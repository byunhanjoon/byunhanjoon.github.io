#!/usr/bin/env python3
"""Run the locked prospective panel exactly once after a verified method freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_replication import git_metadata, prediction_frame  # noqa: E402
from src.basis_dependence import (  # noqa: E402
    anchor_canonical_representation, build_primary_representations, build_rbf_feature_matrix,
    disagreement_metrics, environment_metadata, fit_predict, jsonable, load_dataset,
    oracle_inverse_representation, pca_canonical_representation, prediction_metrics,
    sha256_file, standardize_representation, whiten_representation,
)


CONFIG_PATH = ROOT / "configs" / "development_protocol.yaml"
PANEL_PATH = ROOT / "configs" / "dataset_panel.json"
METHOD_PATH = ROOT / "configs" / "FROZEN_METHOD_CONFIG.json"
LOCK_PATH = ROOT / "results" / "PROSPECTIVE_LOCK.json"
START_PATH = ROOT / "results" / "raw" / "prospective" / "RUN_STARTED.json"
COMPLETE_PATH = ROOT / "results" / "raw" / "prospective" / "RUN_COMPLETE.json"


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def verify_freeze() -> tuple[dict[str, Any], str]:
    """Verify the freeze without touching any dataset or outcome."""
    if not METHOD_PATH.exists():
        raise RuntimeError("prospective lock active: configs/FROZEN_METHOD_CONFIG.json does not exist")
    method_bytes = METHOD_PATH.read_bytes()
    method_hash = digest(method_bytes)
    lock = json.loads(LOCK_PATH.read_text())
    recorded = lock.get("frozen_method_config_sha256")
    if recorded != method_hash:
        raise RuntimeError(f"frozen method hash mismatch: lock={recorded!r}, actual={method_hash}")
    method = json.loads(method_bytes)
    if method.get("status") != "frozen_before_prospective_access":
        raise RuntimeError("method config lacks frozen-before-access status")
    return method, method_hash


def mark_started(method_hash: str, config_hash: str, panel_hash: str) -> None:
    START_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "started", "prospective_outcome_accessed": True,
        "started_unix_time": time.time(), "frozen_method_config_sha256": method_hash,
        "development_protocol_sha256": config_hash, "dataset_panel_sha256": panel_hash,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(START_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        existing = json.loads(START_PATH.read_text())
        if existing.get("frozen_method_config_sha256") != method_hash:
            raise RuntimeError("prospective run already started under a different method freeze")
        return
    with os.fdopen(descriptor, "w") as handle:
        handle.write(encoded)


def primary_builder(method: dict[str, Any], dataset_key: str) -> tuple[str, Callable[[Any], Any]]:
    name = str(method["primary_method"])
    if name == "pca_canonical":
        return name, pca_canonical_representation
    if name == "anchor_canonical":
        anchors = int(method.get("anchor_rows", 16))
        return name, lambda rep: anchor_canonical_representation(rep, dataset_key, anchors=anchors)
    raise RuntimeError(f"prospective representation runner does not support primary method {name!r}")


def run_bundle(
    config: dict[str, Any], config_hash: str, panel_hash: str, method: dict[str, Any],
    method_hash: str, spec: dict[str, Any], model_name: str, seed: int, device: str,
) -> str:
    if spec["panel"] != "prospective":
        raise RuntimeError("prospective runner refuses development datasets")
    destination = ROOT / "results" / "raw" / "prospective" / "evaluation" / model_name / spec["key"] / f"seed_{seed}"
    if (destination / "metadata.json").exists():
        metadata = json.loads((destination / "metadata.json").read_text())
        hashes = (metadata["config_sha256"], metadata["dataset_panel_sha256"], metadata["frozen_method_config_sha256"])
        if hashes != (config_hash, panel_hash, method_hash):
            raise RuntimeError(f"frozen prospective drift at {destination}")
        return "cached"
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite incomplete prospective bundle: {destination}")

    # This is the first point at which a prospective dataset (and its outcomes) is loaded.
    data = load_dataset(spec, config)
    blocks = build_rbf_feature_matrix(data, config)
    primary = build_primary_representations(blocks, int(config["orbit_members"]))
    orbit = [primary[0], *[rep for rep in primary if rep.variant == "orthogonal_all"]]
    primary_name, proposed = primary_builder(method, data.key)
    builders: dict[str, Callable[[Any], Any]] = {
        "raw": lambda rep: rep,
        "standardization": standardize_representation,
        "whitening": whiten_representation,
        primary_name: proposed,
        "ORACLE INVERSE — NOT A METHOD": oracle_inverse_representation,
    }
    predictions_by_repair: dict[str, dict[tuple[str, str], np.ndarray]] = {}
    metrics = []
    prediction_parts = []
    repair_records = []
    started = time.time()
    for repair, builder in builders.items():
        repaired = [builder(rep) for rep in orbit]
        if repair == "anchor_canonical":
            failed = sorted({feature for rep in repaired for feature, audit in rep.metadata["anchor"].items()
                             if not audit["full_rank"]})
            if failed:
                raise RuntimeError(f"frozen AnchorCanonical is rank deficient on prospective blocks: {failed}")
        predictions_by_repair[repair] = {}
        for source, rep in zip(orbit, repaired):
            validation, test, telemetry = fit_predict(
                model_name, data.problem_type, rep, data.y_train, data.y_validation, seed, device, config
            )
            for split, row_ids, y, prediction in (
                ("validation", data.validation_indices, data.y_validation, validation),
                ("test", data.test_indices, data.y_test, test),
            ):
                if len(prediction) != len(y) or not np.isfinite(prediction).all():
                    raise RuntimeError(f"invalid prospective prediction: {repair}/{source.representation_id}/{split}")
                predictions_by_repair[repair][(source.representation_id, split)] = prediction
                frame = prediction_frame(data, model_name, seed, source, split, row_ids, y, prediction)
                frame["repair"] = repair
                prediction_parts.append(frame)
                metrics.append({
                    "dataset": data.key, "panel": data.panel, "problem_type": data.problem_type,
                    "model": model_name, "model_seed": seed, "split": split, "repair": repair,
                    "representation_id": source.representation_id, "variant": source.variant,
                    "member": source.member, "is_reference": source.is_reference,
                    **prediction_metrics(data.problem_type, y, prediction), **telemetry,
                })
            repair_records.append({
                "repair": repair, "source_representation_id": source.representation_id,
                "output_dimension": rep.X_train.shape[1], "metadata": rep.metadata,
            })
        for record in metrics:
            if record["repair"] != repair:
                continue
            rep_id, split = record["representation_id"], record["split"]
            record.update(disagreement_metrics(
                data.problem_type,
                data.y_validation if split == "validation" else data.y_test,
                predictions_by_repair[repair][(orbit[0].representation_id, split)],
                predictions_by_repair[repair][(rep_id, split)],
            ))

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        pd.DataFrame(metrics).to_csv(temporary / "metrics.csv", index=False)
        pd.concat(prediction_parts, ignore_index=True).to_csv(
            temporary / "predictions.csv.gz", index=False, compression="gzip"
        )
        (temporary / "repair_records.json").write_text(json.dumps(jsonable(repair_records), indent=2, sort_keys=True) + "\n")
        metadata = {
            "status": "complete", "stage": "prospective_evaluation", "dataset_spec": spec,
            "model": model_name, "model_seed": seed, "device": device,
            "config_sha256": config_hash, "dataset_panel_sha256": panel_hash,
            "frozen_method_config_sha256": method_hash, "repairs": list(builders),
            "source_representation_count": len(orbit), "fit_count": len(builders) * len(orbit),
            "wall_seconds": time.time() - started, "environment": environment_metadata(),
            "git": git_metadata(), "files": {},
        }
        for filename in ("metrics.csv", "predictions.csv.gz", "repair_records.json"):
            metadata["files"][filename] = {
                "sha256": sha256_file(temporary / filename), "bytes": (temporary / filename).stat().st_size,
            }
        (temporary / "metadata.json").write_text(json.dumps(jsonable(metadata), indent=2, sort_keys=True) + "\n")
        os.rename(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return "complete"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--model", action="append")
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--verify-lock-only", action="store_true")
    args = parser.parse_args()
    method, method_hash = verify_freeze()
    if args.verify_lock_only:
        print(json.dumps({"verified": True, "frozen_method_config_sha256": method_hash}))
        return
    config_bytes, panel_bytes = CONFIG_PATH.read_bytes(), PANEL_PATH.read_bytes()
    config, panel = yaml.safe_load(config_bytes), json.loads(panel_bytes)
    config_hash, panel_hash = digest(config_bytes), digest(panel_bytes)
    mark_started(method_hash, config_hash, panel_hash)
    datasets = [spec for spec in panel["datasets"] if spec["panel"] == "prospective"
                and (not args.dataset or spec["key"] in set(args.dataset))]
    frozen_models = list(method["prospective_models"])
    models = [name for name in frozen_models if not args.model or name in set(args.model)]
    seeds = [int(seed) for seed in method["model_seeds"] if not args.seed or int(seed) in set(args.seed)]
    counts = {"complete": 0, "cached": 0, "failed": 0}
    failures = []
    for spec in datasets:
        for model in models:
            for seed in seeds:
                print(f"=== prospective {spec['key']} {model} seed={seed} ===", flush=True)
                try:
                    status = run_bundle(
                        config, config_hash, panel_hash, method, method_hash, spec, model, seed, args.device
                    )
                    counts[status] += 1
                except Exception as error:
                    counts["failed"] += 1
                    failures.append({"dataset": spec["key"], "model": model, "seed": seed,
                                     "error": repr(error), "traceback": traceback.format_exc()})
                    print(json.dumps(failures[-1]), flush=True)
    expected = len(datasets) * len(models) * len(seeds)
    complete = sum(1 for spec in datasets for model in models for seed in seeds
                   if (ROOT / "results" / "raw" / "prospective" / "evaluation" / model /
                       spec["key"] / f"seed_{seed}" / "metadata.json").exists())
    if not failures and complete == expected:
        payload = {
            "status": "complete", "completed_unix_time": time.time(), "expected_bundles": expected,
            "complete_bundles": complete, "frozen_method_config_sha256": method_hash,
        }
        if not COMPLETE_PATH.exists():
            COMPLETE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"counts": counts, "failures": failures, "coverage": [complete, expected]}, indent=2))
    if failures or complete != expected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
