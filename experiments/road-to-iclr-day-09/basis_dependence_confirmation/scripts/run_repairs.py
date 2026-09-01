#!/usr/bin/env python3
"""Run D0–D4 plus the labeled oracle ceiling on orthogonal all-block orbits."""

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
    anchor_canonical_representation,
    build_primary_representations,
    build_rbf_feature_matrix,
    disagreement_metrics,
    environment_metadata,
    fit_predict,
    jsonable,
    load_dataset,
    oracle_inverse_representation,
    pca_canonical_representation,
    prediction_metrics,
    sha256_file,
    standardize_representation,
    whiten_representation,
)


CONFIG_PATH = ROOT / "configs" / "development_protocol.yaml"
PANEL_PATH = ROOT / "configs" / "dataset_panel.json"


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def run_bundle(
    config: dict[str, Any], config_hash: str, panel_hash: str, spec: dict[str, Any],
    model_name: str, seed: int, device: str,
) -> str:
    if spec["panel"] != "development":
        raise RuntimeError("repair development runner refuses prospective datasets")
    destination = ROOT / "results" / "raw" / "development" / "repairs" / model_name / spec["key"] / f"seed_{seed}"
    if (destination / "metadata.json").exists():
        metadata = json.loads((destination / "metadata.json").read_text())
        if metadata.get("config_sha256") != config_hash or metadata.get("dataset_panel_sha256") != panel_hash:
            raise RuntimeError(f"frozen config drift at {destination}")
        print(f"[cached] {model_name} {spec['key']} seed={seed}", flush=True)
        return "cached"
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite incomplete immutable bundle: {destination}")
    data = load_dataset(spec, config)
    blocks = build_rbf_feature_matrix(data, config)
    primary = build_primary_representations(blocks, int(config["orbit_members"]))
    orbit = [primary[0], *[rep for rep in primary if rep.variant == "orthogonal_all"]]
    builders: dict[str, Callable[[Any], Any]] = {
        "raw": lambda rep: rep,
        "standardization": standardize_representation,
        "whitening": whiten_representation,
        "pca_canonical": pca_canonical_representation,
        "anchor_canonical": lambda rep: anchor_canonical_representation(rep, data.key, anchors=16),
        "ORACLE INVERSE — NOT A METHOD": oracle_inverse_representation,
    }
    excluded_repairs = []
    anchor_probe = anchor_canonical_representation(orbit[0], data.key, anchors=16)
    anchor_audits = anchor_probe.metadata["anchor"]
    if not all(record["full_rank"] for record in anchor_audits.values()):
        failed = sorted(feature for feature, record in anchor_audits.items() if not record["full_rank"])
        builders.pop("anchor_canonical")
        excluded_repairs.append({
            "repair": "anchor_canonical", "reason": "rank-deficient fixed-row anchor matrix",
            "failed_feature_blocks": failed,
        })
    prediction_parts = []
    metrics = []
    records = []
    started = time.time()
    for repair, builder in builders.items():
        repaired = [builder(rep) for rep in orbit]
        reference = repaired[0]
        predictions = {}
        telemetry = {}
        for index, (source, rep) in enumerate(zip(orbit, repaired), start=1):
            validation, test, timing = fit_predict(
                model_name, data.problem_type, rep, data.y_train, data.y_validation, seed, device, config
            )
            telemetry[source.representation_id] = timing
            for split_name, target, prediction in (
                ("validation", data.y_validation, validation), ("test", data.y_test, test)
            ):
                if len(prediction) != len(target) or not np.isfinite(prediction).all():
                    raise RuntimeError(f"invalid prediction: {repair}/{source.representation_id}/{split_name}")
                predictions[(source.representation_id, split_name)] = prediction
                row_ids = data.validation_indices if split_name == "validation" else data.test_indices
                frame = prediction_frame(data, model_name, seed, source, split_name, row_ids, target, prediction)
                frame["repair"] = repair
                prediction_parts.append(frame)
            anchor_records = rep.metadata.get("anchor", {})
            pca_records = rep.metadata.get("pca", {})
            records.append({
                "repair": repair, "source_representation_id": source.representation_id,
                "output_dimension": rep.X_train.shape[1],
                "anchor_full_rank": all(item["full_rank"] for item in anchor_records.values()) if anchor_records else None,
                "max_anchor_condition": max(
                    [item["anchor_condition_number"] for item in anchor_records.values()], default=None
                ),
                "pca_degenerate_blocks": sum(item["degenerate"] for item in pca_records.values()),
                "metadata": rep.metadata,
            })
            print(
                f"[{repair} {index}/{len(repaired)}] {model_name} {data.key} seed={seed} {source.representation_id}",
                flush=True,
            )
        for source in orbit:
            for split_name, target in (("validation", data.y_validation), ("test", data.y_test)):
                prediction = predictions[(source.representation_id, split_name)]
                metrics.append({
                    "dataset": data.key, "panel": data.panel, "problem_type": data.problem_type,
                    "model": model_name, "model_seed": seed, "split": split_name, "repair": repair,
                    "representation_id": source.representation_id, "variant": source.variant,
                    "scope": source.scope, "member": source.member, "is_reference": source.is_reference,
                    **prediction_metrics(data.problem_type, target, prediction),
                    **disagreement_metrics(
                        data.problem_type, target, predictions[(orbit[0].representation_id, split_name)], prediction
                    ),
                    **telemetry[source.representation_id],
                })
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        pd.DataFrame(metrics).to_csv(temporary / "metrics.csv", index=False)
        pd.concat(prediction_parts, ignore_index=True).to_csv(
            temporary / "predictions.csv.gz", index=False, compression="gzip"
        )
        (temporary / "repair_records.json").write_text(json.dumps(jsonable(records), indent=2, sort_keys=True) + "\n")
        files = {}
        for filename in ("metrics.csv", "predictions.csv.gz", "repair_records.json"):
            path = temporary / filename
            files[filename] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        metadata = {
            "status": "complete", "stage": "development_repairs", "config_sha256": config_hash,
            "dataset_panel_sha256": panel_hash, "dataset_spec": spec, "model": model_name,
            "model_seed": seed, "device": device, "repairs": list(builders),
            "excluded_repairs": excluded_repairs,
            "source_representation_count": len(orbit), "fit_count": len(builders) * len(orbit),
            "wall_seconds": time.time() - started,
            "split_audit": {
                "train_validation_disjoint": not bool(set(data.train_indices) & set(data.validation_indices)),
                "train_test_disjoint": not bool(set(data.train_indices) & set(data.test_indices)),
                "validation_test_disjoint": not bool(set(data.validation_indices) & set(data.test_indices)),
                "validation_row_order_sha256": digest_bytes(data.validation_indices.tobytes()),
                "test_row_order_sha256": digest_bytes(data.test_indices.tobytes()),
                "validation_target_sha256": digest_bytes(np.ascontiguousarray(data.y_validation).view(np.uint8)),
                "test_target_sha256": digest_bytes(np.ascontiguousarray(data.y_test).view(np.uint8)),
            },
            "environment": environment_metadata(), "git": git_metadata(), "files": files,
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
    args = parser.parse_args()
    config_bytes, panel_bytes = CONFIG_PATH.read_bytes(), PANEL_PATH.read_bytes()
    config, panel = yaml.safe_load(config_bytes), json.loads(panel_bytes)
    datasets = [spec for spec in panel["datasets"] if spec["panel"] == "development"
                and (not args.dataset or spec["key"] in set(args.dataset))]
    models = [name for name in config["models"] if not args.model or name in set(args.model)]
    seeds = [int(value) for value in config["model_seeds"] if not args.seed or value in set(args.seed)]
    jobs = [(spec, model, seed) for spec in datasets for model in models for seed in seeds]
    counts = {"complete": 0, "cached": 0, "failed": 0}
    failures = []
    for index, (spec, model, seed) in enumerate(jobs, start=1):
        print(f"=== job {index}/{len(jobs)}: {model} {spec['key']} seed={seed} ===", flush=True)
        try:
            status = run_bundle(
                config, digest_bytes(config_bytes), digest_bytes(panel_bytes), spec, model, seed, args.device
            )
            counts[status] += 1
        except Exception as error:
            counts["failed"] += 1
            failures.append({"dataset": spec["key"], "model": model, "seed": seed,
                             "error": repr(error), "traceback": traceback.format_exc()})
            print(json.dumps(failures[-1]), flush=True)
    print(json.dumps({"counts": counts, "failures": failures}, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
