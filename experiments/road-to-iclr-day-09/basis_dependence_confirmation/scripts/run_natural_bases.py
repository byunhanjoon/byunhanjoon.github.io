#!/usr/bin/env python3
"""Run natural equivalent-basis pairs on the locked development panel."""

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
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_replication import git_metadata, prediction_frame  # noqa: E402
from src.basis_dependence import (  # noqa: E402
    build_rbf_feature_matrix,
    disagreement_metrics,
    environment_metadata,
    fit_predict,
    fourier_origin_pairs,
    helmert_pair,
    jsonable,
    load_dataset,
    local_spectral_pair,
    prediction_metrics,
    sha256_file,
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
        raise RuntimeError("natural-basis development runner refuses prospective datasets")
    destination = ROOT / "results" / "raw" / "development" / "natural" / model_name / spec["key"] / f"seed_{seed}"
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
    representations = []
    excluded = []
    helmert_result = helmert_pair(blocks)
    if helmert_result is not None:
        representations.extend(helmert_result)
    try:
        representations.extend(local_spectral_pair(blocks))
    except ValueError as error:
        excluded.append({"family": "C3", "reason": str(error)})
    representations.extend(fourier_origin_pairs(blocks, int(config["orbit_members"])))
    if not representations:
        raise RuntimeError(f"no valid natural pair for {data.key}")
    references = {rep.family: rep for rep in representations if rep.is_reference}
    if len(references) != len({rep.family for rep in representations}):
        raise RuntimeError("natural family reference mismatch")
    predictions = {}
    prediction_parts = []
    telemetry = {}
    records = []
    started = time.time()
    for index, rep in enumerate(representations, start=1):
        reconstruction = float(rep.metadata.get("reconstruction_error", 0.0))
        condition = float(rep.metadata.get("condition_number", 1.0))
        if not rep.is_reference and (reconstruction >= 1e-6 or condition > 10 + 1e-8):
            raise RuntimeError(f"natural equivalence audit failed: {rep.representation_id}")
        validation, test, timing = fit_predict(
            model_name, data.problem_type, rep, data.y_train, data.y_validation, seed, device, config
        )
        telemetry[rep.representation_id] = timing
        for split_name, target, prediction in (
            ("validation", data.y_validation, validation), ("test", data.y_test, test)
        ):
            if len(prediction) != len(target) or not np.isfinite(prediction).all():
                raise RuntimeError(f"invalid prediction: {rep.representation_id}/{split_name}")
            predictions[(rep.representation_id, split_name)] = prediction
            row_ids = data.validation_indices if split_name == "validation" else data.test_indices
            prediction_parts.append(prediction_frame(
                data, model_name, seed, rep, split_name, row_ids, target, prediction
            ))
        records.append({
            "representation_id": rep.representation_id, "family": rep.family, "variant": rep.variant,
            "member": rep.member, "is_reference": rep.is_reference,
            "transforms": rep.transforms, "metadata": rep.metadata,
        })
        print(f"[{index}/{len(representations)}] {model_name} {data.key} seed={seed} {rep.representation_id}", flush=True)
    metrics = []
    for rep in representations:
        reference = references[rep.family]
        for split_name, target in (("validation", data.y_validation), ("test", data.y_test)):
            prediction = predictions[(rep.representation_id, split_name)]
            metrics.append({
                "dataset": data.key, "panel": data.panel, "problem_type": data.problem_type,
                "model": model_name, "model_seed": seed, "split": split_name,
                "family": rep.family, "basis_pair": f"{reference.variant}_vs_{rep.variant}",
                "representation_id": rep.representation_id, "reference_id": reference.representation_id,
                "variant": rep.variant, "member": rep.member, "is_reference": rep.is_reference,
                "reconstruction_error": float(rep.metadata.get("reconstruction_error", 0.0)),
                "condition_number": float(rep.metadata.get("condition_number", 1.0)),
                **prediction_metrics(data.problem_type, target, prediction),
                **disagreement_metrics(
                    data.problem_type, target, predictions[(reference.representation_id, split_name)], prediction
                ),
                **telemetry[rep.representation_id],
            })
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        pd.DataFrame(metrics).to_csv(temporary / "metrics.csv", index=False)
        pd.concat(prediction_parts, ignore_index=True).to_csv(
            temporary / "predictions.csv.gz", index=False, compression="gzip"
        )
        (temporary / "representations.json").write_text(json.dumps(jsonable(records), indent=2, sort_keys=True) + "\n")
        files = {}
        for filename in ("metrics.csv", "predictions.csv.gz", "representations.json"):
            path = temporary / filename
            files[filename] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        metadata = {
            "status": "complete", "stage": "development_natural_bases",
            "config_sha256": config_hash, "dataset_panel_sha256": panel_hash,
            "dataset_spec": spec, "model": model_name, "model_seed": seed, "device": device,
            "representation_count": len(representations), "families": sorted(references), "excluded": excluded,
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
    datasets = [
        spec for spec in panel["datasets"] if spec["panel"] == "development"
        and (not args.dataset or spec["key"] in set(args.dataset))
    ]
    models = [name for name in config["models"] if not args.model or name in set(args.model)]
    seeds = [int(seed) for seed in config["model_seeds"] if not args.seed or seed in set(args.seed)]
    counts = {"complete": 0, "cached": 0, "failed": 0}
    failures = []
    jobs = [(spec, model, seed) for spec in datasets for model in models for seed in seeds]
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
