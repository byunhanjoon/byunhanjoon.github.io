#!/usr/bin/env python3
"""Run immutable Experiment A replication bundles on development data.

Prospective datasets are rejected until the separately hashed method freeze exists; the
prospective runner is intentionally a different entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
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
REPOSITORY = ROOT.parents[2]
sys.path.insert(0, str(ROOT))

from src.basis_dependence import (  # noqa: E402
    build_primary_representations,
    build_rbf_feature_matrix,
    disagreement_metrics,
    environment_metadata,
    fit_predict,
    jsonable,
    load_dataset,
    prediction_metrics,
    sha256_file,
)


CONFIG_PATH = ROOT / "configs" / "development_protocol.yaml"
PANEL_PATH = ROOT / "configs" / "dataset_panel.json"


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def prediction_frame(
    data: Any, model: str, seed: int, rep: Any, split_name: str, row_ids: np.ndarray,
    target: np.ndarray, prediction: np.ndarray,
) -> pd.DataFrame:
    result = pd.DataFrame({
        "dataset": data.key, "panel": data.panel, "model": model, "model_seed": seed,
        "split": split_name, "representation_id": rep.representation_id,
        "variant": rep.variant, "scope": rep.scope, "member": rep.member,
        "is_reference": rep.is_reference, "row_id": row_ids, "target": target,
    })
    values = np.asarray(prediction)
    if values.ndim == 1:
        result["prediction"] = values
    else:
        for class_index in range(values.shape[1]):
            result[f"prediction_{class_index}"] = values[:, class_index]
    return result


def git_metadata() -> dict[str, str]:
    def run(*args: str) -> str:
        return subprocess.run(args, cwd=REPOSITORY, check=True, capture_output=True, text=True).stdout.strip()
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "experiment_status": run("git", "status", "--short", "--", str(ROOT.relative_to(REPOSITORY))),
    }


def run_bundle(
    config: dict[str, Any], panel_hash: str, config_hash: str, spec: dict[str, Any],
    model_name: str, seed: int, device: str,
) -> str:
    if spec["panel"] != "development":
        raise RuntimeError("development runner refuses prospective datasets")
    destination = ROOT / "results" / "raw" / "development" / "replication" / model_name / spec["key"] / f"seed_{seed}"
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
    representations = build_primary_representations(blocks, int(config["orbit_members"]))
    if len(representations) != 25:
        raise RuntimeError(f"unexpected representation count: {len(representations)}")
    prediction_parts = []
    metrics = []
    representation_records = []
    predictions: dict[tuple[str, str], np.ndarray] = {}
    telemetry: dict[str, dict[str, Any]] = {}
    started = time.time()
    for index, rep in enumerate(representations, start=1):
        validation_prediction, test_prediction, timing = fit_predict(
            model_name, data.problem_type, rep, data.y_train, data.y_validation, seed, device, config
        )
        for split_name, target, prediction in (
            ("validation", data.y_validation, validation_prediction), ("test", data.y_test, test_prediction)
        ):
            if len(prediction) != len(target) or not np.isfinite(prediction).all():
                raise RuntimeError(f"invalid {split_name} prediction for {rep.representation_id}")
            predictions[(rep.representation_id, split_name)] = prediction
            row_ids = data.validation_indices if split_name == "validation" else data.test_indices
            prediction_parts.append(prediction_frame(
                data, model_name, seed, rep, split_name, row_ids, target, prediction
            ))
        telemetry[rep.representation_id] = timing
        audits = rep.metadata.get("equivalence", {})
        if audits:
            max_reconstruction = max(record["reconstruction_error"] for record in audits.values())
            max_orthogonality = max(record["orthogonality_error"] for record in audits.values())
            max_condition = max(record["condition_number"] for record in audits.values())
            if max_reconstruction >= 1e-6:
                raise RuntimeError(f"equivalence audit failed: {rep.representation_id}")
            if rep.variant.startswith("orthogonal") and max_orthogonality >= 1e-6:
                raise RuntimeError(f"orthogonality audit failed: {rep.representation_id}")
            if rep.variant == "condition_le_3_all" and max_condition > 3 + 1e-8:
                raise RuntimeError(f"condition audit failed: {rep.representation_id}")
        else:
            max_reconstruction, max_orthogonality, max_condition = 0.0, 0.0, 1.0
        representation_records.append({
            "representation_id": rep.representation_id, "variant": rep.variant, "scope": rep.scope,
            "member": rep.member, "transforms": rep.transforms, "metadata": rep.metadata,
            "max_reconstruction_error": max_reconstruction,
            "max_orthogonality_error": max_orthogonality, "max_condition_number": max_condition,
        })
        print(f"[{index}/{len(representations)}] {model_name} {data.key} seed={seed} {rep.representation_id}", flush=True)

    reference = representations[0]
    for rep, record in zip(representations, representation_records):
        for split_name, target in (("validation", data.y_validation), ("test", data.y_test)):
            prediction = predictions[(rep.representation_id, split_name)]
            task = prediction_metrics(data.problem_type, target, prediction)
            disagreement = disagreement_metrics(
                data.problem_type, target, predictions[(reference.representation_id, split_name)], prediction
            )
            metrics.append({
                "dataset": data.key, "openml_id": data.openml_id, "openml_version": data.openml_version,
                "panel": data.panel, "problem_type": data.problem_type, "model": model_name,
                "model_seed": seed, "split": split_name, "representation_id": rep.representation_id,
                "variant": rep.variant, "scope": rep.scope, "member": rep.member,
                "is_reference": rep.is_reference, "selected_feature": blocks.selected_feature,
                "n_train": len(data.y_train), "n_validation": len(data.y_validation), "n_test": len(data.y_test),
                "n_raw_features": data.X_train_raw.shape[1], "n_encoded_features": rep.X_train.shape[1],
                **task, **disagreement, **telemetry[rep.representation_id],
                "max_reconstruction_error": record["max_reconstruction_error"],
                "max_orthogonality_error": record["max_orthogonality_error"],
                "max_condition_number": record["max_condition_number"],
            })

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        pd.DataFrame(metrics).to_csv(temporary / "metrics.csv", index=False)
        pd.concat(prediction_parts, ignore_index=True).to_csv(
            temporary / "predictions.csv.gz", index=False, compression="gzip"
        )
        (temporary / "representations.json").write_text(
            json.dumps(jsonable(representation_records), indent=2, sort_keys=True) + "\n"
        )
        files = {}
        for filename in ("metrics.csv", "predictions.csv.gz", "representations.json"):
            path = temporary / filename
            files[filename] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        metadata = {
            "status": "complete", "experiment": config["experiment"], "protocol_version": config["protocol_version"],
            "stage": "development_replication", "config_sha256": config_hash,
            "dataset_panel_sha256": panel_hash, "dataset_spec": spec, "model": model_name,
            "model_seed": seed, "device": device, "wall_seconds": time.time() - started,
            "representation_count": len(representations), "selected_feature": blocks.selected_feature,
            "feature_blocks": blocks.feature_blocks, "categorical_blocks": blocks.categorical_blocks,
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
    print(f"[complete] {destination} wall={time.time() - started:.1f}s", flush=True)
    return "complete"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--model", action="append")
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--list-jobs", action="store_true")
    args = parser.parse_args()
    config_bytes = CONFIG_PATH.read_bytes()
    panel_bytes = PANEL_PATH.read_bytes()
    config = yaml.safe_load(config_bytes)
    panel = json.loads(panel_bytes)
    datasets = [
        spec for spec in panel["datasets"] if spec["panel"] == "development"
        and (not args.dataset or spec["key"] in set(args.dataset))
    ]
    models = [name for name in config["models"] if not args.model or name in set(args.model)]
    seeds = [int(seed) for seed in config["model_seeds"] if not args.seed or int(seed) in set(args.seed)]
    jobs = [(spec, model, seed) for spec in datasets for model in models for seed in seeds]
    if args.list_jobs:
        for spec, model, seed in jobs:
            print(json.dumps({"dataset": spec["key"], "model": model, "seed": seed}))
        return
    counts = {"complete": 0, "cached": 0, "failed": 0}
    failures = []
    for index, (spec, model, seed) in enumerate(jobs, start=1):
        print(f"=== job {index}/{len(jobs)}: {model} {spec['key']} seed={seed} ===", flush=True)
        try:
            status = run_bundle(
                config, digest_bytes(panel_bytes), digest_bytes(config_bytes), spec, model, seed, args.device
            )
            counts[status] += 1
        except Exception as error:
            counts["failed"] += 1
            failures.append({
                "dataset": spec["key"], "model": model, "seed": seed,
                "error": repr(error), "traceback": traceback.format_exc(),
            })
            print(json.dumps(failures[-1]), flush=True)
    print(json.dumps({"counts": counts, "failures": failures}, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
