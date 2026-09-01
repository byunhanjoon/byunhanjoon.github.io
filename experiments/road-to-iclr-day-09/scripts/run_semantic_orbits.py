#!/usr/bin/env python3
"""Run immutable model/dataset/seed bundles for the semantic-orbit kill experiment."""

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
REPOSITORY = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from src.semantic_orbits import (  # noqa: E402
    build_representations,
    disagreement_metrics,
    environment_metadata,
    fit_predict,
    jsonable,
    load_dataset,
    prediction_metrics,
    synthetic_sanity,
)


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def frame_digest(*frames: pd.DataFrame, categorical_columns: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(categorical_columns).encode())
    for frame in frames:
        digest.update("|".join(frame.columns).encode())
        digest.update("|".join(map(str, frame.dtypes)).encode())
        digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64).tobytes())
    return digest.hexdigest()


def git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(args, cwd=REPOSITORY, check=True, capture_output=True, text=True).stdout.strip()

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "day09_status": run("git", "status", "--short", "--", str(ROOT.relative_to(REPOSITORY))),
    }


def prediction_frame(
    dataset: str,
    model: str,
    seed: int,
    rep: Any,
    row_ids: np.ndarray,
    y: np.ndarray,
    prediction: np.ndarray,
) -> pd.DataFrame:
    values = np.asarray(prediction)
    result = pd.DataFrame({
        "dataset": dataset,
        "model": model,
        "model_seed": seed,
        "pipeline": rep.pipeline,
        "representation_id": rep.representation_id,
        "reference_id": rep.reference_id,
        "family": rep.family,
        "variant": rep.variant,
        "scope": rep.scope,
        "member": rep.member,
        "repair": rep.repair,
        "is_reference": rep.is_reference,
        "test_row_id": row_ids,
        "target": y,
    })
    if values.ndim == 1:
        result["prediction"] = values
    else:
        for class_index in range(values.shape[1]):
            result[f"prediction_{class_index}"] = values[:, class_index]
    return result


def run_bundle(
    config: dict[str, Any],
    config_path: Path,
    config_hash: str,
    dataset_spec: dict[str, Any],
    model_name: str,
    seed: int,
    device: str,
    force: bool,
) -> str:
    destination = ROOT / "results" / "semantic_orbits" / "raw" / model_name / dataset_spec["name"] / f"seed_{seed}"
    complete_file = destination / "metadata.json"
    if complete_file.exists() and not force:
        existing = json.loads(complete_file.read_text())
        if existing.get("config_sha256") != config_hash:
            raise RuntimeError(f"config drift at existing bundle {destination}")
        print(f"[cached] {model_name} {dataset_spec['name']} seed={seed}", flush=True)
        return "cached"
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite incomplete/forced immutable bundle: {destination}")

    split = load_dataset(dataset_spec, config)
    declared_pipelines = list(config["models"][model_name]["pipelines"])
    representations = []
    for pipeline in declared_pipelines:
        representations.extend(build_representations(split, pipeline, int(config["orbit_members"])))
    references = [rep for rep in representations if rep.is_reference]
    members = [rep for rep in representations if not rep.is_reference]
    representations = references + members
    print(
        f"[start] {model_name} {split.name} seed={seed}: {len(references)} references + {len(members)} members",
        flush=True,
    )

    predictions: dict[str, np.ndarray] = {}
    telemetry_by_id: dict[str, dict[str, Any]] = {}
    prediction_parts: list[pd.DataFrame] = []
    cache: dict[str, tuple[np.ndarray, dict[str, Any], str]] = {}
    started = time.time()
    for index, rep in enumerate(representations, start=1):
        for frame_name, frame in (("train", rep.X_train), ("validation", rep.X_validation), ("test", rep.X_test)):
            numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)
            if np.isinf(numeric).any():
                raise RuntimeError(f"infinite input in {rep.representation_id} {frame_name}")
        key = frame_digest(rep.X_train, rep.X_test, categorical_columns=rep.categorical_columns)
        if key in cache:
            prediction, base_telemetry, source_id = cache[key]
            telemetry = {**base_telemetry, "cached_equivalent_representation": source_id, "fit_seconds": 0.0, "predict_seconds": 0.0}
        else:
            prediction, telemetry = fit_predict(
                model_name, split.problem_type, rep, split.y_train, seed, device, config
            )
            cache[key] = (prediction.copy(), telemetry.copy(), rep.representation_id)
        if len(prediction) != len(split.y_test) or not np.isfinite(prediction).all():
            raise RuntimeError(f"invalid predictions for {rep.representation_id}: shape={prediction.shape}")
        predictions[rep.representation_id] = prediction
        telemetry_by_id[rep.representation_id] = telemetry
        prediction_parts.append(prediction_frame(
            split.name, model_name, seed, rep, split.test_indices, split.y_test, prediction
        ))
        if index == 1 or index % 20 == 0 or index == len(representations):
            elapsed = time.time() - started
            print(f"[{index}/{len(representations)}] {rep.representation_id} elapsed={elapsed:.1f}s", flush=True)

    metric_rows = []
    for rep in representations:
        task = prediction_metrics(split.problem_type, split.y_test, predictions[rep.representation_id])
        disagreement = disagreement_metrics(
            split.problem_type,
            split.y_test,
            predictions[rep.reference_id],
            predictions[rep.representation_id],
        )
        telemetry = telemetry_by_id[rep.representation_id]
        metric_rows.append({
            "dataset": split.name,
            "openml_id": split.openml_id,
            "openml_version": split.openml_version,
            "problem_type": split.problem_type,
            "model": model_name,
            "model_seed": seed,
            "pipeline": rep.pipeline,
            "representation_id": rep.representation_id,
            "reference_id": rep.reference_id,
            "family": rep.family,
            "variant": rep.variant,
            "scope": rep.scope,
            "member": rep.member,
            "repair": rep.repair,
            "is_reference": rep.is_reference,
            "n_train": len(split.y_train),
            "n_validation": len(split.y_validation),
            "n_test": len(split.y_test),
            **task,
            **disagreement,
            "fit_seconds": telemetry.get("fit_seconds"),
            "predict_seconds": telemetry.get("predict_seconds"),
            "checkpoint": telemetry.get("checkpoint"),
            "checkpoint_sha256": telemetry.get("checkpoint_sha256"),
            "checkpoint_version": telemetry.get("checkpoint_version"),
            "cached_equivalent_representation": telemetry.get("cached_equivalent_representation"),
            "transform_metadata_json": json.dumps(jsonable(rep.metadata), sort_keys=True, separators=(",", ":")),
        })

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        pd.concat(prediction_parts, ignore_index=True).to_csv(
            temporary / "predictions.csv.gz", index=False, compression="gzip"
        )
        pd.DataFrame(metric_rows).to_csv(temporary / "metrics.csv", index=False)
        split_audit = {
            "train_validation_disjoint": not bool(set(split.train_indices) & set(split.validation_indices)),
            "train_test_disjoint": not bool(set(split.train_indices) & set(split.test_indices)),
            "validation_test_disjoint": not bool(set(split.validation_indices) & set(split.test_indices)),
            "test_row_order_sha256": digest_bytes(split.test_indices.tobytes()),
            "target_sha256": digest_bytes(np.ascontiguousarray(split.y_test).view(np.uint8)),
        }
        metadata = {
            "status": "complete",
            "experiment": config["experiment"],
            "protocol_version": config["protocol_version"],
            "config": str(config_path.relative_to(ROOT)),
            "config_sha256": config_hash,
            "dataset_spec": dataset_spec,
            "model": model_name,
            "model_seed": seed,
            "device": device,
            "created_unix": time.time(),
            "wall_seconds": time.time() - started,
            "representation_count": len(representations),
            "reference_count": len(references),
            "unique_fit_count": len(cache),
            "split_audit": split_audit,
            "environment": environment_metadata(),
            "git": git_metadata(),
            "files": {
                "predictions.csv.gz": {
                    "sha256": digest_bytes((temporary / "predictions.csv.gz").read_bytes()),
                    "bytes": (temporary / "predictions.csv.gz").stat().st_size,
                },
                "metrics.csv": {
                    "sha256": digest_bytes((temporary / "metrics.csv").read_bytes()),
                    "bytes": (temporary / "metrics.csv").stat().st_size,
                },
            },
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
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "semantic_orbits.yaml")
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--model", action="append")
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--list-jobs", action="store_true")
    parser.add_argument("--force", action="store_true", help="Only valid before an immutable bundle exists")
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    config_hash = digest_bytes(config_bytes)
    selected_datasets = [
        spec for spec in config["datasets"] if not args.dataset or spec["name"] in set(args.dataset)
    ]
    selected_models = [
        model for model in config["models"] if not args.model or model in set(args.model)
    ]
    selected_seeds = [
        int(seed) for seed in config["model_seeds"] if not args.seed or int(seed) in set(args.seed)
    ]
    jobs = [(spec, model, seed) for spec in selected_datasets for model in selected_models for seed in selected_seeds]
    if args.list_jobs:
        for spec, model, seed in jobs:
            print(json.dumps({"dataset": spec["name"], "model": model, "seed": seed}))
        return
    if not jobs:
        raise SystemExit("no jobs selected")

    sanity_path = ROOT / "results" / "semantic_orbits" / "synthetic_sanity.json"
    sanity_path.parent.mkdir(parents=True, exist_ok=True)
    if not sanity_path.exists():
        sanity = synthetic_sanity()
        if sanity["max_structural_function_delta"] > 1e-10 or not sanity["ordinal_strictly_increasing"]:
            raise RuntimeError(f"synthetic sanity failed: {sanity}")
        sanity_path.write_text(json.dumps(sanity, indent=2, sort_keys=True) + "\n")

    counts = {"complete": 0, "cached": 0, "failed": 0}
    failures = []
    for job_index, (spec, model, seed) in enumerate(jobs, start=1):
        print(f"=== job {job_index}/{len(jobs)}: {model} {spec['name']} seed={seed} ===", flush=True)
        try:
            status = run_bundle(config, args.config.resolve(), config_hash, spec, model, seed, args.device, args.force)
            counts[status] += 1
        except Exception as error:
            counts["failed"] += 1
            failure = {
                "dataset": spec["name"], "model": model, "seed": seed,
                "error": repr(error), "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            print(json.dumps(failure), flush=True)
    print(json.dumps({"counts": counts, "failures": failures}, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
