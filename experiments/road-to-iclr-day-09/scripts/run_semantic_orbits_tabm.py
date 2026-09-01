#!/usr/bin/env python3
"""Run the bounded TabM follow-up on the predeclared T6 basis orbits.

This deliberately does not modify the frozen 54-bundle primary grid.  Each dataset/seed
bundle contains the unrotated RBF basis and every member of the three eight-member,
well-conditioned basis orbits.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
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
    jsonable,
    load_dataset,
    prediction_metrics,
    sha256_file,
)


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_metadata() -> dict[str, str]:
    def run(*args: str) -> str:
        return subprocess.run(args, cwd=REPOSITORY, check=True, capture_output=True, text=True).stdout.strip()

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "day09_status": run("git", "status", "--short", "--", str(ROOT.relative_to(REPOSITORY))),
    }


def environment_metadata() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "torch", "pytabkit"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    metadata: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch
        metadata.update({
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        })
    except Exception:
        metadata["gpu"] = None
    return metadata


def fit_tabm(rep: Any, split: Any, seed: int, device: str, model_config: dict[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
    from pytabkit.models.sklearn.sklearn_interfaces import TabM_D_Regressor

    model = TabM_D_Regressor(
        device=device,
        random_state=seed,
        n_cv=int(model_config["n_cv"]),
        n_refit=int(model_config["n_refit"]),
        n_epochs=int(model_config["n_epochs"]),
        patience=int(model_config["patience"]),
        n_threads=int(model_config["n_threads"]),
        verbosity=0,
    )
    started = time.perf_counter()
    model.fit(rep.X_train, split.y_train, X_val=rep.X_validation, y_val=split.y_validation)
    fit_seconds = time.perf_counter() - started
    prediction_started = time.perf_counter()
    prediction = np.asarray(model.predict(rep.X_test), dtype=float).reshape(-1)
    predict_seconds = time.perf_counter() - prediction_started
    del model
    try:
        import torch
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    except Exception:
        pass
    return prediction, {"fit_seconds": fit_seconds, "predict_seconds": predict_seconds}


def run_bundle(
    config: dict[str, Any], config_path: Path, config_hash: str, spec: dict[str, Any], seed: int, device: str
) -> str:
    destination = ROOT / "results" / "semantic_orbits" / "tabm_basis" / spec["name"] / f"seed_{seed}"
    if (destination / "metadata.json").exists():
        metadata = json.loads((destination / "metadata.json").read_text())
        if metadata.get("config_sha256") != config_hash:
            raise RuntimeError(f"config drift at {destination}")
        print(f"[cached] {spec['name']} seed={seed}", flush=True)
        return "cached"
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite incomplete immutable bundle: {destination}")

    split = load_dataset(spec, config)
    # pytabkit's TabM rejects NaNs in continuous columns.  Impute in the original
    # coordinate system before constructing any orbit member so every A phi(x)
    # remains exactly the transform of the same filled phi(x).
    missing_before = {}
    for column in split.numerical_columns:
        train_values = pd.to_numeric(split.X_train_numeric[column], errors="coerce")
        fill_value = float(train_values.median())
        missing_count = 0
        for frame in (split.X_train_numeric, split.X_validation_numeric, split.X_test_numeric):
            missing_count += int(frame[column].isna().sum())
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(fill_value)
        if missing_count:
            missing_before[column] = {"count": missing_count, "train_median": fill_value}
    all_representations = build_representations(split, str(config["pipeline"]), int(config["orbit_members"]))
    variants = set(config["variants"])
    members = [rep for rep in all_representations if rep.family == "T6" and rep.variant in variants]
    reference_ids = {rep.reference_id for rep in members}
    references = [rep for rep in all_representations if rep.is_reference and rep.representation_id in reference_ids]
    representations = references + members
    expected = 1 + len(variants) * int(config["orbit_members"])
    if len(representations) != expected or len(references) != 1:
        raise RuntimeError(f"unexpected T6 panel: {len(references)} references, {len(members)} members")

    predictions: dict[str, np.ndarray] = {}
    prediction_parts = []
    metrics = []
    started = time.time()
    telemetry: dict[str, dict[str, float]] = {}
    for index, rep in enumerate(representations, start=1):
        prediction, timing = fit_tabm(rep, split, seed, device, config["model"])
        if len(prediction) != len(split.y_test) or not np.isfinite(prediction).all():
            raise RuntimeError(f"invalid predictions for {rep.representation_id}")
        predictions[rep.representation_id] = prediction
        telemetry[rep.representation_id] = timing
        prediction_parts.append(pd.DataFrame({
            "dataset": split.name,
            "model": "tabm_d",
            "model_seed": seed,
            "pipeline": rep.pipeline,
            "representation_id": rep.representation_id,
            "reference_id": rep.reference_id,
            "family": rep.family,
            "variant": rep.variant,
            "scope": rep.scope,
            "member": rep.member,
            "is_reference": rep.is_reference,
            "test_row_id": split.test_indices,
            "target": split.y_test,
            "prediction": prediction,
        }))
        print(f"[{index}/{len(representations)}] {split.name} seed={seed} {rep.representation_id}", flush=True)

    for rep in representations:
        task = prediction_metrics("regression", split.y_test, predictions[rep.representation_id])
        disagreement = disagreement_metrics(
            "regression", split.y_test, predictions[rep.reference_id], predictions[rep.representation_id]
        )
        metrics.append({
            "dataset": split.name,
            "openml_id": split.openml_id,
            "openml_version": split.openml_version,
            "problem_type": "regression",
            "model": "tabm_d",
            "model_seed": seed,
            "pipeline": rep.pipeline,
            "representation_id": rep.representation_id,
            "reference_id": rep.reference_id,
            "family": rep.family,
            "variant": rep.variant,
            "scope": rep.scope,
            "member": rep.member,
            "repair": "none",
            "is_reference": rep.is_reference,
            "n_train": len(split.y_train),
            "n_validation": len(split.y_validation),
            "n_test": len(split.y_test),
            **task,
            **disagreement,
            **telemetry[rep.representation_id],
            "transform_metadata_json": json.dumps(jsonable(rep.metadata), sort_keys=True, separators=(",", ":")),
        })

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        pd.concat(prediction_parts, ignore_index=True).to_csv(
            temporary / "predictions.csv.gz", index=False, compression="gzip"
        )
        pd.DataFrame(metrics).to_csv(temporary / "metrics.csv", index=False)
        files = {}
        for filename in ("predictions.csv.gz", "metrics.csv"):
            path = temporary / filename
            files[filename] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        metadata = {
            "status": "complete",
            "experiment": config["experiment"],
            "protocol_version": config["protocol_version"],
            "parent_grid_unchanged": True,
            "bounded_followup": "T6 only on three strongest datasets",
            "preprocessing": {
                "continuous_missing": "training median before orbit construction",
                "imputed_columns": missing_before,
            },
            "config": str(config_path.relative_to(ROOT)),
            "config_sha256": config_hash,
            "dataset_spec": spec,
            "model": "tabm_d",
            "model_seed": seed,
            "device": device,
            "representation_count": len(representations),
            "wall_seconds": time.time() - started,
            "split_audit": {
                "train_validation_disjoint": not bool(set(split.train_indices) & set(split.validation_indices)),
                "train_test_disjoint": not bool(set(split.train_indices) & set(split.test_indices)),
                "validation_test_disjoint": not bool(set(split.validation_indices) & set(split.test_indices)),
                "test_row_order_sha256": digest_bytes(split.test_indices.tobytes()),
                "target_sha256": digest_bytes(np.ascontiguousarray(split.y_test).view(np.uint8)),
            },
            "environment": environment_metadata(),
            "git": git_metadata(),
            "files": files,
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
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "semantic_orbits_tabm_basis.yaml")
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    config_hash = digest_bytes(config_bytes)
    datasets = [spec for spec in config["datasets"] if not args.dataset or spec["name"] in set(args.dataset)]
    seeds = [int(seed) for seed in config["model_seeds"] if not args.seed or int(seed) in set(args.seed)]
    counts = {"complete": 0, "cached": 0, "failed": 0}
    failures = []
    for spec in datasets:
        for seed in seeds:
            try:
                status = run_bundle(config, args.config.resolve(), config_hash, spec, seed, args.device)
                counts[status] += 1
            except Exception as error:
                counts["failed"] += 1
                failures.append({
                    "dataset": spec["name"], "seed": seed, "error": repr(error),
                    "traceback": traceback.format_exc(),
                })
                print(json.dumps(failures[-1]), flush=True)
    print(json.dumps({"counts": counts, "failures": failures}, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
