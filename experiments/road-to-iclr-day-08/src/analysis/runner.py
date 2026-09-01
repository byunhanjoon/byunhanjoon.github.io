"""Four-way matched/mismatch experiment runner."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.analysis.io import (
    append_manifest,
    atomic_save_npz,
    atomic_write_json,
    code_digest,
    git_provenance,
    load_completed_run,
    make_run_id,
    sha256_file,
)
from src.data import load_task, transform_frame
from src.metrics import classification_metrics, disagreement_metrics, regression_metrics
from src.models import available_model, fit_predict_many
from src.transforms import (
    AtomicSpacingTransform,
    AsinhTransform,
    CategoricalBijectionTransform,
    ComposedTransform,
    EmpiricalCDFTransform,
    IdentityTransform,
    MonotoneSplineTransform,
    NegativeAffineTransform,
    PositiveAffineTransform,
    QuantileGaussianTransform,
    RandomMonotonePWLTransform,
    SignedPowerTransform,
)


CONDITIONS = ("clean", "matched", "context_only", "query_only")
PACKAGE_NAMES = (
    "torch",
    "tabpfn",
    "tabicl",
    "xgboost",
    "catboost",
    "lightgbm",
    "scikit-learn",
    "openml",
    "numpy",
    "pandas",
    "scipy",
    "pytabkit",
)


def _packages() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _transform(name: str, value: float, seed: int):
    factories = {
        "identity": lambda: IdentityTransform(),
        "positive_affine": lambda: PositiveAffineTransform(value, seed),
        "negative_affine": lambda: NegativeAffineTransform(value, seed),
        "signed_power": lambda: SignedPowerTransform(value),
        "asinh": lambda: AsinhTransform(value),
        "random_monotone_pwl": lambda: RandomMonotonePWLTransform(value, seed),
        "monotone_spline": lambda: MonotoneSplineTransform(value, seed),
        "empirical_cdf": lambda: EmpiricalCDFTransform(),
        "quantile_gaussian": lambda: QuantileGaussianTransform(value),
        "atomic_spacing": lambda: AtomicSpacingTransform(value, seed),
        "composition": lambda: ComposedTransform(
            [
                PositiveAffineTransform(value, seed),
                RandomMonotonePWLTransform(value, seed + 104_729),
            ]
        ),
    }
    if name not in factories:
        raise ValueError(f"unknown audit transform: {name}")
    return factories[name]()


def _job_key(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _gpu_reset(device: str) -> None:
    try:
        import torch

        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(torch.device(device))
    except Exception:
        pass


def _gpu_peak(device: str) -> int | None:
    try:
        import torch

        if device.startswith("cuda") and torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated(torch.device(device)))
    except Exception:
        pass
    return None


def _combined_gpu_peak(telemetry: dict[str, Any], device: str) -> int | None:
    peaks = [
        item.get("peak_gpu_memory_bytes")
        for item in telemetry.values()
        if item.get("peak_gpu_memory_bytes") is not None
    ]
    parent_peak = _gpu_peak(device)
    if parent_peak is not None:
        peaks.append(parent_peak)
    return max(peaks, default=None)


def _metrics(problem_type: str, y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    return regression_metrics(y, prediction) if problem_type == "regression" else classification_metrics(y, prediction)


def build_run_snapshot(root: Path, repository: Path, config_path: Path) -> dict[str, Any]:
    """Freeze command-level provenance before result files change the worktree."""
    return {
        "code_sha256": code_digest(root),
        "git_provenance": git_provenance(repository),
        "package_versions": _packages(),
        "config_sha256": sha256_file(config_path),
    }


def run_job(
    *,
    root: Path,
    repository: Path,
    config_path: Path,
    config: dict[str, Any],
    dataset_spec: dict[str, Any],
    model: str,
    transform_name: str,
    transform_value: float,
    seed: int,
    split_seed: int | None = None,
    device: str,
    resume: bool,
    command: str,
    run_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_snapshot = run_snapshot or build_run_snapshot(root, repository, config_path)
    code_sha256 = str(run_snapshot["code_sha256"])
    split_seed = int(config.get("split_seed") if split_seed is None else split_seed)
    identity = {
        "protocol_version": config["protocol_version"],
        "phase": config["phase"],
        "dataset": dataset_spec,
        "model": model,
        "transform": transform_name,
        "transform_value": transform_value,
        "seed": seed,
        "max_context": config.get("max_context"),
        "max_query": config.get("max_query"),
        "split_seed": split_seed,
        "precision": config["precision"],
    }
    job_key = _job_key(identity)
    manifest_path = root / "results" / "MANIFEST.jsonl"
    if resume:
        cached = load_completed_run(manifest_path, job_key, code_sha256)
        if cached is not None:
            return {"status": "cached", "job_key": job_key, "run_id": cached["run_id"]}

    available, reason = available_model(model, str(dataset_spec["problem_type"]))
    run_id = make_run_id(job_key)
    output_dir = root / "results" / "raw" / str(config["phase"]) / job_key[:16]
    result_path = (output_dir / f"{run_id}.npz").resolve()
    metadata_path = (output_dir / f"{run_id}.json").resolve()
    started = time.perf_counter()
    provenance = dict(run_snapshot["git_provenance"])
    common: dict[str, Any] = {
        "run_id": run_id,
        "job_key": job_key,
        **provenance,
        "code_sha256": code_sha256,
        "python_version": platform.python_version(),
        "package_versions": dict(run_snapshot["package_versions"]),
        "model": model,
        "model_checkpoint": None,
        "dataset": dataset_spec["dataset"],
        "split_id": "not_loaded",
        "transformation": {
            "name": transform_name,
            "parameter": transform_value,
        },
        "seed": seed,
        "split_seed": split_seed,
        "device": device,
        "wall_clock_seconds": None,
        "peak_gpu_memory_bytes": None,
        "command": command,
        "config": str(config_path.resolve()),
        "config_sha256": str(run_snapshot["config_sha256"]),
        "result_path": str(result_path),
        "metadata_path": str(metadata_path),
        "phase": config["phase"],
        "created_at_utc": run_id.split("__", 1)[0],
        "precision": config["precision"],
    }
    if not available:
        common.update(
            {
                "status": "unavailable",
                "failure": reason,
                "wall_clock_seconds": time.perf_counter() - started,
                "result_sha256": None,
            }
        )
        atomic_write_json(metadata_path, common)
        append_manifest(manifest_path, common)
        return common

    try:
        task = load_task(
            dataset_spec,
            seed=split_seed,
            max_context=config.get("max_context"),
            max_query=config.get("max_query"),
            cache_dir=Path(config["openml_cache"]).expanduser(),
        )
        common["split_id"] = task.split_id
        numerical_train = task.X_train.loc[:, task.numeric_columns].to_numpy(dtype=np.float64)
        numerical_test = task.X_test.loc[:, task.numeric_columns].to_numpy(dtype=np.float64)
        if transform_name == "categorical_bijection":
            transform = CategoricalBijectionTransform(seed)
            transform.fit(task.X_train, task.categorical_columns)
            warped_train = transform.transform(task.X_train)
            warped_test = transform.transform(task.X_test)
            transform_audit = transform.audit(task.X_train, task.X_test)
            transform_scope = "categorical"
        else:
            if not task.numeric_columns:
                raise ValueError(
                    f"numerical transform {transform_name!r} is inapplicable to "
                    f"all-categorical dataset {task.dataset!r}"
                )
            transform = _transform(transform_name, transform_value, seed)
            transform.fit(numerical_train)
            warped_train = transform_frame(task.X_train, task.numeric_columns, transform)
            warped_test = transform_frame(task.X_test, task.numeric_columns, transform)
            transform_audit = transform.audit(numerical_train, numerical_test)
            transform_scope = "numerical"
        predictions: dict[str, np.ndarray] = {}
        telemetry: dict[str, Any] = {}
        _gpu_reset(device)
        fit_groups = (
            (task.X_train, {"clean": task.X_test, "query_only": warped_test}),
            (warped_train, {"matched": warped_test, "context_only": task.X_test}),
        )
        for train, queries in fit_groups:
            outcomes = fit_predict_many(
                model,
                task.problem_type,
                train,
                task.y_train,
                queries,
                categorical_columns=task.categorical_columns,
                categorical_indices=task.categorical_indices,
                seed=seed,
                device=device,
            )
            for condition, outcome in outcomes.items():
                predictions[condition] = outcome.prediction
                telemetry[condition] = outcome.telemetry
        clean_metrics = _metrics(task.problem_type, task.y_test, predictions["clean"])
        metrics: dict[str, Any] = {}
        normalization_scale = float(np.std(task.y_train)) if task.problem_type == "regression" else None
        for condition in CONDITIONS:
            performance = _metrics(task.problem_type, task.y_test, predictions[condition])
            disagreement = disagreement_metrics(
                predictions["clean"],
                predictions[condition],
                task.problem_type,
                normalization_scale=normalization_scale,
            )
            performance["isomorphism_gap"] = float(performance["loss"] - clean_metrics["loss"])
            performance["normalized_isomorphism_gap"] = (
                float(performance["isomorphism_gap"] / max(normalization_scale**2, 1e-12))
                if normalization_scale is not None
                else performance["isomorphism_gap"]
            )
            metrics[condition] = {**performance, **disagreement}
        arrays = {f"prediction__{condition}": prediction for condition, prediction in predictions.items()}
        arrays.update(
            {
                "y_test": task.y_test,
                "test_row_ids": task.test_row_ids,
                "train_row_ids": task.train_row_ids,
            }
        )
        atomic_save_npz(result_path, arrays)
        checkpoint = telemetry["clean"].get("checkpoint")
        common.update(
            {
                "status": "complete",
                "model_checkpoint": checkpoint,
                "task_id": task.task_id,
                "problem_type": task.problem_type,
                "n_classes": task.n_classes,
                "target_normalization_scale": normalization_scale,
                "rows": {"train": len(task.X_train), "validation": len(task.X_validation), "test": len(task.X_test)},
                "features": {"numeric": len(task.numeric_columns), "categorical": len(task.categorical_columns)},
                "dataset_audit": task.audit(),
                "dataset_descriptors": task.descriptors,
                "transform_state": transform.state_dict(),
                "transform_audit": transform_audit,
                "transform_scope": transform_scope,
                "metrics": metrics,
                "telemetry": telemetry,
                "fit_pairing": {
                    "original_context": ["clean", "query_only"],
                    "transformed_context": ["matched", "context_only"],
                },
                "wall_clock_seconds": time.perf_counter() - started,
                "peak_gpu_memory_bytes": _combined_gpu_peak(telemetry, device),
                "result_sha256": sha256_file(result_path),
            }
        )
        atomic_write_json(metadata_path, common)
        append_manifest(manifest_path, common)
        return common
    except Exception as error:
        common.update(
            {
                "status": "failed",
                "failure": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "wall_clock_seconds": time.perf_counter() - started,
                "peak_gpu_memory_bytes": _gpu_peak(device),
                "result_sha256": None,
            }
        )
        atomic_write_json(metadata_path, common)
        append_manifest(manifest_path, common)
        return common


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    required = {"protocol_version", "phase", "precision", "datasets", "models", "transforms", "seeds", "openml_cache"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"config is missing fields: {sorted(missing)}")
    if "split_seed" not in config and "split_seeds" not in config:
        raise ValueError("config must define split_seed or split_seeds")
    if "split_seeds" in config and not config["split_seeds"]:
        raise ValueError("split_seeds must not be empty")
    if config["precision"] != "autocast_bfloat16":
        raise ValueError("paired audit currently supports only fixed autocast_bfloat16 precision")
    return config


def selected_jobs(
    config: dict[str, Any],
    *,
    datasets: set[str] | None,
    models: set[str] | None,
    transforms: set[str] | None = None,
    seeds: set[int] | None = None,
    split_seeds: set[int] | None = None,
    shard_index: int,
    num_shards: int,
):
    jobs = []
    configured_split_seeds = config.get("split_seeds", [config.get("split_seed")])
    for dataset in config["datasets"]:
        if datasets and dataset["dataset"] not in datasets:
            continue
        for model in config["models"]:
            if models and model not in models:
                continue
            for transform in config["transforms"]:
                if transforms and transform["name"] not in transforms:
                    continue
                for value in transform["values"]:
                    for seed in config["seeds"]:
                        if seeds and int(seed) not in seeds:
                            continue
                        for split_seed in configured_split_seeds:
                            if split_seeds and int(split_seed) not in split_seeds:
                                continue
                            jobs.append(
                                (
                                    dataset,
                                    model,
                                    transform["name"],
                                    float(value),
                                    int(seed),
                                    int(split_seed),
                                )
                            )
    return [job for index, job in enumerate(jobs) if index % num_shards == shard_index]
