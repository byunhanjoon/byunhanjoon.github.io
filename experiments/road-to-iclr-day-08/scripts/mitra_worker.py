#!/usr/bin/env python3
"""Run one Mitra fit/predict call inside its dependency-isolated environment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_sha256(path: Path) -> str:
    """Use Hugging Face's content-addressed blob name when available."""
    resolved = path.resolve()
    candidate = resolved.name.lower()
    if len(candidate) == 64 and all(character in "0123456789abcdef" for character in candidate):
        return candidate
    return _sha256(resolved)


def _checkpoint(repo_id: str) -> dict[str, object]:
    config = Path(hf_hub_download(repo_id=repo_id, filename="config.json"))
    weights = Path(hf_hub_download(repo_id=repo_id, filename="model.safetensors"))
    resolved = weights.resolve()
    try:
        snapshot_revision = weights.parent.name
    except Exception:
        snapshot_revision = "unknown"
    return {
        "checkpoint": f"hf://{repo_id}@{snapshot_revision}",
        "checkpoint_repo": repo_id,
        "checkpoint_revision": snapshot_revision,
        "checkpoint_config_sha256": _artifact_sha256(config),
        "checkpoint_sha256": _artifact_sha256(weights),
        "checkpoint_bytes": resolved.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    job_dir = args.request.parent
    problem_type = str(request["problem_type"])
    seed = int(request["seed"])
    fine_tune = bool(request["fine_tune"])
    np.random.seed(seed % (2**32 - 1))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()

    train = pd.read_parquet(job_dir / "train.parquet")
    query_names = list(request["query_names"])
    if not query_names or any(not name.replace("_", "").isalnum() for name in query_names):
        raise ValueError(f"invalid query names: {query_names!r}")
    y_train = np.load(job_dir / "y_train.npy", allow_pickle=False)
    label = "__reparam_target__"
    if label in train.columns:
        raise ValueError(f"reserved label column is already present: {label}")
    train[label] = y_train

    repo_id = "autogluon/mitra-regressor" if problem_type == "regression" else "autogluon/mitra-classifier"
    checkpoint = _checkpoint(repo_id)
    from autogluon.tabular import TabularPredictor

    predictor = TabularPredictor(
        label=label,
        problem_type=problem_type,
        eval_metric="mean_squared_error" if problem_type == "regression" else "log_loss",
        path=str(job_dir / "autogluon"),
        verbosity=0,
    )
    hyperparameters = {
        "MITRA": {
            "n_estimators": 1,
            "fine_tune": fine_tune,
            "fine_tune_steps": 50,
            "seed": seed,
            "hf_model": repo_id,
            "verbose": False,
        }
    }
    started = time.perf_counter()
    predictor.fit(
        train_data=train,
        hyperparameters=hyperparameters,
        fit_weighted_ensemble=False,
        dynamic_stacking=False,
        num_cpus=int(request.get("num_cpus", 8)),
        num_gpus=1 if torch.cuda.is_available() else 0,
        raise_on_model_failure=True,
    )
    fit_seconds = time.perf_counter() - started
    predict_seconds: dict[str, float] = {}
    for name in query_names:
        query = pd.read_parquet(job_dir / f"query__{name}.parquet")
        started = time.perf_counter()
        if problem_type == "regression":
            prediction = predictor.predict(query).to_numpy(dtype=np.float64)
        else:
            prediction = predictor.predict_proba(query, as_multiclass=True).to_numpy(dtype=np.float64)
        predict_seconds[name] = time.perf_counter() - started
        np.save(job_dir / f"prediction__{name}.npy", prediction, allow_pickle=False)
    model_names = predictor.model_names()
    model_info = predictor.model_info(model_names[0]) if model_names else {}
    telemetry = {
        "fit_seconds": fit_seconds,
        "predict_seconds_by_query": predict_seconds,
        "shared_fit_query_count": len(query_names),
        "n_estimators": 1,
        "fine_tune": fine_tune,
        "fine_tune_steps": 50 if fine_tune else 0,
        "preprocessing_policy": "autogluon_default",
        "precision": "bfloat16",
        "model_names": model_names,
        "model_info": {
            "model_type": model_info.get("model_type"),
            "num_features": model_info.get("num_features"),
            "memory_size": model_info.get("memory_size"),
        },
        "autogluon_tabular_version": importlib.metadata.version("autogluon.tabular"),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
        **checkpoint,
    }
    (job_dir / "telemetry.json").write_text(json.dumps(telemetry, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
