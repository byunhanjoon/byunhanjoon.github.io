#!/usr/bin/env python3
"""Isolated pytabkit worker used by the reparameterization audit."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    job_dir = args.request.parent
    X_train = pd.read_parquet(job_dir / "train.parquet")
    y_train = np.load(job_dir / "y_train.npy", allow_pickle=False)
    numeric_columns = list(X_train.select_dtypes(include=[np.number]).columns)
    numeric_fill = X_train[numeric_columns].median(axis=0).fillna(0.0)
    X_train.loc[:, numeric_columns] = X_train[numeric_columns].fillna(numeric_fill)

    from pytabkit.models.sklearn.sklearn_interfaces import (
        RealMLP_TD_Classifier,
        RealMLP_TD_Regressor,
        TabM_D_Classifier,
        TabM_D_Regressor,
    )
    import torch

    classes = {
        ("tabm_default", "classification"): TabM_D_Classifier,
        ("tabm_default", "regression"): TabM_D_Regressor,
        ("realmlp_default", "classification"): RealMLP_TD_Classifier,
        ("realmlp_default", "regression"): RealMLP_TD_Regressor,
    }
    problem_key = "regression" if request["problem_type"] == "regression" else "classification"
    cls = classes[(request["model"], problem_key)]
    model = cls(
        device=request["device"],
        random_state=int(request["seed"]),
        n_cv=1,
        n_refit=0,
        verbosity=0,
    )
    if str(request["device"]).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(torch.device(request["device"]))
    started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started
    predict_seconds = {}
    for key in request["query_names"]:
        query = pd.read_parquet(job_dir / f"query__{key}.parquet")
        query.loc[:, numeric_columns] = query[numeric_columns].fillna(numeric_fill)
        started = time.perf_counter()
        prediction = model.predict(query) if problem_key == "regression" else model.predict_proba(query)
        predict_seconds[key] = time.perf_counter() - started
        np.save(job_dir / f"prediction__{key}.npy", np.asarray(prediction), allow_pickle=False)
    telemetry = {
        "fit_seconds": fit_seconds,
        "predict_seconds_by_query": predict_seconds,
        "adapter": "isolated_pytabkit_worker",
        "pytabkit_version": importlib.metadata.version("pytabkit"),
        "preprocessing_policy": "train_median_imputation_then_pytabkit_recommended_default",
        "n_cv": 1,
        "n_refit": 0,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(torch.device(request["device"])))
            if str(request["device"]).startswith("cuda") and torch.cuda.is_available()
            else None
        ),
    }
    (job_dir / "telemetry.json").write_text(json.dumps(telemetry, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
