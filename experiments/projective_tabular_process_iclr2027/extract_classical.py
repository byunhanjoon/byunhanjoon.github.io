#!/usr/bin/env python3
"""Run coherent classical joint-predictive baselines on the frozen episodes."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from catboost import CatBoostRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.linear_model import BayesianRidge

from common import (
    CACHE,
    CONFIG,
    atomic_json,
    atomic_savez,
    environment_record,
    load_openml_task,
    load_spec,
    native_frame,
    numeric_encode,
    slug,
    stable_seed,
)


METHODS = ["bayesian_linear", "gp_rbf", "gp_matern32", "catboost_process"]


def items(stage: str, shard_index: int, num_shards: int) -> list[tuple[Any, Any]]:
    if stage == "eval":
        task_ids = [task for i, task in enumerate(CONFIG["evaluation_tasks"]) if i % num_shards == shard_index]
        return [load_openml_task(task) for task in task_ids]
    specs = CONFIG["development_datasets"] if stage == "dev" else CONFIG["application_datasets"]
    specs = [spec for i, spec in enumerate(specs) if i % num_shards == shard_index]
    return [(load_spec(spec), None) for spec in specs]


def bayesian_linear(X: np.ndarray, y: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = np.concatenate([X.astype(np.float64), np.ones((len(X), 1))], axis=1)
    test = np.concatenate([query.astype(np.float64), np.ones((len(query), 1))], axis=1)
    model = BayesianRidge(fit_intercept=False, tol=1e-5, max_iter=500)
    model.fit(train, y)
    mean = test @ model.coef_
    covariance = test @ model.sigma_ @ test.T
    covariance += np.eye(len(test)) / max(float(model.alpha_), 1e-10)
    return mean, covariance


def gaussian_process(
    X: np.ndarray, y: np.ndarray, query: np.ndarray, kind: str, seed: int
) -> tuple[np.ndarray, np.ndarray, bool, str]:
    length = math.sqrt(max(X.shape[1], 1))
    smooth = RBF(length, (1e-2, 1e3)) if kind == "rbf" else Matern(length, (1e-2, 1e3), nu=1.5)
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * smooth + WhiteKernel(0.1, (1e-4, 10.0))
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-8,
        normalize_y=False,
        n_restarts_optimizer=0,
        random_state=seed,
    )
    fallback = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        try:
            model.fit(X.astype(np.float64), y.astype(np.float64))
        except Exception:
            fallback = True
            fallback_smooth = RBF(length) if kind == "rbf" else Matern(length, nu=1.5)
            model = GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * fallback_smooth + WhiteKernel(0.1),
                alpha=1e-6,
                normalize_y=False,
                optimizer=None,
            ).fit(X.astype(np.float64), y.astype(np.float64))
    mean, covariance = model.predict(query.astype(np.float64), return_cov=True)
    return mean, covariance, fallback, str(model.kernel_)


def catboost_process(
    X: np.ndarray,
    y: np.ndarray,
    query: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    predictions, context_predictions, inbags = [], [], []
    size = int(CONFIG["catboost"]["ensemble_size"])
    for member in range(size):
        indices = rng.choice(len(X), size=len(X), replace=True)
        inbag = np.zeros(len(X), dtype=bool)
        inbag[np.unique(indices)] = True
        if float(np.std(y[indices])) < 1e-12:
            # CatBoost deliberately rejects a constant-label bootstrap draw.
            # That bootstrap member's fitted function is the constant itself.
            constant = float(np.mean(y[indices]))
            predictions.append(np.full(len(query), constant, dtype=np.float64))
            context_predictions.append(np.full(len(X), constant, dtype=np.float64))
            inbags.append(inbag)
            continue
        model = CatBoostRegressor(
            **config,
            loss_function="RMSE",
            verbose=False,
            random_seed=stable_seed(seed, member),
            allow_writing_files=False,
            thread_count=1,
            l2_leaf_reg=3.0,
        )
        model.fit(X[indices], y[indices])
        predictions.append(np.asarray(model.predict(query), dtype=np.float64))
        context_predictions.append(np.asarray(model.predict(X), dtype=np.float64))
        inbags.append(inbag)
    members = np.stack(predictions)
    context_members = np.stack(context_predictions)
    inbags = np.stack(inbags)
    mean = members.mean(axis=0)
    centered = members - mean
    covariance = centered.T @ centered / size
    oob_prediction = np.empty(len(X), dtype=np.float64)
    for row in range(len(X)):
        available = ~inbags[:, row]
        if not available.any():
            available = np.ones(size, dtype=bool)
        oob_prediction[row] = context_members[available, row].mean()
    noise = float(np.mean((y - oob_prediction) ** 2))
    noise = max(noise, 1e-4)
    covariance += noise * np.eye(len(query))
    return mean, covariance


def valid(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            return data["means"].shape[0] == len(METHODS) and np.isfinite(data["covariances"]).all()
    except Exception:
        return False


def main(args: argparse.Namespace) -> None:
    hpo = json.loads((CACHE / "classical_hpo" / "selected.json").read_text())
    out_dir = CACHE / "classical_episodes" / args.stage
    out_dir.mkdir(parents=True, exist_ok=True)
    records, failures = [], []
    for dataset, _ in items(args.stage, args.shard_index, args.num_shards):
        paths = sorted((CACHE / "tabicl_episodes" / args.stage).glob(f"{slug(dataset.name)}__*.npz"))
        print(f"[{args.stage}] {dataset.name} episodes={len(paths)}", flush=True)
        for source_path in paths:
            output_path = out_dir / source_path.name
            if output_path.exists() and valid(output_path):
                records.append({"path": str(output_path), "status": "cached"})
                continue
            try:
                with np.load(source_path, allow_pickle=False) as source:
                    meta = json.loads(str(source["metadata"].item()))
                    context_indices = source["context_indices"].astype(np.int64)
                    query_indices = source["query_indices"].astype(np.int64)
                X_context_frame = native_frame(dataset.X, context_indices)
                X_query_frame = native_frame(dataset.X, query_indices)
                X_context, X_query = numeric_encode(X_context_frame, X_query_frame, dataset.categorical)
                y_native = dataset.y[context_indices].astype(np.float64)
                context_mean = float(y_native.mean())
                context_scale = float(y_native.std())
                if not np.isfinite(context_scale) or context_scale < 1e-10:
                    context_scale = 1.0
                y = (y_native - context_mean) / context_scale
                metric_mean = float(meta["metric_mean"])
                metric_scale = float(meta["metric_scale"])
                ratio = context_scale / metric_scale
                offset = (context_mean - metric_mean) / metric_scale
                means, covariances, times, extras = [], [], [], []

                start = time.perf_counter()
                mean, covariance = bayesian_linear(X_context, y, X_query)
                times.append(time.perf_counter() - start)
                means.append(offset + ratio * mean)
                covariances.append(ratio**2 * covariance)
                extras.append({})

                for kind in ("rbf", "matern32"):
                    start = time.perf_counter()
                    mean, covariance, fallback, kernel = gaussian_process(
                        X_context,
                        y,
                        X_query,
                        "rbf" if kind == "rbf" else "matern32",
                        stable_seed("gp", dataset.name, source_path.name, kind),
                    )
                    times.append(time.perf_counter() - start)
                    means.append(offset + ratio * mean)
                    covariances.append(ratio**2 * covariance)
                    extras.append({"fallback": fallback, "kernel": kernel})

                cat_config = hpo["selected"][str(meta["context_size"])]
                start = time.perf_counter()
                mean, covariance = catboost_process(
                    X_context,
                    y,
                    X_query,
                    cat_config,
                    stable_seed("catboost-process", dataset.name, source_path.name),
                )
                times.append(time.perf_counter() - start)
                means.append(offset + ratio * mean)
                covariances.append(ratio**2 * covariance)
                extras.append({"config": cat_config})

                means_array = np.stack(means)
                covariance_array = np.stack(covariances)
                covariance_array = 0.5 * (covariance_array + covariance_array.transpose(0, 2, 1))
                for covariance in covariance_array:
                    eigen_min = float(np.linalg.eigvalsh(covariance).min())
                    if eigen_min < 1e-9:
                        covariance += (1e-9 - eigen_min) * np.eye(len(covariance))
                if not np.isfinite(means_array).all() or not np.isfinite(covariance_array).all():
                    raise FloatingPointError("non-finite classical prediction")
                out_meta = {**meta, "methods": METHODS, "elapsed_seconds": times, "extras": extras}
                atomic_savez(
                    output_path,
                    means=means_array.astype(np.float32),
                    covariances=covariance_array.astype(np.float32),
                    metadata=np.asarray(json.dumps(out_meta)),
                )
                records.append({"path": str(output_path), "status": "written", "elapsed_seconds": times})
                print(f"  {source_path.stem} total={sum(times):.2f}s", flush=True)
            except Exception as exc:
                failure = {"dataset": dataset.name, "episode": str(source_path), "error": repr(exc), "traceback": traceback.format_exc()}
                failures.append(failure)
                print(f"  FAILED {failure}", flush=True)
            gc.collect()
    atomic_json(
        CACHE / "logs" / f"extract_classical_{args.stage}_shard{args.shard_index}.json",
        {
            "stage": args.stage,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "records": records,
            "failures": failures,
            "environment": environment_record(),
        },
    )
    if failures:
        raise SystemExit(f"{len(failures)} classical failures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["dev", "eval", "app"], required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
