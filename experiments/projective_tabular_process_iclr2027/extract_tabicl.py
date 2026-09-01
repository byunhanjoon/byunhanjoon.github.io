#!/usr/bin/env python3
"""Extract frozen TabICLv2 marginals and row representations episode by episode."""

from __future__ import annotations

import argparse
import gc
import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tabicl import TabICLRegressor

from common import (
    CACHE,
    CONFIG,
    Dataset,
    atomic_json,
    atomic_savez,
    environment_record,
    load_openml_task,
    load_spec,
    make_coefficients,
    metric_affine,
    native_frame,
    numeric_encode,
    sha256_array,
    slug,
    stable_seed,
)


def eval_splits(dataset: Dataset, task: Any) -> list[tuple[str, np.ndarray, np.ndarray]]:
    result = []
    for fold in CONFIG["evaluation_folds"]:
        train, test = task.get_train_test_split_indices(repeat=0, fold=int(fold), sample=0)
        result.append((f"official_repeat0_fold{fold}", np.asarray(train), np.asarray(test)))
    return result


def random_splits(dataset: Dataset) -> list[tuple[str, np.ndarray, np.ndarray]]:
    indices = np.arange(len(dataset.y))
    result = []
    for seed in CONFIG["development_splits"]:
        train, test = train_test_split(indices, test_size=0.30, random_state=int(seed), shuffle=True)
        result.append((f"random70_30_seed{seed}", np.asarray(train), np.asarray(test)))
    return result


def episode_indices(
    dataset_name: str,
    split_name: str,
    train_pool: np.ndarray,
    test_pool: np.ndarray,
    replicate: int,
) -> tuple[np.ndarray, np.ndarray]:
    maximum = max(CONFIG["context_sizes"])
    n_query = int(CONFIG["query_groups"]) * int(CONFIG["query_size"])
    if len(train_pool) < maximum or len(test_pool) < n_query:
        raise ValueError(
            f"insufficient rows for {dataset_name}/{split_name}: train={len(train_pool)}, test={len(test_pool)}"
        )
    rng_context = np.random.default_rng(stable_seed("context", dataset_name, split_name, replicate))
    rng_query = np.random.default_rng(stable_seed("query", dataset_name, split_name, replicate))
    context_order = rng_context.choice(train_pool, size=maximum, replace=False)
    query = rng_query.choice(test_pool, size=n_query, replace=False)
    return context_order.astype(np.int64), query.astype(np.int64)


class TabICLExtractor:
    def __init__(self, device: str):
        self.device = device
        self.model: TabICLRegressor | None = None
        self.quantile_alphas = ((np.arange(int(CONFIG["tabicl"]["quantile_grid_size"])) + 0.5) / int(CONFIG["tabicl"]["quantile_grid_size"])).tolist()

    def _new_model(self) -> TabICLRegressor:
        return TabICLRegressor(
            n_estimators=int(CONFIG["tabicl"]["n_estimators"]),
            batch_size=int(CONFIG["tabicl"]["n_estimators"]),
            checkpoint_version=CONFIG["tabicl"]["checkpoint"],
            device=self.device,
            random_state=20270100,
            use_fa3=False,
            use_amp="auto",
            verbose=False,
        )

    def fit_context(self, X_context, y_context: np.ndarray) -> None:
        if self.model is None:
            self.model = self._new_model()
            self.model.fit(X_context, y_context)
            shared_model = self.model.model_
            model_path = self.model.model_path_

            def reuse_loaded_model() -> None:
                assert self.model is not None
                self.model.model_ = shared_model
                self.model.model_path_ = model_path
                self.model.model_.eval()

            self.model._load_model = reuse_loaded_model  # type: ignore[method-assign]
        else:
            self.model.fit(X_context, y_context)

    def predict_fitted(
        self, X_query, y_context: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        if self.model is None:
            raise RuntimeError("fit_context must be called before predict_fitted")
        start = time.perf_counter()

        captured: list[np.ndarray] = []

        def capture(_module, args) -> None:
            captured.append(args[0].detach().float().cpu().numpy())

        handle = self.model.model_.icl_predictor.decoder.register_forward_pre_hook(capture)
        try:
            output = self.model.predict(
                X_query,
                output_type=["mean", "quantiles"],
                alphas=self.quantile_alphas,
            )
        finally:
            handle.remove()
        elapsed = time.perf_counter() - start

        mean = np.asarray(output["mean"], dtype=np.float64)
        quantiles = np.asarray(output["quantiles"], dtype=np.float64)
        variance = np.mean((quantiles - mean[:, None]) ** 2, axis=1)
        floor = max(float(np.std(y_context)) ** 2 * 1e-8, 1e-12)
        variance = np.maximum(variance, floor)

        if not captured:
            raise RuntimeError("TabICL decoder hook captured no representations")
        views = np.concatenate(captured, axis=0)
        n_query = len(X_query)
        if views.shape[0] != int(CONFIG["tabicl"]["n_estimators"]):
            raise RuntimeError(f"expected {CONFIG['tabicl']['n_estimators']} views, found {views.shape}")
        if views.shape[1] < n_query:
            raise RuntimeError(f"representation sequence shorter than query: {views.shape}")
        query_hidden = views[:, -n_query:, :]
        return mean, variance, query_hidden, elapsed

    def fit_predict(
        self,
        X_context,
        y_context: np.ndarray,
        X_query,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        start = time.perf_counter()
        self.fit_context(X_context, y_context)
        mean, variance, hidden, _ = self.predict_fitted(X_query, y_context)
        return mean, variance, hidden, time.perf_counter() - start

    def fit_predict_singletons(
        self,
        X_context,
        y_context: np.ndarray,
        X_query,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Evaluate every query row alone after fitting the context once."""
        start = time.perf_counter()
        self.fit_context(X_context, y_context)
        means, variances, hidden_rows = [], [], []
        for row in range(len(X_query)):
            if hasattr(X_query, "iloc"):
                query_row = X_query.iloc[[row]].reset_index(drop=True)
            else:
                query_row = X_query[row : row + 1]
            mean, variance, hidden, _ = self.predict_fitted(query_row, y_context)
            means.append(float(mean[0]))
            variances.append(float(variance[0]))
            hidden_rows.append(hidden[:, 0, :])
        return (
            np.asarray(means, dtype=np.float64),
            np.asarray(variances, dtype=np.float64),
            np.stack(hidden_rows, axis=1),
            time.perf_counter() - start,
        )


def valid_existing(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {"hidden", "mean", "variance", "target", "coefficients", "query_numeric"}
            return required.issubset(data.files) and np.isfinite(data["mean"]).all()
    except Exception:
        return False


def write_episode(
    extractor: TabICLExtractor,
    stage: str,
    dataset: Dataset,
    split_name: str,
    train_pool: np.ndarray,
    test_pool: np.ndarray,
    replicate: int,
    context_size: int,
    out_dir: Path,
    query_mode: str,
) -> dict[str, Any]:
    filename = f"{slug(dataset.name)}__{slug(split_name)}__rep{replicate}__n{context_size}.npz"
    path = out_dir / filename
    if path.exists() and valid_existing(path):
        return {"path": str(path), "status": "cached"}

    context_order, query_indices = episode_indices(
        dataset.name, split_name, train_pool, test_pool, replicate
    )
    context_indices = context_order[:context_size]
    X_context = native_frame(dataset.X, context_indices)
    X_query = native_frame(dataset.X, query_indices)
    y_context = dataset.y[context_indices].astype(np.float64)
    metric_mean, metric_scale = metric_affine(dataset.y[train_pool])
    target = (dataset.y[query_indices] - metric_mean) / metric_scale
    _, query_numeric = numeric_encode(X_context, X_query, dataset.categorical)

    prediction_function = (
        extractor.fit_predict_singletons if query_mode == "singleton" else extractor.fit_predict
    )
    mean_native, variance_native, hidden, elapsed = prediction_function(X_context, y_context, X_query)
    mean = (mean_native - metric_mean) / metric_scale
    variance = variance_native / metric_scale**2
    families, coefficients = make_coefficients(
        stable_seed("coefficients", dataset.name, split_name, replicate)
    )

    if mean.shape != target.shape or hidden.shape[1] != len(target):
        raise AssertionError("episode shape mismatch")
    if not np.isfinite(mean).all() or not np.isfinite(variance).all() or not np.isfinite(hidden).all():
        raise FloatingPointError("non-finite TabICL output")

    metadata = {
        "stage": stage,
        "dataset": dataset.name,
        "source_id": dataset.source_id,
        "target_name": dataset.target,
        "split": split_name,
        "replicate": replicate,
        "context_size": context_size,
        "train_pool_size": int(len(train_pool)),
        "test_pool_size": int(len(test_pool)),
        "query_size": int(len(query_indices)),
        "n_features": int(dataset.X.shape[1]),
        "metric_mean": metric_mean,
        "metric_scale": metric_scale,
        "context_index_sha256": sha256_array(context_indices),
        "query_index_sha256": sha256_array(query_indices),
        "train_pool_sha256": sha256_array(np.sort(train_pool)),
        "test_pool_sha256": sha256_array(np.sort(test_pool)),
        "coefficients_sha256": sha256_array(coefficients),
        "families": families,
        "elapsed_seconds": elapsed,
        "views": int(hidden.shape[0]),
        "hidden_dim": int(hidden.shape[-1]),
        "query_mode": query_mode,
    }
    atomic_savez(
        path,
        hidden=hidden.astype(np.float16),
        mean=mean.astype(np.float32),
        variance=variance.astype(np.float32),
        target=target.astype(np.float32),
        coefficients=coefficients.astype(np.float32),
        query_numeric=query_numeric.astype(np.float32),
        context_indices=context_indices,
        query_indices=query_indices,
        metadata=np.asarray(json.dumps(metadata)),
    )
    return {"path": str(path), "status": "written", **metadata}


def stage_items(stage: str, shard_index: int, num_shards: int) -> list[tuple[Dataset, Any]]:
    if stage == "eval":
        task_ids = [
            task_id
            for position, task_id in enumerate(CONFIG["evaluation_tasks"])
            if position % num_shards == shard_index
        ]
        return [load_openml_task(task_id) for task_id in task_ids]
    specs = CONFIG["development_datasets"] if stage == "dev" else CONFIG["application_datasets"]
    selected = [spec for position, spec in enumerate(specs) if position % num_shards == shard_index]
    return [(load_spec(spec), None) for spec in selected]


def run(args: argparse.Namespace) -> None:
    episode_root = "tabicl_singleton_episodes" if args.query_mode == "singleton" else "tabicl_episodes"
    out_dir = CACHE / episode_root / args.stage
    log_dir = CACHE / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    extractor = TabICLExtractor(args.device)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    selected = stage_items(args.stage, args.shard_index, args.num_shards)
    for dataset, task in selected:
        print(f"[{args.stage}] dataset={dataset.name} rows={len(dataset.y)} features={dataset.X.shape[1]}", flush=True)
        splits = eval_splits(dataset, task) if args.stage == "eval" else random_splits(dataset)
        replicates = int(CONFIG["context_replicates"] if args.stage == "eval" else CONFIG["development_context_replicates"])
        for split_name, train_pool, test_pool in splits:
            for replicate in range(replicates):
                for context_size in CONFIG["context_sizes"]:
                    try:
                        record = write_episode(
                            extractor,
                            args.stage,
                            dataset,
                            split_name,
                            train_pool,
                            test_pool,
                            replicate,
                            int(context_size),
                            out_dir,
                            args.query_mode,
                        )
                        records.append(record)
                        print(
                            f"  {split_name} rep={replicate} n={context_size} {record['status']}"
                            + (f" {record.get('elapsed_seconds', 0):.2f}s" if record["status"] == "written" else ""),
                            flush=True,
                        )
                    except Exception as exc:
                        failure = {
                            "dataset": dataset.name,
                            "split": split_name,
                            "replicate": replicate,
                            "context_size": context_size,
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                        failures.append(failure)
                        print(f"  FAILED {failure}", flush=True)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    payload = {
        "stage": args.stage,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "device": args.device,
        "query_mode": args.query_mode,
        "records": records,
        "failures": failures,
        "environment": environment_record(),
    }
    atomic_json(
        log_dir / f"extract_tabicl_{args.query_mode}_{args.stage}_shard{args.shard_index}.json",
        payload,
    )
    if failures:
        raise SystemExit(f"{len(failures)} episode failures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["dev", "eval", "app"], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--query-mode", choices=["batched", "singleton"], default="batched")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must be in [0, num-shards)")
    return args


if __name__ == "__main__":
    run(parse_args())
