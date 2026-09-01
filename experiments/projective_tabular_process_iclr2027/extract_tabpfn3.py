#!/usr/bin/env python3
"""Run the pinned official TabPFN-3 regressor on benchmark episodes.

Execute this file with the isolated ``tabpfn3_env`` (tabpfn==8.5.0).  The
default v3 checkpoint is pinned explicitly rather than relying on a moving
package default.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tabpfn import TabPFNRegressor

from common import (
    CACHE,
    CONFIG,
    atomic_json,
    atomic_savez,
    environment_record,
    load_openml_task,
    load_spec,
    native_frame,
    slug,
)


MODEL_PATH = CACHE / "tabpfn3_models" / "tabpfn-v3-regressor-v3_default.ckpt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_categorical_fallback(context, query, categorical):
    """Context-only category coding for package validation edge cases."""
    train, test = context.copy(), query.copy()
    for column, is_categorical in zip(context.columns, categorical):
        if not is_categorical and not pd.api.types.is_object_dtype(context[column]):
            continue
        train_values = context[column].astype("string").fillna("<NA>")
        categories = {value: index for index, value in enumerate(pd.unique(train_values))}
        train[column] = train_values.map(categories).fillna(-1).astype(np.float64)
        test[column] = (
            query[column].astype("string").fillna("<NA>").map(categories).fillna(-1).astype(np.float64)
        )
    return train, test


def load_items(stage: str, shard_index: int, num_shards: int) -> list[tuple[Any, Any]]:
    if stage == "eval":
        task_ids = [
            task for index, task in enumerate(CONFIG["evaluation_tasks"])
            if index % num_shards == shard_index
        ]
        return [load_openml_task(task) for task in task_ids]
    specs = CONFIG["development_datasets"] if stage == "dev" else CONFIG["application_datasets"]
    specs = [spec for index, spec in enumerate(specs) if index % num_shards == shard_index]
    return [(load_spec(spec), None) for spec in specs]


def episode_paths(stage: str, dataset_name: str) -> list[Path]:
    return sorted(
        (CACHE / "tabicl_singleton_episodes" / stage).glob(f"{slug(dataset_name)}__*.npz")
    )


def valid(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                {"mean", "variance", "metadata"}.issubset(data.files)
                and np.isfinite(data["mean"]).all()
                and np.isfinite(data["variance"]).all()
            )
    except Exception:
        return False


def main(args: argparse.Namespace) -> None:
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"pinned TabPFN-3 checkpoint missing: {MODEL_PATH}")
    checkpoint = {
        "path": str(MODEL_PATH),
        "bytes": MODEL_PATH.stat().st_size,
        "sha256": sha256_file(MODEL_PATH),
    }
    model = TabPFNRegressor(
        n_estimators=8,
        auto_scale_n_estimators=False,
        device=args.device,
        model_path=str(MODEL_PATH),
        random_state=20270331,
        fit_mode="fit_preprocessors",
        show_progress_bar=False,
    )
    out_dir = CACHE / "tabpfn3_episodes" / args.stage
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for dataset, _ in load_items(args.stage, args.shard_index, args.num_shards):
        paths = episode_paths(args.stage, dataset.name)
        print(f"[{args.stage}] {dataset.name} episodes={len(paths)}", flush=True)
        if not paths:
            failures.append({"dataset": dataset.name, "error": "no matching singleton episode cache"})
            continue
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
                x_context = native_frame(dataset.X, context_indices)
                x_query = native_frame(dataset.X, query_indices)
                dropped_columns = [column for column in x_context if x_context[column].isna().all()]
                if dropped_columns:
                    x_context = x_context.drop(columns=dropped_columns)
                    x_query = x_query.drop(columns=dropped_columns)
                if x_context.shape[1] == 0:
                    raise ValueError("all input columns are missing in the context")
                y_context = dataset.y[context_indices]
                start = time.perf_counter()
                categorical_fallback = False
                try:
                    model.fit(x_context, y_context)
                    output = model.predict(x_query, output_type="full")
                except Exception as initial_error:
                    if "cast object" not in str(initial_error).lower():
                        raise
                    categorical_fallback = True
                    safe_context, safe_query = numeric_categorical_fallback(
                        x_context, x_query, dataset.categorical
                    )
                    model.fit(safe_context, y_context)
                    output = model.predict(safe_query, output_type="full")
                elapsed = time.perf_counter() - start
                mean_native = np.asarray(output["mean"], dtype=np.float64)
                criterion = output["criterion"]
                logits = output["logits"].to(dtype=criterion.borders.dtype)
                variance_native = (
                    criterion.variance(logits).detach().float().cpu().numpy().astype(np.float64)
                )
                mean = (mean_native - float(meta["metric_mean"])) / float(meta["metric_scale"])
                variance = variance_native / float(meta["metric_scale"]) ** 2
                variance = np.maximum(variance, 1e-12)
                if mean.shape != (len(query_indices),) or variance.shape != mean.shape:
                    raise ValueError(f"invalid output shapes mean={mean.shape} variance={variance.shape}")
                out_meta = {
                    **meta,
                    "model": "TabPFN-3",
                    "package": "tabpfn==8.5.0",
                    "checkpoint": checkpoint,
                    "n_estimators": 8,
                    "auto_scale_n_estimators": False,
                    "dropped_all_missing_context_columns": dropped_columns,
                    "categorical_numeric_fallback": categorical_fallback,
                    "elapsed_seconds": elapsed,
                }
                atomic_savez(
                    output_path,
                    mean=mean.astype(np.float32),
                    variance=variance.astype(np.float32),
                    metadata=np.asarray(json.dumps(out_meta)),
                )
                records.append({"path": str(output_path), "status": "written", "elapsed_seconds": elapsed})
                print(f"  {source_path.stem} {elapsed:.2f}s", flush=True)
            except Exception as exc:
                failure = {
                    "dataset": dataset.name,
                    "episode": str(source_path),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                print(f"  FAILED {failure}", flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    atomic_json(
        CACHE / "logs" / f"extract_tabpfn3_{args.stage}_shard{args.shard_index}.json",
        {
            "stage": args.stage,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "checkpoint": checkpoint,
            "records": records,
            "failures": failures,
            "environment": environment_record(),
        },
    )
    if failures:
        raise SystemExit(f"{len(failures)} TabPFN-3 failures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["dev", "eval", "app"], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
