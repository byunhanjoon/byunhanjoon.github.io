#!/usr/bin/env python3
"""Run the official TabDPT-Turbo 1.2 point-regression baseline.

TabDPT-Turbo exposes predictive means, but not a joint predictive covariance.
Accordingly, these cached predictions enter only the point-prediction table.
The package is installed in an isolated, pinned environment; this script does
not modify the environment used by the proposed method.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tabdpt import TabDPTRegressor

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
)


def load_items(stage: str, shard_index: int, num_shards: int) -> list[tuple[Any, Any]]:
    if stage == "eval":
        task_ids = [
            task
            for index, task in enumerate(CONFIG["evaluation_tasks"])
            if index % num_shards == shard_index
        ]
        return [load_openml_task(task) for task in task_ids]
    specs = CONFIG["development_datasets"] if stage == "dev" else CONFIG["application_datasets"]
    specs = [spec for index, spec in enumerate(specs) if index % num_shards == shard_index]
    return [(load_spec(spec), None) for spec in specs]


def episode_paths(stage: str, dataset_name: str) -> list[Path]:
    root = CACHE / "tabicl_singleton_episodes" / stage
    return sorted(root.glob(f"{slug(dataset_name)}__*.npz"))


def valid(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                {"mean", "metadata"}.issubset(data.files)
                and data["mean"].shape == (int(CONFIG["query_groups"]) * int(CONFIG["query_size"]),)
                and np.isfinite(data["mean"]).all()
            )
    except Exception:
        return False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(args: argparse.Namespace) -> None:
    # Construct once per shard so that all episodes reuse the loaded weights.
    # compile=False avoids shape-specific compilation overhead while preserving
    # exactly the same checkpoint and numerical model.
    model = TabDPTRegressor(
        device=args.device,
        use_flash=True,
        compile=False,
        context_reduction="subsample",
        verbose=False,
    )
    weight_path = Path(model.path).resolve()
    weight_record = {
        "path": str(weight_path),
        "bytes": weight_path.stat().st_size,
        "sha256": sha256_file(weight_path),
    }
    out_dir = CACHE / "tabdpt_episodes" / args.stage
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
                context = native_frame(dataset.X, context_indices)
                query = native_frame(dataset.X, query_indices)
                x_context, x_query = numeric_encode(context, query, dataset.categorical)
                y_context = dataset.y[context_indices].astype(np.float64)
                start = time.perf_counter()
                model.fit(x_context, y_context)
                mean_native = np.asarray(
                    model.predict(
                        x_query,
                        n_ensembles=int(args.n_ensembles),
                        context_size=None,
                        seed=20270321,
                    ),
                    dtype=np.float64,
                )
                elapsed = time.perf_counter() - start
                if mean_native.shape != (len(query_indices),) or not np.isfinite(mean_native).all():
                    raise ValueError(f"invalid prediction shape/values: {mean_native.shape}")
                mean = (mean_native - float(meta["metric_mean"])) / float(meta["metric_scale"])
                out_meta = {
                    **meta,
                    "model": "TabDPT-Turbo-1.2",
                    "package": "tabdpt==1.2.0",
                    "weight": weight_record,
                    "n_ensembles": int(args.n_ensembles),
                    "context_size_argument": None,
                    "numeric_encoding": "context-only category/imputation/standardization",
                    "compile": False,
                    "elapsed_seconds": elapsed,
                }
                atomic_savez(
                    output_path,
                    mean=mean.astype(np.float32),
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
        CACHE / "logs" / f"extract_tabdpt_{args.stage}_shard{args.shard_index}.json",
        {
            "stage": args.stage,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "n_ensembles": int(args.n_ensembles),
            "weight": weight_record,
            "records": records,
            "failures": failures,
            "environment": environment_record(),
        },
    )
    if failures:
        raise SystemExit(f"{len(failures)} TabDPT failures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["dev", "eval", "app"], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-ensembles", type=int, default=8)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
