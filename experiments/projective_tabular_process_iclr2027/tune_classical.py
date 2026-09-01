#!/usr/bin/env python3
"""Select CatBoost hyperparameters on the development panel only."""

from __future__ import annotations

import itertools
import json
import math
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from common import CACHE, CONFIG, atomic_json, load_spec, native_frame, numeric_encode, slug, stable_seed


def configs() -> list[dict[str, float | int]]:
    return [
        {"iterations": int(iterations), "depth": int(depth), "learning_rate": float(rate)}
        for iterations, depth, rate in itertools.product(
            CONFIG["catboost"]["iterations"], CONFIG["catboost"]["depth"], CONFIG["catboost"]["learning_rate"]
        )
    ]


def main() -> None:
    rows = []
    grid = configs()
    for spec in CONFIG["development_datasets"]:
        dataset = load_spec(spec)
        paths = sorted((CACHE / "tabicl_episodes" / "dev").glob(f"{slug(dataset.name)}__*.npz"))
        for path in paths:
            with np.load(path, allow_pickle=False) as episode:
                meta = json.loads(str(episode["metadata"].item()))
                context_indices = episode["context_indices"].astype(np.int64)
                query_indices = episode["query_indices"].astype(np.int64)
                target = episode["target"].astype(np.float64)
            X_context_frame = native_frame(dataset.X, context_indices)
            X_query_frame = native_frame(dataset.X, query_indices)
            X_context, X_query = numeric_encode(X_context_frame, X_query_frame, dataset.categorical)
            y_context = dataset.y[context_indices].astype(np.float64)
            y_mean = float(y_context.mean())
            y_scale = float(y_context.std())
            if y_scale < 1e-10:
                y_scale = 1.0
            y_standard = (y_context - y_mean) / y_scale
            for index, config in enumerate(grid):
                start = time.perf_counter()
                model = CatBoostRegressor(
                    **config,
                    loss_function="RMSE",
                    verbose=False,
                    random_seed=stable_seed("catboost-hpo", dataset.name, path.name, index),
                    allow_writing_files=False,
                    thread_count=2,
                    l2_leaf_reg=3.0,
                )
                model.fit(X_context, y_standard)
                prediction_native = y_mean + y_scale * model.predict(X_query)
                prediction = (prediction_native - float(meta["metric_mean"])) / float(meta["metric_scale"])
                rows.append(
                    {
                        "dataset": dataset.name,
                        "episode": path.name,
                        "context_size": int(meta["context_size"]),
                        "config_index": index,
                        **config,
                        "squared_error": float(np.mean((target - prediction) ** 2)),
                        "elapsed_seconds": time.perf_counter() - start,
                    }
                )
        print(f"tuned {dataset.name}", flush=True)
    cells = pd.DataFrame(rows)
    selected = {}
    summary_rows = []
    for context_size in CONFIG["context_sizes"]:
        subset = cells[cells["context_size"] == context_size]
        by_dataset = subset.groupby(["config_index", "dataset"], as_index=False)["squared_error"].mean()
        scores = by_dataset.groupby("config_index")["squared_error"].mean()
        best_index = int(scores.idxmin())
        selected[str(context_size)] = grid[best_index]
        for index, score in scores.items():
            summary_rows.append(
                {
                    "context_size": int(context_size),
                    "config_index": int(index),
                    **grid[int(index)],
                    "dataset_balanced_mse": float(score),
                    "selected": int(index) == best_index,
                }
            )
    out = CACHE / "classical_hpo"
    out.mkdir(parents=True, exist_ok=True)
    cells.to_parquet(out / "catboost_hpo_cells.parquet", index=False)
    pd.DataFrame(summary_rows).to_csv(out / "catboost_hpo_summary.csv", index=False)
    payload = {
        "objective": "dataset-balanced query-row MSE on six development datasets",
        "fits": int(len(cells)),
        "grid_size": len(grid),
        "selected": selected,
        "total_fit_seconds": float(cells["elapsed_seconds"].sum()),
    }
    atomic_json(out / "selected.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
