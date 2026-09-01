"""Nonlinear predictive metrics for randomized confirmed cover actions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from analyze_strength2_cover import strength1_family, strength2_family


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DRAWS = 128
BASE_SEED = 2026082802
METHODS = ("strength2", "iid16", "four_strength1", "four_seed_blocks")


def seed_for(dataset: str, model: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{BASE_SEED}:{dataset}:{model}".encode()).digest()[:8], "little")


def ids_for_actions(
    shape: tuple[int, int, int, int], rng: np.random.Generator, draws: int = DRAWS
) -> dict[str, np.ndarray]:
    feature, category, label, seed = shape
    family2 = strength2_family(category, label, seed)
    selected2 = family2[rng.integers(0, len(family2), size=draws)]
    ids2 = np.ravel_multi_index(np.moveaxis(selected2, -1, 0), shape)
    iid = rng.integers(0, np.prod(shape), size=(draws, 16))
    family1 = strength1_family(category, label, seed)
    selected1 = family1[rng.integers(0, len(family1), size=(draws, 4))]
    ids1 = np.ravel_multi_index(np.moveaxis(selected1.reshape(draws, 16, 4), -1, 0), shape)
    block_count = 16 // seed
    schema = np.column_stack([
        rng.integers(0, size, size=draws * block_count) for size in shape[:3]
    ]).reshape(draws, block_count, 3)
    blocks = np.empty((draws, block_count, seed, 4), dtype=int)
    blocks[..., :3] = schema[:, :, None, :]
    blocks[..., 3] = np.arange(seed)[None, None, :]
    block_ids = np.ravel_multi_index(np.moveaxis(blocks.reshape(draws, 16, 4), -1, 0), shape)
    return {"strength2": ids2, "iid16": iid, "four_strength1": ids1, "four_seed_blocks": block_ids}


def evaluate(y: np.ndarray, predictions: np.ndarray, task: str) -> dict[str, float]:
    if task == "regression":
        residual = predictions[..., 0] - y[None, :]
        mse = np.mean(residual**2, axis=1)
        denominator = float(np.mean((y - y.mean()) ** 2))
        return {"mse": float(mse.mean()), "r2": float((1 - mse / denominator).mean())}
    targets = np.eye(2)[y.astype(int)]
    brier = np.mean(np.sum((predictions - targets[None, ...]) ** 2, axis=-1), axis=1)
    draws = len(predictions)
    clipped = np.clip(predictions[np.arange(draws)[:, None], np.arange(len(y))[None, :], y.astype(int)[None, :]], 1e-12, 1)
    logloss = -np.log(clipped).mean(axis=1)
    accuracy = (np.argmax(predictions, axis=-1) == y[None, :]).mean(axis=1)
    auc = np.asarray([roc_auc_score(y, prediction[:, 1]) for prediction in predictions])
    return {
        "brier": float(brier.mean()), "logloss": float(logloss.mean()),
        "accuracy": float(accuracy.mean()), "roc_auc": float(auc.mean()),
    }


def main() -> None:
    screened = pd.read_csv(RESULTS / "validation_screened_cover_cells.csv")
    material = screened[screened.study == "strength2_confirmation"]
    input_dir = RESULTS / "tier1_confirmation"
    rows = []
    for cell in material.itertuples():
        archive = np.load(input_dir / f"{cell.dataset}__{cell.model}.npz")
        manifest = json.loads((input_dir / f"{cell.dataset}__{cell.model}.json").read_text())
        raw = archive["test_predictions"].astype(np.float64)
        flat = raw.reshape((-1,) + raw.shape[-2:])
        y = archive["test_y"]
        actions = ids_for_actions(tuple(raw.shape[:4]), np.random.default_rng(seed_for(cell.dataset, cell.model)))
        for method, ids in actions.items():
            predictions = flat[ids].mean(axis=1)
            rows.append({
                "dataset": cell.dataset, "model": cell.model,
                "task": manifest["task"], "method": method,
                **evaluate(y, predictions, manifest["task"]),
            })
    frame = pd.DataFrame(rows)
    summaries = {}
    for task, current in frame.groupby("task"):
        metric_names = [name for name in ("brier", "logloss", "accuracy", "roc_auc", "mse", "r2") if name in current and current[name].notna().any()]
        task_summary = {"cells": int(current[["dataset", "model"]].drop_duplicates().shape[0]), "metrics": {}}
        for metric in metric_names:
            pivot = current.pivot(index=["dataset", "model"], columns="method", values=metric)
            minimize = metric in {"brier", "logloss", "mse"}
            task_summary["metrics"][metric] = {
                "mean_by_method": {method: float(pivot[method].mean()) for method in METHODS},
                "strength2_better_cell_counts": {
                    method: int((pivot.strength2 < pivot[method]).sum() if minimize else (pivot.strength2 > pivot[method]).sum())
                    for method in METHODS if method != "strength2"
                },
            }
        summaries[task] = task_summary
    frame.to_csv(RESULTS / "predictive_metric_actions.csv", index=False)
    output = {"status": "complete", "draws_per_cell": DRAWS, "tasks": summaries}
    (RESULTS / "predictive_metric_actions_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
