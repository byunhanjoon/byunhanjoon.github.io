#!/usr/bin/env python3
"""Train-only conditional target-smoothness diagnostics for MPE."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from representations import load_task, split_row_indices, split_state_indices
from ridge_benchmark import DEFAULT_TASKS, fit_ridge, ordinary_design


HERE = Path(__file__).resolve().parent


def atomic_json(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def fold_assignment(row_ids: np.ndarray, split: int, folds: int = 5) -> np.ndarray:
    result = np.empty(len(row_ids), dtype=np.int64)
    for index, row_id in enumerate(row_ids):
        digest = hashlib.sha256(f"20260829|smoothness|{split}|{row_id}".encode()).digest()
        result[index] = int.from_bytes(digest[:8], "little") % folds
    return result


def finite_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or np.ptp(left) <= 1e-15 or np.ptp(right) <= 1e-15:
        return float("nan")
    return float(spearmanr(left, right).statistic)


def diagnostic_from_state_residuals(
    distance: np.ndarray, training_states: np.ndarray, residual_mean: np.ndarray
) -> dict[str, float]:
    d = distance[np.ix_(training_states, training_states)].astype(np.float64)
    upper = np.triu_indices(len(training_states), 1)
    pair_distance = d[upper]
    difference = np.abs(residual_mean[:, None] - residual_mean[None, :])[upper]
    distance_difference_spearman = finite_spearman(pair_distance, difference)

    off_diagonal = d.copy()
    np.fill_diagonal(off_diagonal, np.inf)
    nearest = np.argmin(off_diagonal, axis=1)
    nearest_correlation = finite_spearman(residual_mean, residual_mean[nearest])
    nearest_agreement = -float(np.mean(np.abs(residual_mean - residual_mean[nearest])))

    weights = np.divide(1.0, d, out=np.zeros_like(d), where=d > 1e-12)
    np.fill_diagonal(weights, 0.0)
    centered = residual_mean - residual_mean.mean()
    denominator = float(centered @ centered)
    moran = (
        float(len(centered) / weights.sum() * (centered @ weights @ centered) / denominator)
        if denominator > 1e-15 and weights.sum() > 0
        else float("nan")
    )

    semivariance = 0.5 * difference**2
    design = np.column_stack([np.ones(len(pair_distance)), pair_distance])
    slope = float(np.linalg.lstsq(design, semivariance, rcond=None)[0][1])
    return {
        "distance_residual_difference_spearman": distance_difference_spearman,
        "prespecified_smoothness": -distance_difference_spearman,
        "nearest_neighbor_residual_spearman": nearest_correlation,
        "nearest_neighbor_agreement": nearest_agreement,
        "moran_inverse_distance": moran,
        "empirical_variogram_slope": slope,
        "training_state_residual_std": float(np.std(residual_mean)),
        "state_pairs": int(len(pair_distance)),
    }


def run_cell(task_name: str, split: int, output: Path) -> None:
    cell = f"{task_name}__split{split}"
    path = output / f"{cell}.json"
    state_path = output / f"{cell}__state_residuals.parquet"
    if path.exists() and state_path.exists() and json.loads(path.read_text()).get("status") == "complete":
        print(f"resume smoothness {cell}", flush=True)
        return
    task = load_task(task_name)
    parts = split_row_indices(task, split)
    training_states = split_state_indices(task, split)["train"]
    train_rows = parts["train"]
    design = ordinary_design(task, train_rows)
    raw = pd.to_numeric(task.rows["target"], errors="raise").to_numpy(np.float64)
    mean = float(raw[train_rows].mean())
    scale = float(raw[train_rows].std()) or 1.0
    target = (raw - mean) / scale
    state_indices = task.row_state_indices()
    row_ids = task.rows["row_id"].astype(str).to_numpy()
    local_folds = fold_assignment(row_ids[train_rows], split)
    prediction = np.full(len(task.rows), np.nan, dtype=np.float64)
    fold_rows = []
    for fold in range(5):
        evaluation = train_rows[local_folds == fold]
        fitting = train_rows[local_folds != fold]
        if len(evaluation) == 0 or len(fitting) == 0:
            raise AssertionError("empty cross-fitting fold")
        fold_prediction, elapsed = fit_ridge(
            design, target, state_indices, fitting, evaluation, alpha=1.0
        )
        prediction[evaluation] = fold_prediction
        fold_rows.append({"fold": fold, "train_rows": len(fitting), "held_out_rows": len(evaluation), "seconds": elapsed})
    if not np.isfinite(prediction[train_rows]).all():
        raise AssertionError("cross-fitted training residuals incomplete")
    residual = target - prediction
    residual_means = np.asarray(
        [float(np.mean(residual[train_rows[state_indices[train_rows] == state]])) for state in training_states],
        dtype=np.float64,
    )
    diagnostic = diagnostic_from_state_residuals(task.distance, training_states, residual_means)
    pd.DataFrame(
        {
            "task": task_name,
            "source_unit": task.manifest["source_unit"],
            "split": split,
            "state_id": [task.state_ids[state] for state in training_states],
            "training_rows": [int(np.sum(state_indices[train_rows] == state)) for state in training_states],
            "cross_fitted_residual_mean": residual_means,
        }
    ).to_parquet(state_path, index=False, compression="zstd")
    payload = {
        "status": "complete", "task": task_name, "source_unit": task.manifest["source_unit"],
        "split": split, "split_seed": task.splits[str(split)]["seed"],
        "target_rows_used": "training only", "validation_labels_used": False,
        "test_labels_used": False, "crossfit_folds": 5, "ridge_alpha": 1.0,
        "folds": fold_rows, **diagnostic,
    }
    atomic_json(payload, path)
    print(f"complete smoothness {cell}", flush=True)


def consolidate(output: Path) -> None:
    rows = []
    states = []
    for path in sorted(output.glob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") == "complete":
            rows.append({key: value for key, value in payload.items() if key not in {"status", "folds"}})
        state_path = path.with_name(path.stem + "__state_residuals.parquet")
        if state_path.exists():
            states.append(pd.read_parquet(state_path))
    if rows:
        frame = pd.DataFrame(rows)
        frame.to_parquet(HERE / "raw" / "smoothness_results.parquet", index=False, compression="zstd")
        frame.to_csv(HERE / "raw" / "smoothness_results.csv", index=False)
    if states:
        pd.concat(states, ignore_index=True).to_parquet(
            HERE / "raw" / "smoothness_state_residuals.parquet", index=False, compression="zstd"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=DEFAULT_TASKS + ["all"], default="all")
    parser.add_argument("--split", type=int, choices=range(5))
    parser.add_argument("--consolidate-only", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "raw" / "smoothness_cells")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.consolidate_only:
        tasks = DEFAULT_TASKS if args.task == "all" else [args.task]
        splits = range(5) if args.split is None else [args.split]
        for task in tasks:
            for split in splits:
                run_cell(task, split, args.output)
    consolidate(args.output)


if __name__ == "__main__":
    main()
