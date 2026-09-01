"""High-precision streaming audit of nonlinear predictive metrics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from analyze_predictive_metrics import METHODS, ids_for_actions


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DRAWS = 2_048
BATCH = 32
BASE_SEED = 2026082819
CONTROLS = tuple(method for method in METHODS if method != "strength2")


def seed_for(dataset: str, model: str) -> int:
    digest = hashlib.sha256(f"{BASE_SEED}:{dataset}:{model}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def metric_draws(y: np.ndarray, predictions: np.ndarray, task: str) -> dict[str, np.ndarray]:
    if task == "regression":
        mse = np.mean((predictions[..., 0] - y[None]) ** 2, axis=1)
        denominator = np.mean((y - y.mean()) ** 2)
        return {"mse": mse, "r2": 1 - mse / denominator}
    labels = y.astype(int)
    targets = np.eye(2)[labels]
    brier = np.mean(np.sum((predictions - targets[None]) ** 2, axis=-1), axis=1)
    selected = predictions[
        np.arange(len(predictions))[:, None], np.arange(len(y))[None], labels[None]
    ]
    logloss = -np.mean(np.log(np.clip(selected, 1e-12, 1)), axis=1)
    accuracy = np.mean(np.argmax(predictions, axis=-1) == labels[None], axis=1)
    auc = np.asarray([roc_auc_score(labels, prediction[:, 1]) for prediction in predictions])
    return {"brier": brier, "logloss": logloss, "accuracy": accuracy, "roc_auc": auc}


def main() -> None:
    screened = pd.read_csv(RESULTS / "validation_screened_cover_cells.csv")
    material = screened[screened.study == "strength2_confirmation"]
    exact = pd.read_csv(RESULTS / "strength2_confirmation_cells.csv")
    exact = exact[exact.split == "test"].set_index(["dataset", "model"])
    input_dir = RESULTS / "tier1_confirmation"
    rows: list[dict[str, object]] = []
    calibration_errors = []
    calibration_z_scores = []
    calibration_rows: list[dict[str, object]] = []

    for cell in material.itertuples(index=False):
        stem = f"{cell.dataset}__{cell.model}"
        archive = np.load(input_dir / f"{stem}.npz")
        manifest = json.loads((input_dir / f"{stem}.json").read_text())
        raw = archive["test_predictions"].astype(np.float64)
        flat = raw.reshape((-1,) + raw.shape[-2:])
        y = archive["test_y"]
        rng = np.random.default_rng(seed_for(cell.dataset, cell.model))
        collected: dict[str, dict[str, list[np.ndarray]]] = {method: {} for method in METHODS}
        for _ in range(DRAWS // BATCH):
            actions = ids_for_actions(tuple(raw.shape[:4]), rng, BATCH)
            for method, ids in actions.items():
                values = metric_draws(y, flat[ids].mean(axis=1), manifest["task"])
                for metric, draws in values.items():
                    collected[method].setdefault(metric, []).append(draws)
        arrays = {
            method: {metric: np.concatenate(parts) for metric, parts in metrics.items()}
            for method, metrics in collected.items()
        }

        smooth = "brier" if manifest["task"] != "regression" else "mse"
        exact_row = exact.loc[(cell.dataset, cell.model)]
        exact_means = {
            "strength2": exact_row.quotient_loss + exact_row.strength2_residual,
            "iid16": exact_row.quotient_loss + exact_row.iid16_residual,
            "four_strength1": exact_row.quotient_loss + exact_row.four_strength1_residual,
            "four_seed_blocks": exact_row.quotient_loss + exact_row.four_seed_blocks_residual,
        }
        for method in METHODS:
            error = abs(float(arrays[method][smooth].mean()) - exact_means[method])
            standard_error = float(arrays[method][smooth].std(ddof=1) / np.sqrt(DRAWS))
            calibration_errors.append(error)
            calibration_rows.append({
                "dataset": cell.dataset, "model": cell.model, "method": method,
                "metric": smooth, "mc_mean": float(arrays[method][smooth].mean()),
                "exact_mean": exact_means[method], "absolute_error": error,
                "mc_standard_error": standard_error,
                "standardized_error": error / standard_error if standard_error > 1e-12 else np.nan,
            })
            if standard_error > 1e-12:
                calibration_z_scores.append(error / standard_error)
        for metric in arrays["strength2"]:
            minimize = metric in {"brier", "logloss", "mse"}
            action = arrays["strength2"][metric]
            for control in CONTROLS:
                comparison = arrays[control][metric]
                difference = float(action.mean() - comparison.mean())
                standard_error = float(np.sqrt(action.var(ddof=1) / DRAWS + comparison.var(ddof=1) / DRAWS))
                rows.append({
                    "dataset": cell.dataset, "model": cell.model, "task": manifest["task"],
                    "metric": metric, "control": control, "draws": DRAWS,
                    "strength2_mean": float(action.mean()), "control_mean": float(comparison.mean()),
                    "strength2_minus_control": difference,
                    "mc_standard_error": standard_error,
                    "normal_ci_low": difference - 1.96 * standard_error,
                    "normal_ci_high": difference + 1.96 * standard_error,
                    "favorable_point_estimate": bool(difference < 0 if minimize else difference > 0),
                    "favorable_95_mc_interval": bool(
                        difference + 1.96 * standard_error < 0 if minimize
                        else difference - 1.96 * standard_error > 0
                    ),
                })

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "predictive_metric_uncertainty_cells.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(
        RESULTS / "predictive_metric_exact_calibration.csv", index=False
    )
    summaries: dict[str, object] = {}
    for (task, metric, control), current in frame.groupby(["task", "metric", "control"]):
        summaries[f"{task}:{metric}:{control}"] = {
            "cells": len(current),
            "favorable_point_estimate_cells": int(current.favorable_point_estimate.sum()),
            "favorable_95_mc_interval_cells": int(current.favorable_95_mc_interval.sum()),
            "cell_balanced_mean_strength2_minus_control": float(current.strength2_minus_control.mean()),
            "median_mc_standard_error": float(current.mc_standard_error.median()),
        }
    summary = {
        "status": "complete", "draws_per_method_cell": DRAWS, "batch_size": BATCH,
        "inference_scope": "Monte_Carlo_error_for_randomized_actions_on_fixed_test_sets",
        "maximum_absolute_mc_error_against_exact_brier_or_mse": float(max(calibration_errors)),
        "maximum_standardized_mc_error_against_exact_brier_or_mse": float(max(calibration_z_scores)),
        "comparisons": summaries,
    }
    (RESULTS / "predictive_metric_uncertainty_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
