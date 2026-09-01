"""Candidate-independent action repeat for packed disjoint pairs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_cross_quotient_selection as CQS
from analyze_disjoint_pair32 import paired_ids, prediction_residuals
from analyze_strength2_cover import proper_loss


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DRAWS = 1_024


def analyze_dataset(panel: str, dataset: str, models: list[str], directory: Path):
    scores = {"disjoint_pair_mean32_independent": [], "independent_pair_mean32_independent": []}
    test_losses = {key: [] for key in scores}
    quotient_val, quotient_test, calibration = [], [], []
    validation_y = test_y = None
    task = ""
    for model in models:
        archive = np.load(directory / f"{dataset}__{model}.npz")
        manifest = json.loads((directory / f"{dataset}__{model}.json").read_text())
        validation_y, test_y = archive["validation_y"], archive["test_y"]
        task = manifest["task"]
        shape = tuple(int(value) for value in archive["validation_predictions"].shape[:4])
        val_flat = archive["validation_predictions"].reshape(
            (-1,) + archive["validation_predictions"].shape[-2:]
        ).astype(np.float64)
        test_flat = archive["test_predictions"].reshape(
            (-1,) + archive["test_predictions"].shape[-2:]
        ).astype(np.float64)
        packed, independent = paired_ids(shape, panel, f"{dataset}:{model}")
        exact_val = proper_loss(validation_y, val_flat.mean(axis=0))
        exact_test = proper_loss(test_y, test_flat.mean(axis=0))
        quotient_val.append(exact_val); quotient_test.append(exact_test)
        for method, action in zip(scores, (packed, independent)):
            _, val_loss = CQS.cross_and_mean_scores(validation_y, val_flat, *action)
            _, test_loss = CQS.cross_and_mean_scores(test_y, test_flat, *action)
            scores[method].append(val_loss); test_losses[method].append(test_loss)
            residual = prediction_residuals(val_flat, *action)
            calibration.append({
                "panel": panel, "dataset": dataset, "task": task, "model": model,
                "method": method, "product_cells": int(np.prod(shape)),
                "score_rmse": float(np.sqrt(val_loss.var(ddof=1) + (val_loss.mean() - exact_val) ** 2)),
                "prediction_residual": float(residual.mean()),
                "max_absolute_score_error": float(np.max(np.abs(val_loss - exact_val))),
            })
    quotient_val = np.asarray(quotient_val); quotient_test = np.asarray(quotient_test)
    winner = int(np.argmin(quotient_val))
    rows = []
    for method in scores:
        matrix = np.stack(scores[method], axis=1)
        test_matrix = np.stack(test_losses[method], axis=1)
        selected = np.argmin(matrix, axis=1)
        for draw, chosen in enumerate(selected):
            rows.append({
                "panel": panel, "dataset": dataset, "task": task,
                "method": method, "draw": draw,
                "selection_agreement": bool(chosen == winner),
                "validation_quotient_regret": float(quotient_val[chosen] - quotient_val[winner]),
                "selected_quotient_test_loss": float(quotient_test[chosen]),
                "selected_realized_test_loss": float(test_matrix[draw, chosen]),
            })
    return rows, calibration


def main() -> None:
    rows, calibration = [], []
    for panel, config_name, directory_name in CQS.PANELS:
        config = json.loads((HERE / config_name).read_text())
        for dataset in config["datasets"]:
            current, current_cal = analyze_dataset(
                panel, dataset, config["models"], RESULTS / directory_name
            )
            rows.extend(current); calibration.extend(current_cal)
    draws = pd.DataFrame(rows)
    draws.to_csv(RESULTS / "disjoint_pair_coupling_draws.csv", index=False)
    cells = draws.groupby(["panel", "dataset", "task", "method"], as_index=False).mean(numeric_only=True)
    cells.to_csv(RESULTS / "disjoint_pair_coupling_cells.csv", index=False)
    cal = pd.DataFrame(calibration)
    cal.to_csv(RESULTS / "disjoint_pair_coupling_calibration.csv", index=False)
    summary: dict[str, object] = {"status": "complete", "panels": {}}
    counts = {"agreement": 0, "regret": 0}
    for panel, current in cells.groupby("panel"):
        means = current.groupby("method").mean(numeric_only=True)
        action = means.loc["disjoint_pair_mean32_independent"]
        control = means.loc["independent_pair_mean32_independent"]
        clauses = {
            "agreement_nolower": bool(action.selection_agreement >= control.selection_agreement),
            "regret_nohigher": bool(action.validation_quotient_regret <= control.validation_quotient_regret),
        }
        counts["agreement"] += int(clauses["agreement_nolower"])
        counts["regret"] += int(clauses["regret_nohigher"])
        summary["panels"][panel] = {"clauses": clauses, "method_means": means.reset_index().to_dict(orient="records")}
    exact = cal[(cal.method == "disjoint_pair_mean32_independent") & (cal.product_cells <= 32)]
    summary["exact_partition_max_absolute_error"] = float(exact.max_absolute_score_error.max())
    summary["panels_passing_by_clause"] = counts
    summary["frozen_gate_passed"] = bool(
        counts["agreement"] >= 4 and counts["regret"] >= 4
        and summary["exact_partition_max_absolute_error"] < 1e-12
    )
    (RESULTS / "disjoint_pair_coupling_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
