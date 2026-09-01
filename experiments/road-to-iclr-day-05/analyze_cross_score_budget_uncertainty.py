"""Paired source uncertainty for the 64-fit block-U frontier."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import analyze_robust_model_selection as RMS


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
ACTION = "cover_block_u64"
CONTROL = "iid_u64"
METRICS = {
    "selection_agreement": 1,
    "validation_quotient_regret": -1,
    "selected_quotient_test_loss": -1,
    "selected_realized_test_loss": -1,
}


def main() -> None:
    cells = pd.read_csv(RESULTS / "cross_score_budget_frontier_cells.csv")
    calibration = pd.read_csv(
        RESULTS / "cross_score_budget_frontier_calibration.csv"
    )
    summary: dict[str, object] = {
        "status": "complete", "postgate_uncertainty_audit": True, "panels": {}
    }
    for panel, current in cells.groupby("panel"):
        records = {}
        for metric, direction in METRICS.items():
            pivot = current.pivot(index="dataset", columns="method", values=metric)
            difference = pivot[ACTION] - pivot[CONTROL]
            interval = RMS.cluster_interval(
                difference.to_numpy(),
                RMS.stable_seed("block-u64-uncertainty", panel, metric),
            )
            records[metric] = {
                "mean_difference": float(difference.mean()),
                "favorable_sources": int((difference * direction > 0).sum()),
                "sources": len(difference),
                "source_bootstrap_95_interval": interval,
            }
        current_cal = calibration[calibration.panel == panel]
        pivot = current_cal.pivot(
            index=["dataset", "model"], columns="method", values="score_rmse"
        )
        rmse_difference = pivot[ACTION] - pivot[CONTROL]
        dataset_difference = rmse_difference.groupby(level="dataset").mean()
        records["score_rmse"] = {
            "mean_difference": float(rmse_difference.mean()),
            "favorable_candidate_cells": int((rmse_difference < 0).sum()),
            "candidate_cells": len(rmse_difference),
            "source_bootstrap_95_interval": RMS.cluster_interval(
                dataset_difference.to_numpy(),
                RMS.stable_seed("block-u64-rmse", panel),
            ),
        }
        summary["panels"][panel] = records
    (RESULTS / "cross_score_budget_uncertainty_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
