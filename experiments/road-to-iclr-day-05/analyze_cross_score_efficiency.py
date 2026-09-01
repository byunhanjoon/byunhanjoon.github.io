"""Candidate-level RMSE of quotient-loss estimators."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_robust_model_selection as RMS


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DRAWS = 1_024
ACTION = "strength2_cross32"
CONTROL = "iid_u32"


def main() -> None:
    frame = pd.read_csv(RESULTS / "cross_quotient_score_calibration.csv")
    frame["score_rmse"] = np.sqrt(
        DRAWS * frame.mc_standard_error ** 2 + frame.score_bias ** 2
    )
    frame.to_csv(RESULTS / "cross_score_efficiency_cells.csv", index=False)
    summary: dict[str, object] = {"status": "complete", "panels": {}}
    panel_wins = 0
    total_wins = total_cells = 0
    mean_score_panel_wins = mean_score_total_wins = 0
    for panel, current in frame.groupby("panel"):
        means = current.groupby("method").agg(
            mean_score_rmse=("score_rmse", "mean"),
            median_score_rmse=("score_rmse", "median"),
            mean_score_bias=("score_bias", "mean"),
        )
        pivot = current.pivot(index=["dataset", "model"], columns="method", values="score_rmse")
        difference = pivot[ACTION] - pivot[CONTROL]
        mean_score_difference = pivot[ACTION] - pivot["strength2_mean32"]
        wins = int((difference < 0).sum())
        mean_score_wins = int((mean_score_difference < 0).sum())
        panel_wins += difference.mean() < 0
        mean_score_panel_wins += mean_score_difference.mean() < 0
        total_wins += wins
        mean_score_total_wins += mean_score_wins
        total_cells += len(difference)
        dataset_difference = difference.groupby(level="dataset").mean()
        interval = RMS.cluster_interval(
            dataset_difference.to_numpy(), RMS.stable_seed("cross-score-rmse", panel)
        )
        summary["panels"][panel] = {
            "method_means": means.reset_index().to_dict(orient="records"),
            "strength2_minus_iid_u_mean_rmse": float(difference.mean()),
            "candidate_cells_strength2_lower_rmse": wins,
            "candidate_cells": len(difference),
            "source_bootstrap_95_interval": interval,
            "cross_vs_ordinary_cover_mean": {
                "mean_rmse_difference": float(mean_score_difference.mean()),
                "candidate_cells_cross_lower_rmse": mean_score_wins,
                "candidate_cells": len(mean_score_difference),
            },
        }
    summary["panels_strength2_lower_mean_rmse"] = int(panel_wins)
    summary["candidate_cells_strength2_lower_rmse"] = total_wins
    summary["candidate_cells"] = total_cells
    summary["candidate_fraction_strength2_lower_rmse"] = total_wins / total_cells
    summary["bias_variance_tradeoff_vs_ordinary_cover_mean"] = {
        "panels_cross_lower_mean_rmse": int(mean_score_panel_wins),
        "candidate_cells_cross_lower_rmse": int(mean_score_total_wins),
        "candidate_cells": total_cells,
    }
    summary["diagnostic_gate_passed"] = bool(
        panel_wins >= 4 and total_wins / total_cells > 0.6
    )
    (RESULTS / "cross_score_efficiency_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
