"""Source-bootstrap and score-calibration audit for cross-quotient selection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_robust_model_selection as RMS


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
ACTION = "strength2_cross32"
CONTROL = "iid_u32"
METRICS = {
    "selection_agreement": 1,
    "validation_quotient_regret": -1,
    "selected_quotient_test_loss": -1,
    "selected_realized_test_loss": -1,
}


def main() -> None:
    cells = pd.read_csv(RESULTS / "cross_quotient_selection_cells.csv")
    summary: dict[str, object] = {"status": "complete", "panels": {}}
    differences_rows = []
    for panel, current in cells.groupby("panel"):
        records = {}
        for metric, direction in METRICS.items():
            pivot = current.pivot(index="dataset", columns="method", values=metric)
            difference = pivot[ACTION] - pivot[CONTROL]
            favorable = difference * direction > 0
            interval = RMS.cluster_interval(
                difference.to_numpy(), RMS.stable_seed("cross-uncertainty", panel, metric)
            )
            records[metric] = {
                "strength2_minus_iid_u32_mean": float(difference.mean()),
                "favorable_sources": int(favorable.sum()),
                "sources": len(difference),
                "source_bootstrap_95_interval": interval,
                "interval_excludes_zero_favorably": bool(
                    interval[0] > 0 if direction > 0 else interval[1] < 0
                ),
            }
            for dataset, value in difference.items():
                differences_rows.append({
                    "panel": panel, "dataset": dataset, "metric": metric,
                    "strength2_minus_iid_u32": float(value),
                    "favorable": bool(value * direction > 0),
                })
        summary["panels"][panel] = records
    pd.DataFrame(differences_rows).to_csv(
        RESULTS / "cross_quotient_source_differences.csv", index=False
    )
    calibration = pd.read_csv(RESULTS / "cross_quotient_score_calibration.csv")
    calibration_summary = {}
    for method, current in calibration.groupby("method"):
        finite = current[np.isfinite(current.standardized_bias)]
        calibration_summary[method] = {
            "candidate_cells": len(current),
            "mean_bias": float(current.score_bias.mean()),
            "median_bias": float(current.score_bias.median()),
            "standardized_bias_inside_95_fraction": float(
                (finite.standardized_bias.abs() <= 1.96).mean()
            ),
            "median_absolute_standardized_bias": float(
                finite.standardized_bias.abs().median()
            ),
        }
    summary["score_calibration"] = calibration_summary
    (RESULTS / "cross_quotient_uncertainty_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
