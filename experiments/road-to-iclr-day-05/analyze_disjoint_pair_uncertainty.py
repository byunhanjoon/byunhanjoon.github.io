"""Source uncertainty and non-partition scope for packed cover pairs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import analyze_robust_model_selection as RMS


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
ACTION = "disjoint_pair_mean32"
CONTROL = "independent_pair_mean32"


def main() -> None:
    calibration = pd.read_csv(RESULTS / "disjoint_pair32_calibration.csv")
    cells = pd.read_csv(RESULTS / "disjoint_pair32_cells.csv")
    summary: dict[str, object] = {"status": "complete", "panels": {}}
    interval_passes = {"score_rmse": 0, "prediction_residual": 0}
    for panel, current in calibration.groupby("panel"):
        records = {}
        for metric in ("score_rmse", "prediction_residual"):
            pivot = current.pivot(index=["dataset", "model"], columns="method", values=metric)
            difference = pivot[ACTION] - pivot[CONTROL]
            source = difference.groupby(level="dataset").mean()
            interval = RMS.cluster_interval(
                source.to_numpy(), RMS.stable_seed("disjoint-pair-source", panel, metric)
            )
            favorable_interval = bool(interval[1] < 0)
            interval_passes[metric] += int(favorable_interval)
            records[metric] = {
                "mean_difference": float(difference.mean()),
                "favorable_candidates": int((difference < 0).sum()),
                "candidates": len(difference),
                "favorable_sources": int((source < 0).sum()),
                "sources": len(source),
                "source_bootstrap_95_interval": interval,
                "interval_excludes_zero_favorably": favorable_interval,
            }
        cell_panel = cells[cells.panel == panel]
        for metric, direction in (("selection_agreement", 1), ("validation_quotient_regret", -1)):
            pivot = cell_panel.pivot(index="dataset", columns="method", values=metric)
            difference = pivot[ACTION] - pivot[CONTROL]
            records[metric] = {
                "mean_difference": float(difference.mean()),
                "strictly_favorable_sources": int((difference * direction > 0).sum()),
                "ties": int((difference == 0).sum()), "sources": len(difference),
            }
        summary["panels"][panel] = records

    nonpartition = calibration[calibration.product_cells > 32]
    nonpartition_summary = {}
    nonpartition_better = True
    for metric in ("score_rmse", "prediction_residual"):
        pivot = nonpartition.pivot(index=["panel", "dataset", "model"], columns="method", values=metric)
        action_mean, control_mean = pivot[ACTION].mean(), pivot[CONTROL].mean()
        nonpartition_better &= action_mean < control_mean
        nonpartition_summary[metric] = {
            "candidate_cells": len(pivot),
            "packed_mean": float(action_mean), "independent_mean": float(control_mean),
            "ratio": float(action_mean / control_mean),
            "favorable_candidates": int((pivot[ACTION] < pivot[CONTROL]).sum()),
        }
    summary["nonpartition_over_32_cells"] = nonpartition_summary
    summary["panels_with_favorable_intervals"] = interval_passes
    summary["frozen_addendum_passed"] = bool(
        interval_passes["score_rmse"] >= 4
        and interval_passes["prediction_residual"] >= 4
        and nonpartition_better
    )
    (RESULTS / "disjoint_pair_uncertainty_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
