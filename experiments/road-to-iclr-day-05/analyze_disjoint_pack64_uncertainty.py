"""Source and full-product scope audit for mutually disjoint four-packs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import analyze_robust_model_selection as RMS


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
ACTION = "mutually_disjoint_pack64"
CONTROL = "two_disjoint_pairs64"


def main() -> None:
    frame = pd.read_csv(RESULTS / "disjoint_pack64_calibration.csv")
    summary: dict[str, object] = {"status": "complete", "panels": {}}
    passes = {"score_rmse": 0, "prediction_residual": 0}
    for panel, current in frame.groupby("panel"):
        records = {}
        for metric in passes:
            pivot = current.pivot(index=["dataset", "model"], columns="method", values=metric)
            difference = pivot[ACTION] - pivot[CONTROL]
            source = difference.groupby(level="dataset").mean()
            interval = RMS.cluster_interval(
                source.to_numpy(), RMS.stable_seed("pack64-source", panel, metric)
            )
            favorable = bool(interval[1] < 0)
            passes[metric] += int(favorable)
            records[metric] = {
                "mean_difference": float(difference.mean()),
                "favorable_candidates": int((difference < 0).sum()),
                "candidates": len(difference),
                "favorable_sources": int((source < 0).sum()),
                "sources": len(source),
                "source_bootstrap_95_interval": interval,
                "interval_excludes_zero_favorably": favorable,
            }
        summary["panels"][panel] = records
    full = frame[frame.product_cells == 128]
    full_summary = {}
    full_better = True
    for metric in passes:
        pivot = full.pivot(index=["panel", "dataset", "model"], columns="method", values=metric)
        action, control = pivot[ACTION].mean(), pivot[CONTROL].mean()
        full_better &= action < control
        full_summary[metric] = {
            "candidate_cells": len(pivot), "four_pack_mean": float(action),
            "two_pair_mean": float(control), "ratio": float(action / control),
            "favorable_candidates": int((pivot[ACTION] < pivot[CONTROL]).sum()),
        }
    summary["full_128_cell_subset"] = full_summary
    summary["panels_with_favorable_intervals"] = passes
    summary["frozen_addendum_passed"] = bool(
        passes["score_rmse"] >= 4 and passes["prediction_residual"] >= 4
        and full_better
    )
    (RESULTS / "disjoint_pack64_uncertainty_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
