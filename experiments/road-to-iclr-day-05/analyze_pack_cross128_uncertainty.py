"""Source-cluster uncertainty for non-exhaustive pack-cross128 candidates."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import analyze_robust_model_selection as RMS


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
ACTION = "disjoint_pack_cross128"
CONTROL = "independent_cover_u128"


def main() -> None:
    frame = pd.read_csv(RESULTS / "disjoint_pack_cross128_calibration.csv")
    frame = frame[frame.product_cells == 128]
    pivot = frame.pivot(index=["panel", "dataset", "model"], columns="method", values="score_rmse")
    difference = pivot[ACTION] - pivot[CONTROL]
    panels = {}
    passed = 0
    for panel, current in difference.groupby(level="panel"):
        source = current.groupby(level="dataset").mean()
        interval = RMS.cluster_interval(
            source.to_numpy(), RMS.stable_seed("pack-cross128-source", panel)
        )
        clauses = {
            "mean_negative": bool(current.mean() < 0),
            "all_sources_favorable": bool((source < 0).all()),
            "interval_upper_negative": bool(interval[1] < 0),
        }
        passed += int(all(clauses.values()))
        panels[panel] = {
            "clauses": clauses, "candidates": int(len(current)),
            "sources": int(len(source)), "favorable_sources": int((source < 0).sum()),
            "mean_rmse_difference": float(current.mean()),
            "source_bootstrap_95_interval": interval,
        }
    summary = {
        "status": "complete", "full_product_candidates": int(len(difference)),
        "represented_panels": int(len(panels)), "panels_passing": passed,
        "panels": panels, "frozen_addendum_passed": bool(passed == len(panels)),
    }
    (RESULTS / "pack_cross128_uncertainty_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
