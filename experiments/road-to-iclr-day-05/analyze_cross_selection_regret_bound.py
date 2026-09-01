"""Calibrate the unbiased-score selection-regret bound from Proposition 21."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_robust_model_selection as RMS


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DRAWS = 1_024
METHODS = ("strength2_cross32", "iid_u32")


def main() -> None:
    calibration = pd.read_csv(RESULTS / "cross_quotient_score_calibration.csv")
    calibration = calibration[calibration.method.isin(METHODS)].copy()
    calibration["score_sd"] = calibration.mc_standard_error * np.sqrt(DRAWS)
    bounds = calibration.groupby(["panel", "dataset", "method"], as_index=False).agg(
        regret_upper_bound=("score_sd", lambda values: float(2 * values.sum())),
        candidates=("model", "size"),
    )
    cells = pd.read_csv(RESULTS / "cross_quotient_selection_cells.csv")
    cells = cells[cells.method.isin(METHODS)][
        ["panel", "dataset", "method", "validation_quotient_regret"]
    ]
    frame = bounds.merge(cells, on=["panel", "dataset", "method"], validate="one_to_one")
    frame["bound_covers_observed_regret"] = (
        frame.regret_upper_bound + 1e-15 >= frame.validation_quotient_regret
    )
    frame["bound_to_observed_regret_ratio"] = np.where(
        frame.validation_quotient_regret > 0,
        frame.regret_upper_bound / frame.validation_quotient_regret,
        np.inf,
    )
    frame.to_csv(RESULTS / "cross_selection_regret_bound_cells.csv", index=False)
    summary: dict[str, object] = {"status": "complete", "panels": {}}
    panel_wins = 0
    for panel, current in frame.groupby("panel"):
        pivot = current.pivot(index="dataset", columns="method", values="regret_upper_bound")
        difference = pivot.strength2_cross32 - pivot.iid_u32
        panel_wins += difference.mean() < 0
        records = {}
        for method, method_frame in current.groupby("method"):
            finite = method_frame[np.isfinite(method_frame.bound_to_observed_regret_ratio)]
            records[method] = {
                "mean_regret_upper_bound": float(method_frame.regret_upper_bound.mean()),
                "mean_observed_regret": float(method_frame.validation_quotient_regret.mean()),
                "datasets_bound_covers_observed": int(method_frame.bound_covers_observed_regret.sum()),
                "datasets": len(method_frame),
                "finite_median_bound_to_regret_ratio": float(
                    finite.bound_to_observed_regret_ratio.median()
                ) if len(finite) else None,
            }
        summary["panels"][panel] = {
            "methods": records,
            "datasets_cover_bound_lower": int((difference < 0).sum()),
            "datasets": len(difference),
            "mean_cover_minus_iid_bound": float(difference.mean()),
            "source_bootstrap_95_interval": RMS.cluster_interval(
                difference.to_numpy(), RMS.stable_seed("cross-regret-bound", panel)
            ),
        }
    summary["panels_cover_lower_mean_bound"] = int(panel_wins)
    summary["diagnostic_gate_passed"] = bool(panel_wins == 5)
    (RESULTS / "cross_selection_regret_bound_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
