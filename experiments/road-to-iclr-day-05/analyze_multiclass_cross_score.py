"""Vector-valued multiclass Brier cross-score scope extension."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_cross_quotient_selection as CQS


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PANEL = "openml_multiclass"


def main() -> None:
    config = json.loads((HERE / "openml_multiclass_cover_config.json").read_text())
    directory = RESULTS / "openml_multiclass_cover"
    rows, calibration_rows = [], []
    for dataset in config["datasets"]:
        current_rows, current_calibration = CQS.analyze_dataset(
            PANEL, dataset, config["models"], directory
        )
        rows.extend(current_rows)
        calibration_rows.extend(current_calibration)
    draws = pd.DataFrame(rows)
    draws.to_csv(RESULTS / "multiclass_cross_score_draws.csv", index=False)
    cells = draws.groupby(["dataset", "method"], as_index=False).agg(
        selection_agreement=("selection_agreement", "mean"),
        validation_quotient_regret=("validation_quotient_regret", "mean"),
        selected_quotient_test_loss=("selected_quotient_test_loss", "mean"),
    )
    cells.to_csv(RESULTS / "multiclass_cross_score_cells.csv", index=False)
    calibration = pd.DataFrame(calibration_rows)
    calibration["score_rmse"] = np.sqrt(
        calibration.score_bias ** 2
        + CQS.DRAWS * calibration.mc_standard_error ** 2
    )
    calibration.to_csv(RESULTS / "multiclass_cross_score_calibration.csv", index=False)
    keep = calibration[calibration.method.isin(["strength2_cross32", "iid_u32"])]
    pivot = keep.pivot(
        index=["dataset", "model"], columns="method", values="score_rmse"
    )
    difference = pivot.strength2_cross32 - pivot.iid_u32
    source_means = keep.groupby(["dataset", "method"]).score_rmse.mean().unstack()
    finite_bias = keep[
        (keep.method == "strength2_cross32") & np.isfinite(keep.standardized_bias)
    ]
    clauses = {
        "cover_lower_rmse_candidate_cells_at_least_4_of_6": bool((difference < 0).sum() >= 4),
        "cover_lower_rmse_both_source_means": bool(
            (source_means.strength2_cross32 < source_means.iid_u32).all()
        ),
        "all_cover_standardized_bias_within_3": bool(
            (finite_bias.standardized_bias.abs() <= 3).all()
        ),
    }
    summary = {
        "status": "complete", "draws_per_dataset": CQS.DRAWS,
        "classes": {dataset: int(np.load(directory / f"{dataset}__{config['models'][0]}.npz")["validation_predictions"].shape[-1]) for dataset in config["datasets"]},
        "cover_lower_rmse_candidate_cells": int((difference < 0).sum()),
        "candidate_cells": int(len(difference)),
        "source_mean_rmse": source_means.reset_index().to_dict(orient="records"),
        "max_absolute_cover_standardized_bias": float(finite_bias.standardized_bias.abs().max()),
        "selection_means": cells.groupby("method").mean(numeric_only=True).reset_index().to_dict(orient="records"),
        "clauses": clauses,
        "frozen_gate_passed": bool(all(clauses.values())),
    }
    (RESULTS / "multiclass_cross_score_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
