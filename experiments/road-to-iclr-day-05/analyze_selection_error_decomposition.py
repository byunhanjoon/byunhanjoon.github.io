"""Exact decomposition of held-out selection regret into target shift and nuisance error."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main() -> None:
    draws = pd.concat([
        pd.read_csv(RESULTS / "robust_model_selection_draws.csv"),
        pd.read_csv(RESULTS / "openml_external_selection_draws.csv"),
        pd.read_csv(RESULTS / "taskbalanced_model_selection_draws.csv"),
    ], ignore_index=True)
    shifts = pd.read_csv(RESULTS / "selection_shift_ceiling_datasets.csv")[
        ["panel", "dataset", "validation_winner_test_regret"]
    ].rename(columns={"validation_winner_test_regret": "target_shift_floor"})
    cells = draws.groupby(["panel", "dataset", "method"], as_index=False).agg(
        mean_test_quotient_regret=("test_quotient_regret", "mean"),
        mean_validation_quotient_regret=("validation_quotient_regret", "mean"),
        validation_winner_agreement=("agrees_validation_quotient_winner", "mean"),
    ).merge(shifts, on=["panel", "dataset"], how="left", validate="many_to_one")
    if cells.target_shift_floor.isna().any():
        raise AssertionError("missing exact target-shift floor")
    cells["nuisance_selection_term"] = cells.mean_test_quotient_regret - cells.target_shift_floor
    cells["reconstruction_error"] = abs(
        cells.mean_test_quotient_regret - cells.target_shift_floor - cells.nuisance_selection_term
    )
    cells.to_csv(RESULTS / "selection_error_decomposition_cells.csv", index=False)
    panels = {}
    for (panel, method), current in cells.groupby(["panel", "method"]):
        panels.setdefault(panel, {})[method] = {
            "mean_total_test_quotient_regret": float(current.mean_test_quotient_regret.mean()),
            "mean_target_shift_floor": float(current.target_shift_floor.mean()),
            "mean_nuisance_selection_term": float(current.nuisance_selection_term.mean()),
            "mean_validation_quotient_regret": float(current.mean_validation_quotient_regret.mean()),
            "mean_validation_winner_agreement": float(current.validation_winner_agreement.mean()),
        }
    summary = {
        "status": "complete", "post_outcome_diagnostic": True,
        "maximum_reconstruction_error": float(cells.reconstruction_error.max()),
        "panels": panels,
    }
    (RESULTS / "selection_error_decomposition_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
