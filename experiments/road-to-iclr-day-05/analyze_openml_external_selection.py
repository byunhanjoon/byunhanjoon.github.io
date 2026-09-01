"""Prospective downstream model selection on the external OpenML panel."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import analyze_robust_model_selection as RMS


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main() -> None:
    config = json.loads((HERE / "openml_external_cover_config.json").read_text())
    rows = []
    for dataset in config["datasets"]:
        rows.extend(RMS.analyze_dataset(
            "openml_external", dataset, config["models"], RESULTS / "openml_external_cover"
        ))
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "openml_external_selection_draws.csv", index=False)
    cells = frame.groupby(["dataset", "method"], as_index=False).agg(
        selected_realized_test_loss=("selected_realized_test_loss", "mean"),
        selected_quotient_test_loss=("selected_quotient_test_loss", "mean"),
        validation_quotient_regret=("validation_quotient_regret", "mean"),
        selection_agreement=("agrees_validation_quotient_winner", "mean"),
    )
    cells.to_csv(RESULTS / "openml_external_selection_cells.csv", index=False)
    loss = cells.pivot(index="dataset", columns="method", values="selected_realized_test_loss")
    agreement = cells.pivot(index="dataset", columns="method", values="selection_agreement")
    comparisons = {}
    for control in RMS.ALL_METHODS[1:]:
        difference = loss.strength2 - loss[control]
        comparisons[control] = {
            "mean_strength2_minus_control_test_brier": float(difference.mean()),
            "datasets_strength2_lower": int((difference < 0).sum()),
            "dataset_bootstrap_95_interval": RMS.cluster_interval(
                difference.to_numpy(), RMS.stable_seed("external-selection-bootstrap", control)
            ),
        }
    gate = (
        all(comparisons[control]["mean_strength2_minus_control_test_brier"] < 0 for control in RMS.METHODS[1:])
        and comparisons["iid16"]["datasets_strength2_lower"] >= 6
        and agreement.strength2.mean() > agreement.iid16.mean()
    )
    summary = {
        "status": "complete", "datasets": len(loss), "draws_per_dataset": RMS.DRAWS,
        "comparisons": comparisons,
        "mean_selection_agreement": {
            method: float(agreement[method].mean()) for method in RMS.ALL_METHODS
        },
        "frozen_external_selection_gate_passed": bool(gate),
        "post_failure_qmc_controls_strength2_lower_mean": {
            control: comparisons[control]["mean_strength2_minus_control_test_brier"] < 0
            for control in RMS.QMC_METHODS
        },
    }
    (RESULTS / "openml_external_selection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
