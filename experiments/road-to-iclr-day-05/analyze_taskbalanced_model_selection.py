"""Frozen task-balanced external model-selection replication."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_robust_model_selection as RMS


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def summarize(frame: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    cells = frame.groupby(["dataset", "task", "population", "method"], as_index=False).agg(
        selected_realized_test_loss=("selected_realized_test_loss", "mean"),
        selected_quotient_test_loss=("selected_quotient_test_loss", "mean"),
        validation_quotient_regret=("validation_quotient_regret", "mean"),
        test_quotient_regret=("test_quotient_regret", "mean"),
        selection_agreement=("agrees_validation_quotient_winner", "mean"),
    )
    comparisons = {}
    pivot = cells.pivot(index="dataset", columns="method", values="selected_realized_test_loss")
    for control in RMS.ALL_METHODS[1:]:
        difference = pivot.strength2 - pivot[control]
        comparisons[control] = {
            "mean_strength2_minus_control_test_loss": float(difference.mean()),
            "datasets_strength2_lower": int((difference < 0).sum()),
            "dataset_bootstrap_95_interval": RMS.cluster_interval(
                difference.to_numpy(), RMS.stable_seed("taskbalanced-selection", control)
            ),
        }
    nonenumerating_names = cells.loc[cells.population > 16, "dataset"].unique()
    nonenumerating = pivot.loc[nonenumerating_names]
    nonenumerating_comparisons = {}
    for control in RMS.METHODS[1:]:
        difference = nonenumerating.strength2 - nonenumerating[control]
        nonenumerating_comparisons[control] = {
            "mean_strength2_minus_control_test_loss": float(difference.mean()),
            "datasets_strength2_lower": int((difference < 0).sum()),
            "datasets": len(difference),
        }
    agreement = cells.groupby("method").selection_agreement.mean()
    core_gate = bool(
        all(comparisons[c]["mean_strength2_minus_control_test_loss"] < 0 for c in RMS.METHODS[1:])
        and comparisons["iid16"]["datasets_strength2_lower"] >= 6
        and agreement.strength2 > agreement.iid16
        and all(nonenumerating_comparisons[c]["mean_strength2_minus_control_test_loss"] < 0 for c in RMS.METHODS[1:])
        and nonenumerating_comparisons["iid16"]["datasets_strength2_lower"] >= 4
    )
    summary = {
        "datasets": int(cells.dataset.nunique()),
        "nonenumerating_datasets": int(len(nonenumerating_names)),
        "comparisons": comparisons,
        "nonenumerating_comparisons": nonenumerating_comparisons,
        "mean_selection_agreement": {method: float(value) for method, value in agreement.items()},
        "task_strata_strength2_minus_iid16": {
            task: {
                "datasets": int(current.dataset.nunique()),
                "mean_difference": float(
                    current.pivot(index="dataset", columns="method", values="selected_realized_test_loss")
                    .eval("strength2 - iid16").mean()
                ),
                "datasets_strength2_lower": int((
                    current.pivot(index="dataset", columns="method", values="selected_realized_test_loss")
                    .eval("strength2 - iid16") < 0
                ).sum()),
            }
            for task, current in cells.groupby("task")
        },
        "frozen_core_gate_passed": core_gate,
    }
    return summary, cells


def main() -> None:
    config = json.loads((HERE / "openml_taskbalanced_cover_config.json").read_text())
    input_dir = RESULTS / "openml_taskbalanced_cover"
    rows = []
    for dataset in config["datasets"]:
        current = RMS.analyze_dataset("openml_taskbalanced", dataset, config["models"], input_dir)
        shape = np.load(input_dir / f"{dataset}__{config['models'][0]}.npz")["test_predictions"].shape[:4]
        population = int(np.prod(shape))
        for row in current:
            row["population"] = population
        rows.extend(current)
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "taskbalanced_model_selection_draws.csv", index=False)
    summary, cells = summarize(frame)
    cells.to_csv(RESULTS / "taskbalanced_model_selection_cells.csv", index=False)
    output = {"status": "complete", "draws_per_dataset": RMS.DRAWS, **summary}
    (RESULTS / "taskbalanced_model_selection_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
