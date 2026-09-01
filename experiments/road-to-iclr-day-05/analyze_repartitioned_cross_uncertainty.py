"""Paired Monte Carlo and source uncertainty for repartitioned selection."""

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
    "validation_agreement": 1,
    "test_agreement": 1,
    "validation_quotient_regret": -1,
    "test_quotient_regret": -1,
    "selected_test_quotient_loss": -1,
}


def main() -> None:
    draws = pd.read_csv(RESULTS / "repartitioned_cross_selection_draws.csv")
    cells = pd.read_csv(RESULTS / "repartitioned_cross_selection_cells.csv")
    output: dict[str, object] = {"status": "complete", "panels": {}}
    difference_rows = []
    for panel, current_cells in cells.groupby("panel"):
        panel_record = {}
        current_draws = draws[draws.panel == panel]
        for metric, direction in METRICS.items():
            cell_pivot = current_cells.pivot(index="dataset", columns="method", values=metric)
            source_difference = cell_pivot[ACTION] - cell_pivot[CONTROL]
            source_interval = RMS.cluster_interval(
                source_difference.to_numpy(),
                RMS.stable_seed("repartition-source", panel, metric),
            )
            draw_pivot = current_draws.pivot(
                index=["dataset", "draw"], columns="method", values=metric
            )
            paired = (
                draw_pivot[ACTION].astype(float) - draw_pivot[CONTROL].astype(float)
            ).rename("difference").reset_index()
            by_draw = paired.groupby("draw").difference.mean().to_numpy()
            mc_mean = float(by_draw.mean())
            mc_se = float(by_draw.std(ddof=1) / np.sqrt(len(by_draw)))
            panel_record[metric] = {
                "cover_minus_iid_mean": float(source_difference.mean()),
                "favorable_sources": int((source_difference * direction > 0).sum()),
                "tied_sources": int(np.isclose(source_difference, 0, rtol=0, atol=1e-15).sum()),
                "source_bootstrap_95_interval": source_interval,
                "paired_draw_monte_carlo_standard_error": mc_se,
                "paired_draw_normal_95_interval": [mc_mean - 1.96 * mc_se, mc_mean + 1.96 * mc_se],
                "source_interval_excludes_zero_favorably": bool(
                    source_interval[0] > 0 if direction > 0 else source_interval[1] < 0
                ),
            }
            for dataset, value in source_difference.items():
                difference_rows.append({
                    "panel": panel, "dataset": dataset, "metric": metric,
                    "cover_minus_iid": float(value),
                    "favorable": bool(value * direction > 0),
                })
        classification = current_cells[current_cells.task == "binclass"]
        if len(classification):
            pivot = classification.pivot(
                index="dataset", columns="method", values="test_quotient_regret"
            )
            panel_record["classification_test_regret"] = {
                "cover_minus_iid_mean": float((pivot[ACTION] - pivot[CONTROL]).mean()),
                "favorable_sources": int((pivot[ACTION] < pivot[CONTROL]).sum()),
                "sources": int(len(pivot)),
            }
        output["panels"][panel] = panel_record
    pd.DataFrame(difference_rows).to_csv(
        RESULTS / "repartitioned_cross_source_differences.csv", index=False
    )
    (RESULTS / "repartitioned_cross_uncertainty_summary.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
