"""Translate exact cover risks into equivalent independent-fit budgets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main() -> None:
    selected = pd.read_csv(RESULTS / "validation_screened_cover_cells.csv")
    cells = selected[selected.study == "strength2_confirmation"].copy()
    config = json.loads((HERE / "tier1_confirmation_config.json").read_text())
    cells["source_group"] = cells.dataset.map(config["source_groups"])
    numerical_zero = cells.strength2_residual <= np.maximum(1e-18, 1e-12 * cells.joint_risk)
    denominator = cells.strength2_residual.mask(numerical_zero, np.inf)
    cells["iid_equivalent_fits"] = cells.joint_risk / denominator
    cells["strength1_equivalent_fits"] = 16 * cells.four_strength1_residual / denominator
    cells["seed_block_equivalent_fits"] = 16 * cells.four_seed_blocks_residual / denominator
    cells.loc[numerical_zero, ["iid_equivalent_fits", "strength1_equivalent_fits", "seed_block_equivalent_fits"]] = np.inf
    cells["iid_compute_saving_fraction"] = 1 - 16 / cells.iid_equivalent_fits
    cells["strength1_compute_saving_fraction"] = 1 - 16 / cells.strength1_equivalent_fits
    cells["seed_compute_saving_fraction"] = 1 - 16 / cells.seed_block_equivalent_fits
    cells["expected_brier_or_mse_gain_vs_iid16"] = cells.iid16_residual - cells.strength2_residual
    cells["expected_brier_or_mse_gain_vs_strength1"] = cells.four_strength1_residual - cells.strength2_residual
    cells["expected_brier_or_mse_gain_vs_seed"] = cells.four_seed_blocks_residual - cells.strength2_residual

    group = cells.groupby("source_group", as_index=False).agg(
        cells=("dataset", "size"),
        strength2_residual=("strength2_residual", "mean"),
        iid16_residual=("iid16_residual", "mean"),
        four_strength1_residual=("four_strength1_residual", "mean"),
        four_seed_blocks_residual=("four_seed_blocks_residual", "mean"),
        mean_iid_equivalent_fits=("iid_equivalent_fits", "mean"),
        median_iid_equivalent_fits=("iid_equivalent_fits", "median"),
    )
    for comparator in ("iid16", "four_strength1", "four_seed_blocks"):
        group[f"risk_reduction_vs_{comparator}"] = 1 - group.strength2_residual / group[f"{comparator}_residual"]

    equivalences = ("iid_equivalent_fits", "strength1_equivalent_fits", "seed_block_equivalent_fits")
    summary = {
        "status": "complete", "selection": "validation_material", "test_cells": len(cells),
        "equivalent_fit_budgets": {},
        "sixteen_fit_strength2_mean_expected_risk": float(cells.strength2_residual.mean()),
        "mean_expected_risk_reduction": {
            "vs_iid16": float(1 - cells.strength2_residual.mean() / cells.iid16_residual.mean()),
            "vs_four_strength1": float(1 - cells.strength2_residual.mean() / cells.four_strength1_residual.mean()),
            "vs_four_seed_blocks": float(1 - cells.strength2_residual.mean() / cells.four_seed_blocks_residual.mean()),
        },
        "mean_absolute_expected_loss_gain": {
            "vs_iid16": float(cells.expected_brier_or_mse_gain_vs_iid16.mean()),
            "vs_four_strength1": float(cells.expected_brier_or_mse_gain_vs_strength1.mean()),
            "vs_four_seed_blocks": float(cells.expected_brier_or_mse_gain_vs_seed.mean()),
        },
        "source_groups": len(group),
        "source_groups_strength2_lower_than_all": int((
            (group.strength2_residual < group.iid16_residual)
            & (group.strength2_residual < group.four_strength1_residual)
            & (group.strength2_residual < group.four_seed_blocks_residual)
        ).sum()),
    }
    for name in equivalences:
        finite = cells.loc[np.isfinite(cells[name]), name]
        summary["equivalent_fit_budgets"][name] = {
            "finite_cells": int(len(finite)), "infinite_cells": int((~np.isfinite(cells[name])).sum()),
            "finite_median": float(finite.median()),
            "finite_geometric_mean": float(np.exp(np.log(finite).mean())),
            "finite_minimum": float(finite.min()), "finite_maximum": float(finite.max()),
        }
    keep = [
        "dataset", "model", "task", "source_group", "joint_risk", "strength2_residual",
        "iid16_residual", "four_strength1_residual", "four_seed_blocks_residual",
        *equivalences, "iid_compute_saving_fraction", "strength1_compute_saving_fraction",
        "seed_compute_saving_fraction", "expected_brier_or_mse_gain_vs_iid16",
        "expected_brier_or_mse_gain_vs_strength1", "expected_brier_or_mse_gain_vs_seed",
    ]
    cells[keep].to_csv(RESULTS / "compute_efficiency_cells.csv", index=False)
    group.to_csv(RESULTS / "compute_efficiency_source_groups.csv", index=False)
    (RESULTS / "compute_efficiency_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
