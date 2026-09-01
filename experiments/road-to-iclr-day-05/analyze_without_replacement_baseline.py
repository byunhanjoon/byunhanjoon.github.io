"""Exact finite-population correction baseline for nuisance covers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def analyze_panel(study: str, cells_path: Path) -> pd.DataFrame:
    selected = pd.read_csv(RESULTS / "validation_screened_cover_cells.csv")
    chosen = selected[selected.study == study][["dataset", "model"]]
    cells = pd.read_csv(cells_path)
    cells = cells[cells.split == "test"].merge(chosen, on=["dataset", "model"])
    # Product size follows from IID-16 = joint/16; load shape-independent N from
    # the saved strength-2 design count is not valid, so map from tensor archives
    # in the caller before applying this helper.
    return cells


def main() -> None:
    panels = {
        "strength2_confirmation": (
            RESULTS / "strength2_confirmation_cells.csv", RESULTS / "tier1_confirmation"
        ),
        "strength2_openml_external": (
            RESULTS / "strength2_openml_external_cells.csv", RESULTS / "openml_external_cover"
        ),
        "strength2_openml_taskbalanced": (
            RESULTS / "strength2_openml_taskbalanced_cells.csv", RESULTS / "openml_taskbalanced_cover"
        ),
        "strength2_openml_multiclass": (
            RESULTS / "strength2_openml_multiclass_cells.csv", RESULTS / "openml_multiclass_cover"
        ),
    }
    rows = []
    for study, (cells_path, input_dir) in panels.items():
        cells = analyze_panel(study, cells_path)
        for cell in cells.itertuples(index=False):
            import numpy as np
            shape = np.load(input_dir / f"{cell.dataset}__{cell.model}.npz")["test_predictions"].shape[:4]
            population = int(np.prod(shape))
            if population < 16:
                raise AssertionError("budget exceeds nuisance population")
            srs4 = cell.joint_risk / 4 * (population - 4) / (population - 1)
            srs16 = cell.joint_risk / 16 * (population - 16) / (population - 1)
            rows.append({
                "panel": study, "dataset": cell.dataset, "model": cell.model,
                "population": population, "joint_risk": cell.joint_risk,
                "strength1_b4_residual": cell.four_strength1_residual * 4,
                "srswor_b4_residual": srs4,
                "strength2_b16_residual": cell.strength2_residual,
                "srswor_b16_residual": srs16,
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "without_replacement_baseline_cells.csv", index=False)
    summaries = {}
    gate = True
    for study, current in frame.groupby("panel"):
        s2_wins = int((current.strength2_b16_residual < current.srswor_b16_residual).sum())
        summaries[study] = {
            "cells": len(current),
            "strength1_b4_cells_lower_than_srswor": int((current.strength1_b4_residual < current.srswor_b4_residual).sum()),
            "strength1_b4_pooled_reduction": float(1 - current.strength1_b4_residual.mean() / current.srswor_b4_residual.mean()),
            "strength2_b16_cells_lower_than_srswor": s2_wins,
            "strength2_b16_pooled_reduction": float(1 - current.strength2_b16_residual.mean() / current.srswor_b16_residual.mean()),
        }
        if study == "strength2_confirmation":
            gate &= s2_wins >= 20 and summaries[study]["strength2_b16_pooled_reduction"] > 0
        elif study == "strength2_openml_external":
            gate &= s2_wins >= 9 and summaries[study]["strength2_b16_pooled_reduction"] > 0
    summary = {
        "status": "complete", "panels": summaries,
        "frozen_gate_panels": ["strength2_confirmation", "strength2_openml_external"],
        "posthoc_scope_extension_panels": [
            "strength2_openml_multiclass", "strength2_openml_taskbalanced"
        ],
        "frozen_srswor_gate_passed": bool(gate),
    }
    (RESULTS / "without_replacement_baseline_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
