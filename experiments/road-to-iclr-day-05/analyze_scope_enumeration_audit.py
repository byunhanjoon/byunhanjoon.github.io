"""Audit whether scope-extension gains rely on full nuisance enumeration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main() -> None:
    selected = pd.read_csv(RESULTS / "validation_screened_cover_cells.csv")
    rows = []
    panels = {
        "strength2_openml_taskbalanced": (
            RESULTS / "strength2_openml_taskbalanced_cells.csv",
            RESULTS / "openml_taskbalanced_cover",
            "openml_taskbalanced_cover_config.json",
        ),
        "strength2_openml_multiclass": (
            RESULTS / "strength2_openml_multiclass_cells.csv",
            RESULTS / "openml_multiclass_cover",
            "openml_multiclass_cover_config.json",
        ),
    }
    for panel, (cells_path, input_dir, config_path) in panels.items():
        chosen = selected[selected.study == panel][["dataset", "model"]]
        frame = pd.read_csv(cells_path)
        frame = frame[frame.split == "test"].merge(chosen, on=["dataset", "model"])
        config = json.loads((HERE / config_path).read_text())
        for cell in frame.itertuples(index=False):
            shape = np.load(input_dir / f"{cell.dataset}__{cell.model}.npz")["test_predictions"].shape[:4]
            rows.append({
                "panel": panel, "dataset": cell.dataset, "model": cell.model,
                "task": config["dataset_tasks"][cell.dataset],
                "population": int(np.prod(shape)), "full_enumeration_at_b16": int(np.prod(shape)) == 16,
                "strength2_residual": cell.strength2_residual,
                "iid16_residual": cell.iid16_residual,
                "four_strength1_residual": cell.four_strength1_residual,
                "four_seed_blocks_residual": cell.four_seed_blocks_residual,
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "scope_enumeration_audit_cells.csv", index=False)
    summary: dict[str, object] = {"status": "complete", "panels": {}}
    for panel, current in frame.groupby("panel"):
        strata = {}
        for stratum, values in (
            ("all", current),
            ("nonenumerating_population_gt_16", current[current.population > 16]),
            ("full_enumeration_population_eq_16", current[current.population == 16]),
        ):
            records = {"cells": len(values), "source_datasets": values.dataset.nunique()}
            if len(values):
                for control in ("iid16_residual", "four_strength1_residual", "four_seed_blocks_residual"):
                    records[f"cells_lower_vs_{control}"] = int((values.strength2_residual < values[control]).sum())
                    records[f"pooled_reduction_vs_{control}"] = float(
                        1 - values.strength2_residual.mean() / values[control].mean()
                    )
            strata[stratum] = records
        summary["panels"][panel] = strata
    (RESULTS / "scope_enumeration_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
