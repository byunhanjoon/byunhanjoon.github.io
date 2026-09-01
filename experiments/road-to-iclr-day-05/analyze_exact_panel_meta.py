"""Descriptive roll-up of all validation-screened exact-tensor panels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PANELS = {
    "strength2_confirmation": "tier1_confirmation_config.json",
    "strength2_openml_external": "openml_external_cover_config.json",
    "strength2_openml_taskbalanced": "openml_taskbalanced_cover_config.json",
    "strength2_openml_multiclass": "openml_multiclass_cover_config.json",
}
CONTROLS = ("iid16_residual", "four_strength1_residual", "four_seed_blocks_residual")


def main() -> None:
    frame = pd.read_csv(RESULTS / "validation_screened_cover_cells.csv")
    frame = frame[frame.study.isin(PANELS)].copy()
    source_maps = {
        panel: json.loads((HERE / config).read_text())["source_groups"]
        for panel, config in PANELS.items()
    }
    frame["source_group"] = [
        source_maps[panel][dataset] for panel, dataset in zip(frame.study, frame.dataset)
    ]
    source = frame.groupby(["study", "source_group"], as_index=False)[
        ["strength2_residual", *CONTROLS]
    ].mean()
    source["beats_all"] = np.logical_and.reduce([
        source.strength2_residual < source[control] for control in CONTROLS
    ])
    source.to_csv(RESULTS / "exact_panel_meta_source_groups.csv", index=False)
    panels = {}
    for panel, current in frame.groupby("study"):
        panels[panel] = {
            "cells": len(current), "source_groups": current.source_group.nunique(),
            "cells_lower": {
                control: int((current.strength2_residual < current[control]).sum())
                for control in CONTROLS
            },
            "pooled_reductions": {
                control: float(1 - current.strength2_residual.mean() / current[control].mean())
                for control in CONTROLS
            },
            "source_groups_beating_all": int(source[source.study == panel].beats_all.sum()),
        }
    summary = {
        "status": "complete", "descriptive_not_independence_meta_analysis": True,
        "validation_screened_cells": len(frame),
        "cells_lower": {
            control: int((frame.strength2_residual < frame[control]).sum())
            for control in CONTROLS
        },
        "panel_source_groups": len(source),
        "panel_source_groups_beating_all": int(source.beats_all.sum()),
        "panels": panels,
        "evidence_note": (
            "Panels differ in evidence status and task-balanced sources were reused from a separate failed line; "
            "counts are a roll-up, not an iid significance calculation."
        ),
    }
    (RESULTS / "exact_panel_meta_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
