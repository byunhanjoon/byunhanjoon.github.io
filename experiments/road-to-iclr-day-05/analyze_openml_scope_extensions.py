"""Frozen task-subgroup and multiclass gates for the two scope extensions."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CONTROLS = ("iid16_residual", "four_strength1_residual", "four_seed_blocks_residual")


def reductions(frame: pd.DataFrame) -> dict[str, float]:
    return {control: float(1 - frame.strength2_residual.mean() / frame[control].mean()) for control in CONTROLS}


def main() -> None:
    selected = pd.read_csv(RESULTS / "validation_screened_cover_cells.csv")
    task = selected[selected.study == "strength2_openml_taskbalanced"].copy()
    multiclass = selected[selected.study == "strength2_openml_multiclass"].copy()
    multiclass_initial = multiclass[multiclass.model != "onehot_adam_mlp"]
    multiclass_mlp = multiclass[multiclass.model == "onehot_adam_mlp"]
    task_by_type = {name: reductions(current) for name, current in task.groupby("task")}
    task_group_means = task.groupby("dataset")[["strength2_residual", *CONTROLS]].mean()
    task_gate = bool(
        task.beats_all.mean() >= .75
        and all((task_group_means.strength2_residual < task_group_means[control]).sum() >= 6 for control in CONTROLS)
        and all(value > 0 for current in task_by_type.values() for value in current.values())
    )
    multiclass_gate = bool(
        len(multiclass_initial) >= 2 and multiclass_initial.beats_all.sum() >= 2
        and all(value > 0 for value in reductions(multiclass_initial).values())
    )
    summary = {
        "status": "complete",
        "taskbalanced": {
            "validation_material_cells": len(task), "cells_beating_all": int(task.beats_all.sum()),
            "source_groups": len(task_group_means),
            "source_groups_beating_each_control": {
                control: int((task_group_means.strength2_residual < task_group_means[control]).sum())
                for control in CONTROLS
            },
            "pooled_reductions_by_task": task_by_type,
            "frozen_gate_passed": task_gate,
        },
        "multiclass": {
            "validation_material_cells": len(multiclass), "cells_beating_all": int(multiclass.beats_all.sum()),
            "pooled_reductions": reductions(multiclass),
            "initial_linear_forest_gate_cells": len(multiclass_initial),
            "initial_linear_forest_cells_beating_all": int(multiclass_initial.beats_all.sum()),
            "frozen_gate_passed": multiclass_gate,
            "postgate_mlp_addendum": {
                "validation_material_cells": len(multiclass_mlp),
                "cells_beating_all": int(multiclass_mlp.beats_all.sum()),
                "cells_beating_each_control": {
                    control: int((multiclass_mlp.strength2_residual < multiclass_mlp[control]).sum())
                    for control in CONTROLS
                },
                "pooled_reductions": reductions(multiclass_mlp),
            },
        },
    }
    (RESULTS / "openml_scope_extensions_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
