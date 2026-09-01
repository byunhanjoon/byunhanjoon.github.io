"""Stricter validation-screened evaluation of saved cover outcomes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


STUDIES = (
    ("strength1_initial", "joint_orthogonal_cover_cells.csv", None),
    ("strength1_confirmation", "joint_cover_confirmation_cells.csv", "tier1_confirmation_config.json"),
    ("strength2_initial", "strength2_initial_cells.csv", None),
    ("strength2_confirmation", "strength2_confirmation_cells.csv", "tier1_confirmation_config.json"),
    ("strength1_menu_repeat", "joint_cover_menu_repeat_cells.csv", "tier1_menu_repeat_config.json"),
    ("strength2_menu_repeat", "strength2_menu_repeat_cells.csv", "tier1_menu_repeat_config.json"),
    ("strength2_subsample_repeat", "strength2_subsample_repeat_cells.csv", "tier1_subsample_repeat_config.json"),
    ("strength2_openml_external", "strength2_openml_external_cells.csv", "openml_external_cover_config.json"),
    ("strength2_openml_taskbalanced", "strength2_openml_taskbalanced_cells.csv", "openml_taskbalanced_cover_config.json"),
    ("strength2_openml_multiclass", "strength2_openml_multiclass_cells.csv", "openml_multiclass_cover_config.json"),
)


def paired(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "model"]
    validation = frame[frame.split == "validation"].set_index(keys)
    test = frame[frame.split == "test"].set_index(keys)
    common = validation.index.intersection(test.index)
    output = test.loc[common].copy()
    if "joint_risk_fraction_of_member_loss" in validation:
        screen = validation.loc[common, "joint_risk_fraction_of_member_loss"] >= 0.005
    else:
        screen = validation.loc[common, "joint_risk"] / validation.loc[common, "mean_member_loss"] >= 0.005
    output["validation_material"] = screen
    return output.reset_index()


def main() -> None:
    summaries = {}
    output_rows = []
    for study, filename, config_name in STUDIES:
        full = pd.read_csv(RESULTS / filename)
        evaluation = paired(full)
        material = evaluation[evaluation.validation_material].copy()
        strength2 = study.startswith("strength2")
        if strength2:
            comparisons = ["iid16_residual", "four_strength1_residual", "four_seed_blocks_residual"]
            action = "strength2_residual"
        else:
            comparisons = ["iid_joint_expected_residual", "seed_only_expected_residual"]
            action = "orthogonal_cover_expected_residual"
        wins = np.ones(len(material), dtype=bool)
        for comparator in comparisons:
            wins &= material[action].to_numpy() < material[comparator].to_numpy()
        group_summary = None
        if config_name:
            config = json.loads((HERE / config_name).read_text())
            group_rows = []
            for group in sorted(set(config["source_groups"].values())):
                datasets = [dataset for dataset, value in config["source_groups"].items() if value == group]
                current = material[material.dataset.isin(datasets)]
                if not len(current):
                    group_rows.append((group, False))
                    continue
                group_rows.append((group, all(current[action].mean() < current[c].mean() for c in comparisons)))
            group_summary = {
                "groups": len(group_rows),
                "groups_beating_all": int(sum(value for _, value in group_rows)),
                "details": {group: value for group, value in group_rows},
            }
        validation = full[full.split == "validation"].set_index(["dataset", "model"])
        test = full[full.split == "test"].set_index(["dataset", "model"])
        common = validation.index.intersection(test.index)
        transfer = {}
        for comparator in comparisons:
            val_gain = validation.loc[common, comparator] - validation.loc[common, action]
            test_gain = test.loc[common, comparator] - test.loc[common, action]
            transfer[comparator] = {
                "spearman_gain_correlation": float(val_gain.corr(test_gain, method="spearman")),
                "sign_agreement": float(np.mean((val_gain > 0) == (test_gain > 0))),
                "validation_positive_test_positive_fraction": float(np.mean(test_gain[val_gain > 0] > 0)) if np.any(val_gain > 0) else np.nan,
            }
        summaries[study] = {
            "cells": len(evaluation), "validation_material_cells": len(material),
            "test_cells_action_beats_all": int(wins.sum()),
            "mean_action_residual": float(material[action].mean()) if len(material) else np.nan,
            "mean_comparator_residuals": {c: float(material[c].mean()) if len(material) else np.nan for c in comparisons},
            "source_group_summary": group_summary,
            "validation_test_transfer": transfer,
        }
        material.insert(0, "study", study)
        material["beats_all"] = wins
        output_rows.append(material)
    combined = pd.concat(output_rows, ignore_index=True)
    combined.to_csv(RESULTS / "validation_screened_cover_cells.csv", index=False)
    (RESULTS / "validation_screened_cover_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
