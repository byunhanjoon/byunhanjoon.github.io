#!/usr/bin/env python3
"""Audit equal-compute heterogeneous PLE ensembles without changing frozen gates."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def panel(path: Path, selection: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        primary = "log_loss" if row.task == "classification" else "rmse"
        pair = row.selected_heterogeneous_pair if selection == "recorded" else selection
        for split in ("val", "test"):
            candidate = float(row[f"{pair}_{split}_{primary}"])
            control = float(row[f"tt_{split}_{primary}"])
            rows.append(
                {
                    "dataset": row.dataset,
                    "task": row.task,
                    "model": row.model,
                    "pair": pair,
                    "split": split,
                    "primary_metric": primary,
                    "candidate": candidate,
                    "t_plus_t_control": control,
                    "relative_gain_pct": 100.0 * (control - candidate) / control,
                    "win": bool(candidate < control),
                }
            )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for split, group in frame.groupby("split", sort=False):
        partitions = [("overall", "overall", group)]
        partitions += [("task", str(key), value) for key, value in group.groupby("task")]
        partitions += [
            ("dataset", str(key), value) for key, value in group.groupby("dataset")
        ]
        for level, name, part in partitions:
            records.append(
                {
                    "split": split,
                    "level": level,
                    "name": name,
                    "cells": len(part),
                    "wins": int(part.win.sum()),
                    "mean_relative_gain_pct": float(part.relative_gain_pct.mean()),
                    "median_relative_gain_pct": float(part.relative_gain_pct.median()),
                }
            )
    return pd.DataFrame(records)


def max_parameter_mismatch(path: Path) -> float:
    frame = pd.read_csv(path)
    errors = []
    for column in ("tt_parameters", "q_parameters", "rank_active_parameters"):
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        valid = values.notna()
        errors.extend(
            ((values[valid] - frame.loc[valid, "anchor_parameters"]).abs()
             / frame.loc[valid, "anchor_parameters"]).tolist()
        )
    return float(max(errors, default=np.nan))


def main() -> None:
    specifications = (
        (
            "development",
            RESULTS / "multiview_equal_compute.csv",
            "recorded",
            "Validation-selected between two heterogeneous pairs; developmental only.",
        ),
        (
            "selection_confirmation",
            RESULTS / "multiview_prospective_confirmation.csv",
            "recorded",
            "Prospective data, but trains three candidates to select a two-member pair.",
        ),
        (
            "fixed_policy_confirmation",
            RESULTS / "multiview_fixed_policy_confirmation.csv",
            "recorded",
            "Prospective fixed task policy; exactly two deployed models and no view search.",
        ),
    )
    decision: dict[str, object] = {"panels": {}}
    fixed_panel = None
    for name, source, selection, note in specifications:
        result = panel(source, selection)
        summary = summarize(result)
        result.to_csv(RESULTS / f"multiview_{name}_panel.csv", index=False)
        summary.to_csv(RESULTS / f"multiview_{name}_summary.csv", index=False)
        test = summary.query("split == 'test' and level == 'overall'").iloc[0]
        decision["panels"][name] = {
            "note": note,
            "test_wins": int(test.wins),
            "test_cells": int(test.cells),
            "mean_test_relative_gain_pct": float(test.mean_relative_gain_pct),
            "max_per_member_parameter_relative_mismatch": max_parameter_mismatch(source),
        }
        if name == "fixed_policy_confirmation":
            fixed_panel = result

    assert fixed_panel is not None
    test = fixed_panel.query("split == 'test'")
    task_means = test.groupby("task").relative_gain_pct.mean().to_dict()
    dataset_means = test.groupby("dataset").relative_gain_pct.mean()
    clauses = {
        "at_least_8_of_12_test_wins": bool(test.win.sum() >= 8),
        "positive_overall_mean_relative_test_gain": bool(
            test.relative_gain_pct.mean() > 0
        ),
        "positive_mean_within_both_task_families": bool(
            len(task_means) == 2 and all(value > 0 for value in task_means.values())
        ),
        "positive_dataset_mean_on_at_least_3_of_4": bool(
            (dataset_means > 0).sum() >= 3
        ),
    }
    decision["fixed_policy_predeclared_gate"] = {
        "passed": bool(all(clauses.values())),
        "clauses": clauses,
        "test_wins": int(test.win.sum()),
        "test_cells": int(len(test)),
        "mean_relative_test_gain_pct": float(test.relative_gain_pct.mean()),
        "mean_relative_test_gain_pct_by_task": {
            key: float(value) for key, value in task_means.items()
        },
        "positive_dataset_means": int((dataset_means > 0).sum()),
        "interpretation": (
            "Promising win consistency, but the frozen policy is not promoted because "
            "the full prospective gate failed."
        ),
    }
    (RESULTS / "multiview_equal_compute_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(decision["fixed_policy_predeclared_gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
