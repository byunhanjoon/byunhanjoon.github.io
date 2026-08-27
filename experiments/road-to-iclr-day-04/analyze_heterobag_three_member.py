#!/usr/bin/env python3
"""Apply the frozen prospective gate to three-member HeteroBag."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main() -> None:
    frame = pd.read_csv(RESULTS / "heterobag_three_member.csv")
    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        primary = "log_loss" if row.task == "classification" else "rmse"
        for split in ("val", "test"):
            candidate = float(row[f"heterobag_{split}_{primary}"])
            control = float(row[f"ttt_{split}_{primary}"])
            rows.append(
                {
                    "dataset": row.dataset,
                    "task": row.task,
                    "model": row.model,
                    "alternate_view": row.alternate_view,
                    "split": split,
                    "primary_metric": primary,
                    "heterobag": candidate,
                    "t_plus_t_plus_t": control,
                    "relative_gain_pct": 100.0 * (control - candidate) / control,
                    "win": bool(candidate < control),
                }
            )
    panel = pd.DataFrame(rows)
    panel.to_csv(RESULTS / "heterobag_three_member_panel.csv", index=False)
    summary_rows: list[dict[str, object]] = []
    for split, group in panel.groupby("split", sort=False):
        partitions = [("overall", "overall", group)]
        partitions += [("task", str(key), value) for key, value in group.groupby("task")]
        partitions += [("dataset", str(key), value) for key, value in group.groupby("dataset")]
        for level, name, part in partitions:
            summary_rows.append(
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
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS / "heterobag_three_member_summary.csv", index=False)
    test = panel.query("split == 'test'")
    task_means = test.groupby("task").relative_gain_pct.mean()
    dataset_means = test.groupby("dataset").relative_gain_pct.mean()
    parameter_columns = [
        "anchor_parameters", "t1_parameters", "t2_parameters", "alternate_parameters"
    ]
    exact_parameters = bool(
        frame[parameter_columns].nunique(axis=1).eq(1).all()
    )
    clauses = {
        "at_least_8_of_12_test_wins": bool(test.win.sum() >= 8),
        "positive_overall_mean_relative_test_gain": bool(test.relative_gain_pct.mean() > 0),
        "positive_mean_within_both_task_families": bool((task_means > 0).all()),
        "positive_dataset_mean_on_at_least_3_of_4": bool((dataset_means > 0).sum() >= 3),
        "exact_per_member_parameter_match": exact_parameters,
    }
    decision = {
        "prospective_gate_passed": bool(all(clauses.values())),
        "clauses": clauses,
        "test_wins": int(test.win.sum()),
        "test_cells": len(test),
        "mean_relative_test_gain_pct": float(test.relative_gain_pct.mean()),
        "median_relative_test_gain_pct": float(test.relative_gain_pct.median()),
        "mean_relative_test_gain_pct_by_task": {
            key: float(value) for key, value in task_means.items()
        },
        "mean_relative_test_gain_pct_by_dataset": {
            key: float(value) for key, value in dataset_means.items()
        },
        "positive_dataset_means": int((dataset_means > 0).sum()),
        "claim": (
            "Replacing one of three homogeneous T-PLE members with a fixed "
            "task-level alternate chart improves an equal-compute T-PLE bag "
            "across this frozen prospective panel."
        ),
    }
    (RESULTS / "heterobag_three_member_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
