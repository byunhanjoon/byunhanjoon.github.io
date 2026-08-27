#!/usr/bin/env python3
"""Analyze the prospectively frozen external residual-cascade panel."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def validation_panel() -> pd.DataFrame:
    regression = pd.read_csv(RESULTS / "trichart_external_regression_validation.csv")
    regression["task"] = "regression"
    regression["validation_gain_pct"] = 100.0 * (
        regression.val_anchor_rmse - regression.val_rmse
    ) / regression.val_anchor_rmse
    classification = pd.read_csv(
        RESULTS / "trichart_external_classification_validation.csv"
    )
    classification["task"] = "classification"
    classification["validation_gain_pct"] = 100.0 * (
        classification.val_anchor_log_loss - classification.val_log_loss
    ) / classification.val_anchor_log_loss
    columns = [
        "task",
        "dataset",
        "model",
        "seed",
        "residual_best_epoch",
        "residual_gate",
        "residual_target_parameters",
        "residual_parameters",
        "residual_parameter_difference",
        "validation_gain_pct",
    ]
    return pd.concat(
        [regression[columns], classification[columns]], ignore_index=True
    ).sort_values(["task", "dataset", "model"])


def confirmation_panel() -> pd.DataFrame:
    regression = pd.read_csv(RESULTS / "trichart_external_regression.csv")
    regression["task"] = "regression"
    regression["validation_gain_pct"] = 100.0 * (
        regression.val_anchor_rmse - regression.val_rmse
    ) / regression.val_anchor_rmse
    regression["test_gain_pct"] = 100.0 * (
        regression.test_anchor_rmse - regression.test_rmse
    ) / regression.test_anchor_rmse
    classification = pd.read_csv(RESULTS / "trichart_external_classification.csv")
    classification["task"] = "classification"
    classification["validation_gain_pct"] = 100.0 * (
        classification.val_anchor_log_loss - classification.val_log_loss
    ) / classification.val_anchor_log_loss
    classification["test_gain_pct"] = 100.0 * (
        classification.test_anchor_log_loss - classification.test_log_loss
    ) / classification.test_anchor_log_loss
    columns = [
        "task",
        "dataset",
        "model",
        "seed",
        "residual_best_epoch",
        "residual_gate",
        "validation_gain_pct",
        "test_gain_pct",
    ]
    return pd.concat(
        [regression[columns], classification[columns]], ignore_index=True
    ).sort_values(["task", "dataset", "model"])


def main() -> None:
    panel = validation_panel()
    summary = panel.groupby(["task", "dataset"]).agg(
        cells=("model", "size"),
        safe_cells=("validation_gain_pct", lambda values: int((values >= -1e-9).sum())),
        strict_wins=("validation_gain_pct", lambda values: int((values > 1e-7).sum())),
        mean_validation_gain_pct=("validation_gain_pct", "mean"),
    ).reset_index()
    gates = {}
    for task, group in panel.groupby("task"):
        dataset_summary = summary.query("task == @task")
        gates[task] = {
            "cells": int(len(group)),
            "safe_cells": int((group.validation_gain_pct >= -1e-9).sum()),
            "strict_wins": int((group.validation_gain_pct > 1e-7).sum()),
            "mean_validation_gain_pct": float(group.validation_gain_pct.mean()),
            "positive_datasets": int(
                (dataset_summary.mean_validation_gain_pct > 0).sum()
            ),
            "gate_passed": bool(
                (group.validation_gain_pct >= -1e-9).all()
                and (group.validation_gain_pct > 1e-7).sum() >= 4
                and (dataset_summary.mean_validation_gain_pct > 0).all()
            ),
        }
    decision = {
        "prospective_validation_only": True,
        "test_metrics_opened": False,
        "method": "frozen T-PLE anchor plus parameter-matched zero-start T-PLE residual",
        "tasks": gates,
        "joint_gate_passed": bool(all(task["gate_passed"] for task in gates.values())),
    }
    panel.to_csv(RESULTS / "trichart_external_validation_panel.csv", index=False)
    summary.to_csv(RESULTS / "trichart_external_validation_summary.csv", index=False)
    (RESULTS / "trichart_external_validation_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    print(summary.to_string(index=False))
    print(json.dumps(decision, indent=2, sort_keys=True))

    regression_path = RESULTS / "trichart_external_regression.csv"
    classification_path = RESULTS / "trichart_external_classification.csv"
    if not (regression_path.exists() and classification_path.exists()):
        return
    confirmation = confirmation_panel()
    reproduced = panel.merge(
        confirmation,
        on=["task", "dataset", "model", "seed"],
        suffixes=("_blind", "_opened"),
        validate="one_to_one",
    )
    if not (
        (reproduced.validation_gain_pct_blind - reproduced.validation_gain_pct_opened)
        .abs()
        .max()
        < 1e-12
    ):
        raise RuntimeError("validation rerun did not exactly reproduce")
    confirmation_summary = confirmation.groupby(["task", "dataset"]).agg(
        cells=("model", "size"),
        validation_wins=("validation_gain_pct", lambda values: int((values > 1e-7).sum())),
        mean_validation_gain_pct=("validation_gain_pct", "mean"),
        test_wins=("test_gain_pct", lambda values: int((values > 0).sum())),
        mean_test_gain_pct=("test_gain_pct", "mean"),
    ).reset_index()
    confirmation_decision = {
        "prospective_validation_gate_passed_before_test": decision[
            "joint_gate_passed"
        ],
        "test_metrics_opened": True,
        "validation_exactly_reproduced": True,
        "cells": int(len(confirmation)),
        "test_wins": int((confirmation.test_gain_pct > 0).sum()),
        "mean_test_gain_pct": float(confirmation.test_gain_pct.mean()),
        "tasks": {
            task: {
                "cells": int(len(group)),
                "test_wins": int((group.test_gain_pct > 0).sum()),
                "mean_test_gain_pct": float(group.test_gain_pct.mean()),
            }
            for task, group in confirmation.groupby("task")
        },
    }
    confirmation.to_csv(
        RESULTS / "trichart_external_confirmation.csv", index=False
    )
    confirmation_summary.to_csv(
        RESULTS / "trichart_external_confirmation_summary.csv", index=False
    )
    (RESULTS / "trichart_external_confirmation_decision.json").write_text(
        json.dumps(confirmation_decision, indent=2, sort_keys=True) + "\n"
    )
    print(confirmation_summary.to_string(index=False))
    print(json.dumps(confirmation_decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
