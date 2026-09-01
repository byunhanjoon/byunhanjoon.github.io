#!/usr/bin/env python3
"""Aggregate the frozen conditional second-panel HeteroBag screen."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main() -> None:
    config = json.loads((HERE / "heterobag_phase1_config.json").read_text())
    frames = [
        pd.read_csv(RESULTS / "heterobag_phase1_classification.csv"),
        pd.read_csv(RESULTS / "heterobag_phase1_regression.csv"),
    ]
    frame = pd.concat(frames, ignore_index=True)
    expected = {
        (dataset, architecture)
        for dataset in config["development_datasets"]
        for architecture in config["architectures"]
    }
    observed = set(zip(frame["dataset"], frame["model"]))
    if observed != expected or len(frame) != len(expected):
        raise RuntimeError(
            f"incomplete or duplicate panel: expected={len(expected)}, "
            f"rows={len(frame)}, missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )
    frame["primary_metric"] = np.where(
        frame["task"].eq("classification"), "log_loss", "rmse"
    )
    frame["relative_test_gain_pct"] = np.where(
        frame["task"].eq("classification"),
        frame["relative_test_log_loss_gain_pct"],
        frame["relative_test_rmse_gain_pct"],
    )
    frame["relative_val_gain_pct"] = np.where(
        frame["task"].eq("classification"),
        frame["relative_val_log_loss_gain_pct"],
        frame["relative_val_rmse_gain_pct"],
    )
    frame["test_win"] = frame["relative_test_gain_pct"] > 0
    frame["parameter_relative_mismatch"] = (
        (frame["alternate_parameters"] - frame["anchor_parameters"]).abs()
        / frame["anchor_parameters"]
    )
    dataset = (
        frame.groupby(["dataset", "task"], as_index=False)
        .agg(
            architecture_cells=("model", "size"),
            test_wins=("test_win", "sum"),
            mean_relative_test_gain_pct=("relative_test_gain_pct", "mean"),
            median_relative_test_gain_pct=("relative_test_gain_pct", "median"),
        )
    )
    architecture = (
        frame.groupby("model", as_index=False)
        .agg(
            cells=("dataset", "size"),
            test_wins=("test_win", "sum"),
            mean_relative_test_gain_pct=("relative_test_gain_pct", "mean"),
        )
    )
    task_means = frame.groupby("task")["relative_test_gain_pct"].mean().to_dict()
    rng = np.random.default_rng(20260828)
    values = dataset["mean_relative_test_gain_pct"].to_numpy()
    boot = values[rng.integers(0, len(values), size=(100_000, len(values)))].mean(axis=1)
    gate = config["phase1_gate"]
    clauses = {
        "minimum_test_wins": int(frame["test_win"].sum()) >= gate["minimum_test_wins"],
        "positive_overall_mean": float(frame["relative_test_gain_pct"].mean()) > 0,
        "positive_classification_mean": float(task_means["classification"]) > 0,
        "positive_regression_mean": float(task_means["regression"]) > 0,
        "minimum_positive_dataset_means": int(
            (dataset["mean_relative_test_gain_pct"] > 0).sum()
        ) >= gate["minimum_positive_dataset_means"],
        "no_architecture_below_floor": float(
            architecture["mean_relative_test_gain_pct"].min()
        ) >= gate["minimum_architecture_mean_gain_pct"],
    }
    summary = {
        "status": "complete",
        "evidence_label": config["evidence_label"],
        "cells": int(len(frame)),
        "datasets": int(len(dataset)),
        "architectures": config["architectures"],
        "test_wins": int(frame["test_win"].sum()),
        "positive_dataset_means": int((dataset["mean_relative_test_gain_pct"] > 0).sum()),
        "mean_relative_test_gain_pct": float(frame["relative_test_gain_pct"].mean()),
        "median_relative_test_gain_pct": float(frame["relative_test_gain_pct"].median()),
        "task_mean_relative_test_gain_pct": {key: float(value) for key, value in task_means.items()},
        "dataset_bootstrap_mean_gain_95_ci_pct": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
        "maximum_relative_active_parameter_mismatch": float(
            frame["parameter_relative_mismatch"].max()
        ),
        "exact_active_parameter_match_all_cells": bool(
            (frame["parameter_relative_mismatch"] == 0).all()
        ),
        "protocol_deviation_incremental_test_printing": True,
        "clauses": clauses,
        "phase1_gate_passed": bool(all(clauses.values())),
    }
    frame.to_csv(RESULTS / "heterobag_phase1_cells.csv", index=False)
    dataset.to_csv(RESULTS / "heterobag_phase1_datasets.csv", index=False)
    architecture.to_csv(RESULTS / "heterobag_phase1_architectures.csv", index=False)
    (RESULTS / "heterobag_phase1_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
