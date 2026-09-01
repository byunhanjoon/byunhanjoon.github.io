#!/usr/bin/env python3
"""Aggregate all frozen second-panel HeteroBag triplets and controls."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def ordinary_triplet(name: str, files: list[str]) -> pd.DataFrame:
    frame = pd.concat([pd.read_csv(RESULTS / file) for file in files], ignore_index=True)
    frame["triplet"] = name
    frame["relative_test_gain_pct"] = np.nan
    classification = frame["task"].eq("classification")
    frame.loc[classification, "relative_test_gain_pct"] = frame.loc[
        classification, "relative_test_log_loss_gain_pct"
    ]
    frame.loc[~classification, "relative_test_gain_pct"] = frame.loc[
        ~classification, "relative_test_rmse_gain_pct"
    ]
    return frame[["triplet", "dataset", "task", "model", "relative_test_gain_pct"]]


def main() -> None:
    triplet1 = pd.read_csv(RESULTS / "heterobag_phase1_cells.csv")
    triplet1["triplet"] = "triplet1"
    triplet1 = triplet1[["triplet", "dataset", "task", "model", "relative_test_gain_pct"]]
    triplet2 = ordinary_triplet(
        "triplet2",
        ["heterobag_triplet2_classification.csv", "heterobag_triplet2_regression.csv"],
    )
    mechanism = pd.concat(
        [
            pd.read_csv(RESULTS / "heterobag_mechanism_classification.csv"),
            pd.read_csv(RESULTS / "heterobag_mechanism_regression.csv"),
        ],
        ignore_index=True,
    )
    triplet3 = mechanism.assign(
        triplet="triplet3",
        relative_test_gain_pct=mechanism["heterobag_relative_test_gain_vs_ttt_pct"],
    )[["triplet", "dataset", "task", "model", "relative_test_gain_pct"]]
    pooled = pd.concat([triplet1, triplet2, triplet3], ignore_index=True)
    pooled["win"] = pooled["relative_test_gain_pct"] > 0

    triplet_summary = (
        pooled.groupby("triplet", as_index=False)
        .agg(
            cells=("dataset", "size"),
            wins=("win", "sum"),
            mean_relative_test_gain_pct=("relative_test_gain_pct", "mean"),
            median_relative_test_gain_pct=("relative_test_gain_pct", "median"),
        )
    )
    dataset = (
        pooled.groupby(["dataset", "task"], as_index=False)
        .agg(
            repeated_cells=("model", "size"),
            wins=("win", "sum"),
            mean_relative_test_gain_pct=("relative_test_gain_pct", "mean"),
        )
    )
    task = (
        pooled.groupby("task", as_index=False)
        .agg(
            cells=("dataset", "size"),
            wins=("win", "sum"),
            mean_relative_test_gain_pct=("relative_test_gain_pct", "mean"),
        )
    )
    controls = pd.DataFrame(
        {
            "dataset": mechanism["dataset"],
            "task": mechanism["task"],
            "model": mechanism["model"],
            "heterobag_gain_pct": mechanism["heterobag_relative_test_gain_vs_ttt_pct"],
            "homogeneous_alternate_gain_pct": mechanism[
                "alternate_homogeneous_relative_test_gain_vs_ttt_pct"
            ],
            "coordinate_placebo_gain_pct": mechanism[
                "transformed_t_placebo_relative_test_gain_vs_ttt_pct"
            ],
        }
    )
    controls["heterobag_minus_homogeneous_alternate_pct"] = (
        controls["heterobag_gain_pct"] - controls["homogeneous_alternate_gain_pct"]
    )
    controls["heterobag_minus_coordinate_placebo_pct"] = (
        controls["heterobag_gain_pct"] - controls["coordinate_placebo_gain_pct"]
    )

    mechanism_rows = []
    gain = mechanism["heterobag_relative_test_gain_vs_ttt_pct"]
    for family in ("same_representation", "cross_representation", "coordinate_placebo"):
        for metric in (
            "prediction_correlation",
            "error_correlation",
            "mean_absolute_disagreement",
            "mean_squared_disagreement",
        ):
            value = mechanism[f"{family}_{metric}"]
            result = spearmanr(value, gain, nan_policy="omit")
            mechanism_rows.append(
                {
                    "pair_family": family,
                    "diagnostic": metric,
                    "spearman_with_heterobag_gain": float(result.statistic),
                    "two_sided_p": float(result.pvalue),
                }
            )
    mechanism_summary = pd.DataFrame(mechanism_rows)

    rng = np.random.default_rng(20260828)
    values = dataset["mean_relative_test_gain_pct"].to_numpy()
    boot = values[rng.integers(0, len(values), size=(100_000, len(values)))].mean(axis=1)
    positive_triplets = int((triplet_summary["mean_relative_test_gain_pct"] > 0).sum())
    clauses = {
        "positive_dataset_mean_in_at_least_two_triplets": positive_triplets >= 2,
        "pooled_win_rate_at_least_65pct": float(pooled["win"].mean()) >= 0.65,
        "positive_pooled_classification_mean": float(
            task.loc[task["task"].eq("classification"), "mean_relative_test_gain_pct"].iloc[0]
        ) > 0,
        "positive_pooled_regression_mean": float(
            task.loc[task["task"].eq("regression"), "mean_relative_test_gain_pct"].iloc[0]
        ) > 0,
        "triplet3_beats_homogeneous_alternate_in_panel_mean": float(
            controls["heterobag_minus_homogeneous_alternate_pct"].mean()
        ) > 0,
        "triplet3_beats_coordinate_placebo_in_panel_mean": float(
            controls["heterobag_minus_coordinate_placebo_pct"].mean()
        ) > 0,
    }
    summary = {
        "status": "complete",
        "evidence_label": "PROSPECTIVE_CONFIRMATORY_CONDITIONAL",
        "datasets": int(len(dataset)),
        "architectures": int(pooled["model"].nunique()),
        "seed_triplets": int(pooled["triplet"].nunique()),
        "pooled_cells": int(len(pooled)),
        "pooled_wins": int(pooled["win"].sum()),
        "pooled_win_rate": float(pooled["win"].mean()),
        "pooled_mean_relative_test_gain_pct": float(pooled["relative_test_gain_pct"].mean()),
        "positive_pooled_dataset_means": int((dataset["mean_relative_test_gain_pct"] > 0).sum()),
        "dataset_bootstrap_mean_gain_95_ci_pct": [
            float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
        ],
        "triplet3_mean_heterobag_gain_pct": float(controls["heterobag_gain_pct"].mean()),
        "triplet3_mean_homogeneous_alternate_gain_pct": float(
            controls["homogeneous_alternate_gain_pct"].mean()
        ),
        "triplet3_mean_coordinate_placebo_gain_pct": float(
            controls["coordinate_placebo_gain_pct"].mean()
        ),
        "triplet3_heterobag_minus_coordinate_placebo_pct": float(
            controls["heterobag_minus_coordinate_placebo_pct"].mean()
        ),
        "maximum_relative_active_parameter_mismatch": float(
            mechanism["maximum_alternate_parameter_relative_mismatch"].max()
        ),
        "clauses": clauses,
        "phase2_gate_passed": bool(all(clauses.values())),
        "standalone_promotion_blockers": [
            "coordinate_placebo_control_not_beaten_in_panel_mean"
            if not clauses["triplet3_beats_coordinate_placebo_in_panel_mean"] else None,
            "tabm_not_run",
            "conditional_reused_dataset_panel",
            "active_parameter_match_not_exact",
            "diversity_predictor_not_tested_on_an_untouched_dataset_panel",
        ],
    }
    summary["standalone_promotion_blockers"] = [
        value for value in summary["standalone_promotion_blockers"] if value is not None
    ]

    pooled.to_csv(RESULTS / "heterobag_phase2_pooled_cells.csv", index=False)
    triplet_summary.to_csv(RESULTS / "heterobag_phase2_triplets.csv", index=False)
    dataset.to_csv(RESULTS / "heterobag_phase2_datasets.csv", index=False)
    task.to_csv(RESULTS / "heterobag_phase2_tasks.csv", index=False)
    controls.to_csv(RESULTS / "heterobag_phase2_controls.csv", index=False)
    mechanism_summary.to_csv(RESULTS / "heterobag_phase2_mechanism.csv", index=False)
    (RESULTS / "heterobag_phase2_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
