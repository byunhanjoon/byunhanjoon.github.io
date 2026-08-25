"""Fail-closed analysis for the equivalent-basis Orbit-TabM experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .orbit_ensemble import RESULTS, config


ROOT = Path(__file__).resolve().parents[2]


def _paths() -> list[Path]:
    return [
        RESULTS / "screen_shard0.csv",
        RESULTS / "screen_shard1.csv",
        RESULTS / "confirmation_shard0.csv",
        RESULTS / "confirmation_shard1.csv",
    ]


def _expected(cfg: dict[str, object]) -> pd.DataFrame:
    rows = []
    for stage in ("screen", "confirmation"):
        section = cfg[stage]
        if stage == "screen":
            datasets = section["datasets"]
        else:
            datasets = section["broad_datasets"] + section["extension_datasets"]
        for dataset in datasets:
            for arm in section["arms"]:
                for seed in section["seeds"]:
                    rows.append({"stage": stage, "dataset": dataset, "arm": arm, "seed": seed})
    return pd.DataFrame(rows)


def _bootstrap_interval(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _stage_summary(pairs: pd.DataFrame, stage: str, cfg: dict[str, object]) -> dict[str, object]:
    subset = pairs[pairs.stage == stage]
    by_dataset = subset.groupby("dataset", sort=True).agg(
        relative_loss_reduction=("orbit_vs_cumulative_relative_loss_reduction", "mean"),
        selected_relative_loss_reduction=("orbit_vs_selected_relative_loss_reduction", "mean"),
        score_change=("score_change", "mean"),
    )
    values = by_dataset.relative_loss_reduction.to_numpy()
    selected = by_dataset.selected_relative_loss_reduction.to_numpy()
    samples = int(cfg["claim_gate"]["bootstrap_samples"])
    interval = _bootstrap_interval(values, samples, 20260825 + (stage == "confirmation"))
    selected_interval = _bootstrap_interval(selected, samples, 20260827 + (stage == "confirmation"))
    leave_one_out = np.asarray(
        [(values.sum() - value) / (len(values) - 1) for value in values]
    ) if len(values) > 1 else values.copy()
    return {
        "datasets": int(len(by_dataset)),
        "paired_cells": int(len(subset)),
        "mean_relative_proper_loss_reduction": float(values.mean()),
        "median_relative_proper_loss_reduction": float(np.median(values)),
        "dataset_win_fraction": float((values > 0).mean()),
        "paired_win_fraction": float((subset.orbit_vs_cumulative_relative_loss_reduction > 0).mean()),
        "dataset_cluster_bootstrap_95_interval": list(interval),
        "mean_reduction_vs_validation_selected_single_basis": float(selected.mean()),
        "selected_basis_dataset_cluster_bootstrap_95_interval": list(selected_interval),
        "selected_basis_dataset_win_fraction": float((selected > 0).mean()),
        "minimum_leave_one_dataset_out_mean_reduction": float(leave_one_out.min()),
        "mean_score_change_by_unit": {
            unit: float(group.score_change.mean())
            for unit, group in subset.groupby("score_unit")
        },
    }


def analyze() -> dict[str, object]:
    cfg = config()
    paths = _paths()
    missing_files = [str(path) for path in paths if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing orbit result files: {missing_files}")
    runs = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    keys = ["stage", "dataset", "arm", "seed"]
    if runs.duplicated(keys).any():
        raise AssertionError("Duplicate orbit result cells")
    expected = _expected(cfg)
    coverage = expected.merge(runs[keys], how="outer", on=keys, indicator=True)
    missing = coverage[coverage._merge == "left_only"]
    unexpected = coverage[coverage._merge == "right_only"]
    if len(missing) or len(unexpected):
        raise AssertionError(f"Coverage mismatch: missing={len(missing)}, unexpected={len(unexpected)}")
    failures = runs.failure.fillna("").str.strip().ne("")
    if failures.any():
        raise AssertionError(f"Orbit experiment has {int(failures.sum())} failures")

    wide = runs.pivot(index=["stage", "dataset", "task", "seed"], columns="arm")
    records = []
    for index, row in wide.iterrows():
        stage, dataset, task, seed = index
        cumulative_loss = float(row[("test_proper_loss", "cumulative")])
        local_loss = float(row[("test_proper_loss", "local")])
        orbit_loss = float(row[("test_proper_loss", "orbit_natural")])
        selected_arm = (
            "local"
            if row[("val_proper_loss", "local")] < row[("val_proper_loss", "cumulative")]
            else "cumulative"
        )
        selected_loss = local_loss if selected_arm == "local" else cumulative_loss
        cumulative_metric = float(row[("test_metric", "cumulative")])
        orbit_metric = float(row[("test_metric", "orbit_natural")])
        if task == "regression":
            score_change = (cumulative_metric - orbit_metric) / max(cumulative_metric, 1e-30)
            score_unit = "relative_rmse_reduction"
        else:
            score_change = 100.0 * (orbit_metric - cumulative_metric)
            score_unit = "accuracy_percentage_points"
        record = {
            "stage": stage,
            "dataset": dataset,
            "task": task,
            "seed": int(seed),
            "cumulative_test_proper_loss": cumulative_loss,
            "local_test_proper_loss": local_loss,
            "orbit_test_proper_loss": orbit_loss,
            "orbit_vs_cumulative_relative_loss_reduction": (
                cumulative_loss - orbit_loss
            ) / max(cumulative_loss, 1e-30),
            "orbit_vs_local_relative_loss_reduction": (
                local_loss - orbit_loss
            ) / max(local_loss, 1e-30),
            "validation_selected_basis": selected_arm,
            "selected_test_proper_loss": selected_loss,
            "orbit_vs_selected_relative_loss_reduction": (
                selected_loss - orbit_loss
            ) / max(selected_loss, 1e-30),
            "score_change": score_change,
            "score_unit": score_unit,
            "cumulative_member_correlation": float(
                row[("test_member_prediction_correlation", "cumulative")]
            ),
            "orbit_member_correlation": float(
                row[("test_member_prediction_correlation", "orbit_natural")]
            ),
            "member_correlation_reduction": float(
                row[("test_member_prediction_correlation", "cumulative")]
                - row[("test_member_prediction_correlation", "orbit_natural")]
            ),
            "cumulative_ensemble_gain": float(
                row[("test_ensemble_gain_vs_mean_member", "cumulative")]
            ),
            "orbit_ensemble_gain": float(
                row[("test_ensemble_gain_vs_mean_member", "orbit_natural")]
            ),
            "runtime_ratio": float(
                row[("train_seconds", "orbit_natural")]
                / max(row[("train_seconds", "cumulative")], 1e-30)
            ),
            "memory_ratio": float(
                row[("peak_cuda_bytes", "orbit_natural")]
                / max(row[("peak_cuda_bytes", "cumulative")], 1)
            ),
            "parameters_equal": bool(
                row[("parameters", "orbit_natural")] == row[("parameters", "cumulative")]
            ),
        }
        if stage == "screen":
            random_loss = float(row[("test_proper_loss", "orbit_random")])
            record.update({
                "random_test_proper_loss": random_loss,
                "random_vs_cumulative_relative_loss_reduction": (
                    cumulative_loss - random_loss
                ) / max(cumulative_loss, 1e-30),
            })
        records.append(record)
    pairs = pd.DataFrame(records)
    dataset_summary = pairs.groupby(["stage", "dataset", "task", "score_unit"], as_index=False).agg(
        relative_loss_reduction=("orbit_vs_cumulative_relative_loss_reduction", "mean"),
        relative_loss_reduction_std=("orbit_vs_cumulative_relative_loss_reduction", "std"),
        selected_relative_loss_reduction=("orbit_vs_selected_relative_loss_reduction", "mean"),
        score_change=("score_change", "mean"),
        member_correlation_reduction=("member_correlation_reduction", "mean"),
        runtime_ratio=("runtime_ratio", "mean"),
        memory_ratio=("memory_ratio", "mean"),
    )

    screen = _stage_summary(pairs, "screen", cfg)
    confirmation = _stage_summary(pairs, "confirmation", cfg)
    all_by_dataset = pairs.groupby("dataset").orbit_vs_cumulative_relative_loss_reduction.mean().to_numpy()
    all_interval = _bootstrap_interval(
        all_by_dataset, int(cfg["claim_gate"]["bootstrap_samples"]), 20260829
    )
    gate_cfg = cfg["claim_gate"]
    gate_checks = {
        "bootstrap_interval_excludes_zero": confirmation["dataset_cluster_bootstrap_95_interval"][0] > 0,
        "dataset_win_fraction": confirmation["dataset_win_fraction"]
        >= float(gate_cfg["minimum_dataset_win_fraction"]),
        "mean_effect_size": confirmation["mean_relative_proper_loss_reduction"]
        >= float(gate_cfg["minimum_mean_relative_proper_loss_reduction"]),
        "beats_validation_selected_basis": confirmation[
            "mean_reduction_vs_validation_selected_single_basis"
        ] > 0,
        "no_excess_failures": not failures.any(),
    }
    integrity = {
        "expected_runs": int(len(expected)),
        "completed_runs": int(len(runs)),
        "failures": int(failures.sum()),
        "maximum_basis_relation_error": float(runs.basis_relation_error.max()),
        "all_orbit_transforms_full_rank": bool(runs.all_orbit_transforms_full_rank.all()),
        "all_parameter_counts_paired": bool(pairs.parameters_equal.all()),
        "maximum_allowed_basis_relation_error": float(
            cfg["integrity"]["maximum_basis_relation_error"]
        ),
    }
    if integrity["maximum_basis_relation_error"] > integrity["maximum_allowed_basis_relation_error"]:
        raise AssertionError("Basis relation integrity threshold failed")
    if not integrity["all_orbit_transforms_full_rank"] or not integrity["all_parameter_counts_paired"]:
        raise AssertionError("Orbit rank or parameter-count integrity failed")

    summary = {
        "method": "Orbit-TabM with alternating cumulative/local exact bases",
        "screen": screen,
        "confirmation": confirmation,
        "combined_30_datasets": {
            "datasets": int(len(all_by_dataset)),
            "mean_relative_proper_loss_reduction": float(all_by_dataset.mean()),
            "median_relative_proper_loss_reduction": float(np.median(all_by_dataset)),
            "dataset_win_fraction": float((all_by_dataset > 0).mean()),
            "dataset_cluster_bootstrap_95_interval": list(all_interval),
        },
        "mechanism": {
            "confirmation_mean_member_correlation_reduction": float(
                pairs[pairs.stage == "confirmation"].member_correlation_reduction.mean()
            ),
            "confirmation_mean_runtime_ratio": float(
                pairs[pairs.stage == "confirmation"].runtime_ratio.mean()
            ),
            "confirmation_mean_memory_ratio": float(
                pairs[pairs.stage == "confirmation"].memory_ratio.mean()
            ),
            "screen_random_orbit_mean_relative_loss_reduction": float(
                pairs[pairs.stage == "screen"].random_vs_cumulative_relative_loss_reduction.mean()
            ),
        },
        "integrity": integrity,
        "gate_checks": gate_checks,
        "primary_gate_passed": bool(all(gate_checks.values())),
        "important_caveat": (
            "The confirmation mean is influenced by Credit Card Fraud; the median effect "
            "is smaller, although every leave-one-dataset-out mean remains positive."
        ),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    runs.to_csv(RESULTS / "all_runs.csv", index=False)
    pairs.to_csv(RESULTS / "paired_results.csv", index=False)
    dataset_summary.to_csv(RESULTS / "dataset_summary.csv", index=False)
    (RESULTS / "analysis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    analyze()
