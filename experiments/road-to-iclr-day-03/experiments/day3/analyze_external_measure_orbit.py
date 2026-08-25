"""Analyze the frozen external Selective Measure-Orbit experiment."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .core import loss_numpy, make_prepared, metric
from .external_measure_orbit import (
    RESULTS,
    config,
    load_external_dataset,
    prediction_path,
    verify_locked_sources,
)


def bootstrap(
    values: np.ndarray, samples: int, confidence: float
) -> tuple[float, float]:
    rng = np.random.default_rng(55109)
    draws = np.asarray(
        [rng.choice(values, len(values), replace=True).mean() for _ in range(samples)]
    )
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(draws, alpha)),
        float(np.quantile(draws, 1.0 - alpha)),
    )


def average_predictions(task: str, predictions: list[np.ndarray]) -> np.ndarray:
    if len(predictions) < 1:
        raise ValueError("At least one prediction is required")
    if task == "binclass":
        probabilities = [
            1.0 / (1.0 + np.exp(-np.clip(value.reshape(-1), -40, 40)))
            for value in predictions
        ]
        mean = np.clip(np.mean(probabilities, axis=0), 1e-7, 1 - 1e-7)
        return np.log(mean / (1 - mean))[:, None]
    if task == "multiclass":
        probabilities = []
        for value in predictions:
            shifted = value - value.max(axis=1, keepdims=True)
            probability = np.exp(shifted)
            probabilities.append(probability / probability.sum(axis=1, keepdims=True))
        mean = np.clip(np.mean(probabilities, axis=0), 1e-12, None)
        return np.log(mean)
    return np.mean(predictions, axis=0)


def _predictions(dataset: str, seed: int, arm: str) -> dict[str, np.ndarray]:
    path = prediction_path(dataset, seed, arm)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as values:
        return {"val": values["val"], "test": values["test"]}


def _relative_reduction(control: float, candidate: float) -> float:
    return (control - candidate) / max(abs(control), 1e-12)


def _official_gain(task: str, control: float, candidate: float) -> float:
    if task == "regression":
        return 100.0 * (control - candidate) / max(abs(control), 1e-12)
    return 100.0 * (candidate - control)


def evaluate_gate(
    dataset_values: np.ndarray,
    paired_values: np.ndarray,
    gate: dict[str, Any],
    failures: int,
) -> dict[str, Any]:
    lower, upper = bootstrap(
        dataset_values,
        int(gate.get("bootstrap_samples", config()["primary_gate"]["bootstrap_samples"])),
        float(gate.get("confidence", config()["primary_gate"]["confidence"])),
    )
    clauses = {
        "minimum_mean_loss_reduction": float(dataset_values.mean())
        >= float(gate["minimum_mean_relative_proper_loss_reduction"]),
        "dataset_win_fraction": float((dataset_values > 0).mean())
        >= float(gate["minimum_dataset_win_fraction"]),
        "positive_bootstrap_lower": (lower > 0.0)
        if bool(gate["require_positive_dataset_bootstrap_lower_bound"])
        else True,
        "paired_win_fraction": float((paired_values > 0).mean())
        >= float(gate["minimum_paired_win_fraction"]),
        "no_excess_failures": failures == 0
        if not bool(gate["allow_excess_failures"])
        else True,
    }
    return {
        "mean_relative_proper_loss_reduction": float(dataset_values.mean()),
        "dataset_bootstrap_interval": [lower, upper],
        "positive_dataset_means": int((dataset_values > 0).sum()),
        "datasets": int(len(dataset_values)),
        "positive_paired_cells": int((paired_values > 0).sum()),
        "paired_cells": int(len(paired_values)),
        "clauses": clauses,
        "passed": bool(all(clauses.values())),
    }


def main() -> None:
    verify_locked_sources()
    cfg = config()
    paths = sorted(RESULTS.glob("runs*.csv"))
    if not paths:
        raise FileNotFoundError("No external Measure-Orbit results")
    runs = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    key = ["dataset", "seed", "arm"]
    if runs.duplicated(key).any():
        raise AssertionError("Duplicate external result cells")
    expected = len(cfg["datasets"]) * len(cfg["seeds"]) * len(cfg["arms"])
    if len(runs) != expected:
        raise AssertionError(f"Expected {expected} runs, found {len(runs)}")
    if set(runs.dataset) != set(cfg["datasets"]):
        raise AssertionError("External dataset mismatch")
    if set(runs.seed.astype(int)) != set(cfg["seeds"]):
        raise AssertionError("External seed mismatch")
    if set(runs.arm) != set(cfg["arms"]):
        raise AssertionError("External arm mismatch")
    failures = int(runs.failure.fillna("").ne("").sum())
    numeric_columns = [
        "val_proper_loss",
        "test_proper_loss",
        "test_metric",
        "parameters",
        "gradient_updates",
        "train_seconds",
    ]
    if not np.isfinite(runs[numeric_columns].to_numpy()).all():
        raise AssertionError("Non-finite external result")
    if (runs.groupby(["dataset", "seed"]).parameters.nunique() != 1).any():
        raise AssertionError("Parameter mismatch")

    paired_rows: list[dict[str, Any]] = []
    for dataset_name in cfg["datasets"]:
        dataset = load_external_dataset(dataset_name)
        empty = {
            part: np.empty((len(dataset.y[part]), 0), dtype=np.float32)
            for part in ("train", "val", "test")
        }
        prepared = make_prepared(dataset, empty, {})
        for seed_value in cfg["seeds"]:
            seed = int(seed_value)
            group = runs[(runs.dataset == dataset_name) & (runs.seed == seed)].set_index(
                "arm"
            )
            baseline = group.loc["baseline_anchor"]
            orbit = group.loc["measure_orbit"]
            seedmate = group.loc["baseline_seedmate_update_matched"]
            if int(orbit.gradient_updates) != int(seedmate.gradient_updates):
                raise AssertionError(
                    f"Update mismatch for {dataset_name}/s{seed}: "
                    f"{orbit.gradient_updates} != {seedmate.gradient_updates}"
                )
            predictions = {
                arm: _predictions(dataset_name, seed, arm) for arm in cfg["arms"]
            }
            selected_arm = (
                "measure_orbit"
                if float(orbit.val_proper_loss) < float(baseline.val_proper_loss)
                else "baseline_anchor"
            )
            selected = predictions[selected_arm]
            seed_ensemble = {
                part: average_predictions(
                    dataset.task,
                    [
                        predictions["baseline_anchor"][part],
                        predictions["baseline_seedmate_update_matched"][part],
                    ],
                )
                for part in ("val", "test")
            }
            selected_test_loss = loss_numpy(
                dataset.task, selected["test"], prepared.y["test"]
            )
            selected_test_metric = metric(
                prepared, selected["test"], prepared.y["test"]
            )
            seed_test_loss = loss_numpy(
                dataset.task, seed_ensemble["test"], prepared.y["test"]
            )
            seed_test_metric = metric(
                prepared, seed_ensemble["test"], prepared.y["test"]
            )
            baseline_test_loss = float(baseline.test_proper_loss)
            baseline_test_metric = float(baseline.test_metric)
            orbit_test_loss = float(orbit.test_proper_loss)
            selective_seconds = float(baseline.train_seconds + orbit.train_seconds)
            seed_seconds = float(baseline.train_seconds + seedmate.train_seconds)
            paired_rows.append(
                {
                    "dataset": dataset_name,
                    "task": dataset.task,
                    "seed": seed,
                    "selected_arm": selected_arm,
                    "selected_val_proper_loss": loss_numpy(
                        dataset.task, selected["val"], prepared.y["val"]
                    ),
                    "selected_test_proper_loss": selected_test_loss,
                    "selected_test_metric": selected_test_metric,
                    "seed_ensemble_val_proper_loss": loss_numpy(
                        dataset.task,
                        seed_ensemble["val"],
                        prepared.y["val"],
                    ),
                    "seed_ensemble_test_proper_loss": seed_test_loss,
                    "seed_ensemble_test_metric": seed_test_metric,
                    "baseline_test_proper_loss": baseline_test_loss,
                    "baseline_test_metric": baseline_test_metric,
                    "selective_vs_seed_relative_proper_loss_reduction": _relative_reduction(
                        seed_test_loss, selected_test_loss
                    ),
                    "selective_vs_baseline_relative_proper_loss_reduction": _relative_reduction(
                        baseline_test_loss, selected_test_loss
                    ),
                    "seed_vs_baseline_relative_proper_loss_reduction": _relative_reduction(
                        baseline_test_loss, seed_test_loss
                    ),
                    "raw_orbit_vs_baseline_relative_proper_loss_reduction": _relative_reduction(
                        baseline_test_loss, orbit_test_loss
                    ),
                    "validation_prefers_orbit": selected_arm == "measure_orbit",
                    "orbit_test_better_than_baseline": orbit_test_loss
                    < baseline_test_loss,
                    "selective_vs_seed_official_gain": _official_gain(
                        dataset.task, seed_test_metric, selected_test_metric
                    ),
                    "selective_vs_baseline_official_gain": _official_gain(
                        dataset.task, baseline_test_metric, selected_test_metric
                    ),
                    "selective_gradient_updates": int(
                        baseline.gradient_updates + orbit.gradient_updates
                    ),
                    "seed_ensemble_gradient_updates": int(
                        baseline.gradient_updates + seedmate.gradient_updates
                    ),
                    "selective_train_seconds": selective_seconds,
                    "seed_ensemble_train_seconds": seed_seconds,
                    "seed_over_selective_time_ratio": seed_seconds
                    / max(selective_seconds, 1e-12),
                }
            )

    paired = pd.DataFrame(paired_rows)
    paired.to_csv(RESULTS / "paired.csv", index=False)
    dataset_summary = paired.groupby(["dataset", "task"], as_index=False).agg(
        selective_vs_seed_relative_proper_loss_reduction=(
            "selective_vs_seed_relative_proper_loss_reduction",
            "mean",
        ),
        selective_vs_baseline_relative_proper_loss_reduction=(
            "selective_vs_baseline_relative_proper_loss_reduction",
            "mean",
        ),
        seed_vs_baseline_relative_proper_loss_reduction=(
            "seed_vs_baseline_relative_proper_loss_reduction",
            "mean",
        ),
        raw_orbit_vs_baseline_relative_proper_loss_reduction=(
            "raw_orbit_vs_baseline_relative_proper_loss_reduction",
            "mean",
        ),
        selective_vs_seed_official_gain=(
            "selective_vs_seed_official_gain",
            "mean",
        ),
        selective_vs_baseline_official_gain=(
            "selective_vs_baseline_official_gain",
            "mean",
        ),
        orbit_activations=(
            "selected_arm",
            lambda values: int((values == "measure_orbit").sum()),
        ),
        paired_seed_wins=(
            "selective_vs_seed_relative_proper_loss_reduction",
            lambda values: int((values > 0).sum()),
        ),
        paired_baseline_wins=(
            "selective_vs_baseline_relative_proper_loss_reduction",
            lambda values: int((values > 0).sum()),
        ),
        mean_time_ratio=("seed_over_selective_time_ratio", "mean"),
    )
    dataset_summary.to_csv(RESULTS / "dataset_summary.csv", index=False)

    seed_dataset = dataset_summary[
        "selective_vs_seed_relative_proper_loss_reduction"
    ].to_numpy()
    seed_paired = paired[
        "selective_vs_seed_relative_proper_loss_reduction"
    ].to_numpy()
    baseline_dataset = dataset_summary[
        "selective_vs_baseline_relative_proper_loss_reduction"
    ].to_numpy()
    baseline_paired = paired[
        "selective_vs_baseline_relative_proper_loss_reduction"
    ].to_numpy()
    primary = evaluate_gate(seed_dataset, seed_paired, cfg["primary_gate"], failures)
    preservation = evaluate_gate(
        baseline_dataset,
        baseline_paired,
        cfg["baseline_preservation_gate"],
        failures,
    )
    exact_updates = bool(
        (
            paired.selective_gradient_updates
            == paired.seed_ensemble_gradient_updates
        ).all()
    )
    summary = {
        "integrity": {
            "expected_runs": int(expected),
            "completed_runs": int(len(runs)),
            "failures": failures,
            "datasets": int(len(cfg["datasets"])),
            "seeds": int(len(cfg["seeds"])),
            "parameter_matched": True,
            "exact_gradient_update_match": exact_updates,
            "prediction_files": int(
                sum(
                    prediction_path(dataset, int(seed), arm).exists()
                    for dataset in cfg["datasets"]
                    for seed in cfg["seeds"]
                    for arm in cfg["arms"]
                )
            ),
        },
        "primary_vs_two_seed_prediction_ensemble": primary,
        "preservation_vs_single_baseline": preservation,
        "diagnostics": {
            "mean_seed_over_selective_time_ratio": float(
                paired.seed_over_selective_time_ratio.mean()
            ),
            "median_seed_over_selective_time_ratio": float(
                paired.seed_over_selective_time_ratio.median()
            ),
            "mean_seed_ensemble_vs_baseline_reduction": float(
                dataset_summary.seed_vs_baseline_relative_proper_loss_reduction.mean()
            ),
            "mean_raw_orbit_vs_baseline_reduction": float(
                dataset_summary.raw_orbit_vs_baseline_relative_proper_loss_reduction.mean()
            ),
            "raw_orbit_positive_dataset_means": int(
                (
                    dataset_summary.raw_orbit_vs_baseline_relative_proper_loss_reduction
                    > 0
                ).sum()
            ),
            "raw_orbit_positive_paired_cells": int(
                (
                    paired.raw_orbit_vs_baseline_relative_proper_loss_reduction
                    > 0
                ).sum()
            ),
            "orbit_activations": int(
                (paired.selected_arm == "measure_orbit").sum()
            ),
            "helpful_orbit_activations": int(
                (
                    paired.validation_prefers_orbit
                    & paired.orbit_test_better_than_baseline
                ).sum()
            ),
            "harmful_orbit_activations": int(
                (
                    paired.validation_prefers_orbit
                    & ~paired.orbit_test_better_than_baseline
                ).sum()
            ),
            "missed_helpful_orbit_cells": int(
                (
                    ~paired.validation_prefers_orbit
                    & paired.orbit_test_better_than_baseline
                ).sum()
            ),
            "seed_ensemble_positive_paired_cells": int(
                (paired.seed_vs_baseline_relative_proper_loss_reduction > 0).sum()
            ),
            "paired_cells": int(len(paired)),
        },
        "external_claim_validated": bool(
            primary["passed"] and preservation["passed"] and exact_updates
        ),
    }
    (RESULTS / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(dataset_summary.to_string(index=False))
    raise SystemExit(0)


if __name__ == "__main__":
    main()
