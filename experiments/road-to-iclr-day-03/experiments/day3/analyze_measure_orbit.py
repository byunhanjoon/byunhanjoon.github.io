"""Frozen-gate analysis for the measure-aware Orbit-TabM screen."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .measure_orbit import CONFIG_PATH, RESULTS


def bootstrap(values: np.ndarray, samples: int, confidence: float) -> tuple[float, float]:
    rng = np.random.default_rng(17011)
    draws = np.asarray([rng.choice(values, len(values), replace=True).mean() for _ in range(samples)])
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(draws, alpha)), float(np.quantile(draws, 1.0 - alpha))


def main() -> None:
    cfg = json.loads(CONFIG_PATH.read_text())
    paths = sorted(RESULTS.glob("runs*.csv"))
    if not paths:
        raise FileNotFoundError("No measure-orbit runs")
    runs = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    key = ["dataset", "seed", "arm"]
    if runs.duplicated(key).any():
        raise AssertionError("Duplicate cells")
    expected = len(cfg["screen"]["datasets"]) * len(cfg["screen"]["seeds"]) * len(cfg["screen"]["arms"])
    if len(runs) != expected:
        raise AssertionError(f"Expected {expected} runs, found {len(runs)}")
    if not np.isfinite(runs[["test_proper_loss", "test_metric"]].to_numpy()).all():
        raise AssertionError("Non-finite outcomes")
    parameters = runs.groupby(["dataset", "seed"]).parameters.nunique()
    if (parameters != 1).any():
        raise AssertionError("Parameter mismatch")
    index = ["dataset", "task", "seed"]
    wide = runs.pivot(index=index, columns="arm", values=["test_proper_loss", "test_metric", "train_seconds"]).reset_index()
    wide.columns = ["_".join(value).strip("_") for value in wide.columns]
    wide["relative_proper_loss_reduction"] = (
        wide.test_proper_loss_baseline_tabm - wide.test_proper_loss_measure_orbit_tabm
    ) / wide.test_proper_loss_baseline_tabm.abs().clip(lower=1e-12)
    wide["official_gain"] = np.where(
        wide.task == "binclass",
        100.0 * (wide.test_metric_measure_orbit_tabm - wide.test_metric_baseline_tabm),
        100.0 * (wide.test_metric_baseline_tabm - wide.test_metric_measure_orbit_tabm) / wide.test_metric_baseline_tabm.abs().clip(lower=1e-12),
    )
    wide["runtime_ratio"] = wide.train_seconds_measure_orbit_tabm / wide.train_seconds_baseline_tabm
    wide.to_csv(RESULTS / "paired.csv", index=False)
    dataset = wide.groupby(["dataset", "task"], as_index=False).agg(
        relative_proper_loss_reduction=("relative_proper_loss_reduction", "mean"),
        official_gain=("official_gain", "mean"),
        paired_wins=("relative_proper_loss_reduction", lambda value: int((value > 0).sum())),
        runtime_ratio=("runtime_ratio", "mean"),
    )
    dataset.to_csv(RESULTS / "dataset_summary.csv", index=False)
    gate = cfg["claim_gate"]
    values = dataset.relative_proper_loss_reduction.to_numpy()
    lower, upper = bootstrap(values, int(gate["bootstrap_samples"]), float(gate["confidence"]))
    adult = dataset.loc[dataset.dataset == "adult"].iloc[0]
    clauses = {
        "minimum_mean_loss_reduction": float(values.mean()) >= float(gate["minimum_mean_relative_proper_loss_reduction"]),
        "dataset_win_fraction": float((values > 0).mean()) >= float(gate["minimum_dataset_win_fraction"]),
        "positive_bootstrap_lower": lower > 0.0,
        "adult_accuracy_gain": float(adult.official_gain) >= float(gate["adult_minimum_accuracy_gain_percentage_points"]),
    }
    summary = {
        "integrity": {"expected_runs": expected, "completed_runs": len(runs), "parameter_matched": True, "failures": 0},
        "primary": {
            "mean_relative_proper_loss_reduction": float(values.mean()),
            "dataset_bootstrap_interval": [lower, upper],
            "dataset_wins": int((values > 0).sum()),
            "datasets": len(dataset),
            "paired_wins": int((wide.relative_proper_loss_reduction > 0).sum()),
            "paired_runs": len(wide),
            "mean_official_gain": float(dataset.official_gain.mean()),
            "adult_accuracy_gain_percentage_points": float(adult.official_gain),
            "mean_runtime_ratio": float(wide.runtime_ratio.mean()),
        },
        "gate_clauses": clauses,
        "gate_passed": bool(all(clauses.values())),
    }
    (RESULTS / "analysis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(dataset.to_string(index=False))


if __name__ == "__main__":
    main()
