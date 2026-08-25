"""Prospective analysis for validation-selected Measure-Orbit."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .selective_measure_orbit import CONFIG_PATH, RESULTS


def bootstrap(values: np.ndarray, samples: int, confidence: float) -> tuple[float, float]:
    rng = np.random.default_rng(81283)
    draws = np.asarray([rng.choice(values, len(values), replace=True).mean() for _ in range(samples)])
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(draws, alpha)), float(np.quantile(draws, 1.0 - alpha))


def main() -> None:
    cfg = json.loads(CONFIG_PATH.read_text())
    paths = sorted(RESULTS.glob("runs*.csv"))
    if not paths:
        raise FileNotFoundError("No selective Measure-Orbit results")
    runs = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    key = ["dataset", "seed", "arm"]
    if runs.duplicated(key).any():
        raise AssertionError("Duplicate result cells")
    datasets = list(cfg["confirmation"]["broad_datasets"]) + list(cfg["confirmation"]["extension_datasets"])
    expected = len(datasets) * len(cfg["confirmation"]["seeds"]) * len(cfg["confirmation"]["candidate_arms"])
    if len(runs) != expected:
        raise AssertionError(f"Expected {expected} runs, found {len(runs)}")
    numeric = ["val_proper_loss", "test_proper_loss", "test_metric", "parameters"]
    if not np.isfinite(runs[numeric].to_numpy()).all():
        raise AssertionError("Non-finite outcomes")
    if (runs.groupby(["dataset", "seed"]).parameters.nunique() != 1).any():
        raise AssertionError("Parameter mismatch")

    rows = []
    for (dataset, task, seed), group in runs.groupby(["dataset", "task", "seed"]):
        baseline = group[group.arm == "baseline_tabm"].iloc[0]
        orbit = group[group.arm == "measure_orbit_tabm"].iloc[0]
        selected = orbit if orbit.val_proper_loss < baseline.val_proper_loss else baseline
        selected_gain = (baseline.test_proper_loss - selected.test_proper_loss) / max(abs(baseline.test_proper_loss), 1e-12)
        orbit_gain = (baseline.test_proper_loss - orbit.test_proper_loss) / max(abs(baseline.test_proper_loss), 1e-12)
        if task == "regression":
            official_gain = 100.0 * (baseline.test_metric - selected.test_metric) / max(abs(baseline.test_metric), 1e-12)
        else:
            official_gain = 100.0 * (selected.test_metric - baseline.test_metric)
        rows.append({
            "dataset": dataset,
            "task": task,
            "seed": seed,
            "selected_arm": selected.arm,
            "selected_relative_proper_loss_reduction": selected_gain,
            "orbit_relative_proper_loss_reduction": orbit_gain,
            "selected_official_gain": official_gain,
            "validation_margin": float(baseline.val_proper_loss - orbit.val_proper_loss),
        })
    paired = pd.DataFrame(rows)
    paired.to_csv(RESULTS / "paired.csv", index=False)
    dataset = paired.groupby(["dataset", "task"], as_index=False).agg(
        selected_relative_proper_loss_reduction=("selected_relative_proper_loss_reduction", "mean"),
        orbit_relative_proper_loss_reduction=("orbit_relative_proper_loss_reduction", "mean"),
        selected_official_gain=("selected_official_gain", "mean"),
        activations=("selected_arm", lambda value: int((value == "measure_orbit_tabm").sum())),
        paired_wins=("selected_relative_proper_loss_reduction", lambda value: int((value > 0).sum())),
    )
    dataset.to_csv(RESULTS / "dataset_summary.csv", index=False)
    gate = cfg["claim_gate"]
    values = dataset.selected_relative_proper_loss_reduction.to_numpy()
    lower, upper = bootstrap(values, int(gate["bootstrap_samples"]), float(gate["confidence"]))
    raw_values = dataset.orbit_relative_proper_loss_reduction.to_numpy()
    raw_lower, raw_upper = bootstrap(raw_values, int(gate["bootstrap_samples"]), float(gate["confidence"]))
    baseline_runs = runs[runs.arm == "baseline_tabm"]
    seed_control_rows = []
    for dataset_name, group in baseline_runs.groupby("dataset"):
        by_seed = group.set_index("seed")
        for seed in cfg["confirmation"]["seeds"]:
            alternative_seed = (int(seed) + 1) % len(cfg["confirmation"]["seeds"])
            anchor = by_seed.loc[int(seed)]
            alternative = by_seed.loc[alternative_seed]
            chosen = alternative if alternative.val_proper_loss < anchor.val_proper_loss else anchor
            seed_control_rows.append({
                "dataset": dataset_name,
                "seed": int(seed),
                "relative_proper_loss_reduction": (
                    anchor.test_proper_loss - chosen.test_proper_loss
                ) / max(abs(anchor.test_proper_loss), 1e-12),
            })
    seed_control = pd.DataFrame(seed_control_rows)
    seed_control.to_csv(RESULTS / "seed_selection_control.csv", index=False)
    seed_control_mean = float(seed_control.groupby("dataset").relative_proper_loss_reduction.mean().mean())
    win_fraction = float((values > 0).mean())
    paired_win_fraction = float((paired.selected_relative_proper_loss_reduction > 0).mean())
    clauses = {
        "minimum_mean_loss_reduction": float(values.mean()) >= float(gate["minimum_mean_relative_proper_loss_reduction"]),
        "dataset_win_fraction": win_fraction >= float(gate["minimum_dataset_win_fraction"]),
        "positive_bootstrap_lower": lower > 0.0,
        "paired_win_fraction": paired_win_fraction >= float(gate["minimum_paired_win_fraction"]),
        "no_excess_failures": True,
    }
    summary = {
        "integrity": {"expected_runs": expected, "completed_runs": len(runs), "failures": 0, "parameter_matched": True},
        "primary": {
            "mean_selected_relative_proper_loss_reduction": float(values.mean()),
            "dataset_bootstrap_interval": [lower, upper],
            "positive_dataset_means": int((values > 0).sum()),
            "datasets": len(dataset),
            "positive_paired_cells": int((paired.selected_relative_proper_loss_reduction > 0).sum()),
            "paired_cells": len(paired),
            "activations": int((paired.selected_arm == "measure_orbit_tabm").sum()),
            "mean_official_gain": float(dataset.selected_official_gain.mean()),
            "raw_orbit_mean_reduction": float(dataset.orbit_relative_proper_loss_reduction.mean()),
            "raw_orbit_dataset_bootstrap_interval": [raw_lower, raw_upper],
            "raw_orbit_positive_dataset_means": int((raw_values > 0).sum()),
            "raw_orbit_positive_paired_cells": int((paired.orbit_relative_proper_loss_reduction > 0).sum()),
            "two_baseline_seed_selection_mean_reduction": seed_control_mean,
        },
        "gate_clauses": clauses,
        "gate_passed": bool(all(clauses.values())),
    }
    (RESULTS / "analysis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(dataset.to_string(index=False))


if __name__ == "__main__":
    main()
