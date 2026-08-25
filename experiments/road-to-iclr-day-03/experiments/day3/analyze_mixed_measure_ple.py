"""Audit and analyze the preregistered mixed-measure PLE experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .mixed_measure_ple import CONFIG_PATH, RESULTS


def interval(values: np.ndarray, samples: int, confidence: float, seed: int = 9417) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = np.asarray([
        rng.choice(values, size=len(values), replace=True).mean() for _ in range(samples)
    ])
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(draws, alpha)), float(np.quantile(draws, 1.0 - alpha))


def main() -> None:
    cfg = json.loads(CONFIG_PATH.read_text())
    paths = sorted(RESULTS.glob("runs*.csv"))
    if not paths and (RESULTS / "runs.csv").exists():
        paths = [RESULTS / "runs.csv"]
    if not paths:
        raise FileNotFoundError("No mixed-measure result shards")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    key = ["dataset", "model", "seed", "representation"]
    if frame.duplicated(key).any():
        raise AssertionError("Duplicate result cells")
    expected = len(cfg["datasets"]) * len(cfg["models"]) * len(cfg["seeds"]) * len(cfg["representations"])
    if len(frame) != expected:
        raise AssertionError(f"Expected {expected} completed cells, found {len(frame)}")
    if not np.isfinite(frame[["test_loss", "test_score"]].to_numpy()).all():
        raise AssertionError("Non-finite outcome")
    counts = frame.groupby(["dataset", "model", "seed"]).parameters.nunique()
    if (counts != 1).any():
        raise AssertionError("Representations are not parameter matched")

    index = ["dataset", "task", "model", "seed"]
    wide = frame.pivot(index=index, columns="representation", values=["test_loss", "test_score"]).reset_index()
    wide.columns = ["_".join(value).strip("_") for value in wide.columns]
    for arm in ("mixed_measure_ple", "tail_reallocated_ple"):
        wide[f"{arm}_loss_reduction"] = (
            wide["test_loss_baseline_fixed_ple"] - wide[f"test_loss_{arm}"]
        ) / wide["test_loss_baseline_fixed_ple"].abs().clip(lower=1e-12)
        classification = wide.task == "binclass"
        wide[f"{arm}_official_gain"] = np.where(
            classification,
            100.0 * (wide[f"test_score_{arm}"] - wide["test_score_baseline_fixed_ple"]),
            100.0 * (wide["test_score_baseline_fixed_ple"] - wide[f"test_score_{arm}"]) / wide["test_score_baseline_fixed_ple"].abs().clip(lower=1e-12),
        )
    wide.to_csv(RESULTS / "paired.csv", index=False)

    dataset = wide.groupby(["dataset", "task"], as_index=False).agg(
        mixed_loss_reduction=("mixed_measure_ple_loss_reduction", "mean"),
        mixed_official_gain=("mixed_measure_ple_official_gain", "mean"),
        tail_loss_reduction=("tail_reallocated_ple_loss_reduction", "mean"),
        tail_official_gain=("tail_reallocated_ple_official_gain", "mean"),
    )
    dataset.to_csv(RESULTS / "dataset_summary.csv", index=False)

    gate = cfg["claim_gate"]
    values = dataset.mixed_loss_reduction.to_numpy()
    lower, upper = interval(values, int(gate["bootstrap_samples"]), float(gate["confidence"]))
    adult = wide[wide.dataset == "adult"]
    adult_models = adult.groupby("model").mixed_measure_ple_official_gain.mean()
    adult_gain = float(adult.mixed_measure_ple_official_gain.mean())
    win_fraction = float((dataset.mixed_loss_reduction > 0).mean())
    mean_loss = float(values.mean())
    mean_score = float(dataset.mixed_official_gain.mean())
    clauses = {
        "minimum_mean_loss_reduction": mean_loss >= float(gate["minimum_mean_relative_proper_loss_reduction"]),
        "dataset_win_fraction": win_fraction >= float(gate["minimum_dataset_win_fraction"]),
        "positive_bootstrap_lower": lower > 0.0,
        "positive_official_score": mean_score > 0.0,
        "adult_accuracy_gain": adult_gain >= float(gate["adult_minimum_accuracy_gain_percentage_points"]),
        "adult_all_backbones_positive": bool((adult_models > 0).all()),
    }
    summary = {
        "integrity": {
            "expected_cells": expected,
            "completed_cells": len(frame),
            "failures": 0,
            "parameter_matched": True,
        },
        "primary": {
            "mean_relative_proper_loss_reduction": mean_loss,
            "dataset_bootstrap_interval": [lower, upper],
            "dataset_wins": int((dataset.mixed_loss_reduction > 0).sum()),
            "datasets": len(dataset),
            "mean_official_gain": mean_score,
            "adult_accuracy_gain_percentage_points": adult_gain,
            "adult_backbone_gains": adult_models.to_dict(),
        },
        "control": {
            "tail_mean_relative_proper_loss_reduction": float(dataset.tail_loss_reduction.mean()),
            "tail_dataset_wins": int((dataset.tail_loss_reduction > 0).sum()),
        },
        "gate_clauses": clauses,
        "gate_passed": bool(all(clauses.values())),
    }
    (RESULTS / "analysis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(dataset.to_string(index=False))


if __name__ == "__main__":
    main()
