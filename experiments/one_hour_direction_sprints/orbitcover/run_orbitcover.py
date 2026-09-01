#!/usr/bin/env python3
"""Actual-predictive-loss audit for the frozen OrbitCover panel."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "experiments" / "final_closure" / "summaries" / "experiment_a_cells.csv"
RAW = ROOT / "experiments" / "final_closure" / "raw" / "experiment_a"
OUT = HERE / "results"
PRIMARY = "OC2-COUPLED"
BASELINE = "CANONICAL-INDEPENDENT"
BUDGET = 16
EXPECTED_CELLS = 12 * 3 * 4
BOOTSTRAPS = 50_000


def protocol_hash() -> str:
    return hashlib.sha256((HERE / "PROTOCOL.md").read_bytes()).hexdigest()


def cluster_interval(source_values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAPS, dtype=np.float64)
    for start in range(0, BOOTSTRAPS, 5_000):
        size = min(5_000, BOOTSTRAPS - start)
        indices = rng.integers(0, len(source_values), size=(size, len(source_values)))
        draws[start : start + size] = source_values[indices].mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def paired(frame: pd.DataFrame, method: str, budget: int) -> tuple[pd.DataFrame, dict]:
    subset = frame[
        (frame.budget == budget) & frame.method.isin((method, BASELINE))
    ]
    keys = ["dataset", "split_seed", "model", "task"]
    pivot = subset.pivot(index=keys, columns="method", values="predictive_loss").reset_index()
    if len(pivot) != EXPECTED_CELLS or pivot[[method, BASELINE]].isna().any().any():
        raise AssertionError(f"incomplete paired panel for {method} at B={budget}")
    pivot["absolute_improvement"] = pivot[BASELINE] - pivot[method]
    pivot["fractional_improvement"] = pivot.absolute_improvement / pivot[BASELINE].clip(lower=1e-12)
    source = pivot.groupby("dataset", as_index=False).agg(
        baseline=(BASELINE, "mean"), method=(method, "mean")
    )
    source["fractional_improvement"] = (source.baseline - source.method) / source.baseline.clip(lower=1e-12)
    low, high = cluster_interval(
        source.fractional_improvement.to_numpy(),
        int(hashlib.sha256(f"orbit-predictive|{method}|{budget}".encode()).hexdigest()[:8], 16),
    )
    architecture = pivot.groupby("model").agg(
        baseline=(BASELINE, "mean"), method=(method, "mean")
    )
    architecture["fractional_improvement"] = (
        architecture.baseline - architecture.method
    ) / architecture.baseline.clip(lower=1e-12)
    result = {
        "method": method,
        "budget": budget,
        "cells": len(pivot),
        "cell_wins": int((pivot.absolute_improvement > 0).sum()),
        "cell_win_rate": float((pivot.absolute_improvement > 0).mean()),
        "cell_ties": int((pivot.absolute_improvement == 0).sum()),
        "datasets": len(source),
        "dataset_wins": int((source.fractional_improvement > 0).sum()),
        "equal_dataset_mean_fractional_improvement": float(source.fractional_improvement.mean()),
        "dataset_clustered_95_interval": [low, high],
        "worst_dataset_fractional_improvement": float(source.fractional_improvement.min()),
        "best_dataset_fractional_improvement": float(source.fractional_improvement.max()),
        "architecture_fractional_improvement": {
            key: float(value) for key, value in architecture.fractional_improvement.items()
        },
        "two_sided_cell_sign_p": float(
            binomtest(int((pivot.absolute_improvement > 0).sum()), len(pivot), 0.5).pvalue
        ),
    }
    return pivot, result


def main() -> None:
    started = time.perf_counter()
    frame = pd.read_csv(SOURCE)
    manifests = list(RAW.glob("*/manifest.json"))
    required_columns = {
        "dataset", "split_seed", "model", "task", "method", "budget",
        "predictive_loss", "estimator_draws",
    }
    integrity = {
        "source_exists": SOURCE.exists(),
        "required_columns": required_columns.issubset(frame.columns),
        "raw_manifests_144": len(manifests) == EXPECTED_CELLS,
        "finite_predictive_loss": bool(np.isfinite(frame.predictive_loss).all()),
        "positive_predictive_loss": bool((frame.predictive_loss >= 0).all()),
        "estimator_draws_constant": frame.estimator_draws.nunique() == 1,
        "estimator_draws_at_least_512": int(frame.estimator_draws.min()) >= 512,
        "no_duplicate_method_cells": not frame.duplicated(
            ["dataset", "split_seed", "model", "task", "method", "budget"]
        ).any(),
    }
    if not all(integrity.values()):
        raise AssertionError(f"source integrity failed: {integrity}")

    primary_pivot, primary = paired(frame, PRIMARY, BUDGET)
    comparisons = []
    pivots = []
    for budget in sorted(frame.budget.unique()):
        for method in (PRIMARY, "OC2-INDEPENDENT", "SRS-JOINT"):
            pivot, result = paired(frame, method, int(budget))
            comparisons.append(result)
            pivots.append(pivot.assign(comparison_method=method, comparison_budget=int(budget)))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(pivots, ignore_index=True).to_csv(OUT / "paired_cells.csv", index=False)
    pd.DataFrame(comparisons).to_json(OUT / "comparisons.json", orient="records", indent=2)

    # Test-set oracle is an explicitly invalid selector, used only to measure
    # whether a future validation gate could have meaningful raw-loss headroom.
    oracle_loss = np.minimum(primary_pivot[PRIMARY], primary_pivot[BASELINE])
    oracle_fractional = (primary_pivot[BASELINE] - oracle_loss) / primary_pivot[BASELINE].clip(lower=1e-12)
    oracle_by_source = pd.DataFrame(
        {"dataset": primary_pivot.dataset, "fractional": oracle_fractional}
    ).groupby("dataset").fractional.mean()
    oracle = {
        "invalid_test_oracle_equal_dataset_mean_headroom": float(oracle_by_source.mean()),
        "invalid_test_oracle_cell_mean_headroom": float(oracle_fractional.mean()),
        "cells_where_orbitcover_is_oracle_choice": int((primary_pivot[PRIMARY] < primary_pivot[BASELINE]).sum()),
    }
    gates = {
        "integrity": all(integrity.values()),
        "mean_improvement_at_least_0_005": primary["equal_dataset_mean_fractional_improvement"] >= 0.005,
        "cluster_interval_lower_positive": primary["dataset_clustered_95_interval"][0] > 0,
        "cell_win_rate_at_least_60pct": primary["cell_win_rate"] >= 0.60,
        "at_least_8_dataset_wins": primary["dataset_wins"] >= 8,
        "all_architectures_nonnegative": all(
            value >= 0 for value in primary["architecture_fractional_improvement"].values()
        ),
        "no_dataset_worse_than_minus_1pct": primary["worst_dataset_fractional_improvement"] >= -0.01,
    }
    summary = {
        "status": "complete_cached_predictive_utility_audit",
        "protocol_sha256": protocol_hash(),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "wall_seconds": time.perf_counter() - started,
        "integrity": integrity,
        "primary": primary,
        "oracle_headroom": oracle,
        "gates": gates,
        "passed": all(gates.values()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
