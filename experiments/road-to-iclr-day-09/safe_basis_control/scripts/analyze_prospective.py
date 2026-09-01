#!/usr/bin/env python3
"""Aggregate the prospective run at the dataset x model unit and create five rankings."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safe_basis.common import bd, write_json  # noqa: E402


def bootstrap_ci(units: pd.DataFrame, method: str, column: str, draws: int = 5000) -> tuple[float, float]:
    frame = units[units["method"] == method]
    dataset_values = frame.groupby("dataset")[column].median()
    rng = np.random.default_rng(bd.stable_seed("prospective-bootstrap", method, column))
    values = dataset_values.to_numpy(float)
    samples = np.empty(draws)
    for index in range(draws):
        samples[index] = np.median(rng.choice(values, size=len(values), replace=True))
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def summarize(units: pd.DataFrame, raw_cells: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fixed = units[units["method"] == "GramAnchor-m16"][["dataset", "model", "normalized_excess_risk"]].rename(columns={"normalized_excess_risk": "fixed_C"})
    for method, frame in units.groupby("method", sort=False):
        costs = frame["normalized_excess_risk"].to_numpy(float)
        differences = frame["absolute_task_difference"].to_numpy(float)
        cells = raw_cells[(raw_cells["method"] == method) & (raw_cells["split"] == "test")]
        alpha_counts = cells["alpha"].value_counts(normalize=True).to_dict()
        joined = frame.merge(fixed, on=["dataset", "model"], how="left")
        catastrophic_fixed = joined["fixed_C"] > 0.20
        prevented = float((joined.loc[catastrophic_fixed, "normalized_excess_risk"] <= 0.20).mean()) if catastrophic_fixed.any() else float("nan")
        successful_models = 0
        for _, model_frame in frame.groupby("model"):
            model_costs = model_frame["normalized_excess_risk"].to_numpy(float)
            successful_models += int(
                model_frame["disagreement_reduction"].median() >= 0.70
                and np.median(model_costs) <= 0.01
                and np.quantile(model_costs, 0.95) <= 0.05
                and np.max(model_costs) <= 0.20
            )
        ci_reduction = bootstrap_ci(units, method, "disagreement_reduction")
        ci_cost = bootstrap_ci(units, method, "normalized_excess_risk")
        rows.append(
            {
                "method": method,
                "units": len(frame),
                "datasets": frame["dataset"].nunique(),
                "model_families": frame["model"].nunique(),
                "successful_model_families": successful_models,
                "median_disagreement_reduction": float(frame["disagreement_reduction"].median()),
                "median_reduction_ci_low": ci_reduction[0],
                "median_reduction_ci_high": ci_reduction[1],
                "median_C": float(np.median(costs)),
                "median_C_ci_low": ci_cost[0],
                "median_C_ci_high": ci_cost[1],
                "p90_C": float(np.quantile(costs, 0.90)),
                "p95_C": float(np.quantile(costs, 0.95)),
                "max_C": float(np.max(costs)),
                "task_improvement_fraction": float((differences < -1e-6).mean()),
                "wins": int((differences < -1e-6).sum()),
                "ties": int((np.abs(differences) <= 1e-6).sum()),
                "losses": int((differences > 1e-6).sum()),
                "raw_fallback_rate": float(cells["raw_fallback"].mean()),
                "alpha_0_fraction": float(alpha_counts.get(0.0, 0.0)),
                "alpha_025_fraction": float(alpha_counts.get(0.25, 0.0)),
                "alpha_05_fraction": float(alpha_counts.get(0.5, 0.0)),
                "alpha_075_fraction": float(alpha_counts.get(0.75, 0.0)),
                "alpha_1_fraction": float(alpha_counts.get(1.0, 0.0)),
                "catastrophic_fixed_cells_prevented_fraction": prevented,
                "safety_first_eligible": bool(
                    np.median(costs) <= 0.01
                    and np.quantile(costs, 0.95) <= 0.05
                    and np.max(costs) <= 0.20
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    processed = ROOT / "results" / "processed"
    cells = pd.read_csv(processed / "prospective_cells.csv")
    test = cells[cells["split"] == "test"].copy()
    units = (
        test.groupby(["dataset", "problem_type", "model", "method"], as_index=False)
        .agg(
            alpha=("alpha", "median"),
            raw_fallback=("raw_fallback", "max"),
            raw_disagreement=("raw_disagreement", "median"),
            method_disagreement=("method_disagreement", "median"),
            disagreement_reduction=("disagreement_reduction", "median"),
            raw_loss=("raw_loss", "median"),
            method_loss=("method_loss", "median"),
            trivial_loss=("trivial_loss", "median"),
            absolute_task_difference=("absolute_task_difference", "median"),
            relative_task_difference=("relative_task_difference", "median"),
            normalized_excess_risk=("normalized_excess_risk", "median"),
            denominator_sensitive=("denominator_sensitive", "max"),
            fit_seconds=("fit_seconds", "median"),
        )
    )
    units["predictive_rank"] = units.groupby(["dataset", "model"])["method_loss"].rank(method="average")
    units.to_csv(processed / "prospective_units.csv", index=False)
    summary = summarize(units, cells)
    predictive = units.groupby("method", as_index=False).agg(mean_predictive_rank=("predictive_rank", "mean"), median_predictive_rank=("predictive_rank", "median"))
    summary = summary.merge(predictive, on="method")
    summary["paper_candidate_score"] = (
        summary["median_disagreement_reduction"]
        - 3 * summary["median_C"].clip(lower=0)
        - 2 * (summary["p95_C"] - 0.05).clip(lower=0)
        - 2 * (summary["max_C"] - 0.20).clip(lower=0)
    )
    summary.to_csv(processed / "prospective_aggregate.csv", index=False)

    nonsensitive = units[~units["denominator_sensitive"]]
    nonsensitive_summary = summarize(nonsensitive, cells)
    nonsensitive_summary.to_csv(processed / "prospective_aggregate_excluding_sensitive_denominators.csv", index=False)

    ranking_a = summary.sort_values(["safety_first_eligible", "median_disagreement_reduction"], ascending=[False, False]).copy()
    ranking_a["eligibility_note"] = np.where(ranking_a["safety_first_eligible"], "ELIGIBLE", "EXCLUDED_BY_TAIL_GATE")
    ranking_a.to_csv(processed / "ranking_A_safety_first.csv", index=False)
    ranking_b = summary.sort_values("median_disagreement_reduction", ascending=False)
    ranking_b.to_csv(processed / "ranking_B_invariance.csv", index=False)
    ranking_c = summary.sort_values(["mean_predictive_rank", "task_improvement_fraction"], ascending=[True, False])
    ranking_c.to_csv(processed / "ranking_C_predictive.csv", index=False)
    ranking_d = summary.sort_values(["p95_C", "max_C", "catastrophic_fixed_cells_prevented_fraction"], ascending=[True, True, False])
    ranking_d.to_csv(processed / "ranking_D_tail_robustness.csv", index=False)
    ranking_e = summary.sort_values("paper_candidate_score", ascending=False)
    ranking_e.to_csv(processed / "ranking_E_paper_candidate.csv", index=False)

    write_json(
        processed / "prospective_analysis.json",
        {
            "status": "COMPLETE",
            "primary_unit": "dataset x model",
            "seed_aggregation": "median",
            "bootstrap_unit": "dataset",
            "bootstrap_draws": 5000,
            "cells": len(cells),
            "units": len(units),
            "methods": int(units["method"].nunique()),
            "datasets": int(units["dataset"].nunique()),
            "model_families": int(units["model"].nunique()),
            "denominator_sensitive_units": int(units["denominator_sensitive"].sum()),
            "all_methods_retained_in_rankings": True,
        },
    )
    print(summary[["method", "median_disagreement_reduction", "median_C", "p95_C", "max_C", "raw_fallback_rate", "safety_first_eligible", "paper_candidate_score"]].sort_values("paper_candidate_score", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
