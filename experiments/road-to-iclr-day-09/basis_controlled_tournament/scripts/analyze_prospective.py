#!/usr/bin/env python3
"""Aggregate the untouched prospective panel into the four frozen rankings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_stage2 import aggregate, summarize  # noqa: E402
from scripts.run_prospective import load_finalists  # noqa: E402
from tournament.common import load_protocol, prospective_specs, write_json  # noqa: E402


def category(row: pd.Series, *, require_breadth: bool = True) -> str:
    reduction = float(row["median_disagreement_reduction"])
    cost = float(row["median_relative_task_change"])
    breadth = int(row["model_families"])
    successful_breadth = int(row.get("successful_model_families", breadth))
    if reduction >= 0.70 and cost <= 0.01 and (successful_breadth >= 2 or not require_breadth):
        return "KEEP"
    if require_breadth and successful_breadth == 1 and reduction >= 0.50:
        return "NICHE"
    if reduction >= 0.50 and cost <= 0.03:
        return "PROMISING"
    if (reduction >= 0.50 and breadth == 1) or (reduction >= 0.30 and cost <= 0.03):
        return "NICHE"
    return "FAIL"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    finalist_config, finalist_hash = load_finalists()
    protocol = load_protocol()
    union_models = sorted(
        {model for finalist in finalist_config["finalists"] for model in finalist["applicable_models"]}
    )
    files = sorted((ROOT / "results" / "processed" / "prospective_cells").glob("*.csv"))
    expected = len(union_models) * len(prospective_specs()) * len(protocol["model_seeds"])
    if not args.allow_partial and len(files) != expected:
        raise RuntimeError(f"prospective panel incomplete: {len(files)}/{expected} cells")
    units = aggregate(files, "Raw")
    units["predictive_rank"] = units.groupby(
        ["dataset", "model", "seed", "split"], sort=False
    )["task_error"].rank(method="average", ascending=True)
    processed = ROOT / "results" / "processed"
    units.to_csv(processed / "prospective_units.csv", index=False)
    summary = summarize(units)
    test_units = units[units["split"] == "test"]
    per_model_gate = (
        test_units.groupby(["method", "model"], as_index=False)
        .agg(
            model_median_reduction=("disagreement_reduction", "median"),
            model_median_task_cost=("relative_task_change", "median"),
        )
    )
    per_model_gate["passes_keep_gate"] = (
        (per_model_gate["model_median_reduction"] >= 0.70)
        & (per_model_gate["model_median_task_cost"] <= 0.01)
    )
    successful_breadth = per_model_gate.groupby("method")["passes_keep_gate"].sum()
    summary["successful_model_families"] = summary["method"].map(successful_breadth).fillna(0).astype(int)
    summary["category"] = summary.apply(category, axis=1)
    summary.to_csv(processed / "prospective_method_summary.csv", index=False)

    development = pd.read_csv(processed / "development_all_method_summary.csv")
    method_map = {
        finalist["method_id"]: finalist.get("development_method", finalist["method_id"])
        for finalist in finalist_config["finalists"]
    }
    development_lookup = development.set_index("method")
    consistency = {}
    development_units = pd.read_csv(processed / "development_all_units.csv")
    for prospective_method, development_method in method_map.items():
        selected = development_units[
            (development_units["method"] == development_method)
            & (development_units["split"] == "test")
        ]
        consistency[prospective_method] = float(selected["disagreement_reduction"].std(ddof=0))
    consistency["Raw"] = 0.0
    summary["development_reduction_std"] = summary["method"].map(consistency)
    summary["development_median_reduction"] = summary["method"].map(
        {
            method: float(development_lookup.loc[source, "median_disagreement_reduction"])
            for method, source in method_map.items()
            if source in development_lookup.index
        }
    )
    summary.loc[summary["method"] == "Raw", "development_median_reduction"] = 0.0

    ranking_a = summary[summary["performance_preserving_eligible"]].sort_values(
        ["median_disagreement_reduction", "median_worst_orbit_gain", "development_reduction_std"],
        ascending=[False, False, True],
    )
    ranking_b = summary[summary["pareto_frontier"]].sort_values(
        ["median_relative_task_change", "median_disagreement_reduction"], ascending=[True, False]
    )
    ranking_c = summary.sort_values(
        ["median_predictive_rank", "mean_predictive_rank", "median_relative_task_change"]
    )
    ranking_d = summary.sort_values("paper_method_score", ascending=False)
    ranking_a.to_csv(processed / "prospective_ranking_A.csv", index=False)
    ranking_b.to_csv(processed / "prospective_ranking_B_pareto.csv", index=False)
    ranking_c.to_csv(processed / "prospective_ranking_C_predictive.csv", index=False)
    ranking_d.to_csv(processed / "prospective_ranking_D_score.csv", index=False)

    matrix_records: list[dict[str, Any]] = []
    test = units[units["split"] == "test"]
    for (method, model), frame in test.groupby(["method", "model"], sort=True):
        record = {
            "method": method,
            "model": model,
            "median_disagreement_reduction": float(frame["disagreement_reduction"].median()),
            "median_relative_task_change": float(frame["relative_task_change"].median()),
            "model_families": 1,
        }
        record["category"] = category(pd.Series(record), require_breadth=False)
        matrix_records.append(record)
    matrix = pd.DataFrame(matrix_records)
    matrix.to_csv(processed / "prospective_method_model_matrix.csv", index=False)
    if len(matrix):
        wide = matrix.pivot(index="method", columns="model", values="category").reset_index()
        wide.to_csv(processed / "prospective_method_model_matrix_wide.csv", index=False)

    write_json(
        processed / "prospective_analysis.json",
        {
            "complete": len(files) == expected,
            "files": len(files),
            "expected_files": expected,
            "models": union_models,
            "finalist_configs_sha256": finalist_hash,
            "ranking_A": ranking_a.to_dict(orient="records"),
            "ranking_B": ranking_b.to_dict(orient="records"),
            "ranking_C": ranking_c.to_dict(orient="records"),
            "ranking_D": ranking_d.to_dict(orient="records"),
            "method_model_matrix": matrix_records,
        },
    )
    print(summary.sort_values("paper_method_score", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
