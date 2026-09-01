#!/usr/bin/env python3
"""Aggregate development results into the four required rankings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tournament.common import load_protocol, write_json  # noqa: E402


def aggregate(files: list[Path], baseline_method: str) -> pd.DataFrame:
    if not files:
        return pd.DataFrame()
    rows = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    if "track" not in rows.columns:
        rows["track"] = "optimizer" if baseline_method == "AdamW" else "representation"
    records = []
    keys = ["dataset", "problem_type", "model", "seed", "method", "track", "split"]
    for key, frame in rows.groupby(keys, sort=True):
        reference = frame[frame["is_reference"]]
        orbit = frame[~frame["is_reference"]]
        if len(reference) != 1 or len(orbit) != 8:
            raise RuntimeError(f"incomplete orbit {key}: {len(reference)} reference, {len(orbit)} orbit")
        records.append(
            {
                **dict(zip(keys, key)),
                "disagreement": float(orbit["disagreement"].mean()),
                "max_disagreement": float(orbit["disagreement"].max()),
                "task_error": float(reference["task_error"].iloc[0]),
                "orbit_mean_task_error": float(orbit["task_error"].mean()),
                "worst_orbit_task_error": float(orbit["task_error"].max()),
                "runtime_seconds": float(frame["fit_seconds"].sum()),
            }
        )
    units = pd.DataFrame(records)
    baseline = units[units["method"] == baseline_method][
        ["dataset", "model", "seed", "split", "disagreement", "task_error", "worst_orbit_task_error"]
    ].rename(
        columns={
            "disagreement": "raw_disagreement",
            "task_error": "raw_task_error",
            "worst_orbit_task_error": "raw_worst_orbit_task_error",
        }
    )
    units = units.merge(
        baseline,
        on=["dataset", "model", "seed", "split"],
        how="left",
        validate="many_to_one",
    )
    units["disagreement_reduction"] = 1.0 - units["disagreement"] / units[
        "raw_disagreement"
    ].clip(lower=1e-12)
    units["relative_task_change"] = (
        units["task_error"] - units["raw_task_error"]
    ) / units["raw_task_error"].abs().clip(lower=1e-12)
    units["worst_orbit_gain"] = (
        units["raw_worst_orbit_task_error"] - units["worst_orbit_task_error"]
    ) / units["raw_worst_orbit_task_error"].abs().clip(lower=1e-12)
    units["failure"] = units["disagreement_reduction"] < -0.2
    return units


def summarize(units: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    test = units[units["split"] == "test"]
    for (method, track), frame in test.groupby(["method", "track"], sort=True):
        reduction = float(frame["disagreement_reduction"].median())
        cost = float(frame["relative_task_change"].median())
        failure = float(frame["failure"].mean())
        score = reduction - 5.0 * max(cost, 0.0) - 0.25 * failure
        differences = frame["task_error"] - frame["raw_task_error"]
        tolerance = 1e-10 * frame["raw_task_error"].abs().clip(lower=1.0)
        wins = int((differences < -tolerance).sum())
        ties = int((differences.abs() <= tolerance).sum())
        losses = int((differences > tolerance).sum())
        records.append(
            {
                "method": method,
                "track": track,
                "median_disagreement": float(frame["disagreement"].median()),
                "median_disagreement_reduction": reduction,
                "median_relative_task_change": cost,
                "median_task_error": float(frame["task_error"].median()),
                # Absolute errors cannot be pooled across classification and
                # regression datasets.  Rank the error inside each directly
                # comparable dataset/model/seed unit, then aggregate ranks.
                "median_predictive_rank": float(frame["predictive_rank"].median()),
                "mean_predictive_rank": float(frame["predictive_rank"].mean()),
                "median_worst_orbit_gain": float(frame["worst_orbit_gain"].median()),
                "failure_fraction": failure,
                "paper_method_score": score,
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "units": int(len(frame)),
                "model_families": int(frame["model"].nunique()),
                "datasets": int(frame["dataset"].nunique()),
                "runtime_seconds": float(frame["runtime_seconds"].sum()),
                "performance_preserving_eligible": cost <= 0.01,
            }
        )
    result = pd.DataFrame(records)
    pareto = []
    for row in result.itertuples():
        dominated = (
            (result["median_relative_task_change"] <= row.median_relative_task_change)
            & (result["median_disagreement_reduction"] >= row.median_disagreement_reduction)
            & (
                (result["median_relative_task_change"] < row.median_relative_task_change)
                | (result["median_disagreement_reduction"] > row.median_disagreement_reduction)
            )
        ).any()
        pareto.append(not bool(dominated))
    result["pareto_frontier"] = pareto
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol()
    optimizer_dir = ROOT / "results" / "processed" / "stage2_optimizer_cells"
    representation_dir = ROOT / "results" / "processed" / "stage2_representation_cells"
    optimizer_files = sorted(path for path in optimizer_dir.glob("*.csv") if "coordinate_audit" not in path.name)
    representation_files = sorted(
        path for path in representation_dir.glob("*.csv") if "coordinate_audit" not in path.name
    )
    expected_optimizer = len(protocol["development_datasets"]) * len(protocol["model_seeds"]) * 2
    expected_representation = len(protocol["development_datasets"]) * len(protocol["model_seeds"]) * 5
    if not args.allow_partial:
        if len(optimizer_files) != expected_optimizer:
            raise RuntimeError(
                f"optimizer panel incomplete: {len(optimizer_files)}/{expected_optimizer}"
            )
        if len(representation_files) != expected_representation:
            raise RuntimeError(
                f"representation panel incomplete: {len(representation_files)}/{expected_representation}"
            )
    optimizer = aggregate(optimizer_files, "AdamW")
    representation = aggregate(representation_files, "Raw")
    units = pd.concat([optimizer, representation], ignore_index=True)
    units["predictive_rank"] = units.groupby(
        ["dataset", "model", "seed", "split"], sort=False
    )["task_error"].rank(method="average", ascending=True)
    units.to_csv(ROOT / "results" / "processed" / "development_units.csv", index=False)
    summary = summarize(units)
    summary.to_csv(ROOT / "results" / "processed" / "development_method_summary.csv", index=False)
    ranking_a = summary[summary["performance_preserving_eligible"]].sort_values(
        ["median_disagreement_reduction", "median_worst_orbit_gain"], ascending=[False, False]
    )
    ranking_b = summary[summary["pareto_frontier"]].sort_values(
        "median_relative_task_change"
    )
    ranking_c = summary.sort_values(
        ["median_predictive_rank", "mean_predictive_rank", "median_relative_task_change"]
    )
    ranking_d = summary.sort_values("paper_method_score", ascending=False)
    ranking_a.to_csv(ROOT / "results" / "processed" / "development_ranking_A.csv", index=False)
    ranking_b.to_csv(ROOT / "results" / "processed" / "development_ranking_B_pareto.csv", index=False)
    ranking_c.to_csv(ROOT / "results" / "processed" / "development_ranking_C_predictive.csv", index=False)
    ranking_d.to_csv(ROOT / "results" / "processed" / "development_ranking_D_score.csv", index=False)

    validation_summary = summarize(units[units["split"] == "validation"].assign(split="test"))
    hybrid_validation = validation_summary[validation_summary["track"] == "hybrid_prediction_mixture"].copy()
    hybrid_validation["base_interface"] = hybrid_validation["method"].str.extract(
        r"^Raw\+(.+)@[0-9.]+$"
    )
    chosen_hybrids = []
    for base, frame in hybrid_validation.groupby("base_interface"):
        eligible = frame[frame["median_relative_task_change"] <= 0.01]
        pool = eligible if len(eligible) else frame
        chosen = pool.sort_values(
            ["median_disagreement_reduction", "paper_method_score"], ascending=[False, False]
        ).iloc[0]
        chosen_hybrids.append(chosen.to_dict())
    write_json(
        ROOT / "results" / "processed" / "development_analysis.json",
        {
            "complete": len(optimizer_files) == expected_optimizer
            and len(representation_files) == expected_representation,
            "optimizer_files": len(optimizer_files),
            "expected_optimizer_files": expected_optimizer,
            "representation_files": len(representation_files),
            "expected_representation_files": expected_representation,
            "selected_hybrids_from_validation_only": chosen_hybrids,
            "ranking_A": ranking_a.to_dict(orient="records"),
            "ranking_B": ranking_b.to_dict(orient="records"),
            "ranking_C": ranking_c.to_dict(orient="records"),
            "ranking_D": ranking_d.to_dict(orient="records"),
        },
    )
    print(summary.sort_values("paper_method_score", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
