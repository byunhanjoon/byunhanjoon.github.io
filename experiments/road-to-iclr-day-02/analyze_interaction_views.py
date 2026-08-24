"""Aggregate paired downstream results for the interaction-view benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "results" / "interaction_view_summary.csv",
    )
    args = parser.parse_args()

    raw = pd.concat([pd.read_csv(path) for path in args.inputs], ignore_index=True)
    key = ["dataset", "model", "seed", "representation"]
    raw = raw.drop_duplicates(key)
    baseline = raw[raw.representation.eq("baseline_ple")][
        ["dataset", "model", "seed", "test_score"]
    ].rename(columns={"test_score": "baseline_score"})
    paired = raw.merge(baseline, on=["dataset", "model", "seed"])
    paired["primary_gain"] = np.where(
        paired.task.eq("binclass"),
        100.0 * (paired.test_score - paired.baseline_score),
        100.0 * (paired.baseline_score - paired.test_score)
        / paired.baseline_score,
    )
    paired["win"] = paired.primary_gain > 0
    summary = (
        paired.groupby(
            ["dataset", "task", "model", "representation"], as_index=False
        )
        .agg(
            runs=("seed", "size"),
            test_mean=("test_score", "mean"),
            test_std=("test_score", lambda values: values.std(ddof=0)),
            primary_gain_mean=("primary_gain", "mean"),
            primary_gain_std=("primary_gain", lambda values: values.std(ddof=0)),
            wins=("win", "sum"),
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw.sort_values(key).to_csv(
        args.output.with_name("interaction_view_all.csv"), index=False
    )
    paired.sort_values(key).to_csv(
        args.output.with_name("interaction_view_paired.csv"), index=False
    )
    summary.to_csv(args.output, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
