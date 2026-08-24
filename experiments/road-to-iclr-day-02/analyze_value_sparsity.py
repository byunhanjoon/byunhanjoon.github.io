"""Aggregate Adult value-sparsity runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=HERE / "results" / "adult_value_sparsity_summary.csv"
    )
    args = parser.parse_args()
    raw = pd.concat([pd.read_csv(path) for path in args.inputs], ignore_index=True)
    key = ["model", "seed", "representation"]
    raw = raw.drop_duplicates(key).sort_values(key)
    baseline = raw[raw.representation == "baseline_ple"][["model", "seed", "test_score"]]
    baseline = baseline.rename(columns={"test_score": "baseline_score"})
    paired = raw.merge(baseline, on=["model", "seed"], how="left")
    paired["accuracy_gain_points"] = 100.0 * (
        paired.test_score - paired.baseline_score
    )
    summary = (
        paired.groupby(["model", "representation", "top_k"], as_index=False)
        .agg(
            accuracy_mean=("test_score", "mean"),
            accuracy_std=("test_score", lambda x: x.std(ddof=0)),
            gain_mean=("accuracy_gain_points", "mean"),
            gain_std=("accuracy_gain_points", lambda x: x.std(ddof=0)),
            wins=("accuracy_gain_points", lambda x: int((x > 0).sum())),
        )
    )
    order = {"baseline_ple": 0, "top1": 1, "top2": 2, "top4": 3, "top8": 4,
             "top16": 5, "top32": 6, "top64": 7, "full": 8}
    summary["order"] = summary.representation.map(order)
    summary = summary.sort_values(["model", "order"]).drop(columns="order")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
