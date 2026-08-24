"""Aggregate Adult frequency-stratified generalization metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "results" / "adult_frequency_generalization_summary.csv",
    )
    args = parser.parse_args()
    raw = pd.concat([pd.read_csv(path) for path in args.inputs], ignore_index=True)
    key = ["model", "seed", "profile", "bucket"]
    baseline = raw[raw.representation == "baseline_ple"][key + ["accuracy", "logloss"]]
    baseline = baseline.rename(
        columns={"accuracy": "baseline_accuracy", "logloss": "baseline_logloss"}
    )
    identity = raw[raw.representation == "utility_identity"].merge(baseline, on=key)
    identity["accuracy_gain_points"] = 100.0 * (
        identity.accuracy - identity.baseline_accuracy
    )
    identity["logloss_gain_pct"] = 100.0 * (
        identity.baseline_logloss - identity.logloss
    ) / identity.baseline_logloss
    summary = (
        identity.groupby(["model", "profile", "bucket"], as_index=False)
        .agg(
            rows=("rows", "first"),
            accuracy_gain_mean=("accuracy_gain_points", "mean"),
            accuracy_gain_std=("accuracy_gain_points", lambda x: x.std(ddof=0)),
            logloss_gain_pct_mean=("logloss_gain_pct", "mean"),
            logloss_gain_pct_std=("logloss_gain_pct", lambda x: x.std(ddof=0)),
        )
    )
    bucket_order = {"unseen": 0, "1-9": 1, "10-99": 2, "100-999": 3, "1000+": 4}
    summary["order"] = summary.bucket.map(bucket_order)
    summary = summary.sort_values(["profile", "model", "order"]).drop(columns="order")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
