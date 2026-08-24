"""Aggregate the frequency-shrunk residual-map benchmark."""

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
        "--output", type=Path, default=HERE / "results" / "residual_map_summary.csv"
    )
    args = parser.parse_args()
    raw = pd.concat([pd.read_csv(path) for path in args.inputs], ignore_index=True)
    key = ["dataset", "model", "seed", "representation"]
    duplicates = raw.duplicated(key, keep=False)
    if duplicates.any():
        for _, group in raw.loc[duplicates].groupby(key):
            if not np.allclose(group.test_score, group.test_score.iloc[0]):
                raise ValueError(f"Conflicting duplicate runs:\n{group[key + ['test_score']]}")
        raw = raw.drop_duplicates(key)
    baseline = raw[raw.representation == "baseline_ple"][
        ["dataset", "model", "seed", "test_score", "test_loss"]
    ].rename(columns={"test_score": "baseline_score", "test_loss": "baseline_loss"})
    paired = raw.merge(baseline, on=["dataset", "model", "seed"])
    paired["primary_gain"] = np.where(
        paired.task == "binclass",
        100.0 * (paired.test_score - paired.baseline_score),
        100.0 * (paired.baseline_score - paired.test_score) / paired.baseline_score,
    )
    paired["loss_gain_pct"] = 100.0 * (
        paired.baseline_loss - paired.test_loss
    ) / paired.baseline_loss
    paired["win"] = paired.primary_gain > 0.0
    summary = (
        paired.groupby(["dataset", "task", "model", "representation"], as_index=False)
        .agg(
            runs=("seed", "size"),
            test_mean=("test_score", "mean"),
            test_std=("test_score", lambda x: x.std(ddof=0)),
            primary_gain_mean=("primary_gain", "mean"),
            primary_gain_std=("primary_gain", lambda x: x.std(ddof=0)),
            loss_gain_pct_mean=("loss_gain_pct", "mean"),
            wins=("win", "sum"),
            reused=("reused_baseline", "sum"),
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw.sort_values(key).to_csv(args.output.with_name("residual_map_all.csv"), index=False)
    paired.sort_values(key).to_csv(
        args.output.with_name("residual_map_paired.csv"), index=False
    )
    summary.to_csv(args.output, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
