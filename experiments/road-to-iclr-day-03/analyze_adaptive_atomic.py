"""Paired analysis, matched controls, and uncertainty for adaptive atomic views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent


def bootstrap_mean(values: np.ndarray, seed: int = 2027) -> tuple[float, float]:
    if not len(values):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(10_000, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]).tolist())


def diagnostic_gain(row: pd.Series) -> float:
    scores = json.loads(row.diagnostics)
    selected_text = row.selection
    singleton_text, pair_text = selected_text.split("|")
    singletons = {
        int(value)
        for value in singleton_text.removeprefix("singletons=").split(";")
        if value
    }
    pairs = {
        tuple(map(int, value.split("+")))
        for value in pair_text.removeprefix("pairs=").split(";")
        if value
    }
    gain = 0.0
    for score in scores:
        columns = tuple(score["columns"])
        if score["kind"] == "singleton" and columns[0] in singletons:
            gain += float(score["relative_gain"])
        if score["kind"] == "pair" and columns in pairs:
            gain += float(score["incremental_gain"])
    return gain


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=HERE / "results" / "adaptive_analysis"
    )
    args = parser.parse_args()

    raw = pd.concat([pd.read_csv(path) for path in args.inputs], ignore_index=True)
    if "parameter_count" not in raw.columns and "parameters" in raw.columns:
        raw["parameter_count"] = raw["parameters"]
    split_columns = ["fold"] if "fold" in raw.columns else []
    run_key = ["dataset", *split_columns, "model", "seed"]
    key = [*run_key, "representation"]
    if "test_error" not in raw.columns:
        raw["test_error"] = np.where(
            raw.task.eq("binclass"), 1.0 - raw.test_auc, raw.test_score
        )
    else:
        missing_error = raw.test_error.isna()
        raw.loc[missing_error, "test_error"] = np.where(
            raw.loc[missing_error, "task"].eq("binclass"),
            1.0 - raw.loc[missing_error, "test_auc"],
            raw.loc[missing_error, "test_score"],
        )
    duplicates = raw.duplicated(key, keep=False)
    if duplicates.any():
        for _, group in raw.loc[duplicates].groupby(key):
            if not np.allclose(group.test_error, group.test_error.iloc[0]):
                raise ValueError(f"Conflicting duplicate runs:\n{group[key + ['test_error']]}")
        raw = raw.drop_duplicates(key)
    baseline = raw[raw.representation.eq("baseline_ple")][
        [*run_key, "test_error", "parameter_count"]
    ].rename(
        columns={
            "test_error": "baseline_error",
            "parameter_count": "baseline_parameter_count",
        }
    )
    paired = raw.merge(baseline, on=run_key)
    paired["primary_gain"] = np.where(
        paired.task.eq("binclass"),
        100.0 * (paired.baseline_error - paired.test_error),
        100.0 * (paired.baseline_error - paired.test_error) / paired.baseline_error,
    )
    paired["win"] = paired.primary_gain > 0
    paired["parameter_ratio"] = (
        paired.parameter_count / paired.baseline_parameter_count
    )
    summary = (
        paired.groupby(
            ["dataset", "task", "model", "representation"], as_index=False
        )
        .agg(
            runs=("seed", "size"),
            test_error_mean=("test_error", "mean"),
            primary_gain_mean=("primary_gain", "mean"),
            primary_gain_std=("primary_gain", lambda values: values.std(ddof=0)),
            wins=("win", "sum"),
            parameter_ratio_mean=("parameter_ratio", "mean"),
        )
    )

    random = paired[paired.representation.str.startswith("matched_random")]
    random_mean = (
        random.groupby(run_key, as_index=False)
        .agg(random_gain=("primary_gain", "mean"), random_variants=("representation", "size"))
    )
    selected = paired[paired.representation.eq("adaptive_atomic")].copy()
    selected["diagnostic_gain"] = selected.apply(diagnostic_gain, axis=1)
    selected = selected.merge(random_mean, on=run_key, how="left")
    selected["gain_over_random"] = selected.primary_gain - selected.random_gain

    dataset_selected = (
        selected.groupby("dataset", as_index=False)
        .agg(
            adaptive_gain=("primary_gain", "mean"),
            random_gain=("random_gain", "mean"),
            gain_over_random=("gain_over_random", "mean"),
            diagnostic_gain=("diagnostic_gain", "mean"),
        )
    )
    interval = bootstrap_mean(dataset_selected.adaptive_gain.to_numpy())
    random_interval = bootstrap_mean(dataset_selected.gain_over_random.dropna().to_numpy())
    correlation = (
        float(dataset_selected[["diagnostic_gain", "adaptive_gain"]].corr().iloc[0, 1])
        if len(dataset_selected) >= 3
        else float("nan")
    )
    report = {
        "logical_rows": len(raw),
        "datasets": sorted(raw.dataset.unique().tolist()),
        "models": sorted(raw.model.unique().tolist()),
        "adaptive_dataset_mean_gain": float(dataset_selected.adaptive_gain.mean()),
        "adaptive_dataset_bootstrap_95ci": interval,
        "gain_over_matched_random_mean": float(dataset_selected.gain_over_random.mean()),
        "gain_over_matched_random_bootstrap_95ci": random_interval,
        "diagnostic_to_test_gain_correlation": correlation,
        "maximum_parameter_ratio_deviation": float(
            np.max(np.abs(paired.parameter_ratio - 1.0))
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw.sort_values(key).to_csv(args.output_dir / "all.csv", index=False)
    paired.sort_values(key).to_csv(args.output_dir / "paired.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    selected.to_csv(args.output_dir / "selected_vs_random.csv", index=False)
    dataset_selected.to_csv(args.output_dir / "dataset_summary.csv", index=False)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(summary.to_string(index=False))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
