"""Aggregate the paired Day 2 cross-dataset/backbone experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent


def _column_stability(values: pd.Series) -> str:
    def normalize(value: object) -> str:
        if pd.isna(value) or str(value) == "":
            return ""
        return ";".join(
            str(int(float(part))) if float(part).is_integer() else part
            for part in str(value).split(";")
        )

    counts = values.map(normalize).value_counts().sort_index()
    return "; ".join(f"{key or '-'}:{count}" for key, count in counts.items())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=[
            HERE / "results" / "cross_dataset_mlp_resnet.csv",
            HERE / "results" / "cross_dataset_tabm.csv",
            HERE / "results" / "cross_dataset_california_mlp_resnet.csv",
            HERE / "results" / "cross_dataset_blackfriday_resnet_seed2.csv",
        ],
    )
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()

    frames = [pd.read_csv(path) for path in args.inputs if path.exists()]
    if not frames:
        raise FileNotFoundError("No cross-dataset result CSVs found")
    raw = pd.concat(frames, ignore_index=True)
    key = ["dataset", "model", "seed", "representation"]
    duplicates = raw.duplicated(key, keep=False)
    if duplicates.any():
        duplicate_rows = raw.loc[duplicates]
        for _, group in duplicate_rows.groupby(key):
            if not np.allclose(group.test_score, group.test_score.iloc[0]):
                raise ValueError(f"Conflicting duplicate runs:\n{group[key + ['test_score']]}")
        # Parallel shards may overlap at a completed boundary. Exact repeated
        # runs are safe to collapse; conflicting scores remain an error above.
        raw = raw.drop_duplicates(key, keep="first")
    raw = raw.sort_values(key).reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output_dir / "cross_dataset_all.csv", index=False)

    baseline = raw[raw.representation == "baseline_ple"][
        ["dataset", "model", "seed", "test_score", "test_loss"]
    ].rename(
        columns={"test_score": "baseline_score", "test_loss": "baseline_loss"}
    )
    paired = raw.merge(baseline, on=["dataset", "model", "seed"], how="left")
    paired["primary_gain"] = np.where(
        paired.task == "binclass",
        100.0 * (paired.test_score - paired.baseline_score),
        100.0 * (paired.baseline_score - paired.test_score) / paired.baseline_score,
    )
    paired["loss_gain_pct"] = (
        100.0 * (paired.baseline_loss - paired.test_loss) / paired.baseline_loss
    )
    paired["win"] = paired.primary_gain > 0.0
    paired.to_csv(args.output_dir / "cross_dataset_paired.csv", index=False)

    summary_rows: list[dict[str, object]] = []
    group_columns = ["dataset", "task", "model", "representation"]
    for group_key, group in paired.groupby(group_columns, sort=True):
        dataset, task, model, representation = group_key
        summary_rows.append(
            {
                "dataset": dataset,
                "task": task,
                "model": model,
                "representation": representation,
                "runs": len(group),
                "test_mean": group.test_score.mean(),
                "test_std": group.test_score.std(ddof=0),
                "primary_gain_mean": group.primary_gain.mean(),
                "primary_gain_std": group.primary_gain.std(ddof=0),
                "loss_gain_pct_mean": group.loss_gain_pct.mean(),
                "loss_gain_pct_std": group.loss_gain_pct.std(ddof=0),
                "wins": int(group.win.sum()),
                "mean_parameters": group.parameters.mean(),
                "mean_seconds": group.train_seconds.mean(),
                "selected_columns": _column_stability(group.selected_numeric),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "cross_dataset_summary.csv", index=False)

    selector = (
        raw.sort_values(["dataset", "seed"])
        .drop_duplicates(["dataset", "seed"])[
            [
                "dataset",
                "seed",
                "variance_columns",
                "utility_columns",
                "utility_statistics",
                "top8_values",
            ]
        ]
        .reset_index(drop=True)
    )
    selector.to_csv(args.output_dir / "selector_diagnostics.csv", index=False)

    nonbaseline = summary[summary.representation != "baseline_ple"]
    utility = nonbaseline[nonbaseline.representation == "utility_identity"]
    aggregate = (
        nonbaseline.groupby("representation")
        .agg(
            cells=("primary_gain_mean", "size"),
            positive_cells=("primary_gain_mean", lambda x: int((x > 0).sum())),
            mean_loss_gain_pct=("loss_gain_pct_mean", "mean"),
            median_loss_gain_pct=("loss_gain_pct_mean", "median"),
        )
        .reset_index()
    )
    report = {
        "run_count": int(len(raw)),
        "datasets": sorted(raw.dataset.unique().tolist()),
        "models": sorted(raw.model.unique().tolist()),
        "seeds": sorted(int(seed) for seed in raw.seed.unique()),
        "representations": sorted(raw.representation.unique().tolist()),
        "completed_cells": int(
            raw[["dataset", "model", "representation"]].drop_duplicates().shape[0]
        ),
        "expected_cells": int(
            raw.dataset.nunique() * raw.model.nunique() * raw.representation.nunique()
        ),
        "aggregate": aggregate.to_dict(orient="records"),
        "utility_identity": utility.to_dict(orient="records"),
    }
    (args.output_dir / "cross_dataset_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(f"runs={len(raw)} cells={report['completed_cells']}/{report['expected_cells']}")
    if not aggregate.empty:
        print(aggregate.to_string(index=False))
    if not utility.empty:
        columns = [
            "dataset",
            "model",
            "test_mean",
            "primary_gain_mean",
            "loss_gain_pct_mean",
            "wins",
            "selected_columns",
        ]
        print("\nUtility-selected identity")
        print(utility[columns].to_string(index=False))


if __name__ == "__main__":
    main()
