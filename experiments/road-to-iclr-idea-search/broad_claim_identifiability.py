"""Audit MLP-vs-ResNet claim identification across two equivalent bases.

This reuses the frozen Day-3 natural-encoding grid.  It is a scalar,
label-dependent multiverse analysis rather than a prediction-space OrbitANOVA
decomposition, because per-row predictions were not retained.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    HERE.parent
    / "road-to-iclr-day-03"
    / "results/day3/broad_benchmark/final_natural_encoding_pairs.csv"
)


def percentile_interval(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    draws = rng.choice(values, size=(20_000, len(values)), replace=True).mean(axis=1)
    return np.quantile(draws, [0.025, 0.975]).tolist()


def analyze(path: Path) -> dict[str, object]:
    data = pd.read_csv(path)
    required = {
        "dataset",
        "task",
        "model",
        "seed",
        "cumulative_helmert",
        "local_adjacent",
        "scale",
    }
    if not required.issubset(data.columns):
        raise ValueError(f"missing columns: {sorted(required - set(data.columns))}")
    if set(data["model"]) != {"mlp", "resnet"}:
        raise ValueError("expected exactly the MLP and ResNet arms")

    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(20_260_826)
    for dataset, group in data.groupby("dataset", sort=True):
        task = str(group["task"].iloc[0])
        scale = float(group["scale"].iloc[0])
        model_rows = {model: frame.set_index("seed") for model, frame in group.groupby("model")}
        seeds = sorted(set(model_rows["mlp"].index) & set(model_rows["resnet"].index))
        if not seeds:
            raise ValueError(f"no paired seeds for {dataset}")
        chart_seed_differences = {}
        for chart in ("cumulative_helmert", "local_adjacent"):
            # test_primary is higher-is-better for every task in the source
            # benchmark (negative RMSE for regression).
            chart_seed_differences[chart] = np.asarray(
                [
                    (
                        model_rows["mlp"].loc[seed, chart]
                        - model_rows["resnet"].loc[seed, chart]
                    )
                    / scale
                    for seed in seeds
                ],
                dtype=float,
            )
        chart_means = {
            chart: float(values.mean()) for chart, values in chart_seed_differences.items()
        }
        quotient_seed = np.mean(list(chart_seed_differences.values()), axis=0)
        rows.append(
            {
                "dataset": dataset,
                "task": task,
                "paired_seed_count": len(seeds),
                "cumulative_mlp_minus_resnet_normalized": chart_means["cumulative_helmert"],
                "local_mlp_minus_resnet_normalized": chart_means["local_adjacent"],
                "quotient_mlp_minus_resnet_normalized": float(quotient_seed.mean()),
                "representative_min": min(chart_means.values()),
                "representative_max": max(chart_means.values()),
                "representative_span": abs(
                    chart_means["cumulative_helmert"] - chart_means["local_adjacent"]
                ),
                "point_direction_changes": bool(
                    chart_means["cumulative_helmert"]
                    * chart_means["local_adjacent"]
                    < 0
                ),
                "cumulative_seed_bootstrap_95": percentile_interval(
                    chart_seed_differences["cumulative_helmert"], rng
                ),
                "local_seed_bootstrap_95": percentile_interval(
                    chart_seed_differences["local_adjacent"], rng
                ),
                "quotient_seed_bootstrap_95": percentile_interval(quotient_seed, rng),
            }
        )

    thresholds = (0.0, 0.001, 0.005, 0.01)
    threshold_summaries = {}
    for tau in thresholds:
        identifiable = [
            row
            for row in rows
            if row["representative_min"] > tau or row["representative_max"] < -tau
        ]
        unresolved = [row for row in rows if row not in identifiable]
        crosses_rope = [
            row
            for row in rows
            if row["representative_min"] < -tau
            and row["representative_max"] > tau
        ]
        within_rope = [
            row
            for row in rows
            if row["representative_min"] >= -tau
            and row["representative_max"] <= tau
        ]
        threshold_summaries[str(tau)] = {
            "schema_identifiable_count": len(identifiable),
            "unresolved_count": len(unresolved),
            "representative_range_crosses_rope_count": len(crosses_rope),
            "all_representatives_within_rope_count": len(within_rope),
            "schema_identifiable_datasets": [row["dataset"] for row in identifiable],
            "unresolved_datasets": [row["dataset"] for row in unresolved],
            "representative_range_crosses_rope_datasets": [
                row["dataset"] for row in crosses_rope
            ],
            "all_representatives_within_rope_datasets": [
                row["dataset"] for row in within_rope
            ],
        }

    spans = np.asarray([row["representative_span"] for row in rows])
    change_count = sum(row["point_direction_changes"] for row in rows)
    count = len(rows)
    z = 1.959963984540054
    center = (change_count / count + z**2 / (2 * count)) / (1 + z**2 / count)
    half_width = z * math.sqrt(
        (change_count / count * (1 - change_count / count) + z**2 / (4 * count))
        / count
    ) / (1 + z**2 / count)
    opposite_intervals = []
    for row in rows:
        cumulative = row["cumulative_seed_bootstrap_95"]
        local = row["local_seed_bootstrap_95"]
        if (cumulative[0] > 0 and local[1] < 0) or (
            cumulative[1] < 0 and local[0] > 0
        ):
            opposite_intervals.append(row["dataset"])
    return {
        "source": str(path),
        "estimand": "paired-seed MLP minus ResNet test-primary difference, normalized by the Day-3 task scale; positive favors MLP",
        "warning": "scalar label-dependent claim audit only; the archive does not retain row predictions and cannot estimate schema-representation risk",
        "dataset_count": len(rows),
        "point_direction_change_count": change_count,
        "point_direction_change_fraction": change_count / count,
        "point_direction_change_wilson_95": [center - half_width, center + half_width],
        "point_direction_change_datasets": [
            row["dataset"] for row in rows if row["point_direction_changes"]
        ],
        "opposite_nonzero_seed_bootstrap_intervals": opposite_intervals,
        "representative_span": {
            "median": float(np.median(spans)),
            "mean": float(spans.mean()),
            "p90": float(np.quantile(spans, 0.9)),
            "maximum": float(spans.max()),
        },
        "practical_threshold_summaries": threshold_summaries,
        "datasets": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output", type=Path, default=HERE / "broad_claim_identifiability.json"
    )
    args = parser.parse_args()
    result = analyze(args.input)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
