"""Exploratory claim-level audit for paired architecture comparisons.

Unlike prediction-space schema risk, this layer is label-dependent.  It asks
whether a paired MLP/ResNet comparison is supported across every declared
equivalent chart as well as after averaging over the chart quotient.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


HERE = Path(__file__).resolve().parent


def interval(values: np.ndarray) -> list[float]:
    return list(
        map(
            float,
            stats.t.interval(
                0.95,
                len(values) - 1,
                loc=float(values.mean()),
                scale=float(stats.sem(values)),
            ),
        )
    )


def squared_losses(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.mean((predictions[..., 0] - targets[None, None]) ** 2, axis=-1)


def brier_losses(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    one_hot = np.eye(predictions.shape[-1])[targets]
    return np.mean(
        np.sum((predictions - one_hot[None, None]) ** 2, axis=-1), axis=-1
    )


def audit(path: Path, dataset: str) -> dict[str, object]:
    archive = np.load(path)
    charts = archive["charts"].astype(str)
    mlp = archive["mlp_predictions"].astype(np.float64)
    resnet = archive["resnet_predictions"].astype(np.float64)
    targets = archive["y_test"]
    if dataset == "adult":
        mlp_loss = brier_losses(mlp, targets.astype(int))
        resnet_loss = brier_losses(resnet, targets.astype(int))
        metric = "Brier"
    else:
        mlp_loss = squared_losses(mlp, targets.astype(np.float64))
        resnet_loss = squared_losses(resnet, targets.astype(np.float64))
        metric = "standardized MSE"
    differences = mlp_loss - resnet_loss
    chart_rows = []
    for chart_index, chart in enumerate(charts):
        values = differences[chart_index]
        chart_rows.append(
            {
                "chart": chart,
                "mean_mlp_minus_resnet": float(values.mean()),
                "paired_seed_95_interval": interval(values),
                "fraction_seeds_mlp_better": float(np.mean(values < 0)),
            }
        )
    quotient_by_seed = differences.mean(axis=0)
    means = differences.mean(axis=1)
    intervals = np.asarray([row["paired_seed_95_interval"] for row in chart_rows])
    quotient_interval = interval(quotient_by_seed)
    if quotient_interval[1] < 0 and np.all(intervals[:, 1] < 0):
        status = "MLP advantage is schema-identifiable over this chart set"
    elif quotient_interval[0] > 0 and np.all(intervals[:, 0] > 0):
        status = "ResNet advantage is schema-identifiable over this chart set"
    elif np.min(means) < 0 < np.max(means):
        status = "point-estimate direction changes across equivalent charts"
    else:
        status = "detection/significance depends on the equivalent chart"
    return {
        "dataset": dataset,
        "metric": metric,
        "difference_orientation": "MLP minus ResNet; negative favors MLP",
        "quotient_mean_difference": float(quotient_by_seed.mean()),
        "quotient_paired_seed_95_interval": quotient_interval,
        "representative_mean_range": [float(means.min()), float(means.max())],
        "status": status,
        "by_chart": chart_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adult", type=Path, default=HERE / "adult_architecture_chart.npz"
    )
    parser.add_argument(
        "--diamond", type=Path, default=HERE / "diamond_architecture_chart.npz"
    )
    parser.add_argument(
        "--output", type=Path, default=HERE / "claim_identifiability.json"
    )
    args = parser.parse_args()
    output = {
        "warning": "exploratory paired t intervals; no multiplicity adjustment and no frozen practical-equivalence margin",
        "audits": [audit(args.adult, "adult"), audit(args.diamond, "diamond")],
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
