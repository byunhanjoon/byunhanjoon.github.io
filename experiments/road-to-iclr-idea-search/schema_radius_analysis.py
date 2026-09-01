"""Worst-distribution schema variance via minimum-enclosing-ball duality.

For aligned prediction vectors x_1,...,x_K in a Hilbert space,

  sup_{w in simplex} sum_i w_i ||x_i - sum_j w_j x_j||^2

equals the squared radius of their minimum enclosing ball.  This script
compares that distribution-free radius with the declared uniform chart risk.
The identity is established convex duality, not a new theorem.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


HERE = Path(__file__).resolve().parent


def gram(points: np.ndarray) -> np.ndarray:
    flattened = points.reshape(points.shape[0], -1).astype(np.float64)
    # Inner product averages over evaluation rows, while summing output dims.
    row_count = points.shape[1]
    return flattened @ flattened.T / row_count


def weighted_variance(weights: np.ndarray, matrix: np.ndarray) -> float:
    return float(weights @ np.diag(matrix) - weights @ matrix @ weights)


def radius(points: np.ndarray) -> dict[str, object]:
    matrix = gram(points)
    count = len(points)
    uniform = np.full(count, 1.0 / count)
    result = minimize(
        lambda weights: -weighted_variance(weights, matrix),
        uniform,
        jac=lambda weights: -(np.diag(matrix) - 2.0 * matrix @ weights),
        bounds=[(0.0, 1.0)] * count,
        constraints={
            "type": "eq",
            "fun": lambda weights: float(weights.sum() - 1.0),
            "jac": lambda weights: np.ones_like(weights),
        },
        method="SLSQP",
        options={"ftol": 1e-14, "maxiter": 2_000},
    )
    if not result.success:
        raise RuntimeError(result.message)
    weights = np.clip(result.x, 0.0, 1.0)
    weights /= weights.sum()
    optimum = weighted_variance(weights, matrix)
    center = np.tensordot(weights, points, axes=(0, 0))
    member_distances = np.mean(
        np.sum((points - center[None]) ** 2, axis=-1), axis=1
    )
    uniform_value = weighted_variance(uniform, matrix)
    pair_distances = np.empty((count, count))
    for left in range(count):
        for right in range(count):
            pair_distances[left, right] = np.mean(
                np.sum((points[left] - points[right]) ** 2, axis=-1)
            )
    return {
        "uniform_schema_risk": uniform_value,
        "worst_distribution_schema_radius_squared": optimum,
        "radius_over_uniform_ratio": float(optimum / uniform_value),
        "max_pair_distance_squared": float(pair_distances.max()),
        "optimal_weights": weights.tolist(),
        "active_weight_count": int(np.sum(weights > 1e-7)),
        "minimum_enclosing_ball_max_distance_squared": float(
            member_distances.max()
        ),
        "duality_absolute_error": float(abs(member_distances.max() - optimum)),
        "member_distance_squared_to_robust_center": member_distances.tolist(),
    }


def summarize_archive(
    path: Path, prediction_key: str, name: str, charts: list[str] | None = None
) -> dict[str, object]:
    archive = np.load(path)
    predictions = archive[prediction_key].astype(np.float64)
    chart_names = (
        archive["charts"].astype(str).tolist() if charts is None else charts
    )
    persistent = radius(predictions.mean(axis=1))
    seed_midpoint = predictions.shape[1] // 2
    split_half_persistent = [
        radius(predictions[:, :seed_midpoint].mean(axis=1)),
        radius(predictions[:, seed_midpoint:].mean(axis=1)),
    ]
    conditional = [radius(predictions[:, seed]) for seed in range(predictions.shape[1])]
    fields = (
        "uniform_schema_risk",
        "worst_distribution_schema_radius_squared",
        "radius_over_uniform_ratio",
        "active_weight_count",
        "duality_absolute_error",
    )
    summary = {
        field: {
            "mean": float(np.mean([row[field] for row in conditional])),
            "minimum": float(np.min([row[field] for row in conditional])),
            "maximum": float(np.max([row[field] for row in conditional])),
        }
        for field in fields
    }
    weights = np.asarray([row["optimal_weights"] for row in conditional])
    summary["optimal_weight_by_chart"] = {
        chart: {
            "mean": float(weights[:, index].mean()),
            "standard_deviation": float(weights[:, index].std(ddof=1)),
        }
        for index, chart in enumerate(chart_names)
    }
    return {
        "name": name,
        "charts": chart_names,
        "persistent_mean_predictor": persistent,
        "split_half_persistent": split_half_persistent,
        "same_seed_conditional": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=HERE / "schema_radius_results.json"
    )
    args = parser.parse_args()
    analyses = [
        summarize_archive(
            HERE / "chart_orbit_adult_s32.npz",
            "mlp_predictions",
            "Adult MLP (32 seeds)",
        ),
        summarize_archive(
            HERE / "adult_architecture_chart.npz",
            "resnet_predictions",
            "Adult ResNet (16 seeds)",
        ),
        summarize_archive(
            HERE / "diamond_architecture_chart.npz",
            "mlp_predictions",
            "Diamond MLP (16 seeds)",
        ),
        summarize_archive(
            HERE / "diamond_architecture_chart.npz",
            "resnet_predictions",
            "Diamond ResNet (16 seeds)",
        ),
        summarize_archive(
            HERE / "chart_orbit_black_friday.npz",
            "adamw_predictions",
            "Black Friday MLP (16 seeds)",
        ),
    ]
    output = {
        "estimand": "supremum over all chart probability weights of aligned squared prediction variance",
        "warning": "the radius removes dependence on chart weights but can emphasize a small set of extreme representatives; uniform product measures remain necessary for factor ANOVA",
        "analyses": analyses,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
