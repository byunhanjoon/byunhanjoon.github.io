"""Monte Carlo convergence study for the OrbitANOVA pick-freeze estimator."""

from __future__ import annotations

import json

import numpy as np

from orbit_anova import decompose, pick_freeze
from pick_freeze_validation import FACTOR_NAMES, LEVELS, predictor


def exact_effects() -> tuple[dict[str, float], dict[str, float], float]:
    coordinates = np.asarray(
        np.meshgrid(*(np.arange(level) for level in LEVELS), indexing="ij")
    ).reshape(len(LEVELS), -1).T
    predictions = predictor(coordinates).reshape(LEVELS + (7, 2))
    decomposition = decompose(predictions, FACTOR_NAMES)
    first = {name: decomposition[name] for name in FACTOR_NAMES}
    total = {
        factor: sum(
            value
            for component, value in decomposition.items()
            if component
            not in {
                "total",
                "component_sum_error",
                "prediction_reconstruction_max_error",
            }
            and factor in component.split(":")
        )
        for factor in FACTOR_NAMES
    }
    return first, total, decomposition["total"]


def one_estimate(sample_count: int, rng: np.random.Generator) -> dict[str, object]:
    a = np.column_stack([rng.integers(level, size=sample_count) for level in LEVELS])
    b = np.column_stack([rng.integers(level, size=sample_count) for level in LEVELS])
    hybrids = []
    for factor_index in range(len(LEVELS)):
        coordinates = a.copy()
        coordinates[:, factor_index] = b[:, factor_index]
        hybrids.append(predictor(coordinates))
    return pick_freeze(predictor(a), predictor(b), np.asarray(hybrids), FACTOR_NAMES)


def main() -> None:
    exact_first, exact_total_effect, exact_variance = exact_effects()
    repetitions = 400
    output = {
        "exact_variance": exact_variance,
        "exact_first_order": exact_first,
        "exact_total_effect": exact_total_effect,
        "repetitions": repetitions,
        "sample_sizes": {},
    }
    exact_winner = max(exact_total_effect, key=exact_total_effect.get)
    for sample_count in (8, 16, 32, 64, 128, 256, 512):
        rng = np.random.default_rng(91_000 + sample_count)
        estimates = [one_estimate(sample_count, rng) for _ in range(repetitions)]
        variance_values = np.asarray([estimate["total"] for estimate in estimates])
        summary: dict[str, object] = {
            "pipeline_evaluations_for_five_factors": 7 * sample_count,
            "variance_relative_bias": float(
                (variance_values.mean() - exact_variance) / exact_variance
            ),
            "variance_relative_rmse": float(
                np.sqrt(np.mean((variance_values - exact_variance) ** 2))
                / exact_variance
            ),
            "effects": {},
        }
        winners = []
        for estimate in estimates:
            winners.append(
                max(
                    FACTOR_NAMES,
                    key=lambda name: estimate["effects"][name]["total_effect"],
                )
            )
        summary["correct_top_total_effect_fraction"] = float(
            np.mean(np.asarray(winners) == exact_winner)
        )
        best_effect = exact_total_effect[exact_winner]
        selection_regrets = np.asarray(
            [
                (best_effect - exact_total_effect[winner]) / best_effect
                for winner in winners
            ]
        )
        summary["mean_relative_selection_regret"] = float(selection_regrets.mean())
        summary["p95_relative_selection_regret"] = float(
            np.quantile(selection_regrets, 0.95)
        )
        for factor in FACTOR_NAMES:
            first_values = np.asarray(
                [estimate["effects"][factor]["first_order"] for estimate in estimates]
            )
            total_values = np.asarray(
                [estimate["effects"][factor]["total_effect"] for estimate in estimates]
            )
            summary["effects"][factor] = {
                "first_relative_bias": float(
                    (first_values.mean() - exact_first[factor]) / exact_first[factor]
                ),
                "first_relative_rmse": float(
                    np.sqrt(np.mean((first_values - exact_first[factor]) ** 2))
                    / exact_first[factor]
                ),
                "total_relative_bias": float(
                    (total_values.mean() - exact_total_effect[factor])
                    / exact_total_effect[factor]
                ),
                "total_relative_rmse": float(
                    np.sqrt(
                        np.mean((total_values - exact_total_effect[factor]) ** 2)
                    )
                    / exact_total_effect[factor]
                ),
            }
        output["sample_sizes"][str(sample_count)] = summary
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
