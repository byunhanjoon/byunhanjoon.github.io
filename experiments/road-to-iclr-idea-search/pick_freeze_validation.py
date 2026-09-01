"""Validate sampled vector-output Sobol estimates against exact OrbitANOVA."""

from __future__ import annotations

import json

import numpy as np

from orbit_anova import (
    decompose,
    log_orbit_summary,
    pick_freeze,
    symmetrization_frontier,
)


LEVELS = (3, 4, 2)
FACTOR_NAMES = ("feature", "category", "class")


def predictor(coordinates: np.ndarray) -> np.ndarray:
    """A deterministic vector predictor with known interactions."""
    feature, category, class_id = coordinates.T
    row = np.arange(7)[None, :]
    first_logit = (
        0.25 * feature[:, None] * np.cos(0.4 * row)
        + 0.18 * category[:, None] * np.sin(0.7 * row)
        + 0.12 * class_id[:, None]
        + 0.11 * feature[:, None] * category[:, None] * np.cos(0.2 * row)
        + 0.09 * category[:, None] * class_id[:, None] * np.sin(0.3 * row)
    )
    probability = 1.0 / (1.0 + np.exp(-first_logit))
    return np.stack((probability, 1.0 - probability), axis=-1)


def main() -> None:
    grid_coordinates = np.asarray(
        np.meshgrid(*(np.arange(level) for level in LEVELS), indexing="ij")
    ).reshape(len(LEVELS), -1).T
    exact_predictions = predictor(grid_coordinates).reshape(LEVELS + (7, 2))
    exact = decompose(exact_predictions, FACTOR_NAMES)

    sample_count = 100_000
    rng = np.random.default_rng(20_260_826)
    a_coordinates = np.column_stack(
        [rng.integers(level, size=sample_count) for level in LEVELS]
    )
    b_coordinates = np.column_stack(
        [rng.integers(level, size=sample_count) for level in LEVELS]
    )
    hybrids = []
    for factor_index in range(len(LEVELS)):
        coordinates = a_coordinates.copy()
        coordinates[:, factor_index] = b_coordinates[:, factor_index]
        hybrids.append(predictor(coordinates))
    sampled = pick_freeze(
        predictor(a_coordinates),
        predictor(b_coordinates),
        np.asarray(hybrids),
        FACTOR_NAMES,
    )

    comparison = {}
    for factor_name in FACTOR_NAMES:
        exact_first = exact[factor_name]
        exact_total = sum(
            value
            for component, value in exact.items()
            if component not in {
                "total",
                "component_sum_error",
                "prediction_reconstruction_max_error",
            }
            and factor_name in component.split(":")
        )
        estimate = sampled["effects"][factor_name]
        comparison[factor_name] = {
            "exact_first_order": exact_first,
            "sampled_first_order": estimate["first_order"],
            "exact_total_effect": exact_total,
            "sampled_total_effect": estimate["total_effect"],
        }
    log_identity = log_orbit_summary(
        exact_predictions,
        len(FACTOR_NAMES),
        np.arange(7) % 2,
    )
    frontier = symmetrization_frontier(exact_predictions, FACTOR_NAMES)
    component_names = {
        name
        for name in exact
        if name
        not in {
            "total",
            "component_sum_error",
            "prediction_reconstruction_max_error",
        }
    }
    maximum_coverage_error = 0.0
    for item in frontier:
        selected = set(item["factors"])
        expected_residual = sum(
            exact[name]
            for name in component_names
            if selected.isdisjoint(name.split(":"))
        )
        maximum_coverage_error = max(
            maximum_coverage_error,
            abs(item["residual_risk"] - expected_residual),
        )
    print(
        json.dumps(
            {
                "exact_total": exact["total"],
                "sampled": sampled,
                "comparison": comparison,
                "log_identity": log_identity,
                "symmetrization_coverage_max_error": maximum_coverage_error,
                "symmetrization_frontier": frontier,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
