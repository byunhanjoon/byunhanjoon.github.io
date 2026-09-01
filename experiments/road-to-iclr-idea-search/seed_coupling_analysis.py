"""Many-seed diagnostics for randomized schema representatives.

This separates persistent differences between mean predictors from a declared
same-seed coupling, and treats small-sample optimal transport as a diagnostic
requiring within-representative null calibration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


HERE = Path(__file__).resolve().parent


def prediction_cost(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.mean(
        np.sum((left[:, None] - right[None, :]) ** 2, axis=-1), axis=-1
    )


def assignment_cost(cost: np.ndarray) -> float:
    rows, columns = linear_sum_assignment(cost)
    return float(cost[rows, columns].mean())


def energy_test(
    left: np.ndarray,
    right: np.ndarray,
    rng: np.random.Generator,
    permutations: int,
) -> dict[str, float]:
    if len(left) != len(right):
        raise ValueError("Energy-test samples must be balanced")
    combined = np.concatenate((left, right))
    squared = prediction_cost(combined, combined)
    distances = np.sqrt(np.maximum(squared, 0.0))
    sample_size = len(left)
    off_diagonal = ~np.eye(sample_size, dtype=bool)

    def statistic(indices: np.ndarray) -> float:
        mask = np.zeros(2 * sample_size, dtype=bool)
        mask[indices] = True
        first = np.flatnonzero(mask)
        second = np.flatnonzero(~mask)
        return float(
            2.0 * distances[np.ix_(first, second)].mean()
            - distances[np.ix_(first, first)][off_diagonal].mean()
            - distances[np.ix_(second, second)][off_diagonal].mean()
        )

    observed = statistic(np.arange(sample_size))
    null = np.asarray(
        [
            statistic(rng.choice(2 * sample_size, sample_size, replace=False))
            for _ in range(permutations)
        ]
    )
    return {
        "energy_statistic": observed,
        "permutation_p": float(
            (1 + np.sum(null >= observed)) / (permutations + 1)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=HERE / "chart_orbit_adult_s32.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "chart_seed_coupling_s32.json",
    )
    parser.add_argument("--prediction-key", default="mlp_predictions")
    parser.add_argument("--permutations", type=int, default=3_000)
    parser.add_argument("--transport-splits", type=int, default=500)
    args = parser.parse_args()

    archive = np.load(args.input)
    predictions = archive[args.prediction_key].astype(np.float64)
    chart_names = [str(value) for value in archive["charts"]]
    chart_count, seed_count, row_count = predictions.shape[:3]
    if seed_count % 2:
        raise ValueError("An even seed count is required for split diagnostics")
    rng = np.random.default_rng(260_826)

    pair_grams: dict[tuple[int, int], np.ndarray] = {}
    pair_rows = []
    for left_index in range(chart_count):
        for right_index in range(left_index + 1, chart_count):
            differences = (
                predictions[left_index] - predictions[right_index]
            ).reshape(seed_count, -1)
            gram = differences @ differences.T / row_count
            pair_grams[left_index, right_index] = gram
            plug_in = float(gram.mean())
            unbiased = float(
                (gram.sum() - np.trace(gram))
                / (seed_count * (seed_count - 1))
            )
            signs = rng.choice(
                (-1.0, 1.0), size=(args.permutations, seed_count)
            )
            null = np.einsum("bi,ij,bj->b", signs, gram, signs) / seed_count**2
            pair_rows.append(
                {
                    "left": chart_names[left_index],
                    "right": chart_names[right_index],
                    "mean_difference_squared_plugin": plug_in,
                    "mean_difference_squared_unbiased": unbiased,
                    "paired_sign_flip_p_exploratory": float(
                        (1 + np.sum(null >= plug_in))
                        / (args.permutations + 1)
                    ),
                }
            )

    def persistent_mean_estimate(counts: np.ndarray) -> float:
        sample_size = int(counts.sum())
        total = 0.0
        for gram in pair_grams.values():
            total += float(
                counts @ gram @ counts - counts @ np.diag(gram)
            ) / (sample_size * (sample_size - 1))
        return total / chart_count**2

    all_counts = np.ones(seed_count)
    persistent = persistent_mean_estimate(all_counts)
    jackknife = np.asarray(
        [
            persistent_mean_estimate(
                all_counts - np.eye(seed_count, dtype=float)[index]
            )
            for index in range(seed_count)
        ]
    )
    jackknife_se = float(
        np.sqrt(
            (seed_count - 1)
            / seed_count
            * np.sum((jackknife - jackknife.mean()) ** 2)
        )
    )

    mean_over_charts = predictions.mean(axis=0, keepdims=True)
    same_seed_coupling = float(
        np.mean(np.sum((predictions - mean_over_charts) ** 2, axis=-1))
    )

    half = seed_count // 2
    energy_rows = []
    for left_index in range(chart_count):
        for right_index in range(left_index + 1, chart_count):
            forward = energy_test(
                predictions[left_index, :half],
                predictions[right_index, half:],
                rng,
                args.permutations,
            )
            reverse = energy_test(
                predictions[left_index, half:],
                predictions[right_index, :half],
                rng,
                args.permutations,
            )
            energy_rows.append(
                {
                    "left": chart_names[left_index],
                    "right": chart_names[right_index],
                    "disjoint_seed_forward": forward,
                    "disjoint_seed_reverse": reverse,
                }
            )

    costs = {}
    for left_index in range(chart_count):
        for right_index in range(left_index, chart_count):
            cost = prediction_cost(
                predictions[left_index], predictions[right_index]
            )
            costs[left_index, right_index] = cost
            costs[right_index, left_index] = cost.T
    full_pair_transport = sum(
        assignment_cost(costs[left_index, right_index])
        for left_index in range(chart_count)
        for right_index in range(left_index + 1, chart_count)
    ) / chart_count**2

    excess_transport = []
    for _ in range(args.transport_splits):
        first = np.sort(rng.choice(seed_count, half, replace=False))
        second = np.setdiff1d(np.arange(seed_count), first)
        within = [
            assignment_cost(costs[index, index][np.ix_(first, second)])
            for index in range(chart_count)
        ]
        excess = 0.0
        for left_index in range(chart_count):
            for right_index in range(left_index + 1, chart_count):
                cross = np.mean(
                    [
                        assignment_cost(
                            costs[left_index, right_index][np.ix_(first, second)]
                        ),
                        assignment_cost(
                            costs[left_index, right_index][np.ix_(second, first)]
                        ),
                    ]
                )
                excess += cross - 0.5 * (
                    within[left_index] + within[right_index]
                )
        excess_transport.append(excess / chart_count**2)

    output = {
        "design": {
            "charts": chart_names,
            "seeds": seed_count,
            "rows": row_count,
            "warning": "same-seed dispersion is coupling dependent; empirical OT is upward biased",
        },
        "persistent_mean_schema_risk": {
            "unbiased_u_statistic": persistent,
            "jackknife_standard_error": jackknife_se,
            "normal_95_interval": [
                persistent - 1.96 * jackknife_se,
                persistent + 1.96 * jackknife_se,
            ],
        },
        "same_seed_coupled_schema_risk": same_seed_coupling,
        "mean_pair_diagnostics": pair_rows,
        "disjoint_seed_energy_tests": energy_rows,
        "transport_diagnostic": {
            "naive_full_empirical_pair_lower_bound": full_pair_transport,
            "split_null_corrected_excess_median": float(
                np.median(excess_transport)
            ),
            "split_null_corrected_excess_95_interval": np.quantile(
                excess_transport, (0.025, 0.975)
            ).tolist(),
            "fraction_positive_across_splits": float(
                np.mean(np.asarray(excess_transport) > 0)
            ),
        },
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
