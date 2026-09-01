"""Held-out pilot for row-adaptive approximation of a finite chart orbit.

The policy observes predictions from two charts.  Rows with the largest
two-chart disagreement are escalated to the complete five-chart centroid;
other rows retain the two-chart centroid.  Chart-pair selection uses disjoint
training seeds and rows.  Evaluation uses held-out seeds and rows, and the
threshold for each evaluation seed is calibrated without labels on the held-
out calibration rows.

This is an approximation experiment, not an accuracy-tuned TTA experiment:
neither pair selection nor escalation sees y_test.  The main comparator is a
row-independent escalation rule with exactly the same expected chart cost.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def squared_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sum((left - right) ** 2, axis=-1)


def threshold_from_fraction(scores: np.ndarray, fraction: float) -> float:
    if fraction <= 0:
        return np.inf
    if fraction >= 1:
        return -np.inf
    return float(np.quantile(scores, 1.0 - fraction, method="higher"))


def adaptive_residuals(
    predictions: np.ndarray,
    calibration_rows: np.ndarray,
    evaluation_rows: np.ndarray,
    pair: tuple[int, int],
    escalation_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return residual, escalation mask, and adaptive predictions by seed/row."""
    full = predictions.mean(axis=0)
    pair_mean = predictions[list(pair)].mean(axis=0)
    calibration_score = squared_distance(
        predictions[pair[0]][:, calibration_rows],
        predictions[pair[1]][:, calibration_rows],
    )
    evaluation_score = squared_distance(
        predictions[pair[0]][:, evaluation_rows],
        predictions[pair[1]][:, evaluation_rows],
    )
    masks = np.empty_like(evaluation_score, dtype=bool)
    for seed in range(predictions.shape[1]):
        threshold = threshold_from_fraction(
            calibration_score[seed], escalation_fraction
        )
        masks[seed] = evaluation_score[seed] >= threshold
    adaptive = pair_mean[:, evaluation_rows].copy()
    full_evaluation = full[:, evaluation_rows]
    adaptive[masks] = full_evaluation[masks]
    return squared_distance(adaptive, full_evaluation), masks, adaptive


def selection_score(
    predictions: np.ndarray,
    rows: np.ndarray,
    pair: tuple[int, int],
    fractions: tuple[float, ...],
) -> float:
    """In-sample development objective used only to freeze a chart pair."""
    full = predictions.mean(axis=0)
    pair_mean = predictions[list(pair)].mean(axis=0)
    residual = squared_distance(pair_mean[:, rows], full[:, rows])
    disagreement = squared_distance(
        predictions[pair[0]][:, rows], predictions[pair[1]][:, rows]
    )
    scores = []
    for fraction in fractions:
        kept = []
        for seed in range(predictions.shape[1]):
            threshold = threshold_from_fraction(disagreement[seed], fraction)
            kept.append(residual[seed][disagreement[seed] < threshold])
        scores.append(float(np.concatenate(kept).mean()))
    return float(np.mean(scores))


def two_way_bootstrap_interval(
    values: np.ndarray,
    rng: np.random.Generator,
    repetitions: int,
) -> list[float]:
    """Percentile interval resampling both training seeds and test rows."""
    n_seeds, n_rows = values.shape
    estimates = np.empty(repetitions)
    for repetition in range(repetitions):
        seed_sample = rng.integers(0, n_seeds, n_seeds)
        row_sample = rng.integers(0, n_rows, n_rows)
        estimates[repetition] = values[np.ix_(seed_sample, row_sample)].mean()
    return np.quantile(estimates, (0.025, 0.975)).tolist()


def proper_losses(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if predictions.shape[-1] == 1:
        return (predictions[..., 0] - labels[None]) ** 2
    targets = np.eye(predictions.shape[-1])[labels.astype(int)]
    return np.sum((predictions - targets[None]) ** 2, axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=HERE / "chart_orbit_adult_s32.npz"
    )
    parser.add_argument(
        "--output", type=Path, default=HERE / "orbit_cascade_adult.json"
    )
    parser.add_argument("--prediction-key", default="mlp_predictions")
    parser.add_argument("--bootstrap", type=int, default=1_000)
    args = parser.parse_args()

    archive = np.load(args.input)
    predictions = archive[args.prediction_key].astype(np.float64)
    charts = archive["charts"].astype(str)
    labels = archive["y_test"]
    if predictions.shape[0] != 5 or predictions.shape[1] < 8 or predictions.shape[1] % 2:
        raise ValueError("The design expects five charts and an even count of at least eight seeds")

    rng = np.random.default_rng(260_826)
    row_permutation = rng.permutation(predictions.shape[2])
    split = len(row_permutation) // 2
    calibration_rows = np.sort(row_permutation[:split])
    evaluation_rows = np.sort(row_permutation[split:])
    seed_midpoint = predictions.shape[1] // 2
    selection_seeds = np.arange(seed_midpoint)
    evaluation_seeds = np.arange(seed_midpoint, predictions.shape[1])
    budgets = (2.6, 3.0, 3.5, 4.0)
    fractions = tuple((budget - 2.0) / 3.0 for budget in budgets)

    pair_scores = {}
    for pair in itertools.combinations(range(5), 2):
        score = selection_score(
            predictions[:, selection_seeds], calibration_rows, pair, fractions
        )
        pair_scores["+".join(charts[list(pair)])] = score
    selected_pair = min(
        itertools.combinations(range(5), 2),
        key=lambda pair: pair_scores["+".join(charts[list(pair)])],
    )

    evaluation_predictions = predictions[:, evaluation_seeds]
    full_evaluation = evaluation_predictions.mean(axis=0)[:, evaluation_rows]
    pair_evaluation = evaluation_predictions[list(selected_pair)].mean(axis=0)[
        :, evaluation_rows
    ]
    base_residual = squared_distance(pair_evaluation, full_evaluation)
    y_evaluation = labels[evaluation_rows]
    results = []
    for budget, fraction in zip(budgets, fractions):
        residual, mask, adaptive = adaptive_residuals(
            evaluation_predictions,
            calibration_rows,
            evaluation_rows,
            selected_pair,
            fraction,
        )
        # If escalation is independent of the row, its expected residual is
        # exactly (1-q) times the two-chart residual.  Match each seed's
        # realized escalation rate, including quantile discreteness/shift.
        realized_fraction_by_seed = mask.mean(axis=1)
        random_expected = (
            (1.0 - realized_fraction_by_seed)[:, None] * base_residual
        )
        advantage = random_expected - residual
        full_loss = float(proper_losses(full_evaluation, y_evaluation).mean())
        adaptive_loss = float(proper_losses(adaptive, y_evaluation).mean())
        # Expected Brier of independently escalating each row with probability q.
        pair_losses = proper_losses(pair_evaluation, y_evaluation)
        full_losses = proper_losses(full_evaluation, y_evaluation)
        random_expected_loss = float(
            (
                (1.0 - realized_fraction_by_seed)[:, None] * pair_losses
                + realized_fraction_by_seed[:, None] * full_losses
            ).mean()
        )
        results.append(
            {
                "target_average_chart_passes": budget,
                "escalation_fraction": fraction,
                "realized_average_chart_passes": float(2.0 + 3.0 * mask.mean()),
                "adaptive_residual_to_full_centroid": float(residual.mean()),
                "equal_compute_random_expected_residual": float(
                    random_expected.mean()
                ),
                "relative_residual_reduction_vs_random": float(
                    1.0 - residual.mean() / random_expected.mean()
                ),
                "random_minus_adaptive_residual": float(advantage.mean()),
                "two_way_bootstrap_95_interval_for_advantage": (
                    two_way_bootstrap_interval(advantage, rng, args.bootstrap)
                ),
                "label_dependent_diagnostic_only": {
                    "metric": "MSE" if predictions.shape[-1] == 1 else "Brier",
                    "full_five_chart_loss": full_loss,
                    "adaptive_loss": adaptive_loss,
                    "equal_compute_random_expected_loss": random_expected_loss,
                },
            }
        )

    # Integer-cost static subsets are useful secondary references.  Each
    # subset is selected only on the disjoint development block.
    static_results = []
    for size in (2, 3, 4):
        candidates = list(itertools.combinations(range(5), size))
        development_full = predictions[:, selection_seeds].mean(axis=0)[
            :, calibration_rows
        ]
        selected = min(
            candidates,
            key=lambda subset: squared_distance(
                predictions[list(subset)][:, selection_seeds].mean(axis=0)[
                    :, calibration_rows
                ],
                development_full,
            ).mean(),
        )
        approximation = evaluation_predictions[list(selected)].mean(axis=0)[
            :, evaluation_rows
        ]
        static_results.append(
            {
                "chart_passes": size,
                "selected_charts": charts[list(selected)].tolist(),
                "held_out_residual_to_full_centroid": float(
                    squared_distance(approximation, full_evaluation).mean()
                ),
            }
        )

    conditional_row_risk = np.mean(
        squared_distance(
            predictions,
            predictions.mean(axis=0, keepdims=True),
        ),
        axis=(0, 1),
    )
    sorted_risk = np.sort(conditional_row_risk)[::-1]
    total_risk = sorted_risk.sum()
    concentration = {
        f"top_{percentage:g}_percent_share": float(
            sorted_risk[: max(1, round(len(sorted_risk) * percentage / 100))].sum()
            / total_risk
        )
        for percentage in (1, 5, 10, 20, 50)
    }

    output = {
        "design": {
            "objective": "label-free approximation of the five-chart centroid",
            "selection_seed_indices": selection_seeds.tolist(),
            "evaluation_seed_indices": evaluation_seeds.tolist(),
            "selection_seed_values": archive["seeds"][selection_seeds].tolist(),
            "evaluation_seed_values": archive["seeds"][evaluation_seeds].tolist(),
            "calibration_rows": int(len(calibration_rows)),
            "evaluation_rows": int(len(evaluation_rows)),
            "labels_used_for_policy": False,
            "main_comparator": "row-independent escalation from the same pair to all five charts at equal expected cost",
        },
        "selected_probe_charts": charts[list(selected_pair)].tolist(),
        "development_pair_scores": pair_scores,
        "conditional_chart_risk_concentration": concentration,
        "adaptive_results": results,
        "static_subset_results": static_results,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
