"""Bayes-matched falsification suite for semantic field metrics.

Each trial draws a latent state function from a Gaussian prior whose precision
encodes path, ring, or no topology.  Candidate ridge estimators receive the
same finite function space and differ only in their regularization metric.
Regularization strength is selected on an independent validation sample.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


HERE = Path(__file__).resolve().parent


def graph_laplacian(size: int, ring: bool) -> np.ndarray:
    laplacian = np.zeros((size, size))
    edges = [(index, index + 1) for index in range(size - 1)]
    if ring:
        edges.append((size - 1, 0))
    for left, right in edges:
        laplacian[left, left] += 1.0
        laplacian[right, right] += 1.0
        laplacian[left, right] -= 1.0
        laplacian[right, left] -= 1.0
    positive = np.linalg.eigvalsh(laplacian)
    positive = positive[positive > 1e-10]
    return laplacian / np.median(positive)


def precisions(size: int, topology_strength: float) -> dict[str, np.ndarray]:
    identity = np.eye(size)
    path = graph_laplacian(size, ring=False)
    ring = graph_laplacian(size, ring=True)
    permutation = np.random.default_rng(771_029).permutation(size)
    wrong_path = path[np.ix_(permutation, permutation)]
    return {
        "isotropic": identity,
        "path": identity + topology_strength * path,
        "ring": identity + topology_strength * ring,
        "permuted_path": identity + topology_strength * wrong_path,
    }


def draw_function(rng: np.random.Generator, precision: np.ndarray) -> np.ndarray:
    root = np.linalg.cholesky(precision)
    values = np.linalg.solve(root.T, rng.normal(size=len(precision)))
    return values - values.mean()


def trial(
    seed: int,
    true_precision: np.ndarray,
    candidates: dict[str, np.ndarray],
    strengths: np.ndarray,
    train_size: int,
    validation_size: int,
    noise: float,
) -> tuple[dict[str, float], dict[str, float]]:
    rng = np.random.default_rng(seed)
    size = len(true_precision)
    truth = draw_function(rng, true_precision)
    train_state = rng.integers(size, size=train_size)
    validation_state = rng.integers(size, size=validation_size)
    train_y = truth[train_state] + rng.normal(scale=noise, size=train_size)
    validation_y = truth[validation_state] + rng.normal(
        scale=noise, size=validation_size
    )
    counts = np.bincount(train_state, minlength=size).astype(float) / train_size
    rhs = np.bincount(train_state, weights=train_y, minlength=size) / train_size
    risks = {}
    selected_strengths = {}
    for name, metric in candidates.items():
        fits = []
        for strength in strengths:
            coefficient = np.linalg.solve(
                np.diag(counts) + float(strength) * metric,
                rhs,
            )
            validation_mse = float(
                np.mean((coefficient[validation_state] - validation_y) ** 2)
            )
            fits.append((validation_mse, float(strength), coefficient))
        _, selected_strength, coefficient = min(fits, key=lambda item: item[0])
        risks[name] = float(np.mean((coefficient - truth) ** 2))
        selected_strengths[name] = selected_strength
    return risks, selected_strengths


def bootstrap_interval(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(5_000, len(values)))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def summarize(
    risks: list[dict[str, float]],
    selected_strengths: list[dict[str, float]],
    matched: str,
    seed: int,
) -> dict[str, object]:
    names = list(risks[0])
    matrix = np.asarray([[row[name] for name in names] for row in risks])
    strengths = np.asarray(
        [[row[name] for name in names] for row in selected_strengths]
    )
    winners = np.argmin(matrix, axis=1)
    output: dict[str, object] = {
        "matched_metric": matched,
        "trials": len(risks),
        "metrics": {},
        "paired_matched_contrasts": {},
    }
    for index, name in enumerate(names):
        output["metrics"][name] = {
            "mean_test_mse": float(matrix[:, index].mean()),
            "median_test_mse": float(np.median(matrix[:, index])),
            "win_rate": float(np.mean(winners == index)),
            "median_selected_strength": float(np.median(strengths[:, index])),
        }
    matched_index = names.index(matched)
    for index, name in enumerate(names):
        if name == matched:
            continue
        # Negative means the Bayes-matched metric is better.
        difference = matrix[:, matched_index] - matrix[:, index]
        nonzero = difference[difference != 0]
        wins = int(np.sum(nonzero < 0))
        output["paired_matched_contrasts"][f"matched_minus_{name}"] = {
            "mean_test_mse_difference": float(difference.mean()),
            "mean_difference_ci95": bootstrap_interval(
                difference, seed + 97 * index
            ),
            "matched_win_fraction_excluding_ties": (
                wins / len(nonzero) if len(nonzero) else None
            ),
            "two_sided_sign_test_p": (
                float(binomtest(wins, len(nonzero), 0.5).pvalue)
                if len(nonzero)
                else None
            ),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=24)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--output", type=Path, default=HERE / "field_topology_bayes_suite.json")
    args = parser.parse_args()
    candidate_metrics = precisions(args.states, topology_strength=4.0)
    matched_names = {"ordinal": "path", "cyclic": "ring", "nominal": "isotropic"}
    strengths = np.logspace(-4, 0, 25)
    scenarios = (
        (48, 96, 0.5),
        (96, 144, 0.5),
        (96, 144, 1.0),
    )
    output: dict[str, object] = {
        "status": "frozen_synthetic_semantic_metric_falsification",
        "states": args.states,
        "trials_per_cell": args.trials,
        "candidate_topology_strength": 4.0,
        "true_topology_strengths": [1.0, 4.0, 16.0],
        "candidate_metrics": list(candidate_metrics),
        "scenarios": {},
    }
    scenario_index = 0
    for true_strength in (1.0, 4.0, 16.0):
        true_family = precisions(args.states, topology_strength=true_strength)
        true_metrics = {
            "ordinal": true_family["path"],
            "cyclic": true_family["ring"],
            "nominal": true_family["isotropic"],
        }
        for train_size, validation_size, noise in scenarios:
            scenario_name = (
                f"true{true_strength:g}_n{train_size}_v{validation_size}_noise{noise:g}"
            )
            output["scenarios"][scenario_name] = {}
            for task_index, (task, true_precision) in enumerate(true_metrics.items()):
                risks = []
                selected_strengths = []
                for replicate in range(args.trials):
                    risk, chosen = trial(
                        seed=10_000_000
                        + 100_000 * scenario_index
                        + 10_000 * task_index
                        + replicate,
                        true_precision=true_precision,
                        candidates=candidate_metrics,
                        strengths=strengths,
                        train_size=train_size,
                        validation_size=validation_size,
                        noise=noise,
                    )
                    risks.append(risk)
                    selected_strengths.append(chosen)
                output["scenarios"][scenario_name][task] = summarize(
                    risks,
                    selected_strengths,
                    matched=matched_names[task],
                    seed=88_000 + 100 * scenario_index + task_index,
                )
            scenario_index += 1
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
