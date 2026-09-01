"""Can validation calibrate field stiffness without inventing false topology?"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from field_topology_bayes_suite import (
    bootstrap_interval,
    draw_function,
    graph_laplacian,
    precisions,
)


HERE = Path(__file__).resolve().parent


def metric_families(size: int) -> dict[str, list[tuple[float, np.ndarray]]]:
    identity = np.eye(size)
    path = graph_laplacian(size, ring=False)
    ring = graph_laplacian(size, ring=True)
    permutation = np.random.default_rng(771_029).permutation(size)
    permuted_path = path[np.ix_(permutation, permutation)]
    alphas = (0.0, 1.0, 4.0, 16.0)
    return {
        "isotropic": [(0.0, identity)],
        "path_tuned": [(alpha, identity + alpha * path) for alpha in alphas],
        "ring_tuned": [(alpha, identity + alpha * ring) for alpha in alphas],
        "permuted_path_tuned": [
            (alpha, identity + alpha * permuted_path) for alpha in alphas
        ],
    }


def trial(
    seed: int,
    true_precision: np.ndarray,
    families: dict[str, list[tuple[float, np.ndarray]]],
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
    chosen_alphas = {}
    for family, metrics in families.items():
        fits = []
        for alpha, metric in metrics:
            for strength in strengths:
                coefficient = np.linalg.solve(
                    np.diag(counts) + float(strength) * metric, rhs
                )
                validation_mse = float(
                    np.mean((coefficient[validation_state] - validation_y) ** 2)
                )
                fits.append((validation_mse, alpha, coefficient))
        _, alpha, coefficient = min(fits, key=lambda item: item[0])
        risks[family] = float(np.mean((coefficient - truth) ** 2))
        chosen_alphas[family] = alpha
    return risks, chosen_alphas


def summarize(
    risks: list[dict[str, float]],
    alphas: list[dict[str, float]],
    matched: str,
    seed: int,
) -> dict[str, object]:
    names = list(risks[0])
    matrix = np.asarray([[row[name] for name in names] for row in risks])
    result: dict[str, object] = {"matched_family": matched, "families": {}, "contrasts": {}}
    for index, name in enumerate(names):
        chosen = np.asarray([row[name] for row in alphas])
        result["families"][name] = {
            "mean_test_mse": float(matrix[:, index].mean()),
            "median_test_mse": float(np.median(matrix[:, index])),
            "zero_stiffness_selection_rate": float(np.mean(chosen == 0)),
            "median_selected_stiffness": float(np.median(chosen)),
        }
    matched_index = names.index(matched)
    for index, name in enumerate(names):
        if name == matched:
            continue
        difference = matrix[:, matched_index] - matrix[:, index]
        result["contrasts"][f"matched_minus_{name}"] = {
            "mean_test_mse_difference": float(difference.mean()),
            "mean_difference_ci95": bootstrap_interval(difference, seed + 31 * index),
            "matched_win_rate": float(np.mean(difference < 0)),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=24)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "field_topology_strength_selection.json",
    )
    args = parser.parse_args()
    families = metric_families(args.states)
    strengths = np.logspace(-4, 0, 25)
    scenarios = ((48, 96, 0.5), (96, 144, 0.5), (96, 144, 1.0))
    matched = {
        "ordinal": "path_tuned",
        "cyclic": "ring_tuned",
        "nominal": "isotropic",
    }
    output: dict[str, object] = {
        "status": "post_frozen_calibration_extension",
        "stiffness_grid": [0.0, 1.0, 4.0, 16.0],
        "warning": "topology families search four times as many metric candidates as isotropic",
        "scenarios": {},
    }
    scenario_index = 0
    for true_strength in (1.0, 4.0, 16.0):
        true_family = precisions(args.states, true_strength)
        true_metrics = {
            "ordinal": true_family["path"],
            "cyclic": true_family["ring"],
            "nominal": true_family["isotropic"],
        }
        for train_size, validation_size, noise in scenarios:
            name = f"true{true_strength:g}_n{train_size}_v{validation_size}_noise{noise:g}"
            output["scenarios"][name] = {}
            for task_index, (task, true_precision) in enumerate(true_metrics.items()):
                risks = []
                alphas = []
                for replicate in range(args.trials):
                    risk, chosen = trial(
                        30_000_000 + 100_000 * scenario_index + 10_000 * task_index + replicate,
                        true_precision,
                        families,
                        strengths,
                        train_size,
                        validation_size,
                        noise,
                    )
                    risks.append(risk)
                    alphas.append(chosen)
                output["scenarios"][name][task] = summarize(
                    risks, alphas, matched[task], 190_000 + 100 * scenario_index + task_index
                )
            scenario_index += 1
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
