"""Synthetic falsification test for intrinsic field topology.

This is deliberately a small classical regression experiment.  It asks whether
an interval derivative metric helps a smooth numerical field but hurts an
unordered categorical field.  The result is diagnostic for the FieldRiesz
thesis; it is not a neural-network benchmark.
"""

from __future__ import annotations

import json

import numpy as np


def hat_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    spacing = knots[1] - knots[0]
    return np.maximum(1.0 - np.abs(x[:, None] - knots[None, :]) / spacing, 0.0)


def forms(knots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(knots[0], knots[-1], 20_001)
    phi = hat_basis(grid, knots)
    mass = np.trapz(phi[:, :, None] * phi[:, None, :], grid, axis=0)
    spacing = knots[1] - knots[0]
    stiffness = np.zeros((len(knots), len(knots)))
    for index in range(len(knots) - 1):
        stiffness[index, index] += 1.0 / spacing
        stiffness[index + 1, index + 1] += 1.0 / spacing
        stiffness[index, index + 1] -= 1.0 / spacing
        stiffness[index + 1, index] -= 1.0 / spacing
    return mass, stiffness


def wrong_topology(stiffness: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    size = len(stiffness)
    constant = np.ones(size) / np.sqrt(size)
    full, _ = np.linalg.qr(np.column_stack((constant, rng.normal(size=(size, size - 1)))))
    rotation, _ = np.linalg.qr(rng.normal(size=(size - 1, size - 1)))
    block = np.eye(size)
    block[1:, 1:] = rotation
    chart_rotation = full @ block @ full.T
    return chart_rotation.T @ stiffness @ chart_rotation


def normalize_stiffness(mass: np.ndarray, stiffness: np.ndarray) -> np.ndarray:
    root = np.linalg.cholesky(mass)
    whitened = np.linalg.solve(root, stiffness) @ np.linalg.inv(root.T)
    eigen = np.linalg.eigvalsh((whitened + whitened.T) / 2.0)
    positive = eigen[eigen > eigen.max() * 1e-10]
    return stiffness / np.median(positive)


def fit(
    x: np.ndarray,
    y: np.ndarray,
    knots: np.ndarray,
    metric: np.ndarray,
    strength: float,
) -> np.ndarray:
    design = hat_basis(x, knots)
    system = design.T @ design / len(x) + strength * metric
    rhs = design.T @ y / len(x)
    return np.linalg.solve(system, rhs)


def smooth_target(x: np.ndarray) -> np.ndarray:
    return np.sin(2.0 * np.pi * x) + 0.35 * np.cos(4.0 * np.pi * x)


def numerical_trial(
    seed: int,
    knots: np.ndarray,
    metrics: dict[str, np.ndarray],
    strengths: np.ndarray,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    train_x = rng.uniform(size=96)
    val_x = rng.uniform(size=144)
    test_x = np.linspace(0.0, 1.0, 4001)
    train_y = smooth_target(train_x) + rng.normal(scale=0.25, size=len(train_x))
    val_y = smooth_target(val_x) + rng.normal(scale=0.25, size=len(val_x))
    test_y = smooth_target(test_x)
    val_design = hat_basis(val_x, knots)
    test_design = hat_basis(test_x, knots)
    results = {}
    for name, metric in metrics.items():
        candidates = []
        for strength in strengths:
            coefficient = fit(train_x, train_y, knots, metric, float(strength))
            val_mse = float(np.mean((val_design @ coefficient - val_y) ** 2))
            candidates.append((val_mse, coefficient))
        _, selected = min(candidates, key=lambda item: item[0])
        results[name] = float(np.mean((test_design @ selected - test_y) ** 2))
    return results


def categorical_trial(
    seed: int,
    metrics: dict[str, np.ndarray],
    strengths: np.ndarray,
) -> dict[str, float]:
    """Fit unrelated category effects; an interval graph is false semantics."""
    rng = np.random.default_rng(seed)
    size = next(iter(metrics.values())).shape[0]
    category_effect = np.random.default_rng(202_608_26).normal(size=size)
    category_effect -= category_effect.mean()
    train_category = rng.integers(size, size=8 * size)
    val_category = rng.integers(size, size=12 * size)
    train_design = np.eye(size)[train_category]
    val_design = np.eye(size)[val_category]
    test_design = np.eye(size)
    train_y = category_effect[train_category] + rng.normal(scale=0.25, size=len(train_category))
    val_y = category_effect[val_category] + rng.normal(scale=0.25, size=len(val_category))
    results = {}
    for name, metric in metrics.items():
        candidates = []
        for strength in strengths:
            system = train_design.T @ train_design / len(train_design) + strength * metric
            rhs = train_design.T @ train_y / len(train_design)
            coefficient = np.linalg.solve(system, rhs)
            val_mse = float(np.mean((val_design @ coefficient - val_y) ** 2))
            candidates.append((val_mse, coefficient))
        _, selected = min(candidates, key=lambda item: item[0])
        results[name] = float(np.mean((test_design @ selected - category_effect) ** 2))
    return results


def summarize(values: list[dict[str, float]]) -> dict[str, object]:
    names = list(values[0])
    matrix = np.asarray([[row[name] for name in names] for row in values])
    best = np.argmin(matrix, axis=1)
    return {
        name: {
            "mean_test_mse": float(matrix[:, index].mean()),
            "median_test_mse": float(np.median(matrix[:, index])),
            "win_rate": float(np.mean(best == index)),
        }
        for index, name in enumerate(names)
    }


def main() -> None:
    knots = np.linspace(0.0, 1.0, 24)
    mass, stiffness = forms(knots)
    stiffness = normalize_stiffness(mass, stiffness)
    wrong = normalize_stiffness(mass, wrong_topology(stiffness, seed=991))
    metrics = {
        "mass_only": mass,
        "interval_topology": mass + 0.08 * stiffness,
        "wrong_topology": mass + 0.08 * wrong,
    }
    strengths = np.logspace(-5, 0, 25)
    output: dict[str, object] = {"validation_selected": {}, "fixed_strength": {}}
    output["validation_selected"]["smooth_numerical"] = summarize([
        numerical_trial(50_000 + seed, knots, metrics, strengths) for seed in range(200)
    ])
    output["validation_selected"]["unordered_categorical"] = summarize([
        categorical_trial(60_000 + seed, metrics, strengths) for seed in range(200)
    ])
    for strength in (1e-3, 1e-2, 1e-1):
        output["fixed_strength"][str(strength)] = {
            "smooth_numerical": summarize([
                numerical_trial(70_000 + seed, knots, metrics, np.asarray([strength]))
                for seed in range(100)
            ]),
            "unordered_categorical": summarize([
                categorical_trial(80_000 + seed, metrics, np.asarray([strength]))
                for seed in range(100)
            ]),
        }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
