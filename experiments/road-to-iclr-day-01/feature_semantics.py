"""Synthetic probe for Road to ICLR, Day 1.

The experiment isolates representation choice with a linear ridge readout. It is
not intended as a neural-model benchmark; it asks how much work can be done by
the feature encoding before a backbone sees the table.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CARDINALITIES = (16, 16, 24)
MODEL_NAMES = ("schema_only", "semantic_oracle", "multi_view")


@dataclass
class FeatureState:
    mean: float
    scale: float
    knots: np.ndarray
    cardinality: int


def make_dataset(n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    num_as_cat = rng.integers(0, CARDINALITIES[0], size=n_samples)
    cat_as_num = rng.integers(0, CARDINALITIES[1], size=n_samples)
    hybrid = rng.integers(0, CARDINALITIES[2], size=n_samples)

    num_as_cat_effect = (
        1.15 * np.sin(1.7 * num_as_cat)
        + 0.45 * np.cos(0.7 * np.power(num_as_cat + 1, 1.2))
    )
    cat_as_num_effect = 1.8 * (cat_as_num / (CARDINALITIES[1] - 1) - 0.5)
    hybrid_effect = (
        0.9 * (hybrid / (CARDINALITIES[2] - 1) - 0.5)
        + 0.7 * np.sin(2 * np.pi * hybrid / 6)
    )
    interaction = 0.55 * ((num_as_cat % 4) == (hybrid % 4))

    target = (
        num_as_cat_effect
        + cat_as_num_effect
        + hybrid_effect
        + interaction
        + rng.normal(0, 0.65, size=n_samples)
    )
    features = np.column_stack((num_as_cat, cat_as_num, hybrid)).astype(float)
    return features, target


def fit_state(values: np.ndarray, cardinality: int, bins: int = 8) -> FeatureState:
    quantiles = np.linspace(0, 1, bins + 1)
    knots = np.unique(np.quantile(values, quantiles))
    if len(knots) < 2:
        knots = np.array([values.min(), values.min() + 1.0])
    scale = float(values.std()) or 1.0
    return FeatureState(float(values.mean()), scale, knots, cardinality)


def raw_view(values: np.ndarray, state: FeatureState) -> np.ndarray:
    return ((values - state.mean) / state.scale)[:, None]


def one_hot_view(values: np.ndarray, state: FeatureState) -> np.ndarray:
    encoded = np.zeros((len(values), state.cardinality), dtype=float)
    indices = np.clip(values.astype(int), 0, state.cardinality - 1)
    encoded[np.arange(len(values)), indices] = 1.0
    return encoded


def ple_view(values: np.ndarray, state: FeatureState) -> np.ndarray:
    """Piecewise-linear encoding using training-set quantile knots."""

    knots = state.knots
    widths = np.maximum(np.diff(knots), 1e-12)
    output = np.zeros((len(values), len(widths)), dtype=float)

    for column, (left, right, width) in enumerate(zip(knots[:-1], knots[1:], widths)):
        output[:, column] = np.clip((values - left) / width, 0.0, 1.0)
        output[values >= right, column] = 1.0
    return output


def encode(
    features: np.ndarray,
    states: list[FeatureState],
    model_name: str,
) -> np.ndarray:
    columns: list[np.ndarray] = []

    for index, state in enumerate(states):
        values = features[:, index]
        if model_name == "schema_only":
            # The declared schema says numerical, categorical, numerical.
            columns.append(one_hot_view(values, state) if index == 1 else raw_view(values, state))
        elif model_name == "semantic_oracle":
            if index == 0:
                columns.append(one_hot_view(values, state))
            elif index == 1:
                columns.append(raw_view(values, state))
            else:
                columns.extend(
                    (raw_view(values, state), ple_view(values, state), one_hot_view(values, state))
                )
        elif model_name == "multi_view":
            columns.extend(
                (raw_view(values, state), ple_view(values, state), one_hot_view(values, state))
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

    return np.column_stack(columns)


def ridge_predict(
    train_features: np.ndarray,
    train_target: np.ndarray,
    test_features: np.ndarray,
    alpha: float = 2.0,
) -> np.ndarray:
    train_design = np.column_stack((np.ones(len(train_features)), train_features))
    test_design = np.column_stack((np.ones(len(test_features)), test_features))
    penalty = np.eye(train_design.shape[1]) * np.sqrt(alpha)
    penalty[0, 0] = 0.0
    augmented_design = np.vstack((train_design, penalty))
    augmented_target = np.concatenate((train_target, np.zeros(train_design.shape[1])))
    weights, *_ = np.linalg.lstsq(augmented_design, augmented_target, rcond=None)
    prediction = np.einsum("ij,j->i", test_design, weights)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Ridge prediction produced a non-finite value")
    return prediction


def run(sample_sizes: list[int], seeds: int, test_size: int) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []

    for sample_size in sample_sizes:
        scores = {name: [] for name in MODEL_NAMES}
        for seed in range(seeds):
            train_x, train_y = make_dataset(sample_size, seed)
            test_x, test_y = make_dataset(test_size, 10_000 + seed)
            states = [
                fit_state(train_x[:, index], cardinality)
                for index, cardinality in enumerate(CARDINALITIES)
            ]

            for model_name in MODEL_NAMES:
                encoded_train = encode(train_x, states, model_name)
                encoded_test = encode(test_x, states, model_name)
                prediction = ridge_predict(encoded_train, train_y, encoded_test)
                scores[model_name].append(float(np.sqrt(np.mean((prediction - test_y) ** 2))))

        for model_name in MODEL_NAMES:
            values = np.asarray(scores[model_name])
            rows.append(
                {
                    "train_rows": sample_size,
                    "model": model_name,
                    "rmse_mean": float(values.mean()),
                    "rmse_std": float(values.std(ddof=1)),
                }
            )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=[200, 500, 1000, 2500])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--test-size", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results.csv"))
    args = parser.parse_args()

    rows = run(args.sample_sizes, args.seeds, args.test_size)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"n={row['train_rows']:>4}  {row['model']:<16}  "
            f"RMSE={row['rmse_mean']:.3f} ± {row['rmse_std']:.3f}"
        )


if __name__ == "__main__":
    main()
