"""Probe exact-value pair interactions left after numerical PLE.

For each pair of low-cardinality numerical columns, fit a smoothed lookup table
to numerical-only PLE out-of-fold residuals. A second set of folds evaluates the
lookup table on held-out training rows. This is a diagnostic only: it never uses
validation or test targets, and it does not train a downstream neural model.
"""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

import cross_dataset_models as experiment
from residual_map_benchmark import subsample_dataset


HERE = Path(__file__).resolve().parent


def row_keys(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values)
    return contiguous.view(
        np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))
    ).ravel()


def mapped_correction(
    fit_values: np.ndarray,
    fit_residual: np.ndarray,
    query_values: np.ndarray,
    smoothing: float,
) -> tuple[np.ndarray, int]:
    fit_keys = row_keys(fit_values)
    query_keys = row_keys(query_values)
    keys, inverse, counts = np.unique(
        fit_keys, return_inverse=True, return_counts=True
    )
    means = np.bincount(inverse, weights=fit_residual) / (counts + smoothing)
    positions = np.searchsorted(keys, query_keys)
    valid = positions < len(keys)
    matched = np.zeros(len(query_keys), dtype=bool)
    matched[valid] = keys[positions[valid]] == query_keys[valid]
    output = np.zeros(len(query_keys), dtype=np.float64)
    output[matched] = means[positions[matched]]
    return output, len(keys)


def probe(
    dataset: experiment.benchmark.Dataset,
    seed: int,
    bins: int,
    max_cardinality: int,
    smoothing: float,
    cache: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    cache = {} if cache is None else cache
    experiment.benchmark.encode_dataset(
        dataset,
        "diagnostic_identity",
        seed,
        bins,
        max_cardinality,
        smoothing,
        1e-3,
        cache,
    )
    assert dataset.x_num is not None
    numeric = experiment.benchmark._clean_numeric(dataset.x_num)["train"]
    target = dataset.y["train"].astype(np.float64)
    if dataset.task == "regression":
        target = (target - target.mean()) / max(float(target.std()), 1e-12)
    residual = np.asarray(cache["numeric_residual"], dtype=np.float64)
    base_prediction = target - residual
    candidates = experiment._candidate_columns(numeric, max_cardinality)
    splitter = (
        StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 20_000)
        if dataset.task == "binclass"
        else KFold(n_splits=5, shuffle=True, random_state=seed + 20_000)
    )
    folds = list(
        splitter.split(numeric, target if dataset.task == "binclass" else None)
    )
    baseline_loss = experiment._loss(dataset.task, base_prediction, target)
    rows: list[dict[str, object]] = []
    for left, right in combinations(candidates, 2):
        correction = np.zeros(len(target), dtype=np.float64)
        fold_wins = 0
        cardinalities: list[int] = []
        for fit_index, holdout_index in folds:
            fold_correction, cardinality = mapped_correction(
                numeric[fit_index][:, (left, right)],
                residual[fit_index],
                numeric[holdout_index][:, (left, right)],
                smoothing,
            )
            correction[holdout_index] = fold_correction
            cardinalities.append(cardinality)
            before = experiment._loss(
                dataset.task,
                base_prediction[holdout_index],
                target[holdout_index],
            )
            after = experiment._loss(
                dataset.task,
                base_prediction[holdout_index] + fold_correction,
                target[holdout_index],
            )
            fold_wins += int(after < before)
        corrected_loss = experiment._loss(
            dataset.task, base_prediction + correction, target
        )
        rows.append(
            {
                "dataset": dataset.name,
                "task": dataset.task,
                "seed": seed,
                "bins": bins,
                "left_column": left,
                "right_column": right,
                "left_cardinality": len(np.unique(numeric[:, left])),
                "right_cardinality": len(np.unique(numeric[:, right])),
                "pair_cardinality": len(
                    np.unique(row_keys(numeric[:, (left, right)]))
                ),
                "mean_fit_cardinality": float(np.mean(cardinalities)),
                "baseline_loss": baseline_loss,
                "corrected_loss": corrected_loss,
                "relative_gain": (baseline_loss - corrected_loss)
                / max(baseline_loss, 1e-12),
                "fold_wins": fold_wins,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=experiment.DAY1 / "data")
    parser.add_argument(
        "--datasets", nargs="+", choices=experiment.DATASETS, default=experiment.DATASETS
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--bins", type=int, default=128)
    parser.add_argument("--max-cardinality", type=int, default=128)
    parser.add_argument("--smoothing", type=float, default=20.0)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument(
        "--output", type=Path, default=HERE / "results" / "interaction_utility.csv"
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for dataset_name in args.datasets:
        dataset = experiment.benchmark.load_dataset(args.data, dataset_name)
        dataset = subsample_dataset(
            dataset, args.max_train_rows, None, args.sample_seed
        )
        for seed in args.seeds:
            seed_rows = probe(
                dataset,
                seed,
                args.bins,
                args.max_cardinality,
                args.smoothing,
            )
            rows.extend(seed_rows)
            if seed_rows:
                best = max(seed_rows, key=lambda row: float(row["relative_gain"]))
                print(
                    f"{dataset_name} seed={seed}: best pair "
                    f"{best['left_column']};{best['right_column']}, "
                    f"gain={100 * float(best['relative_gain']):+.4f}%, "
                    f"wins={best['fold_wins']}/5",
                    flush=True,
                )
            else:
                print(f"{dataset_name} seed={seed}: no eligible pairs", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
