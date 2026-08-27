"""Leakage-safe diagnostic for higher-cardinality repeated numerical states.

Day 2 limited exact-state candidates to at most 128 levels.  This diagnostic
keeps its two-stage cross-fitting protocol but separates residual construction
from identity construction, so columns with thousands of levels can be tested
without materializing a huge one-hot matrix.

It also records distribution-only support statistics.  In particular, raw
repeat frequency is not called "atom mass": rounded continuous measurements
can repeat often.  ``local_excess_mass`` compares each ordered level count to a
local median and is only a rough structural-spike diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from sklearn.model_selection import KFold, StratifiedKFold


HERE = Path(__file__).resolve().parent
DAY1 = HERE.parent / "road-to-iclr-day-01"
DAY2 = HERE.parent / "road-to-iclr-day-02"
sys.path.insert(0, str(DAY1))
sys.path.insert(0, str(DAY2))

import real_data_benchmark as benchmark  # noqa: E402
import cross_dataset_models as day2  # noqa: E402


def support_statistics(column: np.ndarray) -> dict[str, float | int]:
    values, counts = np.unique(column, return_counts=True)
    n = len(column)
    repeated = counts >= 2
    if len(counts) >= 5:
        local = median_filter(counts.astype(np.float64), size=5, mode="nearest")
    else:
        local = np.full(len(counts), np.median(counts), dtype=np.float64)
    # Leave a count of one as the irreducible empirical observation.  This is a
    # descriptive spike statistic, not a consistent latent-atom estimator.
    excess = np.maximum(counts - np.maximum(local, 1.0), 0.0)
    return {
        "cardinality": int(len(values)),
        "repeat_level_fraction": float(repeated.mean()),
        "repeat_row_mass": float(counts[repeated].sum() / n),
        "maximum_level_mass": float(counts.max(initial=0) / n),
        "local_excess_mass": float(excess.sum() / n),
    }


def normalized_target(dataset: benchmark.Dataset) -> np.ndarray:
    target = dataset.y["train"].astype(np.float64)
    if dataset.task == "regression":
        scale = float(target.std()) or 1.0
        target = (target - target.mean()) / scale
    return target


def base_residuals(
    dataset: benchmark.Dataset, seed: int, bins: int
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    cache: dict[str, object] = {}
    benchmark.encode_dataset(
        dataset, "schema_ple", seed, bins, 128, 20.0, 1e-3, cache
    )
    clean = benchmark._clean_numeric(dataset.x_num)  # type: ignore[arg-type]
    normalized = cache["normalized_num"]
    ple = cache["numeric_ple"]
    components = [normalized["train"], ple["train"]]  # type: ignore[index]
    if dataset.x_bin is not None:
        components.append(benchmark._clean_numeric(dataset.x_bin)["train"])
    design = np.column_stack(components).astype(np.float32)
    target = normalized_target(dataset)
    residual = benchmark._cross_fitted_numeric_residuals(
        {"train": design}, target, dataset.task, seed
    )
    return clean, residual


def diagnose_column(
    values: np.ndarray,
    target: np.ndarray,
    base_prediction: np.ndarray,
    residual: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    task: str,
    smoothing: float,
) -> dict[str, float | int]:
    correction = np.zeros(len(target), dtype=np.float64)
    fold_wins = 0
    coverage = np.zeros(len(target), dtype=bool)
    for fit_index, holdout_index in folds:
        levels, inverse, counts = np.unique(
            values[fit_index], return_inverse=True, return_counts=True
        )
        means = np.bincount(inverse, weights=residual[fit_index]) / (
            counts + smoothing
        )
        query = values[holdout_index]
        positions = np.searchsorted(levels, query)
        valid = positions < len(levels)
        matched = np.zeros(len(query), dtype=bool)
        matched[valid] = levels[positions[valid]] == query[valid]
        local = np.zeros(len(query), dtype=np.float64)
        local[matched] = means[positions[matched]]
        correction[holdout_index] = local
        coverage[holdout_index] = matched
        before = day2._loss(task, base_prediction[holdout_index], target[holdout_index])
        after = day2._loss(
            task,
            base_prediction[holdout_index] + local,
            target[holdout_index],
        )
        fold_wins += int(after < before)
    before = day2._loss(task, base_prediction, target)
    after = day2._loss(task, base_prediction + correction, target)
    return {
        **support_statistics(values),
        "oof_seen_rate": float(coverage.mean()),
        "relative_residual_gain": float((before - after) / max(before, 1e-12)),
        "fold_wins": fold_wins,
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        fields.extend(field for field in row if field not in fields)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(day2.DATASETS))
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--bins", type=int, default=128)
    parser.add_argument("--max-cardinality", type=int, default=4096)
    parser.add_argument("--smoothing", type=float, default=20.0)
    parser.add_argument("--minimum-relative-gain", type=float, default=5e-4)
    parser.add_argument("--minimum-fold-wins", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=HERE / "results/support_diagnostic.csv"
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for name in args.datasets:
        dataset = benchmark.load_dataset(DAY1 / "data", name)
        if dataset.x_num is None:
            continue
        clean, residual = base_residuals(dataset, args.seed, args.bins)
        target = normalized_target(dataset)
        base_prediction = target - residual
        splitter = (
            StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed + 20_000)
            if dataset.task == "binclass"
            else KFold(n_splits=5, shuffle=True, random_state=args.seed + 20_000)
        )
        folds = list(
            splitter.split(
                clean["train"], target if dataset.task == "binclass" else None
            )
        )
        for column in range(clean["train"].shape[1]):
            cardinality = len(np.unique(clean["train"][:, column]))
            if cardinality > args.max_cardinality:
                continue
            result = diagnose_column(
                clean["train"][:, column],
                target,
                base_prediction,
                residual,
                folds,
                dataset.task,
                args.smoothing,
            )
            selected = (
                result["relative_residual_gain"] >= args.minimum_relative_gain
                and result["fold_wins"] >= args.minimum_fold_wins
            )
            rows.append(
                {
                    "dataset": name,
                    "task": dataset.task,
                    "column": column,
                    "selected": int(selected),
                    **result,
                }
            )
        write_rows(args.output, rows)
        ranked = sorted(
            (row for row in rows if row["dataset"] == name),
            key=lambda row: float(row["relative_residual_gain"]),
            reverse=True,
        )[:8]
        print(
            json.dumps(
                {
                    "dataset": name,
                    "selected": [
                        int(row["column"])
                        for row in rows
                        if row["dataset"] == name and row["selected"]
                    ],
                    "top": ranked,
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
