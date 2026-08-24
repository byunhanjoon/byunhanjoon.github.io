"""Benchmark frequency-shrunk, cross-fitted exact-value residual maps.

The residual map is a compact supervised alternative to a full identity view.
For every utility-selected numerical column, it maps an exact value to the mean
residual left by a numerical-only PLE model. Training rows receive out-of-fold
maps; validation and test rows receive maps fitted on all training OOF
residuals. Values unseen during fitting map to zero.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedKFold

import cross_dataset_models as experiment


HERE = Path(__file__).resolve().parent
ALPHAS = (0.0, 5.0, 20.0, 100.0)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row.setdefault("bins", "128")
        row.setdefault("train_rows", "")
        row.setdefault("val_rows", "")
        row.setdefault("test_rows", "")
    return rows


def subsample_dataset(
    dataset: experiment.benchmark.Dataset,
    max_train_rows: int | None,
    max_eval_rows: int | None,
    sample_seed: int,
) -> experiment.benchmark.Dataset:
    """Take fixed, target-blind row samples while preserving official partitions."""

    limits = {
        "train": max_train_rows,
        "val": max_eval_rows,
        "test": max_eval_rows,
    }
    indices: dict[str, np.ndarray] = {}
    for offset, part in enumerate(("train", "val", "test")):
        size = len(dataset.y[part])
        limit = limits[part]
        if limit is None or limit >= size:
            indices[part] = np.arange(size)
        else:
            rng = np.random.default_rng(sample_seed + offset)
            indices[part] = np.sort(rng.choice(size, size=limit, replace=False))

    def subset(
        values: dict[str, np.ndarray] | None,
    ) -> dict[str, np.ndarray] | None:
        if values is None:
            return None
        return {part: values[part][indices[part]] for part in indices}

    return experiment.benchmark.Dataset(
        name=dataset.name,
        task=dataset.task,
        x_num=subset(dataset.x_num),
        x_bin=subset(dataset.x_bin),
        x_cat=subset(dataset.x_cat),
        y=subset(dataset.y),  # type: ignore[arg-type]
    )


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def residual_map(
    dataset: experiment.benchmark.Dataset,
    seed: int,
    columns: tuple[int, ...],
    alpha: float,
    cache: dict[str, object],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    assert dataset.x_num is not None
    clean = experiment.benchmark._clean_numeric(dataset.x_num)
    residual = np.asarray(cache["numeric_residual"], dtype=np.float64)
    target = dataset.y["train"]
    splitter = (
        StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 20_000)
        if dataset.task == "binclass"
        else KFold(n_splits=5, shuffle=True, random_state=seed + 20_000)
    )
    folds = list(
        splitter.split(
            clean["train"], target if dataset.task == "binclass" else None
        )
    )
    output = {
        part: np.zeros((len(values), len(columns)), dtype=np.float64)
        for part, values in clean.items()
    }
    metadata: dict[str, object] = {"alpha": alpha, "columns": list(columns)}
    column_metadata: dict[int, dict[str, float | int]] = {}
    for output_column, column in enumerate(columns):
        for fit_index, holdout_index in folds:
            fit_values = clean["train"][fit_index, column]
            values, inverse, counts = np.unique(
                fit_values, return_inverse=True, return_counts=True
            )
            means = np.bincount(inverse, weights=residual[fit_index]) / (
                counts + alpha
            )
            query = clean["train"][holdout_index, column]
            positions = np.searchsorted(values, query)
            valid = positions < len(values)
            matched = np.zeros(len(query), dtype=bool)
            matched[valid] = values[positions[valid]] == query[valid]
            output["train"][holdout_index[matched], output_column] = means[
                positions[matched]
            ]

        values, inverse, counts = np.unique(
            clean["train"][:, column], return_inverse=True, return_counts=True
        )
        means = np.bincount(inverse, weights=residual) / (counts + alpha)
        unseen_rates: dict[str, float] = {}
        for part in ("val", "test"):
            query = clean[part][:, column]
            positions = np.searchsorted(values, query)
            valid = positions < len(values)
            matched = np.zeros(len(query), dtype=bool)
            matched[valid] = values[positions[valid]] == query[valid]
            output[part][matched, output_column] = means[positions[matched]]
            unseen_rates[part] = float(1.0 - matched.mean())

        # RMS scaling preserves the semantically important zero fallback.
        scale = max(
            float(np.sqrt(np.mean(output["train"][:, output_column] ** 2))),
            1e-6,
        )
        for part in output:
            output[part][:, output_column] /= scale
        column_metadata[column] = {
            "cardinality": int(len(values)),
            "scale": scale,
            "val_unseen_rate": unseen_rates["val"],
            "test_unseen_rate": unseen_rates["test"],
        }
    metadata["column_metadata"] = column_metadata
    return (
        {part: values.astype(np.float32) for part, values in output.items()},
        metadata,
    )


def encode_variants(
    dataset: experiment.benchmark.Dataset,
    seed: int,
    bins: int,
    alphas: tuple[float, ...],
) -> tuple[
    dict[str, experiment.benchmark.EncodedDataset],
    dict[str, dict[str, object]],
    tuple[int, ...],
    dict[int, dict[str, float | int]],
]:
    cache: dict[str, object] = {}
    base = experiment.benchmark.encode_dataset(
        dataset, "schema_ple", seed, bins, 128, 20.0, 1e-3, cache
    )
    columns, utility_statistics = experiment.utility_diagnostic(
        dataset, seed, bins, 128, 20.0, 5e-4, 3, cache
    )
    assert dataset.x_num is not None
    clean = experiment.benchmark._clean_numeric(dataset.x_num)
    variants = {"baseline_ple": base}
    metadata: dict[str, dict[str, object]] = {"baseline_ple": {}}
    variants["utility_identity"] = experiment._append_view(
        base,
        experiment._full_identity(clean, columns),
        "utility_identity",
        columns,
    )
    metadata["utility_identity"] = {}
    for alpha in alphas:
        name = f"residual_map_a{alpha:g}"
        parts, map_metadata = residual_map(dataset, seed, columns, alpha, cache)
        variants[name] = experiment._append_view(base, parts, name, columns)
        metadata[name] = map_metadata
    return variants, metadata, columns, utility_statistics


def result_row(
    dataset: experiment.benchmark.Dataset,
    model_name: str,
    seed: int,
    bins: int,
    representation: str,
    selected_columns: tuple[int, ...],
    utility_statistics: dict[int, dict[str, float | int]],
    map_metadata: dict[str, object],
    output: experiment.benchmark.TrainOutput,
    encoded: experiment.benchmark.EncodedDataset,
    reused_baseline: bool,
) -> dict[str, object]:
    return {
        "dataset": dataset.name,
        "task": dataset.task,
        "model": model_name,
        "seed": seed,
        "bins": bins,
        "train_rows": len(dataset.y["train"]),
        "val_rows": len(dataset.y["val"]),
        "test_rows": len(dataset.y["test"]),
        "representation": representation,
        "selected_columns": experiment._text(selected_columns),
        "utility_statistics": json.dumps(
            utility_statistics, sort_keys=True, separators=(",", ":")
        ),
        "map_metadata": json.dumps(
            map_metadata, sort_keys=True, separators=(",", ":")
        ),
        "reused_baseline": int(reused_baseline),
        **output.result,
        **experiment._extra_metrics(dataset, encoded, output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=experiment.DAY1 / "data")
    parser.add_argument(
        "--datasets", nargs="+", choices=experiment.DATASETS, default=experiment.DATASETS
    )
    parser.add_argument("--models", nargs="+", choices=experiment.MODELS, default=experiment.MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--alphas", nargs="+", type=float, default=list(ALPHAS))
    parser.add_argument("--bins", type=int, default=128)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-eval-rows", type=int)
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--ensemble-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output", type=Path, default=HERE / "results" / "residual_map_benchmark.csv"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    rows: list[dict[str, object]] = [] if args.force else list(read_rows(args.output))
    completed = {
        (str(row["dataset"]), str(row["model"]), int(row["seed"]), str(row["representation"]))
        for row in rows
    }
    for dataset_name in args.datasets:
        dataset = experiment.benchmark.load_dataset(args.data, dataset_name)
        dataset = subsample_dataset(
            dataset,
            args.max_train_rows,
            args.max_eval_rows,
            args.sample_seed,
        )
        for seed in args.seeds:
            variants, metadata, columns, utility_statistics = encode_variants(
                dataset, seed, args.bins, tuple(args.alphas)
            )
            print(
                f"{dataset_name} seed={seed} selected={experiment._text(columns) or '-'}",
                flush=True,
            )
            for model_name in args.models:
                config = experiment.MODEL_CONFIGS[model_name]
                parameter_budget = experiment.baseline_parameter_count(
                    variants["baseline_ple"], model_name, config, args.ensemble_size
                )
                baseline_output: experiment.benchmark.TrainOutput | None = None
                for representation, encoded in variants.items():
                    key = (dataset_name, model_name, seed, representation)
                    if key in completed:
                        continue
                    reused_baseline = not columns and representation != "baseline_ple"
                    if reused_baseline:
                        assert baseline_output is not None
                        output = baseline_output
                    else:
                        output = experiment.train_model(
                            encoded,
                            model_name,
                            seed,
                            experiment.batch_size(dataset_name),
                            device,
                            config,
                            args.ensemble_size,
                            args.max_epochs,
                            args.patience,
                            parameter_budget,
                        )
                    if representation == "baseline_ple":
                        baseline_output = output
                    rows.append(
                        result_row(
                            dataset,
                            model_name,
                            seed,
                            args.bins,
                            representation,
                            columns,
                            utility_statistics,
                            metadata[representation],
                            output,
                            encoded,
                            reused_baseline,
                        )
                    )
                    completed.add(key)
                    write_rows(args.output, rows)
                    print(
                        f"  {model_name:<6} {representation:<18} "
                        f"test={float(output.result['test_score']):.6f}" +
                        (" reused" if reused_baseline else ""),
                        flush=True,
                    )


if __name__ == "__main__":
    main()
