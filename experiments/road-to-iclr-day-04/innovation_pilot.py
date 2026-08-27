"""Pilot an exact common-factor / innovation decomposition for tabular nets.

The transform is fitted on the training partition only.  It compares the
ordinary quantile/contrast representation with two information-preserving
views:

* ``common_skip`` keeps the baseline coordinates and appends whitened PCA
  scores, giving common variation a short path;
* ``innovation_sum`` replaces the centered coordinates by their orthogonal
  residual and appends the same common scores.  Up to numerical precision,
  the original centered input can be reconstructed from these two blocks.

This is a falsification pilot, not a final benchmark.  It deliberately uses a
small, fixed training recipe and saves every paired cell.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DAY3 = HERE.parent / "road-to-iclr-day-03"
sys.path.insert(0, str(DAY3))

from experiments.day3.core import (  # noqa: E402
    Dataset,
    PARTS,
    base_schema,
    clean_numeric,
    load_dataset,
    make_prepared,
    train_model,
)


WEATHER_ROOT = Path(os.environ.get(
    "DAY4_WEATHER_ROOT",
    "/home/byunhanjoon/proposal/generated_tabred_array/weather",
))


def load_weather(max_train_rows: int, max_eval_rows: int, seed: int) -> Dataset:
    """Load the local official TabReD Weather array with fixed subsampling."""
    rng = np.random.default_rng(seed)
    split_root = WEATHER_ROOT / "split-default"
    raw_indices = {
        "train": np.load(split_root / "train_idx.npy"),
        "val": np.load(split_root / "val_idx.npy"),
        "test": np.load(split_root / "test_idx.npy"),
    }
    indices = {}
    for offset, part in enumerate(PARTS):
        source = raw_indices[part]
        limit = max_train_rows if part == "train" else max_eval_rows
        if len(source) > limit:
            local_rng = np.random.default_rng(seed + offset)
            source = np.sort(local_rng.choice(source, limit, replace=False))
        indices[part] = source
    x_num = np.load(WEATHER_ROOT / "X_num.npy", mmap_mode="r")
    x_bin = np.load(WEATHER_ROOT / "X_bin.npy", mmap_mode="r")
    y = np.load(WEATHER_ROOT / "Y.npy", mmap_mode="r")
    return Dataset(
        name="tabred-weather",
        task="regression",
        x_num={p: np.asarray(x_num[indices[p]]) for p in PARTS},
        x_bin={p: np.asarray(x_bin[indices[p]]) for p in PARTS},
        x_cat=None,
        y={p: np.asarray(y[indices[p]], dtype=np.float32) for p in PARTS},
        n_classes=1,
        split_fingerprint=f"official-subsample-{seed}",
    )


def measure_adaptive_innovation(
    dataset,
    baseline: dict[str, np.ndarray],
    *,
    max_rank: int,
    variance_fraction: float,
    minimum_levels: int = 128,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Apply the innovation split only to high-cardinality numerical fields.

    Low-cardinality numeric fields, binary fields, and categorical contrasts
    stay in their native coordinates.  This is a deliberately simple proxy for
    the continuous component of a mixed marginal measure.
    """
    if dataset.x_num is None:
        return baseline, {"continuous_columns": [], "rank": 0}
    clean = clean_numeric(dataset.x_num)
    n_num = clean["train"].shape[1]
    cardinalities = np.asarray(
        [len(np.unique(clean["train"][:, j])) for j in range(n_num)], dtype=int
    )
    continuous = np.flatnonzero(cardinalities >= minimum_levels)
    if len(continuous) < 2:
        return baseline, {
            "continuous_columns": continuous.tolist(),
            "cardinalities": cardinalities.tolist(),
            "rank": 0,
        }
    selected = {p: baseline[p][:, continuous] for p in PARTS}
    transformed, metadata = factorize(
        selected,
        max_rank=max_rank,
        variance_fraction=variance_fraction,
    )
    innovation = transformed["innovation_sum"]
    keep = np.ones(baseline["train"].shape[1], dtype=bool)
    keep[continuous] = False
    result = {
        p: np.ascontiguousarray(
            np.column_stack((baseline[p][:, keep], innovation[p])), dtype=np.float32
        )
        for p in PARTS
    }
    return result, {
        **metadata,
        "continuous_columns": continuous.tolist(),
        "cardinalities": cardinalities.tolist(),
        "minimum_levels": minimum_levels,
    }


def factorize(
    parts: dict[str, np.ndarray],
    *,
    max_rank: int,
    variance_fraction: float,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, object]]:
    train = np.asarray(parts["train"], dtype=np.float64)
    mean = train.mean(axis=0)
    centered_train = train - mean
    covariance = centered_train.T @ centered_train / max(len(train) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    positive = eigenvalues > max(float(eigenvalues[0]) * 1e-10, 1e-12)
    eigenvalues = eigenvalues[positive]
    eigenvectors = eigenvectors[:, positive]
    if not len(eigenvalues):
        raise ValueError("The baseline representation has zero empirical rank")
    cumulative = np.cumsum(eigenvalues) / eigenvalues.sum()
    requested = int(np.searchsorted(cumulative, variance_fraction) + 1)
    rank = max(1, min(max_rank, requested, len(eigenvalues)))
    values = eigenvalues[:rank]
    vectors = eigenvectors[:, :rank]

    common_skip: dict[str, np.ndarray] = {}
    innovation_sum: dict[str, np.ndarray] = {}
    reconstruction_error: dict[str, float] = {}
    residual_train = centered_train - (centered_train @ vectors) @ vectors.T
    residual_scale = residual_train.std(axis=0)
    residual_keep = residual_scale > 1e-8
    residual_scale = residual_scale[residual_keep]
    for part in PARTS:
        original = np.asarray(parts[part], dtype=np.float64)
        centered = original - mean
        raw_scores = centered @ vectors
        scores = raw_scores / np.sqrt(values)[None, :]
        residual = centered - raw_scores @ vectors.T
        common_skip[part] = np.ascontiguousarray(
            np.column_stack((original, scores)), dtype=np.float32
        )
        innovation_sum[part] = np.ascontiguousarray(
            np.column_stack((residual[:, residual_keep] / residual_scale, scores)),
            dtype=np.float32,
        )
        reconstructed = residual + raw_scores @ vectors.T
        reconstruction_error[part] = float(
            np.max(np.abs(reconstructed - centered), initial=0.0)
        )
    return {
        "common_skip": common_skip,
        "innovation_sum": innovation_sum,
    }, {
        "rank": rank,
        "requested_rank": requested,
        "explained_variance": float(cumulative[rank - 1]),
        "baseline_dimension": int(train.shape[1]),
        "innovation_dimension": int(residual_keep.sum()),
        "eigenvalues": values.tolist(),
        "reconstruction_max_abs": reconstruction_error,
    }


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["adult", "california", "churn", "diamond"],
    )
    parser.add_argument("--models", nargs="+", default=["mlp", "resnet"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260826])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-rank", type=int, default=8)
    parser.add_argument("--variance-fraction", type=float, default=0.5)
    parser.add_argument("--max-train-rows", type=int, default=50000)
    parser.add_argument("--max-eval-rows", type=int, default=15000)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--output", type=Path, default=HERE / "results/innovation_pilot.csv")
    args = parser.parse_args()

    rows: list[dict[str, object]] = list(read_rows(args.output))
    completed = {
        (str(row["dataset"]), str(row["model"]), int(row["seed"]), str(row["method"]))
        for row in rows
    }
    metadata: dict[str, object] = {}
    for dataset_name in args.datasets:
        if dataset_name == "tabred-weather":
            dataset = load_weather(
                args.max_train_rows, args.max_eval_rows, seed=20260826
            )
        else:
            dataset = load_dataset(
                dataset_name,
                max_train_rows=args.max_train_rows,
                max_eval_rows=args.max_eval_rows,
                sample_seed=20260826,
            )
        baseline = base_schema(dataset, seed=20260826)
        variants, factor_metadata = factorize(
            baseline,
            max_rank=args.max_rank,
            variance_fraction=args.variance_fraction,
        )
        adaptive, adaptive_metadata = measure_adaptive_innovation(
            dataset,
            baseline,
            max_rank=args.max_rank,
            variance_fraction=args.variance_fraction,
        )
        variants = {"baseline": baseline, **variants, "measure_innovation": adaptive}
        metadata[dataset_name] = {
            "global_factorization": factor_metadata,
            "measure_adaptive": adaptive_metadata,
        }
        for model in args.models:
            for seed in args.seeds:
                for method, features in variants.items():
                    key = (dataset_name, model, seed, method)
                    if key in completed:
                        continue
                    prepared = make_prepared(
                        dataset,
                        features,
                        {"method": method, "factorization": factor_metadata},
                    )
                    result, _ = train_model(
                        prepared,
                        seed=seed,
                        device=args.device,
                        model_name=model,
                        width=args.width,
                        depth=args.depth,
                        dropout=0.1,
                        learning_rate=1e-3,
                        weight_decay=1e-4,
                        batch_size=512,
                        max_epochs=args.epochs,
                        patience=args.patience,
                    )
                    row = {
                        "dataset": dataset_name,
                        "task": dataset.task,
                        "model": model,
                        "seed": seed,
                        "method": method,
                        "factor_rank": factor_metadata["rank"],
                        "explained_variance": factor_metadata["explained_variance"],
                        **result,
                    }
                    rows.append(row)
                    completed.add(key)
                    write_rows(args.output, rows)
                    print(json.dumps(row, sort_keys=True), flush=True)
    (args.output.parent / "innovation_factorization.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
