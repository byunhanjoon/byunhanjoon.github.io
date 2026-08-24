"""Profile Adult identity gains by exact-value training frequency."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

import cross_dataset_models as experiment


HERE = Path(__file__).resolve().parent
FEATURES = {3: "capital_gain", 4: "capital_loss"}
BUCKETS = (
    ("unseen", 0, 0),
    ("1-9", 1, 9),
    ("10-99", 10, 99),
    ("100-999", 100, 999),
    ("1000+", 1000, np.iinfo(np.int64).max),
)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def frequencies(train: np.ndarray, query: np.ndarray) -> np.ndarray:
    values, counts = np.unique(train, return_counts=True)
    positions = np.searchsorted(values, query)
    valid = positions < len(values)
    matched = np.zeros(len(query), dtype=bool)
    matched[valid] = values[positions[valid]] == query[valid]
    output = np.zeros(len(query), dtype=np.int64)
    output[matched] = counts[positions[matched]]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=experiment.DAY1 / "data")
    parser.add_argument("--models", nargs="+", choices=experiment.MODELS, default=experiment.MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ensemble-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument(
        "--output", type=Path, default=HERE / "results" / "adult_frequency_generalization.csv"
    )
    args = parser.parse_args()

    dataset = experiment.benchmark.load_dataset(args.data, "adult")
    assert dataset.x_num is not None
    clean = experiment.benchmark._clean_numeric(dataset.x_num)
    column_frequency = {
        column: frequencies(clean["train"][:, column], clean["test"][:, column])
        for column in FEATURES
    }
    profiles = {FEATURES[column]: values for column, values in column_frequency.items()}
    profiles["minimum_selected_frequency"] = np.minimum.reduce(
        list(column_frequency.values())
    )
    target = dataset.y["test"].astype(np.float64)
    device = torch.device(args.device)
    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        variants, _, utility_columns, _, _ = experiment.encode_variants(
            dataset, seed, 128, 128, 20.0, 5e-4, 3
        )
        if utility_columns != (3, 4):
            raise RuntimeError(f"Unexpected Adult selection: {utility_columns}")
        selected = {
            "baseline_ple": variants["baseline_ple"],
            "utility_identity": variants["utility_identity"],
        }
        for model_name in args.models:
            config = experiment.MODEL_CONFIGS[model_name]
            budget = experiment.baseline_parameter_count(
                selected["baseline_ple"], model_name, config, args.ensemble_size
            )
            for representation, encoded in selected.items():
                output = experiment.train_model(
                    encoded,
                    model_name,
                    seed,
                    experiment.benchmark.BATCH_SIZES["adult"],
                    device,
                    config,
                    args.ensemble_size,
                    args.max_epochs,
                    args.patience,
                    budget,
                )
                logits = output.test_prediction.astype(np.float64)
                probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
                for profile_name, profile_frequency in profiles.items():
                    for bucket, lower, upper in BUCKETS:
                        mask = (profile_frequency >= lower) & (profile_frequency <= upper)
                        if not mask.any():
                            continue
                        rows.append(
                            {
                                "model": model_name,
                                "seed": seed,
                                "representation": representation,
                                "profile": profile_name,
                                "bucket": bucket,
                                "rows": int(mask.sum()),
                                "accuracy": float(
                                    ((probability[mask] >= 0.5) == target[mask]).mean()
                                ),
                                "logloss": experiment._loss(
                                    "binclass", probability[mask], target[mask]
                                ),
                            }
                        )
                print(
                    f"seed={seed} {model_name:<6} {representation:<16} "
                    f"test={float(output.result['test_score']):.6f}",
                    flush=True,
                )
    write_rows(args.output, rows)


if __name__ == "__main__":
    main()
