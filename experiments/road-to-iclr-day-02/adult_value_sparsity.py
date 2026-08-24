"""Measure how many exact Adult values are needed for the identity gain."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import cross_dataset_models as experiment


HERE = Path(__file__).resolve().parent


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=experiment.DAY1 / "data")
    parser.add_argument("--models", nargs="+", choices=experiment.MODELS, default=experiment.MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--top-k", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--bins", type=int, default=128)
    parser.add_argument("--max-cardinality", type=int, default=128)
    parser.add_argument("--smoothing", type=float, default=20.0)
    parser.add_argument("--minimum-relative-gain", type=float, default=5e-4)
    parser.add_argument("--minimum-fold-wins", type=int, default=3)
    parser.add_argument("--ensemble-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output", type=Path, default=HERE / "results" / "adult_value_sparsity.csv"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = experiment.benchmark.load_dataset(args.data, "adult")
    rows: list[dict[str, object]] = [] if args.force else list(read_rows(args.output))
    completed = {
        (str(row["model"]), int(row["seed"]), str(row["representation"]))
        for row in rows
    }
    for seed in args.seeds:
        (
            base_variants,
            _,
            utility_columns,
            utility_statistics,
            _,
        ) = experiment.encode_variants(
            dataset,
            seed,
            args.bins,
            args.max_cardinality,
            args.smoothing,
            args.minimum_relative_gain,
            args.minimum_fold_wins,
        )
        assert dataset.x_num is not None
        clean = experiment.benchmark._clean_numeric(dataset.x_num)
        variants = {"baseline_ple": base_variants["baseline_ple"]}
        selected_values: dict[str, dict[int, list[float]]] = {}
        for top_k in args.top_k:
            parts, values = experiment._top_k_identity(clean, utility_columns, top_k)
            name = f"top{top_k}"
            variants[name] = experiment._append_view(
                variants["baseline_ple"], parts, name, utility_columns
            )
            selected_values[name] = values
        variants["full"] = base_variants["utility_identity"]
        selected_values["full"] = {}

        for model_name in args.models:
            config = experiment.MODEL_CONFIGS[model_name]
            parameter_budget = experiment.baseline_parameter_count(
                variants["baseline_ple"], model_name, config, args.ensemble_size
            )
            for representation, encoded in variants.items():
                key = (model_name, seed, representation)
                if key in completed:
                    continue
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
                    parameter_budget,
                )
                rows.append(
                    {
                        "dataset": "adult",
                        "task": dataset.task,
                        "model": model_name,
                        "seed": seed,
                        "representation": representation,
                        "top_k": 0
                        if representation == "baseline_ple"
                        else -1
                        if representation == "full"
                        else int(representation.removeprefix("top")),
                        "utility_columns": experiment._text(utility_columns),
                        "utility_statistics": json.dumps(
                            utility_statistics, sort_keys=True, separators=(",", ":")
                        ),
                        "selected_values": json.dumps(
                            selected_values.get(representation, {}),
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        **output.result,
                        **experiment._extra_metrics(dataset, encoded, output),
                    }
                )
                completed.add(key)
                write_rows(args.output, rows)
                print(
                    f"seed={seed} {model_name:<6} {representation:<12} "
                    f"test={float(output.result['test_score']):.6f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
