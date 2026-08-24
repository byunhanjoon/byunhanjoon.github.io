"""Test whether Adult's identity-view gain is really a point-mass effect.

This suite deliberately reuses the frozen Day 1 PLE + ResNet configuration.
It has two stages:

1. attribution on the released split: add identity views one column at a time,
   leave each selected column out, and compare full identity with one indicator
   for the most frequent value of each selected column;
2. split stability: rebuild stratified train/validation/test splits, rerun the
   residual diagnostic inside each training split, and compare its selected
   identity view with matched mode indicators.

No test score participates in feature selection or model selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit


HERE = Path(__file__).resolve().parent
DAY1 = HERE.parent / "road-to-iclr-day-01"
sys.path.insert(0, str(DAY1))

import real_data_benchmark as benchmark  # noqa: E402


FEATURE_NAMES = {
    0: "age",
    1: "fnlwgt",
    2: "education_num",
    3: "capital_gain",
    4: "capital_loss",
    5: "hours_per_week",
}
SELECTED_COLUMNS = (3, 4, 5)
ATTRIBUTION_CONFIGS = {
    "baseline_ple": (),
    "identity_capital_gain": (3,),
    "identity_capital_loss": (4,),
    "identity_hours_per_week": (5,),
    "identity_without_capital_gain": (4, 5),
    "identity_without_capital_loss": (3, 5),
    "identity_without_hours_per_week": (3, 4),
    "identity_all_three": SELECTED_COLUMNS,
    "mode_indicators": SELECTED_COLUMNS,
}

# Frozen from Day 1 candidate base_018.
BINS = 128
WIDTH = 384
DEPTH = 2
DROPOUT = 0.0
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MODEL = "resnet"
ACTIVATION = "gelu"
LOW_CARDINALITY = 128
IDENTITY_EFFECT_THRESHOLD = 1e-3
SMOOTHING = 20.0


def _column_text(columns: Iterable[int]) -> str:
    return ";".join(str(column) for column in columns)


def _mode(values: np.ndarray) -> float:
    unique, counts = np.unique(values, return_counts=True)
    return float(unique[int(np.argmax(counts))])


def _append_view(
    base: benchmark.EncodedDataset,
    parts: dict[str, np.ndarray],
    name: str,
    columns: tuple[int, ...],
) -> benchmark.EncodedDataset:
    return benchmark.EncodedDataset(
        x={
            part: np.ascontiguousarray(
                np.column_stack((base.x[part], parts[part])), dtype=np.float32
            )
            for part in base.x
        },
        y=base.y,
        task=base.task,
        y_mean=base.y_mean,
        y_scale=base.y_scale,
        view_names=base.view_names + (name,),
        view_sizes=base.view_sizes + (parts["train"].shape[1],),
        selected_numeric=columns,
    )


def encode_fixed_columns(
    dataset: benchmark.Dataset,
    seed: int,
    columns: tuple[int, ...],
    representation: str,
    cache: dict[str, object],
) -> tuple[benchmark.EncodedDataset, tuple[float, ...]]:
    base = benchmark.encode_dataset(
        dataset,
        "schema_ple",
        seed,
        BINS,
        LOW_CARDINALITY,
        SMOOTHING,
        IDENTITY_EFFECT_THRESHOLD,
        cache,
    )
    if not columns:
        return base, ()
    assert dataset.x_num is not None
    clean = benchmark._clean_numeric(dataset.x_num)
    if representation == "identity":
        parts = benchmark._one_hot(
            {part: values[:, columns] for part, values in clean.items()}
        )
        return _append_view(base, parts, "numeric_identity", columns), ()
    if representation == "modes":
        modes = tuple(_mode(clean["train"][:, column]) for column in columns)
        parts = {
            part: np.column_stack(
                [
                    (values[:, column] == mode).astype(np.float32)
                    for column, mode in zip(columns, modes)
                ]
            )
            for part, values in clean.items()
        }
        return _append_view(base, parts, "mode_indicators", columns), modes
    raise ValueError(f"Unknown representation: {representation}")


def encode_diagnostic_variants(
    dataset: benchmark.Dataset,
    seed: int,
) -> dict[str, tuple[benchmark.EncodedDataset, tuple[float, ...]]]:
    cache: dict[str, object] = {}
    diagnostic = benchmark.encode_dataset(
        dataset,
        "diagnostic_identity",
        seed,
        BINS,
        LOW_CARDINALITY,
        SMOOTHING,
        IDENTITY_EFFECT_THRESHOLD,
        cache,
    )
    baseline, _ = encode_fixed_columns(dataset, seed, (), "identity", cache)
    modes, mode_values = encode_fixed_columns(
        dataset, seed, diagnostic.selected_numeric, "modes", cache
    )
    return {
        "baseline_ple": (baseline, ()),
        "diagnostic_identity": (diagnostic, ()),
        "diagnostic_modes": (modes, mode_values),
    }


def target_parameter_count(data: benchmark.EncodedDataset) -> int:
    return benchmark._parameter_count(
        benchmark._make_model(
            data,
            WIDTH,
            DEPTH,
            DROPOUT,
            False,
            MODEL,
            ACTIVATION,
        )
    )


def train(
    data: benchmark.EncodedDataset,
    seed: int,
    device: torch.device,
    max_epochs: int,
    patience: int,
    parameter_budget: int,
) -> benchmark.TrainOutput:
    return benchmark.train_one(
        data,
        seed,
        benchmark.BATCH_SIZES["adult"],
        device,
        WIDTH,
        DEPTH,
        DROPOUT,
        LEARNING_RATE,
        WEIGHT_DECAY,
        max_epochs,
        patience,
        gated=False,
        gate_entropy_weight=0.0,
        target_parameters=parameter_budget,
        model_type=MODEL,
        activation=ACTIVATION,
    )


def result_row(
    stage: str,
    split_seed: int,
    model_seed: int,
    configuration: str,
    modes: tuple[float, ...],
    output: benchmark.TrainOutput,
) -> dict[str, object]:
    return {
        "stage": stage,
        "split_seed": split_seed,
        "model_seed": model_seed,
        "configuration": configuration,
        "selected_columns": str(output.result["selected_numeric"]),
        "selected_features": ";".join(
            FEATURE_NAMES[int(column)]
            for column in str(output.result["selected_numeric"]).split(";")
            if column
        ),
        "mode_values": ";".join(f"{value:g}" for value in modes),
        **output.result,
    }


def read_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_attribution(args: argparse.Namespace, device: torch.device) -> None:
    output_path = args.output_dir / "adult_attribution.csv"
    rows = [] if args.force else read_rows(output_path)
    completed = {
        (int(row["model_seed"]), str(row["configuration"])) for row in rows
    }
    dataset = benchmark.load_dataset(args.data, "adult")
    for seed in args.attribution_seeds:
        cache: dict[str, object] = {}
        baseline, _ = encode_fixed_columns(dataset, seed, (), "identity", cache)
        parameter_budget = target_parameter_count(baseline)
        for configuration, columns in ATTRIBUTION_CONFIGS.items():
            if (seed, configuration) in completed:
                continue
            representation = "modes" if configuration == "mode_indicators" else "identity"
            encoded, modes = encode_fixed_columns(
                dataset, seed, columns, representation, cache
            )
            output = train(
                encoded,
                seed,
                device,
                args.max_epochs,
                args.patience,
                parameter_budget,
            )
            rows.append(result_row("attribution", -1, seed, configuration, modes, output))
            write_rows(output_path, rows)
            print(
                f"attribution seed={seed} {configuration} "
                f"test={float(output.result['test_score']):.6f}",
                flush=True,
            )


def resplit_adult(data_root: Path, split_seed: int) -> benchmark.Dataset:
    directory = data_root / "adult"
    arrays = {
        name: np.load(directory / f"{name}.npy")
        for name in ("x_num", "x_bin", "x_cat", "y")
    }
    default = {
        part: np.load(directory / "splits" / "default" / f"{part}.npy")
        for part in ("train", "val", "test")
    }
    indices = np.arange(len(arrays["y"]))
    first = StratifiedShuffleSplit(
        n_splits=1, test_size=len(default["test"]), random_state=split_seed
    )
    train_val, test = next(first.split(indices, arrays["y"]))
    second = StratifiedShuffleSplit(
        n_splits=1, test_size=len(default["val"]), random_state=split_seed + 1
    )
    train_position, val_position = next(
        second.split(train_val, arrays["y"][train_val])
    )
    split_indices = {
        "train": train_val[train_position],
        "val": train_val[val_position],
        "test": test,
    }

    def split(name: str) -> dict[str, np.ndarray]:
        return {
            part: arrays[name][part_indices]
            for part, part_indices in split_indices.items()
        }

    return benchmark.Dataset(
        name="adult",
        task="binclass",
        x_num=split("x_num"),
        x_bin=split("x_bin"),
        x_cat=split("x_cat"),
        y=split("y"),
    )


def run_stability(args: argparse.Namespace, device: torch.device) -> None:
    output_path = args.output_dir / "adult_split_stability.csv"
    rows = [] if args.force else read_rows(output_path)
    completed = {
        (int(row["split_seed"]), int(row["model_seed"]), str(row["configuration"]))
        for row in rows
    }
    for split_seed in args.split_seeds:
        dataset = resplit_adult(args.data, split_seed)
        for model_seed in args.stability_model_seeds:
            variants = encode_diagnostic_variants(dataset, model_seed)
            parameter_budget = target_parameter_count(variants["baseline_ple"][0])
            for configuration, (encoded, modes) in variants.items():
                key = (split_seed, model_seed, configuration)
                if key in completed:
                    continue
                output = train(
                    encoded,
                    model_seed,
                    device,
                    args.max_epochs,
                    args.patience,
                    parameter_budget,
                )
                rows.append(
                    result_row(
                        "split_stability",
                        split_seed,
                        model_seed,
                        configuration,
                        modes,
                        output,
                    )
                )
                write_rows(output_path, rows)
                print(
                    f"stability split={split_seed} seed={model_seed} {configuration} "
                    f"selected={output.result['selected_numeric']} "
                    f"test={float(output.result['test_score']):.6f}",
                    flush=True,
                )


def write_residual_profile(args: argparse.Namespace) -> None:
    dataset = benchmark.load_dataset(args.data, "adult")
    assert dataset.x_num is not None
    cache: dict[str, object] = {}
    encoded = benchmark.encode_dataset(
        dataset,
        "diagnostic_identity",
        0,
        BINS,
        LOW_CARDINALITY,
        SMOOTHING,
        IDENTITY_EFFECT_THRESHOLD,
        cache,
    )
    residual = np.asarray(cache["numeric_residual"])
    numeric = benchmark._clean_numeric(dataset.x_num)["train"]
    target = dataset.y["train"]
    rows: list[dict[str, object]] = []
    for column in encoded.selected_numeric:
        values, inverse, counts = np.unique(
            numeric[:, column], return_inverse=True, return_counts=True
        )
        residual_sums = np.bincount(inverse, weights=residual)
        target_sums = np.bincount(inverse, weights=target)
        for index, value in enumerate(values):
            rows.append(
                {
                    "column": column,
                    "feature": FEATURE_NAMES[column],
                    "selected": int(column in encoded.selected_numeric),
                    "value": float(value),
                    "count": int(counts[index]),
                    "frequency": float(counts[index] / len(numeric)),
                    "target_rate": float(target_sums[index] / counts[index]),
                    "mean_oof_residual": float(residual_sums[index] / counts[index]),
                }
            )
    write_rows(args.output_dir / "adult_residual_profile.csv", rows)


def _float_rows(path: Path) -> list[dict[str, object]]:
    return read_rows(path)


def summarize(args: argparse.Namespace) -> None:
    summary: dict[str, object] = {
        "frozen_configuration": {
            "bins": BINS,
            "width": WIDTH,
            "depth": DEPTH,
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "model": MODEL,
            "activation": ACTIVATION,
            "parameter_matched": True,
        }
    }
    attribution_path = args.output_dir / "adult_attribution.csv"
    if attribution_path.exists():
        rows = _float_rows(attribution_path)
        grouped: dict[str, list[float]] = {}
        by_seed: dict[tuple[int, str], float] = {}
        for row in rows:
            configuration = str(row["configuration"])
            score = float(row["test_score"])
            grouped.setdefault(configuration, []).append(score)
            by_seed[(int(row["model_seed"]), configuration)] = score
        baseline = {
            seed: score
            for (seed, configuration), score in by_seed.items()
            if configuration == "baseline_ple"
        }
        summary["attribution"] = {
            configuration: {
                "test_mean": float(np.mean(scores)),
                "test_std": float(np.std(scores)),
                "mean_paired_accuracy_points_vs_baseline": float(
                    100
                    * np.mean(
                        [
                            by_seed[(seed, configuration)] - baseline[seed]
                            for seed in baseline
                            if (seed, configuration) in by_seed
                        ]
                    )
                ),
                "runs": len(scores),
            }
            for configuration, scores in grouped.items()
        }
    stability_path = args.output_dir / "adult_split_stability.csv"
    if stability_path.exists():
        rows = _float_rows(stability_path)
        grouped: dict[str, list[float]] = {}
        paired: dict[tuple[int, int, str], float] = {}
        selections: Counter[str] = Counter()
        for row in rows:
            configuration = str(row["configuration"])
            score = float(row["test_score"])
            grouped.setdefault(configuration, []).append(score)
            key = (int(row["split_seed"]), int(row["model_seed"]), configuration)
            paired[key] = score
            if configuration == "diagnostic_identity":
                selections[str(row["selected_columns"])] += 1
        differences: dict[str, list[float]] = {}
        for (split_seed, model_seed, configuration), score in paired.items():
            if configuration == "baseline_ple":
                continue
            baseline_key = (split_seed, model_seed, "baseline_ple")
            if baseline_key in paired:
                differences.setdefault(configuration, []).append(
                    100 * (score - paired[baseline_key])
                )
        summary["split_stability"] = {
            "scores": {
                configuration: {
                    "test_mean": float(np.mean(scores)),
                    "test_std": float(np.std(scores)),
                    "runs": len(scores),
                }
                for configuration, scores in grouped.items()
            },
            "paired_accuracy_points_vs_baseline": {
                configuration: {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "wins": int(sum(value > 0 for value in values)),
                    "comparisons": len(values),
                }
                for configuration, values in differences.items()
            },
            "diagnostic_selection_counts": dict(selections),
        }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DAY1 / "data")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("profile", "attribution", "stability"),
        default=("profile", "attribution", "stability"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--attribution-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--split-seeds",
        nargs="+",
        type=int,
        default=[20260825, 20260826, 20260827, 20260828, 20260829],
    )
    parser.add_argument("--stability-model-seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if "profile" in args.stages:
        write_residual_profile(args)
    if "attribution" in args.stages:
        run_attribution(args, device)
    if "stability" in args.stages:
        run_stability(args, device)
    summarize(args)


if __name__ == "__main__":
    main()
