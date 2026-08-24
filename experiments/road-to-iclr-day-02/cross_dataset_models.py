"""Test exact-value residual views across datasets and neural backbones.

The experiment compares five representations while holding the backbone and its
parameter budget fixed:

* ``baseline_ple``: schema features plus numerical PLE;
* ``variance_identity``: full identity views chosen by the Day 1 residual-
  variance diagnostic;
* ``utility_identity``: full identity views chosen by a leakage-safe,
  cross-fitted predictive-utility diagnostic;
* ``utility_top8``: indicators for only the eight most frequent values of each
  utility-selected column;
* ``all_identity``: full identity views for every numerical column with at most
  ``max_cardinality`` training values.

Feature selection uses only the training partition. Test predictions are used
only for final reporting. Runs are resumable and appended to a CSV after every
fit.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


HERE = Path(__file__).resolve().parent
DAY1 = HERE.parent / "road-to-iclr-day-01"
sys.path.insert(0, str(DAY1))

import real_data_benchmark as benchmark  # noqa: E402


DATASETS = (
    "churn",
    "adult",
    "diamond",
    "black-friday",
    "california",
    "higgs-small",
    "house",
    "microsoft",
)
MODELS = ("mlp", "resnet", "tabm")
REPRESENTATIONS = (
    "baseline_ple",
    "variance_identity",
    "utility_identity",
    "utility_top8",
    "all_identity",
)


@dataclass(frozen=True)
class ModelConfig:
    width: int
    depth: int
    dropout: float
    activation: str
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


MODEL_CONFIGS = {
    "mlp": ModelConfig(384, 3, 0.1, "gelu"),
    "resnet": ModelConfig(384, 2, 0.0, "gelu"),
    # These match the official TabM defaults except for a smaller ensemble size.
    "tabm": ModelConfig(512, 3, 0.1, "relu"),
}


def batch_size(dataset_name: str) -> int:
    return benchmark.BATCH_SIZES.get(dataset_name, 512)


def _text(columns: Iterable[int]) -> str:
    return ";".join(map(str, columns))


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _append_view(
    base: benchmark.EncodedDataset,
    parts: dict[str, np.ndarray],
    name: str,
    columns: tuple[int, ...],
) -> benchmark.EncodedDataset:
    if not columns or parts["train"].shape[1] == 0:
        return base
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


def _full_identity(
    clean: dict[str, np.ndarray], columns: tuple[int, ...]
) -> dict[str, np.ndarray]:
    if not columns:
        return {
            part: np.empty((len(values), 0), dtype=np.float32)
            for part, values in clean.items()
        }
    return benchmark._one_hot(
        {part: values[:, columns] for part, values in clean.items()}
    )


def _top_k_identity(
    clean: dict[str, np.ndarray], columns: tuple[int, ...], top_k: int
) -> tuple[dict[str, np.ndarray], dict[int, list[float]]]:
    selected_values: dict[int, list[float]] = {}
    for column in columns:
        values, counts = np.unique(clean["train"][:, column], return_counts=True)
        order = np.lexsort((values, -counts))[:top_k]
        selected_values[column] = [float(value) for value in values[order]]
    parts = {
        part: np.column_stack(
            [
                (values[:, column] == value).astype(np.float32)
                for column in columns
                for value in selected_values[column]
            ]
        )
        if columns
        else np.empty((len(values), 0), dtype=np.float32)
        for part, values in clean.items()
    }
    return parts, selected_values


def _candidate_columns(numeric: np.ndarray, max_cardinality: int) -> tuple[int, ...]:
    return tuple(
        column
        for column in range(numeric.shape[1])
        if len(np.unique(numeric[:, column])) <= max_cardinality
    )


def _loss(task: str, prediction: np.ndarray, target: np.ndarray) -> float:
    if task == "binclass":
        probability = np.clip(prediction, 1e-6, 1.0 - 1e-6)
        return float(
            np.mean(
                -target * np.log(probability)
                - (1.0 - target) * np.log(1.0 - probability)
            )
        )
    return float(np.mean((target - prediction) ** 2))


def utility_diagnostic(
    dataset: benchmark.Dataset,
    seed: int,
    bins: int,
    max_cardinality: int,
    smoothing: float,
    minimum_relative_gain: float,
    minimum_fold_wins: int,
    cache: dict[str, object],
) -> tuple[tuple[int, ...], dict[int, dict[str, float | int]]]:
    """Select columns whose exact-value correction helps on held-out folds.

    First, a numerical-only PLE linear model produces out-of-fold predictions.
    Then a second set of folds learns smoothed exact-value maps of those OOF
    residuals. A column is selected only when the correction improves aggregate
    held-out loss and wins on enough folds. No validation or test target is used.
    """

    encoded = benchmark.encode_dataset(
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
    numeric = benchmark._clean_numeric(dataset.x_num)["train"]
    target = encoded.y["train"].astype(np.float64)
    residual = np.asarray(cache["numeric_residual"], dtype=np.float64)
    base_prediction = target - residual
    splitter = (
        StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 20_000)
        if dataset.task == "binclass"
        else KFold(n_splits=5, shuffle=True, random_state=seed + 20_000)
    )
    fold_indices = list(
        splitter.split(numeric, target if dataset.task == "binclass" else None)
    )

    selected: list[int] = []
    statistics: dict[int, dict[str, float | int]] = {}
    for column in _candidate_columns(numeric, max_cardinality):
        correction = np.zeros(len(target), dtype=np.float64)
        fold_wins = 0
        for fit_index, holdout_index in fold_indices:
            fit_values = numeric[fit_index, column]
            values, inverse, counts = np.unique(
                fit_values, return_inverse=True, return_counts=True
            )
            sums = np.bincount(inverse, weights=residual[fit_index])
            # Shrink rare exact values toward a zero correction.
            means = sums / (counts + smoothing)
            query = numeric[holdout_index, column]
            positions = np.searchsorted(values, query)
            valid = positions < len(values)
            matched = np.zeros(len(query), dtype=bool)
            matched[valid] = values[positions[valid]] == query[valid]
            fold_correction = np.zeros(len(query), dtype=np.float64)
            fold_correction[matched] = means[positions[matched]]
            correction[holdout_index] = fold_correction
            before = _loss(
                dataset.task, base_prediction[holdout_index], target[holdout_index]
            )
            after = _loss(
                dataset.task,
                base_prediction[holdout_index] + fold_correction,
                target[holdout_index],
            )
            fold_wins += int(after < before)

        baseline_loss = _loss(dataset.task, base_prediction, target)
        corrected_loss = _loss(dataset.task, base_prediction + correction, target)
        relative_gain = (baseline_loss - corrected_loss) / max(baseline_loss, 1e-12)
        statistics[column] = {
            "cardinality": int(len(np.unique(numeric[:, column]))),
            "relative_gain": float(relative_gain),
            "fold_wins": fold_wins,
        }
        if relative_gain >= minimum_relative_gain and fold_wins >= minimum_fold_wins:
            selected.append(column)
    return tuple(selected), statistics


def encode_variants(
    dataset: benchmark.Dataset,
    seed: int,
    bins: int,
    max_cardinality: int,
    smoothing: float,
    minimum_relative_gain: float,
    minimum_fold_wins: int,
) -> tuple[
    dict[str, benchmark.EncodedDataset],
    tuple[int, ...],
    tuple[int, ...],
    dict[int, dict[str, float | int]],
    dict[int, list[float]],
]:
    cache: dict[str, object] = {}
    base = benchmark.encode_dataset(
        dataset,
        "schema_ple",
        seed,
        bins,
        max_cardinality,
        smoothing,
        1e-3,
        cache,
    )
    diagnostic = benchmark.encode_dataset(
        dataset,
        "diagnostic_identity",
        seed,
        bins,
        max_cardinality,
        smoothing,
        1e-3,
        cache,
    )
    utility_columns, utility_statistics = utility_diagnostic(
        dataset,
        seed,
        bins,
        max_cardinality,
        smoothing,
        minimum_relative_gain,
        minimum_fold_wins,
        cache,
    )
    assert dataset.x_num is not None
    clean = benchmark._clean_numeric(dataset.x_num)
    all_columns = _candidate_columns(clean["train"], max_cardinality)
    utility_full = _append_view(
        base,
        _full_identity(clean, utility_columns),
        "utility_identity",
        utility_columns,
    )
    top8_parts, top8_values = _top_k_identity(clean, utility_columns, 8)
    utility_top8 = _append_view(
        base, top8_parts, "utility_top8", utility_columns
    )
    all_identity = _append_view(
        base, _full_identity(clean, all_columns), "all_identity", all_columns
    )
    variants = {
        "baseline_ple": base,
        "variance_identity": diagnostic,
        "utility_identity": utility_full,
        "utility_top8": utility_top8,
        "all_identity": all_identity,
    }
    return (
        variants,
        diagnostic.selected_numeric,
        utility_columns,
        utility_statistics,
        top8_values,
    )


def _extra_metrics(
    dataset: benchmark.Dataset,
    data: benchmark.EncodedDataset,
    output: benchmark.TrainOutput,
) -> dict[str, float]:
    if dataset.task == "binclass":
        target = data.y["test"].astype(np.float64)
        logits = output.test_prediction.astype(np.float64)
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        return {
            "test_loss": _loss("binclass", probability, target),
            "test_auc": float(roc_auc_score(target, probability)),
            "test_mae": float("nan"),
        }
    target = dataset.y["test"].astype(np.float64)
    prediction = (
        output.test_prediction.astype(np.float64) * data.y_scale + data.y_mean
    )
    return {
        "test_loss": float(np.mean((prediction - target) ** 2)),
        "test_auc": float("nan"),
        "test_mae": float(np.mean(np.abs(prediction - target))),
    }


class FlatTabM(nn.Module):
    def __init__(
        self,
        input_size: int,
        width: int,
        depth: int,
        dropout: float,
        activation: str,
        ensemble_size: int,
    ) -> None:
        super().__init__()
        try:
            from tabm import TabM
        except ImportError as error:  # pragma: no cover - environment guidance
            raise RuntimeError(
                "The tabm model requires the official `tabm` package."
            ) from error
        names = {"relu": "ReLU", "gelu": "GELU", "silu": "SiLU"}
        self.model = TabM.make(
            n_num_features=input_size,
            d_out=1,
            d_block=width,
            n_blocks=depth,
            dropout=dropout,
            activation=names[activation],
            k=ensemble_size,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs).squeeze(-1)


def _tabm_parameters(
    data: benchmark.EncodedDataset,
    width: int,
    config: ModelConfig,
    ensemble_size: int,
) -> int:
    return benchmark._parameter_count(
        FlatTabM(
            data.x["train"].shape[1],
            width,
            config.depth,
            config.dropout,
            config.activation,
            ensemble_size,
        )
    )


def _matched_tabm_width(
    data: benchmark.EncodedDataset,
    target_parameters: int,
    config: ModelConfig,
    ensemble_size: int,
) -> int:
    low, high = 16, 1024
    while low < high:
        middle = (low + high) // 2
        if _tabm_parameters(data, middle, config, ensemble_size) < target_parameters:
            low = middle + 1
        else:
            high = middle
    return min(
        (max(16, low - 1), low),
        key=lambda width: abs(
            _tabm_parameters(data, width, config, ensemble_size) - target_parameters
        ),
    )


def train_tabm(
    data: benchmark.EncodedDataset,
    seed: int,
    batch_size: int,
    device: torch.device,
    config: ModelConfig,
    ensemble_size: int,
    max_epochs: int,
    patience: int,
    target_parameters: int,
) -> benchmark.TrainOutput:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    width = _matched_tabm_width(data, target_parameters, config, ensemble_size)
    model = FlatTabM(
        data.x["train"].shape[1],
        width,
        config.depth,
        config.dropout,
        config.activation,
        ensemble_size,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_fn: nn.Module = (
        nn.BCEWithLogitsLoss() if data.task == "binclass" else nn.MSELoss()
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(data.x["train"]), torch.from_numpy(data.y["train"])
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    evaluation = {
        part: (
            torch.from_numpy(data.x[part]).to(device),
            torch.from_numpy(data.y[part]).to(device),
        )
        for part in ("val", "test")
    }

    def ensemble_prediction(logits: torch.Tensor) -> torch.Tensor:
        if data.task == "binclass":
            probability = logits.sigmoid().mean(dim=1).clamp(1e-6, 1.0 - 1e-6)
            return torch.logit(probability)
        return logits.mean(dim=1)

    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale_epochs = 0
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        for features, target in train_loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            member_prediction = model(features)
            loss = loss_fn(
                member_prediction, target[:, None].expand_as(member_prediction)
            )
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            prediction = ensemble_prediction(model(evaluation["val"][0]))
            val_loss = float(loss_fn(prediction, evaluation["val"][1]).item())
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
        if stale_epochs > patience:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        val_prediction = ensemble_prediction(model(evaluation["val"][0])).cpu().numpy()
        test_prediction = ensemble_prediction(model(evaluation["test"][0])).cpu().numpy()
    val_score = benchmark._metric(data.task, val_prediction, data.y["val"])
    if data.task == "regression":
        test_score = benchmark._metric(
            data.task,
            test_prediction * data.y_scale + data.y_mean,
            data.y["test"] * data.y_scale + data.y_mean,
        )
        val_score *= data.y_scale
    else:
        test_score = benchmark._metric(data.task, test_prediction, data.y["test"])
    return benchmark.TrainOutput(
        result={
            "input_features": data.x["train"].shape[1],
            "parameters": benchmark._parameter_count(model),
            "width": width,
            "best_epoch": best_epoch,
            "val_score": val_score,
            "test_score": test_score,
            "train_seconds": time.perf_counter() - started,
            "selected_numeric": _text(data.selected_numeric),
            "members": str(ensemble_size),
            "model": "tabm",
            "activation": config.activation,
        },
        val_prediction=val_prediction,
        test_prediction=test_prediction,
    )


def train_model(
    data: benchmark.EncodedDataset,
    model_name: str,
    seed: int,
    batch_size: int,
    device: torch.device,
    config: ModelConfig,
    ensemble_size: int,
    max_epochs: int,
    patience: int,
    target_parameters: int,
) -> benchmark.TrainOutput:
    if model_name == "tabm":
        return train_tabm(
            data,
            seed,
            batch_size,
            device,
            config,
            ensemble_size,
            max_epochs,
            patience,
            target_parameters,
        )
    return benchmark.train_one(
        data,
        seed,
        batch_size,
        device,
        config.width,
        config.depth,
        config.dropout,
        config.learning_rate,
        config.weight_decay,
        max_epochs,
        patience,
        gated=False,
        gate_entropy_weight=0.0,
        target_parameters=target_parameters,
        model_type=model_name,
        activation=config.activation,
    )


def baseline_parameter_count(
    data: benchmark.EncodedDataset,
    model_name: str,
    config: ModelConfig,
    ensemble_size: int,
) -> int:
    if model_name == "tabm":
        return _tabm_parameters(data, config.width, config, ensemble_size)
    return benchmark._parameter_count(
        benchmark._make_model(
            data,
            config.width,
            config.depth,
            config.dropout,
            False,
            model_name,
            config.activation,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DAY1 / "data")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=REPRESENTATIONS,
        default=REPRESENTATIONS,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--bins", type=int, default=128)
    parser.add_argument("--max-cardinality", type=int, default=128)
    parser.add_argument("--smoothing", type=float, default=20.0)
    parser.add_argument("--minimum-relative-gain", type=float, default=5e-4)
    parser.add_argument("--minimum-fold-wins", type=int, default=3)
    parser.add_argument("--ensemble-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument(
        "--output", type=Path, default=HERE / "results" / "cross_dataset_models.csv"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    rows: list[dict[str, object]] = [] if args.force else list(_read_rows(args.output))
    completed = {
        (str(row["dataset"]), str(row["model"]), int(row["seed"]), str(row["representation"]))
        for row in rows
    }
    for dataset_name in args.datasets:
        dataset = benchmark.load_dataset(args.data, dataset_name)
        for seed in args.seeds:
            (
                variants,
                variance_columns,
                utility_columns,
                utility_statistics,
                top8_values,
            ) = encode_variants(
                dataset,
                seed,
                args.bins,
                args.max_cardinality,
                args.smoothing,
                args.minimum_relative_gain,
                args.minimum_fold_wins,
            )
            print(
                f"{dataset_name} seed={seed} variance={_text(variance_columns) or '-'} "
                f"utility={_text(utility_columns) or '-'}",
                flush=True,
            )
            for model_name in args.models:
                config = MODEL_CONFIGS[model_name]
                parameter_budget = baseline_parameter_count(
                    variants["baseline_ple"], model_name, config, args.ensemble_size
                )
                for representation in args.representations:
                    key = (dataset_name, model_name, seed, representation)
                    if key in completed:
                        continue
                    encoded = variants[representation]
                    output = train_model(
                        encoded,
                        model_name,
                        seed,
                        batch_size(dataset_name),
                        device,
                        config,
                        args.ensemble_size,
                        args.max_epochs,
                        args.patience,
                        parameter_budget,
                    )
                    row: dict[str, object] = {
                        "dataset": dataset_name,
                        "task": dataset.task,
                        "model": model_name,
                        "seed": seed,
                        "representation": representation,
                        "bins": args.bins,
                        "max_cardinality": args.max_cardinality,
                        "variance_columns": _text(variance_columns),
                        "utility_columns": _text(utility_columns),
                        "utility_statistics": json.dumps(
                            utility_statistics, sort_keys=True, separators=(",", ":")
                        ),
                        "top8_values": json.dumps(
                            top8_values, sort_keys=True, separators=(",", ":")
                        ),
                        **output.result,
                        **_extra_metrics(dataset, encoded, output),
                    }
                    rows.append(row)
                    completed.add(key)
                    _write_rows(args.output, rows)
                    print(
                        f"  {model_name:<6} {representation:<20} "
                        f"test={float(output.result['test_score']):.6f} "
                        f"epoch={int(output.result['best_epoch'])} "
                        f"seconds={float(output.result['train_seconds']):.1f}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
