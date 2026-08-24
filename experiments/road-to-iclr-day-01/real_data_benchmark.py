"""Focused real-data test of multi-view feature semantics.

The script consumes the public preprocessed arrays and official train/validation/
test splits released with TabPack. It intentionally uses one fixed MLP recipe for
all methods: only the feature views change.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PRIMARY_DATASETS = ("adult", "diamond", "black-friday", "california")
ALL_DATASETS = ("churn",) + PRIMARY_DATASETS
TRAIN_METHODS = (
    "schema",
    "schema_ple",
    "numeric_identity",
    "cat_target",
    "cat_residual",
    "multi_view",
    "multi_view_residual",
    "diagnostic_identity",
    "diagnostic_residual",
    "gated_ple",
    "sparse_gate",
)
METHODS = TRAIN_METHODS + ("late_fusion",)
BIN_OPTIONS = (8, 16, 32, 64, 128)
BATCH_SIZES = {
    "churn": 128,
    "adult": 256,
    "diamond": 512,
    "black-friday": 512,
    "california": 256,
}


@dataclass
class Dataset:
    name: str
    task: Literal["binclass", "regression"]
    x_num: dict[str, np.ndarray] | None
    x_bin: dict[str, np.ndarray] | None
    x_cat: dict[str, np.ndarray] | None
    y: dict[str, np.ndarray]


@dataclass
class EncodedDataset:
    x: dict[str, np.ndarray]
    y: dict[str, np.ndarray]
    task: Literal["binclass", "regression"]
    y_mean: float
    y_scale: float
    view_names: tuple[str, ...]
    view_sizes: tuple[int, ...]
    selected_numeric: tuple[int, ...]


@dataclass
class TrainOutput:
    result: dict[str, str | float | int]
    val_prediction: np.ndarray
    test_prediction: np.ndarray


def _load_optional(path: Path) -> np.ndarray | None:
    return np.load(path) if path.exists() else None


def load_dataset(root: Path, name: str) -> Dataset:
    directory = root / name
    if not directory.exists():
        raise FileNotFoundError(f"Missing {directory}. Run prepare_tabpack_data.py first.")
    info = json.loads((directory / "info.json").read_text())
    task = info["task"]["type"]
    if task not in ("binclass", "regression"):
        raise ValueError(f"This focused benchmark does not support task {task!r}")
    indices = {
        part: np.load(directory / "splits" / "default" / f"{part}.npy")
        for part in ("train", "val", "test")
    }

    def split(array: np.ndarray | None) -> dict[str, np.ndarray] | None:
        return None if array is None else {part: array[index] for part, index in indices.items()}

    target = split(np.load(directory / "y.npy"))
    assert target is not None
    return Dataset(
        name=name,
        task=task,
        x_num=split(_load_optional(directory / "x_num.npy")),
        x_bin=split(_load_optional(directory / "x_bin.npy")),
        x_cat=split(_load_optional(directory / "x_cat.npy")),
        y=target,
    )


def _clean_numeric(parts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    train = parts["train"].astype(np.float64, copy=True)
    medians = np.nanmedian(train, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    output = {}
    for part, values in parts.items():
        values = values.astype(np.float64, copy=True)
        row, column = np.where(~np.isfinite(values))
        values[row, column] = medians[column]
        output[part] = values
    return output


def _piecewise_linear(parts: dict[str, np.ndarray], bins: int) -> dict[str, np.ndarray]:
    """Cumulative piecewise-linear basis with train-only quantile knots."""

    train = parts["train"]
    transformed = {part: [] for part in parts}
    for column in range(train.shape[1]):
        knots = np.unique(np.quantile(train[:, column], np.linspace(0.0, 1.0, bins + 1)))
        if len(knots) < 2:
            knots = np.array([knots[0], knots[0] + 1.0])
        left, right = knots[:-1], knots[1:]
        width = np.maximum(right - left, 1e-12)
        for part, values in parts.items():
            transformed[part].append(
                np.clip(
                    (values[:, column, None] - left[None, :]) / width[None, :],
                    0.0,
                    1.0,
                )
            )
    return {
        part: np.column_stack(columns).astype(np.float32)
        for part, columns in transformed.items()
    }


def _one_hot(parts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    encoder.fit(parts["train"])
    return {part: encoder.transform(values) for part, values in parts.items()}


def _target_map(
    values: np.ndarray,
    target: np.ndarray,
    query: np.ndarray,
    prior: float,
    smoothing: float,
) -> np.ndarray:
    keys, inverse = np.unique(values, return_inverse=True)
    sums = np.bincount(inverse, weights=target)
    counts = np.bincount(inverse)
    means = (sums + smoothing * prior) / (counts + smoothing)
    positions = np.searchsorted(keys, query)
    valid = positions < len(keys)
    matched = np.zeros(len(query), dtype=bool)
    matched[valid] = keys[positions[valid]] == query[valid]
    output = np.full(len(query), prior, dtype=np.float64)
    output[matched] = means[positions[matched]]
    return output


def _cross_fitted_target_views(
    parts: dict[str, np.ndarray],
    target: np.ndarray,
    task: str,
    seed: int,
    folds: int,
    smoothing: float,
) -> dict[str, np.ndarray]:
    train_values = parts["train"]
    output = {
        part: np.zeros((len(values), values.shape[1]), dtype=np.float64)
        for part, values in parts.items()
    }
    splitter = (
        StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        if task == "binclass"
        else KFold(n_splits=folds, shuffle=True, random_state=seed)
    )
    split_target = target if task == "binclass" else None
    fold_indices = list(splitter.split(train_values, split_target))
    for column in range(train_values.shape[1]):
        for fit_index, holdout_index in fold_indices:
            fold_target = target[fit_index]
            prior = float(fold_target.mean())
            output["train"][holdout_index, column] = _target_map(
                train_values[fit_index, column],
                fold_target,
                train_values[holdout_index, column],
                prior,
                smoothing,
            )
            # Average fold-specific maps for validation and test. This keeps their
            # target views on the same 80%-fit distribution as the OOF train view.
            for part in ("val", "test"):
                output[part][:, column] += _target_map(
                    train_values[fit_index, column],
                    fold_target,
                    parts[part][:, column],
                    prior,
                    smoothing,
                ) / len(fold_indices)
    scaler = StandardScaler().fit(output["train"])
    return {part: scaler.transform(values) for part, values in output.items()}


def _cross_fitted_numeric_residuals(
    numeric_views: dict[str, np.ndarray],
    target: np.ndarray,
    task: str,
    seed: int,
    folds: int = 5,
) -> np.ndarray:
    """OOF residuals from a numerical-only linear model."""

    splitter = (
        StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + 10_000)
        if task == "binclass"
        else KFold(n_splits=folds, shuffle=True, random_state=seed + 10_000)
    )
    split_target = target if task == "binclass" else None
    prediction = np.zeros(len(target), dtype=np.float64)
    for fit_index, holdout_index in splitter.split(numeric_views["train"], split_target):
        if task == "binclass":
            model = LogisticRegression(C=1.0, max_iter=500)
            model.fit(numeric_views["train"][fit_index], target[fit_index])
            prediction[holdout_index] = model.predict_proba(
                numeric_views["train"][holdout_index]
            )[:, 1]
        else:
            model = Ridge(alpha=10.0)
            model.fit(numeric_views["train"][fit_index], target[fit_index])
            prediction[holdout_index] = model.predict(numeric_views["train"][holdout_index])
    return target - prediction


def _select_numeric_identity_features(
    numeric: np.ndarray,
    residual: np.ndarray,
    max_cardinality: int,
    effect_threshold: float,
) -> tuple[int, ...]:
    """Select coded numbers with stable level effects left after numerical PLE."""

    total_variance = float(np.sum((residual - residual.mean()) ** 2)) + 1e-12
    selected = []
    for column in range(numeric.shape[1]):
        values, inverse, counts = np.unique(
            numeric[:, column], return_inverse=True, return_counts=True
        )
        if len(values) > max_cardinality:
            continue
        means = np.bincount(inverse, weights=residual) / counts
        between_variance = float(np.sum(counts * (means - residual.mean()) ** 2))
        if between_variance / total_variance >= effect_threshold:
            selected.append(column)
    return tuple(selected)


def encode_dataset(
    dataset: Dataset,
    method: str,
    seed: int,
    bins: int,
    low_cardinality: int,
    smoothing: float,
    identity_effect_threshold: float,
    cache: dict[str, object] | None = None,
) -> EncodedDataset:
    cache = {} if cache is None else cache
    y_mean = float(dataset.y["train"].mean()) if dataset.task == "regression" else 0.0
    y_scale = float(dataset.y["train"].std()) if dataset.task == "regression" else 1.0
    y_scale = y_scale or 1.0
    y = {
        part: ((values.astype(np.float32) - y_mean) / y_scale)
        for part, values in dataset.y.items()
    }
    schema_components: dict[str, list[np.ndarray]] = {part: [] for part in dataset.y}
    view_columns: dict[str, list[np.ndarray]] = {part: [] for part in dataset.y}
    view_names: list[str] = []
    view_sizes: list[int] = []

    def add_view(name: str, parts: dict[str, np.ndarray]) -> None:
        view_names.append(name)
        view_sizes.append(parts["train"].shape[1])
        for part in view_columns:
            view_columns[part].append(parts[part].astype(np.float32))

    clean_num = None
    normalized_num = None
    ple = None
    if dataset.x_num is not None:
        if "clean_num" not in cache:
            cache["clean_num"] = _clean_numeric(dataset.x_num)
        clean_num = cast(dict[str, np.ndarray], cache["clean_num"])
        if "normalized_num" not in cache:
            quantiles = max(min(len(clean_num["train"]) // 30, 1000), 10)
            normalizer = QuantileTransformer(
                n_quantiles=quantiles,
                output_distribution="normal",
                subsample=1_000_000_000,
                random_state=seed,
            ).fit(clean_num["train"])
            cache["normalized_num"] = {
                part: normalizer.transform(values).astype(np.float32)
                for part, values in clean_num.items()
            }
        normalized_num = cast(dict[str, np.ndarray], cache["normalized_num"])
        for part in schema_components:
            schema_components[part].append(normalized_num[part])

        if method != "schema":
            if "numeric_ple" not in cache:
                cache["numeric_ple"] = _piecewise_linear(clean_num, bins)
            ple = cast(dict[str, np.ndarray], cache["numeric_ple"])

    clean_bin = None
    if dataset.x_bin is not None:
        if "clean_bin" not in cache:
            cache["clean_bin"] = _clean_numeric(dataset.x_bin)
        clean_bin = cast(dict[str, np.ndarray], cache["clean_bin"])
        for part in schema_components:
            schema_components[part].append(clean_bin[part].astype(np.float32))

    if dataset.x_cat is not None:
        if "cat_identity" not in cache:
            cache["cat_identity"] = _one_hot(dataset.x_cat)
        identity = cast(dict[str, np.ndarray], cache["cat_identity"])
        for part in schema_components:
            schema_components[part].append(identity[part])

    schema = {
        part: np.column_stack(values).astype(np.float32)
        for part, values in schema_components.items()
    }
    add_view("schema", schema)
    if ple is not None:
        add_view("numeric_ple", ple)

    residual = None
    needs_residual = method in (
        "cat_residual",
        "multi_view_residual",
        "diagnostic_identity",
        "diagnostic_residual",
        "sparse_gate",
    )
    needs_diagnostic = method in (
        "diagnostic_identity",
        "diagnostic_residual",
        "sparse_gate",
    )
    if needs_residual or needs_diagnostic:
        if "numeric_residual" not in cache:
            numeric_components = []
            if normalized_num is not None:
                numeric_components.extend((normalized_num["train"], ple["train"]))
            if clean_bin is not None:
                numeric_components.append(clean_bin["train"].astype(np.float32))
            numeric_train = np.column_stack(numeric_components).astype(np.float32)
            cache["numeric_residual"] = _cross_fitted_numeric_residuals(
                {"train": numeric_train},
                y["train"].astype(np.float64),
                dataset.task,
                seed,
            )
        residual = cast(np.ndarray, cache["numeric_residual"])

    selected_numeric: tuple[int, ...] = ()
    uses_numeric_identity = method in (
        "numeric_identity",
        "multi_view",
        "multi_view_residual",
        "diagnostic_identity",
        "diagnostic_residual",
        "sparse_gate",
    )
    if uses_numeric_identity and clean_num is not None:
        if needs_diagnostic:
            assert residual is not None
            selected_numeric = _select_numeric_identity_features(
                clean_num["train"],
                residual,
                low_cardinality,
                identity_effect_threshold,
            )
        else:
            selected_numeric = tuple(
                column
                for column in range(clean_num["train"].shape[1])
                if len(np.unique(clean_num["train"][:, column])) <= low_cardinality
            )
        if selected_numeric:
            identity_key = f"numeric_identity:{selected_numeric}"
            if identity_key not in cache:
                cache[identity_key] = _one_hot(
                    {
                        part: values[:, selected_numeric]
                        for part, values in clean_num.items()
                    }
                )
            numeric_identity = cast(dict[str, np.ndarray], cache[identity_key])
            add_view("numeric_identity", numeric_identity)

    uses_raw_target = method in ("cat_target", "multi_view")
    uses_residual_target = method in (
        "cat_residual",
        "multi_view_residual",
        "diagnostic_residual",
        "sparse_gate",
    )
    if dataset.x_cat is not None and (uses_raw_target or uses_residual_target):
        target_for_view = y["train"].astype(np.float64)
        target_name = "cat_target"
        if uses_residual_target:
            assert residual is not None
            target_for_view = residual
            target_name = "cat_residual"
        target_key = "raw_target_views" if uses_raw_target else "residual_target_views"
        if target_key not in cache:
            cache[target_key] = _cross_fitted_target_views(
                dataset.x_cat,
                target_for_view,
                dataset.task if uses_raw_target else "regression",
                seed,
                folds=5,
                smoothing=smoothing,
            )
        target_views = cast(dict[str, np.ndarray], cache[target_key])
        target_ple_key = f"{target_key}_ple"
        if target_ple_key not in cache:
            cache[target_ple_key] = _piecewise_linear(target_views, bins)
        target_ple = cast(dict[str, np.ndarray], cache[target_ple_key])
        add_view(target_name, target_views)
        add_view(f"{target_name}_ple", target_ple)

    x = {
        part: np.ascontiguousarray(np.column_stack(values), dtype=np.float32)
        for part, values in view_columns.items()
    }
    return EncodedDataset(
        x=x,
        y=y,
        task=dataset.task,
        y_mean=y_mean,
        y_scale=y_scale,
        view_names=tuple(view_names),
        view_sizes=tuple(view_sizes),
        selected_numeric=selected_numeric,
    )


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unknown activation: {name}")


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        width: int,
        depth: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_size
        for _ in range(depth):
            layers.extend(
                (nn.Linear(current, width), _activation(activation), nn.Dropout(dropout))
            )
            current = width
        layers.append(nn.Linear(current, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float, activation: str) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.block(inputs)


class TabularResNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        width: int,
        depth: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        self.input = nn.Linear(input_size, width)
        self.blocks = nn.Sequential(
            *(ResidualBlock(width, dropout, activation) for _ in range(depth))
        )
        self.output = nn.Sequential(nn.LayerNorm(width), _activation(activation), nn.Linear(width, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(self.blocks(self.input(inputs))).squeeze(-1)


class SparseGatedMLP(nn.Module):
    """Project view groups to one width, then learn a sparse mixture."""

    def __init__(
        self,
        view_sizes: tuple[int, ...],
        width: int,
        depth: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        self.view_sizes = view_sizes
        self.projections = nn.ModuleList(nn.Linear(size, width) for size in view_sizes)
        initial_logits = torch.full((len(view_sizes),), -2.0)
        initial_logits[: min(2, len(view_sizes))] = 2.0
        self.gate_logits = nn.Parameter(initial_logits)
        layers: list[nn.Module] = []
        for _ in range(max(depth - 1, 0)):
            layers.extend(
                (nn.Linear(width, width), _activation(activation), nn.Dropout(dropout))
            )
        layers.append(nn.Linear(width, 1))
        self.backbone = nn.Sequential(*layers)

    def gate_weights(self) -> torch.Tensor:
        return torch.softmax(self.gate_logits, dim=0)

    def gate_entropy(self) -> torch.Tensor:
        weights = self.gate_weights()
        return -(weights * torch.log(weights.clamp_min(1e-12))).sum()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        views = torch.split(inputs, self.view_sizes, dim=-1)
        projected = torch.stack(
            [torch.relu(layer(view)) for layer, view in zip(self.projections, views)],
            dim=1,
        )
        mixed = torch.sum(projected * self.gate_weights()[None, :, None], dim=1)
        return self.backbone(mixed).squeeze(-1)


def _make_model(
    data: EncodedDataset,
    width: int,
    depth: int,
    dropout: float,
    gated: bool,
    model_type: str = "mlp",
    activation: str = "relu",
) -> nn.Module:
    if gated:
        return SparseGatedMLP(data.view_sizes, width, depth, dropout, activation)
    if model_type == "mlp":
        return MLP(data.x["train"].shape[1], width, depth, dropout, activation)
    if model_type == "resnet":
        return TabularResNet(
            data.x["train"].shape[1], width, depth, dropout, activation
        )
    raise ValueError(f"Unknown model type: {model_type}")


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _matched_width(
    data: EncodedDataset,
    target_parameters: int,
    depth: int,
    dropout: float,
    gated: bool,
    model_type: str = "mlp",
    activation: str = "relu",
) -> int:
    low, high = 16, 1024
    while low < high:
        middle = (low + high) // 2
        count = _parameter_count(
            _make_model(data, middle, depth, dropout, gated, model_type, activation)
        )
        if count < target_parameters:
            low = middle + 1
        else:
            high = middle
    candidates = (max(16, low - 1), low)
    return min(
        candidates,
        key=lambda width: abs(
            _parameter_count(
                _make_model(data, width, depth, dropout, gated, model_type, activation)
            )
            - target_parameters
        ),
    )


def _metric(task: str, prediction: np.ndarray, target: np.ndarray) -> float:
    if task == "binclass":
        return float(((prediction >= 0.0) == target).mean())
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def train_one(
    data: EncodedDataset,
    seed: int,
    batch_size: int,
    device: torch.device,
    width: int,
    depth: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    gated: bool,
    gate_entropy_weight: float,
    target_parameters: int | None,
    model_type: str = "mlp",
    activation: str = "relu",
) -> TrainOutput:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    if target_parameters is not None:
        width = _matched_width(
            data,
            target_parameters,
            depth,
            dropout,
            gated,
            model_type,
            activation,
        )
    model = _make_model(
        data, width, depth, dropout, gated, model_type, activation
    ).to(device)
    parameter_count = _parameter_count(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn: nn.Module = nn.BCEWithLogitsLoss() if data.task == "binclass" else nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(data.x["train"]), torch.from_numpy(data.y["train"])),
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
            loss = loss_fn(model(features), target)
            if isinstance(model, SparseGatedMLP):
                loss = loss + gate_entropy_weight * model.gate_entropy()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            val_logits = model(evaluation["val"][0])
            val_loss = float(loss_fn(val_logits, evaluation["val"][1]).item())
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
        if stale_epochs > patience:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        val_prediction = model(evaluation["val"][0]).cpu().numpy()
        test_prediction = model(evaluation["test"][0]).cpu().numpy()
    val_task_score = _metric(data.task, val_prediction, data.y["val"])
    if data.task == "regression":
        test_prediction_report = test_prediction * data.y_scale + data.y_mean
        test_target = data.y["test"] * data.y_scale + data.y_mean
        test_score = _metric(data.task, test_prediction_report, test_target)
        val_score_report = val_task_score * data.y_scale
    else:
        test_score = _metric(data.task, test_prediction, data.y["test"])
        val_score_report = val_task_score
    gate_summary = ""
    if isinstance(model, SparseGatedMLP):
        weights = model.gate_weights().detach().cpu().numpy()
        gate_summary = ";".join(
            f"{name}:{weight:.4f}" for name, weight in zip(data.view_names, weights)
        )
    return TrainOutput(
        result={
            "input_features": data.x["train"].shape[1],
            "parameters": parameter_count,
            "width": width,
            "best_epoch": best_epoch,
            "val_score": val_score_report,
            "test_score": test_score,
            "train_seconds": time.perf_counter() - started,
            "selected_numeric": ";".join(map(str, data.selected_numeric)),
            "members": gate_summary,
            "model": "gated" if gated else model_type,
            "activation": activation,
        },
        val_prediction=val_prediction,
        test_prediction=test_prediction,
    )


def _prediction_loss(task: str, prediction: np.ndarray, target: np.ndarray) -> float:
    if task == "binclass":
        return float(np.mean(np.logaddexp(0.0, prediction) - target * prediction))
    return float(np.mean((prediction - target) ** 2))


def late_fusion(
    data: EncodedDataset,
    runs: dict[str, TrainOutput],
) -> TrainOutput:
    """Greedily select preprocessing models using validation predictions."""

    candidate_names = [
        name for name in ("schema_ple", "numeric_identity", "cat_residual") if name in runs
    ]
    if not candidate_names:
        raise ValueError("late_fusion requires at least one component run")
    selected: list[str] = []
    remaining = candidate_names.copy()
    val_prediction = np.zeros_like(next(iter(runs.values())).val_prediction)
    test_prediction = np.zeros_like(next(iter(runs.values())).test_prediction)
    best_loss = math.inf
    while remaining:
        best_candidate = None
        best_candidate_loss = best_loss
        for name in remaining:
            count = len(selected)
            proposal = (val_prediction * count + runs[name].val_prediction) / (count + 1)
            loss = _prediction_loss(data.task, proposal, data.y["val"])
            if loss < best_candidate_loss:
                best_candidate = name
                best_candidate_loss = loss
        if best_candidate is None:
            break
        count = len(selected)
        val_prediction = (
            val_prediction * count + runs[best_candidate].val_prediction
        ) / (count + 1)
        test_prediction = (
            test_prediction * count + runs[best_candidate].test_prediction
        ) / (count + 1)
        selected.append(best_candidate)
        remaining.remove(best_candidate)
        best_loss = best_candidate_loss

    val_score = _metric(data.task, val_prediction, data.y["val"])
    if data.task == "regression":
        test_score = _metric(
            data.task,
            test_prediction * data.y_scale + data.y_mean,
            data.y["test"] * data.y_scale + data.y_mean,
        )
        val_score *= data.y_scale
    else:
        test_score = _metric(data.task, test_prediction, data.y["test"])
    return TrainOutput(
        result={
            "input_features": sum(int(runs[name].result["input_features"]) for name in selected),
            "parameters": sum(int(runs[name].result["parameters"]) for name in selected),
            "width": 0,
            "best_epoch": 0,
            "val_score": val_score,
            "test_score": test_score,
            "train_seconds": sum(float(runs[name].result["train_seconds"]) for name in selected),
            "selected_numeric": "",
            "members": ";".join(selected),
        },
        val_prediction=val_prediction,
        test_prediction=test_prediction,
    )


def write_rows(path: Path, rows: list[dict[str, str | float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("data"))
    parser.add_argument("--datasets", nargs="+", choices=ALL_DATASETS, default=PRIMARY_DATASETS)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("real_results.csv"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bins", nargs="+", type=int, choices=BIN_OPTIONS, default=[16])
    parser.add_argument("--low-cardinality", type=int, default=32)
    parser.add_argument("--identity-effect-threshold", type=float, default=0.01)
    parser.add_argument("--smoothing", nargs="+", type=float, default=[20.0])
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--model", choices=("mlp", "resnet"), default="mlp")
    parser.add_argument("--activation", choices=("relu", "gelu", "silu"), default="relu")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--gate-entropy-weight", type=float, default=1e-3)
    parser.add_argument(
        "--match-parameters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match each single model to the schema+PLE parameter budget.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    rows: list[dict[str, str | float | int]] = []
    for dataset_name in args.datasets:
        dataset = load_dataset(args.data, dataset_name)
        for bins in args.bins:
            for smoothing in args.smoothing:
                for seed in args.seeds:
                    encoded_cache: dict[str, EncodedDataset] = {}
                    preprocessing_cache: dict[str, object] = {}

                    def get_encoded(method: str) -> EncodedDataset:
                        if method not in encoded_cache:
                            encoded_cache[method] = encode_dataset(
                                dataset,
                                method,
                                seed,
                                bins,
                                args.low_cardinality,
                                smoothing,
                                args.identity_effect_threshold,
                                preprocessing_cache,
                            )
                        return encoded_cache[method]

                    reference = get_encoded("schema_ple")
                    target_parameters = (
                        _parameter_count(
                            _make_model(
                                reference,
                                args.width,
                                args.depth,
                                args.dropout,
                                False,
                                args.model,
                                args.activation,
                            )
                        )
                        if args.match_parameters
                        else None
                    )
                    train_methods = [method for method in args.methods if method in TRAIN_METHODS]
                    if "late_fusion" in args.methods:
                        for dependency in ("schema_ple", "numeric_identity", "cat_residual"):
                            if dependency not in train_methods:
                                train_methods.append(dependency)

                    runs: dict[str, TrainOutput] = {}
                    for method in train_methods:
                        encoded = get_encoded(method)
                        output = train_one(
                            encoded,
                            seed,
                            BATCH_SIZES[dataset_name],
                            device,
                            args.width,
                            args.depth,
                            args.dropout,
                            args.learning_rate,
                            args.weight_decay,
                            args.max_epochs,
                            args.patience,
                            gated=method in ("gated_ple", "sparse_gate"),
                            gate_entropy_weight=args.gate_entropy_weight,
                            target_parameters=target_parameters,
                            model_type=args.model,
                            activation=args.activation,
                        )
                        runs[method] = output
                        if method not in args.methods:
                            continue
                        row = {
                            "dataset": dataset_name,
                            "task": dataset.task,
                            "metric": "accuracy" if dataset.task == "binclass" else "rmse",
                            "method": method,
                            "bins": bins,
                            "smoothing": smoothing,
                            "seed": seed,
                            **output.result,
                        }
                        rows.append(row)
                        write_rows(args.output, rows)
                        print(
                            f"{dataset_name:<13} bins={bins:<3} {method:<24} seed={seed} "
                            f"test={output.result['test_score']:.5g} "
                            f"epoch={output.result['best_epoch']}"
                        )

                    if "late_fusion" in args.methods:
                        output = late_fusion(reference, runs)
                        row = {
                            "dataset": dataset_name,
                            "task": dataset.task,
                            "metric": "accuracy" if dataset.task == "binclass" else "rmse",
                            "method": "late_fusion",
                            "bins": bins,
                            "smoothing": smoothing,
                            "seed": seed,
                            **output.result,
                        }
                        rows.append(row)
                        write_rows(args.output, rows)
                        print(
                            f"{dataset_name:<13} bins={bins:<3} {'late_fusion':<24} seed={seed} "
                            f"test={output.result['test_score']:.5g} "
                            f"members={output.result['members']}"
                        )


if __name__ == "__main__":
    main()
