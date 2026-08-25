"""Screen an identity-inspired local PLE basis on every TabPack dataset.

For one numerical column, cumulative PLE emits one ramp per knot interval.  A
linear change of coordinates turns those ramps into compact hat functions:

    h_0 = 1 - p_0,  h_j = p_{j-1} - p_j.

After dropping the final redundant hat, the local and cumulative bases have the
same dimension and, with a bias term, exactly the same span.  They differ only
in geometry: cumulative PLE is dense while the local basis has at most two
active coordinates per column.  This script tests whether that Adult identity
mechanism transfers to continuous columns and multiclass data.

The screen is resumable.  Model selection and blending use validation loss;
the official test target is touched only for final reporting.
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
from typing import Literal

import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parent.parent / "road-to-iclr-day-01" / "data"
DATASETS = (
    "adult",
    "black-friday",
    "california",
    "churn",
    "diamond",
    "higgs-small",
    "house",
    "microsoft",
    "otto",
)
MODELS = ("mlp", "resnet")
TRAINED_REPRESENTATIONS = (
    "cumulative_ple",
    "local_ple",
    "local_ple_energy",
    "dual_ple",
    "cumulative_seedmate",
)
DERIVED_REPRESENTATIONS = (
    "basis_blend",
    "basis_select",
    "energy_basis_blend",
    "energy_basis_select",
    "seed_blend",
    "seed_select",
)
Task = Literal["binclass", "multiclass", "regression"]


@dataclass
class Dataset:
    name: str
    task: Task
    x_num: dict[str, np.ndarray] | None
    x_bin: dict[str, np.ndarray] | None
    x_cat: dict[str, np.ndarray] | None
    y: dict[str, np.ndarray]
    n_classes: int


@dataclass
class Encoded:
    x: dict[str, np.ndarray]
    y: dict[str, np.ndarray]
    task: Task
    n_classes: int
    y_mean: float
    y_scale: float
    basis_features: int


@dataclass
class Fit:
    result: dict[str, object]
    val_prediction: np.ndarray
    test_prediction: np.ndarray


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


def _row_indices(
    size: int, limit: int | None, seed: int
) -> np.ndarray:
    if limit is None or size <= limit:
        return np.arange(size)
    return np.sort(np.random.default_rng(seed).choice(size, limit, replace=False))


def load_dataset(
    root: Path,
    name: str,
    max_train_rows: int | None,
    max_eval_rows: int | None,
    sample_seed: int,
) -> Dataset:
    directory = root / name
    info = json.loads((directory / "info.json").read_text())
    task = info["task"]["type"]
    if task not in ("binclass", "multiclass", "regression"):
        raise ValueError(f"Unsupported task: {task}")
    split_indices = {
        part: np.load(directory / "splits" / "default" / f"{part}.npy")
        for part in ("train", "val", "test")
    }
    limits = {"train": max_train_rows, "val": max_eval_rows, "test": max_eval_rows}
    for offset, part in enumerate(("train", "val", "test")):
        local = _row_indices(len(split_indices[part]), limits[part], sample_seed + offset)
        split_indices[part] = split_indices[part][local]

    def load_optional(stem: str) -> dict[str, np.ndarray] | None:
        path = directory / f"{stem}.npy"
        if not path.exists():
            return None
        array = np.load(path, mmap_mode="r")
        return {part: np.asarray(array[index]) for part, index in split_indices.items()}

    target = load_optional("y")
    assert target is not None
    if task == "multiclass":
        classes = np.unique(target["train"])
        if not all(np.array_equal(np.unique(target[part]), classes) for part in target):
            raise ValueError("Every multiclass partition must contain the train classes")
        target = {
            part: np.searchsorted(classes, values).astype(np.int64)
            for part, values in target.items()
        }
        n_classes = len(classes)
    else:
        target = {part: values.astype(np.float32) for part, values in target.items()}
        n_classes = 2 if task == "binclass" else 1
    return Dataset(
        name=name,
        task=task,
        x_num=load_optional("x_num"),
        x_bin=load_optional("x_bin"),
        x_cat=load_optional("x_cat"),
        y=target,
        n_classes=n_classes,
    )


def _clean_numeric(parts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    train = parts["train"].astype(np.float64, copy=True)
    medians = np.nanmedian(train, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    output: dict[str, np.ndarray] = {}
    for part, source in parts.items():
        values = source.astype(np.float64, copy=True)
        row, column = np.where(~np.isfinite(values))
        values[row, column] = medians[column]
        output[part] = values
    return output


def piecewise_bases(
    parts: dict[str, np.ndarray], bins: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[int]]:
    """Return equal-width cumulative and local coordinates over shared knots."""

    cumulative_columns: dict[str, list[np.ndarray]] = {part: [] for part in parts}
    local_columns: dict[str, list[np.ndarray]] = {part: [] for part in parts}
    intervals: list[int] = []
    train = parts["train"]
    for column in range(train.shape[1]):
        knots = np.unique(
            np.quantile(train[:, column], np.linspace(0.0, 1.0, bins + 1))
        )
        if len(knots) < 2:
            knots = np.array([knots[0], knots[0] + 1.0])
        left, right = knots[:-1], knots[1:]
        width = np.maximum(right - left, 1e-12)
        intervals.append(len(left))
        for part, values in parts.items():
            ramps = np.clip(
                (values[:, column, None] - left[None, :]) / width[None, :],
                0.0,
                1.0,
            ).astype(np.float32)
            # Drop the final redundant hat. Together with a model intercept,
            # these K local columns span exactly the same functions as K ramps.
            local = np.empty_like(ramps)
            local[:, 0] = 1.0 - ramps[:, 0]
            if ramps.shape[1] > 1:
                local[:, 1:] = ramps[:, :-1] - ramps[:, 1:]
            cumulative_columns[part].append(ramps)
            local_columns[part].append(local)
    cumulative = {
        part: np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)
        for part, columns in cumulative_columns.items()
    }
    local = {
        part: np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)
        for part, columns in local_columns.items()
    }
    return cumulative, local, intervals


def energy_match(
    local: dict[str, np.ndarray],
    cumulative: dict[str, np.ndarray],
    intervals: list[int],
) -> tuple[dict[str, np.ndarray], list[float]]:
    """Match each local block's train RMS energy to its cumulative block."""

    output = {part: values.copy() for part, values in local.items()}
    scales: list[float] = []
    start = 0
    for width in intervals:
        stop = start + width
        cumulative_energy = float(
            np.mean(np.sum(cumulative["train"][:, start:stop] ** 2, axis=1))
        )
        local_energy = float(
            np.mean(np.sum(local["train"][:, start:stop] ** 2, axis=1))
        )
        scale = math.sqrt(cumulative_energy / max(local_energy, 1e-12))
        for values in output.values():
            values[:, start:stop] *= scale
        scales.append(scale)
        start = stop
    return output, scales


def encode(dataset: Dataset, bins: int, seed: int) -> dict[str, Encoded]:
    schema_components: dict[str, list[np.ndarray]] = {
        part: [] for part in dataset.y
    }
    cumulative: dict[str, np.ndarray] | None = None
    local: dict[str, np.ndarray] | None = None
    local_energy: dict[str, np.ndarray] | None = None
    intervals: list[int] = []
    if dataset.x_num is not None:
        clean = _clean_numeric(dataset.x_num)
        quantiles = max(min(len(clean["train"]) // 30, 1000), 10)
        transformer = QuantileTransformer(
            n_quantiles=quantiles,
            output_distribution="normal",
            subsample=1_000_000_000,
            random_state=seed,
        ).fit(clean["train"])
        for part, values in clean.items():
            schema_components[part].append(
                transformer.transform(values).astype(np.float32)
            )
        cumulative, local, intervals = piecewise_bases(clean, bins)
        local_energy, _ = energy_match(local, cumulative, intervals)
    if dataset.x_bin is not None:
        binary = _clean_numeric(dataset.x_bin)
        for part, values in binary.items():
            schema_components[part].append(values.astype(np.float32))
    if dataset.x_cat is not None:
        encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, dtype=np.float32
        ).fit(dataset.x_cat["train"])
        for part, values in dataset.x_cat.items():
            schema_components[part].append(encoder.transform(values))
    schema = {
        part: np.ascontiguousarray(np.column_stack(values), dtype=np.float32)
        for part, values in schema_components.items()
    }
    if cumulative is None or local is None or local_energy is None:
        raise ValueError("The local-basis experiment requires numerical columns")
    y_mean = (
        float(dataset.y["train"].mean()) if dataset.task == "regression" else 0.0
    )
    y_scale = (
        float(dataset.y["train"].std()) if dataset.task == "regression" else 1.0
    ) or 1.0
    y = {
        part: (
            ((values.astype(np.float32) - y_mean) / y_scale)
            if dataset.task == "regression"
            else values
        )
        for part, values in dataset.y.items()
    }
    basis_features = int(sum(intervals))

    def make(parts: dict[str, list[np.ndarray]]) -> Encoded:
        return Encoded(
            x={
                part: np.ascontiguousarray(np.column_stack(values), dtype=np.float32)
                for part, values in parts.items()
            },
            y=y,
            task=dataset.task,
            n_classes=dataset.n_classes,
            y_mean=y_mean,
            y_scale=y_scale,
            basis_features=basis_features,
        )

    return {
        "cumulative_ple": make(
            {part: [schema[part], cumulative[part]] for part in schema}
        ),
        "local_ple": make({part: [schema[part], local[part]] for part in schema}),
        "local_ple_energy": make(
            {part: [schema[part], local_energy[part]] for part in schema}
        ),
        "dual_ple": make(
            {
                part: [schema[part], cumulative[part], local[part]]
                for part in schema
            }
        ),
    }


def _activation(name: str) -> nn.Module:
    return {"relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU()}[name]


class MLP(nn.Module):
    def __init__(
        self, input_size: int, output_size: int, width: int, depth: int, dropout: float
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_size
        for _ in range(depth):
            layers.extend((nn.Linear(current, width), nn.GELU(), nn.Dropout(dropout)))
            current = width
        layers.append(nn.Linear(current, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.block(inputs)


class ResNet(nn.Module):
    def __init__(
        self, input_size: int, output_size: int, width: int, depth: int, dropout: float
    ) -> None:
        super().__init__()
        self.input = nn.Linear(input_size, width)
        self.blocks = nn.Sequential(
            *(ResidualBlock(width, dropout) for _ in range(depth))
        )
        self.output = nn.Sequential(nn.LayerNorm(width), nn.GELU(), nn.Linear(width, output_size))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(self.blocks(self.input(inputs)))


def make_model(
    data: Encoded, model_name: str, width: int, depth: int, dropout: float
) -> nn.Module:
    output_size = data.n_classes if data.task == "multiclass" else 1
    cls = MLP if model_name == "mlp" else ResNet
    return cls(data.x["train"].shape[1], output_size, width, depth, dropout)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def matched_width(
    data: Encoded,
    model_name: str,
    target_parameters: int,
    depth: int,
    dropout: float,
) -> int:
    low, high = 16, 1024
    while low < high:
        middle = (low + high) // 2
        count = parameter_count(make_model(data, model_name, middle, depth, dropout))
        if count < target_parameters:
            low = middle + 1
        else:
            high = middle
    return min(
        (max(16, low - 1), low),
        key=lambda width: abs(
            parameter_count(make_model(data, model_name, width, depth, dropout))
            - target_parameters
        ),
    )


def torch_loss(task: Task) -> nn.Module:
    if task == "binclass":
        return nn.BCEWithLogitsLoss()
    if task == "multiclass":
        return nn.CrossEntropyLoss()
    return nn.MSELoss()


def numpy_loss(task: Task, prediction: np.ndarray, target: np.ndarray) -> float:
    if task == "binclass":
        logits = prediction.reshape(-1).astype(np.float64)
        return float(np.mean(np.logaddexp(0.0, logits) - target * logits))
    if task == "multiclass":
        logits = prediction.astype(np.float64)
        maximum = logits.max(axis=1, keepdims=True)
        log_partition = maximum[:, 0] + np.log(
            np.exp(logits - maximum).sum(axis=1)
        )
        return float(np.mean(log_partition - logits[np.arange(len(target)), target]))
    return float(np.mean((prediction.reshape(-1) - target) ** 2))


def score(data: Encoded, prediction: np.ndarray, target: np.ndarray) -> float:
    if data.task == "binclass":
        return float(((prediction.reshape(-1) >= 0.0) == target).mean())
    if data.task == "multiclass":
        return float((prediction.argmax(axis=1) == target).mean())
    return float(
        np.sqrt(np.mean((prediction.reshape(-1) - target) ** 2)) * data.y_scale
    )


def predict_in_batches(
    model: nn.Module, features: np.ndarray, device: torch.device, batch_size: int
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).to(device)
            outputs.append(model(batch).float().cpu().numpy())
    return np.concatenate(outputs)


def train(
    data: Encoded,
    model_name: str,
    seed: int,
    device: torch.device,
    width: int,
    depth: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    target_parameters: int,
) -> Fit:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    width = matched_width(
        data, model_name, target_parameters, depth, dropout
    )
    model = make_model(data, model_name, width, depth, dropout).to(device)
    loss_fn = torch_loss(data.task)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    target_tensor = torch.from_numpy(data.y["train"])
    loader = DataLoader(
        TensorDataset(torch.from_numpy(data.x["train"]), target_tensor),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        for features, target in loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features)
            if data.task != "multiclass":
                prediction = prediction.squeeze(-1)
            loss = loss_fn(prediction, target)
            loss.backward()
            optimizer.step()
        val_prediction = predict_in_batches(model, data.x["val"], device, batch_size * 4)
        val_loss = numpy_loss(data.task, val_prediction, data.y["val"])
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            stale = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        if stale > patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    val_prediction = predict_in_batches(model, data.x["val"], device, batch_size * 4)
    test_prediction = predict_in_batches(model, data.x["test"], device, batch_size * 4)
    return Fit(
        result={
            "input_features": data.x["train"].shape[1],
            "basis_features": data.basis_features,
            "parameters": parameter_count(model),
            "width": width,
            "best_epoch": best_epoch,
            "val_loss": numpy_loss(data.task, val_prediction, data.y["val"]),
            "val_score": score(data, val_prediction, data.y["val"]),
            "test_loss": numpy_loss(data.task, test_prediction, data.y["test"]),
            "test_score": score(data, test_prediction, data.y["test"]),
            "train_seconds": time.perf_counter() - started,
        },
        val_prediction=val_prediction,
        test_prediction=test_prediction,
    )


def derived_fit(
    data: Encoded,
    cumulative: Fit,
    local: Fit,
    representation: str,
) -> tuple[Fit, float]:
    if representation.endswith("_select"):
        alpha = float(
            numpy_loss(data.task, local.val_prediction, data.y["val"])
            < numpy_loss(data.task, cumulative.val_prediction, data.y["val"])
        )
    else:
        grid = np.linspace(0.0, 1.0, 21)
        alpha = min(
            grid,
            key=lambda value: numpy_loss(
                data.task,
                (1.0 - value) * cumulative.val_prediction
                + value * local.val_prediction,
                data.y["val"],
            ),
        )
        alpha = float(alpha)
    val_prediction = (
        (1.0 - alpha) * cumulative.val_prediction + alpha * local.val_prediction
    )
    test_prediction = (
        (1.0 - alpha) * cumulative.test_prediction + alpha * local.test_prediction
    )
    return (
        Fit(
            result={
                "input_features": cumulative.result["input_features"],
                "basis_features": cumulative.result["basis_features"],
                "parameters": int(cumulative.result["parameters"])
                + int(local.result["parameters"]),
                "width": cumulative.result["width"],
                "best_epoch": max(
                    int(cumulative.result["best_epoch"]),
                    int(local.result["best_epoch"]),
                ),
                "val_loss": numpy_loss(data.task, val_prediction, data.y["val"]),
                "val_score": score(data, val_prediction, data.y["val"]),
                "test_loss": numpy_loss(data.task, test_prediction, data.y["test"]),
                "test_score": score(data, test_prediction, data.y["test"]),
                "train_seconds": float(cumulative.result["train_seconds"])
                + float(local.result["train_seconds"]),
            },
            val_prediction=val_prediction,
            test_prediction=test_prediction,
        ),
        alpha,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=["mlp"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--max-train-rows", type=int, default=100_000)
    parser.add_argument("--max-eval-rows", type=int, default=25_000)
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--seed-control-offset", type=int, default=10_000)
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=TRAINED_REPRESENTATIONS,
        default=TRAINED_REPRESENTATIONS,
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output", type=Path, default=HERE / "results" / "local_basis_screen.csv"
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = list(_read_rows(args.output))
    completed = {
        (row["dataset"], row["model"], int(row["seed"]), int(row["bins"]), row["representation"])
        for row in rows
    }
    device = torch.device(args.device)
    for dataset_name in args.datasets:
        dataset = load_dataset(
            args.data,
            dataset_name,
            args.max_train_rows,
            args.max_eval_rows,
            args.sample_seed,
        )
        print(
            f"{dataset_name}: task={dataset.task} rows="
            f"{len(dataset.y['train'])}/{len(dataset.y['val'])}/{len(dataset.y['test'])}",
            flush=True,
        )
        for seed in args.seeds:
            variants = encode(dataset, args.bins, seed)
            for model_name in args.models:
                baseline = variants["cumulative_ple"]
                target_parameters = parameter_count(
                    make_model(baseline, model_name, args.width, args.depth, args.dropout)
                )
                fits: dict[str, Fit] = {}
                required = set(args.representations)
                for representation in TRAINED_REPRESENTATIONS:
                    if representation not in required:
                        continue
                    encoded_name = (
                        "cumulative_ple"
                        if representation == "cumulative_seedmate"
                        else representation
                    )
                    training_seed = (
                        seed + args.seed_control_offset
                        if representation == "cumulative_seedmate"
                        else seed
                    )
                    fit = train(
                        variants[encoded_name],
                        model_name,
                        training_seed,
                        device,
                        args.width,
                        args.depth,
                        args.dropout,
                        args.learning_rate,
                        args.weight_decay,
                        args.batch_size,
                        args.max_epochs,
                        args.patience,
                        target_parameters,
                    )
                    fits[representation] = fit
                    key = (dataset_name, model_name, seed, args.bins, representation)
                    if key not in completed:
                        rows.append(
                            {
                                "dataset": dataset_name,
                                "task": dataset.task,
                                "model": model_name,
                                "seed": seed,
                                "bins": args.bins,
                                "representation": representation,
                                "blend_alpha_local": "",
                                "train_rows": len(dataset.y["train"]),
                                "val_rows": len(dataset.y["val"]),
                                "test_rows": len(dataset.y["test"]),
                                **fit.result,
                            }
                        )
                        completed.add(key)
                        _write_rows(args.output, rows)
                    print(
                        f"  {model_name:<6} {representation:<16} "
                        f"val_loss={float(fit.result['val_loss']):.6f} "
                        f"test={float(fit.result['test_score']):.6f}",
                        flush=True,
                    )
                if "cumulative_ple" in fits and "local_ple" in fits:
                    for representation in ("basis_blend", "basis_select"):
                        fit, alpha = derived_fit(
                            baseline,
                            fits["cumulative_ple"],
                            fits["local_ple"],
                            representation,
                        )
                        key = (dataset_name, model_name, seed, args.bins, representation)
                        if key not in completed:
                            rows.append(
                                {
                                    "dataset": dataset_name,
                                    "task": dataset.task,
                                    "model": model_name,
                                    "seed": seed,
                                    "bins": args.bins,
                                    "representation": representation,
                                    "blend_alpha_local": alpha,
                                    "train_rows": len(dataset.y["train"]),
                                    "val_rows": len(dataset.y["val"]),
                                    "test_rows": len(dataset.y["test"]),
                                    **fit.result,
                                }
                            )
                            completed.add(key)
                            _write_rows(args.output, rows)
                        print(
                            f"  {model_name:<6} {representation:<16} "
                            f"alpha={alpha:.2f} test={float(fit.result['test_score']):.6f}",
                            flush=True,
                        )
                if "cumulative_ple" in fits and "local_ple_energy" in fits:
                    for representation in (
                        "energy_basis_blend",
                        "energy_basis_select",
                    ):
                        fit, alpha = derived_fit(
                            baseline,
                            fits["cumulative_ple"],
                            fits["local_ple_energy"],
                            representation,
                        )
                        key = (dataset_name, model_name, seed, args.bins, representation)
                        if key not in completed:
                            rows.append(
                                {
                                    "dataset": dataset_name,
                                    "task": dataset.task,
                                    "model": model_name,
                                    "seed": seed,
                                    "bins": args.bins,
                                    "representation": representation,
                                    "blend_alpha_local": alpha,
                                    "train_rows": len(dataset.y["train"]),
                                    "val_rows": len(dataset.y["val"]),
                                    "test_rows": len(dataset.y["test"]),
                                    **fit.result,
                                }
                            )
                            completed.add(key)
                            _write_rows(args.output, rows)
                        print(
                            f"  {model_name:<6} {representation:<20} "
                            f"alpha={alpha:.2f} test={float(fit.result['test_score']):.6f}",
                            flush=True,
                        )
                if "cumulative_ple" in fits and "cumulative_seedmate" in fits:
                    for representation in ("seed_blend", "seed_select"):
                        fit, alpha = derived_fit(
                            baseline,
                            fits["cumulative_ple"],
                            fits["cumulative_seedmate"],
                            representation,
                        )
                        key = (dataset_name, model_name, seed, args.bins, representation)
                        if key not in completed:
                            rows.append(
                                {
                                    "dataset": dataset_name,
                                    "task": dataset.task,
                                    "model": model_name,
                                    "seed": seed,
                                    "bins": args.bins,
                                    "representation": representation,
                                    "blend_alpha_local": alpha,
                                    "train_rows": len(dataset.y["train"]),
                                    "val_rows": len(dataset.y["val"]),
                                    "test_rows": len(dataset.y["test"]),
                                    **fit.result,
                                }
                            )
                            completed.add(key)
                            _write_rows(args.output, rows)
                        print(
                            f"  {model_name:<6} {representation:<16} "
                            f"alpha={alpha:.2f} test={float(fit.result['test_score']):.6f}",
                            flush=True,
                        )


if __name__ == "__main__":
    main()
