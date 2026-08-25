"""Optimizer-level remedies for exact equivalent-basis sensitivity.

The screen deliberately keeps the predictor class fixed and compares:

* ordinary AdamW/Adam/AdaGrad/SGD;
* diagonal scaling and full whitening controls;
* a basis-invariant anchor canonicalization layer;
* fixed input-covariance gradient preconditioning of the first affine layer;
* a hybrid natural-gradient first layer with AdamW on all later parameters.

All covariance/canonical transforms are fitted on training rows only.
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

import numpy as np
import torch
from scipy.linalg import block_diag, qr
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .core import (
    MLP,
    PARTS,
    ResNet,
    Prepared,
    apply_transform,
    base_schema,
    combine,
    condition_transform,
    diagonal_standardize,
    geometry,
    load_dataset,
    loss_numpy,
    make_prepared,
    metric,
    ple_blocks,
    whiten,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "day3" / "optimizer_remedies"


@dataclass(frozen=True)
class Remedy:
    name: str
    input_transform: str
    optimizer: str
    learning_rate: float
    first_precondition_power: float = 0.0
    first_learning_rate: float = 0.0
    invariant_initialization: bool = False
    function_matched_initialization: bool = False


REMEDIES = {
    remedy.name: remedy
    for remedy in (
        Remedy("adamw", "raw", "adamw", 1e-3),
        Remedy("adam_no_wd", "raw", "adam", 1e-3),
        Remedy("adagrad", "raw", "adagrad", 1e-2),
        Remedy("sgd_momentum", "raw", "sgd", 3e-2),
        Remedy("diagonal_adamw", "diagonal", "adamw", 1e-3),
        Remedy("whiten_adamw", "whiten", "adamw", 1e-3),
        Remedy("whiten_sgd", "whiten", "sgd", 3e-2),
        Remedy("anchor_canonical_adamw", "anchor", "adamw", 1e-3),
        Remedy("anchor_canonical_sgd", "anchor", "sgd", 3e-2),
        Remedy("anchor_whiten_adamw", "anchor_whiten", "adamw", 1e-3),
        Remedy("anchor_whiten_sgd", "anchor_whiten", "sgd", 3e-2),
        Remedy("cov_invsqrt_hybrid", "raw", "hybrid", 1e-3, 0.5, 1e-2),
        Remedy("natural_hybrid", "raw", "hybrid", 1e-3, 1.0, 3e-2),
        Remedy("natural_hybrid_invariant_init", "raw", "hybrid", 1e-3, 1.0, 3e-2, True),
        Remedy("natural_hybrid_invariant_init_lr01", "raw", "hybrid", 1e-3, 1.0, 1e-2, True),
        Remedy("natural_hybrid_invariant_init_lr003", "raw", "hybrid", 1e-3, 1.0, 3e-3, True),
        Remedy("natural_hybrid_matched_init", "raw", "hybrid", 1e-3, 1.0, 3e-2, False, True),
        Remedy("natural_sgd_all", "raw", "sgd", 3e-2, 1.0),
    )
}


def controlled_numeric(
    dataset_name: str,
    kappa: float,
    bins: int = 32,
    transform_seed: int = 81000,
) -> tuple[object, dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Return dataset, reference coordinates, transformed coordinates, and B.

    The representation intentionally omits the duplicate raw-numeric view so
    the augmented input second moment is full rank.  It retains categorical and
    binary schema blocks plus a full-rank whitened PLE block per numeric field.
    For every split, ``x_kappa = x_reference @ B`` up to float rounding.
    """

    dataset = load_dataset(dataset_name)
    assert dataset.x_num is not None
    schema = base_schema(dataset, include_num=False, include_cat=True)
    ple, _ = ple_blocks(dataset.x_num, bins=bins)
    white_blocks = [whiten(block)[0] for block in ple]
    reference = {
        part: np.ascontiguousarray(
            np.column_stack([schema[part], *[block[part] for block in white_blocks]]),
            dtype=np.float64,
        )
        for part in PARTS
    }
    matrices = [np.eye(schema["train"].shape[1], dtype=np.float64)]
    transformed_blocks = []
    for column, block in enumerate(white_blocks):
        transform = condition_transform(block["train"].shape[1], kappa, transform_seed + column)
        matrices.append(transform)
        transformed_blocks.append(apply_transform(block, transform))
    basis_transform = block_diag(*matrices)
    transformed = {
        part: np.ascontiguousarray(
            np.column_stack([schema[part], *[block[part] for block in transformed_blocks]]),
            dtype=np.float64,
        )
        for part in PARTS
    }
    return dataset, reference, transformed, basis_transform


def anchor_canonicalize(parts: dict[str, np.ndarray], rtol: float = 1e-10) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Create coordinates invariant to any invertible right basis transform.

    Pivot rows are selected from an orthonormal basis of the train column space,
    which depends only on that space. Every row is then represented by its
    unique coefficients in the selected row basis. If ``X' = X A``, those
    coefficients are unchanged because both the row and anchor equations are
    multiplied by the same invertible ``A``.
    """

    train = np.asarray(parts["train"], dtype=np.float64)
    mean = train.mean(axis=0)
    centered = train - mean
    u, singular, _ = np.linalg.svd(centered, full_matrices=False)
    keep = singular > singular[0] * rtol
    q = u[:, keep]
    rank = q.shape[1]
    _, _, pivots = qr(q.T, pivoting=True, mode="economic")
    anchors = np.asarray(pivots[:rank], dtype=np.int64)
    anchor_rows = centered[anchors]
    output = {}
    reconstruction_errors = {}
    for part, values in parts.items():
        centered_part = np.asarray(values, dtype=np.float64) - mean
        coefficients = np.linalg.lstsq(anchor_rows.T, centered_part.T, rcond=rtol)[0].T
        reconstructed = coefficients @ anchor_rows
        reconstruction_errors[part] = float(
            np.linalg.norm(reconstructed - centered_part)
            / max(np.linalg.norm(centered_part), 1e-12)
        )
        output[part] = coefficients
    # Invariant diagonal normalization improves numerical scale without changing
    # the exact canonical relationship.
    output = diagonal_standardize(output)
    return output, {
        "canonical_rank": rank,
        "anchor_rows": anchors.tolist(),
        "reconstruction_errors": reconstruction_errors,
    }


def fit_input_preconditioner(
    train: np.ndarray,
    power: float,
    relative_floor: float = 1e-10,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return ``E[[x,1][x,1]^T]^-power`` for first-layer gradients."""

    augmented = np.column_stack((np.asarray(train, dtype=np.float64), np.ones(len(train))))
    second_moment = augmented.T @ augmented / len(augmented)
    eigen, vectors = np.linalg.eigh(second_moment)
    floor = max(float(eigen[-1]) * relative_floor, 1e-14)
    clipped = np.maximum(eigen, floor)
    preconditioner = (vectors * (clipped ** (-power))[None, :]) @ vectors.T
    return preconditioner, {
        "input_second_moment_condition": float(eigen[-1] / max(eigen[0], floor)),
        "preconditioner_condition": float(np.linalg.cond(preconditioner)),
        "precondition_floor": floor,
    }


def _predict(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            outputs.append(model(torch.from_numpy(x[start : start + batch_size]).to(device)).cpu().numpy())
    return np.concatenate(outputs)


def _make_optimizer(name: str, parameters, learning_rate: float):
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=0.0)
    if name == "adagrad":
        return torch.optim.Adagrad(parameters, lr=learning_rate, weight_decay=1e-4)
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    if name == "hybrid":
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
    raise ValueError(name)


def _invariant_initialize(first: nn.Linear, train: np.ndarray, seed: int) -> None:
    covariance = np.asarray(train, dtype=np.float64).T @ np.asarray(train, dtype=np.float64) / len(train)
    eigen, vectors = np.linalg.eigh(covariance)
    floor = max(float(eigen[-1]) * 1e-10, 1e-14)
    inverse_sqrt = (vectors * np.maximum(eigen, floor) ** -0.5) @ vectors.T
    rng = np.random.default_rng(seed + 99173)
    gaussian = rng.normal(scale=math.sqrt(1.0 / (3.0 * train.shape[1])), size=first.weight.shape)
    weight = gaussian @ inverse_sqrt
    with torch.no_grad():
        first.weight.copy_(torch.from_numpy(weight).to(first.weight))
        first.bias.zero_()


def _function_match_initialize(first: nn.Linear, basis_transform: np.ndarray) -> None:
    inverse_transpose = np.linalg.inv(basis_transform).T
    mapped = first.weight.detach().cpu().numpy().astype(np.float64) @ inverse_transpose
    with torch.no_grad():
        first.weight.copy_(torch.from_numpy(mapped).to(first.weight))


def transform_for_remedy(
    x: dict[str, np.ndarray], remedy: Remedy
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if remedy.input_transform == "raw":
        return x, {}
    if remedy.input_transform == "diagonal":
        return diagonal_standardize(x), {}
    if remedy.input_transform == "whiten":
        transformed, meta = whiten(x)
        return transformed, {"canonical_rank": meta["retained_rank"]}
    if remedy.input_transform == "anchor":
        return anchor_canonicalize(x)
    if remedy.input_transform == "anchor_whiten":
        canonical, canonical_meta = anchor_canonicalize(x)
        transformed, white_meta = whiten(canonical)
        return transformed, {
            **canonical_meta,
            "canonical_rank": white_meta["retained_rank"],
        }
    raise ValueError(remedy.input_transform)


def train_remedy(
    data: Prepared,
    remedy: Remedy,
    seed: int,
    device: str,
    model_name: str = "mlp",
    basis_transform: np.ndarray | None = None,
    max_epochs: int = 40,
    patience: int = 6,
    batch_size: int = 512,
) -> tuple[dict[str, object], list[dict[str, float | int]]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch_device = torch.device(device)
    output_size = data.n_classes if data.task == "multiclass" else 1
    cls = MLP if model_name == "mlp" else ResNet
    model = cls(data.x["train"].shape[1], output_size, 256, 3, 0.1).to(torch_device)
    first = model.first
    if remedy.invariant_initialization:
        _invariant_initialize(first, data.x["train"], seed)
    if remedy.function_matched_initialization and basis_transform is not None:
        _function_match_initialize(first, basis_transform)

    first_parameters = {id(first.weight), id(first.bias)}
    later = [parameter for parameter in model.parameters() if id(parameter) not in first_parameters]
    parameters = model.parameters() if remedy.optimizer != "hybrid" else later
    optimizer = _make_optimizer(remedy.optimizer, parameters, remedy.learning_rate)
    preconditioner = None
    precondition_meta: dict[str, float] = {}
    if remedy.first_precondition_power:
        matrix, precondition_meta = fit_input_preconditioner(
            data.x["train"], remedy.first_precondition_power
        )
        preconditioner = torch.from_numpy(matrix).to(torch_device, torch.float64)
    momentum_buffer = None

    if data.task == "binclass":
        criterion: nn.Module = nn.BCEWithLogitsLoss()
    elif data.task == "multiclass":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(data.x["train"]), torch.from_numpy(data.y["train"])),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=torch_device.type == "cuda",
    )
    best_loss, best_epoch, stale, best_state = math.inf, 0, 0, None
    curves = []
    started = time.perf_counter()
    first_gradient_norm = math.nan
    first_preconditioned_gradient_norm = math.nan
    for epoch in range(1, max_epochs + 1):
        model.train()
        total, count = 0.0, 0
        for batch_index, (features, target) in enumerate(loader):
            features = features.to(torch_device, non_blocking=True)
            target = target.to(torch_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if remedy.optimizer == "hybrid":
                first.weight.grad = None
                first.bias.grad = None
            prediction = model(features)
            if data.task != "multiclass":
                prediction = prediction.squeeze(-1)
            loss = criterion(prediction, target)
            loss.backward()
            if preconditioner is not None:
                gradient = torch.cat((first.weight.grad, first.bias.grad[:, None]), dim=1)
                preconditioned = (gradient.double() @ preconditioner).to(gradient.dtype)
                if epoch == 1 and batch_index == 0:
                    first_gradient_norm = float(gradient.norm().detach().cpu())
                    first_preconditioned_gradient_norm = float(preconditioned.norm().detach().cpu())
                if remedy.optimizer == "hybrid":
                    if momentum_buffer is None:
                        momentum_buffer = preconditioned.detach().clone()
                    else:
                        momentum_buffer.mul_(0.9).add_(preconditioned)
                    with torch.no_grad():
                        first.weight.add_(momentum_buffer[:, :-1], alpha=-remedy.first_learning_rate)
                        first.bias.add_(momentum_buffer[:, -1], alpha=-remedy.first_learning_rate)
                else:
                    first.weight.grad.copy_(preconditioned[:, :-1])
                    first.bias.grad.copy_(preconditioned[:, -1])
            optimizer.step()
            total += float(loss.detach().cpu()) * len(features)
            count += len(features)
        val_prediction = _predict(model, data.x["val"], torch_device, batch_size * 4)
        val_loss = loss_numpy(data.task, val_prediction, data.y["val"])
        curves.append({"epoch": epoch, "train_loss": total / count, "val_loss": val_loss})
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale > patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    train_prediction = _predict(model, data.x["train"], torch_device, batch_size * 4)
    val_prediction = _predict(model, data.x["val"], torch_device, batch_size * 4)
    test_prediction = _predict(model, data.x["test"], torch_device, batch_size * 4)
    result = {
        "input_features": data.x["train"].shape[1],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "train_loss": loss_numpy(data.task, train_prediction, data.y["train"]),
        "val_loss": loss_numpy(data.task, val_prediction, data.y["val"]),
        "test_loss": loss_numpy(data.task, test_prediction, data.y["test"]),
        "val_metric": metric(data, val_prediction, data.y["val"]),
        "test_metric": metric(data, test_prediction, data.y["test"]),
        "first_gradient_norm": first_gradient_norm,
        "first_preconditioned_gradient_norm": first_preconditioned_gradient_norm,
        "first_weight_norm": float(first.weight.detach().norm().cpu()),
        "train_seconds": time.perf_counter() - started,
        **precondition_meta,
    }
    return result, curves


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run(args: argparse.Namespace) -> None:
    output = args.output
    curves_output = output.with_name(output.stem + "_curves.csv")
    rows, curve_rows = _read(output), _read(curves_output)
    complete = {
        (row["dataset"], row["model"], int(row["seed"]), float(row["target_kappa"]), row["remedy"])
        for row in rows
    }
    equivalence = {}
    for dataset_name in args.datasets:
        for kappa in args.kappas:
            dataset, reference, transformed, basis_transform = controlled_numeric(dataset_name, kappa)
            relation_error = float(
                np.linalg.norm(reference["train"].astype(np.float64) @ basis_transform - transformed["train"])
                / np.linalg.norm(transformed["train"])
            )
            equivalence[f"{dataset_name}/{kappa:g}"] = relation_error
            for remedy_name in args.remedies:
                remedy = REMEDIES[remedy_name]
                x, transform_meta = transform_for_remedy(transformed, remedy)
                for model_name in args.models:
                    for seed in args.seeds:
                        key = (dataset_name, model_name, seed, float(kappa), remedy_name)
                        if key in complete:
                            continue
                        # Matched initialization is defined only for raw controlled
                        # coordinates. Other remedies derive their own coordinates.
                        matched = basis_transform if remedy.function_matched_initialization else None
                        failure = ""
                        try:
                            fit, curves = train_remedy(
                                make_prepared(dataset, x, {}),
                                remedy,
                                seed,
                                args.device,
                                model_name,
                                matched,
                            )
                        except (AssertionError, FloatingPointError) as error:
                            failure = f"{type(error).__name__}: {error}"
                            fit, curves = {
                                "input_features": x["train"].shape[1],
                                "parameters": math.nan,
                                "best_epoch": 0,
                                "train_loss": math.nan,
                                "val_loss": math.nan,
                                "test_loss": math.nan,
                                "val_metric": math.nan,
                                "test_metric": math.nan,
                                "first_gradient_norm": math.nan,
                                "first_preconditioned_gradient_norm": math.nan,
                                "first_weight_norm": math.nan,
                                "train_seconds": math.nan,
                            }, []
                        row = {
                            "experiment": "optimizer_remedy",
                            "intervention_class": "B",
                            "dataset": dataset_name,
                            "task": dataset.task,
                            "model": model_name,
                            "seed": seed,
                            "target_kappa": kappa,
                            "remedy": remedy_name,
                            "input_transform": remedy.input_transform,
                            "optimizer": remedy.optimizer,
                            "learning_rate": remedy.learning_rate,
                            "first_precondition_power": remedy.first_precondition_power,
                            "first_learning_rate": remedy.first_learning_rate,
                            "invariant_initialization": remedy.invariant_initialization,
                            "function_matched_initialization": remedy.function_matched_initialization,
                            "basis_relation_error": relation_error,
                            "split_fingerprint": dataset.split_fingerprint,
                            "failure": failure,
                            **geometry(x["train"]),
                            **transform_meta,
                            **fit,
                        }
                        rows.append(row)
                        _write(output, rows)
                        curve_rows.extend(
                            {
                                "dataset": dataset_name,
                                "model": model_name,
                                "seed": seed,
                                "target_kappa": kappa,
                                "remedy": remedy_name,
                                **curve,
                            }
                            for curve in curves
                        )
                        _write(curves_output, curve_rows)
                        status = f"FAILED {failure}" if failure else f"metric={fit['test_metric']:.6f}"
                        print(
                            f"{dataset_name:12s} {model_name:6s} κ={kappa:4g} s{seed} "
                            f"{remedy_name:31s} {status}",
                            flush=True,
                        )
    output.with_name(output.stem + "_equivalence.json").write_text(json.dumps(equivalence, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["adult", "diamond"])
    parser.add_argument("--kappas", nargs="+", type=float, default=[1.0, 3000.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--models", nargs="+", choices=["mlp", "resnet"], default=["mlp"])
    parser.add_argument("--remedies", nargs="+", choices=sorted(REMEDIES), default=list(REMEDIES))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=RESULTS / "screen.csv")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
