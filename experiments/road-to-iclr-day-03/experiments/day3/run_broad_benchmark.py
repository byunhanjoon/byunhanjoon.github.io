"""Resumable runner for the preregistered 25-dataset Day 3 benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .broad_data import (
    CONFIG_PATH,
    Representation,
    config,
    controlled_representation,
    load_broad_dataset,
    natural_blockwise_equivalence_errors,
    natural_representation,
    sketched_anchor_canonicalize,
    standard_representation,
)
from .broad_models import (
    FirstLayerMatrixUpdater,
    ShampooOptimizer,
    SOAPOptimizer,
    covariance_initialize,
    make_model,
    member_loss,
    metrics,
    predictions,
)
from .core import Prepared, diagonal_standardize, geometry, make_prepared, whiten
from .optimizer_remedies import anchor_canonicalize


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/day3/broad_benchmark"
MATRIX_METHODS = {"input_natural", "first_layer_kfac", "shampoo", "soap"}
MANUAL_FIRST_METHODS = {"input_natural", "first_layer_kfac"}


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
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


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _transform(rep: Representation, remedy: str) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if remedy in ("adamw", *MATRIX_METHODS):
        scope = "first_affine" if remedy in MANUAL_FIRST_METHODS else "all_parameters"
        implementation = {
            "input_natural": "exact_static_input_second_moment_right_factor",
            "first_layer_kfac": "scoped_first_layer_two_factor_kfac",
            "shampoo": "full_model_adam_grafted_shampoo",
            "soap": "full_model_public_algorithm_equivalent_soap",
        }.get(remedy, "pytorch_adamw")
        return rep.parts, {"optimizer_scope": scope, "implementation": implementation}
    if remedy == "diagonal_adamw":
        return diagonal_standardize(rep.parts), {"optimizer_scope": "all_parameters"}
    if remedy == "whiten_adamw":
        transformed, metadata = whiten(rep.parts)
        return transformed, {"optimizer_scope": "all_parameters", "whiten_rank": metadata["retained_rank"]}
    if remedy == "anchor_whiten_adamw":
        canonical, canonical_meta = anchor_canonicalize(rep.parts)
        transformed, white_meta = whiten(canonical)
        return transformed, {
            "optimizer_scope": "all_parameters",
            "canonical_rank": canonical_meta["canonical_rank"],
            "whiten_rank": white_meta["retained_rank"],
            "canonical_reconstruction_errors": canonical_meta["reconstruction_errors"],
        }
    if remedy == "sketch_anchor_whiten_adamw":
        canonical, canonical_meta = sketched_anchor_canonicalize(rep.parts)
        transformed, white_meta = whiten(canonical)
        return transformed, {
            "optimizer_scope": "all_parameters",
            "canonical_rank": canonical_meta["canonical_rank"],
            "sketch_attempts": canonical_meta["sketch_attempts"],
            "whiten_rank": white_meta["retained_rank"],
            "canonical_reconstruction_errors": canonical_meta["reconstruction_errors"],
        }
    raise KeyError(remedy)


def _predict_numpy(
    model: nn.Module,
    x: np.ndarray,
    task: str,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            tensor = torch.from_numpy(x[start : start + batch_size]).to(device)
            output.append(predictions(model, tensor, task).float().cpu().numpy())
    return np.concatenate(output)


def train_cell(
    data: Prepared,
    *,
    model_name: str,
    remedy: str,
    seed: int,
    device: str,
    learning_rate: float | None = None,
    ridge: float = 1e-8,
    precondition_frequency: int = 10,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    cfg = config()["training"]
    _seed(seed)
    resolved = torch.device(device)
    memory_reset_ok = True
    if resolved.type == "cuda":
        try:
            torch.cuda.reset_peak_memory_stats(resolved.index)
        except RuntimeError:
            # CUDA's peak-stat reset can race with another process on a shared
            # physical GPU. Training remains valid; mark the memory observation
            # as unavailable instead of failing the scientific cell.
            memory_reset_ok = False
    output_size = data.n_classes if data.task == "multiclass" else 1
    model = make_model(model_name, data.x["train"].shape[1], output_size).to(resolved)
    if remedy in ("input_natural", "first_layer_kfac"):
        covariance_initialize(model.first, data.x["train"], seed, ridge)
    first_ids = {id(model.first.weight), id(model.first.bias)}
    later = [parameter for parameter in model.parameters() if id(parameter) not in first_ids]
    default_lr = {
        "adamw": 1e-3,
        "diagonal_adamw": 1e-3,
        "whiten_adamw": 1e-3,
        "anchor_whiten_adamw": 1e-3,
        "sketch_anchor_whiten_adamw": 1e-3,
        "input_natural": 3e-2,
        "first_layer_kfac": 3e-3,
        "shampoo": 3e-3,
        "soap": 3e-3,
    }[remedy]
    lr = default_lr if learning_rate is None else learning_rate
    if remedy in MANUAL_FIRST_METHODS:
        optimizer = torch.optim.AdamW(later, lr=1e-3, weight_decay=float(config()["optimizer_calibration"]["weight_decay"]))
        first_updater = FirstLayerMatrixUpdater(
            model.first,
            data.x["train"],
            remedy,
            lr,
            ridge=ridge,
            precondition_frequency=precondition_frequency,
        )
    elif remedy == "shampoo":
        optimizer = ShampooOptimizer(
            model.parameters(),
            lr=lr,
            weight_decay=float(config()["optimizer_calibration"]["weight_decay"]),
            precondition_frequency=precondition_frequency,
        )
        first_updater = None
    elif remedy == "soap":
        optimizer = SOAPOptimizer(
            model.parameters(),
            lr=lr,
            weight_decay=float(config()["optimizer_calibration"]["weight_decay"]),
            precondition_frequency=precondition_frequency,
        )
        first_updater = None
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=float(config()["optimizer_calibration"]["weight_decay"])
        )
        first_updater = None
    batch_size = int(cfg["batch_size"])
    if len(data.x["train"]) >= int(cfg["large_dataset_threshold"]):
        batch_size = int(cfg["large_batch_size"])
    generator = torch.Generator().manual_seed(seed + 70000)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(data.x["train"]), torch.from_numpy(data.y["train"])),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=resolved.type == "cuda",
    )
    best_utility = -math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    curves = []
    started = time.perf_counter()
    try:
        for epoch in range(1, int(cfg["max_epochs"]) + 1):
            model.train()
            total_loss = 0.0
            seen = 0
            for features, target in loader:
                features = features.to(resolved, non_blocking=True)
                target = target.to(resolved, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                if first_updater is not None:
                    model.first.weight.grad = None
                    model.first.bias.grad = None
                loss = member_loss(model, features, target, data.task)
                loss.backward()
                if first_updater is not None:
                    first_updater.step(len(features))
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * len(features)
                seen += len(features)
            val_logits = _predict_numpy(model, data.x["val"], data.task, resolved, batch_size * 2)
            val_metrics = metrics(data, val_logits, data.y["val"])
            utility = val_metrics["primary"]
            curves.append(
                {
                    "epoch": epoch,
                    "train_loss": total_loss / seen,
                    "val_primary": utility,
                    **{f"val_{key}": value for key, value in val_metrics.items() if key != "primary"},
                }
            )
            if utility > best_utility:
                best_utility = utility
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            if stale > int(cfg["patience"]):
                break
        if best_state is None:
            raise RuntimeError("No finite validation checkpoint")
        model.load_state_dict(best_state)
        all_metrics = {}
        for part in ("val", "test"):
            logits = _predict_numpy(model, data.x[part], data.task, resolved, batch_size * 2)
            all_metrics.update({f"{part}_{key}": value for key, value in metrics(data, logits, data.y[part]).items()})
        peak = (
            int(torch.cuda.max_memory_allocated(resolved.index))
            if resolved.type == "cuda" and memory_reset_ok
            else math.nan
        )
        diagnostics = first_updater.diagnostics if first_updater is not None else None
        result = {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "best_epoch": best_epoch,
            "epochs_trained": epoch,
            "learning_rate": lr,
            "ridge": ridge,
            "precondition_frequency": precondition_frequency,
            "train_seconds": time.perf_counter() - started,
            "peak_cuda_bytes": peak,
            "peak_memory_observation_valid": memory_reset_ok,
            "first_gradient_norm": diagnostics.raw_gradient_norm if diagnostics else math.nan,
            "first_update_norm": diagnostics.update_norm if diagnostics else math.nan,
            "preconditioner_condition": diagnostics.preconditioner_condition if diagnostics else math.nan,
            **all_metrics,
        }
        return result, curves
    finally:
        if first_updater is not None:
            first_updater.close()


def _representation(dataset, family: str, kappa: float) -> Representation:
    if family == "controlled":
        return controlled_representation(dataset, kappa)
    if family in ("cumulative_helmert", "local_adjacent"):
        return natural_representation(dataset, family)
    if family in ("raw_standard", "quantile_standard"):
        return standard_representation(dataset, family)
    raise KeyError(family)


def run(args: argparse.Namespace) -> None:
    rows = _read(args.output)
    curve_path = args.output.with_name(args.output.stem + "_curves.csv")
    curve_rows = _read(curve_path)
    complete = {
        (
            row["dataset"],
            row["representation"],
            float(row["target_kappa"]),
            row["model"],
            row["remedy"],
            int(row["seed"]),
            float(row["learning_rate_requested"]),
            float(row["ridge_requested"]),
            int(row["precondition_frequency_requested"]),
        )
        for row in rows
        if not row.get("failure", "").strip()
    }
    for dataset_name in args.datasets:
        dataset = load_broad_dataset(dataset_name)
        natural_errors = natural_blockwise_equivalence_errors(dataset)
        for representation_name in args.representations:
            kappas = args.kappas if representation_name == "controlled" else [1.0]
            for kappa in kappas:
                representation = _representation(dataset, representation_name, kappa)
                for remedy in args.remedies:
                    transform_started = time.perf_counter()
                    transformed, transform_meta = _transform(representation, remedy)
                    preprocessing_seconds = time.perf_counter() - transform_started
                    prepared = make_prepared(dataset, transformed, {})
                    geometry_meta = geometry(transformed["train"])
                    for model_name in args.models:
                        for seed in args.seeds:
                            requested_lr = args.learning_rate if args.learning_rate is not None else -1.0
                            key = (
                                dataset_name,
                                representation_name,
                                float(kappa),
                                model_name,
                                remedy,
                                seed,
                                requested_lr,
                                args.ridge,
                                args.precondition_frequency,
                            )
                            if key in complete:
                                continue
                            failure = ""
                            try:
                                fit, curves = train_cell(
                                    prepared,
                                    model_name=model_name,
                                    remedy=remedy,
                                    seed=seed,
                                    device=args.device,
                                    learning_rate=args.learning_rate,
                                    ridge=args.ridge,
                                    precondition_frequency=args.precondition_frequency,
                                )
                            except Exception as error:  # failures are scientific outcomes
                                failure = f"{type(error).__name__}: {error}"
                                fit, curves = {}, []
                            row = {
                                "experiment": "broad_basis_benchmark",
                                "dataset": dataset_name,
                                "task": dataset.task,
                                "split_fingerprint": dataset.split_fingerprint,
                                "representation": representation_name,
                                "target_kappa": kappa,
                                "model": model_name,
                                "remedy": remedy,
                                "seed": seed,
                                "learning_rate_requested": requested_lr,
                                "ridge_requested": args.ridge,
                                "precondition_frequency_requested": args.precondition_frequency,
                                "failure": failure,
                                "natural_equivalence_max_error": max(
                                    value
                                    for direction in natural_errors.values()
                                    for value in direction.values()
                                ),
                                "representation_metadata": json.dumps(representation.metadata, sort_keys=True),
                                "transform_metadata": json.dumps(transform_meta, sort_keys=True),
                                "preprocessing_seconds": preprocessing_seconds,
                                **geometry_meta,
                                **fit,
                            }
                            rows.append(row)
                            _write(args.output, rows)
                            curve_rows.extend(
                                {
                                    "dataset": dataset_name,
                                    "representation": representation_name,
                                    "target_kappa": kappa,
                                    "model": model_name,
                                    "remedy": remedy,
                                    "seed": seed,
                                    **curve,
                                }
                                for curve in curves
                            )
                            _write(curve_path, curve_rows)
                            status = failure or f"test={fit['test_primary']:.6f}"
                            print(
                                f"{dataset_name:34s} {representation_name:20s} κ={kappa:4g} "
                                f"{model_name:28s} {remedy:24s} s{seed} {status}",
                                flush=True,
                            )


def main() -> None:
    cfg = config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=cfg["datasets"])
    parser.add_argument("--representations", nargs="+", default=["controlled"])
    parser.add_argument("--kappas", nargs="+", type=float, default=cfg["kappas"])
    parser.add_argument("--models", nargs="+", default=["mlp"])
    parser.add_argument("--remedies", nargs="+", default=["adamw"])
    parser.add_argument("--seeds", nargs="+", type=int, default=cfg["broad_seeds"])
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--precondition-frequency", type=int, default=10)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=RESULTS / "runs.csv")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
