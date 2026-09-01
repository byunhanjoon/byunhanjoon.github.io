"""Controlled MLP, direct TabM-D, and frozen-model training adapters."""

from __future__ import annotations

import copy
import gc
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from .common import bd, max_gpu_memory_mb, reset_gpu_memory
from .optimizers import make_optimizers, step as optimizer_step, zero_grad


@dataclass
class FitResult:
    validation: np.ndarray
    test: np.ndarray
    telemetry: dict[str, Any]


class ControlledMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, width: int, hidden_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(current, width), nn.GELU()])
            current = width
        layers.append(nn.Linear(current, output_dim))
        self.network = nn.Sequential(*layers)

    @property
    def first_weight(self) -> nn.Parameter:
        return self.network[0].weight

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def _seed_everything(seed: int, device: str) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)


def build_model(
    model_name: str,
    input_dim: int,
    output_dim: int,
    config: dict[str, Any],
    seed: int,
    device: str,
) -> tuple[nn.Module, nn.Parameter, dict[str, Any]]:
    _seed_everything(seed, device)
    if model_name == "controlled_mlp":
        model = ControlledMLP(
            input_dim,
            output_dim,
            int(config["width"]),
            int(config["hidden_layers"]),
        ).to(device)
        return model, model.first_weight, {"architecture": "3x256-GELU-no_batchnorm-no_dropout"}
    if model_name == "tabm_d":
        from pytabkit.models.nn_models.tabm import Model

        model = Model(
            n_num_features=input_dim,
            cat_cardinalities=[],
            n_classes=output_dim,
            backbone={
                "type": "MLP",
                "n_blocks": int(config["n_blocks"]),
                "d_block": int(config["d_block"]),
                "dropout": float(config["dropout"]),
            },
            bins=None,
            num_embeddings=None,
            arch_type=str(config["arch_type"]),
            k=int(config["k"]),
            share_training_batches=bool(config["share_training_batches"]),
        ).to(device)
        first = model.backbone.blocks[0][0]
        metadata = {
            "architecture": "TabM-D direct/no-coordinate-preprocessing",
            "tabm_k": int(config["k"]),
            "first_layer_input_scaling_parameter": first.r is not None,
            "optimizer_scope_note": (
                "The block optimizer acts on the shared first-layer weight. TabM's per-ensemble "
                "input scaling r remains under AdamW and is a separately reported architectural "
                "source of coordinate dependence."
            ),
        }
        return model, first.weight, metadata
    raise ValueError(f"unsupported trainable model {model_name}")


@torch.no_grad()
def data_equivariant_initialization(
    first_weight: nn.Parameter,
    X_train: np.ndarray,
    blocks: dict[str, list[int]],
    seed: int,
) -> dict[str, Any]:
    """Initialize feature blocks as ``R.T @ Z / sqrt(n)`` with invariant scaling."""

    values = np.asarray(X_train, dtype=np.float64)
    fan_in = first_weight.shape[1]
    target_element_std = (3.0 * fan_in) ** -0.5
    records: dict[str, Any] = {}
    for feature, indices in sorted(blocks.items(), key=lambda item: min(item[1])):
        rng = np.random.default_rng(bd.stable_seed(seed, "data_equivariant_init", feature))
        random_targets = rng.standard_normal((len(values), first_weight.shape[0]))
        initialized = random_targets.T @ values[:, indices] / np.sqrt(max(len(values), 1))
        target_norm = target_element_std * np.sqrt(initialized.size)
        source_norm = max(float(np.linalg.norm(initialized)), 1e-12)
        initialized *= target_norm / source_norm
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=first_weight.device)
        first_weight.index_copy_(
            1,
            index_tensor,
            torch.as_tensor(initialized, dtype=first_weight.dtype, device=first_weight.device),
        )
        records[feature] = {
            "indices": list(indices),
            "source_frobenius_norm": source_norm,
            "target_frobenius_norm": target_norm,
        }
    return records


def _forward(model_name: str, model: nn.Module, values: torch.Tensor) -> torch.Tensor:
    result = model(values)
    if model_name == "tabm_d":
        # (batch, ensemble, output)
        return result
    return result


def _prediction_from_logits(
    model_name: str,
    problem_type: str,
    logits: torch.Tensor,
    y_mean: float,
    y_scale: float,
) -> torch.Tensor:
    if model_name == "tabm_d":
        if problem_type == "classification":
            return logits.softmax(dim=-1).mean(dim=1)
        return logits.squeeze(-1).mean(dim=1) * y_scale + y_mean
    if problem_type == "classification":
        return logits.softmax(dim=-1)
    return logits.squeeze(-1) * y_scale + y_mean


def _loss(
    model_name: str,
    problem_type: str,
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if model_name == "tabm_d":
        ensemble = logits.shape[1]
        if problem_type == "classification":
            return nn.functional.cross_entropy(
                logits.flatten(0, 1), target.repeat_interleave(ensemble)
            )
        return nn.functional.mse_loss(
            logits.squeeze(-1).flatten(), target.repeat_interleave(ensemble)
        )
    if problem_type == "classification":
        return nn.functional.cross_entropy(logits, target)
    return nn.functional.mse_loss(logits.squeeze(-1), target)


def fit_trainable(
    model_name: str,
    problem_type: str,
    rep: Any,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    seed: int,
    device: str,
    model_config: dict[str, Any],
    *,
    optimizer_method: str = "adamw",
    optimizer_overrides: dict[str, Any] | None = None,
    initialization: str = "default",
    return_model: bool = False,
) -> FitResult | tuple[FitResult, nn.Module]:
    reset_gpu_memory(device)
    output_dim = 1 if problem_type == "regression" else int(np.max(y_train)) + 1
    model, first_weight, architecture_metadata = build_model(
        model_name, rep.X_train.shape[1], output_dim, model_config, seed, device
    )
    initialization_metadata: dict[str, Any] = {"kind": initialization}
    if initialization == "data_equivariant":
        initialization_metadata["blocks"] = data_equivariant_initialization(
            first_weight, rep.X_train, rep.feature_blocks, seed
        )
        if model_name == "tabm_d":
            first_module = model.backbone.blocks[0][0]
            if first_module.r is not None:
                with torch.no_grad():
                    first_module.r.fill_(1.0)
                first_module.r.requires_grad_(False)
                initialization_metadata["tabm_first_input_adapter"] = (
                    "frozen_to_one because a diagonal per-coordinate input adapter is not "
                    "closed under general within-block rotations"
                )
    elif initialization != "default":
        raise ValueError(f"unknown initialization {initialization}")
    override = {} if optimizer_overrides is None else dict(optimizer_overrides)
    learning_rate = float(override.pop("learning_rate", model_config["learning_rate"]))
    optimizers = make_optimizers(
        model,
        first_weight,
        rep.feature_blocks,
        method=optimizer_method,
        lr=learning_rate,
        weight_decay=float(override.pop("weight_decay", model_config["weight_decay"])),
        beta1=float(override.pop("beta1", 0.9)),
        beta2=float(override.pop("beta2", 0.999)),
        epsilon=float(override.pop("epsilon", 1e-8)),
        alpha=float(override.pop("alpha", 0.0)),
        eigenvalue_floor=float(override.pop("eigenvalue_floor", 1e-8)),
        normalization=str(override.pop("normalization", "mean")),
    )
    if override:
        raise ValueError(f"unused optimizer overrides: {sorted(override)}")
    train_values = torch.as_tensor(np.asarray(rep.X_train, dtype=np.float32), device=device)
    validation_values = torch.as_tensor(np.asarray(rep.X_validation, dtype=np.float32), device=device)
    test_values = torch.as_tensor(np.asarray(rep.X_test, dtype=np.float32), device=device)
    if problem_type == "regression":
        y_mean = float(np.mean(y_train))
        y_scale = max(float(np.std(y_train)), 1e-8)
        train_target = torch.as_tensor(((y_train - y_mean) / y_scale).astype(np.float32), device=device)
        validation_target = torch.as_tensor(
            ((y_validation - y_mean) / y_scale).astype(np.float32), device=device
        )
    else:
        y_mean, y_scale = 0.0, 1.0
        train_target = torch.as_tensor(y_train.astype(np.int64), device=device)
        validation_target = torch.as_tensor(y_validation.astype(np.int64), device=device)
    rng = np.random.default_rng(seed)
    best_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    no_improvement = 0
    started = time.perf_counter()
    for epoch in range(int(model_config["max_epochs"])):
        model.train()
        order = rng.permutation(len(y_train))
        for start in range(0, len(order), int(model_config["batch_size"])):
            indices = torch.as_tensor(
                order[start : start + int(model_config["batch_size"])], device=device
            )
            zero_grad(optimizers)
            loss = _loss(
                model_name,
                problem_type,
                _forward(model_name, model, train_values[indices]),
                train_target[indices],
            )
            loss.backward()
            optimizer_step(optimizers)
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                _loss(
                    model_name,
                    problem_type,
                    _forward(model_name, model, validation_values),
                    validation_target,
                ).item()
            )
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy({name: value.detach().cpu() for name, value in model.state_dict().items()})
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= int(model_config["patience"]):
                break
    if best_state is None:
        raise RuntimeError("trainable model did not produce a checkpoint")
    model.load_state_dict(best_state)
    fit_seconds = time.perf_counter() - started
    predict_started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        validation_prediction = _prediction_from_logits(
            model_name,
            problem_type,
            _forward(model_name, model, validation_values),
            y_mean,
            y_scale,
        ).cpu().numpy()
        test_prediction = _prediction_from_logits(
            model_name,
            problem_type,
            _forward(model_name, model, test_values),
            y_mean,
            y_scale,
        ).cpu().numpy()
    telemetry = {
        **architecture_metadata,
        "optimizer": optimizer_method,
        "learning_rate": learning_rate,
        "initialization": initialization_metadata,
        "fit_seconds": fit_seconds,
        "predict_seconds": time.perf_counter() - predict_started,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "gpu_peak_memory_mb": max_gpu_memory_mb(device),
    }
    result = FitResult(np.asarray(validation_prediction), np.asarray(test_prediction), telemetry)
    if return_model:
        return result, model
    del model
    del optimizers
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def fit_external(
    model_name: str,
    problem_type: str,
    rep: Any,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    seed: int,
    device: str,
    protocol: dict[str, Any],
) -> FitResult:
    reset_gpu_memory(device)
    validation, test, telemetry = bd.fit_predict(
        model_name,
        problem_type,
        rep,
        y_train,
        y_validation,
        seed,
        device,
        protocol,
    )
    telemetry["gpu_peak_memory_mb"] = max_gpu_memory_mb(device)
    return FitResult(np.asarray(validation), np.asarray(test), telemetry)


def fit_model(
    model_name: str,
    problem_type: str,
    rep: Any,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    seed: int,
    device: str,
    protocol: dict[str, Any],
    *,
    optimizer_method: str = "adamw",
    optimizer_overrides: dict[str, Any] | None = None,
    initialization: str = "default",
) -> FitResult:
    if model_name in {"controlled_mlp", "tabm_d"}:
        return fit_trainable(
            model_name,
            problem_type,
            rep,
            y_train,
            y_validation,
            seed,
            device,
            protocol["models"][model_name],
            optimizer_method=optimizer_method,
            optimizer_overrides=optimizer_overrides,
            initialization=initialization,
        )
    if optimizer_method != "adamw" or initialization != "default":
        raise ValueError("custom optimizers and initialization are only valid for trainable models")
    return fit_external(
        model_name,
        problem_type,
        rep,
        y_train,
        y_validation,
        seed,
        device,
        protocol,
    )


@torch.no_grad()
def initial_predictions(
    model_name: str,
    problem_type: str,
    rep: Any,
    seed: int,
    device: str,
    model_config: dict[str, Any],
    initialization: str,
) -> np.ndarray:
    output_dim = 1 if problem_type == "regression" else 2
    model, first_weight, _ = build_model(
        model_name, rep.X_train.shape[1], output_dim, model_config, seed, device
    )
    if initialization == "data_equivariant":
        data_equivariant_initialization(first_weight, rep.X_train, rep.feature_blocks, seed)
        if model_name == "tabm_d":
            first_module = model.backbone.blocks[0][0]
            if first_module.r is not None:
                first_module.r.fill_(1.0)
                first_module.r.requires_grad_(False)
    model.eval()
    values = torch.as_tensor(np.asarray(rep.X_test, dtype=np.float32), device=device)
    logits = _forward(model_name, model, values)
    result = _prediction_from_logits(model_name, problem_type, logits, 0.0, 1.0)
    output = result.cpu().numpy()
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return output
