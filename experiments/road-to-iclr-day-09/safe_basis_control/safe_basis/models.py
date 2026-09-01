"""Backbone adapters with optional training-partition predictions."""

from __future__ import annotations

import copy
import dataclasses
import gc
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from .common import PRIOR_PROTOCOL_PATH, bd


def prior_model_config() -> dict[str, Any]:
    return yaml.safe_load(PRIOR_PROTOCOL_PATH.read_text())


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.linear1 = nn.Linear(width, width)
        self.linear2 = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.norm(values)
        residual = torch.relu(self.linear1(residual))
        residual = self.dropout(residual)
        residual = self.linear2(residual)
        return values + residual


class TabularResNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, width: int = 256, blocks: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, width)
        self.blocks = nn.Sequential(*(ResidualBlock(width, dropout) for _ in range(blocks)))
        self.output_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, output_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.input(values))
        hidden = self.blocks(hidden)
        return self.output(torch.relu(self.output_norm(hidden)))


def _fit_resnet(
    problem_type: str,
    rep: Any,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    seed: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    output_dim = 1 if problem_type == "regression" else int(np.max(y_train)) + 1
    model = TabularResNet(rep.X_train.shape[1], output_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_x = torch.as_tensor(np.asarray(rep.X_train, dtype=np.float32), device=device)
    validation_x = torch.as_tensor(np.asarray(rep.X_validation, dtype=np.float32), device=device)
    test_x = torch.as_tensor(np.asarray(rep.X_test, dtype=np.float32), device=device)
    if problem_type == "regression":
        y_mean = float(np.mean(y_train))
        y_scale = max(float(np.std(y_train)), 1e-8)
        train_y = torch.as_tensor(((y_train - y_mean) / y_scale).astype(np.float32), device=device)
        validation_y = torch.as_tensor(((y_validation - y_mean) / y_scale).astype(np.float32), device=device)
        loss_fn: Any = nn.MSELoss()
    else:
        y_mean, y_scale = 0.0, 1.0
        train_y = torch.as_tensor(y_train.astype(np.int64), device=device)
        validation_y = torch.as_tensor(y_validation.astype(np.int64), device=device)
        loss_fn = nn.CrossEntropyLoss()
    generator = np.random.default_rng(seed)
    best_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    started = time.perf_counter()
    for epoch in range(80):
        model.train()
        order = generator.permutation(len(y_train))
        for start in range(0, len(order), 256):
            rows = torch.as_tensor(order[start : start + 256], device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(train_x[rows]).squeeze(-1)
            loss = loss_fn(prediction, train_y[rows])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            value = float(loss_fn(model(validation_x).squeeze(-1), validation_y).item())
        if value < best_loss - 1e-7:
            best_loss = value
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 12:
                break
    if best_state is None:
        raise RuntimeError("ResNet failed to produce a checkpoint")
    model.load_state_dict(best_state)
    fit_seconds = time.perf_counter() - started
    model.eval()

    def predict(values: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            logits = model(values).squeeze(-1)
            if problem_type == "regression":
                return logits.cpu().numpy() * y_scale + y_mean
            return torch.softmax(logits, dim=1).cpu().numpy()

    prediction_started = time.perf_counter()
    train, validation, test = predict(train_x), predict(validation_x), predict(test_x)
    telemetry = {
        "fit_seconds": fit_seconds,
        "predict_seconds": time.perf_counter() - prediction_started,
        "best_epoch": best_epoch,
        "best_validation_objective": best_loss,
        "architecture": "ResNet-style-tabular-3x256-LayerNorm-ReLU-dropout0.1",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
    }
    del model
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return train, validation, test, telemetry


def fit_predictions(
    model_name: str,
    problem_type: str,
    rep: Any,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    seed: int,
    device: str,
    *,
    include_train: bool = False,
    learning_rate_multiplier: float = 1.0,
    weight_decay: float | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if model_name == "resnet_tabular":
        train, validation, test, telemetry = _fit_resnet(
            problem_type, rep, y_train, y_validation, seed, device
        )
        predictions = {"validation": validation, "test": test}
        if include_train:
            predictions["train"] = train
        return predictions, telemetry

    config = copy.deepcopy(prior_model_config())
    model_config = config["models"][model_name]
    if "learning_rate" in model_config:
        model_config["learning_rate"] = float(model_config["learning_rate"]) * float(learning_rate_multiplier)
    elif model_name == "tabm_d" and learning_rate_multiplier != 1.0:
        model_config["learning_rate"] = 0.002 * float(learning_rate_multiplier)
    if weight_decay is not None:
        model_config["weight_decay"] = float(weight_decay)
    if include_train:
        joined = np.concatenate([rep.X_train, rep.X_test], axis=0)
        fit_rep = dataclasses.replace(rep, X_test=joined)
    else:
        fit_rep = rep
    validation, combined_test, telemetry = bd.fit_predict(
        model_name,
        problem_type,
        fit_rep,
        y_train,
        y_validation,
        seed,
        device,
        config,
    )
    if include_train:
        train = np.asarray(combined_test[: len(rep.X_train)])
        test = np.asarray(combined_test[len(rep.X_train) :])
        predictions = {"train": train, "validation": np.asarray(validation), "test": test}
    else:
        predictions = {"validation": np.asarray(validation), "test": np.asarray(combined_test)}
    telemetry = {
        **telemetry,
        "learning_rate_multiplier": float(learning_rate_multiplier),
        "weight_decay_override": weight_decay,
    }
    return predictions, telemetry
