"""Controlled MLP utilities for the optimization-geometry mechanism test."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from .basis_dependence import Representation


@dataclass(frozen=True)
class OptimizerCondition:
    name: str
    function_matched: bool
    optimizer: str
    momentum: float
    weight_decay: float


def optimizer_conditions(default_weight_decay: float) -> list[OptimizerCondition]:
    return [
        OptimizerCondition("ordinary_adamw", False, "adamw", 0.0, default_weight_decay),
        OptimizerCondition("matched_adamw", True, "adamw", 0.0, default_weight_decay),
        OptimizerCondition("matched_sgd_momentum", True, "sgd", 0.9, default_weight_decay),
        OptimizerCondition("matched_sgd_plain", True, "sgd", 0.0, 0.0),
        OptimizerCondition("matched_adamw_no_weight_decay", True, "adamw", 0.0, 0.0),
    ]


def minibatch_orders(n_rows: int, epochs: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.permutation(n_rows).astype(np.int64) for _ in range(epochs)]


def order_sha256(order: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(order).view(np.uint8)).hexdigest()


def make_controlled_mlp(
    input_dimension: int, output_dimension: int, model_config: dict[str, Any], seed: int, device: str,
) -> Any:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    activation = nn.GELU if model_config["activation"] == "GELU" else nn.ReLU
    layers: list[nn.Module] = []
    size = input_dimension
    for _ in range(int(model_config["hidden_layers"])):
        layers.extend([nn.Linear(size, int(model_config["width"])), activation()])
        size = int(model_config["width"])
    layers.append(nn.Linear(size, output_dimension))
    return nn.Sequential(*layers).to(device)


def function_matched_copy(reference_model: Any, transformed: Representation, device: str) -> Any:
    """Copy a model and adjust its first layer for ``X' = X A`` block transforms.

    PyTorch stores a linear layer as ``x @ W.T + b``. Consequently, the exact
    function-preserving update is ``W' = W @ inv(A).T`` on every transformed block.
    """
    import torch
    from torch import nn

    matched = copy.deepcopy(reference_model).to(device)
    first = next(module for module in matched.modules() if isinstance(module, nn.Linear))
    with torch.no_grad():
        source_weight = first.weight.detach().clone()
        for feature, matrix in transformed.transforms.items():
            indices = transformed.feature_blocks[feature]
            a = torch.as_tensor(np.asarray(matrix), dtype=source_weight.dtype, device=device)
            first.weight[:, indices] = source_weight[:, indices] @ torch.linalg.inv(a).T
    return matched


def max_logit_difference(
    reference_model: Any, transformed_model: Any, X_reference: np.ndarray,
    X_transformed: np.ndarray, device: str, maximum_rows: int = 1000,
) -> float:
    import torch

    count = min(maximum_rows, len(X_reference))
    left = torch.as_tensor(np.asarray(X_reference[:count], dtype=np.float32), device=device)
    right = torch.as_tensor(np.asarray(X_transformed[:count], dtype=np.float32), device=device)
    reference_model.eval()
    transformed_model.eval()
    with torch.no_grad():
        return float(torch.max(torch.abs(reference_model(left) - transformed_model(right))).item())


def make_optimizer(model: Any, condition: OptimizerCondition, learning_rate: float) -> Any:
    import torch

    if condition.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=condition.weight_decay)
    if condition.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=condition.momentum,
            weight_decay=condition.weight_decay,
        )
    raise ValueError(condition.optimizer)
