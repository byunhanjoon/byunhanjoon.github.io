"""Training loop for differentiable program selection."""

from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from ..models.executor import OPERATORS, ExecutableProgram
from ..models.discrete_search import search_chain_program
from ..models.program_search import DifferentiableProgram


@dataclass(frozen=True)
class ProgramTrainingConfig:
    n_nodes: int = 3
    operators: tuple[str, ...] = OPERATORS
    selector: str = "straight_through_gumbel"
    epochs: int = 600
    learning_rate: float = 0.025
    weight_decay: float = 1.0e-5
    start_temperature: float = 2.0
    end_temperature: float = 0.12
    entropy_weight: float = 2.0e-4
    gradient_clip: float = 5.0
    patience: int = 150
    device: str = "cuda"
    discrete_warm_start: bool = True

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["operators"] = list(self.operators)
        return value


@dataclass
class TrainedProgram:
    model: DifferentiableProgram
    compiled: ExecutableProgram
    history: list[dict[str, float]]
    training_seconds: float
    best_validation_nrmse: float


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def train_program(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    *,
    seed: int,
    config: ProgramTrainingConfig,
    warm_start_program: ExecutableProgram | None = None,
) -> TrainedProgram:
    set_deterministic_seed(seed)
    device = _device(config.device)
    x_train = torch.as_tensor(train_features, dtype=torch.float32, device=device)
    y_train = torch.as_tensor(train_targets, dtype=torch.float32, device=device)
    x_validation = torch.as_tensor(validation_features, dtype=torch.float32, device=device)
    y_validation = torch.as_tensor(validation_targets, dtype=torch.float32, device=device)
    target_variance = y_train.var(unbiased=False).clamp_min(1.0e-8)
    validation_scale = y_validation.std(unbiased=False).clamp_min(1.0e-8)
    model = DifferentiableProgram(
        n_features=x_train.shape[1],
        n_nodes=config.n_nodes,
        operators=config.operators,
        selector=config.selector,
    ).to(device)
    if config.discrete_warm_start:
        warm_start = warm_start_program or search_chain_program(
                train_features,
                train_targets,
                validation_features,
                validation_targets,
                max_depth=config.n_nodes,
                operators=tuple(operator for operator in config.operators if operator != "identity"),
            )
        model.initialize_from_program(warm_start)
    else:
        with torch.no_grad():
            model.output_bias.copy_(y_train.mean())
            model.output_scale.copy_(y_train.std(unbiased=False).clamp_min(0.1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    model.eval()
    with torch.no_grad():
        initial_prediction, _ = model(x_validation, temperature=config.end_temperature, hard=True)
        best_validation = float(
            (initial_prediction - y_validation).square().mean().sqrt() / validation_scale
        )
    epochs_without_improvement = 0
    started = time.perf_counter()
    for epoch in range(config.epochs):
        progress = epoch / max(config.epochs - 1, 1)
        temperature = config.start_temperature * (
            config.end_temperature / config.start_temperature
        ) ** progress
        entropy_weight = 0.0 if progress < 0.30 else config.entropy_weight
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction, diagnostics = model(x_train, temperature=temperature, hard=progress > 0.70)
        normalized_mse = (prediction - y_train).square().mean() / target_variance
        entropy = diagnostics.operator_entropy + diagnostics.input_entropy + diagnostics.output_entropy
        loss = normalized_mse + entropy_weight * entropy
        if not torch.isfinite(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            validation_prediction, _ = model(x_validation, temperature=temperature, hard=True)
            validation_nrmse = float(
                (validation_prediction - y_validation).square().mean().sqrt() / validation_scale
            )
        if validation_nrmse + 1.0e-6 < best_validation:
            best_validation = validation_nrmse
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch % 20 == 0 or epoch == config.epochs - 1:
            history.append(
                {
                    "epoch": float(epoch),
                    "temperature": float(temperature),
                    "train_nrmse": float(normalized_mse.detach().sqrt()),
                    "validation_nrmse": validation_nrmse,
                    "operator_entropy": float(diagnostics.operator_entropy.detach()),
                    "input_entropy": float(diagnostics.input_entropy.detach()),
                    "output_entropy": float(diagnostics.output_entropy.detach()),
                }
            )
        if epoch > config.epochs // 2 and epochs_without_improvement >= config.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    compiled = model.compile().fit_output_affine(train_features, train_targets)
    return TrainedProgram(model, compiled, history, time.perf_counter() - started, best_validation)
