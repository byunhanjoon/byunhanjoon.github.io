"""End-to-end training for operand estimators with a fixed exact program."""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass

import numpy as np
import torch

from ..models.executor import ExecutableProgram
from ..models.operand import OperandEstimator, build_operand_estimator


@dataclass
class TrainedOperand:
    estimator: OperandEstimator
    training_seconds: float
    best_validation_nrmse: float


def train_operand_estimator(
    variant: str,
    program: ExecutableProgram,
    latent_train: np.ndarray,
    targets_train: np.ndarray,
    latent_validation: np.ndarray,
    targets_validation: np.ndarray,
    *,
    seed: int,
    noise_strength: float,
    epochs: int,
    learning_rate: float,
    correction_weight: float,
    device: str,
) -> TrainedOperand:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    latent_train_t = torch.as_tensor(latent_train, dtype=torch.float32, device=torch_device)
    targets_train_t = torch.as_tensor(targets_train, dtype=torch.float32, device=torch_device)
    latent_validation_t = torch.as_tensor(latent_validation, dtype=torch.float32, device=torch_device)
    targets_validation_t = torch.as_tensor(targets_validation, dtype=torch.float32, device=torch_device)
    mean = latent_train_t.mean(dim=0)
    scale = latent_train_t.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    estimator = build_operand_estimator(variant, mean, scale).to(torch_device)
    if variant == "raw":
        prediction = program(latent_validation_t)
        nrmse = float(
            (prediction - targets_validation_t).square().mean().sqrt()
            / targets_validation_t.std(unbiased=False).clamp_min(1.0e-8)
        )
        return TrainedOperand(estimator, 0.0, nrmse)
    optimizer = torch.optim.AdamW(estimator.parameters(), lr=learning_rate, weight_decay=1.0e-5)
    target_variance = targets_train_t.var(unbiased=False).clamp_min(1.0e-8)
    validation_scale = targets_validation_t.std(unbiased=False).clamp_min(1.0e-8)
    generator = torch.Generator(device=torch_device).manual_seed(seed * 1009 + 37)
    validation_generator = torch.Generator(device=torch_device).manual_seed(seed * 1009 + 73)
    validation_observed = latent_validation_t + noise_strength * scale * torch.randn(
        latent_validation_t.shape, generator=validation_generator, device=torch_device
    )
    best_state = copy.deepcopy(estimator.state_dict())
    best_validation = float("inf")
    stale = 0
    started = time.perf_counter()
    for _ in range(epochs):
        estimator.train()
        observed = latent_train_t + noise_strength * scale * torch.randn(
            latent_train_t.shape, generator=generator, device=torch_device
        )
        optimizer.zero_grad(set_to_none=True)
        estimated = estimator(observed)
        prediction = program(estimated)
        task_loss = (prediction - targets_train_t).square().mean() / target_variance
        correction = ((estimated - observed) / scale).square().mean()
        loss = task_loss + correction_weight * correction
        if not torch.isfinite(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(estimator.parameters(), 5.0)
        optimizer.step()
        estimator.eval()
        with torch.no_grad():
            validation_prediction = program(estimator(validation_observed))
            validation_nrmse = float(
                (validation_prediction - targets_validation_t).square().mean().sqrt()
                / validation_scale
            )
        if validation_nrmse + 1.0e-6 < best_validation:
            best_validation = validation_nrmse
            best_state = copy.deepcopy(estimator.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= 80:
            break
    estimator.load_state_dict(best_state)
    estimator.eval()
    return TrainedOperand(estimator, time.perf_counter() - started, best_validation)
