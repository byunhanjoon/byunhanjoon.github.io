"""Executable shared graph with regime- or context-dependent coefficients."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .executor import ExecutableProgram


@dataclass
class RegimeParameterizedProgram:
    program: ExecutableProgram
    scales: np.ndarray
    biases: np.ndarray

    def __call__(self, features: np.ndarray, regimes: np.ndarray) -> np.ndarray:
        base = np.asarray(self.program(features), dtype=np.float64)
        return self.scales[regimes] * base + self.biases[regimes]


def fit_regime_coefficients(
    program: ExecutableProgram,
    features: np.ndarray,
    targets: np.ndarray,
    regimes: np.ndarray,
    n_regimes: int,
) -> RegimeParameterizedProgram:
    base = np.asarray(program(features), dtype=np.float64)
    scales = np.zeros(n_regimes, dtype=np.float64)
    biases = np.zeros(n_regimes, dtype=np.float64)
    for regime in range(n_regimes):
        mask = regimes == regime
        design = np.column_stack([base[mask], np.ones(mask.sum())])
        scales[regime], biases[regime] = np.linalg.lstsq(design, targets[mask], rcond=None)[0]
    return RegimeParameterizedProgram(program, scales, biases)


class ContextCoefficientModel(nn.Module):
    def __init__(self, context_width: int = 1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(context_width, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 2),
        )
        nn.init.zeros_(self.network[-1].weight)
        with torch.no_grad():
            self.network[-1].bias.copy_(torch.tensor([1.0, 0.0]))

    def forward(self, base_prediction: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        scale, bias = self.network(context).unbind(dim=-1)
        return scale * base_prediction + bias
