"""Penalized neural escape hatch on top of a fixed executable program."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import nn

from .executor import ExecutableProgram


GateMode = Literal["none", "scalar", "adaptive"]


class ResidualModule(nn.Module):
    def __init__(self, input_width: int, gate_mode: GateMode) -> None:
        super().__init__()
        self.gate_mode = gate_mode
        self.residual = nn.Sequential(
            nn.Linear(input_width, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Flatten(0),
        )
        if gate_mode == "scalar":
            self.gate_logit = nn.Parameter(torch.tensor(-3.0))
        elif gate_mode == "adaptive":
            self.gate = nn.Sequential(
                nn.Linear(input_width, 32),
                nn.Tanh(),
                nn.Linear(32, 1),
                nn.Flatten(0),
            )
            nn.init.zeros_(self.gate[-2].weight)
            nn.init.constant_(self.gate[-2].bias, -3.0)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = self.residual(features)
        if self.gate_mode == "none":
            gate = torch.ones_like(residual)
        elif self.gate_mode == "scalar":
            gate = torch.sigmoid(self.gate_logit).expand_as(residual)
        else:
            gate = torch.sigmoid(self.gate(features))
        return gate * residual, gate


@dataclass
class ResidualFit:
    training_seconds: float
    validation_objective: float


class ProgramResidualRegressor:
    def __init__(
        self,
        program: ExecutableProgram,
        *,
        seed: int,
        gate_mode: GateMode,
        residual_penalty: float,
        gate_penalty: float,
        epochs: int,
        device: str,
    ) -> None:
        self.program = program
        self.seed = seed
        self.gate_mode = gate_mode
        self.residual_penalty = residual_penalty
        self.gate_penalty = gate_penalty
        self.epochs = epochs
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.x_mean = np.zeros(program.n_features)
        self.x_scale = np.ones(program.n_features)
        self.residual_scale = 1.0
        self.model: ResidualModule | None = None

    def _features(self, features: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(
            (features - self.x_mean) / self.x_scale,
            dtype=torch.float32,
            device=self.device,
        )

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        validation: tuple[np.ndarray, np.ndarray],
    ) -> ResidualFit:
        torch.manual_seed(self.seed)
        self.x_mean = features.mean(axis=0)
        self.x_scale = features.std(axis=0).clip(1.0e-6)
        base_train = np.asarray(self.program(features), dtype=np.float64)
        self.residual_scale = max(float(base_train.std()), 1.0e-6)
        train_x = self._features(features)
        train_residual = torch.as_tensor(
            (targets - base_train) / self.residual_scale,
            dtype=torch.float32,
            device=self.device,
        )
        validation_features, validation_targets = validation
        validation_x = self._features(validation_features)
        validation_base = np.asarray(self.program(validation_features), dtype=np.float64)
        validation_residual = torch.as_tensor(
            (validation_targets - validation_base) / self.residual_scale,
            dtype=torch.float32,
            device=self.device,
        )
        self.model = ResidualModule(features.shape[1], self.gate_mode).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2.0e-3, weight_decay=1.0e-5)
        best_state = copy.deepcopy(self.model.state_dict())
        best = float("inf")
        stale = 0
        started = time.perf_counter()
        for _ in range(self.epochs):
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            contribution, gate = self.model(train_x)
            loss = (contribution - train_residual).square().mean()
            loss = loss + self.residual_penalty * contribution.abs().mean()
            if self.gate_mode != "none":
                loss = loss + self.gate_penalty * gate.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            optimizer.step()
            self.model.eval()
            with torch.no_grad():
                val_contribution, val_gate = self.model(validation_x)
                objective = (val_contribution - validation_residual).square().mean()
                objective = objective + self.residual_penalty * val_contribution.abs().mean()
                if self.gate_mode != "none":
                    objective = objective + self.gate_penalty * val_gate.mean()
                score = float(objective)
            if score + 1.0e-7 < best:
                best = score
                best_state = copy.deepcopy(self.model.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= 100:
                break
        self.model.load_state_dict(best_state)
        self.model.eval()
        return ResidualFit(time.perf_counter() - started, best)

    def predict_with_usage(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("residual model has not been fitted")
        base = np.asarray(self.program(features), dtype=np.float64)
        with torch.no_grad():
            contribution, gate = self.model(self._features(features))
        contribution_array = contribution.cpu().numpy().astype(np.float64) * self.residual_scale
        return base + contribution_array, contribution_array, gate.cpu().numpy().astype(np.float64)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.predict_with_usage(features)[0]
