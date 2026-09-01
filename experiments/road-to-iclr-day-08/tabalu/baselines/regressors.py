"""Phase-A baselines with explicit, documented compute budgets."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from torch import nn


class Regressor(Protocol):
    def fit(self, features: np.ndarray, targets: np.ndarray, validation: tuple[np.ndarray, np.ndarray]) -> None: ...

    def predict(self, features: np.ndarray) -> np.ndarray: ...


@dataclass
class SklearnRegressor:
    estimator: object

    def fit(
        self, features: np.ndarray, targets: np.ndarray, validation: tuple[np.ndarray, np.ndarray]
    ) -> None:
        del validation
        self.estimator.fit(features, targets)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self.estimator.predict(features), dtype=np.float64)


class PolynomialSymbolicRegressor:
    """A deliberately transparent sparse polynomial symbolic baseline."""

    def __init__(self, degree: int = 3) -> None:
        self.features = PolynomialFeatures(degree=degree, include_bias=False)
        self.scaler = StandardScaler()
        self.model: Lasso | None = None

    def fit(
        self, features: np.ndarray, targets: np.ndarray, validation: tuple[np.ndarray, np.ndarray]
    ) -> None:
        transformed = self.scaler.fit_transform(self.features.fit_transform(features))
        validation_features, validation_targets = validation
        validation_transformed = self.scaler.transform(self.features.transform(validation_features))
        best_score = float("inf")
        for alpha in (1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2):
            candidate = Lasso(alpha=alpha, max_iter=10_000, tol=1.0e-6)
            candidate.fit(transformed, targets)
            score = float(np.mean((candidate.predict(validation_transformed) - validation_targets) ** 2))
            if score < best_score:
                best_score = score
                self.model = candidate

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("baseline has not been fitted")
        transformed = self.scaler.transform(self.features.transform(features))
        return np.asarray(self.model.predict(transformed), dtype=np.float64)


class EquationLayer(nn.Module):
    def __init__(self, input_width: int, linear_units: int, product_units: int) -> None:
        super().__init__()
        self.linear_units = linear_units
        self.product_units = product_units
        self.projection = nn.Linear(input_width, linear_units + 2 * product_units)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        projected = self.projection(values)
        linear = projected[:, : self.linear_units]
        factors = projected[:, self.linear_units :]
        products = factors[:, : self.product_units] * factors[:, self.product_units :]
        return torch.cat([linear, products], dim=-1)


class EQLNetwork(nn.Module):
    def __init__(self, input_width: int) -> None:
        super().__init__()
        self.layer1 = EquationLayer(input_width, linear_units=12, product_units=8)
        self.layer2 = EquationLayer(20, linear_units=8, product_units=8)
        self.output = nn.Linear(16, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.layer1(values).clamp(-1.0e3, 1.0e3)
        values = self.layer2(values).clamp(-1.0e5, 1.0e5)
        return self.output(values).squeeze(-1)


class NeuralMixtureNetwork(nn.Module):
    def __init__(self, input_width: int, n_experts: int = 2) -> None:
        super().__init__()
        self.router = nn.Sequential(nn.Linear(input_width, 32), nn.Tanh(), nn.Linear(32, n_experts))
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_width, 64),
                    nn.SiLU(),
                    nn.Linear(64, 64),
                    nn.SiLU(),
                    nn.Linear(64, 1),
                )
                for _ in range(n_experts)
            ]
        )

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probabilities = self.router(values).softmax(dim=-1)
        predictions = torch.cat([expert(values) for expert in self.experts], dim=-1)
        return (predictions * probabilities).sum(dim=-1), probabilities


class NeuralMixtureRegressor:
    def __init__(
        self, seed: int, epochs: int = 600, device: str = "cuda", n_experts: int = 2
    ) -> None:
        self.seed = seed
        self.epochs = epochs
        self.n_experts = n_experts
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.x_mean: np.ndarray | None = None
        self.x_scale: np.ndarray | None = None
        self.y_mean = 0.0
        self.y_scale = 1.0
        self.model: NeuralMixtureNetwork | None = None

    def fit(
        self, features: np.ndarray, targets: np.ndarray, validation: tuple[np.ndarray, np.ndarray]
    ) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.x_mean = features.mean(axis=0)
        self.x_scale = features.std(axis=0).clip(1.0e-6)
        self.y_mean = float(targets.mean())
        self.y_scale = float(max(targets.std(), 1.0e-6))
        train_x = torch.as_tensor((features - self.x_mean) / self.x_scale, dtype=torch.float32, device=self.device)
        train_y = torch.as_tensor((targets - self.y_mean) / self.y_scale, dtype=torch.float32, device=self.device)
        validation_x, validation_y = validation
        validation_x_t = torch.as_tensor(
            (validation_x - self.x_mean) / self.x_scale, dtype=torch.float32, device=self.device
        )
        validation_y_t = torch.as_tensor(
            (validation_y - self.y_mean) / self.y_scale, dtype=torch.float32, device=self.device
        )
        self.model = NeuralMixtureNetwork(features.shape[1], self.n_experts).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2.0e-3, weight_decay=1.0e-5)
        best_state = copy.deepcopy(self.model.state_dict())
        best_loss = float("inf")
        stale = 0
        for _ in range(self.epochs):
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction, probabilities = self.model(train_x)
            balance = (probabilities.mean(dim=0) - 1.0 / self.n_experts).square().mean()
            loss = (prediction - train_y).square().mean() + 0.01 * balance
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            optimizer.step()
            self.model.eval()
            with torch.no_grad():
                validation_prediction, _ = self.model(validation_x_t)
                validation_loss = float((validation_prediction - validation_y_t).square().mean())
            if validation_loss + 1.0e-7 < best_loss:
                best_loss = validation_loss
                best_state = copy.deepcopy(self.model.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= 100:
                break
        self.model.load_state_dict(best_state)
        self.model.eval()

    def predict_with_routing(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None or self.x_mean is None or self.x_scale is None:
            raise RuntimeError("baseline has not been fitted")
        values = torch.as_tensor(
            (features - self.x_mean) / self.x_scale, dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            prediction, probabilities = self.model(values)
        return (
            prediction.cpu().numpy() * self.y_scale + self.y_mean,
            probabilities.cpu().numpy(),
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.predict_with_routing(features)[0]


class NeuralRegressor:
    def __init__(self, kind: str, seed: int, epochs: int = 600, device: str = "cuda") -> None:
        self.kind = kind
        self.seed = seed
        self.epochs = epochs
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.x_mean: np.ndarray | None = None
        self.x_scale: np.ndarray | None = None
        self.y_mean = 0.0
        self.y_scale = 1.0
        self.model: nn.Module | None = None

    def fit(
        self, features: np.ndarray, targets: np.ndarray, validation: tuple[np.ndarray, np.ndarray]
    ) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.x_mean = features.mean(axis=0)
        self.x_scale = features.std(axis=0).clip(1.0e-6)
        self.y_mean = float(targets.mean())
        self.y_scale = float(max(targets.std(), 1.0e-6))
        train_x = torch.as_tensor((features - self.x_mean) / self.x_scale, dtype=torch.float32, device=self.device)
        train_y = torch.as_tensor((targets - self.y_mean) / self.y_scale, dtype=torch.float32, device=self.device)
        validation_features, validation_targets = validation
        validation_x = torch.as_tensor(
            (validation_features - self.x_mean) / self.x_scale, dtype=torch.float32, device=self.device
        )
        validation_y = torch.as_tensor(
            (validation_targets - self.y_mean) / self.y_scale, dtype=torch.float32, device=self.device
        )
        if self.kind == "MLP":
            self.model = nn.Sequential(
                nn.Linear(features.shape[1], 64),
                nn.SiLU(),
                nn.Linear(64, 64),
                nn.SiLU(),
                nn.Linear(64, 64),
                nn.SiLU(),
                nn.Linear(64, 1),
                nn.Flatten(0),
            ).to(self.device)
            l1_weight = 0.0
        elif self.kind == "EQL":
            self.model = EQLNetwork(features.shape[1]).to(self.device)
            l1_weight = 2.0e-6
        else:
            raise ValueError(self.kind)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2.0e-3, weight_decay=1.0e-5)
        best_state = copy.deepcopy(self.model.state_dict())
        best_validation = float("inf")
        stale = 0
        for _ in range(self.epochs):
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = self.model(train_x)
            loss = (prediction - train_y).square().mean()
            if l1_weight:
                loss = loss + l1_weight * sum(parameter.abs().sum() for parameter in self.model.parameters())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            optimizer.step()
            self.model.eval()
            with torch.no_grad():
                validation_loss = float((self.model(validation_x) - validation_y).square().mean())
            if validation_loss + 1.0e-7 < best_validation:
                best_validation = validation_loss
                best_state = copy.deepcopy(self.model.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= 100:
                break
        self.model.load_state_dict(best_state)
        self.model.eval()

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model is None or self.x_mean is None or self.x_scale is None:
            raise RuntimeError("baseline has not been fitted")
        values = torch.as_tensor(
            (features - self.x_mean) / self.x_scale, dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            prediction = self.model(values).detach().cpu().numpy()
        return np.asarray(prediction * self.y_scale + self.y_mean, dtype=np.float64)


def build_baseline(name: str, seed: int, device: str = "cuda") -> Regressor:
    if name == "Linear":
        return SklearnRegressor(LinearRegression())
    if name == "RandomForest":
        return SklearnRegressor(
            RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed)
        )
    if name == "PolynomialSR":
        return PolynomialSymbolicRegressor(degree=3)
    if name in {"MLP", "EQL"}:
        return NeuralRegressor(name, seed=seed, device=device)
    if name == "NeuralMoE":
        return NeuralMixtureRegressor(seed=seed, device=device)
    raise KeyError(f"unknown baseline {name}")
