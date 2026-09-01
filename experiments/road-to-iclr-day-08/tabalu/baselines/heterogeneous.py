"""Neural controls for heterogeneous typed inputs."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from tabalu.models.typed import HeterogeneousBatch, datetime_parts, ordinal_threshold

from .regressors import NeuralRegressor


def manual_preprocessing(batch: HeterogeneousBatch) -> np.ndarray:
    parts = datetime_parts(batch.timestamp_hours)
    categories = np.eye(3, dtype=np.float64)[batch.categorical]
    rank = batch.ordinal.astype(np.float64) / 3.0
    thresholds = np.column_stack([ordinal_threshold(batch.ordinal, level) for level in (1, 2, 3)])
    hour = 2 * np.pi * parts["hour"] / 24
    weekday = 2 * np.pi * parts["weekday"] / 7
    year = 2 * np.pi * parts["day_of_year"] / 365.25
    elapsed = ((parts["elapsed_days"] - 18_262.0) / 365.25)[:, None]
    return np.column_stack(
        (
            batch.continuous,
            categories,
            rank,
            thresholds,
            elapsed,
            np.sin(hour),
            np.cos(hour),
            np.sin(weekday),
            np.cos(weekday),
            np.sin(year),
            np.cos(year),
        )
    )


class ManualPreprocessingMLP:
    def __init__(self, seed: int, device: str, epochs: int) -> None:
        self.regressor = NeuralRegressor("MLP", seed=seed, device=device, epochs=epochs)

    def fit(self, batch: HeterogeneousBatch, targets: np.ndarray, validation: tuple[HeterogeneousBatch, np.ndarray]) -> None:
        validation_batch, validation_targets = validation
        self.regressor.fit(
            manual_preprocessing(batch),
            targets,
            (manual_preprocessing(validation_batch), validation_targets),
        )

    def predict(self, batch: HeterogeneousBatch) -> np.ndarray:
        return self.regressor.predict(manual_preprocessing(batch))


class EmbeddingNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.continuous = nn.Sequential(nn.Linear(2, 8), nn.SiLU())
        self.category = nn.Embedding(3, 4)
        self.ordinal = nn.Embedding(4, 4)
        self.hour = nn.Embedding(24, 4)
        self.weekday = nn.Embedding(7, 3)
        self.month = nn.Embedding(12, 4)
        self.output = nn.Sequential(
            nn.Linear(28, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, values: dict[str, torch.Tensor]) -> torch.Tensor:
        joined = torch.cat(
            (
                self.continuous(values["continuous"]),
                self.category(values["category"]),
                self.ordinal(values["ordinal"]),
                self.hour(values["hour"]),
                self.weekday(values["weekday"]),
                self.month(values["month"]),
                values["elapsed"].unsqueeze(-1),
            ),
            dim=-1,
        )
        return self.output(joined).squeeze(-1)


@dataclass
class EmbeddingFit:
    training_seconds: float
    validation_mse: float


class LearnedEmbeddingRegressor:
    def __init__(self, seed: int, device: str, epochs: int) -> None:
        self.seed = seed
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.continuous_mean = np.zeros(2)
        self.continuous_scale = np.ones(2)
        self.elapsed_mean = 0.0
        self.elapsed_scale = 1.0
        self.target_mean = 0.0
        self.target_scale = 1.0
        self.model: EmbeddingNetwork | None = None

    def _tensor_batch(self, batch: HeterogeneousBatch) -> dict[str, torch.Tensor]:
        parts = datetime_parts(batch.timestamp_hours)
        elapsed = (parts["elapsed_days"] - self.elapsed_mean) / self.elapsed_scale
        return {
            "continuous": torch.as_tensor(
                (batch.continuous - self.continuous_mean) / self.continuous_scale,
                dtype=torch.float32,
                device=self.device,
            ),
            "category": torch.as_tensor(batch.categorical, dtype=torch.long, device=self.device),
            "ordinal": torch.as_tensor(batch.ordinal, dtype=torch.long, device=self.device),
            "hour": torch.as_tensor(parts["hour"], dtype=torch.long, device=self.device),
            "weekday": torch.as_tensor(parts["weekday"], dtype=torch.long, device=self.device),
            "month": torch.as_tensor(parts["month"], dtype=torch.long, device=self.device),
            "elapsed": torch.as_tensor(elapsed, dtype=torch.float32, device=self.device),
        }

    def fit(self, batch: HeterogeneousBatch, targets: np.ndarray, validation: tuple[HeterogeneousBatch, np.ndarray]) -> EmbeddingFit:
        torch.manual_seed(self.seed)
        self.continuous_mean = batch.continuous.mean(axis=0)
        self.continuous_scale = batch.continuous.std(axis=0).clip(1.0e-6)
        train_parts = datetime_parts(batch.timestamp_hours)
        self.elapsed_mean = float(train_parts["elapsed_days"].mean())
        self.elapsed_scale = max(float(train_parts["elapsed_days"].std()), 1.0)
        self.target_mean = float(targets.mean())
        self.target_scale = max(float(targets.std()), 1.0e-6)
        train_x = self._tensor_batch(batch)
        train_y = torch.as_tensor((targets - self.target_mean) / self.target_scale, dtype=torch.float32, device=self.device)
        validation_batch, validation_targets = validation
        validation_x = self._tensor_batch(validation_batch)
        validation_y = torch.as_tensor(
            (validation_targets - self.target_mean) / self.target_scale,
            dtype=torch.float32,
            device=self.device,
        )
        self.model = EmbeddingNetwork().to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2.0e-3, weight_decay=1.0e-5)
        best_state = copy.deepcopy(self.model.state_dict())
        best = float("inf")
        stale = 0
        started = time.perf_counter()
        for _ in range(self.epochs):
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = (self.model(train_x) - train_y).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            optimizer.step()
            self.model.eval()
            with torch.no_grad():
                score = float((self.model(validation_x) - validation_y).square().mean())
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
        return EmbeddingFit(time.perf_counter() - started, best)

    def predict(self, batch: HeterogeneousBatch) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("embedding model has not been fitted")
        with torch.no_grad():
            values = self.model(self._tensor_batch(batch)).cpu().numpy()
        return np.asarray(values * self.target_scale + self.target_mean, dtype=np.float64)
