"""Targets with a controlled non-symbolic fraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .programs import SyntheticTask, generate_program_task, regenerate_split


@dataclass(frozen=True)
class ResidualTask:
    symbolic_task: SyntheticTask
    residual_mean: float
    residual_scale: float
    symbolic_scale: float

    @property
    def task_id(self) -> str:
        return f"residual-{self.symbolic_task.seed}"


def _raw_non_symbolic(features: np.ndarray) -> np.ndarray:
    return (
        np.sin(1.3 * features[:, 0] * features[:, 1])
        + 0.55 * np.tanh(features[:, 2] ** 2 - 0.7)
        + 0.35 * np.cos(features[:, 3] + 0.5 * features[:, 0])
    )


def generate_residual_task(seed: int) -> ResidualTask:
    symbolic = generate_program_task(seed, n_features=4)
    reference_x, symbolic_y = regenerate_split(symbolic, "train", 8192)
    raw = _raw_non_symbolic(reference_x)
    return ResidualTask(
        symbolic,
        float(raw.mean()),
        max(float(raw.std()), 1.0e-6),
        max(float(symbolic_y.std()), 1.0e-6),
    )


def sample_residual_split(
    task: ResidualTask,
    split: str,
    rows: int,
    *,
    alpha: float,
    magnitude_multiplier: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features, symbolic = regenerate_split(
        task.symbolic_task,
        "ood_test" if split == "ood" else split,
        rows,
        magnitude_multiplier,
    )
    residual = (
        (_raw_non_symbolic(features) - task.residual_mean)
        / task.residual_scale
        * task.symbolic_scale
    )
    targets = symbolic.astype(np.float64) + alpha * residual
    return features, targets, symbolic.astype(np.float64), residual.astype(np.float64)
