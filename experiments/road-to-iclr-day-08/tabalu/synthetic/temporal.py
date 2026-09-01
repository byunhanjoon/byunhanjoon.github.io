"""Shared-structure, changing-coefficient temporal tasks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .programs import PHASE_A_OPERATORS, generate_program_task
from ..models.executor import ExecutableProgram


@dataclass(frozen=True)
class TemporalCoefficientTask:
    task_id: str
    seed: int
    program: ExecutableProgram
    scales: tuple[float, float]
    biases: tuple[float, float]
    change_point: float


def generate_temporal_task(seed: int, n_features: int = 4, change_point: float = 0.70) -> TemporalCoefficientTask:
    program = generate_program_task(
        seed, n_features=n_features, depth_range=(1, 3), operators=PHASE_A_OPERATORS
    ).program
    rng = np.random.default_rng(seed + 511)
    probe = rng.uniform(-2, 2, size=(4096, n_features)).astype(np.float32)
    program = ExecutableProgram(
        n_features,
        program.nodes,
        program.output,
        program.output_scale / max(float(np.std(program(probe))), 1.0e-3),
        program.output_bias,
        program.epsilon,
    )
    scales = (float(rng.uniform(0.6, 1.0)), float(rng.uniform(1.5, 2.5)))
    biases = (float(rng.uniform(-0.3, 0.3)), float(rng.uniform(-0.8, 0.8)))
    return TemporalCoefficientTask(
        f"temporal-{seed:06d}", seed, program, scales, biases, change_point
    )


def sample_temporal_split(
    task: TemporalCoefficientTask,
    split: str,
    rows: int,
    *,
    seed: int,
    magnitude_multiplier: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    offsets = {"train": 13, "validation": 31, "iid_test": 53, "future_test": 79}
    rng = np.random.default_rng(task.seed * 10007 + seed * 107 + offsets[split])
    if split == "future_test":
        time = rng.uniform(0.80, 1.0, size=rows)
    else:
        time = rng.uniform(0.0, 0.80, size=rows)
    regimes = (time >= task.change_point).astype(np.int64)
    if magnitude_multiplier <= 1:
        features = rng.uniform(-2, 2, size=(rows, task.program.n_features))
    else:
        magnitude = rng.uniform(2.25, 2 * magnitude_multiplier, size=(rows, task.program.n_features))
        features = magnitude * rng.choice([-1.0, 1.0], size=magnitude.shape)
    features = features.astype(np.float32)
    base = np.asarray(task.program(features), dtype=np.float64)
    scales = np.asarray(task.scales)
    biases = np.asarray(task.biases)
    targets = (scales[regimes] * base + biases[regimes]).astype(np.float32)
    return features, time.astype(np.float32)[:, None], regimes, targets
