"""Known categorical-regime program mixtures for the Phase-D pilot."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .programs import PHASE_A_OPERATORS, generate_program_task
from ..models.executor import ExecutableProgram


@dataclass(frozen=True)
class RegimeTask:
    task_id: str
    seed: int
    programs: tuple[ExecutableProgram, ExecutableProgram]


def generate_regime_task(seed: int, n_features: int = 4) -> RegimeTask:
    first = generate_program_task(
        seed * 2, n_features=n_features, depth_range=(1, 3), operators=PHASE_A_OPERATORS
    ).program
    offset = 1
    while True:
        second = generate_program_task(
            seed * 2 + offset,
            n_features=n_features,
            depth_range=(1, 3),
            operators=PHASE_A_OPERATORS,
        ).program
        if second.expression() != first.expression():
            break
        offset += 1
    rng = np.random.default_rng(seed + 913)
    probe = rng.uniform(-2, 2, size=(4096, n_features)).astype(np.float32)
    normalized: list[ExecutableProgram] = []
    for program in (first, second):
        scale = max(float(np.std(program(probe))), 1.0e-3)
        normalized.append(
            ExecutableProgram(
                program.n_features,
                program.nodes,
                program.output,
                program.output_scale / scale,
                program.output_bias,
                program.epsilon,
            )
        )
    return RegimeTask(f"regime-{seed:06d}", seed, (normalized[0], normalized[1]))


def sample_regime_split(
    task: RegimeTask,
    split: str,
    rows: int,
    *,
    seed: int,
    magnitude_multiplier: float,
    regime_one_probability: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    offsets = {"train": 11, "validation": 29, "iid_test": 47, "ood_test": 71}
    rng = np.random.default_rng(task.seed * 10007 + seed * 101 + offsets[split])
    if magnitude_multiplier <= 1:
        features = rng.uniform(-2, 2, size=(rows, task.programs[0].n_features))
    else:
        magnitude = rng.uniform(2.25, 2 * magnitude_multiplier, size=(rows, task.programs[0].n_features))
        features = magnitude * rng.choice([-1.0, 1.0], size=magnitude.shape)
    features = features.astype(np.float32)
    regimes = rng.binomial(1, regime_one_probability, size=rows).astype(np.int64)
    context = ((2 * regimes - 1) + rng.normal(0, 0.10, size=rows)).astype(np.float32)[:, None]
    expert_predictions = np.stack([program(features) for program in task.programs], axis=-1)
    targets = expert_predictions[np.arange(rows), regimes].astype(np.float32)
    return features, context, regimes, targets
