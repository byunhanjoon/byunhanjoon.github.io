"""Multi-regime arithmetic tasks for router-count scaling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tabalu.models.executor import ExecutableProgram

from .programs import generate_program_task


@dataclass(frozen=True)
class MultiRegimeTask:
    task_id: str
    seed: int
    programs: tuple[ExecutableProgram, ...]

    @property
    def n_regimes(self) -> int:
        return len(self.programs)


def generate_multi_regime_task(seed: int, n_regimes: int) -> MultiRegimeTask:
    programs = tuple(
        generate_program_task(
            seed * 31 + regime,
            n_features=4,
            depth_range=(1, 2),
            operators=("add", "subtract", "multiply", "safe_divide", "abs", "square"),
        ).program
        for regime in range(n_regimes)
    )
    return MultiRegimeTask(f"regime-scale-{seed}-r{n_regimes}", seed, programs)


def sample_multi_regime_split(
    task: MultiRegimeTask,
    split: str,
    rows: int,
    *,
    seed: int,
    magnitude_multiplier: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    offset = {"train": 11, "validation": 23, "iid": 37, "ood": 51}[split]
    rng = np.random.default_rng(task.seed * 1009 + seed * 67 + offset)
    if split in {"train", "validation", "iid"}:
        regimes = np.resize(np.arange(task.n_regimes), rows)
        rng.shuffle(regimes)
        features = rng.uniform(-2, 2, size=(rows, 4)).astype(np.float32)
    else:
        weights = 2.0 ** np.arange(task.n_regimes)
        weights /= weights.sum()
        regimes = rng.choice(task.n_regimes, size=rows, p=weights)
        magnitudes = rng.uniform(2.25, 2.0 * magnitude_multiplier, size=(rows, 4))
        features = (magnitudes * rng.choice((-1.0, 1.0), size=(rows, 4))).astype(np.float32)
    context = np.eye(task.n_regimes, dtype=np.float32)[regimes]
    targets = np.empty(rows, dtype=np.float64)
    for regime, program in enumerate(task.programs):
        mask = regimes == regime
        targets[mask] = np.asarray(program(features[mask]), dtype=np.float64)
    return features, context, regimes.astype(np.int64), targets
