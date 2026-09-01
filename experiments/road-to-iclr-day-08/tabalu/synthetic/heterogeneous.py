"""Synthetic heterogeneous tasks with continuous, category, ordinal and time semantics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tabalu.models.typed import HeterogeneousBatch, typed_design_matrix


@dataclass(frozen=True)
class HeterogeneousTask:
    task_id: str
    seed: int
    active_feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]


def generate_heterogeneous_task(seed: int) -> HeterogeneousTask:
    rng = np.random.default_rng(seed)
    # Every task needs each type to test family removals; exact choices vary.
    active = (
        rng.choice(("x0*x1", "x0/x1", "x0+x1")),
        f"[cat=={rng.integers(0, 3)}]*({rng.choice(('x0*x1', 'x0/x1', 'x0+x1'))})",
        rng.choice(("rank", "rank>=2", "bounded_rank_diff", "rank*x0")),
        rng.choice(("sin(hour)", "cos(hour)")),
        rng.choice(("sin(weekday)", "cos(weekday)")),
        rng.choice(("elapsed", "sin(year)", "cos(year)")),
    )
    coefficients = tuple(rng.choice((-1.0, 1.0)) * rng.uniform(0.7, 1.8) for _ in active)
    return HeterogeneousTask(f"heterogeneous-{seed}", seed, active, coefficients)


def sample_heterogeneous_split(
    task: HeterogeneousTask,
    split: str,
    n_rows: int,
    *,
    seed: int,
    magnitude_multiplier: float = 1.0,
) -> tuple[HeterogeneousBatch, np.ndarray]:
    split_offset = {"train": 11, "validation": 23, "iid": 37, "future": 53}[split]
    rng = np.random.default_rng(task.seed * 1009 + seed * 67 + split_offset)
    continuous = rng.uniform(-2, 2, size=(n_rows, 2)) * magnitude_multiplier
    # Keep protected-division denominators away from zero in the clean typed test.
    continuous[:, 1] = np.sign(continuous[:, 1]) * np.maximum(np.abs(continuous[:, 1]), 0.35)
    categorical = rng.integers(0, 3, size=n_rows)
    ordinal = rng.integers(0, 4, size=n_rows)
    if split == "future":
        start, stop = 24 * 548, 24 * 730  # July–December 2021.
    else:
        start, stop = 0, 24 * 365  # Calendar year 2020.
    timestamp_hours = 24 * 18_262 + rng.integers(start, stop, size=n_rows)
    batch = HeterogeneousBatch(continuous, categorical, ordinal, timestamp_hours)
    design, names = typed_design_matrix(batch)
    lookup = {name: design[:, index] for index, name in enumerate(names)}
    targets = sum(coefficient * lookup[name] for name, coefficient in zip(task.active_feature_names, task.coefficients))
    return batch, np.asarray(targets, dtype=np.float64)
