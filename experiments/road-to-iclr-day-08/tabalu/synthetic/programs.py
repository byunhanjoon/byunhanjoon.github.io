"""Regenerable numerical DAG tasks for the arithmetic extrapolation gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..models.executor import BINARY_OPERATORS, ExecutableProgram, ProgramNode


PHASE_A_OPERATORS = ("add", "subtract", "multiply", "safe_divide", "abs", "square")


@dataclass(frozen=True)
class SyntheticTask:
    task_id: str
    seed: int
    program: ExecutableProgram
    n_irrelevant: int
    generation_attempt: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "program": self.program.to_dict(),
            "n_irrelevant": self.n_irrelevant,
            "generation_attempt": self.generation_attempt,
            "expression": self.program.expression(),
        }


def _sample_inputs(rng: np.random.Generator, rows: int, n_features: int, multiplier: float) -> np.ndarray:
    if multiplier <= 1.0:
        return rng.uniform(-2.0, 2.0, size=(rows, n_features)).astype(np.float32)
    lower = 2.25
    upper = 2.0 * multiplier
    magnitudes = rng.uniform(lower, upper, size=(rows, n_features))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(rows, n_features))
    return (magnitudes * signs).astype(np.float32)


def regenerate_split(
    task: SyntheticTask,
    split: str,
    rows: int,
    multiplier: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = {"train": 11, "validation": 23, "iid_test": 37, "ood_test": 51}
    if split not in offsets:
        raise ValueError(f"unknown split {split}")
    split_seed = task.seed * 1009 + offsets[split] + int(multiplier * 100)
    rng = np.random.default_rng(split_seed)
    features = _sample_inputs(rng, rows, task.program.n_features, multiplier)
    targets = np.asarray(task.program(features), dtype=np.float32)
    return features, targets


def _random_program(
    rng: np.random.Generator,
    n_features: int,
    n_relevant_features: int,
    depth: int,
    operators: tuple[str, ...],
) -> ExecutableProgram:
    nodes: list[ProgramNode] = []
    previous_output: int | None = None
    for node_index in range(depth):
        upper = n_features + node_index
        operator = str(rng.choice(operators))
        left = (
            int(rng.integers(0, n_relevant_features))
            if previous_output is None
            else previous_output
        )
        right = None
        if operator in BINARY_OPERATORS:
            candidates = [index for index in range(n_relevant_features) if index != left]
            right = int(rng.choice(candidates)) if candidates else left
        nodes.append(ProgramNode(operator, left, right))
        previous_output = n_features + node_index
    return ExecutableProgram(n_features=n_features, nodes=nodes, output=n_features + depth - 1).compile()


def generate_program_task(
    seed: int,
    *,
    n_features: int = 4,
    depth_range: tuple[int, int] = (1, 3),
    n_irrelevant: int = 0,
    operators: tuple[str, ...] = PHASE_A_OPERATORS,
    max_attempts: int = 100,
) -> SyntheticTask:
    """Generate and numerically screen a ground-truth executable task."""

    rng = np.random.default_rng(seed)
    total_features = n_features + n_irrelevant
    for attempt in range(max_attempts):
        depth = int(rng.integers(depth_range[0], depth_range[1] + 1))
        program = _random_program(rng, total_features, n_features, depth, operators)
        stable = True
        for multiplier in (1.0, 2.0, 4.0, 8.0):
            probe = _sample_inputs(rng, 2048, total_features, multiplier=multiplier)
            values = np.asarray(program(probe), dtype=np.float64)
            if (
                not np.isfinite(values).all()
                or np.std(values) < 0.10
                or np.quantile(np.abs(values), 0.999) > 2.0e4
            ):
                stable = False
                break
        if not stable:
            continue
        task_id = f"program-{seed:06d}"
        return SyntheticTask(task_id, seed, program, n_irrelevant, attempt)
    raise RuntimeError(f"failed to generate a stable task after {max_attempts} attempts")
