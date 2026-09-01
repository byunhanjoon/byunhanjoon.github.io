from __future__ import annotations

import numpy as np

from tabalu.evaluation import regression_metrics
from tabalu.models.discrete_search import beam_search_chain_program, search_chain_program
from tabalu.models.executor import ExecutableProgram, ProgramNode


def test_chain_search_recovers_exact_extrapolating_computation() -> None:
    truth = ExecutableProgram(
        3,
        [ProgramNode("multiply", 0, 1), ProgramNode("safe_divide", 3, 2)],
        4,
    )
    rng = np.random.default_rng(8)
    train = rng.uniform(-2, 2, size=(256, 3)).astype(np.float32)
    validation = rng.uniform(-2, 2, size=(128, 3)).astype(np.float32)
    recovered = search_chain_program(
        train,
        truth(train),
        validation,
        truth(validation),
        max_depth=2,
        operators=("add", "subtract", "multiply", "safe_divide", "abs", "square"),
    )
    ood = (rng.uniform(3, 16, size=(256, 3)) * rng.choice([-1, 1], size=(256, 3))).astype(
        np.float32
    )
    assert regression_metrics(truth(ood), recovered(ood))["nrmse"] < 1.0e-6


def test_beam_search_recovers_short_chain() -> None:
    rng = np.random.default_rng(12)
    train = rng.uniform(-2, 2, size=(256, 3)).astype(np.float32)
    validation = rng.uniform(-2, 2, size=(128, 3)).astype(np.float32)
    train_y = train[:, 0] * train[:, 1] + train[:, 2]
    validation_y = validation[:, 0] * validation[:, 1] + validation[:, 2]
    program = beam_search_chain_program(
        train,
        train_y,
        validation,
        validation_y,
        max_depth=2,
        operators=("add", "multiply"),
        beam_width=64,
    )
    np.testing.assert_allclose(program(validation), validation_y, atol=1.0e-6)
