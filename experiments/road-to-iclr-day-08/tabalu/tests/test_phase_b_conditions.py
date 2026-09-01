from __future__ import annotations

import numpy as np

from tabalu.models.executor import ExecutableProgram, ProgramNode
from tabalu.scripts.run_phase_b_recovery import prepare_condition, rebase_program


def test_rebase_preserves_program_with_extra_inputs() -> None:
    program = ExecutableProgram(2, [ProgramNode("multiply", 0, 1)], 2)
    rebased = rebase_program(program, 5)
    features = np.random.default_rng(3).normal(size=(32, 5)).astype(np.float32)
    np.testing.assert_allclose(program(features[:, :2]), rebased(features))
    assert rebased.output == 5


def test_irrelevant_condition_keeps_targets_and_expands_width() -> None:
    truth = ExecutableProgram(2, [ProgramNode("add", 0, 1)], 2)
    rng = np.random.default_rng(8)
    values = rng.normal(size=(16, 2)).astype(np.float32)
    targets = truth(values)
    conditioned_truth, train, validation, test = prepare_condition(
        "irrelevant_features",
        {"count": 3},
        0,
        truth,
        (values, targets),
        (values, targets),
        (values, targets),
        8,
    )
    assert train[0].shape[1] == validation[0].shape[1] == test[0].shape[1] == 5
    np.testing.assert_array_equal(train[1], targets)
    np.testing.assert_allclose(conditioned_truth(train[0]), targets)
