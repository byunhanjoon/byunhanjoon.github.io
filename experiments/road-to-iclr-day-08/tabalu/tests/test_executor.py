from __future__ import annotations

import json

import numpy as np
import torch

from tabalu.models.executor import ExecutableProgram, ProgramNode, apply_operator


def test_protected_operators_are_finite() -> None:
    values = torch.tensor([-2.0, 0.0, 3.0])
    for operator in ("safe_divide", "safe_sqrt", "safe_log"):
        result = apply_operator(operator, values, values)
        assert torch.isfinite(result).all()


def test_executor_matches_known_arithmetic_program() -> None:
    program = ExecutableProgram(
        n_features=3,
        nodes=[ProgramNode("multiply", 0, 1), ProgramNode("safe_divide", 3, 2)],
        output=4,
    )
    features = np.array([[2.0, 3.0, 1.0], [4.0, -2.0, 2.0]], dtype=np.float32)
    np.testing.assert_allclose(program(features), np.array([6.0, -4.0]), rtol=1e-6)


def test_compiler_prunes_identity_unused_and_duplicate_nodes() -> None:
    program = ExecutableProgram(
        n_features=2,
        nodes=[
            ProgramNode("add", 0, 1),
            ProgramNode("identity", 2),
            ProgramNode("add", 1, 0),
            ProgramNode("multiply", 3, 4),
            ProgramNode("square", 0),
        ],
        output=5,
    )
    compiled = program.compile()
    assert len(compiled.nodes) == 2
    assert compiled.nodes[0] == ProgramNode("add", 0, 1)
    assert compiled.nodes[1].operator == "multiply"
    features = np.random.default_rng(4).normal(size=(32, 2)).astype(np.float32)
    np.testing.assert_allclose(program(features), compiled(features), rtol=1e-6, atol=1e-6)


def test_program_json_round_trip() -> None:
    program = ExecutableProgram(2, [ProgramNode("subtract", 0, 1)], 2, 2.5, -0.3)
    restored = ExecutableProgram.from_dict(json.loads(json.dumps(program.to_dict())))
    assert restored == program
