from __future__ import annotations

import numpy as np
import torch

from tabalu.models.executor import ExecutableProgram, ProgramNode
from tabalu.models.neural_executor import exact_node_data


def test_exact_node_supervision_matches_executor() -> None:
    program = ExecutableProgram(
        3,
        [ProgramNode("multiply", 0, 1), ProgramNode("safe_divide", 3, 2)],
        4,
    )
    features = torch.as_tensor(np.random.default_rng(2).uniform(-2, 2, size=(64, 3)).astype(np.float32))
    nodes = exact_node_data(program, features)
    torch.testing.assert_close(nodes[-1][1], program(features))
    assert nodes[0][0].shape == nodes[1][0].shape == (64, 2)
