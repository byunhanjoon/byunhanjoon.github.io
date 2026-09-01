from __future__ import annotations

import numpy as np
import torch

from tabalu.models.residual import ResidualModule
from tabalu.synthetic.residuals import generate_residual_task, sample_residual_split


def test_alpha_zero_is_exact_symbolic_target() -> None:
    task = generate_residual_task(202611001)
    _, targets, symbolic, residual = sample_residual_split(task, "iid_test", 128, alpha=0.0)
    np.testing.assert_allclose(targets, symbolic)
    assert residual.std() > 0


def test_residual_gate_modes_are_bounded() -> None:
    features = torch.randn(16, 4)
    for mode in ("none", "scalar", "adaptive"):
        contribution, gate = ResidualModule(4, mode)(features)
        assert contribution.shape == gate.shape == (16,)
        assert torch.all((gate >= 0) & (gate <= 1))
