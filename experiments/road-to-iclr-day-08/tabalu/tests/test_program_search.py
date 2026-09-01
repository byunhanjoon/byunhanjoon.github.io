from __future__ import annotations

import numpy as np
import torch

from tabalu.models.program_search import DifferentiableProgram


def select(logits: torch.Tensor, index: int) -> None:
    with torch.no_grad():
        logits.fill_(-20.0)
        logits[index] = 20.0


def test_hard_selection_compiles_to_same_exact_program() -> None:
    model = DifferentiableProgram(
        n_features=3,
        n_nodes=2,
        operators=("add", "multiply", "safe_divide"),
        selector="softmax",
    )
    select(model.operator_logits[0], 1)
    select(model.left_logits[0], 0)
    select(model.right_logits[0], 1)
    select(model.operator_logits[1], 2)
    select(model.left_logits[1], 3)
    select(model.right_logits[1], 2)
    select(model.output_logits, 4)
    features = torch.tensor([[2.0, 3.0, 1.0], [4.0, -2.0, 2.0]])
    model.eval()
    hard, _ = model(features, temperature=0.1, hard=True)
    compiled = model.compile()
    np.testing.assert_allclose(hard.detach().numpy(), compiled(features).detach().numpy(), rtol=1e-6)
    assert compiled.expression() == "safe_divide(multiply(x0, x1), x2)"


def test_soft_and_hard_modes_are_reportable_separately() -> None:
    model = DifferentiableProgram(2, n_nodes=1, selector="straight_through_gumbel")
    model.eval()
    features = torch.tensor([[0.5, -1.0], [2.0, 1.0]])
    soft, _ = model(features, temperature=1.0, hard=False)
    hard, _ = model(features, temperature=1.0, hard=True)
    assert soft.shape == hard.shape == (2,)
    assert not torch.allclose(soft, hard)
