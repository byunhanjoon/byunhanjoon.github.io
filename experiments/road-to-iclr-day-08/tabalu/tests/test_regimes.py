from __future__ import annotations

import numpy as np
import torch

from tabalu.models.executor import ExecutableProgram, ProgramNode
from tabalu.models.router import ProgramMixture, SparseProgramRouter
from tabalu.synthetic import generate_regime_task, sample_regime_split


def test_regime_generator_is_reproducible_and_frequency_shifts() -> None:
    task = generate_regime_task(123)
    first = sample_regime_split(
        task, "train", 1000, seed=4, magnitude_multiplier=1, regime_one_probability=0.2
    )
    second = sample_regime_split(
        task, "train", 1000, seed=4, magnitude_multiplier=1, regime_one_probability=0.2
    )
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    assert 0.15 < first[2].mean() < 0.25


def test_program_mixture_hard_routes_between_exact_experts() -> None:
    programs = [
        ExecutableProgram(2, [ProgramNode("add", 0, 1)], 2),
        ExecutableProgram(2, [ProgramNode("multiply", 0, 1)], 2),
    ]
    router = SparseProgramRouter(1, 2, hidden_width=1)
    with torch.no_grad():
        router.network[0].weight.fill_(1.0)
        router.network[0].bias.zero_()
        router.network[2].weight.copy_(torch.tensor([[-5.0], [5.0]]))
        router.network[2].bias.zero_()
    mixture = ProgramMixture(programs, router)
    features = torch.tensor([[2.0, 3.0], [2.0, 3.0]])
    context = torch.tensor([[-2.0], [2.0]])
    prediction, _ = mixture(features, context, hard=True)
    torch.testing.assert_close(prediction, torch.tensor([5.0, 6.0]))
