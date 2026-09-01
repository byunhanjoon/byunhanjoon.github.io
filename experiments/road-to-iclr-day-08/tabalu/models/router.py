"""Sparse regime router and executable-program mixture."""

from __future__ import annotations

import torch
from torch import nn

from .executor import ExecutableProgram


class SparseProgramRouter(nn.Module):
    def __init__(self, context_width: int, n_regimes: int, hidden_width: int = 16) -> None:
        super().__init__()
        self.n_regimes = n_regimes
        self.network = nn.Sequential(
            nn.Linear(context_width, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, n_regimes),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.network(context)

    def probabilities(self, context: torch.Tensor) -> torch.Tensor:
        return self(context).softmax(dim=-1)


class ProgramMixture(nn.Module):
    def __init__(self, programs: list[ExecutableProgram], router: SparseProgramRouter) -> None:
        super().__init__()
        if len(programs) != router.n_regimes:
            raise ValueError("program/router regime count mismatch")
        self.programs = programs
        self.router = router

    def forward(
        self, features: torch.Tensor, context: torch.Tensor, *, hard: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probabilities = self.router.probabilities(context)
        expert_predictions = torch.stack([program(features) for program in self.programs], dim=-1)
        if hard:
            weights = torch.nn.functional.one_hot(
                probabilities.argmax(dim=-1), len(self.programs)
            ).to(probabilities)
        else:
            weights = probabilities
        return (expert_predictions * weights).sum(dim=-1), probabilities
