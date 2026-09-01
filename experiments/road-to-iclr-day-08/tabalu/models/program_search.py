"""Differentiable DAG discovery with soft or straight-through selectors."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .executor import BINARY_OPERATORS, OPERATORS, ExecutableProgram, ProgramNode, apply_operator


@dataclass(frozen=True)
class SelectorDiagnostics:
    operator_entropy: torch.Tensor
    input_entropy: torch.Tensor
    output_entropy: torch.Tensor


def categorical_entropy(logits: torch.Tensor) -> torch.Tensor:
    probabilities = logits.softmax(dim=-1)
    return -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum()


class DifferentiableProgram(nn.Module):
    """A fixed-capacity DAG whose operation and input edges are learned."""

    def __init__(
        self,
        n_features: int,
        n_nodes: int = 3,
        operators: tuple[str, ...] = OPERATORS,
        selector: str = "straight_through_gumbel",
        epsilon: float = 0.25,
    ) -> None:
        super().__init__()
        if selector not in {"softmax", "gumbel", "straight_through_gumbel"}:
            raise ValueError(f"unknown selector {selector}")
        if any(operator not in OPERATORS for operator in operators):
            raise ValueError("operator library contains an unsupported primitive")
        self.n_features = n_features
        self.n_nodes = n_nodes
        self.operators = tuple(operators)
        self.selector = selector
        self.epsilon = epsilon
        self.operator_logits = nn.ParameterList(
            [nn.Parameter(torch.empty(len(operators))) for _ in range(n_nodes)]
        )
        self.left_logits = nn.ParameterList(
            [nn.Parameter(torch.empty(n_features + index)) for index in range(n_nodes)]
        )
        self.right_logits = nn.ParameterList(
            [nn.Parameter(torch.empty(n_features + index)) for index in range(n_nodes)]
        )
        self.output_logits = nn.Parameter(torch.empty(n_features + n_nodes))
        self.output_scale = nn.Parameter(torch.tensor(1.0))
        self.output_bias = nn.Parameter(torch.tensor(0.0))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in list(self.operator_logits) + list(self.left_logits) + list(self.right_logits):
            nn.init.normal_(parameter, mean=0.0, std=0.05)
        nn.init.normal_(self.output_logits, mean=0.0, std=0.05)
        with torch.no_grad():
            self.output_scale.fill_(1.0)
            self.output_bias.zero_()

    def initialize_from_program(self, program: ExecutableProgram, strength: float = 8.0) -> None:
        """Warm-start selectors from a compatible chain/DAG program."""

        if program.n_features != self.n_features or len(program.nodes) > self.n_nodes:
            raise ValueError("warm-start program is incompatible with model capacity")
        with torch.no_grad():
            for index in range(self.n_nodes):
                self.operator_logits[index].fill_(-strength)
                self.left_logits[index].fill_(-strength)
                self.right_logits[index].fill_(-strength)
                if index < len(program.nodes):
                    node = program.nodes[index]
                    self.operator_logits[index][self.operators.index(node.operator)] = strength
                    self.left_logits[index][node.left] = strength
                    self.right_logits[index][node.right if node.right is not None else node.left] = strength
                else:
                    self.operator_logits[index][self.operators.index("identity")] = strength
                    previous = self.n_features + index - 1
                    self.left_logits[index][previous] = strength
                    self.right_logits[index][previous] = strength
            self.output_logits.fill_(-strength)
            self.output_logits[program.output] = strength
            self.output_scale.fill_(program.output_scale)
            self.output_bias.fill_(program.output_bias)

    def _weights(self, logits: torch.Tensor, temperature: float, hard: bool) -> torch.Tensor:
        if self.selector == "softmax":
            probabilities = (logits / temperature).softmax(dim=-1)
            if not hard:
                return probabilities
            one_hot = F.one_hot(probabilities.argmax(), probabilities.numel()).to(probabilities)
            return one_hot - probabilities.detach() + probabilities
        if self.training:
            use_hard = hard or self.selector == "straight_through_gumbel"
            return F.gumbel_softmax(logits, tau=temperature, hard=use_hard, dim=-1)
        probabilities = (logits / temperature).softmax(dim=-1)
        if hard:
            return F.one_hot(probabilities.argmax(), probabilities.numel()).to(probabilities)
        return probabilities

    def forward(
        self,
        features: torch.Tensor,
        *,
        temperature: float = 1.0,
        hard: bool = False,
    ) -> tuple[torch.Tensor, SelectorDiagnostics]:
        if features.ndim != 2 or features.shape[1] != self.n_features:
            raise ValueError(f"expected [rows, {self.n_features}] input")
        values = [features[:, index] for index in range(self.n_features)]
        operator_entropy = features.new_zeros(())
        input_entropy = features.new_zeros(())
        for node_index in range(self.n_nodes):
            available = torch.stack(values, dim=-1)
            operator_weights = self._weights(self.operator_logits[node_index], temperature, hard)
            left_weights = self._weights(self.left_logits[node_index], temperature, hard)
            right_weights = self._weights(self.right_logits[node_index], temperature, hard)
            left = available @ left_weights
            right = available @ right_weights
            candidates = torch.stack(
                [apply_operator(name, left, right, epsilon=self.epsilon) for name in self.operators],
                dim=-1,
            )
            value = (candidates * operator_weights).sum(dim=-1).clamp(-1.0e5, 1.0e5)
            values.append(value)
            operator_entropy = operator_entropy + categorical_entropy(self.operator_logits[node_index])
            input_entropy = input_entropy + categorical_entropy(self.left_logits[node_index])
            input_entropy = input_entropy + categorical_entropy(self.right_logits[node_index])
        output_weights = self._weights(self.output_logits, temperature, hard)
        output = torch.stack(values, dim=-1) @ output_weights
        prediction = output * self.output_scale + self.output_bias
        diagnostics = SelectorDiagnostics(
            operator_entropy=operator_entropy / max(self.n_nodes, 1),
            input_entropy=input_entropy / max(2 * self.n_nodes, 1),
            output_entropy=categorical_entropy(self.output_logits),
        )
        return prediction, diagnostics

    def compile(self) -> ExecutableProgram:
        nodes: list[ProgramNode] = []
        for index in range(self.n_nodes):
            operator = self.operators[int(self.operator_logits[index].argmax())]
            left = int(self.left_logits[index].argmax())
            right = int(self.right_logits[index].argmax()) if operator in BINARY_OPERATORS else None
            nodes.append(ProgramNode(operator, left, right))
        program = ExecutableProgram(
            n_features=self.n_features,
            nodes=nodes,
            output=int(self.output_logits.argmax()),
            output_scale=float(self.output_scale.detach()),
            output_bias=float(self.output_bias.detach()),
            epsilon=self.epsilon,
        )
        return program.compile()

    def selector_probabilities(self) -> dict[str, list[list[float]] | list[float]]:
        return {
            "operators": [logits.softmax(-1).detach().cpu().tolist() for logits in self.operator_logits],
            "left": [logits.softmax(-1).detach().cpu().tolist() for logits in self.left_logits],
            "right": [logits.softmax(-1).detach().cpu().tolist() for logits in self.right_logits],
            "output": self.output_logits.softmax(-1).detach().cpu().tolist(),
        }
