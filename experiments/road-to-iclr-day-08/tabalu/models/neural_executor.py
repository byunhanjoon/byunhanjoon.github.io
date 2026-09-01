"""Oracle-graph executor whose arithmetic nodes are learned small MLPs."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .executor import BINARY_OPERATORS, ExecutableProgram, apply_operator


class PrimitiveApproximator(nn.Module):
    def __init__(
        self,
        input_mean: torch.Tensor,
        input_scale: torch.Tensor,
        output_mean: torch.Tensor,
        output_scale: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("input_mean", input_mean)
        self.register_buffer("input_scale", input_scale.clamp_min(1.0e-6))
        self.register_buffer("output_mean", output_mean)
        self.register_buffer("output_scale", output_scale.clamp_min(1.0e-6))
        width = len(input_mean)
        self.network = nn.Sequential(
            nn.Linear(width, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, operands: torch.Tensor) -> torch.Tensor:
        normalized = (operands - self.input_mean) / self.input_scale
        return self.network(normalized).squeeze(-1) * self.output_scale + self.output_mean


class NeuralPrimitiveExecutor(nn.Module):
    def __init__(self, program: ExecutableProgram, approximators: list[PrimitiveApproximator]) -> None:
        super().__init__()
        if len(program.nodes) != len(approximators):
            raise ValueError("one approximator is required per executable node")
        self.program = program
        self.approximators = nn.ModuleList(approximators)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        values = [features[:, index] for index in range(self.program.n_features)]
        for node, approximator in zip(self.program.nodes, self.approximators):
            operands = [values[node.left]]
            if node.operator in BINARY_OPERATORS:
                assert node.right is not None
                operands.append(values[node.right])
            values.append(approximator(torch.stack(operands, dim=-1)))
        return values[self.program.output] * self.program.output_scale + self.program.output_bias


def exact_node_data(program: ExecutableProgram, features: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
    values = [features[:, index] for index in range(program.n_features)]
    output: list[tuple[torch.Tensor, torch.Tensor]] = []
    for node in program.nodes:
        operands = [values[node.left]]
        right = None
        if node.operator in BINARY_OPERATORS:
            assert node.right is not None
            right = values[node.right]
            operands.append(right)
        result = apply_operator(node.operator, values[node.left], right, epsilon=program.epsilon)
        output.append((torch.stack(operands, dim=-1), result))
        values.append(result)
    return output


@dataclass
class TrainedNeuralExecutor:
    model: NeuralPrimitiveExecutor
    training_seconds: float
    node_validation_nrmse: list[float]


def train_neural_primitive_executor(
    program: ExecutableProgram,
    train_features: np.ndarray,
    validation_features: np.ndarray,
    *,
    seed: int,
    epochs: int,
    device: str,
) -> TrainedNeuralExecutor:
    torch.manual_seed(seed)
    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    train_t = torch.as_tensor(train_features, dtype=torch.float32, device=torch_device)
    validation_t = torch.as_tensor(validation_features, dtype=torch.float32, device=torch_device)
    train_nodes = exact_node_data(program, train_t)
    validation_nodes = exact_node_data(program, validation_t)
    approximators: list[PrimitiveApproximator] = []
    validation_scores: list[float] = []
    started = time.perf_counter()
    for node_index, ((train_operands, train_target), (val_operands, val_target)) in enumerate(
        zip(train_nodes, validation_nodes)
    ):
        torch.manual_seed(seed * 101 + node_index)
        approximator = PrimitiveApproximator(
            train_operands.mean(dim=0),
            train_operands.std(dim=0, unbiased=False),
            train_target.mean(),
            train_target.std(unbiased=False),
        ).to(torch_device)
        optimizer = torch.optim.AdamW(approximator.parameters(), lr=0.002, weight_decay=1.0e-5)
        best_state = copy.deepcopy(approximator.state_dict())
        best = float("inf")
        stale = 0
        scale = val_target.std(unbiased=False).clamp_min(1.0e-8)
        for _ in range(epochs):
            approximator.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = approximator(train_operands)
            loss = (prediction - train_target).square().mean() / train_target.var(
                unbiased=False
            ).clamp_min(1.0e-8)
            loss.backward()
            optimizer.step()
            approximator.eval()
            with torch.no_grad():
                score = float((approximator(val_operands) - val_target).square().mean().sqrt() / scale)
            if score + 1.0e-7 < best:
                best = score
                best_state = copy.deepcopy(approximator.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= 80:
                break
        approximator.load_state_dict(best_state)
        approximator.eval()
        approximators.append(approximator)
        validation_scores.append(best)
    model = NeuralPrimitiveExecutor(program, approximators).to(torch_device)
    model.eval()
    return TrainedNeuralExecutor(model, time.perf_counter() - started, validation_scores)
