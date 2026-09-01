"""Typed deterministic numerical program execution and compilation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import torch


UNARY_OPERATORS = ("identity", "abs", "square", "safe_sqrt", "safe_log")
BINARY_OPERATORS = ("add", "subtract", "multiply", "safe_divide", "min", "max")
OPERATORS = UNARY_OPERATORS + BINARY_OPERATORS
COMMUTATIVE_OPERATORS = {"add", "multiply", "min", "max"}


def protected_denominator(value: torch.Tensor, epsilon: float) -> torch.Tensor:
    sign = torch.where(value < 0, -torch.ones_like(value), torch.ones_like(value))
    return sign * value.abs().clamp_min(epsilon)


def apply_operator(
    name: str,
    left: torch.Tensor,
    right: torch.Tensor | None = None,
    *,
    epsilon: float = 0.25,
    value_limit: float = 1.0e6,
) -> torch.Tensor:
    """Apply one protected primitive without learned arithmetic."""

    if name == "identity":
        result = left
    elif name == "abs":
        result = left.abs()
    elif name == "square":
        result = left.square()
    elif name == "safe_sqrt":
        result = left.abs().clamp_min(epsilon).sqrt()
    elif name == "safe_log":
        result = left.abs().clamp_min(epsilon).log()
    else:
        if right is None:
            raise ValueError(f"binary operator {name!r} requires a right operand")
        if name == "add":
            result = left + right
        elif name == "subtract":
            result = left - right
        elif name == "multiply":
            result = left * right
        elif name == "safe_divide":
            result = left / protected_denominator(right, epsilon)
        elif name == "min":
            result = torch.minimum(left, right)
        elif name == "max":
            result = torch.maximum(left, right)
        else:
            raise KeyError(f"unknown operator: {name}")
    return torch.nan_to_num(result, nan=0.0, posinf=value_limit, neginf=-value_limit).clamp(
        -value_limit, value_limit
    )


@dataclass(frozen=True)
class ProgramNode:
    operator: str
    left: int
    right: int | None = None

    def __post_init__(self) -> None:
        if self.operator not in OPERATORS:
            raise ValueError(f"unsupported operator {self.operator!r}")
        if self.operator in BINARY_OPERATORS and self.right is None:
            raise ValueError(f"binary operator {self.operator!r} requires right")


@dataclass
class ExecutableProgram:
    """A DAG whose references index inputs first and then preceding nodes."""

    n_features: int
    nodes: list[ProgramNode]
    output: int
    output_scale: float = 1.0
    output_bias: float = 0.0
    epsilon: float = 0.25

    def __post_init__(self) -> None:
        if self.n_features < 1:
            raise ValueError("n_features must be positive")
        for index, node in enumerate(self.nodes):
            upper = self.n_features + index
            refs = [node.left]
            if node.operator in BINARY_OPERATORS:
                refs.append(node.right)
            if any(ref is None or ref < 0 or ref >= upper for ref in refs):
                raise ValueError(f"node {index} contains a forward/invalid reference: {node}")
        if self.output < 0 or self.output >= self.n_features + len(self.nodes):
            raise ValueError("invalid output reference")

    def __call__(self, features: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        was_numpy = isinstance(features, np.ndarray)
        tensor = torch.as_tensor(features)
        if tensor.ndim != 2 or tensor.shape[1] != self.n_features:
            raise ValueError(f"expected [rows, {self.n_features}] features, got {tuple(tensor.shape)}")
        values = [tensor[:, i] for i in range(self.n_features)]
        for node in self.nodes:
            right = values[node.right] if node.right is not None else None
            values.append(apply_operator(node.operator, values[node.left], right, epsilon=self.epsilon))
        result = values[self.output] * self.output_scale + self.output_bias
        if was_numpy:
            return result.detach().cpu().numpy()
        return result

    @property
    def operation_count(self) -> int:
        return len(self.compile().nodes)

    @property
    def features_used(self) -> tuple[int, ...]:
        compiled = self.compile()
        used: set[int] = set()
        for node in compiled.nodes:
            for ref in (node.left, node.right):
                if ref is not None and ref < self.n_features:
                    used.add(ref)
        if compiled.output < self.n_features:
            used.add(compiled.output)
        return tuple(sorted(used))

    def compile(self) -> "ExecutableProgram":
        """Prune unreachable nodes, eliminate identities, and merge equal nodes."""

        old_to_new: dict[int, int] = {i: i for i in range(self.n_features)}
        expression_to_ref: dict[tuple[Any, ...], int] = {}

        def materialize(reference: int) -> int:
            if reference in old_to_new:
                return old_to_new[reference]
            old_node_index = reference - self.n_features
            node = self.nodes[old_node_index]
            left = materialize(node.left)
            right = materialize(node.right) if node.right is not None else None
            if node.operator == "identity":
                old_to_new[reference] = left
                return left
            if node.operator in COMMUTATIVE_OPERATORS and right is not None and right < left:
                left, right = right, left
            key = (node.operator, left, right)
            if key in expression_to_ref:
                new_ref = expression_to_ref[key]
            else:
                new_ref = self.n_features + len(new_nodes)
                new_nodes.append(ProgramNode(node.operator, left, right))
                expression_to_ref[key] = new_ref
            old_to_new[reference] = new_ref
            return new_ref

        new_nodes: list[ProgramNode] = []
        output = materialize(self.output)
        return ExecutableProgram(
            n_features=self.n_features,
            nodes=new_nodes,
            output=output,
            output_scale=float(self.output_scale),
            output_bias=float(self.output_bias),
            epsilon=float(self.epsilon),
        )

    def fit_output_affine(self, features: np.ndarray, targets: np.ndarray) -> "ExecutableProgram":
        """Fine-tune only the compiled program's scalar output constants."""

        base = self.compile()
        base.output_scale = 1.0
        base.output_bias = 0.0
        values = np.asarray(base(features), dtype=np.float64)
        design = np.column_stack([values, np.ones_like(values)])
        scale, bias = np.linalg.lstsq(design, np.asarray(targets, dtype=np.float64), rcond=None)[0]
        base.output_scale = float(scale)
        base.output_bias = float(bias)
        return base

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_features": self.n_features,
            "nodes": [asdict(node) for node in self.nodes],
            "output": self.output,
            "output_scale": self.output_scale,
            "output_bias": self.output_bias,
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ExecutableProgram":
        return cls(
            n_features=int(value["n_features"]),
            nodes=[ProgramNode(**node) for node in value["nodes"]],
            output=int(value["output"]),
            output_scale=float(value.get("output_scale", 1.0)),
            output_bias=float(value.get("output_bias", 0.0)),
            epsilon=float(value.get("epsilon", 0.25)),
        )

    def expression(self, feature_names: Iterable[str] | None = None) -> str:
        names = list(feature_names or [f"x{i}" for i in range(self.n_features)])
        values = names[:]
        for node in self.nodes:
            left = values[node.left]
            right = values[node.right] if node.right is not None else None
            if node.operator == "identity":
                expr = left
            elif node.operator in UNARY_OPERATORS:
                expr = f"{node.operator}({left})"
            else:
                expr = f"{node.operator}({left}, {right})"
            values.append(expr)
        result = values[self.output]
        if self.output_scale != 1.0 or self.output_bias != 0.0:
            result = f"({self.output_scale:.8g} * {result} + {self.output_bias:.8g})"
        return result
