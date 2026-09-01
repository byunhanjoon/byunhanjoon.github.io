"""Prediction and structural metrics for generated programs."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from ..models.executor import COMMUTATIVE_OPERATORS, ExecutableProgram


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    residual = predictions - targets
    mse = float(np.mean(residual**2))
    variance = float(np.var(targets))
    denominator = np.maximum(np.abs(targets), 1.0e-3)
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(mse)),
        "r2": float(1.0 - mse / max(variance, 1.0e-12)),
        "nrmse": float(np.sqrt(mse) / max(np.std(targets), 1.0e-12)),
        "relative_error": float(np.median(np.abs(residual) / denominator)),
    }


def _canonical(program: ExecutableProgram, reference: int) -> tuple[Any, ...]:
    if reference < program.n_features:
        return ("feature", reference)
    node = program.nodes[reference - program.n_features]
    children = [_canonical(program, node.left)]
    if node.right is not None:
        children.append(_canonical(program, node.right))
    if node.operator in COMMUTATIVE_OPERATORS:
        children.sort(key=repr)
    return (node.operator, *children)


def program_recovery_metrics(
    truth: ExecutableProgram, recovered: ExecutableProgram
) -> dict[str, float | bool]:
    truth = truth.compile()
    recovered = recovered.compile()
    truth_features = set(truth.features_used)
    recovered_features = set(recovered.features_used)
    overlap = len(truth_features & recovered_features)
    precision = overlap / max(len(recovered_features), 1)
    recall = overlap / max(len(truth_features), 1)
    truth_operators = Counter(node.operator for node in truth.nodes)
    recovered_operators = Counter(node.operator for node in recovered.nodes)
    operator_overlap = sum((truth_operators & recovered_operators).values())
    operator_accuracy = operator_overlap / max(sum(truth_operators.values()), sum(recovered_operators.values()), 1)
    exact_graph = _canonical(truth, truth.output) == _canonical(recovered, recovered.output)
    return {
        "feature_precision": float(precision),
        "feature_recall": float(recall),
        "feature_f1": float(2 * precision * recall / max(precision + recall, 1.0e-12)),
        "operator_accuracy": float(operator_accuracy),
        "exact_program": bool(exact_graph),
        "depth_error": float(abs(len(truth.nodes) - len(recovered.nodes))),
        "coefficient_error": float(
            abs(truth.output_scale - recovered.output_scale) + abs(truth.output_bias - recovered.output_bias)
        ),
    }
