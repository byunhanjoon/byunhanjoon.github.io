"""Exact chain-DAG search used as a Phase-A selector warm start.

This is deliberately constrained to the same short chain family used by the
first falsification panel.  It is not presented as a general symbolic-regression
algorithm; it isolates execution from the harder general-DAG search problem.
"""

from __future__ import annotations

import numpy as np

from .executor import BINARY_OPERATORS, ExecutableProgram, ProgramNode


def _apply_numpy(name: str, left: np.ndarray, right: np.ndarray | None, epsilon: float) -> np.ndarray:
    if name == "identity":
        result = left
    elif name == "abs":
        result = np.abs(left)
    elif name == "square":
        result = np.square(left)
    elif name == "safe_sqrt":
        result = np.sqrt(np.maximum(np.abs(left), epsilon))
    elif name == "safe_log":
        result = np.log(np.maximum(np.abs(left), epsilon))
    elif name == "add":
        result = left + right
    elif name == "subtract":
        result = left - right
    elif name == "multiply":
        result = left * right
    elif name == "safe_divide":
        assert right is not None
        denominator = np.where(right < 0, -1.0, 1.0) * np.maximum(np.abs(right), epsilon)
        result = left / denominator
    elif name == "min":
        result = np.minimum(left, right)
    elif name == "max":
        result = np.maximum(left, right)
    else:
        raise KeyError(name)
    return np.nan_to_num(result, nan=0.0, posinf=1.0e6, neginf=-1.0e6).clip(-1.0e6, 1.0e6)


def _fit_affine(values: np.ndarray, targets: np.ndarray) -> tuple[float, float, float]:
    values64 = values.astype(np.float64, copy=False)
    targets64 = targets.astype(np.float64, copy=False)
    centered = values64 - values64.mean()
    denominator = float(np.dot(centered, centered))
    scale = float(np.dot(centered, targets64 - targets64.mean()) / max(denominator, 1.0e-12))
    bias = float(targets64.mean() - scale * values64.mean())
    residual = scale * values64 + bias - targets64
    nrmse = float(np.sqrt(np.mean(residual**2)) / max(np.std(targets64), 1.0e-12))
    return scale, bias, nrmse


def search_chain_program(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    *,
    max_depth: int,
    operators: tuple[str, ...],
    epsilon: float = 0.25,
    validation_candidates: int = 64,
) -> ExecutableProgram:
    """Enumerate the complete depth-limited chain family and select on validation."""

    n_features = train_features.shape[1]
    previous_values = np.asarray(train_features, dtype=np.float32).T
    level_values: list[np.ndarray] = [previous_values]
    level_metadata: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    finalists: list[tuple[int, int, float]] = []
    for depth in range(1, max_depth + 1):
        value_blocks: list[np.ndarray] = []
        parent_blocks: list[np.ndarray] = []
        operator_blocks: list[np.ndarray] = []
        right_blocks: list[np.ndarray] = []
        parent_count = previous_values.shape[0]
        parent_indices = np.arange(parent_count, dtype=np.int32)
        for operator_index, operator in enumerate(operators):
            right_features = range(n_features) if operator in BINARY_OPERATORS else (None,)
            for right_feature in right_features:
                right_values = (
                    train_features[:, right_feature][None, :] if right_feature is not None else None
                )
                value_blocks.append(_apply_numpy(operator, previous_values, right_values, epsilon))
                parent_blocks.append(parent_indices)
                operator_blocks.append(np.full(parent_count, operator_index, dtype=np.int16))
                right_blocks.append(
                    np.full(parent_count, -1 if right_feature is None else right_feature, dtype=np.int16)
                )
        current_values = np.concatenate(value_blocks, axis=0).astype(np.float32, copy=False)
        parents = np.concatenate(parent_blocks)
        operator_ids = np.concatenate(operator_blocks)
        right_ids = np.concatenate(right_blocks)
        level_values.append(current_values)
        level_metadata.append((parents, operator_ids, right_ids))

        target = np.asarray(train_targets, dtype=np.float64)
        count = current_values.shape[1]
        sums = current_values.sum(axis=1, dtype=np.float64)
        sum_squares = np.einsum("ij,ij->i", current_values, current_values, dtype=np.float64)
        sum_products = np.einsum("ij,j->i", current_values, target, dtype=np.float64)
        centered_squares = np.maximum(sum_squares - sums * sums / count, 1.0e-12)
        centered_products = sum_products - sums * target.sum() / count
        target_sse = float(np.square(target - target.mean()).sum())
        residual_sse = np.maximum(target_sse - centered_products**2 / centered_squares, 0.0)
        scores = np.sqrt(residual_sse / max(target_sse, 1.0e-12))
        keep = min(validation_candidates, len(scores))
        indices = np.argpartition(scores, keep - 1)[:keep]
        indices = indices[np.argsort(scores[indices])]
        finalists.extend((depth, int(index), float(scores[index])) for index in indices)
        previous_values = current_values
    if not finalists:
        raise RuntimeError("discrete program search produced no finite candidates")
    def reconstruct(depth: int, index: int) -> ExecutableProgram:
        reversed_nodes: list[tuple[str, int | None]] = []
        current_index = index
        for current_depth in range(depth, 0, -1):
            parents, operator_ids, right_ids = level_metadata[current_depth - 1]
            operator = operators[int(operator_ids[current_index])]
            right = int(right_ids[current_index])
            reversed_nodes.append((operator, None if right < 0 else right))
            current_index = int(parents[current_index])
        base_feature = current_index
        nodes: list[ProgramNode] = []
        left = base_feature
        for operator, right in reversed(reversed_nodes):
            nodes.append(ProgramNode(operator, left, right))
            left = n_features + len(nodes) - 1
        values = level_values[depth][index]
        scale, bias, _ = _fit_affine(values, train_targets)
        return ExecutableProgram(n_features, nodes, left, scale, bias, epsilon)

    best_program: ExecutableProgram | None = None
    best_key = (float("inf"), float("inf"))
    for depth, index, _ in finalists:
        candidate = reconstruct(depth, index)
        prediction = np.asarray(candidate(validation_features), dtype=np.float64)
        score = float(
            np.sqrt(np.mean((prediction - validation_targets) ** 2))
            / max(np.std(validation_targets), 1.0e-12)
        )
        key = (score, len(candidate.nodes))
        if key < best_key:
            best_key = key
            best_program = candidate
    assert best_program is not None
    return best_program.compile()


def beam_search_chain_program(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    *,
    max_depth: int,
    operators: tuple[str, ...],
    epsilon: float = 0.25,
    beam_width: int = 256,
) -> ExecutableProgram:
    """Bounded beam approximation to chain search for depth diagnostics."""

    n_features = train_features.shape[1]
    # nodes, train values, original base feature, affine scale/bias, train NRMSE
    beam: list[tuple[tuple[ProgramNode, ...], np.ndarray, int, float, float, float]] = []
    finalists: list[ExecutableProgram] = []
    for depth in range(1, max_depth + 1):
        if depth == 1:
            parents = [
                (tuple(), np.asarray(train_features[:, feature], dtype=np.float64), feature)
                for feature in range(n_features)
            ]
        else:
            parents = [(nodes, values, base) for nodes, values, base, _, _, _ in beam]
        expanded: list[
            tuple[tuple[ProgramNode, ...], np.ndarray, int, float, float, float]
        ] = []
        for nodes, left_values, base_feature in parents:
            left_index = base_feature if not nodes else n_features + len(nodes) - 1
            for operator in operators:
                right_features = range(n_features) if operator in BINARY_OPERATORS else (None,)
                for right in right_features:
                    right_values = None if right is None else train_features[:, right]
                    values = _apply_numpy(operator, left_values, right_values, epsilon)
                    scale, bias, score = _fit_affine(values, train_targets)
                    expanded.append(
                        (
                            nodes + (ProgramNode(operator, left_index, right),),
                            values,
                            base_feature,
                            scale,
                            bias,
                            score,
                        )
                    )
        expanded.sort(key=lambda item: item[-1])
        beam = expanded[:beam_width]
        for nodes, _, _, scale, bias, _ in beam:
            finalists.append(
                ExecutableProgram(
                    n_features,
                    list(nodes),
                    n_features + len(nodes) - 1,
                    scale,
                    bias,
                    epsilon,
                ).compile()
            )
    if not finalists:
        raise RuntimeError("beam search produced no candidates")

    def validation_key(candidate: ExecutableProgram) -> tuple[float, int]:
        prediction = np.asarray(candidate(validation_features), dtype=np.float64)
        score = float(
            np.sqrt(np.mean((prediction - validation_targets) ** 2))
            / max(np.std(validation_targets), 1.0e-12)
        )
        return score, len(candidate.nodes)

    return min(finalists, key=validation_key)
