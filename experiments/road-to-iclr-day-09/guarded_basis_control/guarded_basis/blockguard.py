"""Feature-selective Gram representations and deterministic BlockGuard rules."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Iterable

import numpy as np

from .common import bd, build_interface


EPS = 1e-8


def selection_key(features: Iterable[str]) -> str:
    ordered = sorted(str(feature) for feature in features)
    digest = hashlib.sha256(json.dumps(ordered, separators=(",", ":")).encode()).hexdigest()[:16]
    return f"n{len(ordered)}__{digest}"


def gram_interface(rep: Any, dataset_key: str) -> Any:
    return build_interface(
        rep,
        "gram_anchor",
        dataset_key,
        anchors=16,
        selection="gram_pivot",
        normalize=True,
    )


def _records(rep: Any) -> list[tuple[str | None, str, list[int]]]:
    named: list[tuple[str, str, list[int]]] = []
    for kind, mapping in (("continuous", rep.feature_blocks), ("categorical", rep.categorical_blocks)):
        named.extend((str(name), kind, list(indices)) for name, indices in mapping.items())
    named.sort(key=lambda row: min(row[2]))
    records: list[tuple[str | None, str, list[int]]] = []
    cursor = 0
    for name, kind, indices in named:
        start, end = min(indices), max(indices) + 1
        if indices != list(range(start, end)):
            raise RuntimeError(f"BlockGuard requires contiguous block {name}")
        if cursor < start:
            records.append((None, "passthrough", list(range(cursor, start))))
        records.append((name, kind, indices))
        cursor = end
    if cursor < rep.X_train.shape[1]:
        records.append((None, "passthrough", list(range(cursor, rep.X_train.shape[1]))))
    return records


def mixed_representation(
    raw_rep: Any,
    dataset_key: str,
    selected_features: Iterable[str],
    *,
    gram_rep: Any | None = None,
) -> Any:
    """Replace only selected continuous blocks with target-free Gram coordinates."""

    selected = frozenset(str(feature) for feature in selected_features)
    missing = selected.difference(raw_rep.feature_blocks)
    if missing:
        raise KeyError(f"unknown BlockGuard features: {sorted(missing)}")
    gram_rep = gram_interface(raw_rep, dataset_key) if gram_rep is None else gram_rep
    raw_splits = (raw_rep.X_train, raw_rep.X_validation, raw_rep.X_test)
    gram_splits = (gram_rep.X_train, gram_rep.X_validation, gram_rep.X_test)
    output: list[list[np.ndarray]] = [[], [], []]
    columns: list[str] = []
    feature_blocks: dict[str, list[int]] = {}
    categorical_blocks: dict[str, list[int]] = {}

    for name, kind, raw_indices in _records(raw_rep):
        if name is not None and kind == "continuous" and name in selected:
            source_indices = list(gram_rep.feature_blocks[name])
            source_splits = gram_splits
            source_columns = gram_rep.columns
        else:
            source_indices = raw_indices
            source_splits = raw_splits
            source_columns = raw_rep.columns
        start = len(columns)
        for split_index, values in enumerate(source_splits):
            output[split_index].append(values[:, source_indices])
        columns.extend(source_columns[index] for index in source_indices)
        if name is not None:
            mapping = feature_blocks if kind == "continuous" else categorical_blocks
            mapping[name] = list(range(start, start + len(source_indices)))

    matrices = tuple(np.concatenate(parts, axis=1).astype(np.float64) for parts in output)
    key = selection_key(selected)
    return dataclasses.replace(
        raw_rep,
        representation_id=f"blockguard__{key}__{raw_rep.representation_id}",
        family="BlockGuard",
        X_train=matrices[0],
        X_validation=matrices[1],
        X_test=matrices[2],
        columns=columns,
        feature_blocks=feature_blocks,
        categorical_blocks=categorical_blocks,
        metadata={
            **raw_rep.metadata,
            "interface": "BlockGuard",
            "selected_features": sorted(selected),
            "selection_key": key,
            "gram_parameters": {
                "anchors": 16,
                "selection": "gram_pivot",
                "normalize": True,
                "coordinate_standardization": True,
            },
        },
    )


def one_block_raw_representation(reference: Any, dataset_key: str, feature: str, member: int) -> Any:
    """Construct the deterministic raw orbit that rotates exactly one feature block."""

    indices = list(reference.feature_blocks[feature])
    matrix = bd.orthogonal_matrix(
        len(indices), bd.stable_seed(dataset_key, "BlockGuard", "raw_one_block", member, feature)
    )
    splits = [values.copy() for values in (reference.X_train, reference.X_validation, reference.X_test)]
    for values in splits:
        values[:, indices] = values[:, indices] @ matrix
    return dataclasses.replace(
        reference,
        representation_id=f"blockguard_raw_one__{feature}__m{int(member)}",
        variant="orthogonal_one_blockguard",
        scope="one",
        member=int(member),
        X_train=splits[0],
        X_validation=splits[1],
        X_test=splits[2],
        transforms={feature: matrix},
        metadata={
            **reference.metadata,
            "blockguard_raw_one_block": feature,
            "orthogonality_error": float(np.linalg.norm(matrix.T @ matrix - np.eye(len(indices)))),
        },
        is_reference=False,
    )


def target_free_descriptors(reference: Any, orbit: list[Any]) -> list[dict[str, Any]]:
    """Compute stable spectral and coordinate-orbit descriptors for every block."""

    rows: list[dict[str, Any]] = []
    for feature_index, (feature, indices) in enumerate(reference.feature_blocks.items()):
        values = np.asarray(reference.X_train[:, indices], dtype=np.float64)
        singular = np.linalg.svd(values, compute_uv=False)
        tolerance = max(values.shape) * np.finfo(float).eps * (singular[0] if len(singular) else 0.0)
        nonzero = singular[singular > tolerance]
        weights = singular**2
        weights = weights / max(float(weights.sum()), np.finfo(float).tiny)
        positive_weights = weights[weights > 0]
        entropy = float(-np.sum(positive_weights * np.log(positive_weights)))
        orbit_values = [
            float(
                np.linalg.norm(np.asarray(rep.X_train[:, indices]) - values)
                / max(np.linalg.norm(values), np.finfo(float).tiny)
            )
            for rep in orbit[1:]
        ]
        rows.append(
            {
                "feature": str(feature),
                "feature_index": int(feature_index),
                "empirical_rank": int(len(nonzero)),
                "block_dimension": int(len(indices)),
                "spectrum_entropy": entropy,
                "largest_nonzero_singular_value": float(nonzero[0]) if len(nonzero) else 0.0,
                "smallest_nonzero_singular_value": float(nonzero[-1]) if len(nonzero) else 0.0,
                "condition_proxy": float(nonzero[0] / nonzero[-1]) if len(nonzero) else float("inf"),
                "mean_gram_diagonal": float(np.mean(np.sum(values**2, axis=1))),
                "embedding_type": "RBF",
                "raw_input_block_orbit_relative_frobenius": float(np.mean(orbit_values)),
            }
        )
    return rows


def grouped_candidates(one_block_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Stable quartile groups of validation C with feature-index tie breaking."""

    ordered = sorted(
        one_block_rows,
        key=lambda row: (float(row["normalized_excess_risk"]), int(row["feature_index"])),
    )
    labels = ("very_safe", "safe", "uncertain", "dangerous")
    groups = np.array_split(np.asarray(ordered, dtype=object), 4)
    assignment: dict[str, str] = {}
    by_label: dict[str, list[str]] = {}
    for label, group in zip(labels, groups):
        features = [str(row["feature"]) for row in group.tolist()]
        by_label[label] = features
        assignment.update({feature: label for feature in features})
    candidates = [
        {"candidate": "raw_only", "features": []},
        {"candidate": "very_safe_gram", "features": by_label["very_safe"]},
        {
            "candidate": "very_safe_plus_safe_gram",
            "features": by_label["very_safe"] + by_label["safe"],
        },
        {
            "candidate": "all_except_dangerous_gram",
            "features": by_label["very_safe"] + by_label["safe"] + by_label["uncertain"],
        },
        {"candidate": "all_gram", "features": [str(row["feature"]) for row in ordered]},
    ]
    return candidates, assignment


def greedy_candidates(one_block_rows: list[dict[str, Any]], maximum_stages: int = 8) -> list[dict[str, Any]]:
    """Stable benefit/cost ordering, batched into at most eight cumulative stages."""

    ordered = sorted(
        one_block_rows,
        key=lambda row: (
            -float(row["basis_disagreement_benefit"])
            / max(float(row["normalized_excess_risk"]) + EPS, EPS),
            int(row["feature_index"]),
        ),
    )
    groups = np.array_split(np.asarray(ordered, dtype=object), min(int(maximum_stages), len(ordered)))
    candidates: list[dict[str, Any]] = [{"candidate": "stage_0_raw", "stage": 0, "features": []}]
    selected: list[str] = []
    for stage, group in enumerate(groups, start=1):
        selected.extend(str(row["feature"]) for row in group.tolist())
        candidates.append(
            {"candidate": f"stage_{stage}", "stage": int(stage), "features": list(selected)}
        )
    return candidates


def select_grouped(candidate_rows: list[dict[str, Any]], tau: float) -> dict[str, Any]:
    eligible = [row for row in candidate_rows if float(row["validation_C"]) <= float(tau)]
    if not eligible:
        raise RuntimeError("raw BlockGuard candidate must always satisfy every nonnegative tau")
    return max(eligible, key=lambda row: (int(row["selected_dimensions"]), int(row["selected_blocks"])))


def select_greedy(candidate_rows: list[dict[str, Any]], tau: float) -> dict[str, Any]:
    ordered = sorted(candidate_rows, key=lambda row: int(row["stage"]))
    selected = ordered[0]
    for row in ordered[1:]:
        if float(row["validation_C"]) > float(tau):
            break
        selected = row
    return selected


def invariant_fractions(reference: Any, selected_features: Iterable[str]) -> tuple[float, float]:
    selected = set(selected_features)
    total_blocks = len(reference.feature_blocks)
    total_dimensions = sum(len(indices) for indices in reference.feature_blocks.values())
    selected_dimensions = sum(len(reference.feature_blocks[name]) for name in selected)
    return len(selected) / max(total_blocks, 1), selected_dimensions / max(total_dimensions, 1)


def coordinate_audit(mixed_orbit: list[Any], selected_features: Iterable[str]) -> dict[str, Any]:
    selected = set(selected_features)
    reference = mixed_orbit[0]
    errors: dict[str, float] = {}
    for feature in sorted(selected):
        indices = reference.feature_blocks[feature]
        denominator = max(np.linalg.norm(reference.X_train[:, indices]), np.finfo(float).tiny)
        errors[feature] = max(
            float(np.linalg.norm(rep.X_train[:, indices] - reference.X_train[:, indices]) / denominator)
            for rep in mixed_orbit[1:]
        )
    maximum = max(errors.values(), default=0.0)
    return {
        "selected_block_relative_errors": errors,
        "maximum_selected_block_relative_error": maximum,
        "passes_1e_minus_6": bool(maximum < 1e-6),
    }
