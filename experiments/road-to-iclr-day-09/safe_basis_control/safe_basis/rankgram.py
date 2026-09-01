"""Rank-adaptive deterministic Gram interfaces and information diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from .common import EPS, bd, ensure_finite

from tournament.representations import gram_pivot_indices


def empirical_rank(values: np.ndarray, relative_threshold: float) -> tuple[int, np.ndarray]:
    singular = np.linalg.svd(np.asarray(values, dtype=float), compute_uv=False)
    if not len(singular) or singular[0] <= EPS:
        return 1, singular
    rank = int(np.sum(singular / singular[0] > float(relative_threshold)))
    return max(rank, 1), singular


def anchor_count(rank: int, rule: str) -> int:
    if rule == "rank":
        return int(rank)
    if rule == "rank_plus_one":
        return int(rank + 1)
    if rule == "double_rank_capped_16":
        return int(min(2 * rank, 16))
    if rule == "fixed_16":
        return 16
    raise ValueError(f"unknown anchor rule {rule}")


def _normalize(
    splits: tuple[np.ndarray, np.ndarray, np.ndarray],
    anchors: np.ndarray,
    normalization: str,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, float]]:
    anchor_norm = np.maximum(np.linalg.norm(anchors, axis=1), EPS)
    raw = tuple(np.asarray(values, dtype=float) @ anchors.T for values in splits)
    if normalization == "N0_raw_inner_product":
        output = raw
        scale = 1.0
    elif normalization == "N1_anchor_norm":
        output = tuple(values / anchor_norm[None, :] for values in raw)
        scale = 1.0
    elif normalization == "N2_cosine":
        output = tuple(
            values / (np.maximum(np.linalg.norm(source, axis=1), EPS)[:, None] * anchor_norm[None, :])
            for values, source in zip(raw, splits)
        )
        scale = 1.0
    elif normalization == "N3_block_rms":
        source_rms = float(np.sqrt(np.mean(np.asarray(splits[0], dtype=float) ** 2)))
        coordinate_rms = float(np.sqrt(np.mean(raw[0] ** 2)))
        scale = source_rms / max(coordinate_rms, EPS)
        output = tuple(values * scale for values in raw)
    else:
        raise ValueError(f"unknown normalization {normalization}")
    return output, {"global_rms_scale": float(scale)}


def _standardize(
    splits: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, Any]]:
    mean = np.asarray(splits[0]).mean(axis=0)
    scale = np.asarray(splits[0]).std(axis=0)
    constant = scale < 1e-12
    scale[constant] = 1.0
    return tuple((np.asarray(values) - mean) / scale for values in splits), {
        "standardized": True,
        "constant_coordinates": int(constant.sum()),
    }


def _reconstruction_error(coordinates: np.ndarray, source: np.ndarray) -> float:
    design = np.column_stack([np.asarray(coordinates, dtype=float), np.ones(len(coordinates))])
    inverse, *_ = np.linalg.lstsq(design, np.asarray(source, dtype=float), rcond=None)
    reconstructed = design @ inverse
    return float(np.linalg.norm(source - reconstructed) / max(np.linalg.norm(source), EPS))


def rank_adaptive_block(
    splits: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    relative_threshold: float,
    anchor_rule: str,
    normalization: str,
    standardize: bool,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, Any]]:
    train = np.asarray(splits[0], dtype=float)
    rank, singular = empirical_rank(train, relative_threshold)
    count = min(anchor_count(rank, anchor_rule), len(train))
    indices, pivot_rank = gram_pivot_indices(train, count)
    anchors = train[indices]
    outputs, normalization_metadata = _normalize(splits, anchors, normalization)
    diagnostic_coordinates = outputs[0]
    reconstruction = _reconstruction_error(diagnostic_coordinates, train)
    gram = anchors @ anchors.T
    eigenvalues = np.linalg.eigvalsh(0.5 * (gram + gram.T))
    positive = eigenvalues[eigenvalues > max(float(eigenvalues.max(initial=0.0)) * 1e-12, 1e-14)]
    effective_condition = float(positive.max() / positive.min()) if len(positive) else float("inf")
    exact_condition = float(np.linalg.cond(gram))
    standardization_metadata: dict[str, Any] = {"standardized": False}
    if standardize:
        outputs, standardization_metadata = _standardize(outputs)
    metadata = {
        "raw_block_dimension": int(train.shape[1]),
        "empirical_rank": int(rank),
        "singular_values": singular.tolist(),
        "relative_rank_threshold": float(relative_threshold),
        "anchor_rule": anchor_rule,
        "anchor_count": int(count),
        "anchor_indices": indices.tolist(),
        "pivot_rank": int(pivot_rank),
        "anchor_gram_condition_number": exact_condition,
        "anchor_gram_effective_condition_number": effective_condition,
        "coordinate_dimension": int(outputs[0].shape[1]),
        "reconstruction_error": reconstruction,
        "normalization": normalization,
        **normalization_metadata,
        **standardization_metadata,
    }
    return outputs, metadata


def _blocks(rep: Any) -> list[tuple[str, list[int], str]]:
    records = []
    for kind, mapping in (("continuous", rep.feature_blocks), ("categorical", rep.categorical_blocks)):
        records.extend((str(name), list(indices), kind) for name, indices in mapping.items())
    records.sort(key=lambda item: min(item[1]))
    return records


def build_rank_adaptive_interface(
    rep: Any,
    dataset_key: str,
    *,
    relative_threshold: float,
    anchor_rule: str,
    normalization: str,
    standardize: bool = False,
) -> Any:
    splits = (rep.X_train, rep.X_validation, rep.X_test)
    pieces: list[list[np.ndarray]] = [[], [], []]
    columns: list[str] = []
    feature_blocks: dict[str, list[int]] = {}
    categorical_blocks: dict[str, list[int]] = {}
    audits: dict[str, Any] = {}
    cursor = 0
    for name, indices, kind in _blocks(rep):
        start, end = min(indices), max(indices) + 1
        if indices != list(range(start, end)):
            raise RuntimeError(f"non-contiguous feature block {name}")
        if cursor < start:
            for split_index, values in enumerate(splits):
                pieces[split_index].append(values[:, cursor:start])
            columns.extend(rep.columns[cursor:start])
        outputs, metadata = rank_adaptive_block(
            tuple(values[:, start:end] for values in splits),
            relative_threshold=relative_threshold,
            anchor_rule=anchor_rule,
            normalization=normalization,
            standardize=standardize,
        )
        new_start = len(columns)
        width = outputs[0].shape[1]
        mapping = feature_blocks if kind == "continuous" else categorical_blocks
        mapping[name] = list(range(new_start, new_start + width))
        columns.extend(f"{name}::rankgram::{index}" for index in range(width))
        for split_index, values in enumerate(outputs):
            pieces[split_index].append(values)
        audits[name] = {"kind": kind, **metadata}
        cursor = end
    if cursor < rep.X_train.shape[1]:
        for split_index, values in enumerate(splits):
            pieces[split_index].append(values[:, cursor:])
        columns.extend(rep.columns[cursor:])
    matrices = tuple(np.concatenate(part, axis=1).astype(float) for part in pieces)
    ensure_finite(matrices, f"{dataset_key}/RankAdaptiveGram")
    return bd.Representation(
        representation_id=f"rank_adaptive_gram__{rep.representation_id}",
        family="rank_adaptive_gram",
        variant=rep.variant,
        scope=rep.scope,
        member=rep.member,
        X_train=matrices[0],
        X_validation=matrices[1],
        X_test=matrices[2],
        columns=columns,
        feature_blocks=feature_blocks,
        categorical_blocks=categorical_blocks,
        transforms=rep.transforms,
        metadata={
            **rep.metadata,
            "interface": "RankAdaptiveGram",
            "relative_threshold": relative_threshold,
            "anchor_rule": anchor_rule,
            "normalization": normalization,
            "standardize": standardize,
            "block_audits": audits,
        },
        is_reference=rep.is_reference,
    )


def orbit_coordinate_audit(reference: Any, transformed: list[Any]) -> list[dict[str, Any]]:
    records = []
    for candidate in transformed:
        shape = candidate.X_train.shape == reference.X_train.shape
        errors = {}
        for split in ("train", "validation", "test"):
            left = getattr(reference, f"X_{split}")
            right = getattr(candidate, f"X_{split}")
            errors[f"{split}_relative_error"] = (
                float(np.linalg.norm(left - right) / max(np.linalg.norm(left), EPS)) if shape else float("inf")
            )
        records.append({"representation_id": candidate.representation_id, "shape_match": shape, **errors})
    return records
