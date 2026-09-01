"""PLE/RBF numerical embedding construction and within-block basis rotations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .common import EPS, bd, ensure_finite


def _prepare_numeric(frames: list[pd.DataFrame], column: str) -> tuple[list[np.ndarray], float]:
    train = pd.to_numeric(frames[0][column], errors="coerce")
    median = float(train.median())
    arrays = [pd.to_numeric(frame[column], errors="coerce").fillna(median).to_numpy(dtype=float) for frame in frames]
    return arrays, median


def _ple(arrays: list[np.ndarray], dimension: int) -> tuple[list[np.ndarray], dict[str, Any]]:
    edges = np.quantile(arrays[0], np.linspace(0.0, 1.0, int(dimension) + 1))
    for index in range(1, len(edges)):
        if edges[index] <= edges[index - 1]:
            edges[index] = edges[index - 1] + max(abs(edges[index - 1]) * 1e-9, 1e-9)
    widths = np.maximum(np.diff(edges), EPS)
    outputs = [np.clip((values[:, None] - edges[:-1][None, :]) / widths[None, :], 0.0, 1.0) for values in arrays]
    return outputs, {"edges": edges.tolist()}


def _rbf(arrays: list[np.ndarray], dimension: int) -> tuple[list[np.ndarray], dict[str, Any]]:
    centers = np.quantile(arrays[0], np.linspace(0.0, 1.0, int(dimension)))
    positive = np.diff(np.unique(centers))
    width = float(np.median(positive)) if len(positive) else 1.0
    width = max(width, EPS)
    outputs = [np.exp(-0.5 * ((values[:, None] - centers[None, :]) / width) ** 2) for values in arrays]
    return outputs, {"centers": centers.tolist(), "width": width}


def _one_hot(train: pd.Series, series: list[pd.Series]) -> tuple[list[np.ndarray], list[str]]:
    levels = sorted(train.astype(str).unique().tolist())
    mapping = {level: index for index, level in enumerate(levels)}
    output = []
    for values in series:
        codes = values.astype(str).map(mapping).fillna(-1).to_numpy(dtype=int)
        matrix = np.zeros((len(codes), len(levels)), dtype=float)
        valid = codes >= 0
        matrix[np.arange(len(codes))[valid], codes[valid]] = 1.0
        output.append(matrix)
    return output, levels


def build_embedding_representation(
    dataset: Any,
    embedding: str,
    dimension: int,
    *,
    rotated: bool,
    rotation_member: int,
) -> Any:
    frames = [dataset.X_train_raw.copy(), dataset.X_validation_raw.copy(), dataset.X_test_raw.copy()]
    pieces: list[list[np.ndarray]] = [[], [], []]
    columns: list[str] = []
    feature_blocks: dict[str, list[int]] = {}
    categorical_blocks: dict[str, list[int]] = {}
    transforms: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {"embedding": embedding, "dimension": int(dimension), "interface_location": "between_embedding_and_backbone", "features": {}}
    for column in frames[0].columns:
        start = len(columns)
        if column in dataset.numerical_columns and frames[0][column].nunique(dropna=True) >= 16:
            arrays, median = _prepare_numeric(frames, column)
            if embedding == "PLE":
                blocks, fitted = _ple(arrays, dimension)
            elif embedding == "RBF":
                blocks, fitted = _rbf(arrays, dimension)
            else:
                raise ValueError(f"unknown embedding {embedding}")
            if rotated:
                matrix = bd.orthogonal_matrix(int(dimension), bd.stable_seed(dataset.key, embedding, dimension, column, rotation_member))
                blocks = [block @ matrix for block in blocks]
            else:
                matrix = np.eye(int(dimension))
            transforms[column] = matrix
            for split, block in enumerate(blocks):
                pieces[split].append(block)
            columns.extend(f"{column}::{embedding}::{index}" for index in range(int(dimension)))
            feature_blocks[column] = list(range(start, start + int(dimension)))
            metadata["features"][column] = {"imputation_median": median, **fitted}
        else:
            values = [frame[column].astype("string").fillna("__MISSING__") for frame in frames]
            blocks, levels = _one_hot(values[0], values)
            for split, block in enumerate(blocks):
                pieces[split].append(block)
            columns.extend(f"{column}::onehot::{level}" for level in levels)
            categorical_blocks[column] = list(range(start, start + len(levels)))
            transforms[column] = np.eye(len(levels))
            metadata["features"][column] = {"levels": levels}
    matrices = tuple(np.concatenate(part, axis=1).astype(float) for part in pieces)
    ensure_finite(matrices, f"{dataset.key}/{embedding}/k{dimension}")
    variant = "embedding_rotated" if rotated else "embedding_original"
    return bd.Representation(
        representation_id=f"{embedding.lower()}_k{dimension}_{variant}_m{rotation_member}",
        family=f"{embedding}_numerical_embedding",
        variant=variant,
        scope="all_numerical_embedding_blocks" if rotated else "reference",
        member=int(rotation_member),
        X_train=matrices[0],
        X_validation=matrices[1],
        X_test=matrices[2],
        columns=columns,
        feature_blocks=feature_blocks,
        categorical_blocks=categorical_blocks,
        transforms=transforms,
        metadata=metadata,
        is_reference=not rotated,
    )


def embedding_orbit(dataset: Any, embedding: str, dimension: int, members: int) -> list[Any]:
    return [build_embedding_representation(dataset, embedding, dimension, rotated=False, rotation_member=-1)] + [
        build_embedding_representation(dataset, embedding, dimension, rotated=True, rotation_member=member)
        for member in range(int(members))
    ]
