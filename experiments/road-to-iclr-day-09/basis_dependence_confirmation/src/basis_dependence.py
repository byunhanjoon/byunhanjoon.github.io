"""Data, feature-block, transformation, remedy, and model primitives.

All fitted preprocessing accepts train/validation/test explicitly and fits only on train.
Within-feature matrices use the row-vector convention ``Z_transformed = Z @ A``.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.fft import dct
from scipy.linalg import helmert
from scipy.stats import spearmanr
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


EPS = 1e-12


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "little")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass
class DatasetData:
    key: str
    openml_id: int
    openml_version: int
    panel: str
    problem_type: str
    X_train_raw: pd.DataFrame
    X_validation_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    nominal_columns: list[str]
    numerical_columns: list[str]
    cyclic_periods: dict[str, int]


@dataclass
class BlockData:
    dataset: DatasetData
    X_train: np.ndarray
    X_validation: np.ndarray
    X_test: np.ndarray
    columns: list[str]
    feature_blocks: dict[str, list[int]]
    categorical_blocks: dict[str, list[int]]
    passthrough_blocks: dict[str, list[int]]
    selected_feature: str
    basis_metadata: dict[str, Any]


@dataclass
class Representation:
    representation_id: str
    family: str
    variant: str
    scope: str
    member: int
    X_train: np.ndarray
    X_validation: np.ndarray
    X_test: np.ndarray
    columns: list[str]
    feature_blocks: dict[str, list[int]]
    categorical_blocks: dict[str, list[int]]
    transforms: dict[str, np.ndarray]
    metadata: dict[str, Any]
    is_reference: bool = False


def _subsample(indices: np.ndarray, y: np.ndarray, limit: int, seed: int, classification: bool) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) <= limit:
        return np.sort(indices)
    selected, _ = train_test_split(
        indices, train_size=limit, random_state=seed,
        stratify=y[indices] if classification else None,
    )
    return np.sort(np.asarray(selected, dtype=np.int64))


def load_dataset(spec: dict[str, Any], config: dict[str, Any]) -> DatasetData:
    bunch = fetch_openml(data_id=int(spec["openml_id"]), as_frame=True, parser="auto")
    version = int(bunch.details["version"])
    if version != int(spec["openml_version"]):
        raise RuntimeError(f"OpenML version drift for {spec['key']}: {version}")
    X = bunch.data.copy()
    X.columns = [str(column) for column in X.columns]
    raw_y = np.asarray(bunch.target)
    classification = spec["problem_type"] == "classification"
    all_indices = np.arange(len(X), dtype=np.int64)
    outer, test = train_test_split(
        all_indices, test_size=0.2, random_state=int(config["split_seed"]),
        stratify=raw_y if classification else None,
    )
    train, validation = train_test_split(
        outer, test_size=0.2, random_state=int(config["split_seed"]) + 1,
        stratify=raw_y[outer] if classification else None,
    )
    train = _subsample(train, raw_y, int(config["max_train_rows"]), int(config["split_seed"]) + 2, classification)
    validation = _subsample(
        validation, raw_y, int(config["max_validation_rows"]), int(config["split_seed"]) + 3, classification
    )
    test = _subsample(test, raw_y, int(config["max_test_rows"]), int(config["split_seed"]) + 4, classification)
    nominal = [
        column for column in X.columns
        if isinstance(X[column].dtype, pd.CategoricalDtype)
        or X[column].dtype == object or X[column].dtype.name == "string"
    ]
    numerical = [column for column in X.columns if column not in nominal]
    prepared = X.copy()
    for column in nominal:
        prepared[column] = prepared[column].astype("string").fillna("__MISSING__")
    for column in numerical:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce").astype(float)
    frames = [prepared.iloc[idx].reset_index(drop=True) for idx in (train, validation, test)]
    if classification:
        encoder = LabelEncoder().fit(raw_y[train])
        unknown = (set(raw_y[validation]) | set(raw_y[test])) - set(encoder.classes_)
        if unknown:
            raise RuntimeError(f"unseen target classes for {spec['key']}: {unknown}")
        targets = [encoder.transform(raw_y[idx]).astype(int) for idx in (train, validation, test)]
    else:
        targets = [pd.to_numeric(raw_y[idx], errors="raise").astype(float) for idx in (train, validation, test)]
    return DatasetData(
        key=str(spec["key"]), openml_id=int(spec["openml_id"]), openml_version=version,
        panel=str(spec["panel"]), problem_type=str(spec["problem_type"]),
        X_train_raw=frames[0], X_validation_raw=frames[1], X_test_raw=frames[2],
        y_train=targets[0], y_validation=targets[1], y_test=targets[2],
        train_indices=train, validation_indices=validation, test_indices=test,
        nominal_columns=nominal, numerical_columns=numerical,
        cyclic_periods={str(key): int(value) for key, value in spec.get("cyclic_periods", {}).items()},
    )


def _rbf_block(train: np.ndarray, arrays: Iterable[np.ndarray], dimension: int) -> tuple[list[np.ndarray], dict[str, Any]]:
    quantiles = np.linspace(0.0, 1.0, dimension)
    centers = np.quantile(train, quantiles)
    gaps = np.diff(np.unique(centers))
    if len(gaps) == 0:
        raise ValueError("RBF centers are degenerate")
    width = float(np.median(gaps[gaps > 0]))
    width = max(width, 1e-12)
    blocks = [np.exp(-0.5 * ((values[:, None] - centers[None, :]) / width) ** 2) for values in arrays]
    return blocks, {"centers": centers.tolist(), "width": width, "quantiles": quantiles.tolist()}


def _onehot_block(train: pd.Series, series: Iterable[pd.Series]) -> tuple[list[np.ndarray], list[str]]:
    levels = sorted(train.astype(str).unique().tolist())
    mapping = {level: index for index, level in enumerate(levels)}
    outputs = []
    for values in series:
        codes = values.astype(str).map(mapping).fillna(-1).to_numpy(dtype=int)
        matrix = np.zeros((len(codes), len(levels)), dtype=float)
        valid = codes >= 0
        matrix[np.arange(len(codes))[valid], codes[valid]] = 1.0
        outputs.append(matrix)
    return outputs, levels


def build_rbf_feature_matrix(data: DatasetData, config: dict[str, Any]) -> BlockData:
    frames = [data.X_train_raw.copy(), data.X_validation_raw.copy(), data.X_test_raw.copy()]
    minimum_unique = int(config["minimum_continuous_unique_values"])
    dimension = int(config["basis_dimension"])
    valid_continuous = []
    imputation = {}
    for column in data.numerical_columns:
        train = pd.to_numeric(frames[0][column], errors="coerce")
        fill = float(train.median())
        for frame in frames:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(fill)
        imputation[column] = fill
        if frames[0][column].nunique(dropna=True) >= minimum_unique:
            valid_continuous.append(column)
    if not valid_continuous:
        raise RuntimeError(f"no valid continuous feature for {data.key}")
    variances = {column: float(frames[0][column].var(ddof=0)) for column in valid_continuous}
    selected = sorted(valid_continuous, key=lambda column: (-variances[column], column))[0]

    pieces: list[list[np.ndarray]] = [[], [], []]
    columns: list[str] = []
    feature_blocks: dict[str, list[int]] = {}
    categorical_blocks: dict[str, list[int]] = {}
    passthrough_blocks: dict[str, list[int]] = {}
    metadata: dict[str, Any] = {"rbf": {}, "categorical_levels": {}, "imputation": imputation}
    for column in frames[0].columns:
        start = len(columns)
        if column in valid_continuous:
            arrays = [frame[column].to_numpy(dtype=float) for frame in frames]
            blocks, fitted = _rbf_block(arrays[0], arrays, dimension)
            for split_index, block in enumerate(blocks):
                pieces[split_index].append(block)
            names = [f"{column}::rbf::{index}" for index in range(dimension)]
            columns.extend(names)
            feature_blocks[column] = list(range(start, start + dimension))
            metadata["rbf"][column] = fitted
        else:
            values = [frame[column] for frame in frames]
            if column in data.nominal_columns or frames[0][column].nunique(dropna=True) < minimum_unique:
                blocks, levels = _onehot_block(values[0], values)
                for split_index, block in enumerate(blocks):
                    pieces[split_index].append(block)
                names = [f"{column}::onehot::{level}" for level in levels]
                columns.extend(names)
                target = categorical_blocks if column in data.nominal_columns else passthrough_blocks
                target[column] = list(range(start, start + len(levels)))
                metadata["categorical_levels"][column] = levels
            else:
                raise AssertionError("all numerical columns must be routed")
    matrices = [np.concatenate(split_pieces, axis=1).astype(np.float64) for split_pieces in pieces]
    if not all(np.isfinite(matrix).all() for matrix in matrices):
        raise RuntimeError(f"non-finite feature matrix for {data.key}")
    return BlockData(
        dataset=data, X_train=matrices[0], X_validation=matrices[1], X_test=matrices[2],
        columns=columns, feature_blocks=feature_blocks, categorical_blocks=categorical_blocks,
        passthrough_blocks=passthrough_blocks, selected_feature=selected, basis_metadata=metadata,
    )


def orthogonal_matrix(dimension: int, seed: int, positive_determinant: bool = True) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    signs = np.where(np.diag(r) < 0, -1.0, 1.0)
    q = q @ np.diag(signs)
    if positive_determinant and np.linalg.det(q) < 0:
        q[:, -1] *= -1
    return q


def conditioned_matrix(dimension: int, seed: int) -> np.ndarray:
    left = orthogonal_matrix(dimension, stable_seed(seed, "left"))
    right = orthogonal_matrix(dimension, stable_seed(seed, "right"))
    singular = np.geomspace(1 / math.sqrt(3), math.sqrt(3), dimension)
    rng = np.random.default_rng(stable_seed(seed, "singular"))
    singular = singular[rng.permutation(dimension)]
    return left @ np.diag(singular) @ right.T


def _transform_matrices(blocks: BlockData, transforms: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    output = [blocks.X_train.copy(), blocks.X_validation.copy(), blocks.X_test.copy()]
    for feature, matrix in transforms.items():
        indices = blocks.feature_blocks[feature]
        for values in output:
            values[:, indices] = values[:, indices] @ matrix
    return output[0], output[1], output[2]


def build_primary_representations(blocks: BlockData, orbit_members: int) -> list[Representation]:
    identity = {feature: np.eye(len(indices)) for feature, indices in blocks.feature_blocks.items()}
    reference = Representation(
        "rbf_reference", "A", "reference", "reference", -1,
        blocks.X_train.copy(), blocks.X_validation.copy(), blocks.X_test.copy(), list(blocks.columns),
        blocks.feature_blocks, blocks.categorical_blocks, identity,
        {"basis": blocks.basis_metadata, "selected_feature": blocks.selected_feature}, True,
    )
    reps = [reference]
    specifications = (
        ("orthogonal_one", "one", lambda feature, member: orthogonal_matrix(
            len(blocks.feature_blocks[feature]), stable_seed(blocks.dataset.key, "orthogonal_one", member, feature)
        )),
        ("orthogonal_all", "all", lambda feature, member: orthogonal_matrix(
            len(blocks.feature_blocks[feature]), stable_seed(blocks.dataset.key, "orthogonal_all", member, feature)
        )),
        ("condition_le_3_all", "all", lambda feature, member: conditioned_matrix(
            len(blocks.feature_blocks[feature]), stable_seed(blocks.dataset.key, "condition_le_3_all", member, feature)
        )),
    )
    for variant, scope, builder in specifications:
        for member in range(orbit_members):
            features = [blocks.selected_feature] if scope == "one" else sorted(blocks.feature_blocks)
            transforms = {feature: builder(feature, member) for feature in features}
            matrices = _transform_matrices(blocks, transforms)
            audit = {}
            for feature, matrix in transforms.items():
                indices = blocks.feature_blocks[feature]
                reconstructed = matrices[0][:, indices] @ np.linalg.inv(matrix)
                reference_values = blocks.X_train[:, indices]
                relative = float(np.linalg.norm(reference_values - reconstructed) / max(np.linalg.norm(reference_values), EPS))
                audit[feature] = {
                    "condition_number": float(np.linalg.cond(matrix)),
                    "orthogonality_error": float(np.linalg.norm(matrix.T @ matrix - np.eye(len(indices)))),
                    "reconstruction_error": relative,
                }
            reps.append(Representation(
                f"{variant}__m{member}", "A", variant, scope, member,
                *matrices, list(blocks.columns), blocks.feature_blocks, blocks.categorical_blocks,
                transforms, {"equivalence": audit}, False,
            ))
    return reps


def standardize_representation(rep: Representation) -> Representation:
    mean = rep.X_train.mean(axis=0)
    scale = rep.X_train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    matrices = tuple((values - mean) / scale for values in (rep.X_train, rep.X_validation, rep.X_test))
    return Representation(
        f"standardize__{rep.representation_id}", "D", rep.variant, rep.scope, rep.member,
        *matrices, rep.columns, rep.feature_blocks, rep.categorical_blocks, rep.transforms,
        {**rep.metadata, "repair": "standardization"}, rep.is_reference,
    )


def whiten_representation(rep: Representation) -> Representation:
    matrices = [values.copy() for values in (rep.X_train, rep.X_validation, rep.X_test)]
    audits = {}
    for feature, indices in rep.feature_blocks.items():
        train = rep.X_train[:, indices]
        mean = train.mean(axis=0)
        covariance = np.cov(train - mean, rowvar=False, ddof=0)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        floor = max(float(eigenvalues.max()) * 1e-10, 1e-12)
        inverse_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(np.maximum(eigenvalues, floor))) @ eigenvectors.T
        for split_index, values in enumerate((rep.X_train, rep.X_validation, rep.X_test)):
            matrices[split_index][:, indices] = (values[:, indices] - mean) @ inverse_sqrt
        audits[feature] = {"min_eigenvalue": float(eigenvalues.min()), "floor": floor}
    return Representation(
        f"whiten__{rep.representation_id}", "D", rep.variant, rep.scope, rep.member,
        *matrices, rep.columns, rep.feature_blocks, rep.categorical_blocks, rep.transforms,
        {**rep.metadata, "repair": "whitening", "whitening": audits}, rep.is_reference,
    )


def pca_canonical_representation(rep: Representation) -> Representation:
    matrices = [values.copy() for values in (rep.X_train, rep.X_validation, rep.X_test)]
    audits = {}
    for feature, indices in rep.feature_blocks.items():
        train = rep.X_train[:, indices]
        mean = train.mean(axis=0)
        _, singular, vt = np.linalg.svd(train - mean, full_matrices=False)
        vectors = vt.T
        train_scores = (train - mean) @ vectors
        for component in range(vectors.shape[1]):
            row = int(np.argmax(np.abs(train_scores[:, component])))
            if train_scores[row, component] < 0:
                vectors[:, component] *= -1
                train_scores[:, component] *= -1
        for split_index, values in enumerate((rep.X_train, rep.X_validation, rep.X_test)):
            matrices[split_index][:, indices] = (values[:, indices] - mean) @ vectors
        ratios = singular[1:] / np.maximum(singular[:-1], EPS)
        audits[feature] = {
            "singular_values": singular.tolist(),
            "degenerate": bool(np.any(ratios > 0.99)),
            "closest_adjacent_ratio": float(ratios.max()) if len(ratios) else 0.0,
        }
    return Representation(
        f"pca__{rep.representation_id}", "D", rep.variant, rep.scope, rep.member,
        *matrices, rep.columns, rep.feature_blocks, rep.categorical_blocks, rep.transforms,
        {**rep.metadata, "repair": "pca_canonical", "pca": audits}, rep.is_reference,
    )


def anchor_canonical_representation(rep: Representation, dataset_key: str, anchors: int = 16) -> Representation:
    output_splits: list[list[np.ndarray]] = [[], [], []]
    new_columns: list[str] = []
    new_feature_blocks: dict[str, list[int]] = {}
    audits = {}
    block_by_start = sorted(rep.feature_blocks.items(), key=lambda item: min(item[1]))
    used = set(index for indices in rep.feature_blocks.values() for index in indices)
    segments: list[tuple[int, int, str | None]] = []
    cursor = 0
    for feature, indices in block_by_start:
        start, end = min(indices), max(indices) + 1
        if cursor < start:
            segments.append((cursor, start, None))
        segments.append((start, end, feature))
        cursor = end
    if cursor < rep.X_train.shape[1]:
        segments.append((cursor, rep.X_train.shape[1], None))
    if len(used) != sum(len(indices) for indices in rep.feature_blocks.values()):
        raise RuntimeError("overlapping feature blocks")
    for start, end, feature in segments:
        new_start = len(new_columns)
        if feature is None:
            for split_index, values in enumerate((rep.X_train, rep.X_validation, rep.X_test)):
                output_splits[split_index].append(values[:, start:end])
            new_columns.extend(rep.columns[start:end])
            continue
        dimension = end - start
        m = max(2 * dimension, anchors)
        rng = np.random.default_rng(stable_seed(dataset_key, "AnchorCanonical", feature))
        if len(rep.X_train) < m:
            anchor_indices = rng.choice(len(rep.X_train), size=m, replace=True)
        else:
            anchor_indices = rng.choice(len(rep.X_train), size=m, replace=False)
        anchor_matrix = rep.X_train[anchor_indices, start:end].astype(np.float64)
        rank = int(np.linalg.matrix_rank(anchor_matrix))
        pinv = np.linalg.pinv(anchor_matrix)
        for split_index, values in enumerate((rep.X_train, rep.X_validation, rep.X_test)):
            output_splits[split_index].append(values[:, start:end].astype(np.float64) @ pinv)
        new_feature_blocks[feature] = list(range(new_start, new_start + m))
        new_columns.extend([f"{feature}::anchor::{index}" for index in range(m)])
        audits[feature] = {
            "anchor_row_indices": anchor_indices.tolist(), "rank": rank, "dimension": dimension,
            "full_rank": rank == dimension, "anchor_condition_number": float(np.linalg.cond(anchor_matrix)),
        }
    matrices = [np.concatenate(parts, axis=1) for parts in output_splits]
    return Representation(
        f"anchor__{rep.representation_id}", "D", rep.variant, rep.scope, rep.member,
        matrices[0], matrices[1], matrices[2], new_columns, new_feature_blocks, {}, rep.transforms,
        {**rep.metadata, "repair": "anchor_canonical", "anchor": audits}, rep.is_reference,
    )


def oracle_inverse_representation(rep: Representation) -> Representation:
    matrices = [values.copy() for values in (rep.X_train, rep.X_validation, rep.X_test)]
    for feature, transform in rep.transforms.items():
        indices = rep.feature_blocks[feature]
        if not np.allclose(transform, np.eye(len(indices))):
            for values in matrices:
                values[:, indices] = values[:, indices] @ np.linalg.inv(transform)
    return Representation(
        f"oracle__{rep.representation_id}", "D", rep.variant, rep.scope, rep.member,
        *matrices, rep.columns, rep.feature_blocks, rep.categorical_blocks, rep.transforms,
        {**rep.metadata, "repair": "ORACLE INVERSE — NOT A METHOD"}, rep.is_reference,
    )


def helmert_pair(blocks: BlockData) -> tuple[Representation, Representation] | None:
    candidates = [(feature, indices) for feature, indices in blocks.categorical_blocks.items() if len(indices) >= 3]
    if not candidates:
        return None
    feature, indices = sorted(candidates, key=lambda item: (-len(item[1]), item[0]))[0]
    matrix = helmert(len(indices), full=True).T
    transformed = [values.copy() for values in (blocks.X_train, blocks.X_validation, blocks.X_test)]
    for values in transformed:
        values[:, indices] = values[:, indices] @ matrix
    base = Representation(
        f"onehot__{feature}", "C1", "onehot", "one", -1,
        blocks.X_train.copy(), blocks.X_validation.copy(), blocks.X_test.copy(), blocks.columns,
        blocks.feature_blocks, blocks.categorical_blocks, {}, {"feature": feature}, True,
    )
    reconstruction = float(np.linalg.norm(blocks.X_train[:, indices] - transformed[0][:, indices] @ matrix.T) /
                           max(np.linalg.norm(blocks.X_train[:, indices]), EPS))
    pair = Representation(
        f"helmert__{feature}", "C1", "helmert", "one", 0,
        *transformed, blocks.columns, blocks.feature_blocks, blocks.categorical_blocks, {feature: matrix},
        {"feature": feature, "condition_number": float(np.linalg.cond(matrix)), "reconstruction_error": reconstruction},
    )
    return base, pair


def _hat_values(values: np.ndarray, knots: np.ndarray) -> np.ndarray:
    result = np.zeros((len(values), len(knots)), dtype=float)
    clipped = np.clip(values, knots[0], knots[-1])
    right = np.searchsorted(knots, clipped, side="right")
    right = np.clip(right, 1, len(knots) - 1)
    left = right - 1
    denominator = np.maximum(knots[right] - knots[left], EPS)
    weight_right = (clipped - knots[left]) / denominator
    result[np.arange(len(values)), left] = 1.0 - weight_right
    result[np.arange(len(values)), right] += weight_right
    return result


def local_spectral_pair(blocks: BlockData) -> tuple[Representation, Representation]:
    raw_frames = [blocks.dataset.X_train_raw, blocks.dataset.X_validation_raw, blocks.dataset.X_test_raw]
    candidates = []
    for feature in blocks.feature_blocks:
        train = pd.to_numeric(raw_frames[0][feature], errors="coerce")
        fill = float(train.median())
        values = train.fillna(fill).to_numpy(dtype=float)
        candidates.append((feature, float(np.var(values)), fill, values))
    candidates.sort(key=lambda item: (-item[1], item[0]))
    selected = None
    rejected = []
    for feature, _, fill, train_values in candidates:
        knots = np.quantile(train_values, np.linspace(0, 1, 8))
        if len(np.unique(knots)) == 8:
            selected = (feature, fill, knots)
            break
        rejected.append(feature)
    if selected is None:
        raise ValueError(f"no continuous feature has 8 distinct hat knots for {blocks.dataset.key}")
    feature, fill, knots = selected
    arrays = [pd.to_numeric(frame[feature], errors="coerce").fillna(fill).to_numpy(dtype=float) for frame in raw_frames]
    local_blocks = [_hat_values(values, knots) for values in arrays]
    q = dct(np.eye(8), norm="ortho", axis=0)
    spectral_blocks = [values @ q for values in local_blocks]
    indices = blocks.feature_blocks[feature]
    local = [values.copy() for values in (blocks.X_train, blocks.X_validation, blocks.X_test)]
    spectral = [values.copy() for values in (blocks.X_train, blocks.X_validation, blocks.X_test)]
    for split_index in range(3):
        local[split_index][:, indices] = local_blocks[split_index]
        spectral[split_index][:, indices] = spectral_blocks[split_index]
    base = Representation(
        f"local_hat__{feature}", "C3", "local_hat", "one", -1,
        *local, blocks.columns, blocks.feature_blocks, blocks.categorical_blocks, {},
        {"feature": feature, "knots": knots.tolist(),
         "selection_rule": "highest_training_variance_then_lexicographic_with_8_distinct_quantile_knots",
         "rejected_duplicate_knot_features": rejected}, True,
    )
    reconstruction = float(np.linalg.norm(local_blocks[0] - spectral_blocks[0] @ q.T) /
                           max(np.linalg.norm(local_blocks[0]), EPS))
    pair = Representation(
        f"spectral_hat__{feature}", "C3", "spectral_hat", "one", 0,
        *spectral, blocks.columns, blocks.feature_blocks, blocks.categorical_blocks, {feature: q},
        {"feature": feature, "knots": knots.tolist(),
         "selection_rule": "highest_training_variance_then_lexicographic_with_8_distinct_quantile_knots",
         "rejected_duplicate_knot_features": rejected, "condition_number": float(np.linalg.cond(q)),
         "reconstruction_error": reconstruction},
    )
    return base, pair


def fourier_origin_pairs(blocks: BlockData, members: int = 8) -> list[Representation]:
    candidates = [feature for feature in blocks.dataset.cyclic_periods if feature in blocks.dataset.X_train_raw.columns]
    if not candidates:
        return []
    feature = sorted(candidates, key=lambda name: (-blocks.dataset.cyclic_periods[name], name))[0]
    period = blocks.dataset.cyclic_periods[feature]
    frames = [blocks.dataset.X_train_raw, blocks.dataset.X_validation_raw, blocks.dataset.X_test_raw]
    arrays = [pd.to_numeric(frame[feature], errors="coerce").fillna(0).to_numpy(dtype=float) for frame in frames]
    def encode(values: np.ndarray, shift: float) -> np.ndarray:
        theta = 2 * np.pi * (values + shift) / period
        return np.column_stack([
            component
            for frequency in range(1, 5)
            for component in (np.sin(frequency * theta), np.cos(frequency * theta))
        ])
    base_blocks = [encode(values, 0.0) for values in arrays]
    old_indices = blocks.feature_blocks.get(feature) or blocks.passthrough_blocks.get(feature)
    if old_indices is None or len(old_indices) != 8:
        return []
    reps = []
    base_matrices = [values.copy() for values in (blocks.X_train, blocks.X_validation, blocks.X_test)]
    for split_index in range(3):
        base_matrices[split_index][:, old_indices] = base_blocks[split_index]
    reps.append(Representation(
        f"fourier_origin__{feature}", "C2", "fourier_origin", "one", -1,
        *base_matrices, blocks.columns, blocks.feature_blocks, blocks.categorical_blocks, {},
        {"feature": feature, "period": period, "origin": 0}, True,
    ))
    shifts = np.linspace(1, max(period - 1, 1), members, dtype=int)
    for member, shift in enumerate(shifts):
        shifted_blocks = [encode(values, float(shift)) for values in arrays]
        matrices = [values.copy() for values in base_matrices]
        for split_index in range(3):
            matrices[split_index][:, old_indices] = shifted_blocks[split_index]
        rotations = []
        for frequency in range(1, 5):
            angle = 2 * np.pi * frequency * shift / period
            rotations.append(np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]))
        q = np.zeros((8, 8))
        for frequency_index, rotation in enumerate(rotations):
            start = 2 * frequency_index
            q[start:start + 2, start:start + 2] = rotation
        reconstruction = float(np.linalg.norm(base_blocks[0] @ q - shifted_blocks[0]) /
                               max(np.linalg.norm(base_blocks[0]), EPS))
        reps.append(Representation(
            f"fourier_shift__{feature}__m{member}", "C2", "fourier_shift", "one", member,
            *matrices, blocks.columns, blocks.feature_blocks, blocks.categorical_blocks, {feature: q},
            {"feature": feature, "period": period, "shift": int(shift),
             "condition_number": float(np.linalg.cond(q)), "reconstruction_error": reconstruction},
        ))
    return reps


def prediction_metrics(problem_type: str, y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if problem_type == "classification":
        probs = np.clip(np.asarray(prediction, dtype=float), 1e-8, 1.0)
        probs /= probs.sum(axis=1, keepdims=True)
        result = {
            "log_loss": float(log_loss(y, probs, labels=np.arange(probs.shape[1]))),
            "accuracy": float(accuracy_score(y, probs.argmax(axis=1))),
        }
        try:
            result["roc_auc"] = float(
                roc_auc_score(y, probs[:, 1]) if probs.shape[1] == 2
                else roc_auc_score(y, probs, multi_class="ovr")
            )
        except ValueError:
            result["roc_auc"] = float("nan")
        return result
    pred = np.asarray(prediction, dtype=float).reshape(-1)
    return {"rmse": float(mean_squared_error(y, pred) ** 0.5), "mae": float(mean_absolute_error(y, pred))}


def disagreement_metrics(problem_type: str, y: np.ndarray, reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if problem_type == "classification":
        p = np.clip(np.asarray(reference, dtype=float), 1e-8, 1.0)
        q = np.clip(np.asarray(prediction, dtype=float), 1e-8, 1.0)
        p /= p.sum(axis=1, keepdims=True)
        q /= q.sum(axis=1, keepdims=True)
        midpoint = 0.5 * (p + q)
        js = 0.5 * np.sum(p * np.log(p / midpoint), axis=1) + 0.5 * np.sum(q * np.log(q / midpoint), axis=1)
        return {
            "probability_rmse": float(np.sqrt(np.mean((p - q) ** 2))),
            "probability_mad": float(np.mean(np.abs(p - q))),
            "js_divergence": float(np.mean(js)),
            "label_flip_rate": float(np.mean(p.argmax(axis=1) != q.argmax(axis=1))),
        }
    p = np.asarray(reference, dtype=float).reshape(-1)
    q = np.asarray(prediction, dtype=float).reshape(-1)
    scale = max(float(np.std(y)), EPS)
    pearson = float(np.corrcoef(p, q)[0, 1]) if np.std(p) > EPS and np.std(q) > EPS else float("nan")
    return {
        "prediction_rmse_normalized": float(np.sqrt(np.mean((p - q) ** 2)) / scale),
        "prediction_pearson": pearson,
        "prediction_spearman": float(spearmanr(p, q).statistic) if len(p) > 1 else float("nan"),
    }


def environment_metadata() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "tabpfn", "tabicl", "catboost", "pytabkit"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    metadata: dict[str, Any] = {
        "python": platform.python_version(), "platform": platform.platform(), "packages": packages,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch
        metadata["torch_cuda"] = torch.version.cuda
        metadata["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        metadata["gpu"] = None
    return metadata


def _tabpfn_checkpoint(problem_type: str) -> Path:
    filename = f"tabpfn-v2.6-{'regressor' if problem_type == 'regression' else 'classifier'}-v2.6_default.ckpt"
    matches = list(Path.home().glob(f".cache/huggingface/hub/models--Prior-Labs--tabpfn_2_6/snapshots/*/{filename}"))
    if not matches:
        raise FileNotFoundError(f"missing exact TabPFN-2.6 checkpoint {filename}")
    return matches[0].resolve()


def _fit_controlled_mlp(
    problem_type: str,
    X_train: np.ndarray,
    X_validation: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    seed: int,
    device: str,
    model_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], Any]:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    activation = nn.GELU if model_config["activation"] == "GELU" else nn.ReLU
    n_outputs = 1 if problem_type == "regression" else int(np.max(y_train)) + 1
    layers: list[nn.Module] = []
    size = X_train.shape[1]
    for _ in range(int(model_config["hidden_layers"])):
        layers.extend([nn.Linear(size, int(model_config["width"])), activation()])
        size = int(model_config["width"])
    layers.append(nn.Linear(size, n_outputs))
    model = nn.Sequential(*layers).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    X_train_tensor = torch.as_tensor(np.asarray(X_train, dtype=np.float32), device=device)
    X_validation_tensor = torch.as_tensor(np.asarray(X_validation, dtype=np.float32), device=device)
    if problem_type == "regression":
        y_mean = float(np.mean(y_train))
        y_scale = max(float(np.std(y_train)), 1e-8)
        y_train_fit = ((y_train - y_mean) / y_scale).astype(np.float32)
        y_validation_fit = ((y_validation - y_mean) / y_scale).astype(np.float32)
        loss_fn: Any = nn.MSELoss()
        train_target = torch.as_tensor(y_train_fit, device=device)
        validation_target = torch.as_tensor(y_validation_fit, device=device)
    else:
        y_mean, y_scale = 0.0, 1.0
        loss_fn = nn.CrossEntropyLoss()
        train_target = torch.as_tensor(y_train.astype(np.int64), device=device)
        validation_target = torch.as_tensor(y_validation.astype(np.int64), device=device)
    rng = np.random.default_rng(seed)
    best_state = None
    best_loss = float("inf")
    best_epoch = -1
    no_improvement = 0
    batch_size = int(model_config["batch_size"])
    started = time.perf_counter()
    for epoch in range(int(model_config["max_epochs"])):
        model.train()
        order = rng.permutation(len(y_train))
        for start in range(0, len(order), batch_size):
            indices = torch.as_tensor(order[start:start + batch_size], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(X_train_tensor[indices]).squeeze(-1), train_target[indices])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_fn(model(X_validation_tensor).squeeze(-1), validation_target).item())
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= int(model_config["patience"]):
                break
    if best_state is None:
        raise RuntimeError("controlled MLP did not produce a checkpoint")
    model.load_state_dict(best_state)
    fit_seconds = time.perf_counter() - started
    model.eval()
    predict_started = time.perf_counter()
    with torch.no_grad():
        validation_logits = model(X_validation_tensor).squeeze(-1)
        test_logits = model(torch.as_tensor(np.asarray(X_test, dtype=np.float32), device=device)).squeeze(-1)
        if problem_type == "regression":
            validation_prediction = validation_logits.cpu().numpy() * y_scale + y_mean
            test_prediction = test_logits.cpu().numpy() * y_scale + y_mean
        else:
            validation_prediction = torch.softmax(validation_logits, dim=1).cpu().numpy()
            test_prediction = torch.softmax(test_logits, dim=1).cpu().numpy()
    telemetry = {
        "fit_seconds": fit_seconds, "predict_seconds": time.perf_counter() - predict_started,
        "best_epoch": best_epoch, "best_validation_loss": best_loss,
        "architecture": "3x256-GELU-no_batchnorm-no_dropout",
    }
    return np.asarray(validation_prediction), np.asarray(test_prediction), telemetry, model


def fit_predict(
    model_name: str,
    problem_type: str,
    rep: Representation,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    seed: int,
    device: str,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    if model_name == "controlled_mlp":
        validation, test, telemetry, model = _fit_controlled_mlp(
            problem_type, rep.X_train, rep.X_validation, rep.X_test,
            y_train, y_validation, seed, device, config["models"][model_name],
        )
    elif model_name == "tabpfn_2_6":
        from tabpfn import TabPFNClassifier, TabPFNRegressor

        checkpoint = _tabpfn_checkpoint(problem_type)
        cls = TabPFNRegressor if problem_type == "regression" else TabPFNClassifier
        model = cls(
            n_estimators=int(config["models"][model_name]["n_estimators"]),
            model_path=checkpoint, device=device, random_state=seed,
            inference_precision="autocast", fit_mode="fit_preprocessors", show_progress_bar=False,
        )
        model.fit(rep.X_train, y_train)
        fit_seconds = time.perf_counter() - started
        predict_started = time.perf_counter()
        if problem_type == "regression":
            validation, test = model.predict(rep.X_validation), model.predict(rep.X_test)
        else:
            validation, test = model.predict_proba(rep.X_validation), model.predict_proba(rep.X_test)
        telemetry = {
            "fit_seconds": fit_seconds, "predict_seconds": time.perf_counter() - predict_started,
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
        }
    elif model_name == "tabicl_v2":
        from tabicl import TabICLClassifier, TabICLRegressor

        cls = TabICLRegressor if problem_type == "regression" else TabICLClassifier
        model = cls(
            n_estimators=int(config["models"][model_name]["n_estimators"]),
            device=device, use_amp=True, random_state=seed, batch_size=1, allow_auto_download=True,
        )
        model.fit(rep.X_train, y_train)
        fit_seconds = time.perf_counter() - started
        predict_started = time.perf_counter()
        if problem_type == "regression":
            validation, test = model.predict(rep.X_validation), model.predict(rep.X_test)
        else:
            validation, test = model.predict_proba(rep.X_validation), model.predict_proba(rep.X_test)
        checkpoint = Path(model.model_path_).resolve()
        telemetry = {
            "fit_seconds": fit_seconds, "predict_seconds": time.perf_counter() - predict_started,
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size, "checkpoint_version": model.checkpoint_version,
        }
    elif model_name == "tabm_d":
        from pytabkit.models.sklearn.sklearn_interfaces import TabM_D_Classifier, TabM_D_Regressor

        cls = TabM_D_Regressor if problem_type == "regression" else TabM_D_Classifier
        model_config = config["models"][model_name]
        model = cls(
            device=device, random_state=seed, n_cv=int(model_config["n_cv"]),
            n_refit=int(model_config["n_refit"]), n_epochs=int(model_config["n_epochs"]),
            patience=int(model_config["patience"]), n_threads=4, verbosity=0,
            lr=float(model_config["learning_rate"]) if "learning_rate" in model_config else None,
            weight_decay=float(model_config["weight_decay"]) if "weight_decay" in model_config else None,
        )
        model.fit(rep.X_train, y_train, X_val=rep.X_validation, y_val=y_validation)
        fit_seconds = time.perf_counter() - started
        predict_started = time.perf_counter()
        if problem_type == "regression":
            validation, test = model.predict(rep.X_validation), model.predict(rep.X_test)
        else:
            validation, test = model.predict_proba(rep.X_validation), model.predict_proba(rep.X_test)
        telemetry = {"fit_seconds": fit_seconds, "predict_seconds": time.perf_counter() - predict_started}
    elif model_name == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor

        model_config = config["models"][model_name]
        common = dict(
            iterations=int(model_config["iterations"]), depth=int(model_config["depth"]),
            learning_rate=float(model_config["learning_rate"]), random_seed=seed,
            thread_count=4, allow_writing_files=False, verbose=False,
        )
        model = CatBoostRegressor(loss_function="RMSE", **common) if problem_type == "regression" else CatBoostClassifier(
            loss_function="MultiClass" if len(np.unique(y_train)) > 2 else "Logloss", **common
        )
        model.fit(rep.X_train, y_train, eval_set=(rep.X_validation, y_validation), early_stopping_rounds=30)
        fit_seconds = time.perf_counter() - started
        predict_started = time.perf_counter()
        if problem_type == "regression":
            validation = model.predict(rep.X_validation).reshape(-1)
            test = model.predict(rep.X_test).reshape(-1)
        else:
            validation, test = model.predict_proba(rep.X_validation), model.predict_proba(rep.X_test)
        telemetry = {"fit_seconds": fit_seconds, "predict_seconds": time.perf_counter() - predict_started}
    else:
        raise ValueError(f"unknown model {model_name}")
    validation = np.asarray(validation, dtype=float)
    test = np.asarray(test, dtype=float)
    clear_model(model, device)
    return validation, test, telemetry


def clear_model(model: Any, device: str) -> None:
    del model
    gc.collect()
    try:
        import torch
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    except Exception:
        pass
