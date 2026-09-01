"""Representation-orbit construction and model adapters for Kill Experiment 2.

All data-dependent operations are fitted on the training partition.  Representation objects
carry their own reference identifier so disagreement is never computed against a non-equivalent
feature space (for example, an RBF basis is compared with the unrotated RBF basis).
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
from scipy.stats import rankdata, spearmanr
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


EPS = 1e-12
MONTHS = {name: index for index, name in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
)}


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
class DatasetSplit:
    name: str
    openml_id: int
    openml_version: int
    problem_type: str
    X_train_native: pd.DataFrame
    X_validation_native: pd.DataFrame
    X_test_native: pd.DataFrame
    X_train_numeric: pd.DataFrame
    X_validation_numeric: pd.DataFrame
    X_test_numeric: pd.DataFrame
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    nominal_columns: list[str]
    numerical_columns: list[str]
    ordinal_orders: dict[str, list[str]]
    cyclic_periods: dict[str, int]

    def frames(self, pipeline: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if pipeline == "native_categorical":
            return self.X_train_native.copy(), self.X_validation_native.copy(), self.X_test_native.copy()
        if pipeline == "numeric_code":
            return self.X_train_numeric.copy(), self.X_validation_numeric.copy(), self.X_test_numeric.copy()
        raise ValueError(f"unknown pipeline {pipeline}")


@dataclass
class Representation:
    representation_id: str
    reference_id: str
    family: str
    variant: str
    scope: str
    member: int
    pipeline: str
    repair: str
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    categorical_columns: list[str]
    metadata: dict[str, Any]
    is_reference: bool = False


def _subsample(indices: np.ndarray, y: np.ndarray, limit: int, seed: int, classification: bool) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) <= limit:
        return np.sort(indices)
    selected, _ = train_test_split(
        indices,
        train_size=limit,
        random_state=seed,
        stratify=y[indices] if classification else None,
    )
    return np.sort(np.asarray(selected, dtype=np.int64))


def load_dataset(spec: dict[str, Any], config: dict[str, Any]) -> DatasetSplit:
    bunch = fetch_openml(data_id=int(spec["openml_id"]), as_frame=True, parser="auto")
    actual_version = int(bunch.details["version"])
    if actual_version != int(spec["openml_version"]):
        raise RuntimeError(f"OpenML version drift for {spec['name']}: {actual_version}")
    X = bunch.data.copy()
    X.columns = [str(column) for column in X.columns]
    raw_y = np.asarray(bunch.target)
    classification = spec["problem_type"] == "classification"
    all_indices = np.arange(len(X), dtype=np.int64)
    outer, test = train_test_split(
        all_indices,
        test_size=0.2,
        random_state=int(config["split_seed"]),
        stratify=raw_y if classification else None,
    )
    train, validation = train_test_split(
        outer,
        test_size=0.2,
        random_state=int(config["split_seed"]) + 1,
        stratify=raw_y[outer] if classification else None,
    )
    train = _subsample(train, raw_y, int(config["max_train_rows"]), int(config["split_seed"]) + 2, classification)
    validation = _subsample(
        validation, raw_y, int(config["max_validation_rows"]), int(config["split_seed"]) + 3, classification
    )
    test = _subsample(test, raw_y, int(config["max_test_rows"]), int(config["split_seed"]) + 4, classification)

    nominal = [
        column for column in X.columns
        if isinstance(X[column].dtype, pd.CategoricalDtype) or X[column].dtype == object or X[column].dtype.name == "string"
    ]
    numerical = [column for column in X.columns if column not in nominal]
    native = X.copy()
    for column in nominal:
        native[column] = native[column].astype("string").fillna("__MISSING__")
    for column in numerical:
        native[column] = pd.to_numeric(native[column], errors="coerce").astype(float)

    train_native = native.iloc[train].reset_index(drop=True)
    validation_native = native.iloc[validation].reset_index(drop=True)
    test_native = native.iloc[test].reset_index(drop=True)
    train_numeric = train_native.copy()
    validation_numeric = validation_native.copy()
    test_numeric = test_native.copy()
    for column in nominal:
        levels = sorted(train_native[column].astype(str).unique().tolist())
        mapping = {value: index for index, value in enumerate(levels)}
        for frame in (train_numeric, validation_numeric, test_numeric):
            frame[column] = frame[column].astype(str).map(mapping).fillna(-1).astype(float)

    if classification:
        encoder = LabelEncoder().fit(raw_y[train])
        unknown = (set(raw_y[validation]) | set(raw_y[test])) - set(encoder.classes_)
        if unknown:
            raise RuntimeError(f"test classes absent from train for {spec['name']}: {unknown}")
        y_train = encoder.transform(raw_y[train]).astype(int)
        y_validation = encoder.transform(raw_y[validation]).astype(int)
        y_test = encoder.transform(raw_y[test]).astype(int)
    else:
        y_train = pd.to_numeric(raw_y[train], errors="raise").astype(float)
        y_validation = pd.to_numeric(raw_y[validation], errors="raise").astype(float)
        y_test = pd.to_numeric(raw_y[test], errors="raise").astype(float)

    return DatasetSplit(
        name=str(spec["name"]),
        openml_id=int(spec["openml_id"]),
        openml_version=actual_version,
        problem_type=str(spec["problem_type"]),
        X_train_native=train_native,
        X_validation_native=validation_native,
        X_test_native=test_native,
        X_train_numeric=train_numeric,
        X_validation_numeric=validation_numeric,
        X_test_numeric=test_numeric,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
        train_indices=train,
        validation_indices=validation,
        test_indices=test,
        nominal_columns=nominal,
        numerical_columns=numerical,
        ordinal_orders={str(key): list(value) for key, value in spec.get("ordinal_orders", {}).items()},
        cyclic_periods={str(key): int(value) for key, value in spec.get("cyclic_periods", {}).items()},
    )


def _native_categories(frame: pd.DataFrame, nominal: Iterable[str]) -> list[str]:
    columns = set(frame.columns)
    return [column for column in nominal if column in columns and not pd.api.types.is_numeric_dtype(frame[column])]


def _reference(
    rep_id: str,
    family: str,
    variant: str,
    pipeline: str,
    frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    nominal: list[str],
    metadata: dict[str, Any] | None = None,
) -> Representation:
    return Representation(
        rep_id, rep_id, family, variant, "reference", -1, pipeline, "none",
        *frames, _native_categories(frames[0], nominal), metadata or {}, True,
    )


def _zscore_frames(frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], columns: list[str]):
    train, validation, test = (frame.copy() for frame in frames)
    means = train[columns].mean(axis=0)
    scales = train[columns].std(axis=0, ddof=0).replace(0, 1.0).fillna(1.0)
    for frame in (train, validation, test):
        frame.loc[:, columns] = (frame[columns] - means) / scales
    return (train, validation, test), {"means": means.to_dict(), "scales": scales.to_dict()}


def _quantile_column(train: pd.Series, values: pd.Series) -> np.ndarray:
    finite = np.sort(pd.to_numeric(train, errors="coerce").dropna().unique())
    source = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if len(finite) <= 1:
        result = np.zeros(len(source), dtype=float)
    else:
        result = np.interp(source, finite, np.linspace(0.0, 1.0, len(finite)), left=0.0, right=1.0)
    result[~np.isfinite(source)] = np.nan
    return result


def _quantile_frames(frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], columns: list[str]):
    train, validation, test = (frame.copy() for frame in frames)
    fit = train.copy()
    for column in columns:
        for frame in (train, validation, test):
            frame[column] = _quantile_column(fit[column], frame[column])
    return (train, validation, test)


def _category_signature_mapping(
    frame: pd.DataFrame, column: str, nominal_columns: Iterable[str]
) -> dict[str, int]:
    """Target-free relabeling invariant to the feature's category names.

    Frequency is primary.  A deterministic hash of the aligned rows with this feature removed
    resolves ties using other-feature distributions without consulting the target.
    """
    labels = frame[column].astype(str)
    other = frame.drop(columns=[column]).copy()
    nominal_set = set(nominal_columns) - {column}
    for name in other.columns:
        if name in nominal_set:
            other_labels = other[name].astype(str)
            counts = other_labels.value_counts(dropna=False).to_dict()
            other[name] = other_labels.map(counts).astype("string")
        elif pd.api.types.is_numeric_dtype(other[name]):
            other[name] = pd.to_numeric(other[name], errors="coerce").round(10).astype("string").fillna("NA")
        else:
            other[name] = other[name].astype("string").fillna("NA")
    row_signatures = pd.util.hash_pandas_object(other, index=False).to_numpy(dtype=np.uint64)
    keys: list[tuple[tuple[int, str], str]] = []
    for label in sorted(labels.unique()):
        mask = labels.to_numpy() == label
        membership = np.sort(row_signatures[mask])
        digest = hashlib.sha256(membership.tobytes()).hexdigest()
        keys.append(((-int(mask.sum()), digest), label))
    unique_keys = {key: index for index, key in enumerate(sorted({key for key, _ in keys}))}
    return {label: unique_keys[key] for key, label in keys}


def _canonicalize_nominal(frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], columns: list[str]):
    train, validation, test = (frame.copy() for frame in frames)
    mappings = {}
    for column in columns:
        mapping = _category_signature_mapping(train, column, columns)
        mappings[column] = mapping
        for frame in (train, validation, test):
            frame[column] = frame[column].astype(str).map(mapping).fillna(-1).astype(float)
    return (train, validation, test), mappings


def _random_monotone_values(n_levels: int, rng: np.random.Generator) -> np.ndarray:
    if n_levels <= 1:
        return np.zeros(n_levels)
    gaps = np.exp(rng.uniform(-3.0, 1.5, size=n_levels - 1))
    values = np.concatenate([[0.0], np.cumsum(gaps)])
    return values / max(values[-1], EPS) * float(rng.uniform(1.0, 5.0))


def _monotone_pwl_fit(values: pd.Series, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    finite = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    knots = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, 9)))
    if len(knots) <= 1:
        return knots, knots.copy()
    slopes = np.exp(rng.uniform(-1.0, 1.0, size=len(knots) - 1))
    warped = np.concatenate([[0.0], np.cumsum(np.diff(knots) * slopes)])
    warped -= np.median(warped)
    return knots, warped


def _pwl_apply(values: pd.Series, knots: np.ndarray, warped: np.ndarray) -> np.ndarray:
    source = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if len(knots) <= 1:
        result = source.copy()
    else:
        result = np.interp(source, knots, warped)
        left_slope = (warped[1] - warped[0]) / max(knots[1] - knots[0], EPS)
        right_slope = (warped[-1] - warped[-2]) / max(knots[-1] - knots[-2], EPS)
        left = source < knots[0]
        right = source > knots[-1]
        result[left] = warped[0] + left_slope * (source[left] - knots[0])
        result[right] = warped[-1] + right_slope * (source[right] - knots[-1])
    result[~np.isfinite(source)] = np.nan
    return result


def _replace_cyclic_with_sincos(
    frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], periods: dict[str, int]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output = []
    for source in frames:
        frame = source.copy()
        for column, period in periods.items():
            theta = 2.0 * np.pi * pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float) / period
            position = frame.columns.get_loc(column)
            frame.insert(position, f"{column}__sin", np.sin(theta))
            frame.insert(position + 1, f"{column}__cos", np.cos(theta))
            frame = frame.drop(columns=[column])
        output.append(frame)
    return tuple(output)  # type: ignore[return-value]


def _rbf_frames(
    frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], feature: str, n_basis: int = 8
) -> tuple[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], dict[str, Any]]:
    train_values = pd.to_numeric(frames[0][feature], errors="coerce").to_numpy(dtype=float)
    finite = train_values[np.isfinite(train_values)]
    centers = np.quantile(finite, np.linspace(0.05, 0.95, n_basis))
    gaps = np.diff(np.unique(centers))
    width = float(np.median(gaps)) if len(gaps) else float(np.std(finite))
    width = max(width, float(np.std(finite)) * 0.05, EPS)
    output = []
    for source in frames:
        frame = source.copy()
        values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=float)
        basis = np.exp(-0.5 * ((values[:, None] - centers[None, :]) / width) ** 2)
        basis[~np.isfinite(values), :] = np.nan
        position = frame.columns.get_loc(feature)
        frame = frame.drop(columns=[feature])
        for index in reversed(range(n_basis)):
            frame.insert(position, f"{feature}__rbf{index}", basis[:, index])
        output.append(frame)
    return tuple(output), {"feature": feature, "centers": centers.tolist(), "width": width}  # type: ignore[return-value]


def _basis_matrix(kind: str, rng: np.random.Generator, n: int) -> tuple[np.ndarray, float]:
    left, _ = np.linalg.qr(rng.normal(size=(n, n)))
    right, _ = np.linalg.qr(rng.normal(size=(n, n)))
    if kind == "orthogonal":
        singular = np.ones(n)
    else:
        upper = 3.0 if kind == "cond_le_3" else 10.0
        condition = float(rng.uniform(1.25, upper))
        singular = np.geomspace(1.0, condition, n)
    matrix = left @ np.diag(singular) @ right.T
    return matrix, float(np.linalg.cond(matrix))


def _apply_basis_matrix(
    frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], feature: str, matrix: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    columns = [f"{feature}__rbf{index}" for index in range(matrix.shape[0])]
    output = []
    for source in frames:
        frame = source.copy()
        frame.loc[:, columns] = frame[columns].to_numpy(dtype=float) @ matrix.T
        output.append(frame)
    return tuple(output)  # type: ignore[return-value]


def build_representations(split: DatasetSplit, pipeline: str, orbit_members: int) -> list[Representation]:
    frames = split.frames(pipeline)
    nominal_metadata = split.nominal_columns
    numeric = split.numerical_columns
    categorical = _native_categories(frames[0], nominal_metadata)
    reps: list[Representation] = []

    original_id = f"original__{pipeline}"
    reps.append(_reference(original_id, "R0", "original", pipeline, frames, nominal_metadata))

    # T1 is the only family deliberately evaluated in both categorical pipelines.
    if nominal_metadata:
        for scope in ("one", "all"):
            for member in range(orbit_members):
                rng = np.random.default_rng(stable_seed(split.name, "T1", pipeline, scope, member))
                selected = [nominal_metadata[member % len(nominal_metadata)]] if scope == "one" else nominal_metadata
                changed = [frame.copy() for frame in frames]
                maps: dict[str, dict[str, str | float]] = {}
                for column in selected:
                    levels = sorted(changed[0][column].astype(str).unique().tolist())
                    permuted = rng.permutation(len(levels))
                    if pipeline == "native_categorical":
                        mapping = {level: f"opaque_{int(code):04d}_{member:02d}" for level, code in zip(levels, permuted)}
                        for frame in changed:
                            frame[column] = frame[column].astype(str).map(mapping).fillna("opaque_unknown")
                    else:
                        numeric_levels = sorted(pd.to_numeric(changed[0][column], errors="coerce").dropna().unique())
                        mapping = {str(level): float(code) for level, code in zip(numeric_levels, permuted)}
                        actual = {float(level): float(code) for level, code in zip(numeric_levels, permuted)}
                        for frame in changed:
                            frame[column] = pd.to_numeric(frame[column], errors="coerce").map(actual).fillna(-1.0)
                    maps[column] = mapping
                rep_id = f"T1_nominal_{pipeline}_{scope}_m{member}"
                reps.append(Representation(
                    rep_id, original_id, "T1", "nominal_relabeling", scope, member, pipeline, "none",
                    *changed, _native_categories(changed[0], nominal_metadata), {"mappings": maps, "bijective": True},
                ))

        # Target-free repair, generated once per pipeline from the all-feature relabelings.
        canonical_ref_id = f"nominal_canonical_ref__{pipeline}"
        canonical_ref, mapping = _canonicalize_nominal(frames, nominal_metadata)
        reps.append(_reference(
            canonical_ref_id, "R3", "nominal_canonical_reference", pipeline, canonical_ref, [],
            {"target_free": True, "mappings": mapping},
        ))
        for member in range(orbit_members):
            source = next(rep for rep in reps if rep.representation_id == f"T1_nominal_{pipeline}_all_m{member}")
            repaired, mapping = _canonicalize_nominal(
                (source.X_train, source.X_validation, source.X_test), nominal_metadata
            )
            reps.append(Representation(
                f"R3_nominal_canonical_{pipeline}_all_m{member}", canonical_ref_id, "T1", "nominal_relabeling",
                "all", member, pipeline, "nominal_canonicalization", *repaired, [],
                {"target_free": True, "mappings": mapping},
            ))

    # Remaining headline families use the common numeric-code pipeline, except categorical T4.
    if pipeline == "numeric_code":
        for member in range(orbit_members):
            rng = np.random.default_rng(stable_seed(split.name, "T0", member))
            order = rng.permutation(frames[0].columns).tolist()
            changed = tuple(frame.loc[:, order].copy() for frame in frames)
            reps.append(Representation(
                f"T0_column_permutation_m{member}", original_id, "T0", "column_permutation", "all", member,
                pipeline, "none", *changed, [], {"column_order": order},
            ))

        zref, zmeta = _zscore_frames(frames, numeric)
        zref_id = "zscore_reference"
        reps.append(_reference(zref_id, "R1", "zscore_reference", pipeline, zref, [], zmeta))
        for transform in ("scaling", "affine"):
            for scope in ("one", "all"):
                for member in range(orbit_members):
                    rng = np.random.default_rng(stable_seed(split.name, "T2", transform, scope, member))
                    selected = [numeric[member % len(numeric)]] if scope == "one" else numeric
                    changed = [frame.copy() for frame in frames]
                    metadata: dict[str, Any] = {"parameters": {}}
                    for column in selected:
                        factor = float(10.0 ** rng.uniform(-1.0, 1.0))
                        std = float(pd.to_numeric(frames[0][column], errors="coerce").std(ddof=0))
                        offset = 0.0 if transform == "scaling" else float(rng.uniform(-3.0, 3.0) * max(std, EPS))
                        for frame in changed:
                            frame[column] = pd.to_numeric(frame[column], errors="coerce") * factor + offset
                        metadata["parameters"][column] = {"a": factor, "b": offset}
                    rep_id = f"T2_{transform}_{scope}_m{member}"
                    reps.append(Representation(
                        rep_id, original_id, "T2", transform, scope, member, pipeline, "none", *changed, [], metadata,
                    ))
                    repaired, repair_meta = _zscore_frames(tuple(changed), numeric)
                    reps.append(Representation(
                        f"R1_{transform}_{scope}_m{member}", zref_id, "T2", transform, scope, member, pipeline,
                        "standardization", *repaired, [], {**metadata, "repair_fit": repair_meta},
                    ))

        qref = _quantile_frames(frames, numeric)
        qref_id = "quantile_reference"
        reps.append(_reference(qref_id, "R2", "quantile_reference", pipeline, qref, []))
        kinds = ("signed_log", "rank_quantile", "random_pwl")
        for member in range(orbit_members):
            kind = kinds[member % len(kinds)]
            rng = np.random.default_rng(stable_seed(split.name, "T3", kind, member))
            changed = [frame.copy() for frame in frames]
            metadata: dict[str, Any] = {"mapping_kind": kind, "columns": {}}
            for column in numeric:
                if kind == "signed_log":
                    center = float(pd.to_numeric(frames[0][column], errors="coerce").median())
                    scale = float(pd.to_numeric(frames[0][column], errors="coerce").std(ddof=0))
                    scale = max(scale, EPS)
                    for frame in changed:
                        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
                        frame[column] = np.sign(values - center) * np.log1p(np.abs(values - center) / scale)
                    metadata["columns"][column] = {"center": center, "scale": scale}
                elif kind == "rank_quantile":
                    fit = frames[0][column]
                    for frame in changed:
                        frame[column] = _quantile_column(fit, frame[column])
                    metadata["columns"][column] = {"train_unique": int(pd.Series(fit).nunique(dropna=True))}
                else:
                    knots, warped = _monotone_pwl_fit(frames[0][column], rng)
                    for frame in changed:
                        frame[column] = _pwl_apply(frame[column], knots, warped)
                    metadata["columns"][column] = {"knots": knots.tolist(), "warped": warped.tolist()}
            reps.append(Representation(
                f"T3_monotone_{kind}_m{member}", original_id, "T3", kind, "all", member, pipeline, "none",
                *changed, [], metadata,
            ))
            repaired = _quantile_frames(tuple(changed), numeric)
            reps.append(Representation(
                f"R2_monotone_{kind}_m{member}", qref_id, "T3", kind, "all", member, pipeline,
                "quantile_rank", *repaired, [], metadata,
            ))

        if split.ordinal_orders:
            rank_frames = [frame.copy() for frame in split.frames("native_categorical")]
            for column, order in split.ordinal_orders.items():
                mapping = {value: float(index) for index, value in enumerate(order)}
                for frame in rank_frames:
                    frame[column] = frame[column].astype(str).map(mapping).astype(float)
            ordinal_ref_id = "ordinal_integer_reference"
            reps.append(_reference(ordinal_ref_id, "T4", "ordinal_integer_reference", pipeline, tuple(rank_frames), []))
            rank_ref_id = "ordinal_rank_reference"
            normalized = tuple(frame.assign(**{
                column: frame[column] / max(len(order) - 1, 1)
                for column, order in split.ordinal_orders.items()
            }) for frame in rank_frames)
            reps.append(_reference(rank_ref_id, "R4", "ordinal_rank_reference", pipeline, normalized, []))
            columns = list(split.ordinal_orders)
            for scope in ("one", "all"):
                for member in range(orbit_members):
                    rng = np.random.default_rng(stable_seed(split.name, "T4", scope, member))
                    selected = [columns[member % len(columns)]] if scope == "one" else columns
                    changed = [frame.copy() for frame in rank_frames]
                    maps = {}
                    for column in selected:
                        values = _random_monotone_values(len(split.ordinal_orders[column]), rng)
                        maps[column] = values.tolist()
                        for frame in changed:
                            codes = frame[column].to_numpy(dtype=int)
                            frame[column] = values[codes]
                    reps.append(Representation(
                        f"T4_ordinal_integer_{scope}_m{member}", ordinal_ref_id, "T4", "ordinal_integer", scope,
                        member, pipeline, "none", *changed, [], {"spacing": maps, "strict_order": True},
                    ))
                    repaired = [frame.copy() for frame in changed]
                    for column, order in split.ordinal_orders.items():
                        source_values = np.asarray(maps.get(column, np.arange(len(order))), dtype=float)
                        value_to_rank = {float(value): rank / max(len(order) - 1, 1) for rank, value in enumerate(source_values)}
                        for frame in repaired:
                            frame[column] = frame[column].map(value_to_rank).astype(float)
                    reps.append(Representation(
                        f"R4_ordinal_rank_{scope}_m{member}", rank_ref_id, "T4", "ordinal_integer", scope,
                        member, pipeline, "ordinal_canonicalization", *repaired, [],
                        {"spacing": maps, "known_order": split.ordinal_orders},
                    ))

        if split.cyclic_periods:
            sincos_ref = _replace_cyclic_with_sincos(frames, split.cyclic_periods)
            sincos_ref_id = "cyclic_sincos_reference"
            reps.append(_reference(
                sincos_ref_id, "T5", "cyclic_sincos_reference", pipeline, sincos_ref, [],
                {"periods": split.cyclic_periods, "origin": 0},
            ))
            cyclic_columns = list(split.cyclic_periods)
            for scope in ("one", "all"):
                for member in range(orbit_members):
                    rng = np.random.default_rng(stable_seed(split.name, "T5", "shift", scope, member))
                    selected = [cyclic_columns[member % len(cyclic_columns)]] if scope == "one" else cyclic_columns
                    changed = [frame.copy() for frame in frames]
                    shifts = {}
                    for column in selected:
                        period = split.cyclic_periods[column]
                        shift = int(rng.integers(1, period))
                        shifts[column] = shift
                        for frame in changed:
                            frame[column] = (pd.to_numeric(frame[column], errors="coerce") + shift) % period
                    reps.append(Representation(
                        f"T5_cyclic_shift_{scope}_m{member}", original_id, "T5", "cyclic_shift", scope, member,
                        pipeline, "none", *changed, [], {"periods": split.cyclic_periods, "shifts": shifts},
                    ))
                    canonical = [frame.copy() for frame in changed]
                    for column, shift in shifts.items():
                        period = split.cyclic_periods[column]
                        for frame in canonical:
                            frame[column] = (pd.to_numeric(frame[column], errors="coerce") - shift) % period
                    canonical_sincos = _replace_cyclic_with_sincos(tuple(canonical), split.cyclic_periods)
                    reps.append(Representation(
                        f"R5_cyclic_frontend_{scope}_m{member}", sincos_ref_id, "T5", "cyclic_shift", scope,
                        member, pipeline, "cyclic_frontend", *canonical_sincos, [],
                        {"periods": split.cyclic_periods, "shifts": shifts, "metadata_used": True},
                    ))
            for member in range(orbit_members):
                rng = np.random.default_rng(stable_seed(split.name, "T5", "rotation", member))
                rotated = [frame.copy() for frame in sincos_ref]
                angles = {}
                for column in cyclic_columns:
                    angle = float(rng.uniform(-np.pi, np.pi))
                    angles[column] = angle
                    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
                    names = [f"{column}__sin", f"{column}__cos"]
                    for frame in rotated:
                        frame.loc[:, names] = frame[names].to_numpy(dtype=float) @ rotation.T
                reps.append(Representation(
                    f"T5_cyclic_rotation_all_m{member}", sincos_ref_id, "T5", "cyclic_rotation", "all", member,
                    pipeline, "none", *rotated, [], {"angles": angles, "determinant": 1.0},
                ))

        basis_candidates = [column for column in numeric if frames[0][column].nunique(dropna=True) >= 8]
        if basis_candidates:
            feature = max(basis_candidates, key=lambda column: frames[0][column].nunique(dropna=True))
            basis_frames, basis_meta = _rbf_frames(frames, feature, 8)
            basis_ref_id = f"basis_reference__{feature}"
            reps.append(_reference(basis_ref_id, "T6", "rbf_basis_reference", pipeline, basis_frames, [], basis_meta))
            for kind in ("orthogonal", "cond_le_3", "cond_le_10"):
                for member in range(orbit_members):
                    rng = np.random.default_rng(stable_seed(split.name, "T6", kind, member))
                    matrix, condition = _basis_matrix(kind, rng, 8)
                    changed = _apply_basis_matrix(basis_frames, feature, matrix)
                    reps.append(Representation(
                        f"T6_basis_{kind}_m{member}", basis_ref_id, "T6", kind, "one", member, pipeline, "none",
                        *changed, [], {**basis_meta, "matrix": matrix.tolist(), "condition_number": condition},
                    ))

    # Native categorical treatment for T4: arbitrary numeric labels retain categorical metadata.
    if pipeline == "native_categorical" and split.ordinal_orders:
        source = split.frames("native_categorical")
        columns = list(split.ordinal_orders)
        reference = [frame.copy() for frame in source]
        for column, order in split.ordinal_orders.items():
            mapping = {value: f"rank_{index}" for index, value in enumerate(order)}
            for frame in reference:
                frame[column] = frame[column].astype(str).map(mapping)
        reference_id = "ordinal_categorical_reference"
        reps.append(_reference(
            reference_id, "T4", "ordinal_categorical_reference", pipeline, tuple(reference), nominal_metadata
        ))
        for scope in ("one", "all"):
            for member in range(orbit_members):
                rng = np.random.default_rng(stable_seed(split.name, "T4", "categorical", scope, member))
                selected = [columns[member % len(columns)]] if scope == "one" else columns
                changed = [frame.copy() for frame in reference]
                maps = {}
                for column in selected:
                    values = _random_monotone_values(len(split.ordinal_orders[column]), rng)
                    mapping = {f"rank_{index}": f"space_{value:.12g}" for index, value in enumerate(values)}
                    maps[column] = mapping
                    for frame in changed:
                        frame[column] = frame[column].astype(str).map(mapping)
                reps.append(Representation(
                    f"T4_ordinal_categorical_{scope}_m{member}", reference_id, "T4", "ordinal_categorical",
                    scope, member, pipeline, "none", *changed, nominal_metadata,
                    {"spacing_labels": maps, "categorical_metadata": True},
                ))

    ids = [rep.representation_id for rep in reps]
    if len(ids) != len(set(ids)):
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        raise RuntimeError(f"duplicate representation ids: {duplicates}")
    references = {rep.representation_id for rep in reps if rep.is_reference}
    missing = sorted({rep.reference_id for rep in reps} - references)
    if missing:
        raise RuntimeError(f"missing references: {missing}")
    return reps


def _tabpfn_checkpoint(problem_type: str) -> Path:
    filename = f"tabpfn-v2.6-{'regressor' if problem_type == 'regression' else 'classifier'}-v2.6_default.ckpt"
    matches = list(Path.home().glob(f".cache/huggingface/hub/models--Prior-Labs--tabpfn_2_6/snapshots/*/{filename}"))
    if not matches:
        raise FileNotFoundError(f"missing exact TabPFN-2.6 checkpoint {filename}")
    return matches[0].resolve()


def fit_predict(
    model_name: str,
    problem_type: str,
    rep: Representation,
    y_train: np.ndarray,
    seed: int,
    device: str,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    cat_indices = [rep.X_train.columns.get_loc(column) for column in rep.categorical_columns]
    if model_name == "tabpfn_2_6":
        from tabpfn import TabPFNClassifier, TabPFNRegressor

        checkpoint = _tabpfn_checkpoint(problem_type)
        cls = TabPFNRegressor if problem_type == "regression" else TabPFNClassifier
        model = cls(
            n_estimators=int(config["tfm_estimators"]),
            categorical_features_indices=cat_indices or None,
            model_path=checkpoint,
            device=device,
            random_state=seed,
            inference_precision="autocast",
            fit_mode="fit_preprocessors",
            show_progress_bar=False,
        )
        model.fit(rep.X_train, y_train)
        fit_seconds = time.perf_counter() - started
        predict_started = time.perf_counter()
        prediction = model.predict(rep.X_test) if problem_type == "regression" else model.predict_proba(rep.X_test)
        telemetry = {
            "fit_seconds": fit_seconds,
            "predict_seconds": time.perf_counter() - predict_started,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "n_estimators": int(config["tfm_estimators"]),
        }
    elif model_name == "tabicl_v2":
        from tabicl import TabICLClassifier, TabICLRegressor

        if rep.categorical_columns:
            raise ValueError("TabICLv2 is restricted to the declared numeric-code pipeline")
        cls = TabICLRegressor if problem_type == "regression" else TabICLClassifier
        model = cls(
            n_estimators=int(config["tfm_estimators"]),
            device=device,
            use_amp=True,
            random_state=seed,
            batch_size=1,
            allow_auto_download=True,
        )
        model.fit(rep.X_train, y_train)
        fit_seconds = time.perf_counter() - started
        predict_started = time.perf_counter()
        prediction = model.predict(rep.X_test) if problem_type == "regression" else model.predict_proba(rep.X_test)
        checkpoint = Path(model.model_path_).resolve()
        telemetry = {
            "fit_seconds": fit_seconds,
            "predict_seconds": time.perf_counter() - predict_started,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_version": model.checkpoint_version,
            "n_estimators": int(config["tfm_estimators"]),
        }
    elif model_name == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor

        train, test = rep.X_train.copy(), rep.X_test.copy()
        for column in rep.categorical_columns:
            train[column] = train[column].astype("string").fillna("__MISSING__")
            test[column] = test[column].astype("string").fillna("__MISSING__")
        common = dict(
            iterations=int(config["catboost_iterations"]), depth=7, learning_rate=0.05,
            random_seed=seed, thread_count=4, allow_writing_files=False, verbose=False,
        )
        if problem_type == "regression":
            model = CatBoostRegressor(loss_function="RMSE", **common)
        else:
            model = CatBoostClassifier(loss_function="Logloss", **common)
        model.fit(train, y_train, cat_features=rep.categorical_columns)
        fit_seconds = time.perf_counter() - started
        predict_started = time.perf_counter()
        prediction = model.predict(test).reshape(-1) if problem_type == "regression" else model.predict_proba(test)
        telemetry = {
            "fit_seconds": fit_seconds,
            "predict_seconds": time.perf_counter() - predict_started,
            "iterations": int(config["catboost_iterations"]),
        }
    else:
        raise ValueError(f"unknown model {model_name}")
    result = np.asarray(prediction, dtype=float)
    del model
    gc.collect()
    try:
        import torch
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    except Exception:
        pass
    return result, telemetry


def prediction_metrics(problem_type: str, y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if problem_type == "classification":
        probs = np.asarray(prediction, dtype=float)
        probs = np.clip(probs, 1e-8, 1.0)
        probs /= probs.sum(axis=1, keepdims=True)
        result = {
            "log_loss": float(log_loss(y, probs, labels=np.arange(probs.shape[1]))),
            "accuracy": float(accuracy_score(y, probs.argmax(axis=1))),
        }
        try:
            result["roc_auc"] = float(
                roc_auc_score(y, probs[:, 1]) if probs.shape[1] == 2 else roc_auc_score(y, probs, multi_class="ovr")
            )
        except ValueError:
            result["roc_auc"] = float("nan")
        return result
    pred = np.asarray(prediction, dtype=float).reshape(-1)
    return {
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "mae": float(mean_absolute_error(y, pred)),
    }


def disagreement_metrics(
    problem_type: str, y: np.ndarray, reference: np.ndarray, prediction: np.ndarray
) -> dict[str, float]:
    if problem_type == "classification":
        p = np.clip(np.asarray(reference, dtype=float), 1e-8, 1.0)
        q = np.clip(np.asarray(prediction, dtype=float), 1e-8, 1.0)
        p /= p.sum(axis=1, keepdims=True)
        q /= q.sum(axis=1, keepdims=True)
        midpoint = 0.5 * (p + q)
        js = 0.5 * np.sum(p * np.log(p / midpoint), axis=1) + 0.5 * np.sum(q * np.log(q / midpoint), axis=1)
        return {
            "probability_mad": float(np.mean(np.abs(p - q))),
            "js_divergence": float(np.mean(js)),
            "label_flip_rate": float(np.mean(p.argmax(axis=1) != q.argmax(axis=1))),
        }
    p = np.asarray(reference, dtype=float).reshape(-1)
    q = np.asarray(prediction, dtype=float).reshape(-1)
    y_scale = max(float(np.std(y)), EPS)
    pearson = float(np.corrcoef(p, q)[0, 1]) if np.std(p) > EPS and np.std(q) > EPS else float("nan")
    spear = float(spearmanr(p, q).statistic) if len(p) > 1 else float("nan")
    return {
        "prediction_rmse_normalized": float(np.sqrt(np.mean((p - q) ** 2)) / y_scale),
        "prediction_pearson": pearson,
        "prediction_spearman": spear,
    }


def environment_metadata() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "tabpfn", "tabicl", "catboost"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    metadata: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch
        metadata["torch_cuda"] = torch.version.cuda
        metadata["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        metadata["gpu"] = None
    return metadata


def synthetic_sanity(seed: int = 20260901, n: int = 4096) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    nominal = rng.integers(0, 4, size=n)
    ordinal = rng.integers(0, 5, size=n)
    cyclic = rng.integers(0, 24, size=n)
    ratio = rng.lognormal(0.2, 0.8, size=n)
    interval = rng.normal(10.0, 3.0, size=n)
    noise = rng.normal(size=(n, 3))
    nominal_effect = np.array([-0.7, 0.2, 0.8, -0.1])
    structural = (
        nominal_effect[nominal] + 0.7 * ordinal + np.sin(2 * np.pi * cyclic / 24)
        + 0.5 * np.log1p(ratio) + 0.3 * interval
    )
    target = structural + rng.normal(0.0, 0.1, size=n)
    nominal_map = rng.permutation(4)
    ordinal_spacing = _random_monotone_values(5, rng)
    shift = 7
    scale, offset = 3.4, -12.0
    transformed = {
        "nominal": nominal_map[nominal],
        "ordinal": ordinal_spacing[ordinal],
        "cyclic": (cyclic + shift) % 24,
        "ratio": scale * ratio,
        "interval": interval + offset,
        "noise": noise,
    }
    inverse_nominal = np.argsort(nominal_map)[transformed["nominal"]]
    inverse_ordinal = np.searchsorted(ordinal_spacing, transformed["ordinal"])
    inverse_cyclic = (transformed["cyclic"] - shift) % 24
    inverse_ratio = transformed["ratio"] / scale
    inverse_interval = transformed["interval"] - offset
    reconstructed = (
        nominal_effect[inverse_nominal] + 0.7 * inverse_ordinal + np.sin(2 * np.pi * inverse_cyclic / 24)
        + 0.5 * np.log1p(inverse_ratio) + 0.3 * inverse_interval
    )
    return {
        "seed": seed,
        "n_rows": n,
        "target_unchanged": True,
        "max_structural_function_delta": float(np.max(np.abs(structural - reconstructed))),
        "nominal_bijection": len(np.unique(nominal_map)) == 4,
        "ordinal_strictly_increasing": bool(np.all(np.diff(ordinal_spacing) > 0)),
        "cyclic_inverse_exact": bool(np.array_equal(cyclic, inverse_cyclic)),
        "ratio_inverse_max_error": float(np.max(np.abs(ratio - inverse_ratio))),
        "interval_inverse_max_error": float(np.max(np.abs(interval - inverse_interval))),
        "target_mean": float(target.mean()),
    }
