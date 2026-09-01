from __future__ import annotations

import numpy as np
import pandas as pd

from src.semantic_orbits import (
    DatasetSplit,
    _basis_matrix,
    _canonicalize_nominal,
    build_representations,
    disagreement_metrics,
    synthetic_sanity,
)


def _tiny_split() -> DatasetSplit:
    rng = np.random.default_rng(42)
    n_train, n_validation, n_test = 96, 24, 32
    n = n_train + n_validation + n_test
    nominal = np.array(["red", "blue", "green", "yellow"])[np.arange(n) % 4]
    ordinal = np.array(["low", "middle", "high"])[np.arange(n) % 3]
    hour = np.arange(n) % 24
    ratio = rng.lognormal(size=n)
    interval = rng.normal(size=n)
    native = pd.DataFrame({
        "nominal": nominal,
        "ordinal": ordinal,
        "hour": hour.astype(float),
        "ratio": ratio,
        "interval": interval,
    })
    numeric = native.copy()
    numeric["nominal"] = numeric["nominal"].map({"blue": 0, "green": 1, "red": 2, "yellow": 3}).astype(float)
    numeric["ordinal"] = numeric["ordinal"].map({"low": 0, "middle": 1, "high": 2}).astype(float)
    slices = (slice(0, n_train), slice(n_train, n_train + n_validation), slice(n_train + n_validation, n))
    frames_native = [native.iloc[item].reset_index(drop=True) for item in slices]
    frames_numeric = [numeric.iloc[item].reset_index(drop=True) for item in slices]
    y = 0.3 * interval + np.sin(2 * np.pi * hour / 24)
    return DatasetSplit(
        "tiny", 0, 1, "regression",
        *frames_native, *frames_numeric,
        y[:n_train], y[n_train:n_train + n_validation], y[-n_test:],
        np.arange(n_train), np.arange(n_train, n_train + n_validation), np.arange(n_train + n_validation, n),
        ["nominal", "ordinal"], ["hour", "ratio", "interval"],
        {"ordinal": ["low", "middle", "high"]}, {"hour": 24},
    )


def test_synthetic_semantics_are_exactly_reconstructable() -> None:
    audit = synthetic_sanity(n=512)
    assert audit["nominal_bijection"]
    assert audit["ordinal_strictly_increasing"]
    assert audit["cyclic_inverse_exact"]
    assert audit["max_structural_function_delta"] < 1e-10


def test_basis_matrices_are_invertible_and_conditioned() -> None:
    rng = np.random.default_rng(1)
    for kind, upper in (("orthogonal", 1.000001), ("cond_le_3", 3.0), ("cond_le_10", 10.0)):
        matrix, condition = _basis_matrix(kind, rng, 8)
        assert np.linalg.matrix_rank(matrix) == 8
        assert condition <= upper + 1e-10
        assert np.allclose(np.linalg.solve(matrix, matrix), np.eye(8), atol=1e-10)


def test_nominal_canonicalization_ignores_arbitrary_names() -> None:
    split = _tiny_split()
    frames = split.frames("native_categorical")
    relabeled = tuple(frame.assign(
        nominal=frame["nominal"].map({"red": "x7", "blue": "x2", "green": "x9", "yellow": "x4"}),
        ordinal=frame["ordinal"].map({"low": "z3", "middle": "z8", "high": "z1"}),
    ) for frame in frames)
    original, _ = _canonicalize_nominal(frames, split.nominal_columns)
    transformed, _ = _canonicalize_nominal(relabeled, split.nominal_columns)
    for left, right in zip(original, transformed):
        assert np.array_equal(left[split.nominal_columns].to_numpy(), right[split.nominal_columns].to_numpy())


def test_builder_has_eight_members_and_valid_references() -> None:
    split = _tiny_split()
    reps = build_representations(split, "numeric_code", 8)
    references = {rep.representation_id for rep in reps if rep.is_reference}
    assert all(rep.reference_id in references for rep in reps)
    for family in ("T0", "T1", "T2", "T3", "T4", "T5", "T6"):
        assert sum(rep.family == family and not rep.is_reference for rep in reps) >= 8
    for rep in reps:
        assert len(rep.X_train) == len(split.y_train)
        assert len(rep.X_test) == len(split.y_test)


def test_classification_disagreement_metrics() -> None:
    y = np.array([0, 1, 0, 1])
    p = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.4, 0.6]])
    same = disagreement_metrics("classification", y, p, p)
    assert same["probability_mad"] == 0.0
    assert same["js_divergence"] == 0.0
    assert same["label_flip_rate"] == 0.0
