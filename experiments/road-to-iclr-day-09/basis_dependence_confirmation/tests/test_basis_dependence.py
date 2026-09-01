from __future__ import annotations

import numpy as np
import pandas as pd

from src.basis_dependence import (
    BlockData,
    DatasetData,
    anchor_canonical_representation,
    build_primary_representations,
    build_rbf_feature_matrix,
    conditioned_matrix,
    fourier_origin_pairs,
    helmert_pair,
    local_spectral_pair,
    orthogonal_matrix,
    pca_canonical_representation,
)
from src.mechanism import (
    function_matched_copy,
    make_controlled_mlp,
    max_logit_difference,
    minibatch_orders,
    order_sha256,
)


def synthetic_dataset(seed: int = 5) -> DatasetData:
    rng = np.random.default_rng(seed)
    n_train, n_validation, n_test = 128, 32, 48
    frames = []
    for n, offset in ((n_train, 0), (n_validation, 1), (n_test, 2)):
        frames.append(pd.DataFrame({
            "x": rng.normal(offset * 0.1, 2.0, n),
            "u": rng.uniform(-2, 2, n),
            "category": np.resize(np.array(["a", "b", "c", "d"]), n),
        }))
    targets = [rng.normal(size=n) for n in (n_train, n_validation, n_test)]
    return DatasetData(
        "synthetic", 0, 1, "development", "regression", *frames, *targets,
        np.arange(n_train), np.arange(n_validation) + n_train,
        np.arange(n_test) + n_train + n_validation,
        ["category"], ["x", "u"], {},
    )


def config() -> dict:
    return {"minimum_continuous_unique_values": 16, "basis_dimension": 8}


def test_orthogonal_and_conditioned_matrices() -> None:
    q = orthogonal_matrix(8, 17)
    assert np.linalg.norm(q.T @ q - np.eye(8)) < 1e-12
    assert np.linalg.det(q) > 0
    a = conditioned_matrix(8, 17)
    assert np.linalg.cond(a) <= 3 + 1e-12
    assert np.linalg.matrix_rank(a) == 8


def test_primary_transforms_reconstruct_and_never_mix_blocks() -> None:
    blocks = build_rbf_feature_matrix(synthetic_dataset(), config())
    reps = build_primary_representations(blocks, 8)
    assert len(reps) == 25
    for rep in reps[1:]:
        for audit in rep.metadata["equivalence"].values():
            assert audit["reconstruction_error"] < 1e-12
            if rep.variant.startswith("orthogonal"):
                assert audit["orthogonality_error"] < 1e-12
        if rep.scope == "one":
            assert set(rep.transforms) == {blocks.selected_feature}
        else:
            assert set(rep.transforms) == set(blocks.feature_blocks)


def test_anchor_canonical_is_invariant_without_transform_matrix() -> None:
    blocks = build_rbf_feature_matrix(synthetic_dataset(), config())
    reps = build_primary_representations(blocks, 1)
    reference = anchor_canonical_representation(reps[0], blocks.dataset.key)
    for rep in reps[1:]:
        transformed = anchor_canonical_representation(rep, blocks.dataset.key)
        assert all(item["full_rank"] for item in transformed.metadata["anchor"].values())
        assert np.linalg.norm(reference.X_train - transformed.X_train) / np.linalg.norm(reference.X_train) < 1e-10
        assert np.linalg.norm(reference.X_test - transformed.X_test) / np.linalg.norm(reference.X_test) < 1e-10


def test_pca_canonical_is_invariant_when_spectrum_is_distinct() -> None:
    rng = np.random.default_rng(9)
    train = rng.normal(size=(256, 8)) @ np.diag(np.arange(1, 9))
    validation = rng.normal(size=(32, 8)) @ np.diag(np.arange(1, 9))
    test = rng.normal(size=(48, 8)) @ np.diag(np.arange(1, 9))
    data = synthetic_dataset()
    blocks = BlockData(data, train, validation, test, [f"z{i}" for i in range(8)], {"x": list(range(8))}, {}, {}, "x", {})
    reps = build_primary_representations(blocks, 1)
    base = pca_canonical_representation(reps[0])
    rotated = pca_canonical_representation(next(rep for rep in reps if rep.variant == "orthogonal_one"))
    assert not base.metadata["pca"]["x"]["degenerate"]
    assert np.allclose(base.X_train, rotated.X_train, atol=1e-10)
    assert np.allclose(base.X_test, rotated.X_test, atol=1e-10)


def test_natural_pairs_are_exactly_reconstructable() -> None:
    blocks = build_rbf_feature_matrix(synthetic_dataset(), config())
    helmert_result = helmert_pair(blocks)
    assert helmert_result is not None
    _, helmert_rep = helmert_result
    assert helmert_rep.metadata["reconstruction_error"] < 1e-12
    _, spectral_rep = local_spectral_pair(blocks)
    assert spectral_rep.metadata["reconstruction_error"] < 1e-12
    assert spectral_rep.metadata["condition_number"] <= 1 + 1e-12


def test_fourier_origin_is_an_exact_block_rotation() -> None:
    data = synthetic_dataset()
    for frame in (data.X_train_raw, data.X_validation_raw, data.X_test_raw):
        frame["hour"] = np.arange(len(frame)) % 24
    data.numerical_columns.append("hour")
    data.cyclic_periods = {"hour": 24}
    blocks = build_rbf_feature_matrix(data, config())
    pairs = fourier_origin_pairs(blocks, 8)
    assert len(pairs) == 9
    assert all(rep.metadata.get("reconstruction_error", 0.0) < 1e-12 for rep in pairs)


def test_preprocessing_fits_train_only() -> None:
    first = synthetic_dataset()
    second = synthetic_dataset()
    second.X_validation_raw.loc[:, "x"] = 1e9
    second.X_test_raw.loc[:, "x"] = -1e9
    left = build_rbf_feature_matrix(first, config())
    right = build_rbf_feature_matrix(second, config())
    assert left.basis_metadata == right.basis_metadata
    assert np.array_equal(left.X_train, right.X_train)


def test_function_matched_initial_predictions() -> None:
    blocks = build_rbf_feature_matrix(synthetic_dataset(), config())
    transformed = next(
        rep for rep in build_primary_representations(blocks, 1)
        if rep.variant == "orthogonal_all"
    )
    model_config = {
        "hidden_layers": 3, "width": 256, "activation": "GELU",
    }
    reference = make_controlled_mlp(blocks.X_train.shape[1], 1, model_config, 0, "cpu")
    matched = function_matched_copy(reference, transformed, "cpu")
    difference = max_logit_difference(
        reference, matched, blocks.X_train, transformed.X_train, "cpu", maximum_rows=1000
    )
    assert difference < 1e-5


def test_paired_minibatch_orders_are_byte_identical() -> None:
    left = minibatch_orders(1024, 25, 7)
    right = minibatch_orders(1024, 25, 7)
    assert [order_sha256(order) for order in left] == [order_sha256(order) for order in right]
    assert all(np.array_equal(a, b) for a, b in zip(left, right))
