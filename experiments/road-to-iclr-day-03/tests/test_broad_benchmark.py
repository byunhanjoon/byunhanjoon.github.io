import numpy as np
import torch

from experiments.day3.broad_data import (
    controlled_representation,
    natural_blockwise_equivalence_errors,
    paired_natural_representations,
    sketched_anchor_canonicalize,
)
from experiments.day3.broad_models import make_model
from experiments.day3.core import Dataset


def synthetic_dataset() -> Dataset:
    rng = np.random.default_rng(42)
    sizes = {"train": 300, "val": 80, "test": 90}
    x_num = {part: rng.normal(size=(size, 3)) for part, size in sizes.items()}
    x_cat = {
        part: np.column_stack(
            (rng.choice(["a", "b", "c"], size=size), rng.choice(["x", "y"], size=size))
        )
        for part, size in sizes.items()
    }
    y = {part: rng.integers(0, 2, size=size) for part, size in sizes.items()}
    return Dataset("synthetic", "binclass", x_num, None, x_cat, y, 2, "synthetic")


def test_controlled_transform_has_requested_condition_and_fixed_energy():
    dataset = synthetic_dataset()
    low = controlled_representation(dataset, 1.0)
    high = controlled_representation(dataset, 1000.0)
    assert np.isclose(high.metadata["basis_condition"], 1000.0, rtol=1e-10)
    assert np.isclose(
        high.metadata["transformed_trace"], high.metadata["reference_trace"], rtol=1e-12
    )
    assert max(high.metadata["basis_relation_errors"].values()) < 1e-14
    assert low.parts["train"].shape == high.parts["train"].shape


def test_natural_pair_is_blockwise_affine_equivalent():
    errors = natural_blockwise_equivalence_errors(synthetic_dataset())
    assert max(value for direction in errors.values() for value in direction.values()) < 1e-10


def test_natural_pair_exposes_an_exact_invertible_global_map():
    reference, changed, transform = paired_natural_representations(synthetic_dataset())
    assert transform.shape[0] == transform.shape[1]
    assert np.linalg.matrix_rank(transform) == len(transform)
    for part in ("train", "val", "test"):
        assert np.allclose(reference.parts[part] @ transform, changed.parts[part], atol=1e-9)


def test_all_broad_models_have_dense_first_affine_layer():
    for name in ("mlp", "resnet", "dense_stem_ft_transformer", "dense_stem_tabm"):
        model = make_model(name, 13, 3)
        output = model(torch.randn(5, 13))
        assert output.shape == (5, 3)
        assert model.first.in_features == 13


def test_sketched_anchor_coordinates_are_gl_invariant():
    rng = np.random.default_rng(7)
    train = rng.normal(size=(600, 12))
    parts = {"train": train, "val": rng.normal(size=(80, 12)), "test": rng.normal(size=(90, 12))}
    q1, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    q2, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    transform = q1 @ np.diag(np.geomspace(1 / np.sqrt(1000), np.sqrt(1000), 12)) @ q2
    changed = {part: values @ transform for part, values in parts.items()}
    canonical, meta = sketched_anchor_canonicalize(parts, initial_rows=128)
    canonical_changed, changed_meta = sketched_anchor_canonicalize(changed, initial_rows=128)
    assert meta["anchor_rows"] == changed_meta["anchor_rows"]
    for part in parts:
        assert np.allclose(canonical[part], canonical_changed[part], atol=2e-9, rtol=2e-9)
