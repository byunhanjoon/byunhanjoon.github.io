from __future__ import annotations

import numpy as np

from experiments.day3.mixed_measure_ple import (
    atom_values,
    fixed_ple_block,
    mixed_measure_block,
)
from experiments.day3.measure_orbit import _member_tensor
import torch


def parts(train: np.ndarray, query: np.ndarray | None = None) -> dict[str, np.ndarray]:
    train = np.asarray(train, dtype=np.float64)[:, None]
    query = train if query is None else np.asarray(query, dtype=np.float64)[:, None]
    return {"train": train, "val": query, "test": query.copy()}


def test_atom_selection_is_frequency_only_and_leaves_nonatom_support() -> None:
    train = np.asarray([0.0] * 80 + [1.0] * 10 + list(range(2, 12)))
    selected = atom_values(train, budget=10, maximum_atoms=3, minimum_nonatom_rows=15)
    np.testing.assert_array_equal(selected, [0.0])


def test_all_arms_use_exactly_the_fixed_budget() -> None:
    values = parts(np.asarray([0.0] * 80 + list(range(1, 21))), [-1.0, 0.0, 5.0, 30.0])
    baseline, _ = fixed_ple_block(values, 0, budget=16)
    tail, _ = mixed_measure_block(values, 0, 16, 4, 8, include_atom_coordinates=False)
    mixed, _ = mixed_measure_block(values, 0, 16, 4, 8, include_atom_coordinates=True)
    for block in (baseline, tail, mixed):
        assert {matrix.shape[1] for matrix in block.values()} == {16}


def test_atom_and_nonatom_components_have_disjoint_support() -> None:
    values = parts(np.asarray([0.0] * 80 + list(range(1, 21))), [0.0, 1.0, 7.0, 30.0])
    mixed, metadata = mixed_measure_block(values, 0, 16, 4, 8, include_atom_coordinates=True)
    assert metadata["atoms"] == [0.0]
    query = mixed["val"]
    assert query[0, 0] == 1.0
    assert np.all(query[0, 1:] == 0.0)
    assert np.all(query[1:, 0] == 0.0)
    assert np.all(query[1:, 1] == 1.0)


def test_unknown_values_use_continuous_fallback() -> None:
    values = parts(np.asarray([0.0] * 80 + list(range(1, 21))), [100.0])
    mixed, _ = mixed_measure_block(values, 0, 16, 4, 8, include_atom_coordinates=True)
    assert mixed["val"][0, 0] == 0.0
    assert mixed["val"][0, 1] == 1.0
    assert mixed["val"][0, 2:].sum() > 0.0


def test_member_assignment_preserves_requested_order() -> None:
    features = {
        "a": torch.full((2, 3), 1.0),
        "b": torch.full((2, 3), 2.0),
    }
    output = _member_tensor(features, ["a", "b", "a"])
    assert output.shape == (2, 3, 3)
    torch.testing.assert_close(output[:, 0], features["a"])
    torch.testing.assert_close(output[:, 1], features["b"])
