from __future__ import annotations

import numpy as np

from tabalu.models.typed import (
    HeterogeneousBatch,
    categorical_membership,
    datetime_parts,
    typed_design_matrix,
)
from tabalu.synthetic.heterogeneous import generate_heterogeneous_task, sample_heterogeneous_split


def test_typed_operators_have_expected_semantics() -> None:
    hours = np.array([0, 24, 24 * 4 + 23])
    parts = datetime_parts(hours)
    np.testing.assert_array_equal(parts["hour"], [0, 0, 23])
    np.testing.assert_array_equal(parts["weekday"], [3, 4, 0])
    np.testing.assert_array_equal(categorical_membership(np.array([0, 1, 2]), (0, 2)), [1, 0, 1])


def test_heterogeneous_target_is_in_full_typed_library() -> None:
    task = generate_heterogeneous_task(73)
    batch, targets = sample_heterogeneous_split(task, "train", 128, seed=2)
    design, names = typed_design_matrix(batch)
    lookup = {name: design[:, index] for index, name in enumerate(names)}
    expected = sum(coefficient * lookup[name] for name, coefficient in zip(task.active_feature_names, task.coefficients))
    np.testing.assert_allclose(targets, expected)
    assert isinstance(batch, HeterogeneousBatch)
