from __future__ import annotations

import numpy as np

from tabalu.synthetic import generate_program_task, regenerate_split


def test_task_and_splits_are_exactly_regenerable() -> None:
    first = generate_program_task(1234)
    second = generate_program_task(1234)
    assert first.to_dict() == second.to_dict()
    x1, y1 = regenerate_split(first, "train", 64)
    x2, y2 = regenerate_split(second, "train", 64)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)


def test_magnitude_ood_is_outside_training_support() -> None:
    task = generate_program_task(42)
    features, targets = regenerate_split(task, "ood_test", 128, multiplier=4)
    assert np.all(np.abs(features) >= 2.25)
    assert np.all(np.abs(features) <= 8.0)
    assert np.isfinite(targets).all()
