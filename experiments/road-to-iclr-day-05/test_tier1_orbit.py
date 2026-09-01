from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("tier1_orbit", HERE / "tier1_orbit.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_category_and_feature_rendering_tracks_categorical_indices() -> None:
    numerical = np.asarray([[1.0], [2.0]])
    categorical = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    permutation = np.asarray([2, 0, 1])
    rendered, indices = MODULE.render(
        numerical,
        categorical,
        permutation,
        [np.asarray([1, 0]), np.asarray([0, 1])],
    )
    assert indices == (0, 2)
    np.testing.assert_array_equal(rendered[:, 0], categorical[:, 1])
    np.testing.assert_array_equal(rendered[:, 2], 1 - categorical[:, 0])


def test_binary_probability_alignment() -> None:
    semantic = np.asarray([[0.8, 0.2], [0.3, 0.7]])
    swap = np.asarray([1, 0])
    rendered_outputs = semantic[:, swap]
    np.testing.assert_array_equal(rendered_outputs[:, swap], semantic)
