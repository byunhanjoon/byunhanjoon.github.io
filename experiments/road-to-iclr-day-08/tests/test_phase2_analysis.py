from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.phase2 import applicable_transformed, hierarchical_bootstrap


def test_hierarchical_bootstrap_is_deterministic_and_equal_dataset_weighted():
    frame = pd.DataFrame(
        {
            "dataset": ["a", "a", "b", "b"],
            "split_seed": [1, 2, 1, 2],
            "effect": [0.0, 2.0, 10.0, 14.0],
        }
    )
    first = hierarchical_bootstrap(frame, "effect", seed=7, draws=2_000)
    second = hierarchical_bootstrap(frame, "effect", seed=7, draws=2_000)
    assert first == second
    assert first["mean"] == 6.5
    assert first["ci_low"] <= first["mean"] <= first["ci_high"]


def test_applicable_transformed_removes_identity_and_categorical_noops():
    frame = pd.DataFrame(
        {
            "transform": ["identity", "categorical_bijection", "categorical_bijection", "asinh"],
            "transform_scope": ["numerical", "categorical", "categorical", "numerical"],
            "n_categorical": [0, 0, 2, 0],
        }
    )
    result = applicable_transformed(frame)
    assert result.index.tolist() == [2, 3]
