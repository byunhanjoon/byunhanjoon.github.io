from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("analyze_joint_cover", HERE / "analyze_joint_orthogonal_cover.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_design_is_balanced_and_cell_inclusion_uniform() -> None:
    designs = MODULE.orthogonal_designs(2)
    assert designs.shape == (3456, 4, 3)
    for factor, cardinality in ((0, 4), (1, 4), (2, 2)):
        counts = np.stack([(designs[:, :, factor] == level).sum(axis=1) for level in range(cardinality)], axis=1)
        assert np.all(counts == 4 // cardinality)
    inclusion = np.zeros((4, 4, 2, 4), dtype=int)
    for design in designs:
        for seed, (feature, category, label) in enumerate(design):
            inclusion[feature, category, label, seed] += 1
    assert np.unique(inclusion).size == 1


def test_main_effects_are_annihilated() -> None:
    designs = MODULE.orthogonal_designs(2)
    feature = np.arange(4)[:, None, None, None, None, None]
    category = 10 * np.arange(4)[None, :, None, None, None, None]
    label = 100 * np.arange(2)[None, None, :, None, None, None]
    seed = 1000 * np.arange(4)[None, None, None, :, None, None]
    predictions = np.broadcast_to(feature + category + label + seed, (4, 4, 2, 4, 3, 1)).astype(float)
    assert MODULE.exact_cover_residual(predictions, designs) < 1e-20


def test_singleton_class_design_count() -> None:
    designs = MODULE.orthogonal_designs(1)
    assert designs.shape == (576, 4, 3)


def test_singleton_category_designs_still_balance_other_factors() -> None:
    designs = MODULE.orthogonal_designs(2, 1)
    assert designs.shape == (144, 4, 3)
    assert np.all(designs[:, :, 1] == 0)
    assert np.all(np.sort(designs[:, :, 0], axis=1) == np.arange(4))
