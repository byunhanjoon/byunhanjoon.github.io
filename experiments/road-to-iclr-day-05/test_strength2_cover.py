from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("strength2", HERE / "analyze_strength2_cover.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_mixed_base_has_strength_two() -> None:
    for category in (1, 2, 4):
        for label in (1, 2, 4):
            for seed in (1, 2, 4):
                base = MODULE.strength2_base(category, label, seed)
                MODULE.assert_strength(base, (4, category, label, seed), 2)


def test_strength2_annihilates_main_and_pairwise_fields() -> None:
    cardinalities = (4, 4, 2, 4)
    family = MODULE.strength2_family(4, 2)
    covariance = MODULE.incidence_covariance(family, cardinalities)
    indices = np.indices(cardinalities)
    values = (
        indices[0] + 2 * indices[1] + 3 * indices[2] + 4 * indices[3]
        + indices[0] * indices[1] + indices[2] * indices[3]
    )[..., None, None].astype(float)
    assert abs(MODULE.expected_residual(values, covariance)) < 1e-12


def test_incidence_estimator_is_unbiased() -> None:
    family = MODULE.strength2_family(4, 2)
    cardinalities = (4, 4, 2, 4)
    incidence = np.zeros((len(family), np.prod(cardinalities)))
    for i, design in enumerate(family):
        ids = np.ravel_multi_index(design.T, cardinalities)
        incidence[i, ids] += 1 / len(design)
    assert np.max(np.abs(incidence.mean(axis=0) - 1 / np.prod(cardinalities))) < 1e-14


def test_strength2_component_coefficients_kill_order_two() -> None:
    cardinalities = (4, 4, 2, 4)
    family = MODULE.strength2_family(4, 2)
    covariance = MODULE.incidence_covariance(family, cardinalities)
    coefficients = MODULE.component_coefficients(covariance, cardinalities)
    for name, coefficient in coefficients.items():
        if name.count(":") <= 1:
            assert abs(coefficient) < 1e-12
    assert any(coefficient > 0 for name, coefficient in coefficients.items() if name.count(":") >= 2)


def test_binary_seed_families_are_balanced() -> None:
    family1 = MODULE.strength1_family(4, 2, 2)
    family2 = MODULE.strength2_family(4, 2, 2)
    assert family1.shape == (20736, 4, 4)
    assert family2.shape == (2304, 16, 4)
    for family in (family1, family2):
        for factor, levels in enumerate((4, 4, 2, 2)):
            counts = np.stack([(family[:, :, factor] == level).sum(axis=1) for level in range(levels)], axis=1)
            assert np.all(counts == family.shape[1] // levels)


def test_binary_category_families_are_balanced() -> None:
    family1 = MODULE.strength1_family(2, 2, 4)
    family2 = MODULE.strength2_family(2, 2, 4)
    for design in family1[:10]:
        MODULE.assert_strength(design, (4, 2, 2, 4), 1)
    for design in family2[:10]:
        MODULE.assert_strength(design, (4, 2, 2, 4), 2)


def test_four_level_target_family_is_pairwise_balanced() -> None:
    family = MODULE.strength2_family(1, 4, 4)
    for design in family[:20]:
        MODULE.assert_strength(design, (4, 1, 4, 4), 2)
