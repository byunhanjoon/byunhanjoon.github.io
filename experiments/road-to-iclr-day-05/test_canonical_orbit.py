from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("canonical_orbit", HERE / "canonical_orbit.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_category_and_feature_permutations_canonicalize_identically() -> None:
    train = np.asarray([[3.0, 0, 9.0], [1.0, 1, 8.0], [2.0, 0, 7.0], [4.0, 2, 6.0]])
    query = np.asarray([[5.0, 2, 5.0], [6.0, 1, 4.0]])
    base = MODULE.canonicalize_tables(train, [query], (1,))
    feature = np.asarray([2, 0, 1])
    category_map = np.asarray([2, 0, 1])
    changed_train = train.copy()
    changed_query = query.copy()
    changed_train[:, 1] = category_map[changed_train[:, 1].astype(int)]
    changed_query[:, 1] = category_map[changed_query[:, 1].astype(int)]
    changed = MODULE.canonicalize_tables(changed_train[:, feature], [changed_query[:, feature]], (2,))
    assert np.array_equal(base[0], changed[0])
    assert np.array_equal(base[1][0], changed[1][0])
    assert base[2] == changed[2]


def test_target_permutation_canonicalizes_identically() -> None:
    y = np.asarray([0, 1, 0, 0, 1, 1, 0])
    first, mapping = MODULE.canonicalize_target(y)
    second, swapped_mapping = MODULE.canonicalize_target(1 - y)
    assert np.array_equal(first, second)
    assert np.array_equal(mapping, swapped_mapping[::-1])


def test_duplicate_fields_are_harmless() -> None:
    train = np.asarray([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    first = MODULE.canonicalize_tables(train, [train], ())
    second = MODULE.canonicalize_tables(train[:, ::-1], [train[:, ::-1]], ())
    assert np.array_equal(first[0], second[0])

