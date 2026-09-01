import importlib.util
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("highdim_row_tested", HERE / "highdim_row_cover.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_marginal_design_is_balanced_but_generally_not_pairwise():
    design = MODULE.marginal_design(12, np.random.default_rng(91))
    assert design.shape == (32, 14)
    assert np.array_equal(np.bincount(design[:, 0]), np.repeat(8, 4))
    assert np.array_equal(np.bincount(design[:, 1]), np.repeat(8, 4))
    for column in range(2, design.shape[1]):
        assert np.array_equal(np.bincount(design[:, column]), np.repeat(16, 2))
    first_pair = np.zeros((4, 4), dtype=int)
    for row in design:
        first_pair[row[0], row[1]] += 1
    assert np.unique(first_pair).size > 1


def test_oa_accepts_row_plus_german_field_count():
    # class + row order + 13 German categorical fields
    design = MODULE.FIELD.base_oa(15)
    MODULE.FIELD.assert_pairwise_balance(design, (4, 4) + (2,) * 15)

