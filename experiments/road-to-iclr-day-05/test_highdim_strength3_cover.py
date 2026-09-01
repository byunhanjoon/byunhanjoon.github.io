import numpy as np

from analyze_strength2_cover import assert_strength
from highdim_strength3_cover import block_design, strength3_base


def test_mixed_oa_balances_all_three_factor_margins():
    for binary_count in (5, 11, 15, 28):
        base = strength3_base(binary_count)
        assert base.shape == (128, 2 + binary_count)
        assert_strength(base, (4, 4) + (2,) * binary_count, 3)


def test_all_methods_have_equal_budget():
    rng = np.random.default_rng(8)
    for method in ("strength3_oa128", "four_strength2_oa32", "four_marginal32", "iid128"):
        assert block_design(method, 15, rng).shape == (128, 17)
