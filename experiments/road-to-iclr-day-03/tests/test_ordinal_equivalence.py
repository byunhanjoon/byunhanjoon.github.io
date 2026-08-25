import numpy as np

from experiments.day3.core import contrast_block, cumulative_ordinal, equivalence_diagnostics


def test_local_and_cumulative_ordinal_are_affine_equivalent():
    codes = np.tile(np.arange(9), 30)
    parts = {"train": codes, "val": codes[:50], "test": codes[50:100]}
    local = contrast_block(parts, 9)
    cumulative = cumulative_ordinal(parts, 9)
    diagnostics = equivalence_diagnostics(local, cumulative)
    assert diagnostics["rank_a"] == diagnostics["rank_b"] == 8
    assert diagnostics["a_to_b"]["train"] < 1e-12
    assert diagnostics["b_to_a"]["test"] < 1e-12
    assert diagnostics["max_principal_angle_deg"] < 1e-8
