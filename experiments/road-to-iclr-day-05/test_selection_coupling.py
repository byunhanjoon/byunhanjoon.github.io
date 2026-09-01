import numpy as np

import analyze_robust_model_selection as rms


def test_model_specific_seeds_produce_distinct_valid_actions():
    shape = (4, 4, 2, 4)
    first = rms.action_ids(shape, rms.stable_seed("a"))
    second = rms.action_ids(shape, rms.stable_seed("b"))
    assert all(first[name].shape == (rms.DRAWS, 16) for name in rms.METHODS)
    assert any(not np.array_equal(first[name], second[name]) for name in rms.METHODS)
