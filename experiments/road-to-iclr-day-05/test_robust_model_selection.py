import numpy as np

from analyze_robust_model_selection import DRAWS, action_ids, batched_losses
from analyze_strength2_cover import assert_strength


def test_actions_have_equal_budget_and_strength2_balances_pairs():
    shape = (4, 4, 2, 4)
    actions = action_ids(shape, 17)
    assert set(actions) == {
        "strength2", "iid16", "srswor16", "four_strength1", "four_seed_blocks",
        "sobol16", "lhs16",
    }
    assert all(values.shape == (DRAWS, 16) for values in actions.values())
    assert all(len(np.unique(row)) == 16 for row in actions["srswor16"])
    coordinates = np.column_stack(np.unravel_index(actions["strength2"][0], shape))
    assert_strength(coordinates, shape, 2)


def test_batched_loss_matches_direct_binary_brier():
    rng = np.random.default_rng(4)
    raw = rng.random((128, 7, 2))
    raw /= raw.sum(axis=-1, keepdims=True)
    y = rng.integers(0, 2, size=7)
    ids = rng.integers(0, 128, size=(DRAWS, 16))
    actual = batched_losses(y, raw, ids)
    predictions = raw[ids].mean(axis=1)
    expected = np.mean(np.sum((predictions - np.eye(2)[y][None]) ** 2, axis=-1), axis=1)
    np.testing.assert_allclose(actual, expected)
