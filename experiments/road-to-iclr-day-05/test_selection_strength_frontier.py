import numpy as np

from analyze_selection_strength_frontier import DRAWS, frontier_actions
from analyze_strength2_cover import assert_strength


def test_frontier_budgets_and_cover_strengths():
    shape = (4, 4, 2, 4)
    actions = frontier_actions(shape, 7)
    expected = {4: 3, 16: 4, 64: 5}
    for budget, count in expected.items():
        current = [values for values in actions.values() if values.shape == (DRAWS, budget)]
        assert len(current) == count
    for name, strength in (("strength1_b4", 1), ("strength2_b16", 2), ("strength3_b64", 3)):
        coordinates = np.column_stack(np.unravel_index(actions[name][0], shape))
        assert_strength(coordinates, shape, strength)
