import itertools
import numpy as np


def test_finite_population_formula_exhaustively():
    values = np.array([-1.5, -0.5, 0.5, 1.5])
    for budget in (1, 2, 3):
        means = np.array([values[list(index)].mean() for index in itertools.combinations(range(4), budget)])
        expected = np.mean(values**2) / budget * (4 - budget) / 3
        assert np.isclose(np.mean(means**2), expected)
