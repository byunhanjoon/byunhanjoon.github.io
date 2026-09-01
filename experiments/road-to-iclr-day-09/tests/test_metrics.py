import numpy as np

from src.metrics import brier_score, jensen_shannon, total_variation


def test_probability_metrics_known_cases():
    p = np.array([[1.0, 0.0], [0.5, 0.5]])
    q = np.array([[0.0, 1.0], [0.5, 0.5]])
    np.testing.assert_allclose(total_variation(p, q), [1.0, 0.0])
    np.testing.assert_allclose(jensen_shannon(p, q), [np.log(2.0), 0.0])
    assert brier_score(np.array([0, 1]), p) == 0.25

