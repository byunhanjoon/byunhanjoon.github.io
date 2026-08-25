import numpy as np

from experiments.day3.core import condition_transform, invariant_penalty, transformed_invariant_penalty


def test_function_space_regularizer_is_basis_invariant():
    rng = np.random.default_rng(4)
    z = rng.normal(size=(1000, 9))
    covariance = z.T @ z / len(z)
    weight = rng.normal(size=(13, 9))
    transform = condition_transform(9, 300, 12)
    left = invariant_penalty(weight, covariance)
    right = transformed_invariant_penalty(weight, covariance, transform)
    assert np.isclose(left, right, rtol=1e-10, atol=1e-10)
