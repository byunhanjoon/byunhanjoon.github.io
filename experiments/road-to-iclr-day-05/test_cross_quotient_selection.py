import numpy as np

from analyze_cross_quotient_selection import cross_and_mean_scores, iid_u_scores
from analyze_cross_score_budget_frontier import cover_block_scores


def test_cross_score_exactly_averages_to_quotient_loss_over_independent_population_draws():
    y = np.asarray([0.25, -0.5])
    flat = np.asarray([[[0.0], [-1.0]], [[0.5], [0.0]]])
    ids = np.asarray([[0], [1]])
    left = np.repeat(ids, 2, axis=0)
    right = np.tile(ids, (2, 1))
    cross, _ = cross_and_mean_scores(y, flat, left, right)
    quotient = flat.mean(axis=0)
    expected = np.mean((y - quotient[:, 0]) ** 2)
    assert np.isclose(cross.mean(), expected)


def test_iid_u_statistic_exactly_averages_to_quotient_loss():
    y = np.asarray([0.25, -0.5])
    flat = np.asarray([[[0.0], [-1.0]], [[0.5], [0.0]]])
    # Average over every ordered IID sample, including repeated population cells.
    ids = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])
    scores = iid_u_scores(y, flat, ids)
    quotient = flat.mean(axis=0)
    expected = np.mean((y - quotient[:, 0]) ** 2)
    assert np.isclose(scores.mean(), expected)


def test_independent_cross_score_variance_identity():
    quotient_residual = 0.3
    errors = np.asarray([-1.0, 1.0])
    scores = np.asarray([
        (quotient_residual - left) * (quotient_residual - right)
        for left in errors for right in errors
    ])
    covariance = np.mean(errors ** 2)
    expected_variance = 2 * quotient_residual ** 2 * covariance + covariance ** 2
    assert np.isclose(scores.mean(), quotient_residual ** 2)
    assert np.isclose(np.mean((scores - scores.mean()) ** 2), expected_variance)


def test_four_block_u_score_is_unbiased_over_independent_blocks():
    y = np.asarray([0.25, -0.5])
    flat = np.asarray([[[0.0], [-1.0]], [[0.5], [0.0]]])
    combinations = np.asarray([
        [(mask >> bit) & 1 for bit in range(4)] for mask in range(16)
    ])
    block_ids = combinations[:, :, None]
    scores, _ = cover_block_scores(y, flat, block_ids)
    quotient = flat.mean(axis=0)
    expected = np.mean((y - quotient[:, 0]) ** 2)
    assert np.isclose(scores.mean(), expected)
