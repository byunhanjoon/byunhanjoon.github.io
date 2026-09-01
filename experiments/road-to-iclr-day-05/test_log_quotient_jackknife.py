import itertools

import numpy as np

import analyze_log_quotient_jackknife as LQJ
import analyze_log_jackknife_frontier as LJF


def test_batched_log_loss_matches_scalar_calls() -> None:
    y = np.array([0, 1])
    predictions = np.array([
        [[.8, .2], [.1, .9]],
        [[.6, .4], [.3, .7]],
    ])
    batched = LQJ.log_loss(y, predictions)
    scalar = np.array([LQJ.log_loss(y, prediction) for prediction in predictions])
    np.testing.assert_allclose(batched, scalar)


def test_two_block_jackknife_reduces_symmetric_log_bias() -> None:
    y = np.array([1])
    members = np.array([[[.4, .6]], [[.2, .8]]])
    pairs = np.array(list(itertools.product(range(2), repeat=2)))
    left, right = pairs[:, :1], pairs[:, 1:]
    ordinary, jackknife = LQJ.block_scores(y, members, left, right)
    exact = float(LQJ.log_loss(y, members.mean(axis=0)))
    assert abs(jackknife.mean() - exact) < abs(ordinary.mean() - exact)


def test_four_block_jackknife_reduces_symmetric_log_bias() -> None:
    y = np.array([1])
    members = np.array([[[.4, .6]], [[.2, .8]]])
    combinations = np.array(list(itertools.product(range(2), repeat=4)))
    block_ids = combinations[:, :, None]
    jackknife = LJF.multiblock_scores(y, members, block_ids)
    full_predictions = members[combinations].mean(axis=1)
    ordinary = LQJ.log_loss(y, full_predictions)
    exact = float(LQJ.log_loss(y, members.mean(axis=0)))
    assert abs(jackknife.mean() - exact) < abs(ordinary.mean() - exact)
