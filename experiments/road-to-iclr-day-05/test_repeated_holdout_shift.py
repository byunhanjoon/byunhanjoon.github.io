import numpy as np

import analyze_repeated_holdout_shift as RHS


def test_row_losses_match_brier_and_mse() -> None:
    y_binary = np.array([0, 1])
    prediction = np.array([[.75, .25], [.1, .9]])
    np.testing.assert_allclose(RHS.row_losses(y_binary, prediction), [.125, .02])
    y_regression = np.array([1.0, -1.0])
    regression = np.array([[1.5], [-2.0]])
    np.testing.assert_allclose(RHS.row_losses(y_regression, regression), [.25, 1.0])


def test_split_metrics_decomposes_validation_winner_floor() -> None:
    validation = np.array([[0.0, 1.0], [0.0, 1.0]])
    test = np.array([[2.0, 0.0], [2.0, 0.0]])
    metrics = RHS.split_metrics(validation, test)
    assert metrics["validation_winner"] == 0
    assert metrics["test_winner"] == 1
    assert not metrics["winner_agreement"]
    assert metrics["target_shift_floor"] == 2.0
    assert metrics["validation_margin"] == 1.0
    np.testing.assert_allclose(metrics["rank_correlation"], -1.0)


def test_stratified_indices_preserve_validation_counts() -> None:
    pooled = np.array([0] * 12 + [1] * 8)
    validation = np.array([0] * 5 + [1] * 3)
    selected = RHS.stratified_indices(pooled, validation, np.random.default_rng(7))
    assert len(np.unique(selected)) == len(selected)
    np.testing.assert_array_equal(np.unique(pooled[selected], return_counts=True)[1], [5, 3])
