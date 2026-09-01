import numpy as np

import analyze_cross_quotient_selection as CQS
import analyze_repartitioned_cross_selection as RCS


def test_validation_action_scores_match_whole_sample_scores() -> None:
    rng = np.random.default_rng(11)
    y = np.array([0, 1, 1, 0, 1])
    raw = rng.uniform(size=(20, len(y), 2))
    flat = raw / raw.sum(axis=-1, keepdims=True)
    left = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    right = np.array([[8, 9, 10, 11], [12, 13, 14, 15]])
    masks = np.ones((2, len(y)), dtype=bool)
    expected_cross, _ = CQS.cross_and_mean_scores(y, flat, left, right)
    actual_cross = RCS.validation_action_scores(y, flat, masks, left, right)
    np.testing.assert_allclose(actual_cross, expected_cross)
    iid = np.concatenate((left, right), axis=1)
    expected_u = CQS.iid_u_scores(y, flat, iid)
    actual_u = RCS.validation_action_scores(y, flat, masks, iid)
    np.testing.assert_allclose(actual_u, expected_u)


def test_complement_loss_identity() -> None:
    pooled = np.array([[1.0, 2.0], [3.0, 0.0], [2.0, 4.0]])
    validation = pooled[:1].mean(axis=0)
    test = pooled[1:].mean(axis=0)
    pooled_mean = pooled.mean(axis=0)
    reconstructed = 3 / 2 * pooled_mean - 1 / 2 * validation
    np.testing.assert_allclose(test, reconstructed)
