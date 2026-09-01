import numpy as np
import pytest

from src.methods import (
    EXPERTS,
    competence_weights,
    cross_validated_expert_losses,
    fit_predict_experts,
    weighted_prediction,
)


def test_competence_weights_are_normalized_and_shrink_to_uniform():
    losses = np.arange(len(EXPERTS), dtype=float)
    sharp = competence_weights(losses, temperature=0.1, uniform_shrinkage=0.0)
    uniform = competence_weights(losses, temperature=0.1, uniform_shrinkage=1.0)
    assert sharp.argmax() == 0
    assert sharp.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(uniform, np.full(len(EXPERTS), 1 / len(EXPERTS)))


def test_competence_rejects_invalid_hyperparameters():
    with pytest.raises(ValueError):
        competence_weights(np.zeros(len(EXPERTS)), 0.0, 0.0)
    with pytest.raises(ValueError):
        competence_weights(np.zeros(len(EXPERTS)), 1.0, 1.1)


def test_context_cv_and_full_predictions_are_deterministic_and_finite():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(48, 4))
    y = x[:, 0] - 0.5 * x[:, 1] + rng.normal(0, 0.1, 48)
    query = rng.normal(size=(12, 4))
    first = cross_validated_expert_losses(x, y, "regression", seed=9, folds=3)
    second = cross_validated_expert_losses(x, y, "regression", seed=9, folds=3)
    np.testing.assert_allclose(first, second)
    assert first.shape == (len(EXPERTS),)
    prediction = fit_predict_experts(x, y, query, "regression", seed=9)
    assert prediction.shape == (len(EXPERTS), len(query))
    assert np.all(np.isfinite(prediction))


def test_weighted_prediction_uses_only_declared_weights():
    predictions = np.arange(len(EXPERTS) * 3, dtype=float).reshape(len(EXPERTS), 3)
    weights = np.zeros(len(EXPERTS))
    weights[2] = 1.0
    np.testing.assert_array_equal(weighted_prediction(predictions, weights), predictions[2])
