import numpy as np

from src.methods import (
    context_gate_descriptor,
    episode_loss,
    featurewise_pooled_gate_descriptor,
    mixture_loss_curve,
)


def test_gate_descriptor_uses_fixed_context_shape():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(32, 4))
    y = (x[:, 0] > 0).astype(int)
    first = context_gate_descriptor(x, y)
    second = context_gate_descriptor(x[::-1], y[::-1])
    assert first.ndim == 1
    assert np.all(np.isfinite(first))
    np.testing.assert_allclose(first, second, atol=1e-12)


def test_featurewise_descriptor_is_row_and_column_permutation_invariant():
    rng = np.random.default_rng(13)
    x = rng.normal(size=(40, 5)) * np.arange(1, 6)
    y = x[:, 1] + rng.normal(scale=0.1, size=40)
    reference = featurewise_pooled_gate_descriptor(x, y)
    row_order = rng.permutation(len(x))
    column_order = rng.permutation(x.shape[1])
    permuted = featurewise_pooled_gate_descriptor(x[row_order][:, column_order], y[row_order])
    np.testing.assert_allclose(reference, permuted, atol=1e-10)


def test_mixture_curve_endpoints_match_expert_losses():
    y = np.array([0, 1, 1, 0])
    raw = np.array([0.2, 0.8, 0.7, 0.1])
    rank = np.array([0.4, 0.6, 0.55, 0.3])
    curve = mixture_loss_curve(y, raw, rank, "classification", np.array([0.0, 1.0]), 1e-6)
    assert np.isclose(curve[0], episode_loss(y, rank, "classification", 1e-6))
    assert np.isclose(curve[1], episode_loss(y, raw, "classification", 1e-6))
