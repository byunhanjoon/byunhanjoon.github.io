import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core import FM, _split_indices, membership, sample_context
from src.selectors import (
    aggregate_pair_feature,
    complementarity_matrix,
    dpp_logdet,
    kcenter,
    one_swap,
    pairwise_greedy,
    pairwise_objective,
)


def test_split_is_sized_disjoint_and_class_stratified():
    y = np.tile(np.arange(2), 1000)
    candidate, selector, test = _split_indices(y, "classification", seed=0)
    assert (len(candidate), len(selector), len(test)) == (256, 128, 256)
    assert len(set(candidate) | set(selector) | set(test)) == 640
    for split in (candidate, selector, test):
        assert set(y[split]) == {0, 1}


def test_regression_split_uses_quantile_strata_without_overlap():
    y = np.linspace(-1, 1, 2000)
    candidate, selector, test = _split_indices(y, "regression", seed=0)
    assert len(set(candidate) & set(selector)) == 0
    assert len(set(candidate) & set(test)) == 0
    assert len(set(selector) & set(test)) == 0


def test_context_sampling_is_deterministic_and_covers_classes():
    y = np.repeat(np.arange(4), 64)
    first = sample_context(y, 16, np.random.default_rng(7), "classification")
    second = sample_context(y, 16, np.random.default_rng(7), "classification")
    assert np.array_equal(first, second)
    assert len(first) == len(np.unique(first)) == 16
    assert set(y[first]) == {0, 1, 2, 3}
    assert membership(first).sum() == 16


def test_fm_matches_explicit_pair_sum():
    model = FM(5, 3)
    with torch.no_grad():
        model.bias.fill_(0.2)
        model.linear.copy_(torch.arange(5, dtype=torch.float32) / 10)
        model.v.copy_(torch.arange(15, dtype=torch.float32).reshape(5, 3) / 20)
    x = torch.tensor([[1, 0, 1, 1, 0]], dtype=torch.float32)
    selected = np.array([0, 2, 3])
    expected = 0.2 + model.linear[selected].sum().item()
    for pos, i in enumerate(selected):
        for j in selected[pos + 1 :]:
            expected += float(model.v[i] @ model.v[j])
    assert np.isclose(float(model(x)), expected, atol=1e-6)


def test_pair_feature_matches_upper_triangle_sum():
    pair = np.arange(25, dtype=float).reshape(5, 5)
    pair = (pair + pair.T) / 2
    np.fill_diagonal(pair, 0)
    x = membership([0, 2, 4], 5)[None, :]
    expected = np.triu(pair[np.ix_([0, 2, 4], [0, 2, 4])], 1).sum()
    assert np.isclose(aggregate_pair_feature(x, pair)[0], expected)


def test_pairwise_greedy_and_swap_respect_budget_and_coverage():
    y = np.repeat([0, 1], 6)
    additive = np.linspace(0, 1, 12)
    pair = np.zeros((12, 12))
    pair[0, 11] = pair[11, 0] = 10
    initial = pairwise_greedy(additive, pair, 4, y, True)
    improved = one_swap(initial, additive, pair, y, True)
    assert len(initial) == len(improved) == 4
    assert set(y[initial]) == set(y[improved]) == {0, 1}
    assert pairwise_objective(improved, additive, pair) >= pairwise_objective(initial, additive, pair)


def test_geometry_selectors_return_unique_budget_and_coverage():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(32, 6))
    y = np.repeat([0, 1], 16)
    for selected in (kcenter(z, 8, y, True), dpp_logdet(z, 8, y, True)):
        assert len(selected) == len(np.unique(selected)) == 8
        assert set(y[selected]) == {0, 1}


def test_complementarity_rewards_different_binary_classes():
    bins = np.array([0, 0, 1, 1])
    matrix = complementarity_matrix(bins)
    assert matrix[0, 2] > matrix[0, 1]
    assert np.allclose(np.diag(matrix), 0)
