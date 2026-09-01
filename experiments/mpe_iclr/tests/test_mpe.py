from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mpe  # noqa: E402


def tree_metric() -> np.ndarray:
    adjacency = np.zeros((31, 31), dtype=np.float64)
    for child in range(1, 31):
        parent = (child - 1) // 2
        adjacency[parent, child] = adjacency[child, parent] = 1.0
    return mpe.graph_shortest_path(adjacency)


@pytest.mark.parametrize(
    "distance",
    [mpe.path_distance(23), mpe.cycle_distance(24), tree_metric(), mpe.equality_distance(17)],
)
def test_metric_axioms(distance: np.ndarray) -> None:
    summary = mpe.validate_distance_matrix(distance)
    assert summary["passed"] is True
    assert summary["max_triangle_violation"] <= 1e-12


def test_farthest_landmarks_are_unique_training_states_only() -> None:
    distance = mpe.cycle_distance(32)
    training = np.asarray([state for state in range(32) if state % 4])
    landmarks = mpe.farthest_point_landmarks(distance, training, 32)
    assert len(landmarks) == len(training)
    assert len(np.unique(landmarks)) == len(landmarks)
    assert np.isin(landmarks, training).all()


@pytest.mark.parametrize("kernel", ["gaussian", "laplacian", "triangular", "inverse_distance"])
def test_partition_weights_sum_to_one(kernel: str) -> None:
    distance = mpe.path_distance(11)[:, [0, 5, 10]]
    bandwidth = 5.1 if kernel == "triangular" else 2.0
    weights = mpe.mpe_weights(distance, bandwidth, kernel=kernel)
    assert np.isfinite(weights).all()
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-12)


def test_sparse_partition_has_exact_support_limit() -> None:
    distance = mpe.path_distance(21)[:, np.arange(0, 21, 2)]
    weights = mpe.mpe_weights(distance, 2.0, sparse_k=4)
    assert np.max(np.sum(weights > 0, axis=1)) <= 4
    assert np.allclose(weights.sum(axis=1), 1.0)


@pytest.mark.parametrize("kernel", ["gaussian", "laplacian"])
def test_strictly_positive_kernel_partition_is_stable_at_extreme_distance(kernel: str) -> None:
    distance = np.asarray([[1.0e6, 1.0e6 + 1.0, 1.0e6 + 2.0]])
    weights = mpe.mpe_weights(distance, 1.0, kernel=kernel)
    assert np.isfinite(weights).all()
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert int(np.argmax(weights[0])) == 0


def test_exact_transported_relabeling_invariance() -> None:
    distance = tree_metric()
    landmarks = mpe.farthest_point_landmarks(distance, np.arange(24), 16)
    reference = mpe.state_weight_table(distance, landmarks, 2.0)
    for seed in range(32):
        new_of_old = mpe.codebook_permutation(len(distance), seed)
        transported = np.empty_like(distance)
        transported[np.ix_(new_of_old, new_of_old)] = distance
        transported_landmarks = new_of_old[landmarks]
        candidate = mpe.state_weight_table(transported, transported_landmarks, 2.0)
        aligned = candidate[new_of_old]
        assert np.array_equal(reference, aligned)


def test_equality_unseen_states_collapse() -> None:
    distance = mpe.equality_distance(20)
    landmarks = np.arange(8)
    weights = mpe.state_weight_table(distance, landmarks, 1.0)
    assert np.max(np.abs(weights[8:] - weights[8])) == 0.0


def test_partition_interpolation_bound() -> None:
    states = np.linspace(0.0, 1.0, 101)
    distance = np.abs(states[:, None] - states[None, :])
    landmarks = np.arange(0, 101, 10)
    weights = mpe.state_weight_table(distance, landmarks, 0.08)
    f = np.sin(states)
    estimate = weights @ f[landmarks]
    radius = mpe.weighted_metric_radius(weights, distance[:, landmarks])
    assert np.max(np.abs(estimate - f) - radius) <= 2e-15


def test_linear_head_rank_realizability() -> None:
    rng = np.random.default_rng(7)
    left = rng.normal(size=(12, 4))
    right = rng.normal(size=(4, 7))
    target = left @ right
    u, singular, vt = np.linalg.svd(target, full_matrices=False)
    root = np.sqrt(singular[:4])
    tokens = u[:, :4] * root[None, :]
    head = root[:, None] * vt[:4]
    assert np.max(np.abs(tokens @ head - target)) < 1e-12


def test_metric_perturbation_bound() -> None:
    rng = np.random.default_rng(9)
    distances = rng.uniform(0.0, 3.0, size=(40, 12))
    delta = 1e-4
    perturbed = np.maximum(0.0, distances + rng.uniform(-delta, delta, size=distances.shape))
    h = 0.7
    tokens = rng.normal(size=(12, 6))
    tokens /= np.maximum(1.0, np.linalg.norm(tokens, axis=1, keepdims=True))
    original_affinity = mpe.kernel_affinity(distances / h, "gaussian")
    z0 = float(np.min(original_affinity.sum(axis=1)))
    original = mpe.mpe_weights(distances, h) @ tokens
    changed = mpe.mpe_weights(perturbed, h) @ tokens
    lhs = np.linalg.norm(original - changed, axis=1)
    rhs = 2.0 * 1.0 * distances.shape[1] * np.exp(-0.5) * delta / (h * z0)
    assert float(np.max(lhs)) <= rhs + 1e-12


def test_triangular_interval_is_piecewise_linear() -> None:
    landmarks = np.linspace(0.0, 1.0, 11)
    queries = np.linspace(0.0, 1.0, 10001)
    gap = landmarks[1] - landmarks[0]
    distance = np.abs(queries[:, None] - landmarks[None, :])
    weights = mpe.mpe_weights(distance, gap * (1.0 + 1e-14), kernel="triangular")
    values = np.sin(landmarks * 3.0)
    expected = np.interp(queries, landmarks, values)
    assert np.max(np.abs(weights @ values - expected)) < 1e-12


def test_nystrom_is_finite_and_has_expected_dimension() -> None:
    distance = mpe.cycle_distance(32)
    landmarks = mpe.farthest_point_landmarks(distance, np.arange(24), 16)
    features, rank = mpe.nystrom_features(distance, np.arange(32), landmarks, 2.0)
    assert features.shape == (32, 16)
    assert np.isfinite(features).all()
    assert 1 <= rank <= 16


def test_corrupt_metric_preserves_distribution() -> None:
    distance = tree_metric()
    permutation = mpe.codebook_permutation(len(distance), 123)
    corrupt = mpe.corrupt_state_association(distance, permutation)
    assert np.array_equal(np.sort(distance.ravel()), np.sort(corrupt.ravel()))
    assert not np.array_equal(distance, corrupt)


def test_state_partition_is_disjoint_and_deterministic() -> None:
    states = [f"state-{index}" for index in range(60)]
    left = mpe.make_state_partition(states, 42)
    right = mpe.make_state_partition(states, 42)
    mpe.assert_disjoint_states(left)
    assert all(np.array_equal(left[key], right[key]) for key in left)


def test_state_balanced_metric_does_not_reduce_to_row_weighted() -> None:
    losses = np.asarray([0.0] * 100 + [10.0])
    states = np.asarray(["frequent"] * 100 + ["rare"])
    assert mpe.state_balanced_mean(losses, states) == 5.0
    assert float(np.mean(losses)) < 0.1


def test_string_jaccard_distance_is_metric() -> None:
    values = ["chief executive", "chief exec", "nurse", "registered nurse", "data scientist"]
    distance = mpe.trigram_jaccard_distance(values)
    mpe.validate_distance_matrix(distance)


def test_zero_denominator_fails_loudly() -> None:
    with pytest.raises(FloatingPointError):
        mpe.mpe_weights(np.asarray([[2.0, 3.0]]), 1.0, kernel="triangular")
