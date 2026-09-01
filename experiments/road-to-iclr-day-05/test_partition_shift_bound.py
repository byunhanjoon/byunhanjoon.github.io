"""Independent checks for Proposition 34's finite selection inequality."""

from __future__ import annotations

import numpy as np


def test_partition_shift_bound_random_finite_candidate_sets() -> None:
    rng = np.random.default_rng(2026082934)
    for _ in range(10_000):
        candidates = int(rng.integers(2, 20))
        validation = rng.normal(size=candidates)
        estimate = validation + rng.uniform(-.2, .2, size=candidates)
        test = validation + rng.uniform(-.3, .3, size=candidates)
        epsilon = float(np.max(np.abs(estimate - validation)))
        delta = float(np.max(np.abs(test - validation)))
        selected = int(np.argmin(estimate))
        test_best = int(np.argmin(test))
        assert test[selected] - test[test_best] <= 2 * epsilon + 2 * delta + 1e-12


def test_partition_shift_bound_constant_is_tight_with_tie_breaking() -> None:
    epsilon, delta = .4, .7
    validation = np.asarray([2 * epsilon, 0.0])
    estimate = np.asarray([epsilon, epsilon])
    test = np.asarray([2 * epsilon + delta, -delta])
    selected = int(np.argmin(estimate))
    test_best = int(np.argmin(test))
    regret = float(test[selected] - test[test_best])
    assert regret == 2 * epsilon + 2 * delta
