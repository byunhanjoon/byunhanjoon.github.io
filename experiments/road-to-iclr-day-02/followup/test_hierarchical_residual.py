from __future__ import annotations

import unittest

import numpy as np

from hierarchical_residual import (
    DiscoveryConfig,
    discover,
    fit_lookup,
    fit_pure_interaction,
)


class HierarchicalResidualTest(unittest.TestCase):
    def test_rejects_impossible_fold_win_threshold(self) -> None:
        numeric = np.arange(40, dtype=np.float64).reshape(20, 2)
        target = np.linspace(0.0, 1.0, 20)
        with self.assertRaisesRegex(ValueError, "minimum_fold_wins"):
            discover(
                numeric,
                numeric,
                target,
                "regression",
                0,
                DiscoveryConfig(folds=4, minimum_fold_wins=5),
            )

    def test_unseen_state_falls_back_to_zero(self) -> None:
        lookup = fit_lookup(
            np.array([0.0, 0.0, 1.0]),
            np.array([2.0, 4.0, 6.0]),
            smoothing=0.0,
        )
        prediction = lookup.predict(np.array([0.0, 1.0, 2.0]))
        np.testing.assert_allclose(prediction, [3.0, 6.0, 0.0])

    def test_pure_interaction_has_zero_weighted_marginals(self) -> None:
        values = np.array(
            [[left, right] for left in range(4) for right in range(3) for _ in range(5)],
            dtype=np.float64,
        )
        residual = (
            2.0 * (values[:, 0] == 1)
            - 3.0 * (values[:, 1] == 2)
            + 4.0 * ((values[:, 0] == 3) & (values[:, 1] == 0))
        )
        lookup = fit_pure_interaction(values, residual, 5.0, 10)
        prediction = lookup.predict(values)
        for column in (0, 1):
            for state in np.unique(values[:, column]):
                self.assertAlmostEqual(
                    float(prediction[values[:, column] == state].mean()),
                    0.0,
                    places=8,
                )

    def test_discovery_distinguishes_additive_and_interaction(self) -> None:
        rng = np.random.default_rng(7)
        rows = 6_000
        left = rng.integers(0, 8, rows)
        right = rng.integers(0, 7, rows)
        nuisance = rng.integers(0, 6, rows)
        continuous = rng.normal(size=rows)
        numeric = np.column_stack((left, right, nuisance, continuous)).astype(float)
        design = np.column_stack((numeric, continuous**2))
        smooth = 1.2 * continuous - 0.3 * continuous**2
        config = DiscoveryConfig(
            smoothing=10.0,
            max_cardinality=16,
            max_pair_cardinality=128,
            minimum_relative_gain=2e-3,
            minimum_pair_fold_wins=5,
            maximum_pairs=2,
        )

        additive = (
            smooth
            + 1.2 * (left == 3)
            - 0.9 * (right == 5)
            + rng.normal(scale=0.4, size=rows)
        )
        selection, _, _ = discover(
            design, numeric, additive, "regression", 0, config
        )
        self.assertEqual(set(selection.singletons), {0, 1})
        self.assertNotIn((0, 1), selection.pairs)

        left_centered = (left == 3).astype(float) - 1.0 / 8.0
        right_centered = (right == 5).astype(float) - 1.0 / 7.0
        interactive = (
            smooth
            + 8.0 * left_centered * right_centered
            + rng.normal(scale=0.4, size=rows)
        )
        selection, _, _ = discover(
            design, numeric, interactive, "regression", 0, config
        )
        self.assertIn((0, 1), selection.pairs)


if __name__ == "__main__":
    unittest.main()
