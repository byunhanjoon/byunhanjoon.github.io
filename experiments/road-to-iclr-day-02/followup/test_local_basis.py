"""Invariants for the information-equivalent local PLE construction."""

from __future__ import annotations

import unittest

import numpy as np

from local_basis_benchmark import energy_match, piecewise_bases


class LocalBasisTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(2026)
        self.parts = {
            "train": rng.normal(size=(1_000, 3)),
            "val": rng.normal(size=(200, 3)),
            "test": rng.normal(size=(300, 3)),
        }

    def test_equal_dimensions_and_local_support(self) -> None:
        cumulative, local, intervals = piecewise_bases(self.parts, 32)
        self.assertEqual(cumulative["train"].shape, local["train"].shape)
        start = 0
        for width in intervals:
            stop = start + width
            active = np.count_nonzero(local["test"][:, start:stop], axis=1)
            self.assertLessEqual(int(active.max()), 2)
            start = stop

    def test_bases_have_the_same_affine_span(self) -> None:
        cumulative, local, _ = piecewise_bases(self.parts, 16)
        for source, target in ((cumulative, local), (local, cumulative)):
            train = np.column_stack((np.ones(len(source["train"])), source["train"]))
            coefficients = np.linalg.lstsq(
                train,
                np.column_stack((np.ones(len(target["train"])), target["train"])),
                rcond=None,
            )[0]
            prediction = np.column_stack(
                (np.ones(len(source["test"])), source["test"])
            ) @ coefficients
            expected = np.column_stack(
                (np.ones(len(target["test"])), target["test"])
            )
            self.assertLess(float(np.max(np.abs(prediction - expected))), 1e-5)

    def test_energy_matching_uses_train_blocks(self) -> None:
        cumulative, local, intervals = piecewise_bases(self.parts, 16)
        matched, _ = energy_match(local, cumulative, intervals)
        start = 0
        for width in intervals:
            stop = start + width
            cumulative_energy = np.mean(
                np.sum(cumulative["train"][:, start:stop] ** 2, axis=1)
            )
            matched_energy = np.mean(
                np.sum(matched["train"][:, start:stop] ** 2, axis=1)
            )
            self.assertAlmostEqual(
                float(cumulative_energy), float(matched_energy), places=5
            )
            start = stop


if __name__ == "__main__":
    unittest.main()
