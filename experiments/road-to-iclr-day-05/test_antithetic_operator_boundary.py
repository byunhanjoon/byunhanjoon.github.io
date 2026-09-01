"""Independent algebra checks for Proposition 35."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def test_finite_population_pair_operator_and_pack_mean() -> None:
    rng = np.random.default_rng(3501)
    population = rng.normal(size=(9, 4))
    population -= population.mean(axis=0)
    sigma = population.T @ population / len(population)
    pairs = np.asarray(list(itertools.combinations(range(len(population)), 2)))
    cross = sum(
        np.outer(population[i], population[j]) + np.outer(population[j], population[i])
        for i, j in pairs
    ) / (len(population) * (len(population) - 1))
    assert np.allclose(cross, -sigma / (len(population) - 1), atol=1e-14)
    for k in (2, 4, 8, 9):
        packs = np.asarray(list(itertools.combinations(range(len(population)), k)))
        means = population[packs].mean(axis=1)
        observed = np.mean(np.sum(means ** 2, axis=1))
        independent = np.mean(np.sum(population ** 2, axis=1)) / k
        assert np.isclose(observed / independent, (len(population) - k) / 8)


def test_exchangeable_zero_sum_tuple_has_maximal_k_antithesis() -> None:
    rng = np.random.default_rng(3502)
    for k in (2, 4, 7):
        values = rng.normal(size=(k, 5))
        values -= values.mean(axis=0)
        sigma = values.T @ values / k
        cross = sum(
            np.outer(values[i], values[j])
            for i in range(k) for j in range(k) if i != j
        ) / (k * (k - 1))
        assert np.allclose(cross, -sigma / (k - 1), atol=1e-14)


def test_controlled_antithetic_boundary_artifact_passes() -> None:
    result = json.loads(
        (HERE / "results/antithetic_operator_boundary_summary.json").read_text()
    )
    assert result["finite_exact_identity_passed"]
    assert result["gaussian_calibration_passed"]
    assert result["same_pair_coefficient_only_at_full_resolution"]
