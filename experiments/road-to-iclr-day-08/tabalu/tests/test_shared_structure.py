from __future__ import annotations

import numpy as np

from tabalu.models.executor import ExecutableProgram, ProgramNode
from tabalu.models.shared_structure import fit_regime_coefficients
from tabalu.synthetic import generate_temporal_task, sample_temporal_split


def test_regime_coefficients_recover_known_affine_changes() -> None:
    program = ExecutableProgram(2, [ProgramNode("multiply", 0, 1)], 2)
    rng = np.random.default_rng(4)
    features = rng.normal(size=(200, 2)).astype(np.float32)
    regimes = rng.integers(0, 2, size=200)
    base = program(features)
    targets = np.where(regimes == 0, 2.0 * base + 1.0, -0.5 * base + 3.0)
    fitted = fit_regime_coefficients(program, features, targets, regimes, 2)
    np.testing.assert_allclose(fitted(features, regimes), targets, rtol=1e-6, atol=1e-6)


def test_future_temporal_split_contains_only_post_change_regime() -> None:
    task = generate_temporal_task(77, change_point=0.7)
    _, time, regimes, _ = sample_temporal_split(
        task, "future_test", 128, seed=0, magnitude_multiplier=8
    )
    assert np.all(time >= 0.8)
    assert np.all(regimes == 1)
