import numpy as np

from experiments.day3.core import PARTS, residualize


def test_block_residualization_is_exact_and_orthogonal_on_train():
    rng = np.random.default_rng(3)
    n = rng.normal(size=(400, 6))
    c = n @ rng.normal(size=(6, 5)) + rng.normal(size=(400, 5))
    numeric = {p: n.copy() for p in PARTS}
    categorical = {p: c.copy() for p in PARTS}
    perpendicular, beta = residualize(numeric, categorical)
    assert np.linalg.norm(n.T @ perpendicular["train"]) / (np.linalg.norm(n) * np.linalg.norm(perpendicular["train"])) < 1e-12
    assert np.allclose(c, perpendicular["train"] + n @ beta, atol=1e-10)
