import numpy as np

from experiments.day3.core import PARTS, geometry, whiten


def test_whitening_uses_training_fit_and_unit_covariance():
    rng = np.random.default_rng(1)
    train = rng.normal(size=(500, 7)) @ np.diag(np.geomspace(1, 100, 7))
    parts = {"train": train, "val": rng.normal(size=(50, 7)), "test": rng.normal(size=(60, 7))}
    transformed, meta = whiten(parts)
    covariance = transformed["train"].T @ transformed["train"] / len(train)
    assert meta["retained_rank"] == 7
    assert np.allclose(covariance, np.eye(7), atol=1e-9)
    assert geometry(transformed["train"])["condition_number"] < 1.000001
    assert set(transformed) == set(PARTS)
