import importlib.util
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("highdim_bootstrap_tested", HERE / "analyze_highdim_uncertainty.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_gram_bootstrap_matches_direct_sample_variance_and_brier():
    rng = np.random.default_rng(3)
    raw = rng.random((5, 7, 2))
    predictions = raw / raw.sum(axis=-1, keepdims=True)
    y = rng.integers(0, 2, size=7)
    counts = np.asarray([[1, 0, 2, 1, 1], [0, 3, 0, 2, 0]])
    risk, brier = MODULE.bootstrap_method(y, predictions, counts)
    for draw, count in enumerate(counts):
        sample = predictions[np.repeat(np.arange(5), count)]
        expected_risk = np.mean(np.sum((sample - sample.mean(axis=0)) ** 2, axis=-1)) * 5 / 4
        targets = np.eye(2)[y]
        expected_brier = np.mean(np.sum((sample - targets[None, ...]) ** 2, axis=-1))
        assert np.isclose(risk[draw], expected_risk)
        assert np.isclose(brier[draw], expected_brier)
