from __future__ import annotations

import numpy as np

from tabalu.synthetic.regime_scaling import generate_multi_regime_task, sample_multi_regime_split


def test_multi_regime_splits_cover_and_shift_regimes() -> None:
    task = generate_multi_regime_task(41, 4)
    _, context, regimes, targets = sample_multi_regime_split(task, "train", 400, seed=0)
    np.testing.assert_array_equal(np.bincount(regimes), [100, 100, 100, 100])
    assert context.shape == (400, 4)
    assert np.isfinite(targets).all()
    _, _, shifted, _ = sample_multi_regime_split(task, "ood", 4000, seed=0, magnitude_multiplier=4)
    assert np.mean(shifted == 3) > 0.45
