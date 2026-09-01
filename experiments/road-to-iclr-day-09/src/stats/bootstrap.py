from __future__ import annotations

import numpy as np


def mean_interval(values: np.ndarray, draws: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or x.size < 2 or not np.all(np.isfinite(x)):
        raise ValueError("values must be a finite vector with at least two entries")
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    chunk = min(draws, 1000)
    for start in range(0, draws, chunk):
        stop = min(start + chunk, draws)
        index = rng.integers(0, x.size, size=(stop - start, x.size))
        means[start:stop] = x[index].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(x.mean()), float(low), float(high)

