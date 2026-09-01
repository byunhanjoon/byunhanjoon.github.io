from __future__ import annotations

import numpy as np


def _probabilities(p: np.ndarray) -> np.ndarray:
    values = np.asarray(p, dtype=np.float64)
    if values.ndim != 2 or np.any(values < 0):
        raise ValueError("probabilities must be a nonnegative rows-by-classes matrix")
    sums = values.sum(axis=1, keepdims=True)
    if np.any(sums <= 0):
        raise ValueError("probability rows must have positive mass")
    return values / sums


def total_variation(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p, q = _probabilities(p), _probabilities(q)
    if p.shape != q.shape:
        raise ValueError("shape mismatch")
    return 0.5 * np.abs(p - q).sum(axis=1)


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p, q = _probabilities(p), _probabilities(q)
    if p.shape != q.shape:
        raise ValueError("shape mismatch")
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        safe_a = np.clip(a, 1e-300, None)
        safe_b = np.clip(b, 1e-300, None)
        terms = np.where(a > 0, a * (np.log(safe_a) - np.log(safe_b)), 0.0)
        return terms.sum(axis=1)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    probs = _probabilities(p)
    labels = np.asarray(y, dtype=int)
    if labels.shape != (probs.shape[0],):
        raise ValueError("label shape mismatch")
    one_hot = np.eye(probs.shape[1])[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
