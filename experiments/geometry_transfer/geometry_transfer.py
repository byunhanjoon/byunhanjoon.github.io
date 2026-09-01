"""Core operators and exact risk arithmetic for the Geometry Transfer Law.

The module is intentionally small: every predictor considered here is a fixed
linear map from observed-state residual means to held-out-state predictions.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.linalg import solve


EPS = 1e-12


@dataclass(frozen=True)
class Decomposition:
    possible_signal: float
    approximation_error: float
    transferable_signal: float
    noise_cost: float
    delta: float
    gtr: float


def weighted_norm2(x: np.ndarray, q: np.ndarray) -> float:
    """Return x'Qx for diagonal Q represented by its diagonal."""
    x = np.asarray(x, dtype=float)
    q = np.asarray(q, dtype=float)
    return float(np.dot(q, x * x))


def decompose(
    mu_u: np.ndarray,
    mu_t: np.ndarray,
    operator: np.ndarray,
    sigma: np.ndarray,
    q: np.ndarray | None = None,
) -> Decomposition:
    """Evaluate the population identity for a diagonal or full Sigma."""
    mu_u = np.asarray(mu_u, dtype=float)
    mu_t = np.asarray(mu_t, dtype=float)
    a = np.asarray(operator, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if q is None:
        q = np.full(len(mu_u), 1.0 / len(mu_u))
    q = np.asarray(q, dtype=float)
    transferred = a @ mu_t
    possible = weighted_norm2(mu_u, q)
    approximation = weighted_norm2(mu_u - transferred, q)
    signal = possible - approximation
    if sigma.ndim == 1:
        noise = float(np.dot(q, (a * a) @ sigma))
    else:
        noise = float(np.dot(q, np.einsum("ij,jk,ik->i", a, sigma, a)))
    delta = signal - noise
    gtr = signal / noise if noise > 0 else (np.inf if signal > 0 else np.nan)
    return Decomposition(possible, approximation, signal, noise, delta, gtr)


def empirical_gain(
    residual_u: np.ndarray,
    state_index_u: np.ndarray,
    state_prediction: np.ndarray,
) -> float:
    """State-balanced MSE(fallback)-MSE(geometry) on held-out rows."""
    values = []
    for state in range(len(state_prediction)):
        mask = state_index_u == state
        if mask.any():
            r = residual_u[mask]
            g = state_prediction[state]
            values.append(float(np.mean(r * r - (r - g) ** 2)))
    return float(np.mean(values))


def _row_normalize(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    sums = weights.sum(axis=1, keepdims=True)
    bad = sums[:, 0] <= EPS
    if bad.any():
        weights[bad] = 1.0
        sums = weights.sum(axis=1, keepdims=True)
    return weights / sums


def median_bandwidth(distance: np.ndarray, train: np.ndarray) -> float:
    d = np.asarray(distance, dtype=float)
    block = d[np.ix_(train, train)]
    positive = block[np.isfinite(block) & (block > EPS)]
    return float(np.median(positive)) if len(positive) else 1.0


def knn_operator(distance: np.ndarray, train: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    block = np.asarray(distance, dtype=float)[np.ix_(query, train)]
    result = np.zeros_like(block)
    for row, values in enumerate(block):
        order = np.argsort(values, kind="stable")[: min(k, len(train))]
        selected = values[order]
        zero = selected <= EPS
        if zero.any():
            result[row, order[zero]] = 1.0 / zero.sum()
        else:
            inv = 1.0 / np.maximum(selected, EPS)
            result[row, order] = inv / inv.sum()
    return result


def rbf_operator(
    distance: np.ndarray,
    train: np.ndarray,
    query: np.ndarray,
    bandwidth: float,
    scale: float = 1.0,
) -> np.ndarray:
    block = np.asarray(distance, dtype=float)[np.ix_(query, train)]
    h = max(float(bandwidth) * float(scale), EPS)
    shifted = block - np.min(block, axis=1, keepdims=True)
    return _row_normalize(np.exp(-0.5 * (shifted / h) ** 2))


def kernel_ridge_operator(
    distance: np.ndarray,
    train: np.ndarray,
    query: np.ndarray,
    bandwidth: float,
    ridge: float = 0.1,
) -> np.ndarray:
    d = np.asarray(distance, dtype=float)
    h = max(float(bandwidth), EPS)
    k_tt = np.exp(-0.5 * (d[np.ix_(train, train)] / h) ** 2)
    k_ut = np.exp(-0.5 * (d[np.ix_(query, train)] / h) ** 2)
    lam = ridge * max(float(np.trace(k_tt)) / max(len(train), 1), EPS)
    # A Gaussian of an arbitrary supplied metric need not be positive definite;
    # the frozen ridge still defines a linear interpolator when the system is
    # nonsingular, so use the general solver rather than silently assuming PSD.
    return solve(k_tt + lam * np.eye(len(train)), k_ut.T, assume_a="gen").T


def harmonic_operator(
    distance: np.ndarray,
    train: np.ndarray,
    query: np.ndarray,
    bandwidth: float,
    ridge: float = 1e-5,
) -> np.ndarray:
    """Transductive harmonic extension over the train+query metric graph."""
    nodes = np.concatenate([train, query])
    d = np.asarray(distance, dtype=float)[np.ix_(nodes, nodes)]
    h = max(float(bandwidth), EPS)
    w = np.exp(-0.5 * (d / h) ** 2)
    np.fill_diagonal(w, 0.0)
    lap = np.diag(w.sum(axis=1)) - w
    nt = len(train)
    l_uu = lap[nt:, nt:] + ridge * np.eye(len(query))
    l_ut = lap[nt:, :nt]
    return -solve(l_uu, l_ut, assume_a="pos")


def operator_family(distance: np.ndarray, train: np.ndarray, query: np.ndarray) -> Dict[str, np.ndarray]:
    """Frozen generic operator menu used by synthetic and real experiments."""
    h = median_bandwidth(distance, train)
    return {
        "knn_1": knn_operator(distance, train, query, 1),
        "knn_3": knn_operator(distance, train, query, 3),
        "knn_5": knn_operator(distance, train, query, 5),
        "knn_10": knn_operator(distance, train, query, 10),
        "rbf_half": rbf_operator(distance, train, query, h, 0.5),
        "rbf": rbf_operator(distance, train, query, h, 1.0),
        "rbf_double": rbf_operator(distance, train, query, h, 2.0),
        "kernel_ridge": kernel_ridge_operator(distance, train, query, h),
        "harmonic": harmonic_operator(distance, train, query, h),
    }


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def state_means(values: np.ndarray, states: np.ndarray, ordered_states: np.ndarray) -> np.ndarray:
    return np.asarray([np.mean(values[states == state]) for state in ordered_states], dtype=float)


def state_mean_variance(values: np.ndarray, states: np.ndarray, ordered_states: np.ndarray) -> np.ndarray:
    result = []
    for state in ordered_states:
        group = np.asarray(values[states == state], dtype=float)
        result.append(float(np.var(group, ddof=1) / len(group)) if len(group) > 1 else float(np.var(values) / max(len(group), 1)))
    return np.asarray(result)
