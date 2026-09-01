"""Stable/order and explicit coordinate-shape representations."""

from __future__ import annotations

import numpy as np


class TieAwareECDF:
    """Context-fitted mid-ECDF; query values never affect fitted state."""

    def __init__(self) -> None:
        self.columns_: list[np.ndarray] | None = None

    def fit(self, context_x: np.ndarray) -> "TieAwareECDF":
        x = np.asarray(context_x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("expected rows by features")
        self.columns_ = [np.sort(col[np.isfinite(col)]) for col in x.T]
        if any(col.size == 0 for col in self.columns_):
            raise ValueError("each feature needs one finite context value")
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.columns_ is None:
            raise RuntimeError("fit before transform")
        values = np.asarray(x, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.columns_):
            raise ValueError("feature count mismatch")
        out = np.full_like(values, np.nan)
        for j, fitted in enumerate(self.columns_):
            finite = np.isfinite(values[:, j])
            v = values[finite, j]
            left = np.searchsorted(fitted, v, side="left")
            right = np.searchsorted(fitted, v, side="right")
            out[finite, j] = (left + right) / (2.0 * fitted.size)
        return out


def robust_affine(context_x: np.ndarray, x: np.ndarray) -> np.ndarray:
    context = np.asarray(context_x, dtype=np.float64)
    values = np.asarray(x, dtype=np.float64)
    med = np.nanmedian(context, axis=0)
    q25, q75 = np.nanquantile(context, [0.25, 0.75], axis=0)
    scale = q75 - q25
    fallback = np.nanmedian(np.abs(context - med), axis=0) * 1.4826
    scale = np.where(scale > 1e-12, scale, np.where(fallback > 1e-12, fallback, 1.0))
    return (values - med) / scale


def marginal_descriptors(context_x: np.ndarray) -> np.ndarray:
    """Permutation-invariant episode descriptor with invariant and shape blocks."""
    x = np.asarray(context_x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("expected rows by features")
    probs = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    per_feature: list[np.ndarray] = []
    for col in x.T:
        finite = col[np.isfinite(col)]
        if finite.size < 4:
            raise ValueError("descriptor requires four finite values per feature")
        med = np.median(finite)
        q25, q75 = np.quantile(finite, [0.25, 0.75])
        iqr = max(float(q75 - q25), 1e-12)
        standardized = (finite - med) / iqr
        q = np.quantile(standardized, probs)
        spacings = np.diff(q)
        centered = standardized - standardized.mean()
        sd = max(float(standardized.std()), 1e-12)
        skew = float(np.mean((centered / sd) ** 3))
        kurt = float(np.mean((centered / sd) ** 4) - 3.0)
        unique_fraction = np.unique(finite).size / finite.size
        missing_fraction = 1.0 - finite.size / col.size
        raw_scale = np.log(max(iqr, 1e-12))
        per_feature.append(
            np.r_[q, spacings, skew, kurt, raw_scale, unique_fraction, missing_fraction]
        )
    matrix = np.vstack(per_feature)
    return np.r_[matrix.mean(axis=0), matrix.std(axis=0)]

