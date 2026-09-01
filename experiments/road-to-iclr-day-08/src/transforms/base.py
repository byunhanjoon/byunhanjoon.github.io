"""Base API and audit helpers for featurewise task transformations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


EXACTNESS_CLASSES = {
    "exact analytic bijection",
    "bijection on observed support",
    "order-preserving but lossy because of ties/finite precision",
    "non-bijective control",
}


@dataclass(frozen=True)
class TransformMetadata:
    """Scientifically relevant properties that every transform must declare."""

    name: str
    exactness_class: str
    monotonicity: str
    order_preserved: bool
    distances_preserved: bool
    data_dependent: bool
    severity: float

    def __post_init__(self) -> None:
        if self.exactness_class not in EXACTNESS_CLASSES:
            raise ValueError(f"invalid exactness class: {self.exactness_class}")
        if self.monotonicity not in {"increasing", "decreasing", "mixed", "identity"}:
            raise ValueError(f"invalid monotonicity: {self.monotonicity}")


class FeatureTransform(ABC):
    """Composable train-fit feature transformation.

    Implementations operate on two-dimensional floating-point arrays. Missing
    values are never fitted, modified, or filled; their masks are part of the
    task and must remain unchanged.
    """

    metadata: TransformMetadata

    def __init__(self) -> None:
        self.n_features_: int | None = None

    def fit(
        self,
        X_train: np.ndarray,
        feature_metadata: dict[str, Any] | None = None,
    ) -> "FeatureTransform":
        X = self._validate_X(X_train)
        self.n_features_ = X.shape[1]
        self._fit(X, feature_metadata or {})
        return self

    @abstractmethod
    def _fit(self, X_train: np.ndarray, feature_metadata: dict[str, Any]) -> None:
        """Fit state from context/train rows only."""

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_fitted_X(X)
        before_missing = np.isnan(X)
        result = np.asarray(self._transform(X.copy()), dtype=np.float64)
        if result.shape != X.shape:
            raise ValueError(f"transform changed shape from {X.shape} to {result.shape}")
        if not np.array_equal(before_missing, np.isnan(result)):
            raise ValueError("transform changed the missingness mask")
        return result

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_fitted_X(X)
        before_missing = np.isnan(X)
        result = np.asarray(self._inverse_transform(X.copy()), dtype=np.float64)
        if result.shape != X.shape:
            raise ValueError(f"inverse changed shape from {X.shape} to {result.shape}")
        if not np.array_equal(before_missing, np.isnan(result)):
            raise ValueError("inverse changed the missingness mask")
        return result

    @abstractmethod
    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted forward transformation."""

    @abstractmethod
    def _inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted inverse or documented approximate inverse."""

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable fitted state."""

    def audit(self, X_train: np.ndarray, X_test: np.ndarray) -> dict[str, Any]:
        """Audit missingness, finite outputs, reconstruction, and order."""
        train = self._validate_fitted_X(X_train)
        test = self._validate_fitted_X(X_test)
        joined = np.concatenate([train, test], axis=0)
        warped = self.transform(joined)
        restored = self.inverse_transform(warped)
        finite = np.isfinite(joined)
        abs_error = np.abs(restored[finite] - joined[finite])
        denom = np.maximum(1.0, np.abs(joined[finite]))
        rel_error = abs_error / denom
        violations = 0
        comparisons = 0
        reversals = 0
        max_tie_relative_input_gap = 0.0
        for column in range(joined.shape[1]):
            mask = np.isfinite(joined[:, column]) & np.isfinite(warped[:, column])
            x = joined[mask, column]
            y = warped[mask, column]
            if x.size < 2:
                continue
            order = np.argsort(x, kind="mergesort")
            dx = np.diff(x[order])
            dy = np.diff(y[order])
            distinct = dx > 0
            comparisons += int(distinct.sum())
            if self.metadata.monotonicity in {"increasing", "identity"}:
                tied = distinct & (dy == 0)
                reversed_order = distinct & (dy < 0)
            elif self.metadata.monotonicity == "decreasing":
                tied = distinct & (dy == 0)
                reversed_order = distinct & (dy > 0)
            else:
                tied = np.zeros_like(distinct)
                reversed_order = np.zeros_like(distinct)
            violations += int(np.sum(tied | reversed_order))
            reversals += int(np.sum(reversed_order))
            if np.any(tied):
                ordered_x = x[order]
                denominator = np.maximum.reduce(
                    [np.ones_like(dx), np.abs(ordered_x[:-1]), np.abs(ordered_x[1:])]
                )
                max_tie_relative_input_gap = max(
                    max_tie_relative_input_gap,
                    float(np.max((dx / denominator)[tied])),
                )
        return {
            "metadata": asdict(self.metadata),
            "train_rows": int(train.shape[0]),
            "test_rows": int(test.shape[0]),
            "n_features": int(joined.shape[1]),
            "missing_mask_preserved": bool(
                np.array_equal(np.isnan(joined), np.isnan(warped))
            ),
            "all_finite_inputs_have_finite_outputs": bool(np.isfinite(warped[finite]).all()),
            "max_abs_reconstruction_error": float(abs_error.max(initial=0.0)),
            "max_rel_reconstruction_error": float(rel_error.max(initial=0.0)),
            "strict_order_comparisons": comparisons,
            "strict_order_violations": violations,
            "strict_order_reversals": reversals,
            "max_order_tie_relative_input_gap": max_tie_relative_input_gap,
        }

    def _validate_fitted_X(self, X: np.ndarray) -> np.ndarray:
        if self.n_features_ is None:
            raise RuntimeError("transform has not been fitted")
        result = self._validate_X(X)
        if result.shape[1] != self.n_features_:
            raise ValueError(
                f"expected {self.n_features_} features, received {result.shape[1]}"
            )
        return result

    @staticmethod
    def _validate_X(X: np.ndarray) -> np.ndarray:
        result = np.asarray(X, dtype=np.float64)
        if result.ndim != 2:
            raise ValueError(f"expected a 2D array, received shape {result.shape}")
        if np.isinf(result).any():
            raise ValueError("infinite values are not supported")
        return result


def robust_location_scale(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Train-only median and nondegenerate IQR, with deterministic fallbacks."""
    center = np.nanmedian(X, axis=0)
    center = np.where(np.isfinite(center), center, 0.0)
    q25 = np.nanquantile(X, 0.25, axis=0)
    q75 = np.nanquantile(X, 0.75, axis=0)
    scale = q75 - q25
    std = np.nanstd(X, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, std)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    return center.astype(np.float64), scale.astype(np.float64)


def json_array(value: np.ndarray) -> list[Any]:
    """Convert an ndarray to a nested JSON-safe list."""
    return np.asarray(value).tolist()
