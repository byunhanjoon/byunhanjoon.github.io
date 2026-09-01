"""Numerical feature transformations for matched-task reparameterization."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import ndtr, ndtri

from .base import FeatureTransform, TransformMetadata, json_array, robust_location_scale


def _copy_finite(column: np.ndarray, fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    result = column.copy()
    mask = np.isfinite(column)
    result[mask] = fn(column[mask])
    return result


class IdentityTransform(FeatureTransform):
    def __init__(self) -> None:
        super().__init__()
        self.metadata = TransformMetadata(
            "identity", "exact analytic bijection", "identity", True, True, False, 0.0
        )

    def _fit(self, X_train: np.ndarray, feature_metadata: dict[str, Any]) -> None:
        return None

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return X

    def _inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X

    def state_dict(self) -> dict[str, Any]:
        return {"type": "identity", "n_features": self.n_features_}


class PositiveAffineTransform(FeatureTransform):
    """Random positive unit changes with shifts expressed in robust train units."""

    def __init__(self, severity: float = 1.0, seed: int = 0) -> None:
        super().__init__()
        self.severity, self.seed = float(severity), int(seed)
        self.metadata = TransformMetadata(
            "positive_affine",
            "exact analytic bijection",
            "increasing",
            True,
            False,
            True,
            self.severity,
        )

    def _fit(self, X_train: np.ndarray, feature_metadata: dict[str, Any]) -> None:
        _, scale = robust_location_scale(X_train)
        rng = np.random.default_rng(self.seed)
        direction = rng.choice(np.asarray([-1.0, 1.0]), size=X_train.shape[1])
        magnitude = rng.uniform(0.55, 1.0, size=X_train.shape[1])
        self.a_ = np.exp(direction * magnitude * self.severity)
        self.b_ = rng.uniform(-1.0, 1.0, size=X_train.shape[1]) * scale * self.severity

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.a_ + self.b_

    def _inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.b_) / self.a_

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "positive_affine",
            "severity": self.severity,
            "seed": self.seed,
            "n_features": self.n_features_,
            "a": json_array(self.a_),
            "b": json_array(self.b_),
        }


class NegativeAffineTransform(FeatureTransform):
    """Information-preserving order-reversing affine control."""

    def __init__(self, severity: float = 1.0, seed: int = 0) -> None:
        super().__init__()
        self.severity, self.seed = float(severity), int(seed)
        self.metadata = TransformMetadata(
            "negative_affine",
            "exact analytic bijection",
            "decreasing",
            False,
            False,
            True,
            self.severity,
        )

    def _fit(self, X_train: np.ndarray, feature_metadata: dict[str, Any]) -> None:
        center, scale = robust_location_scale(X_train)
        rng = np.random.default_rng(self.seed)
        magnitude = np.exp(rng.uniform(-0.5, 0.5, X_train.shape[1]) * self.severity)
        self.a_ = -magnitude
        self.b_ = 2.0 * center + rng.uniform(-0.5, 0.5, X_train.shape[1]) * scale

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.a_ + self.b_

    def _inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.b_) / self.a_

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "negative_affine",
            "severity": self.severity,
            "seed": self.seed,
            "n_features": self.n_features_,
            "a": json_array(self.a_),
            "b": json_array(self.b_),
        }


class SignedPowerTransform(FeatureTransform):
    def __init__(self, power: float = 2.0) -> None:
        if power <= 0:
            raise ValueError("power must be positive")
        super().__init__()
        self.power = float(power)
        self.metadata = TransformMetadata(
            "signed_power",
            "exact analytic bijection",
            "increasing",
            True,
            False,
            True,
            abs(float(np.log2(power))),
        )

    def _fit(self, X_train: np.ndarray, feature_metadata: dict[str, Any]) -> None:
        self.center_, self.scale_ = robust_location_scale(X_train)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        z = (X - self.center_) / self.scale_
        return np.sign(z) * np.power(np.abs(z), self.power)

    def _inverse_transform(self, X: np.ndarray) -> np.ndarray:
        z = np.sign(X) * np.power(np.abs(X), 1.0 / self.power)
        return self.center_ + self.scale_ * z

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "signed_power",
            "power": self.power,
            "n_features": self.n_features_,
            "center": json_array(self.center_),
            "scale": json_array(self.scale_),
        }


class AsinhTransform(FeatureTransform):
    """Smooth saturating bijection with a closed-form inverse."""

    def __init__(self, severity: float = 1.0) -> None:
        if severity <= 0:
            raise ValueError("severity must be positive")
        super().__init__()
        self.severity = float(severity)
        self.metadata = TransformMetadata(
            "asinh",
            "exact analytic bijection",
            "increasing",
            True,
            False,
            True,
            self.severity,
        )

    def _fit(self, X_train: np.ndarray, feature_metadata: dict[str, Any]) -> None:
        self.center_, self.scale_ = robust_location_scale(X_train)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        z = (X - self.center_) / self.scale_
        return np.arcsinh(self.severity * z) / self.severity

    def _inverse_transform(self, X: np.ndarray) -> np.ndarray:
        z = np.sinh(self.severity * X) / self.severity
        return self.center_ + self.scale_ * z

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "asinh",
            "severity": self.severity,
            "n_features": self.n_features_,
            "center": json_array(self.center_),
            "scale": json_array(self.scale_),
        }


class RandomMonotonePWLTransform(FeatureTransform):
    """Strictly increasing random piecewise-linear bijection with linear tails."""

    DEFAULT_KNOTS = np.asarray([-4.0, -2.0, -1.0, -0.4, 0.0, 0.4, 1.0, 2.0, 4.0])

    def __init__(self, severity: float = 1.0, seed: int = 0) -> None:
        if severity < 0:
            raise ValueError("severity must be nonnegative")
        super().__init__()
        self.severity, self.seed = float(severity), int(seed)
        self.metadata = TransformMetadata(
            "random_monotone_pwl",
            "exact analytic bijection",
            "increasing",
            True,
            False,
            True,
            self.severity,
        )

    def _fit(self, X_train: np.ndarray, feature_metadata: dict[str, Any]) -> None:
        self.center_, self.scale_ = robust_location_scale(X_train)
        self.x_knots_ = np.tile(self.DEFAULT_KNOTS, (X_train.shape[1], 1))
        rng = np.random.default_rng(self.seed)
        log_slopes = rng.uniform(
            -self.severity, self.severity, size=(X_train.shape[1], len(self.DEFAULT_KNOTS) - 1)
        )
        slopes = np.exp(log_slopes)
        widths = np.diff(self.DEFAULT_KNOTS)
        y = np.concatenate(
            [np.zeros((X_train.shape[1], 1)), np.cumsum(slopes * widths, axis=1)], axis=1
        )
        zero_at = np.asarray([np.interp(0.0, self.DEFAULT_KNOTS, row) for row in y])
        y -= zero_at[:, None]
        z_train = (X_train - self.center_) / self.scale_
        mapped = np.empty_like(z_train)
        for j in range(X_train.shape[1]):
            mapped[:, j] = self._forward_column(z_train[:, j], self.DEFAULT_KNOTS, y[j])
        _, output_scale = robust_location_scale(mapped)
        self.y_knots_ = y / output_scale[:, None]

    @staticmethod
    def _forward_column(values: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        result = values.copy()
        mask = np.isfinite(values)
        v = values[mask]
        out = np.interp(v, x, y)
        left_slope = (y[1] - y[0]) / (x[1] - x[0])
        right_slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
        left, right = v < x[0], v > x[-1]
        out[left] = y[0] + left_slope * (v[left] - x[0])
        out[right] = y[-1] + right_slope * (v[right] - x[-1])
        result[mask] = out
        return result

    def _transform(self, X: np.ndarray) -> np.ndarray:
        z = (X - self.center_) / self.scale_
        result = np.empty_like(z)
        for j in range(z.shape[1]):
            result[:, j] = self._forward_column(z[:, j], self.x_knots_[j], self.y_knots_[j])
        return result

    def _inverse_transform(self, X: np.ndarray) -> np.ndarray:
        result = np.empty_like(X)
        for j in range(X.shape[1]):
            result[:, j] = self._forward_column(X[:, j], self.y_knots_[j], self.x_knots_[j])
        return self.center_ + self.scale_ * result

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "random_monotone_pwl",
            "severity": self.severity,
            "seed": self.seed,
            "n_features": self.n_features_,
            "center": json_array(self.center_),
            "scale": json_array(self.scale_),
            "x_knots": json_array(self.x_knots_),
            "y_knots": json_array(self.y_knots_),
        }


class MonotoneSplineTransform(FeatureTransform):
    """Held-out strictly monotone PCHIP warp with positive linear tails."""

    DEFAULT_KNOTS = np.asarray([-4.0, -2.5, -1.2, -0.5, 0.0, 0.5, 1.2, 2.5, 4.0])

    def __init__(self, severity: float = 1.0, seed: int = 0) -> None:
        if severity < 0:
            raise ValueError("severity must be nonnegative")
        super().__init__()
        self.severity, self.seed = float(severity), int(seed)
        self.metadata = TransformMetadata(
            "monotone_spline",
            "exact analytic bijection",
            "increasing",
            True,
            False,
            True,
            self.severity,
        )

    def _fit(self, X_train: np.ndarray, feature_metadata: dict[str, Any]) -> None:
        self.center_, self.scale_ = robust_location_scale(X_train)
        self.x_knots_ = np.tile(self.DEFAULT_KNOTS, (X_train.shape[1], 1))
        rng = np.random.default_rng(self.seed)
        increments = np.exp(
            rng.uniform(-self.severity, self.severity, (X_train.shape[1], len(self.DEFAULT_KNOTS) - 1))
        ) * np.diff(self.DEFAULT_KNOTS)
        y = np.concatenate(
            [np.zeros((X_train.shape[1], 1)), np.cumsum(increments, axis=1)], axis=1
        )
        for j in range(X_train.shape[1]):
            y[j] -= np.interp(0.0, self.DEFAULT_KNOTS, y[j])
        z_train = (X_train - self.center_) / self.scale_
        mapped = np.empty_like(z_train)
        for j in range(X_train.shape[1]):
            mapped[:, j] = self._forward_column(z_train[:, j], self.DEFAULT_KNOTS, y[j])
        _, output_scale = robust_location_scale(mapped)
        self.y_knots_ = y / output_scale[:, None]

    @staticmethod
    def _forward_column(values: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        result = values.copy()
        mask = np.isfinite(values)
        v = values[mask]
        interpolator = PchipInterpolator(x, y, extrapolate=False)
        clipped = np.clip(v, x[0], x[-1])
        out = np.asarray(interpolator(clipped), dtype=np.float64)
        left_slope = max(float(interpolator.derivative()(x[0])), 1e-8)
        right_slope = max(float(interpolator.derivative()(x[-1])), 1e-8)
        left, right = v < x[0], v > x[-1]
        out[left] = y[0] + left_slope * (v[left] - x[0])
        out[right] = y[-1] + right_slope * (v[right] - x[-1])
        result[mask] = out
        return result

    @classmethod
    def _inverse_column(cls, values: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        result = values.copy()
        mask = np.isfinite(values)
        v = values[mask]
        interp = PchipInterpolator(x, y, extrapolate=False)
        left_slope = max(float(interp.derivative()(x[0])), 1e-8)
        right_slope = max(float(interp.derivative()(x[-1])), 1e-8)
        out = np.empty_like(v)
        left, right = v < y[0], v > y[-1]
        out[left] = x[0] + (v[left] - y[0]) / left_slope
        out[right] = x[-1] + (v[right] - y[-1]) / right_slope
        middle = ~(left | right)
        targets = v[middle]
        lo = x[np.maximum(0, np.searchsorted(y, targets, side="right") - 1)]
        hi = x[np.minimum(len(x) - 1, np.searchsorted(y, targets, side="right"))]
        for _ in range(52):
            mid = (lo + hi) / 2.0
            below = np.asarray(interp(mid)) < targets
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        out[middle] = (lo + hi) / 2.0
        result[mask] = out
        return result

    def _transform(self, X: np.ndarray) -> np.ndarray:
        z = (X - self.center_) / self.scale_
        result = np.empty_like(z)
        for j in range(z.shape[1]):
            result[:, j] = self._forward_column(z[:, j], self.x_knots_[j], self.y_knots_[j])
        return result

    def _inverse_transform(self, X: np.ndarray) -> np.ndarray:
        z = np.empty_like(X)
        for j in range(X.shape[1]):
            z[:, j] = self._inverse_column(X[:, j], self.x_knots_[j], self.y_knots_[j])
        return self.center_ + self.scale_ * z

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "monotone_spline",
            "severity": self.severity,
            "seed": self.seed,
            "n_features": self.n_features_,
            "center": json_array(self.center_),
            "scale": json_array(self.scale_),
            "x_knots": json_array(self.x_knots_),
            "y_knots": json_array(self.y_knots_),
        }


class EmpiricalCDFTransform(FeatureTransform):
    """Train-fit empirical-CDF baseline, explicitly labeled lossy with ties."""

    def __init__(self) -> None:
        super().__init__()
        self.metadata = TransformMetadata(
            "empirical_cdf",
            "order-preserving but lossy because of ties/finite precision",
            "increasing",
            True,
            False,
            True,
            1.0,
        )

    def _fit(self, X_train: np.ndarray, feature_metadata: dict[str, Any]) -> None:
        self.values_: list[np.ndarray] = []
        self.quantiles_: list[np.ndarray] = []
        for j in range(X_train.shape[1]):
            values = X_train[np.isfinite(X_train[:, j]), j]
            unique, counts = np.unique(values, return_counts=True)
            if len(unique) == 0:
                unique = np.asarray([0.0])
                counts = np.asarray([1])
            cumulative = np.cumsum(counts)
            mid = (cumulative - 0.5 * counts) / counts.sum()
            self.values_.append(unique.astype(np.float64))
            self.quantiles_.append(mid.astype(np.float64))

    @staticmethod
    def _interp(column: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return _copy_finite(column, lambda v: np.interp(v, x, y, left=y[0], right=y[-1]))

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [self._interp(X[:, j], self.values_[j], self.quantiles_[j]) for j in range(X.shape[1])]
        )

    def _inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [self._interp(X[:, j], self.quantiles_[j], self.values_[j]) for j in range(X.shape[1])]
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "empirical_cdf",
            "n_features": self.n_features_,
            "values": [json_array(x) for x in self.values_],
            "quantiles": [json_array(x) for x in self.quantiles_],
        }


class QuantileGaussianTransform(EmpiricalCDFTransform):
    def __init__(self, epsilon: float = 1e-5) -> None:
        if not 0 < epsilon < 0.5:
            raise ValueError("epsilon must be between zero and one half")
        super().__init__()
        self.epsilon = float(epsilon)
        self.metadata = TransformMetadata(
            "quantile_gaussian",
            "order-preserving but lossy because of ties/finite precision",
            "increasing",
            True,
            False,
            True,
            1.0,
        )

    def _transform(self, X: np.ndarray) -> np.ndarray:
        uniform = super()._transform(X)
        return _copy_finite(uniform, lambda v: ndtri(np.clip(v, self.epsilon, 1 - self.epsilon)))

    def _inverse_transform(self, X: np.ndarray) -> np.ndarray:
        uniform = _copy_finite(X, ndtr)
        return super()._inverse_transform(uniform)

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state.update({"type": "quantile_gaussian", "epsilon": self.epsilon})
        return state


class AtomicSpacingTransform(FeatureTransform):
    """Order-preserving random remapping of train-observed equality classes."""

    def __init__(self, severity: float = 1.0, seed: int = 0) -> None:
        super().__init__()
        self.severity, self.seed = float(severity), int(seed)
        self.metadata = TransformMetadata(
            "atomic_spacing",
            "bijection on observed support",
            "increasing",
            True,
            False,
            True,
            self.severity,
        )

    def _fit(self, X_train: np.ndarray, feature_metadata: dict[str, Any]) -> None:
        rng = np.random.default_rng(self.seed)
        self.values_: list[np.ndarray] = []
        self.remapped_: list[np.ndarray] = []
        self.atom_statistics_: list[dict[str, float | int]] = []
        for j in range(X_train.shape[1]):
            finite = X_train[np.isfinite(X_train[:, j]), j]
            unique, counts = np.unique(finite, return_counts=True)
            if len(unique) == 0:
                unique, counts = np.asarray([0.0]), np.asarray([1])
            if len(unique) == 1:
                remapped = unique.copy()
            else:
                gaps = np.exp(rng.uniform(-self.severity, self.severity, len(unique) - 1))
                remapped = np.concatenate([[0.0], np.cumsum(gaps)])
                remapped -= np.median(remapped)
                source_iqr = np.subtract(*np.quantile(unique, [0.75, 0.25]))
                target_iqr = np.subtract(*np.quantile(remapped, [0.75, 0.25]))
                if source_iqr > 0 and target_iqr > 0:
                    remapped *= source_iqr / target_iqr
            probabilities = counts / counts.sum()
            self.values_.append(unique.astype(np.float64))
            self.remapped_.append(remapped.astype(np.float64))
            self.atom_statistics_.append(
                {
                    "unique_levels": int(len(unique)),
                    "largest_atom_mass": float(probabilities.max()),
                    "entropy": float(-(probabilities * np.log(probabilities)).sum()),
                    "repeated_fraction": float(1.0 - len(unique) / max(1, len(finite))),
                }
            )

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [RandomMonotonePWLTransform._forward_column(X[:, j], self.values_[j], self.remapped_[j])
             if len(self.values_[j]) > 1 else X[:, j].copy()
             for j in range(X.shape[1])]
        )

    def _inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [RandomMonotonePWLTransform._forward_column(X[:, j], self.remapped_[j], self.values_[j])
             if len(self.values_[j]) > 1 else X[:, j].copy()
             for j in range(X.shape[1])]
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "atomic_spacing",
            "severity": self.severity,
            "seed": self.seed,
            "n_features": self.n_features_,
            "values": [json_array(x) for x in self.values_],
            "remapped": [json_array(x) for x in self.remapped_],
            "atom_statistics": self.atom_statistics_,
        }


class ComposedTransform(FeatureTransform):
    def __init__(self, transforms: list[FeatureTransform]) -> None:
        if not 2 <= len(transforms) <= 3:
            raise ValueError("compositions must contain two or three transforms")
        super().__init__()
        self.transforms = transforms
        exactness = (
            "exact analytic bijection"
            if all(t.metadata.exactness_class == "exact analytic bijection" for t in transforms)
            else "order-preserving but lossy because of ties/finite precision"
        )
        self.metadata = TransformMetadata(
            "composition",
            exactness,
            "increasing" if all(t.metadata.order_preserved for t in transforms) else "mixed",
            all(t.metadata.order_preserved for t in transforms),
            all(t.metadata.distances_preserved for t in transforms),
            any(t.metadata.data_dependent for t in transforms),
            float(sum(t.metadata.severity for t in transforms)),
        )

    def _fit(self, X_train: np.ndarray, feature_metadata: dict[str, Any]) -> None:
        current = X_train
        for transform in self.transforms:
            transform.fit(current, feature_metadata)
            current = transform.transform(current)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        for transform in self.transforms:
            X = transform.transform(X)
        return X

    def _inverse_transform(self, X: np.ndarray) -> np.ndarray:
        for transform in reversed(self.transforms):
            X = transform.inverse_transform(X)
        return X

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "composition",
            "n_features": self.n_features_,
            "transforms": [t.state_dict() for t in self.transforms],
        }


def _restore_common(transform: FeatureTransform, state: dict[str, Any]) -> FeatureTransform:
    transform.n_features_ = int(state["n_features"])
    for key in ("a", "b", "center", "scale", "x_knots", "y_knots"):
        if key in state:
            setattr(transform, f"{key}_", np.asarray(state[key], dtype=np.float64))
    if "values" in state:
        transform.values_ = [np.asarray(x, dtype=np.float64) for x in state["values"]]  # type: ignore[attr-defined]
    if "quantiles" in state:
        transform.quantiles_ = [np.asarray(x, dtype=np.float64) for x in state["quantiles"]]  # type: ignore[attr-defined]
    if "remapped" in state:
        transform.remapped_ = [np.asarray(x, dtype=np.float64) for x in state["remapped"]]  # type: ignore[attr-defined]
    if "atom_statistics" in state:
        transform.atom_statistics_ = list(state["atom_statistics"])  # type: ignore[attr-defined]
    return transform


def transform_from_state(state: dict[str, Any]) -> FeatureTransform:
    """Deterministically reconstruct a fitted transform from JSON state."""
    kind = state["type"]
    constructors: dict[str, Callable[[], FeatureTransform]] = {
        "identity": IdentityTransform,
        "positive_affine": lambda: PositiveAffineTransform(state["severity"], state["seed"]),
        "negative_affine": lambda: NegativeAffineTransform(state["severity"], state["seed"]),
        "signed_power": lambda: SignedPowerTransform(state["power"]),
        "asinh": lambda: AsinhTransform(state["severity"]),
        "random_monotone_pwl": lambda: RandomMonotonePWLTransform(state["severity"], state["seed"]),
        "monotone_spline": lambda: MonotoneSplineTransform(state["severity"], state["seed"]),
        "empirical_cdf": EmpiricalCDFTransform,
        "quantile_gaussian": lambda: QuantileGaussianTransform(state["epsilon"]),
        "atomic_spacing": lambda: AtomicSpacingTransform(state["severity"], state["seed"]),
    }
    if kind == "composition":
        transform = ComposedTransform([transform_from_state(x) for x in state["transforms"]])
        transform.n_features_ = int(state["n_features"])
        return transform
    if kind not in constructors:
        raise ValueError(f"unknown transform type: {kind}")
    return _restore_common(constructors[kind](), state)

