"""Serializable strictly increasing scalar transforms used by PriorDial."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


def _array(x: np.ndarray | list[float]) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64)
    if out.ndim != 1:
        raise ValueError("scalar transforms require a one-dimensional array")
    return out


class ScalarTransform:
    name = "base"

    def fit(self, context_x: np.ndarray) -> "ScalarTransform":
        values = _array(context_x)
        if not np.all(np.isfinite(values)):
            raise ValueError("fit data must be finite")
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        state = asdict(self) if hasattr(self, "__dataclass_fields__") else {}
        return {"name": self.name, **_jsonable(state)}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


@dataclass
class IdentityTransform(ScalarTransform):
    name = "identity"

    def transform(self, x: np.ndarray) -> np.ndarray:
        return _array(x).copy()

    inverse_transform = transform


@dataclass
class PositiveAffineTransform(ScalarTransform):
    scale: float = 1.0
    shift: float = 0.0
    name = "affine"

    def __post_init__(self) -> None:
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scale must be finite and positive")

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self.scale * _array(x) + self.shift

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return (_array(x) - self.shift) / self.scale


@dataclass
class SignedPowerTransform(ScalarTransform):
    power: float = 1.0
    center: float = 0.0
    scale: float = 1.0
    name = "signed_power"

    def __post_init__(self) -> None:
        if self.power <= 0 or self.scale <= 0:
            raise ValueError("power and scale must be positive")

    def transform(self, x: np.ndarray) -> np.ndarray:
        u = (_array(x) - self.center) / self.scale
        return self.center + self.scale * np.sign(u) * np.abs(u) ** self.power

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        u = (_array(x) - self.center) / self.scale
        return self.center + self.scale * np.sign(u) * np.abs(u) ** (1.0 / self.power)


@dataclass
class AsinhTransform(ScalarTransform):
    severity: float = 0.8
    center: float = 0.0
    scale: float = 1.0
    name = "asinh"

    def __post_init__(self) -> None:
        if self.severity <= 0 or self.scale <= 0:
            raise ValueError("severity and scale must be positive")

    def transform(self, x: np.ndarray) -> np.ndarray:
        u = (_array(x) - self.center) / self.scale
        return self.center + self.scale * np.arcsinh(self.severity * u) / self.severity

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        u = (_array(x) - self.center) / self.scale
        return self.center + self.scale * np.sinh(self.severity * u) / self.severity


@dataclass
class SinhTransform(ScalarTransform):
    severity: float = 0.5
    center: float = 0.0
    scale: float = 1.0
    name = "sinh"

    def __post_init__(self) -> None:
        if self.severity <= 0 or self.scale <= 0:
            raise ValueError("severity and scale must be positive")

    def transform(self, x: np.ndarray) -> np.ndarray:
        u = (_array(x) - self.center) / self.scale
        return self.center + self.scale * np.sinh(self.severity * u) / self.severity

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        u = (_array(x) - self.center) / self.scale
        return self.center + self.scale * np.arcsinh(self.severity * u) / self.severity


def _linear_with_tails(x: np.ndarray, xp: np.ndarray, yp: np.ndarray) -> np.ndarray:
    out = np.interp(x, xp, yp)
    left = x < xp[0]
    right = x > xp[-1]
    left_slope = (yp[1] - yp[0]) / (xp[1] - xp[0])
    right_slope = (yp[-1] - yp[-2]) / (xp[-1] - xp[-2])
    out[left] = yp[0] + left_slope * (x[left] - xp[0])
    out[right] = yp[-1] + right_slope * (x[right] - xp[-1])
    return out


@dataclass
class MonotonePWLTransform(ScalarTransform):
    seed: int = 0
    n_knots: int = 7
    slope_sigma: float = 0.7
    x_knots: list[float] = field(default_factory=list)
    y_knots: list[float] = field(default_factory=list)
    name = "pwl"

    def fit(self, context_x: np.ndarray) -> "MonotonePWLTransform":
        values = _array(context_x)
        if not np.all(np.isfinite(values)):
            raise ValueError("fit data must be finite")
        if np.unique(values).size < 3:
            raise ValueError("PWL fit requires at least three unique values")
        probs = np.linspace(0.02, 0.98, self.n_knots)
        xk = np.unique(np.quantile(values, probs))
        if xk.size < 3:
            raise ValueError("quantile knots collapsed")
        rng = np.random.default_rng(self.seed)
        dx = np.diff(xk)
        slopes = np.exp(rng.normal(0.0, self.slope_sigma, size=dx.size))
        increments = dx * slopes
        increments *= (xk[-1] - xk[0]) / increments.sum()
        yk = np.r_[xk[0], xk[0] + np.cumsum(increments)]
        self.x_knots = xk.tolist()
        self.y_knots = yk.tolist()
        return self

    def _knots(self) -> tuple[np.ndarray, np.ndarray]:
        if len(self.x_knots) < 3:
            raise RuntimeError("PWL transform must be fit before use")
        return np.asarray(self.x_knots), np.asarray(self.y_knots)

    def transform(self, x: np.ndarray) -> np.ndarray:
        xk, yk = self._knots()
        return _linear_with_tails(_array(x), xk, yk)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        xk, yk = self._knots()
        return _linear_with_tails(_array(x), yk, xk)


def make_warp(name: str, rng: np.random.Generator) -> ScalarTransform:
    """Sample a development-range warp without inspecting labels or query values."""
    if name == "identity":
        return IdentityTransform()
    if name == "affine":
        return PositiveAffineTransform(
            scale=float(np.exp(rng.uniform(-1.0, 1.0))), shift=float(rng.normal(0, 1.0))
        )
    if name == "signed_power":
        return SignedPowerTransform(power=float(rng.uniform(0.45, 1.9)))
    if name == "asinh":
        return AsinhTransform(severity=float(rng.uniform(0.5, 1.4)))
    if name == "sinh":
        return SinhTransform(severity=float(rng.uniform(0.25, 0.7)))
    if name == "pwl":
        return MonotonePWLTransform(seed=int(rng.integers(0, 2**31 - 1)))
    raise KeyError(f"unknown warp family: {name}")


def audit_transform(transform: ScalarTransform, x: np.ndarray) -> dict[str, float | bool]:
    values = _array(x)
    unique = np.unique(values[np.isfinite(values)])
    if unique.size < 3:
        raise ValueError("audit needs at least three unique finite values")
    transformed = transform.transform(unique)
    reconstructed = transform.inverse_transform(transformed)
    scale = max(float(np.ptp(unique)), 1.0)
    ties = np.array([values[0], values[0], values[-1], values[-1]])
    transformed_ties = transform.transform(ties)
    return {
        "strictly_increasing": bool(np.all(np.diff(transformed) > 0)),
        "inverse_max_scaled_error": float(np.max(np.abs(reconstructed - unique)) / scale),
        "ties_preserved": bool(
            transformed_ties[0] == transformed_ties[1]
            and transformed_ties[2] == transformed_ties[3]
        ),
        "all_finite": bool(np.all(np.isfinite(transformed)) and np.all(np.isfinite(reconstructed))),
    }
