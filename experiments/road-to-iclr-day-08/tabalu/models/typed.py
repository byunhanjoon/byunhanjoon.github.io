"""Deterministic typed operators and a sparse linear executable over them."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class HeterogeneousBatch:
    continuous: np.ndarray
    categorical: np.ndarray
    ordinal: np.ndarray
    timestamp_hours: np.ndarray

    def __post_init__(self) -> None:
        n_rows = len(self.continuous)
        if self.continuous.ndim != 2 or self.continuous.shape[1] != 2:
            raise ValueError("continuous must have shape [rows, 2]")
        if any(len(values) != n_rows for values in (self.categorical, self.ordinal, self.timestamp_hours)):
            raise ValueError("all typed columns must have the same number of rows")


def safe_divide(left: np.ndarray, right: np.ndarray, epsilon: float = 1.0e-3) -> np.ndarray:
    sign = np.where(right < 0, -1.0, 1.0)
    denominator = sign * np.maximum(np.abs(right), epsilon)
    return left / denominator


def categorical_equal(values: np.ndarray, category: int) -> np.ndarray:
    return (values == category).astype(np.float64)


def categorical_membership(values: np.ndarray, categories: tuple[int, ...]) -> np.ndarray:
    return np.isin(values, categories).astype(np.float64)


def ordinal_threshold(values: np.ndarray, threshold: int) -> np.ndarray:
    return (values >= threshold).astype(np.float64)


def bounded_ordinal_difference(values: np.ndarray, center: int, bound: float = 2.0) -> np.ndarray:
    return np.clip(values.astype(np.float64) - center, -bound, bound)


def datetime_parts(timestamp_hours: np.ndarray) -> dict[str, np.ndarray]:
    hours = np.asarray(timestamp_hours, dtype=np.int64)
    days = np.floor_divide(hours, 24)
    hour = np.mod(hours, 24)
    weekday = np.mod(days + 3, 7)  # Unix epoch 1970-01-01 was Thursday.
    dates = hours.astype("datetime64[h]").astype("datetime64[D]")
    year_start = dates.astype("datetime64[Y]").astype("datetime64[D]")
    day_of_year = (dates - year_start).astype(np.int64)
    month = dates.astype("datetime64[M]").astype(np.int64) % 12
    return {
        "elapsed_days": days.astype(np.float64),
        "hour": hour,
        "weekday": weekday,
        "month": month,
        "day_of_year": day_of_year,
    }


def typed_design_matrix(
    batch: HeterogeneousBatch,
    *,
    include_categorical_conditions: bool = True,
    include_ordinal: bool = True,
    include_time: bool = True,
) -> tuple[np.ndarray, list[str]]:
    x0, x1 = batch.continuous.T
    columns: list[np.ndarray] = [
        x0,
        x1,
        x0 + x1,
        x0 - x1,
        x0 * x1,
        safe_divide(x0, x1),
        np.abs(x0),
        x0**2,
        np.minimum(x0, x1),
        np.maximum(x0, x1),
    ]
    names = ["x0", "x1", "x0+x1", "x0-x1", "x0*x1", "x0/x1", "abs(x0)", "x0^2", "min", "max"]
    if include_categorical_conditions:
        for category in range(3):
            indicator = categorical_equal(batch.categorical, category)
            columns.extend((indicator, indicator * (x0 * x1), indicator * (x0 + x1), indicator * safe_divide(x0, x1)))
            names.extend(
                (
                    f"cat=={category}",
                    f"[cat=={category}]*(x0*x1)",
                    f"[cat=={category}]*(x0+x1)",
                    f"[cat=={category}]*(x0/x1)",
                )
            )
        columns.append(categorical_membership(batch.categorical, (0, 2)))
        names.append("cat in {0,2}")
    if include_ordinal:
        rank = batch.ordinal.astype(np.float64)
        columns.extend(
            (
                rank,
                ordinal_threshold(batch.ordinal, 1),
                ordinal_threshold(batch.ordinal, 2),
                ordinal_threshold(batch.ordinal, 3),
                bounded_ordinal_difference(batch.ordinal, 2),
                rank * x0,
            )
        )
        names.extend(("rank", "rank>=1", "rank>=2", "rank>=3", "bounded_rank_diff", "rank*x0"))
    if include_time:
        parts = datetime_parts(batch.timestamp_hours)
        elapsed = (parts["elapsed_days"] - 18_262.0) / 365.25
        hour_phase = 2 * np.pi * parts["hour"] / 24
        weekday_phase = 2 * np.pi * parts["weekday"] / 7
        year_phase = 2 * np.pi * parts["day_of_year"] / 365.25
        columns.extend(
            (
                elapsed,
                np.sin(hour_phase),
                np.cos(hour_phase),
                np.sin(weekday_phase),
                np.cos(weekday_phase),
                np.sin(year_phase),
                np.cos(year_phase),
                (parts["month"] >= 6).astype(np.float64),
            )
        )
        names.extend(("elapsed", "sin(hour)", "cos(hour)", "sin(weekday)", "cos(weekday)", "sin(year)", "cos(year)", "month>=6"))
    return np.column_stack(columns).astype(np.float64), names


class SparseTypedProgram:
    """A fitted affine program over an explicit deterministic typed library."""

    def __init__(self, *, include_categorical_conditions: bool = True, include_ordinal: bool = True, include_time: bool = True) -> None:
        self.options = {
            "include_categorical_conditions": include_categorical_conditions,
            "include_ordinal": include_ordinal,
            "include_time": include_time,
        }
        self.scaler = StandardScaler()
        self.model: OrthogonalMatchingPursuit | None = None
        self.feature_names: list[str] = []

    def fit(self, batch: HeterogeneousBatch, targets: np.ndarray, validation: tuple[HeterogeneousBatch, np.ndarray]) -> None:
        design, self.feature_names = typed_design_matrix(batch, **self.options)
        val_batch, val_targets = validation
        val_design, _ = typed_design_matrix(val_batch, **self.options)
        scaled = self.scaler.fit_transform(design)
        scaled_val = self.scaler.transform(val_design)
        best_score = float("inf")
        for operation_budget in (4, 6, 8, 10, 12):
            candidate = OrthogonalMatchingPursuit(
                n_nonzero_coefs=min(operation_budget, scaled.shape[1] - 1),
                tol=None,
            )
            candidate.fit(scaled, targets)
            score = float(np.mean((candidate.predict(scaled_val) - val_targets) ** 2))
            if score < best_score:
                best_score = score
                self.model = candidate

    def predict(self, batch: HeterogeneousBatch) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("typed program has not been fitted")
        design, _ = typed_design_matrix(batch, **self.options)
        return np.asarray(self.model.predict(self.scaler.transform(design)), dtype=np.float64)

    @property
    def operation_count(self) -> int:
        if self.model is None:
            return 0
        return int(np.count_nonzero(np.abs(self.model.coef_) > 1.0e-10))
