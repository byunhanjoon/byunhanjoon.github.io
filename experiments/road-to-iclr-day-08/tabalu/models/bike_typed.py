"""Sparse typed executable models for the UCI Bike Sharing pilot."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.preprocessing import StandardScaler


def bike_typed_design(
    frame: pd.DataFrame, *, bounded_time_only: bool = False
) -> tuple[np.ndarray, list[str]]:
    temp = frame["temp"].to_numpy(dtype=np.float64)
    atemp = frame["atemp"].to_numpy(dtype=np.float64)
    humidity = frame["hum"].to_numpy(dtype=np.float64)
    wind = frame["windspeed"].to_numpy(dtype=np.float64)
    hour = frame["hr"].to_numpy(dtype=np.float64)
    weekday = frame["weekday"].to_numpy(dtype=np.float64)
    month = frame["mnth"].to_numpy(dtype=np.float64)
    elapsed = (
        (frame["dteday"] - pd.Timestamp("2011-01-01")).dt.total_seconds().to_numpy()
        / (365.25 * 24 * 3600)
    )
    hour_phase = 2 * np.pi * hour / 24
    weekday_phase = 2 * np.pi * weekday / 7
    month_phase = 2 * np.pi * (month - 1) / 12
    columns: list[np.ndarray] = [
        temp,
        atemp,
        humidity,
        wind,
        temp**2,
        humidity**2,
        temp * humidity,
        temp * wind,
    ]
    names = [
        "temp",
        "atemp",
        "humidity",
        "wind",
        "temp^2",
        "humidity^2",
        "temp*humidity",
        "temp*wind",
    ]
    if not bounded_time_only:
        columns.extend((elapsed, elapsed**2))
        names.extend(("elapsed", "elapsed^2"))
    columns.extend(
        (
        np.sin(hour_phase),
        np.cos(hour_phase),
        np.sin(2 * hour_phase),
        np.cos(2 * hour_phase),
        np.sin(weekday_phase),
        np.cos(weekday_phase),
        np.sin(month_phase),
        np.cos(month_phase),
        )
    )
    names.extend(
        (
        "sin(hour)",
        "cos(hour)",
        "sin(2hour)",
        "cos(2hour)",
        "sin(weekday)",
        "cos(weekday)",
        "sin(month)",
        "cos(month)",
        )
    )
    working = frame["workingday"].to_numpy(dtype=np.float64)
    columns.extend((working * np.sin(hour_phase), working * np.cos(hour_phase)))
    names.extend(("working*sin(hour)", "working*cos(hour)"))
    for source, cardinality in (("season", 4), ("weathersit", 4), ("holiday", 2), ("workingday", 2)):
        values = frame[source].to_numpy()
        for category in range(1 if source in {"season", "weathersit"} else 0, cardinality + (1 if source in {"season", "weathersit"} else 0)):
            indicator = (values == category).astype(np.float64)
            columns.append(indicator)
            names.append(f"{source}=={category}")
            if source == "weathersit":
                columns.append(indicator * temp)
                names.append(f"[{source}=={category}]*temp")
    return np.column_stack(columns), names


class SparseBikeProgram:
    def __init__(
        self,
        operation_budgets: tuple[int, ...] = (8, 12, 16, 24, 32),
        *,
        bounded_time_only: bool = False,
    ) -> None:
        self.operation_budgets = operation_budgets
        self.bounded_time_only = bounded_time_only
        self.scaler = StandardScaler()
        self.model: OrthogonalMatchingPursuit | None = None
        self.feature_names: list[str] = []

    def fit(self, frame: pd.DataFrame, targets: np.ndarray, validation: tuple[pd.DataFrame, np.ndarray]) -> None:
        design, self.feature_names = bike_typed_design(frame, bounded_time_only=self.bounded_time_only)
        val_frame, val_targets = validation
        val_design, _ = bike_typed_design(val_frame, bounded_time_only=self.bounded_time_only)
        scaled = self.scaler.fit_transform(design)
        scaled_val = self.scaler.transform(val_design)
        best = float("inf")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*linear dependence in the dictionary.*")
            for budget in self.operation_budgets:
                candidate = OrthogonalMatchingPursuit(n_nonzero_coefs=min(budget, scaled.shape[1] - 1))
                candidate.fit(scaled, targets)
                score = float(np.mean((candidate.predict(scaled_val) - val_targets) ** 2))
                if score < best:
                    best = score
                    self.model = candidate

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("bike program has not been fitted")
        design, _ = bike_typed_design(frame, bounded_time_only=self.bounded_time_only)
        return np.asarray(self.model.predict(self.scaler.transform(design)), dtype=np.float64)

    @property
    def operation_count(self) -> int:
        return 0 if self.model is None else int(np.count_nonzero(np.abs(self.model.coef_) > 1.0e-10))


class SeasonRoutedBikeProgram:
    def __init__(self, *, bounded_time_only: bool = False) -> None:
        self.programs = {
            season: SparseBikeProgram(bounded_time_only=bounded_time_only)
            for season in range(1, 5)
        }

    def fit(self, frame: pd.DataFrame, targets: np.ndarray, validation: tuple[pd.DataFrame, np.ndarray]) -> None:
        val_frame, val_targets = validation
        for season, program in self.programs.items():
            train_mask = frame["season"].to_numpy() == season
            val_mask = val_frame["season"].to_numpy() == season
            program.fit(
                frame.loc[train_mask],
                targets[train_mask],
                (val_frame.loc[val_mask], val_targets[val_mask]),
            )

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        output = np.empty(len(frame), dtype=np.float64)
        seasons = frame["season"].to_numpy()
        for season, program in self.programs.items():
            mask = seasons == season
            output[mask] = program.predict(frame.loc[mask])
        return output

    @property
    def operation_count(self) -> int:
        return sum(program.operation_count for program in self.programs.values())
