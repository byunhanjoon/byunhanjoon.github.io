"""Exact category-identity controls for matched-task audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def _json_scalar(value: Any) -> Any:
    """Convert numpy-backed category labels to JSON-native scalars."""
    return value.item() if isinstance(value, np.generic) else value


@dataclass(frozen=True)
class CategoricalBijectionMetadata:
    name: str = "categorical_bijection"
    exactness_class: str = "bijection on declared categorical support"
    membership_preserved: bool = True
    data_dependent: bool = True
    severity: float = 1.0


class CategoricalBijectionTransform:
    """Permute declared category identities featurewise.

    The declared pandas ``CategoricalDtype`` is treated as feature metadata. This
    lets a train-fitted transform define a total bijection for categories that
    are declared but absent from the sampled context, without reading query row
    values. Missing values remain missing. Input columns must remain categorical
    at application time so accidental metadata removal is caught explicitly.
    """

    metadata = CategoricalBijectionMetadata()

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)
        self.columns_: list[str] | None = None
        self.categories_: dict[str, list[Any]] = {}
        self.permuted_: dict[str, list[Any]] = {}
        self.ordered_: dict[str, bool] = {}

    def fit(
        self, frame: pd.DataFrame, categorical_columns: list[str]
    ) -> "CategoricalBijectionTransform":
        missing = set(categorical_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"categorical columns are absent: {sorted(missing)}")
        self.columns_ = list(categorical_columns)
        rng = np.random.default_rng(self.seed)
        for name in self.columns_:
            series = frame[name]
            if not isinstance(series.dtype, pd.CategoricalDtype):
                raise TypeError(f"column {name!r} must have categorical dtype")
            categories = list(series.cat.categories)
            order = rng.permutation(len(categories))
            self.categories_[name] = categories
            self.permuted_[name] = [categories[int(index)] for index in order]
            self.ordered_[name] = bool(series.cat.ordered)
        return self

    def _require_fit(self) -> list[str]:
        if self.columns_ is None:
            raise RuntimeError("categorical transform has not been fitted")
        return self.columns_

    def _apply(self, frame: pd.DataFrame, inverse: bool) -> pd.DataFrame:
        columns = self._require_fit()
        result = frame.copy()
        for name in columns:
            series = result[name]
            if not isinstance(series.dtype, pd.CategoricalDtype):
                raise TypeError(f"column {name!r} must have categorical dtype")
            expected = self.permuted_[name] if inverse else self.categories_[name]
            actual = list(series.cat.categories)
            if actual != expected:
                raise ValueError(
                    f"declared support changed for {name!r}: expected {expected!r}, got {actual!r}"
                )
            target = self.categories_[name] if inverse else self.permuted_[name]
            mapping = dict(zip(expected, target))
            result[name] = series.cat.rename_categories(mapping)
        return result

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self._apply(frame, inverse=False)

    def inverse_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self._apply(frame, inverse=True)

    def audit(self, train: pd.DataFrame, query: pd.DataFrame) -> dict[str, Any]:
        joined = pd.concat([train, query], ignore_index=True)
        warped = self.transform(joined)
        restored = self.inverse_transform(warped)
        equality_preserved = True
        missing_preserved = True
        round_trip = True
        support: dict[str, dict[str, Any]] = {}
        for name in self._require_fit():
            original = joined[name]
            changed = warped[name]
            original_codes = original.cat.codes.to_numpy()
            changed_codes = changed.cat.codes.to_numpy()
            # A pairwise equality matrix is quadratic in rows.  Equality-class
            # preservation is equivalent to a one-to-one mapping between the
            # observed original and changed codes, including the missing code.
            forward: dict[int, set[int]] = {}
            backward: dict[int, set[int]] = {}
            for source, target in zip(original_codes.tolist(), changed_codes.tolist()):
                forward.setdefault(int(source), set()).add(int(target))
                backward.setdefault(int(target), set()).add(int(source))
            equality_preserved &= bool(
                all(len(values) == 1 for values in forward.values())
                and all(len(values) == 1 for values in backward.values())
            )
            missing_preserved &= bool(original.isna().equals(changed.isna()))
            restored_series = restored[name]
            round_trip &= bool(
                np.array_equal(original.cat.codes.to_numpy(), restored_series.cat.codes.to_numpy())
                and list(original.cat.categories) == list(restored_series.cat.categories)
                and original.cat.ordered == restored_series.cat.ordered
            )
            support[name] = {
                "declared_levels": len(self.categories_[name]),
                "observed_train_levels": int(train[name].nunique(dropna=True)),
                "observed_query_levels": int(query[name].nunique(dropna=True)),
                "ordered": self.ordered_[name],
            }
        return {
            "metadata": self.metadata.__dict__,
            "equality_classes_preserved": equality_preserved,
            "missing_mask_preserved": missing_preserved,
            "exact_round_trip": round_trip,
            "support": support,
        }

    def state_dict(self) -> dict[str, Any]:
        columns = self._require_fit()
        return {
            "type": "categorical_bijection",
            "seed": self.seed,
            "columns": columns,
            "categories": {
                name: [_json_scalar(value) for value in self.categories_[name]]
                for name in columns
            },
            "permuted": {
                name: [_json_scalar(value) for value in self.permuted_[name]]
                for name in columns
            },
            "ordered": self.ordered_,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "CategoricalBijectionTransform":
        if state.get("type") != "categorical_bijection":
            raise ValueError(f"unexpected transform type: {state.get('type')}")
        transform = cls(int(state["seed"]))
        transform.columns_ = list(state["columns"])
        transform.categories_ = {name: list(values) for name, values in state["categories"].items()}
        transform.permuted_ = {name: list(values) for name, values in state["permuted"].items()}
        transform.ordered_ = {name: bool(value) for name, value in state["ordered"].items()}
        return transform
