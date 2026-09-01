"""Sparse exact numeric libraries for a small general-tabular pilot."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression, OrthogonalMatchingPursuit
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


class ExactNumericLibrary:
    def __init__(self) -> None:
        self.polynomial = PolynomialFeatures(degree=2, include_bias=False)
        self.scaler = StandardScaler()
        self.names: list[str] = []

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        design = self.polynomial.fit_transform(features)
        self.names = list(self.polynomial.get_feature_names_out())
        return self.scaler.fit_transform(design)

    def transform(self, features: np.ndarray) -> np.ndarray:
        return self.scaler.transform(self.polynomial.transform(features))


class SparseExactRegressor:
    def __init__(self) -> None:
        self.library = ExactNumericLibrary()
        self.model: OrthogonalMatchingPursuit | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray, validation: tuple[np.ndarray, np.ndarray]) -> None:
        train = self.library.fit_transform(features)
        val_x, val_y = validation
        val = self.library.transform(val_x)
        best = float("inf")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*linear dependence in the dictionary.*")
            for budget in (4, 8, 12, 16, 24, 32):
                candidate = OrthogonalMatchingPursuit(n_nonzero_coefs=min(budget, train.shape[1] - 1))
                candidate.fit(train, targets)
                score = float(np.mean((candidate.predict(val) - val_y) ** 2))
                if score < best:
                    best = score
                    self.model = candidate

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model has not been fitted")
        return np.asarray(self.model.predict(self.library.transform(features)), dtype=np.float64)

    @property
    def operation_count(self) -> int:
        return 0 if self.model is None else int(np.count_nonzero(np.abs(self.model.coef_) > 1.0e-10))


class SparseExactClassifier:
    def __init__(self) -> None:
        self.library = ExactNumericLibrary()
        self.model: LogisticRegression | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray, validation: tuple[np.ndarray, np.ndarray]) -> None:
        train = self.library.fit_transform(features)
        val_x, val_y = validation
        val = self.library.transform(val_x)
        best = float("inf")
        from sklearn.metrics import log_loss

        for regularization in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
            candidate = LogisticRegression(
                penalty="l1",
                C=regularization,
                solver="liblinear",
                max_iter=5000,
                random_state=0,
            )
            candidate.fit(train, targets)
            score = float(log_loss(val_y, candidate.predict_proba(val), labels=candidate.classes_))
            if score < best:
                best = score
                self.model = candidate

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model has not been fitted")
        return np.asarray(self.model.predict_proba(self.library.transform(features)), dtype=np.float64)

    @property
    def operation_count(self) -> int:
        return 0 if self.model is None else int(np.count_nonzero(np.abs(self.model.coef_) > 1.0e-10))
