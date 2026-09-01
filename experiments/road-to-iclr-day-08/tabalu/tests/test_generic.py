from __future__ import annotations

import numpy as np

from tabalu.models.generic import SparseExactClassifier, SparseExactRegressor


def test_sparse_exact_general_models_fit_small_tasks() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(100, 3))
    reg_y = x[:, 0] * x[:, 1] + x[:, 2] ** 2
    reg = SparseExactRegressor()
    reg.fit(x[:70], reg_y[:70], (x[70:85], reg_y[70:85]))
    assert np.mean((reg.predict(x[85:]) - reg_y[85:]) ** 2) < 1.0e-10
    cls_y = (x[:, 0] * x[:, 1] > 0).astype(int)
    cls = SparseExactClassifier()
    cls.fit(x[:70], cls_y[:70], (x[70:85], cls_y[70:85]))
    assert cls.predict_proba(x[85:]).shape == (15, 2)
