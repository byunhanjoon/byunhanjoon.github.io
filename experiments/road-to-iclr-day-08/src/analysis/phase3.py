"""Cross-dataset descriptor analysis for mechanism discovery."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


DESCRIPTOR_COLUMNS = [
    "missing_rate",
    "unique_fraction",
    "largest_atom_mass",
    "binned_entropy",
    "zero_mass",
    "skewness",
    "excess_kurtosis",
    "robust_scale_iqr",
    "tail_heaviness_q99_q01_over_iqr",
    "spacing_irregularity",
]


def aggregate_descriptors(features: pd.DataFrame) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {}
    for column in DESCRIPTOR_COLUMNS:
        aggregations[f"mean_{column}"] = (column, "mean")
        aggregations[f"max_{column}"] = (column, "max")
    return features.groupby(["dataset", "split_seed"], as_index=False).agg(**aggregations)


def cross_dataset_meta_models(
    table: pd.DataFrame,
    target: str,
    *,
    seed: int = 20260831,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_columns = [column for column in table if column.startswith(("mean_", "max_"))]
    usable = table.replace([np.inf, -np.inf], np.nan).dropna(subset=[target, "dataset"])
    X, y = usable[feature_columns], usable[target].to_numpy(dtype=np.float64)
    groups = usable["dataset"].to_numpy()
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("cross-dataset validation requires at least two datasets")
    cv = GroupKFold(n_splits=n_splits)
    models: dict[str, Any] = {
        "ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=seed, n_jobs=8),
        ),
    }
    predictions = usable[["dataset", "split_seed"]].copy()
    predictions["target"] = y
    baseline_prediction = np.empty_like(y)
    for train_indices, test_indices in cv.split(X, y, groups):
        baseline_prediction[test_indices] = float(np.mean(y[train_indices]))
    baseline_mae = float(mean_absolute_error(y, baseline_prediction))
    predictions["prediction_fold_mean_baseline"] = baseline_prediction
    metrics = []
    importances = []
    for name, model in models.items():
        prediction = cross_val_predict(model, X, y, groups=groups, cv=cv, n_jobs=1)
        predictions[f"prediction_{name}"] = prediction
        metrics.append(
            {
                "meta_model": name,
                "target": target,
                "datasets": len(np.unique(groups)),
                "rows": len(usable),
                "group_folds": n_splits,
                "mae": float(mean_absolute_error(y, prediction)),
                "fold_mean_baseline_mae": baseline_mae,
                "mae_improvement_fraction": (
                    float(1.0 - mean_absolute_error(y, prediction) / baseline_mae)
                    if baseline_mae > 0
                    else np.nan
                ),
                "r2": float(r2_score(y, prediction)),
                "target_std": float(np.std(y)),
            }
        )
        model.fit(X, y)
        fitted = model[-1]
        values = fitted.coef_ if name == "ridge" else fitted.feature_importances_
        for feature, value in zip(feature_columns, np.asarray(values).reshape(-1)):
            importances.append({"meta_model": name, "target": target, "feature": feature, "importance": float(value)})
    return pd.DataFrame(metrics), pd.concat([predictions, pd.DataFrame(importances)], axis=0, ignore_index=True, sort=False)
