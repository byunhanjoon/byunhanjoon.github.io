"""Cheap raw/rank experts and context-only descriptors for the E3 kill test."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from src.representations import TieAwareECDF, marginal_descriptors, robust_affine
from src.transforms import MonotonePWLTransform


def context_gate_descriptor(context_x: np.ndarray, context_y: np.ndarray) -> np.ndarray:
    """Return fixed-width summaries without query rows, labels, or regime metadata."""
    rank = TieAwareECDF().fit(context_x).transform(context_x)
    y = np.asarray(context_y, dtype=np.float64)
    y_centered = y - y.mean()
    correlations = []
    for column in rank.T:
        centered = column - column.mean()
        denominator = np.sqrt(np.sum(centered**2) * np.sum(y_centered**2))
        correlations.append(0.0 if denominator <= 1e-12 else float(centered @ y_centered / denominator))
    correlation = np.asarray(correlations)
    stable = np.r_[
        np.sort(correlation),
        np.sort(np.abs(correlation)),
        y.mean(), y.std(), np.quantile(y, [0.1, 0.5, 0.9]),
        np.log1p(context_x.shape[0]), np.log1p(context_x.shape[1]),
    ]
    return np.r_[stable, marginal_descriptors(context_x)]


def featurewise_pooled_gate_descriptor(context_x: np.ndarray, context_y: np.ndarray) -> np.ndarray:
    """Pool aligned per-feature shape/association blocks without depending on column order."""
    x = np.asarray(context_x, dtype=np.float64)
    y = np.asarray(context_y, dtype=np.float64)
    rank = TieAwareECDF().fit(x).transform(x)
    y_centered = y - y.mean()
    probabilities = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    blocks = []
    for feature, column in enumerate(x.T):
        finite = column[np.isfinite(column)]
        median = np.median(finite)
        q25, q75 = np.quantile(finite, [0.25, 0.75])
        iqr = max(float(q75 - q25), 1e-12)
        standardized = (finite - median) / iqr
        quantiles = np.quantile(standardized, probabilities)
        centered = standardized - standardized.mean()
        sd = max(float(standardized.std()), 1e-12)
        rank_centered = rank[:, feature] - rank[:, feature].mean()
        denominator = np.sqrt(np.sum(rank_centered**2) * np.sum(y_centered**2))
        association = 0.0 if denominator <= 1e-12 else float(rank_centered @ y_centered / denominator)
        blocks.append(np.r_[
            association, abs(association), quantiles, np.diff(quantiles),
            np.mean((centered / sd) ** 3), np.mean((centered / sd) ** 4) - 3,
            np.log(iqr), np.unique(finite).size / finite.size, 1 - finite.size / column.size,
        ])
    feature_matrix = np.vstack(blocks)
    pooled = np.r_[
        feature_matrix.mean(axis=0), feature_matrix.std(axis=0),
        feature_matrix.min(axis=0), feature_matrix.max(axis=0),
    ]
    # These interaction pools retain whether the predictive features, specifically,
    # have a given marginal geometry. The computation is still column-permutation invariant.
    shape = feature_matrix[:, 2:]
    weight = feature_matrix[:, 1]
    weighted_shape = (shape * weight[:, None]).sum(axis=0) / max(weight.sum(), 1e-12)
    if x.shape[1] > 1 and np.std(weight) > 1e-12:
        alignment = np.asarray([
            np.corrcoef(weight, shape[:, index])[0, 1]
            if np.std(shape[:, index]) > 1e-12 else 0.0
            for index in range(shape.shape[1])
        ])
    else:
        alignment = np.zeros(shape.shape[1])
    global_values = np.r_[
        y.mean(), y.std(), np.quantile(y, [0.1, 0.5, 0.9]),
        np.log1p(x.shape[0]), np.log1p(x.shape[1]),
    ]
    result = np.r_[pooled, weighted_shape, alignment, global_values]
    if not np.all(np.isfinite(result)):
        raise AssertionError("featurewise pooled descriptor is nonfinite")
    return result


def _classification_prediction(model: KNeighborsClassifier, query_x: np.ndarray) -> np.ndarray:
    probability = model.predict_proba(query_x)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(query_x.shape[0], dtype=np.float64)
    if 0 not in classes:
        return np.ones(query_x.shape[0], dtype=np.float64)
    return np.asarray(probability[:, classes.index(1)], dtype=np.float64)


def _fit_predict(
    task_type: str,
    context_x: np.ndarray,
    context_y: np.ndarray,
    query_x: np.ndarray,
    neighbors: int,
    metric_p: int,
) -> np.ndarray:
    if task_type == "classification":
        model = KNeighborsClassifier(n_neighbors=neighbors, weights="distance", p=metric_p)
        model.fit(context_x, context_y)
        return _classification_prediction(model, query_x)
    model = KNeighborsRegressor(n_neighbors=neighbors, weights="distance", p=metric_p)
    model.fit(context_x, context_y)
    return np.asarray(model.predict(query_x), dtype=np.float64)


def fit_knn_views(
    episode,
    neighbors: int,
    metric_p: int,
    augmentation_seed: int,
    augmentation_knots: int,
    augmentation_slope_sigma: float,
) -> dict[str, np.ndarray]:
    """Fit M0/M1/M2 and one extra transformed M3 view on the same episode."""
    scaler = StandardScaler().fit(episode.context_x)
    raw_context = scaler.transform(episode.context_x)
    raw_query = scaler.transform(episode.query_x)

    robust_context = robust_affine(episode.context_x, episode.context_x)
    robust_query = robust_affine(episode.context_x, episode.query_x)

    ranker = TieAwareECDF().fit(episode.context_x)
    rank_context = ranker.transform(episode.context_x)
    rank_query = ranker.transform(episode.query_x)

    augmented_context = np.empty_like(episode.context_x)
    augmented_query = np.empty_like(episode.query_x)
    for feature in range(episode.context_x.shape[1]):
        transform = MonotonePWLTransform(
            seed=augmentation_seed + feature,
            n_knots=augmentation_knots,
            slope_sigma=augmentation_slope_sigma,
        ).fit(episode.context_x[:, feature])
        augmented_context[:, feature] = transform.transform(episode.context_x[:, feature])
        augmented_query[:, feature] = transform.transform(episode.query_x[:, feature])
    augmented_scaler = StandardScaler().fit(augmented_context)
    augmented_context = augmented_scaler.transform(augmented_context)
    augmented_query = augmented_scaler.transform(augmented_query)

    raw = _fit_predict(task_type=episode.metadata["task_type"], context_x=raw_context,
                       context_y=episode.context_y, query_x=raw_query,
                       neighbors=neighbors, metric_p=metric_p)
    robust = _fit_predict(task_type=episode.metadata["task_type"], context_x=robust_context,
                          context_y=episode.context_y, query_x=robust_query,
                          neighbors=neighbors, metric_p=metric_p)
    rank = _fit_predict(task_type=episode.metadata["task_type"], context_x=rank_context,
                        context_y=episode.context_y, query_x=rank_query,
                        neighbors=neighbors, metric_p=metric_p)
    transformed = _fit_predict(task_type=episode.metadata["task_type"], context_x=augmented_context,
                               context_y=episode.context_y, query_x=augmented_query,
                               neighbors=neighbors, metric_p=metric_p)
    return {"raw": raw, "robust": robust, "rank": rank, "augmentation": 0.5 * (raw + transformed)}


def episode_loss(y: np.ndarray, prediction: np.ndarray, task_type: str, clip: float) -> float:
    y = np.asarray(y)
    prediction = np.asarray(prediction, dtype=np.float64)
    if task_type == "classification":
        p = np.clip(prediction, clip, 1 - clip)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return float(np.mean((y - prediction) ** 2))


def mixture_loss_curve(
    y: np.ndarray,
    raw: np.ndarray,
    rank: np.ndarray,
    task_type: str,
    alphas: np.ndarray,
    clip: float,
) -> np.ndarray:
    return np.asarray([
        episode_loss(y, alpha * raw + (1 - alpha) * rank, task_type, clip)
        for alpha in alphas
    ])
