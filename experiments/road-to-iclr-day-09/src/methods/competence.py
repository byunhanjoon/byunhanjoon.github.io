"""Frozen six-expert panel and context-only competence routing."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


EXPERTS = ("linear", "additive", "threshold", "interaction", "partition", "periodic")


def _periodic_features(x: np.ndarray) -> np.ndarray:
    frequencies = (0.75, 1.5, 2.5, 4.0)
    return np.concatenate(
        [function(frequency * x) for frequency in frequencies for function in (np.sin, np.cos)],
        axis=1,
    )


def _model(name: str, task_type: str, seed: int):
    classification = task_type == "classification"
    if name == "linear":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=300) if classification else Ridge(alpha=1.0),
        )
    if name == "additive":
        return make_pipeline(
            SplineTransformer(n_knots=4, degree=2, include_bias=False),
            StandardScaler(),
            LogisticRegression(C=0.5, max_iter=300) if classification else Ridge(alpha=2.0),
        )
    if name == "threshold":
        cls = RandomForestClassifier if classification else RandomForestRegressor
        return cls(n_estimators=40, min_samples_leaf=3, max_features="sqrt", random_state=seed, n_jobs=1)
    if name == "interaction":
        return make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            StandardScaler(),
            LogisticRegression(C=0.25, max_iter=300) if classification else Ridge(alpha=4.0),
        )
    if name == "partition":
        cls = DecisionTreeClassifier if classification else DecisionTreeRegressor
        return cls(max_depth=3, min_samples_leaf=3, random_state=seed)
    if name == "periodic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.5, max_iter=300) if classification else Ridge(alpha=2.0),
        )
    raise KeyError(name)


def _design(name: str, x: np.ndarray) -> np.ndarray:
    return _periodic_features(x) if name == "periodic" else x


def _classification_prediction(model, x: np.ndarray, training_y: np.ndarray) -> np.ndarray:
    classes = list(model.classes_)
    if 1 not in classes:
        prediction = np.zeros(x.shape[0], dtype=float)
    elif 0 not in classes:
        prediction = np.ones(x.shape[0], dtype=float)
    else:
        prediction = np.asarray(model.predict_proba(x)[:, classes.index(1)], dtype=float)
    prior_strength = 20.0
    return (len(training_y) * prediction + prior_strength * float(np.mean(training_y))) / (
        len(training_y) + prior_strength
    )


def fit_predict_experts(
    context_x: np.ndarray,
    context_y: np.ndarray,
    query_x: np.ndarray,
    task_type: str,
    seed: int,
) -> np.ndarray:
    """Fit the frozen experts on context and return expert-by-query predictions."""
    x = np.asarray(context_x, dtype=float)
    y = np.asarray(context_y)
    query = np.asarray(query_x, dtype=float)
    if task_type == "classification" and np.unique(y).size == 1:
        return np.full((len(EXPERTS), query.shape[0]), float(y[0]))
    predictions = []
    for name in EXPERTS:
        model = _model(name, task_type, seed)
        model.fit(_design(name, x), y)
        design = _design(name, query)
        if task_type == "classification":
            prediction = _classification_prediction(model, design, y)
        else:
            prediction = np.asarray(model.predict(design), dtype=float)
        predictions.append(prediction)
    result = np.asarray(predictions)
    if result.shape != (len(EXPERTS), query.shape[0]) or not np.all(np.isfinite(result)):
        raise AssertionError("invalid expert prediction array")
    return result


def prediction_loss(y: np.ndarray, prediction: np.ndarray, task_type: str) -> float:
    if task_type == "classification":
        return float(log_loss(y, np.clip(prediction, 1e-6, 1 - 1e-6), labels=[0, 1]))
    return float(mean_squared_error(y, prediction))


def cross_validated_expert_losses(
    context_x: np.ndarray,
    context_y: np.ndarray,
    task_type: str,
    seed: int,
    folds: int = 3,
) -> np.ndarray:
    """Estimate each expert's loss using context rows and labels only."""
    x = np.asarray(context_x, dtype=float)
    y = np.asarray(context_y)
    if folds < 2 or folds > len(y):
        raise ValueError("invalid fold count")
    if task_type == "classification":
        counts = np.bincount(y.astype(int), minlength=2)
        splitter = (
            StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
            if counts.min() >= folds
            else KFold(n_splits=folds, shuffle=True, random_state=seed)
        )
        split = splitter.split(x, y) if isinstance(splitter, StratifiedKFold) else splitter.split(x)
    else:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        split = splitter.split(x)
    oof = np.empty((len(EXPERTS), len(y)), dtype=float)
    for fold_index, (train, validation) in enumerate(split):
        train_y = y[train]
        if task_type == "classification" and np.unique(train_y).size == 1:
            oof[:, validation] = float(train_y[0])
            continue
        for expert_index, name in enumerate(EXPERTS):
            model = _model(name, task_type, seed + fold_index * 100)
            model.fit(_design(name, x[train]), train_y)
            validation_x = _design(name, x[validation])
            if task_type == "classification":
                prediction = _classification_prediction(model, validation_x, train_y)
            else:
                prediction = np.asarray(model.predict(validation_x), dtype=float)
            oof[expert_index, validation] = prediction
    if not np.all(np.isfinite(oof)):
        raise AssertionError("nonfinite cross-validated predictions")
    return np.asarray([prediction_loss(y, row, task_type) for row in oof])


def competence_weights(losses: np.ndarray, temperature: float, uniform_shrinkage: float) -> np.ndarray:
    losses = np.asarray(losses, dtype=float)
    if losses.shape != (len(EXPERTS),) or not np.all(np.isfinite(losses)):
        raise ValueError("expected one finite loss per expert")
    if temperature <= 0 or not 0 <= uniform_shrinkage <= 1:
        raise ValueError("invalid routing hyperparameters")
    logits = -(losses - losses.min()) / temperature
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()
    weights = (1.0 - uniform_shrinkage) * weights + uniform_shrinkage / len(EXPERTS)
    return weights


def weighted_prediction(expert_predictions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    predictions = np.asarray(expert_predictions, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if predictions.ndim != 2 or predictions.shape[0] != len(EXPERTS):
        raise ValueError("expert predictions must be expert by query")
    if weights.shape != (len(EXPERTS),) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("invalid expert weights")
    return weights @ predictions
