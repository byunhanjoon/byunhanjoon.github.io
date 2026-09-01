"""Performance, calibration, and paired prediction-disagreement metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from scipy.stats import spearmanr


def _probabilities(prediction: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(prediction, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError(f"classification predictions must be 2D, received {probabilities.shape}")
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def expected_calibration_error(y: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            result += mask.mean() * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(result)


def classification_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    probabilities = _probabilities(prediction)
    labels = np.arange(probabilities.shape[1])
    one_hot = np.eye(probabilities.shape[1])[np.asarray(y, dtype=int)]
    try:
        auc = (
            roc_auc_score(y, probabilities[:, 1])
            if probabilities.shape[1] == 2
            else roc_auc_score(y, probabilities, multi_class="ovr", average="macro")
        )
    except ValueError:
        auc = None
    return {
        "loss": float(log_loss(y, probabilities, labels=labels)),
        "accuracy": float(accuracy_score(y, probabilities.argmax(axis=1))),
        "auc": None if auc is None else float(auc),
        "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "ece": expected_calibration_error(np.asarray(y), probabilities),
    }


def regression_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(y, dtype=np.float64)
    values = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if values.shape != target.shape:
        raise ValueError(f"prediction shape {values.shape} does not match target {target.shape}")
    return {
        "loss": float(mean_squared_error(target, values)),
        "rmse": float(np.sqrt(mean_squared_error(target, values))),
        "mae": float(mean_absolute_error(target, values)),
        "r2": float(r2_score(target, values)),
    }


def disagreement_metrics(
    clean: np.ndarray,
    candidate: np.ndarray,
    problem_type: str,
    *,
    normalization_scale: float | None = None,
) -> dict[str, float]:
    if problem_type == "regression":
        left = np.asarray(clean, dtype=np.float64).reshape(-1)
        right = np.asarray(candidate, dtype=np.float64).reshape(-1)
        scale = max(float(normalization_scale or np.std(left)), 1e-12)
        correlation = spearmanr(left, right).statistic if len(left) > 1 else 1.0
        return {
            "normalized_absolute_disagreement": float(np.mean(np.abs(left - right)) / scale),
            "prediction_spearman": float(correlation if np.isfinite(correlation) else 0.0),
        }
    left, right = _probabilities(clean), _probabilities(candidate)
    midpoint = 0.5 * (left + right)
    js = 0.5 * np.sum(left * np.log(left / midpoint), axis=1)
    js += 0.5 * np.sum(right * np.log(right / midpoint), axis=1)
    return {
        "js_divergence": float(js.mean()),
        "total_variation": float((0.5 * np.abs(left - right).sum(axis=1)).mean()),
        "argmax_flip_rate": float(np.mean(left.argmax(axis=1) != right.argmax(axis=1))),
    }
