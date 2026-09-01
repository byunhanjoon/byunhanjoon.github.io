"""Validation-only SafeGram alpha selection and its preregistered ablations."""

from __future__ import annotations

from typing import Any

import numpy as np

from .common import EPS, mix_predictions, normalized_excess_risk, task_error, trivial_prediction


def bootstrap_normalized_cost(
    problem_type: str,
    y: np.ndarray,
    y_train: np.ndarray,
    raw: np.ndarray,
    method: np.ndarray,
    *,
    resamples: int,
    seed: int,
    epsilon: float,
) -> np.ndarray:
    """Row bootstrap of the exact normalized cost statistic.

    Regression resamples squared residuals and applies the RMSE square root
    inside every replicate, rather than incorrectly bootstrapping RMSE values.
    """

    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(y), size=(int(resamples), len(y)))
    output = np.empty(int(resamples), dtype=float)
    trivial = trivial_prediction(problem_type, y_train, len(y))
    for index, rows in enumerate(indices):
        raw_loss = task_error(problem_type, y[rows], np.asarray(raw)[rows])
        method_loss = task_error(problem_type, y[rows], np.asarray(method)[rows])
        trivial_loss = task_error(problem_type, y[rows], np.asarray(trivial)[rows])
        output[index] = (method_loss - raw_loss) / max(trivial_loss - raw_loss, epsilon)
    return output


def alpha_evidence(
    problem_type: str,
    y_validation: np.ndarray,
    y_train: np.ndarray,
    raw_validation: np.ndarray,
    invariant_validation: np.ndarray,
    *,
    alphas: list[float],
    bootstrap_resamples: int,
    seed: int,
    epsilon: float = 1e-8,
) -> list[dict[str, Any]]:
    records = []
    for alpha in alphas:
        prediction = mix_predictions(raw_validation, invariant_validation, alpha)
        metrics = normalized_excess_risk(
            problem_type, y_validation, raw_validation, prediction, y_train, epsilon
        )
        samples = bootstrap_normalized_cost(
            problem_type,
            y_validation,
            y_train,
            raw_validation,
            prediction,
            resamples=bootstrap_resamples,
            seed=seed,
            epsilon=epsilon,
        )
        records.append(
            {
                "alpha": float(alpha),
                **metrics,
                "bootstrap_mean_C": float(np.mean(samples)),
                "bootstrap_se_C": float(np.std(samples, ddof=1)),
                "bootstrap_ucb95_C": float(np.quantile(samples, 0.95)),
                "bootstrap_lcb95_C": float(np.quantile(samples, 0.05)),
                "bootstrap_resamples": int(bootstrap_resamples),
            }
        )
    return records


def _largest(records: list[dict[str, Any]], predicate: Any) -> float:
    passing = [float(record["alpha"]) for record in records if predicate(record)]
    return max(passing, default=0.0)


def select_gates(
    records: list[dict[str, Any]],
    *,
    taus: list[float],
    constrained_lambda_multipliers: list[float],
) -> dict[str, float]:
    result: dict[str, float] = {}
    labels = {0.0: "t0", 0.005: "t005", 0.01: "t01", 0.02: "t02"}
    for tau in taus:
        suffix = labels.get(float(tau), f"t{str(tau).replace('.', '')}")
        key = f"SafeGram-{suffix}"
        result[key] = _largest(records, lambda row, threshold=tau: row["bootstrap_ucb95_C"] <= threshold)
    result["G1-point-t01"] = _largest(records, lambda row: row["normalized_excess_risk"] <= 0.01)
    result["G2-oneSE-t01"] = _largest(
        records, lambda row: row["normalized_excess_risk"] + row["bootstrap_se_C"] <= 0.01
    )
    minimum_loss = min(float(row["method_loss"]) for row in records)
    result["G3-validation-min"] = max(
        float(row["alpha"]) for row in records if abs(float(row["method_loss"]) - minimum_loss) <= 1e-12
    )
    trivial_loss = float(records[0]["trivial_loss"])
    for multiplier in constrained_lambda_multipliers:
        scored = [
            (
                float(row["method_loss"]) + float(multiplier) * trivial_loss * (1.0 - float(row["alpha"])),
                -float(row["alpha"]),
                float(row["alpha"]),
            )
            for row in records
        ]
        result[f"G4-constrained-{multiplier:g}"] = min(scored)[2]
    return result


def verify_alpha_zero(records: list[dict[str, Any]]) -> None:
    zero = next(record for record in records if float(record["alpha"]) == 0.0)
    for key in ("normalized_excess_risk", "bootstrap_mean_C", "bootstrap_se_C", "bootstrap_ucb95_C"):
        if abs(float(zero[key])) > 1e-12:
            raise RuntimeError(f"alpha=0 safety identity failed for {key}: {zero[key]}")
