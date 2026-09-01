"""Frozen GuardedGram G1/G2/G3 validation-only selection rules."""

from __future__ import annotations

from typing import Any

import numpy as np

from safe_basis.common import mix_predictions, normalized_excess_risk
from safe_basis.gating import bootstrap_normalized_cost


def guarded_evidence(
    problem_type: str,
    y_validation: np.ndarray,
    y_train: np.ndarray,
    raw_validation: np.ndarray,
    gram_validation: np.ndarray,
    *,
    alphas: list[float],
    resamples: int,
    seed: int,
    epsilon: float = 1e-8,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, alpha in enumerate(alphas):
        prediction = mix_predictions(raw_validation, gram_validation, alpha)
        metrics = normalized_excess_risk(
            problem_type, y_validation, raw_validation, prediction, y_train, epsilon
        )
        samples = bootstrap_normalized_cost(
            problem_type,
            y_validation,
            y_train,
            raw_validation,
            prediction,
            resamples=resamples,
            seed=int(seed) + index * 1009,
            epsilon=epsilon,
        )
        records.append(
            {
                "alpha": float(alpha),
                **metrics,
                "bootstrap_mean_C": float(np.mean(samples)),
                "bootstrap_se_C": float(np.std(samples, ddof=1)),
                "bootstrap_q10_C": float(np.quantile(samples, 0.10)),
                "bootstrap_q20_C": float(np.quantile(samples, 0.20)),
                "bootstrap_q80_C": float(np.quantile(samples, 0.80)),
                "bootstrap_q90_C": float(np.quantile(samples, 0.90)),
                "bootstrap_samples": samples,
                "bootstrap_resamples": int(resamples),
            }
        )
    return records


def _record(evidence: list[dict[str, Any]], alpha: float) -> dict[str, Any]:
    return next(row for row in evidence if np.isclose(float(row["alpha"]), float(alpha)))


def centered_harm_p_value(record: dict[str, Any], tau: float) -> float:
    estimate = float(record["normalized_excess_risk"])
    if estimate <= float(tau):
        return 1.0
    samples = np.asarray(record["bootstrap_samples"], dtype=float)
    centered = samples - estimate
    observed_excess = estimate - float(tau)
    return float((1 + np.sum(centered >= observed_excess)) / (len(centered) + 1))


def select_g1(
    evidence: list[dict[str, Any]], *, tau: float, significance: float = 0.05
) -> tuple[float, list[dict[str, float]]]:
    decisions = []
    for alpha in (0.75, 0.5, 0.25, 0.0):
        record = _record(evidence, alpha)
        p_value = centered_harm_p_value(record, tau)
        reject_for_harm = bool(p_value < significance)
        decisions.append(
            {
                "alpha": alpha,
                "C_hat": float(record["normalized_excess_risk"]),
                "tau": float(tau),
                "p_harm": p_value,
                "reject_for_harm": reject_for_harm,
            }
        )
        if not reject_for_harm:
            return float(alpha), decisions
    return 0.0, decisions


def select_g2(evidence: list[dict[str, Any]], *, tau: float, gamma: float) -> float:
    for alpha in (0.75, 0.5, 0.25, 0.0):
        record = _record(evidence, alpha)
        guarded = float(record["normalized_excess_risk"]) + float(gamma) * float(
            record["bootstrap_se_C"]
        )
        if guarded <= float(tau):
            return float(alpha)
    return 0.0


def select_g3(evidence: list[dict[str, Any]], *, tau: float = 0.01) -> tuple[float, str]:
    primary = _record(evidence, 0.75)
    if float(primary["bootstrap_q80_C"]) <= tau:
        return 0.75, "clearly_safe"
    if float(primary["bootstrap_q20_C"]) <= tau:
        return 0.5, "ambiguous_default"
    for alpha in (0.5, 0.25):
        record = _record(evidence, alpha)
        if float(record["bootstrap_q80_C"]) <= tau:
            return float(alpha), "unsafe_recursive_safe"
        if float(record["bootstrap_q20_C"]) <= tau:
            return float(alpha), "unsafe_recursive_ambiguous"
    return 0.0, "unsafe_fallback"


def strip_samples(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in row.items() if key != "bootstrap_samples"} for row in evidence]

