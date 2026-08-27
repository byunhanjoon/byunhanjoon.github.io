"""Exact balanced product-factor ANOVA for aligned predictive probabilities."""

from __future__ import annotations

import itertools

import numpy as np


def brier(y: np.ndarray, probabilities: np.ndarray) -> float:
    targets = np.eye(probabilities.shape[-1])[y]
    return float(np.mean(np.sum((probabilities - targets) ** 2, axis=-1)))


def log_loss(y: np.ndarray, probabilities: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    selected = probabilities[np.arange(len(y)), y]
    return float(-np.mean(np.log(np.clip(selected, np.finfo(float).tiny, 1.0))))


def log_orbit_summary(
    predictions: np.ndarray,
    factor_count: int,
    y: np.ndarray | None = None,
) -> dict[str, float]:
    """Label-free log-loss gap using the left KL/Bregman centroid.

    The centroid is the normalized geometric mean of aligned probabilities.
    For probabilities in the simplex interior,

    ``E_g logloss(y, p_g) - logloss(y, q) = E_g KL(q || p_g)``.

    The right side is label-free.  This is established Bregman-information
    machinery; unlike squared variance, it does not use orthogonal fANOVA.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.ndim != factor_count + 2:
        raise ValueError("Prediction rank does not match the factor count")
    flat = predictions.reshape((-1,) + predictions.shape[-2:])
    log_probabilities = np.log(
        np.clip(flat, np.finfo(float).tiny, 1.0)
    )
    mean_log = log_probabilities.mean(axis=0)
    maximum = mean_log.max(axis=-1, keepdims=True)
    log_normalizer = maximum + np.log(
        np.exp(mean_log - maximum).sum(axis=-1, keepdims=True)
    )
    centroid = np.exp(mean_log - log_normalizer)
    diversity = float(
        np.mean(
            np.sum(
                centroid[None, ...]
                * (
                    (mean_log - log_normalizer)[None, ...]
                    - log_probabilities
                ),
                axis=-1,
            )
        )
    )
    output = {
        "label_free_gap": diversity,
        "mean_log_normalizer_gap": float(-log_normalizer.mean()),
    }
    if y is not None:
        mean_member_loss = float(np.mean([log_loss(y, member) for member in flat]))
        centroid_loss = log_loss(y, centroid)
        output.update(
            {
                "mean_member_log_loss": mean_member_loss,
                "centroid_log_loss": centroid_loss,
                "risk_identity_absolute_error": abs(
                    mean_member_loss - centroid_loss - diversity
                ),
            }
        )
    return output


def decompose(
    predictions: np.ndarray,
    factor_names: tuple[str, ...],
) -> dict[str, float]:
    """Return orthogonal variance components for a balanced factor grid.

    ``predictions`` has shape ``factor_1 x ... x factor_k x rows x classes``.
    Components use the uniform product measure over the supplied factor levels.
    """
    factor_count = len(factor_names)
    if predictions.ndim != factor_count + 2:
        raise ValueError("Prediction rank does not match the factor count")
    axes = tuple(range(factor_count))
    grand = predictions.mean(axis=axes, keepdims=True)
    effects: dict[tuple[int, ...], np.ndarray] = {(): grand}
    components: dict[str, float] = {}
    for order in range(1, factor_count + 1):
        for subset in itertools.combinations(range(factor_count), order):
            complement = tuple(axis for axis in axes if axis not in subset)
            conditional = (
                predictions.mean(axis=complement, keepdims=True)
                if complement
                else predictions
            )
            effect = conditional.copy()
            for lower_order in range(order):
                for proper_subset in itertools.combinations(subset, lower_order):
                    effect -= effects[proper_subset]
            effects[subset] = effect
            name = ":".join(factor_names[index] for index in subset)
            components[name] = float(np.mean(np.sum(effect**2, axis=-1)))

    total = float(np.mean(np.sum((predictions - grand) ** 2, axis=-1)))
    reconstruction = np.zeros_like(predictions)
    for effect in effects.values():
        reconstruction += effect
    return {
        "total": total,
        **components,
        "component_sum_error": abs(total - sum(components.values())),
        "prediction_reconstruction_max_error": float(
            np.max(np.abs(predictions - reconstruction))
        ),
    }


def sample_variance(predictions: np.ndarray) -> float:
    """Unbiased Hilbert-valued variance across the leading sample axis.

    The final two axes are evaluation rows and output coordinates, matching
    :func:`decompose`.  Any axes between the sample and final two axes are not
    supported deliberately: Monte Carlo samples should already be flattened.
    """
    if predictions.ndim != 3:
        raise ValueError("Predictions must have shape samples x rows x outputs")
    if predictions.shape[0] < 2:
        raise ValueError("At least two samples are required")
    centered = predictions - predictions.mean(axis=0, keepdims=True)
    return float(
        np.sum(centered**2, axis=0).sum(axis=-1).mean()
        / (predictions.shape[0] - 1)
    )


def _mean_squared_distance(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or left.ndim != 3:
        raise ValueError("Prediction pairs must share samples x rows x outputs shape")
    return float(np.mean(np.sum((left - right) ** 2, axis=-1)))


def pick_freeze(
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    hybrids: np.ndarray,
    factor_names: tuple[str, ...],
) -> dict[str, object]:
    """Estimate total variance and Sobol main/total effects in ``O(KN)``.

    ``A`` and ``B`` are independent product-measure samples.  ``hybrids[j]``
    uses every coordinate from ``A`` except factor ``j``, which comes from
    ``B``.  The estimators are the vector-output pick-freeze identities

    ``T_j = 1/2 E ||f(A) - f(A_-j, B_j)||^2`` and
    ``V_j = V - 1/2 E ||f(B) - f(A_-j, B_j)||^2``.

    ``T_j`` is the sum of all ANOVA components containing factor ``j``;
    ``V_j`` is its first-order component.  Finite-sample main effects can be
    slightly negative, so raw estimates are returned without clipping.
    """
    if predictions_a.shape != predictions_b.shape or predictions_a.ndim != 3:
        raise ValueError("A and B must share samples x rows x outputs shape")
    if hybrids.shape != (len(factor_names),) + predictions_a.shape:
        raise ValueError("Hybrids must have shape factors x samples x rows x outputs")

    variance = sample_variance(np.concatenate((predictions_a, predictions_b), axis=0))
    effects: dict[str, dict[str, float]] = {}
    for factor_index, factor_name in enumerate(factor_names):
        hybrid = hybrids[factor_index]
        total = 0.5 * _mean_squared_distance(predictions_a, hybrid)
        main = variance - 0.5 * _mean_squared_distance(predictions_b, hybrid)
        effects[factor_name] = {
            "first_order": main,
            "total_effect": total,
            "first_order_fraction": main / variance if variance else np.nan,
            "total_effect_fraction": total / variance if variance else np.nan,
        }
    return {"total": variance, "effects": effects}


def symmetrization_frontier(
    predictions: np.ndarray,
    factor_names: tuple[str, ...],
) -> list[dict[str, object]]:
    """Enumerate exact factor-marginalization costs and residual squared risk.

    Averaging a set ``J`` of product factors deletes exactly the fANOVA
    components whose index sets intersect ``J``.  The number of full-factorial
    members needed for the wrapper is the product of selected level counts.
    This is group symmetrization only when the selected levels form a closed
    action orbit; an arbitrary finite chart menu instead requires a canonical
    semantic renderer.
    """
    factor_count = len(factor_names)
    if predictions.ndim != factor_count + 2:
        raise ValueError("Prediction rank does not match the factor count")
    axes = tuple(range(factor_count))
    grand = predictions.mean(axis=axes, keepdims=True)
    total = float(np.mean(np.sum((predictions - grand) ** 2, axis=-1)))
    frontier = []
    for subset_size in range(factor_count + 1):
        for subset in itertools.combinations(range(factor_count), subset_size):
            symmetrized = (
                predictions.mean(axis=subset, keepdims=True)
                if subset
                else predictions
            )
            residual = float(
                np.mean(np.sum((symmetrized - grand) ** 2, axis=-1))
            )
            cost = int(np.prod([predictions.shape[index] for index in subset]))
            frontier.append(
                {
                    "factors": [factor_names[index] for index in subset],
                    "full_orbit_cost": cost,
                    "removed_risk": total - residual,
                    "residual_risk": residual,
                    "removed_fraction": (total - residual) / total if total else np.nan,
                }
            )
    return frontier


def budgeted_marginalization_frontier(
    predictions: np.ndarray,
    factor_names: tuple[str, ...],
    budget: int,
) -> list[dict[str, object]]:
    """Expected residual for exact factor averaging plus iid complement draws.

    For a selected factor set ``J``, one conditional Monte Carlo draw averages
    all ``c_J`` combinations of those factors while drawing the complement
    once. With ``m = floor(budget / c_J)`` independent complement draws, the
    average has expected squared schema risk ``SR(Q_J p) / m``. The empty set
    recovers ordinary iid schema averaging. A remainder smaller than ``c_J``
    is not used, so callers should report realized cost as well as the budget.

    This is a Rao--Blackwellized Monte Carlo frontier. It is group
    symmetrization only when the chosen transformations form a group action.
    """
    if budget < 1:
        raise ValueError("budget must be a positive integer")
    exact_frontier = symmetrization_frontier(predictions, factor_names)
    total = float(exact_frontier[0]["residual_risk"])
    frontier = []
    for item in exact_frontier:
        batch_cost = int(item["full_orbit_cost"])
        repeats = budget // batch_cost
        if repeats < 1:
            continue
        expected_residual = float(item["residual_risk"]) / repeats
        expected_removed = total - expected_residual
        frontier.append(
            {
                **item,
                "budget": budget,
                "conditional_draws": repeats,
                "realized_cost": repeats * batch_cost,
                "unused_budget": budget - repeats * batch_cost,
                "expected_residual_risk": expected_residual,
                "expected_removed_risk": expected_removed,
                "expected_removed_fraction": (
                    expected_removed / total if total else np.nan
                ),
            }
        )
    return frontier


def risk_summary(
    predictions: np.ndarray,
    y: np.ndarray,
    factor_names: tuple[str, ...],
) -> dict[str, object]:
    # Several libraries emit float32 probabilities.  The Brier identity
    # subtracts nearby risks, so promote before any reductions.
    predictions = np.asarray(predictions, dtype=np.float64)
    factor_axes = tuple(range(len(factor_names)))
    orbit_mean = predictions.mean(axis=factor_axes)
    flat = predictions.reshape((-1,) + predictions.shape[-2:])
    risks = np.asarray([brier(y, member) for member in flat])
    mean_member = float(risks.mean())
    mean_risk = brier(y, orbit_mean)
    decomposition = decompose(predictions, factor_names)
    row_schema_risk = np.mean(
        np.sum((flat - orbit_mean[None, ...]) ** 2, axis=-1), axis=0
    )
    hard_predictions = np.argmax(flat, axis=-1)
    modal_fractions = np.max(
        np.stack(
            [np.mean(hard_predictions == label, axis=0) for label in range(flat.shape[-1])]
        ),
        axis=0,
    )
    probability_ranges = np.max(
        np.max(flat, axis=0) - np.min(flat, axis=0), axis=-1
    )
    return {
        "reference_brier": brier(y, predictions[(0,) * len(factor_names)]),
        "mean_member_brier": mean_member,
        "best_member_brier": float(risks.min()),
        "worst_member_brier": float(risks.max()),
        "member_brier_std": float(risks.std()),
        "orbit_mean_brier": mean_risk,
        "brier_reduction_by_averaging": mean_member - mean_risk,
        "risk_identity_absolute_error": abs(
            mean_member - mean_risk - float(decomposition["total"])
        ),
        "instance_audit": {
            "hard_label_flip_fraction": float(np.mean(modal_fractions < 1.0)),
            "mean_nonmodal_vote_fraction": float(np.mean(1.0 - modal_fractions)),
            "row_schema_risk_median": float(np.median(row_schema_risk)),
            "row_schema_risk_p95": float(np.quantile(row_schema_risk, 0.95)),
            "row_schema_risk_max": float(row_schema_risk.max()),
            "max_class_probability_range_mean": float(probability_ranges.mean()),
            "max_class_probability_range_p95": float(
                np.quantile(probability_ranges, 0.95)
            ),
            "max_class_probability_range_max": float(probability_ranges.max()),
        },
        "anova": decomposition,
    }
