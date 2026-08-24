"""Nested discovery of exact-value main effects and pure pair interactions.

The smooth baseline is supplied as a numerical design matrix (typically raw
quantile-normalized values plus PLE). Candidate exact-state corrections are fit
only on residuals from the fitting partition and evaluated on held-out rows.
Pair effects are estimated after backfitting both singleton main effects, so a
pair is credited only for utility beyond an additive exact-value explanation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Iterable, Literal

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold


Task = Literal["binclass", "regression"]


@dataclass(frozen=True)
class DiscoveryConfig:
    folds: int = 5
    smoothing: float = 20.0
    max_cardinality: int = 128
    max_pair_cardinality: int = 512
    minimum_relative_gain: float = 5e-4
    minimum_fold_wins: int = 5
    minimum_pair_fold_wins: int = 5
    maximum_singletons: int = 4
    maximum_pairs: int = 4
    representation_budget: int = 512
    backfit_iterations: int = 4


@dataclass(frozen=True)
class CandidateScore:
    kind: Literal["singleton", "pair"]
    columns: tuple[int, ...]
    cardinality: int
    relative_gain: float
    incremental_gain: float
    fold_wins: int
    incremental_fold_wins: int


@dataclass(frozen=True)
class Selection:
    singletons: tuple[int, ...]
    pairs: tuple[tuple[int, int], ...]


@dataclass
class Lookup:
    keys: np.ndarray
    means: np.ndarray

    def predict(self, values: np.ndarray) -> np.ndarray:
        query = state_keys(values)
        positions = np.searchsorted(self.keys, query)
        valid = positions < len(self.keys)
        matched = np.zeros(len(query), dtype=bool)
        matched[valid] = self.keys[positions[valid]] == query[valid]
        output = np.zeros(len(query), dtype=np.float64)
        output[matched] = self.means[positions[matched]]
        return output


def state_keys(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    return contiguous.view(
        np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))
    ).ravel()


def fit_lookup(values: np.ndarray, residual: np.ndarray, smoothing: float) -> Lookup:
    keys, inverse, counts = np.unique(
        state_keys(values), return_inverse=True, return_counts=True
    )
    sums = np.bincount(inverse, weights=residual)
    return Lookup(keys=keys, means=sums / (counts + smoothing))


def loss(task: Task, prediction: np.ndarray, target: np.ndarray) -> float:
    if task == "binclass":
        probability = np.clip(prediction, 1e-6, 1.0 - 1e-6)
        return float(
            np.mean(
                -target * np.log(probability)
                - (1.0 - target) * np.log(1.0 - probability)
            )
        )
    return float(np.mean((prediction - target) ** 2))


def fit_smooth(design: np.ndarray, target: np.ndarray, task: Task):
    if task == "binclass":
        model = LogisticRegression(C=1.0, max_iter=1_000)
    else:
        model = Ridge(alpha=10.0)
    model.fit(design, target)
    return model


def predict_smooth(model, design: np.ndarray, task: Task) -> np.ndarray:
    if task == "binclass":
        return model.predict_proba(design)[:, 1].astype(np.float64)
    return model.predict(design).astype(np.float64)


def split_indices(
    target: np.ndarray, task: Task, folds: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = (
        StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        if task == "binclass"
        else KFold(n_splits=folds, shuffle=True, random_state=seed)
    )
    return list(splitter.split(target, target if task == "binclass" else None))


def candidate_columns(numeric: np.ndarray, max_cardinality: int) -> tuple[int, ...]:
    return tuple(
        column
        for column in range(numeric.shape[1])
        if 1 < len(np.unique(numeric[:, column])) <= max_cardinality
    )


def fit_additive_pair(
    values: np.ndarray,
    residual: np.ndarray,
    smoothing: float,
    iterations: int,
) -> tuple[Lookup, Lookup, np.ndarray]:
    """Backfit two singleton maps and return their in-sample prediction."""

    left_prediction = np.zeros(len(residual), dtype=np.float64)
    right_prediction = np.zeros(len(residual), dtype=np.float64)
    left_lookup = fit_lookup(values[:, 0], residual, smoothing)
    right_lookup = fit_lookup(values[:, 1], residual, smoothing)
    for _ in range(iterations):
        left_lookup = fit_lookup(
            values[:, 0], residual - right_prediction, smoothing
        )
        left_prediction = left_lookup.predict(values[:, 0])
        right_lookup = fit_lookup(
            values[:, 1], residual - left_prediction, smoothing
        )
        right_prediction = right_lookup.predict(values[:, 1])
    return left_lookup, right_lookup, left_prediction + right_prediction


def fit_pure_interaction(
    values: np.ndarray,
    residual_after_additive: np.ndarray,
    smoothing: float,
    iterations: int,
) -> Lookup:
    """Fit a joint lookup and project out both weighted singleton marginals."""

    raw = fit_lookup(values, residual_after_additive, smoothing)
    centered_prediction = raw.predict(values)
    for _ in range(iterations):
        left_margin = fit_lookup(values[:, 0], centered_prediction, 0.0)
        centered_prediction -= left_margin.predict(values[:, 0])
        right_margin = fit_lookup(values[:, 1], centered_prediction, 0.0)
        centered_prediction -= right_margin.predict(values[:, 1])
    # The centered prediction is constant within a joint cell. Reconstructing
    # with zero smoothing preserves that projection for unseen-query lookup.
    return fit_lookup(values, centered_prediction, 0.0)


def _relative_gain(before: float, after: float) -> float:
    return (before - after) / max(before, 1e-12)


def discover(
    design: np.ndarray,
    numeric: np.ndarray,
    target: np.ndarray,
    task: Task,
    seed: int,
    config: DiscoveryConfig,
) -> tuple[Selection, list[CandidateScore], np.ndarray]:
    """Cross-validated structure discovery using training data only."""

    if config.minimum_fold_wins > config.folds:
        raise ValueError(
            "minimum_fold_wins cannot exceed the number of discovery folds"
        )
    if config.minimum_pair_fold_wins > config.folds:
        raise ValueError(
            "minimum_pair_fold_wins cannot exceed the number of discovery folds"
        )

    candidates = candidate_columns(numeric, config.max_cardinality)
    pairs = tuple(
        pair
        for pair in combinations(candidates, 2)
        if len(np.unique(state_keys(numeric[:, pair])))
        <= config.max_pair_cardinality
    )
    base_prediction = np.zeros(len(target), dtype=np.float64)
    singleton_prediction = {
        column: np.zeros(len(target), dtype=np.float64) for column in candidates
    }
    pair_additive_prediction = {
        pair: np.zeros(len(target), dtype=np.float64) for pair in pairs
    }
    pair_prediction = {
        pair: np.zeros(len(target), dtype=np.float64) for pair in pairs
    }
    singleton_fold_wins = {column: 0 for column in candidates}
    pair_fold_wins = {pair: 0 for pair in pairs}
    incremental_fold_wins = {pair: 0 for pair in pairs}

    for fit_index, holdout_index in split_indices(
        target, task, config.folds, seed
    ):
        model = fit_smooth(design[fit_index], target[fit_index], task)
        fit_prediction = predict_smooth(model, design[fit_index], task)
        holdout_prediction = predict_smooth(model, design[holdout_index], task)
        base_prediction[holdout_index] = holdout_prediction
        fit_residual = target[fit_index] - fit_prediction
        before = loss(task, holdout_prediction, target[holdout_index])

        fold_singletons: dict[int, tuple[Lookup, np.ndarray]] = {}
        for column in candidates:
            lookup = fit_lookup(
                numeric[fit_index, column], fit_residual, config.smoothing
            )
            correction = lookup.predict(numeric[holdout_index, column])
            singleton_prediction[column][holdout_index] = correction
            fold_singletons[column] = (lookup, correction)
            singleton_fold_wins[column] += int(
                loss(task, holdout_prediction + correction, target[holdout_index])
                < before
            )

        for pair in pairs:
            fit_values = numeric[fit_index][:, pair]
            holdout_values = numeric[holdout_index][:, pair]
            left, right, fit_additive = fit_additive_pair(
                fit_values,
                fit_residual,
                config.smoothing,
                config.backfit_iterations,
            )
            holdout_additive = left.predict(holdout_values[:, 0]) + right.predict(
                holdout_values[:, 1]
            )
            interaction = fit_pure_interaction(
                fit_values,
                fit_residual - fit_additive,
                config.smoothing,
                config.backfit_iterations,
            )
            holdout_interaction = interaction.predict(holdout_values)
            pair_additive_prediction[pair][holdout_index] = holdout_additive
            pair_prediction[pair][holdout_index] = (
                holdout_additive + holdout_interaction
            )
            additive_loss = loss(
                task,
                holdout_prediction + holdout_additive,
                target[holdout_index],
            )
            joint_loss = loss(
                task,
                holdout_prediction + holdout_additive + holdout_interaction,
                target[holdout_index],
            )
            pair_fold_wins[pair] += int(joint_loss < before)
            incremental_fold_wins[pair] += int(joint_loss < additive_loss)

    baseline_loss = loss(task, base_prediction, target)
    scores: list[CandidateScore] = []
    for column in candidates:
        corrected_loss = loss(
            task, base_prediction + singleton_prediction[column], target
        )
        scores.append(
            CandidateScore(
                kind="singleton",
                columns=(column,),
                cardinality=len(np.unique(numeric[:, column])),
                relative_gain=_relative_gain(baseline_loss, corrected_loss),
                incremental_gain=_relative_gain(baseline_loss, corrected_loss),
                fold_wins=singleton_fold_wins[column],
                incremental_fold_wins=singleton_fold_wins[column],
            )
        )
    for pair in pairs:
        additive_loss = loss(
            task, base_prediction + pair_additive_prediction[pair], target
        )
        corrected_loss = loss(
            task, base_prediction + pair_prediction[pair], target
        )
        scores.append(
            CandidateScore(
                kind="pair",
                columns=pair,
                cardinality=len(np.unique(state_keys(numeric[:, pair]))),
                relative_gain=_relative_gain(baseline_loss, corrected_loss),
                incremental_gain=_relative_gain(additive_loss, corrected_loss),
                fold_wins=pair_fold_wins[pair],
                incremental_fold_wins=incremental_fold_wins[pair],
            )
        )

    eligible_singletons = sorted(
        (
            score
            for score in scores
            if score.kind == "singleton"
            and score.relative_gain >= config.minimum_relative_gain
            and score.fold_wins >= config.minimum_fold_wins
        ),
        key=lambda score: score.relative_gain,
        reverse=True,
    )[: config.maximum_singletons]
    eligible_pairs = sorted(
        (
            score
            for score in scores
            if score.kind == "pair"
            and score.relative_gain >= config.minimum_relative_gain
            and score.incremental_gain >= config.minimum_relative_gain
            and score.fold_wins >= config.minimum_pair_fold_wins
            and score.incremental_fold_wins >= config.minimum_pair_fold_wins
        ),
        key=lambda score: score.incremental_gain,
        reverse=True,
    )[: config.maximum_pairs]
    ranked_terms = sorted(
        [*eligible_singletons, *eligible_pairs],
        key=lambda score: (
            (
                score.relative_gain
                if score.kind == "singleton"
                else score.incremental_gain
            )
            / max(np.log1p(score.cardinality), 1e-12)
        ),
        reverse=True,
    )
    kept: list[CandidateScore] = []
    used_states = 0
    for score in ranked_terms:
        if used_states + score.cardinality <= config.representation_budget:
            kept.append(score)
            used_states += score.cardinality
    eligible_singletons = [score for score in kept if score.kind == "singleton"]
    eligible_pairs = [score for score in kept if score.kind == "pair"]
    return (
        Selection(
            singletons=tuple(score.columns[0] for score in eligible_singletons),
            pairs=tuple(
                (score.columns[0], score.columns[1]) for score in eligible_pairs
            ),
        ),
        scores,
        base_prediction,
    )


def fit_selected_correction(
    numeric: np.ndarray,
    residual: np.ndarray,
    selection: Selection,
    config: DiscoveryConfig,
) -> tuple[list[tuple[int, Lookup]], list[tuple[tuple[int, int], Lookup]]]:
    """Fit selected hierarchical corrections on one training partition."""

    singleton_models: list[tuple[int, Lookup]] = []
    fitted = np.zeros(len(residual), dtype=np.float64)
    # Coordinate descent lets several selected singleton effects share residuals.
    singleton_predictions = {
        column: np.zeros(len(residual), dtype=np.float64)
        for column in selection.singletons
    }
    lookup_by_column: dict[int, Lookup] = {}
    for _ in range(config.backfit_iterations):
        for column in selection.singletons:
            other = fitted - singleton_predictions[column]
            lookup = fit_lookup(
                numeric[:, column], residual - other, config.smoothing
            )
            new_prediction = lookup.predict(numeric[:, column])
            fitted += new_prediction - singleton_predictions[column]
            singleton_predictions[column] = new_prediction
            lookup_by_column[column] = lookup
    singleton_models = [
        (column, lookup_by_column[column]) for column in selection.singletons
    ]

    pair_models: list[tuple[tuple[int, int], Lookup]] = []
    for pair in selection.pairs:
        lookup = fit_pure_interaction(
            numeric[:, pair],
            residual - fitted,
            config.smoothing,
            config.backfit_iterations,
        )
        fitted += lookup.predict(numeric[:, pair])
        pair_models.append((pair, lookup))
    return singleton_models, pair_models


def predict_selected_correction(
    numeric: np.ndarray,
    singleton_models: Iterable[tuple[int, Lookup]],
    pair_models: Iterable[tuple[tuple[int, int], Lookup]],
) -> np.ndarray:
    output = np.zeros(len(numeric), dtype=np.float64)
    for column, lookup in singleton_models:
        output += lookup.predict(numeric[:, column])
    for pair, lookup in pair_models:
        output += lookup.predict(numeric[:, pair])
    return output


def nonnested_selected_gain(
    design: np.ndarray,
    numeric: np.ndarray,
    target: np.ndarray,
    task: Task,
    seed: int,
    selection: Selection,
    config: DiscoveryConfig,
) -> float:
    """OOF gain after choosing ``selection`` on those same OOF outcomes.

    This intentionally reuses the discovery folds and is therefore an optimism
    diagnostic, not an unbiased performance estimate. Compare it only with the
    outer result from :func:`nested_audit`.
    """

    baseline = np.zeros(len(target), dtype=np.float64)
    corrected = np.zeros(len(target), dtype=np.float64)
    for fit_index, holdout_index in split_indices(target, task, config.folds, seed):
        smooth = fit_smooth(design[fit_index], target[fit_index], task)
        fit_prediction = predict_smooth(smooth, design[fit_index], task)
        holdout_prediction = predict_smooth(smooth, design[holdout_index], task)
        singleton_models, pair_models = fit_selected_correction(
            numeric[fit_index],
            target[fit_index] - fit_prediction,
            selection,
            config,
        )
        correction = predict_selected_correction(
            numeric[holdout_index], singleton_models, pair_models
        )
        baseline[holdout_index] = holdout_prediction
        corrected[holdout_index] = holdout_prediction + correction
    return _relative_gain(
        loss(task, baseline, target), loss(task, corrected, target)
    )


def nested_audit(
    design: np.ndarray,
    numeric: np.ndarray,
    target: np.ndarray,
    task: Task,
    seed: int,
    config: DiscoveryConfig,
    outer_folds: int = 5,
    inner_folds: int = 5,
) -> list[dict[str, object]]:
    """Evaluate the full discovery-and-correction procedure on outer folds."""

    rows: list[dict[str, object]] = []
    outer = split_indices(target, task, outer_folds, seed + 50_000)
    inner_config = DiscoveryConfig(
        **{**asdict(config), "folds": inner_folds}
    )
    for outer_fold, (fit_index, holdout_index) in enumerate(outer):
        selection, scores, _ = discover(
            design[fit_index],
            numeric[fit_index],
            target[fit_index],
            task,
            seed + 1_000 * outer_fold,
            inner_config,
        )
        base = fit_smooth(design[fit_index], target[fit_index], task)
        fit_prediction = predict_smooth(base, design[fit_index], task)
        holdout_prediction = predict_smooth(base, design[holdout_index], task)
        singleton_models, pair_models = fit_selected_correction(
            numeric[fit_index],
            target[fit_index] - fit_prediction,
            selection,
            config,
        )
        correction = predict_selected_correction(
            numeric[holdout_index], singleton_models, pair_models
        )
        before = loss(task, holdout_prediction, target[holdout_index])
        after = loss(
            task, holdout_prediction + correction, target[holdout_index]
        )
        rows.append(
            {
                "outer_fold": outer_fold,
                "fit_rows": len(fit_index),
                "holdout_rows": len(holdout_index),
                "singletons": ";".join(map(str, selection.singletons)),
                "pairs": ";".join(f"{a}+{b}" for a, b in selection.pairs),
                "candidate_count": len(scores),
                "baseline_loss": before,
                "corrected_loss": after,
                "relative_gain": _relative_gain(before, after),
                "win": int(after < before),
            }
        )
    return rows
