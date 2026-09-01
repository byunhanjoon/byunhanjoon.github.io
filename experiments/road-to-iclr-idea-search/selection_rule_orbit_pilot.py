"""Pilot fixed-recipe versus representation-wise model-selection estimands.

This is a deliberately small falsification experiment.  The semantic training
rows are split once, before any representation is rendered.  A finite
validation-only rule either selects a configuration on the identity chart and
freezes it, or is rerun within every element of the schema orbit.  Test labels
are used only after both predictive orbits have been produced.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from cross_model_orbit_gate import load_dataset, official_subsample
from orbit_anova import brier, decompose, risk_summary


HERE = Path(__file__).resolve().parent
FACTOR_NAMES = ("feature", "category", "class")


def candidate_specs(family: str) -> list[dict[str, object]]:
    if family == "ordinal_forest":
        return [
            {"min_samples_leaf": 1, "max_features": "sqrt"},
            {"min_samples_leaf": 3, "max_features": "sqrt"},
            {"min_samples_leaf": 10, "max_features": "sqrt"},
            {"min_samples_leaf": 3, "max_features": 1.0},
        ]
    if family == "native_histgb":
        return [
            {"min_samples_leaf": 5, "l2_regularization": 0.0, "max_leaf_nodes": 31},
            {"min_samples_leaf": 15, "l2_regularization": 1.0, "max_leaf_nodes": 31},
            {"min_samples_leaf": 30, "l2_regularization": 1.0, "max_leaf_nodes": 31},
            {"min_samples_leaf": 15, "l2_regularization": 5.0, "max_leaf_nodes": 15},
        ]
    if family == "catboost_native":
        return [
            {"depth": 4, "l2_leaf_reg": 1.0},
            {"depth": 6, "l2_leaf_reg": 3.0},
            {"depth": 8, "l2_leaf_reg": 3.0},
            {"depth": 6, "l2_leaf_reg": 10.0},
        ]
    raise ValueError(family)


def fit_predict(
    family: str,
    spec: dict[str, object],
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    categorical_indices: tuple[int, ...],
    random_state: int,
) -> np.ndarray:
    if family == "ordinal_forest":
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=120,
                n_jobs=4,
                random_state=random_state,
                **spec,
            ),
        )
    elif family == "native_histgb":
        model = HistGradientBoostingClassifier(
            categorical_features=list(categorical_indices),
            learning_rate=0.08,
            max_iter=120,
            random_state=random_state,
            **spec,
        )
    elif family == "catboost_native":
        from catboost import CatBoostClassifier

        train_frame = pd.DataFrame(train_x)
        query_frame = pd.DataFrame(query_x)
        for column in categorical_indices:
            train_frame[column] = train_frame[column].astype(int).astype(str)
            query_frame[column] = query_frame[column].astype(int).astype(str)
        model = CatBoostClassifier(
            iterations=120,
            learning_rate=0.08,
            loss_function=("Logloss" if len(np.unique(train_y)) == 2 else "MultiClass"),
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
            thread_count=4,
            **spec,
        )
        model.fit(train_frame, train_y, cat_features=list(categorical_indices))
        return model.predict_proba(query_frame)
    else:
        raise ValueError(family)
    model.fit(train_x, train_y)
    return model.predict_proba(query_x)


def entropy(counts: Counter[int], total: int) -> float:
    probabilities = np.asarray([count / total for count in counts.values()])
    return float(-np.sum(probabilities * np.log2(probabilities)))


def paired_row_bootstrap(
    frozen: np.ndarray,
    selected: np.ndarray,
    y: np.ndarray,
    draws: int = 5_000,
) -> dict[str, object]:
    """Uncertainty over evaluation rows, conditional on the fitted orbit."""
    frozen_flat = frozen.reshape((-1,) + frozen.shape[-2:])
    selected_flat = selected.reshape((-1,) + selected.shape[-2:])
    frozen_centroid = frozen_flat.mean(axis=0)
    selected_centroid = selected_flat.mean(axis=0)
    frozen_row_risk = np.mean(
        np.sum((frozen_flat - frozen_centroid[None, ...]) ** 2, axis=-1), axis=0
    )
    selected_row_risk = np.mean(
        np.sum((selected_flat - selected_centroid[None, ...]) ** 2, axis=-1), axis=0
    )
    targets = np.eye(frozen.shape[-1])[y]
    frozen_row_loss = np.sum((frozen_centroid - targets) ** 2, axis=-1)
    selected_row_loss = np.sum((selected_centroid - targets) ** 2, axis=-1)
    rng = np.random.default_rng(814_203)
    indices = rng.integers(0, len(y), size=(draws, len(y)))
    frozen_boot = frozen_row_risk[indices].mean(axis=1)
    selected_boot = selected_row_risk[indices].mean(axis=1)
    risk_difference = selected_boot - frozen_boot
    risk_ratio = np.divide(
        selected_boot,
        frozen_boot,
        out=np.full_like(selected_boot, np.nan),
        where=frozen_boot > 0,
    )
    loss_difference = (
        selected_row_loss[indices] - frozen_row_loss[indices]
    ).mean(axis=1)

    def interval(values: np.ndarray) -> list[float]:
        return [float(value) for value in np.quantile(values, (0.025, 0.975))]

    return {
        "resampling_unit": "test_query_row",
        "conditional_on_fitted_models": True,
        "draws": draws,
        "schema_risk_difference_ci95": interval(risk_difference),
        "schema_risk_ratio_ci95": interval(risk_ratio[np.isfinite(risk_ratio)]),
        "orbit_mean_brier_difference_ci95": interval(loss_difference),
    }


def switch_decomposition(
    frozen: np.ndarray,
    selected: np.ndarray,
) -> dict[str, float]:
    """Exact Hilbert-variance change induced by configuration switching."""
    axes = tuple(range(len(FACTOR_NAMES)))
    frozen_centered = frozen - frozen.mean(axis=axes, keepdims=True)
    switch = selected - frozen
    switch_centered = switch - switch.mean(axis=axes, keepdims=True)
    frozen_risk = float(np.mean(np.sum(frozen_centered**2, axis=-1)))
    switch_dispersion = float(np.mean(np.sum(switch_centered**2, axis=-1)))
    twice_cross_covariance = float(
        2 * np.mean(np.sum(frozen_centered * switch_centered, axis=-1))
    )
    selected_risk = float(
        np.mean(
            np.sum(
                (selected - selected.mean(axis=axes, keepdims=True)) ** 2,
                axis=-1,
            )
        )
    )
    return {
        "frozen_schema_risk": frozen_risk,
        "selected_schema_risk": selected_risk,
        "schema_risk_change": selected_risk - frozen_risk,
        "switch_dispersion": switch_dispersion,
        "twice_frozen_switch_cross_covariance": twice_cross_covariance,
        "change_reconstruction_error": abs(
            selected_risk
            - frozen_risk
            - switch_dispersion
            - twice_cross_covariance
        ),
    }


def selection_margin_diagnostic(losses: np.ndarray) -> dict[str, object]:
    """Sufficient identity-winner stability certificate over a finite orbit."""
    identity = losses[(0,) * (losses.ndim - 1)]
    winner = int(np.argmin(identity))
    competitors = np.asarray([index for index in range(len(identity)) if index != winner])
    identity_gaps = identity[competitors] - identity[winner]
    orbit_gaps = losses[..., competitors] - losses[..., winner, None]
    gamma = float(identity_gaps.min())
    delta = float(np.max(np.abs(orbit_gaps - identity_gaps)))
    ordered = np.sort(losses, axis=-1)
    local_margins = ordered[..., 1] - ordered[..., 0]
    return {
        "identity_winner": winner,
        "identity_minimum_competitor_gap": gamma,
        "maximum_relative_gap_shift": delta,
        "sufficient_certificate_holds": bool(delta < gamma),
        "certificate_ratio_delta_over_gamma": delta / gamma if gamma else None,
        "minimum_identity_winner_gap_over_orbit": float(orbit_gaps.min()),
        "minimum_local_selection_margin": float(local_margins.min()),
        "median_local_selection_margin": float(np.median(local_margins)),
    }


def build_orbit(
    x: np.ndarray,
    y: np.ndarray,
    original_categorical: tuple[int, ...],
    original_cardinalities: dict[int, int],
    n_feature: int,
    n_category: int,
    n_class: int,
) -> tuple[list[np.ndarray], list[dict[int, np.ndarray]], list[np.ndarray]]:
    feature_rng = np.random.default_rng(91_773)
    feature_permutations = [np.arange(x.shape[1])] + [
        feature_rng.permutation(x.shape[1]) for _ in range(n_feature - 1)
    ]
    category_rng = np.random.default_rng(821_571)
    category_maps = []
    for category_level in range(n_category if original_categorical else 1):
        mappings = {}
        for column in original_categorical:
            size = original_cardinalities[column]
            mappings[column] = (
                np.arange(size)
                if category_level == 0
                else category_rng.permutation(size)
            )
        category_maps.append(mappings)
    class_count = int(y.max() + 1)
    target_count = min(n_class, math.factorial(class_count))
    class_rng = np.random.default_rng(430_921)
    label_permutations = [np.arange(class_count)]
    while len(label_permutations) < target_count:
        candidate = class_rng.permutation(class_count)
        if not any(np.array_equal(candidate, old) for old in label_permutations):
            label_permutations.append(candidate)
    return feature_permutations, category_maps, label_permutations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("experiments/road-to-iclr-day-01/data/adult"),
    )
    parser.add_argument(
        "--family",
        choices=("ordinal_forest", "native_histgb", "catboost_native"),
        default="ordinal_forest",
    )
    parser.add_argument("--train-size", type=int, default=3_000)
    parser.add_argument("--test-size", type=int, default=1_000)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--n-feature", type=int, default=4)
    parser.add_argument("--n-category", type=int, default=4)
    parser.add_argument("--n-class", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=77)
    parser.add_argument("--split-seed", type=int, default=20_260_826)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prediction-output", type=Path)
    args = parser.parse_args()

    x, y, original_categorical, original_cardinalities = load_dataset(args.data_root)
    train_x, test_x, train_y, test_y = official_subsample(
        args.data_root, x, y, args.train_size, args.test_size
    )
    fit_x, validation_x, fit_y, validation_y = train_test_split(
        train_x,
        train_y,
        test_size=args.validation_fraction,
        random_state=args.split_seed,
        stratify=train_y,
    )
    feature_permutations, category_maps, label_permutations = build_orbit(
        x,
        y,
        original_categorical,
        original_cardinalities,
        args.n_feature,
        args.n_category,
        args.n_class,
    )
    specs = candidate_specs(args.family)
    shape = (
        len(feature_permutations),
        len(category_maps),
        len(label_permutations),
    )
    validation_losses = np.empty(shape + (len(specs),), dtype=np.float64)
    selected_configs = np.empty(shape, dtype=int)
    selected_predictions = np.empty(shape + (len(test_y), int(y.max() + 1)))
    rendered = {}

    for index in itertools.product(*(range(size) for size in shape)):
        feature_index, category_index, class_index = index
        feature_permutation = feature_permutations[feature_index]
        mappings = category_maps[category_index]
        label_permutation = label_permutations[class_index]
        transformed_categorical = tuple(
            new_index
            for new_index, old_index in enumerate(feature_permutation)
            if old_index in original_categorical
        )

        def render(values: np.ndarray) -> np.ndarray:
            transformed = values.copy()
            for column, mapping in mappings.items():
                transformed[:, column] = mapping[values[:, column].astype(int)]
            return transformed[:, feature_permutation]

        orbit_fit_x = render(fit_x)
        orbit_validation_x = render(validation_x)
        orbit_test_x = render(test_x)
        orbit_fit_y = label_permutation[fit_y]
        orbit_validation_y = label_permutation[validation_y]
        for config_index, spec in enumerate(specs):
            validation_probabilities = fit_predict(
                args.family,
                spec,
                orbit_fit_x,
                orbit_fit_y,
                orbit_validation_x,
                transformed_categorical,
                args.random_state,
            )
            validation_losses[index + (config_index,)] = brier(
                orbit_validation_y, validation_probabilities
            )
        selected = int(np.argmin(validation_losses[index]))
        selected_configs[index] = selected
        rendered[index] = (
            render(train_x),
            label_permutation[train_y],
            orbit_test_x,
            transformed_categorical,
            label_permutation,
        )
        full_x, full_y, query_x, categorical_indices, alignment = rendered[index]
        raw = fit_predict(
            args.family,
            specs[selected],
            full_x,
            full_y,
            query_x,
            categorical_indices,
            args.random_state,
        )
        selected_predictions[index] = raw[:, alignment]

    frozen_config = int(selected_configs[(0, 0, 0)])
    orbit_average_config = int(
        np.argmin(validation_losses.mean(axis=tuple(range(len(FACTOR_NAMES)))))
    )
    development_slices = (
        slice(0, min(2, shape[0])),
        slice(0, min(2, shape[1])),
        slice(None),
    )
    evaluation_slices = (
        slice(min(2, shape[0]), shape[0]),
        slice(min(2, shape[1]), shape[1]),
        slice(None),
    )
    if any(length == 0 for length in (
        shape[0] - min(2, shape[0]),
        shape[1] - min(2, shape[1]),
    )):
        development_pooled_config = orbit_average_config
        evaluation_slices = tuple(slice(None) for _ in shape)
        heldout_nuisance_levels = False
    else:
        development_losses = validation_losses[
            development_slices + (slice(None),)
        ]
        development_pooled_config = int(
            np.argmin(development_losses.mean(axis=tuple(range(len(FACTOR_NAMES)))))
        )
        heldout_nuisance_levels = True
    frozen_predictions = np.empty_like(selected_predictions)
    orbit_average_predictions = np.empty_like(selected_predictions)
    development_pooled_predictions = np.empty_like(selected_predictions)
    for index in itertools.product(*(range(size) for size in shape)):
        full_x, full_y, query_x, categorical_indices, alignment = rendered[index]
        raw = fit_predict(
            args.family,
            specs[frozen_config],
            full_x,
            full_y,
            query_x,
            categorical_indices,
            args.random_state,
        )
        frozen_predictions[index] = raw[:, alignment]
        orbit_average_raw = fit_predict(
            args.family,
            specs[orbit_average_config],
            full_x,
            full_y,
            query_x,
            categorical_indices,
            args.random_state,
        )
        orbit_average_predictions[index] = orbit_average_raw[:, alignment]
        development_raw = fit_predict(
            args.family,
            specs[development_pooled_config],
            full_x,
            full_y,
            query_x,
            categorical_indices,
            args.random_state,
        )
        development_pooled_predictions[index] = development_raw[:, alignment]

    counts = Counter(int(value) for value in selected_configs.flat)
    selected_summary = risk_summary(selected_predictions, test_y, FACTOR_NAMES)
    frozen_summary = risk_summary(frozen_predictions, test_y, FACTOR_NAMES)
    orbit_average_summary = risk_summary(
        orbit_average_predictions, test_y, FACTOR_NAMES
    )
    development_pooled_summary = risk_summary(
        development_pooled_predictions, test_y, FACTOR_NAMES
    )
    heldout_selected_summary = risk_summary(
        selected_predictions[evaluation_slices], test_y, FACTOR_NAMES
    )
    heldout_frozen_summary = risk_summary(
        frozen_predictions[evaluation_slices], test_y, FACTOR_NAMES
    )
    heldout_development_pooled_summary = risk_summary(
        development_pooled_predictions[evaluation_slices], test_y, FACTOR_NAMES
    )
    selected_risk = float(selected_summary["anova"]["total"])
    frozen_risk = float(frozen_summary["anova"]["total"])
    result = {
        "status": "exploratory_estimand_falsification_pilot",
        "dataset": args.data_root.name,
        "family": args.family,
        "fit_size": len(fit_y),
        "validation_size": len(validation_y),
        "test_size": len(test_y),
        "split_seed": args.split_seed,
        "factor_shape": list(shape),
        "candidate_specs": specs,
        "selection_protocol": {
            "split_before_rendering": True,
            "validation_labels_used": True,
            "test_labels_used_for_selection": False,
            "same_candidates_and_random_seed_in_every_representation": True,
            "canonical_identity_selected_config": frozen_config,
            "orbit_average_validation_selected_config": orbit_average_config,
            "development_pooled_selected_config": development_pooled_config,
        },
        "selected_config_grid": selected_configs.tolist(),
        "selection_counts": {str(key): value for key, value in sorted(counts.items())},
        "selection_entropy_bits": entropy(counts, selected_configs.size),
        "fraction_different_from_identity_config": float(
            np.mean(selected_configs != frozen_config)
        ),
        "selection_decision_anova": decompose(
            np.eye(len(specs))[selected_configs][..., None, :], FACTOR_NAMES
        ),
        "mean_validation_brier_by_config": validation_losses.mean(axis=(0, 1, 2)).tolist(),
        "selection_margin_diagnostic": selection_margin_diagnostic(
            validation_losses
        ),
        "frozen_identity_selection_orbit": frozen_summary,
        "representation_wise_selection_orbit": selected_summary,
        "orbit_average_validation_selection_orbit": orbit_average_summary,
        "development_pooled_selection_orbit": development_pooled_summary,
        "heldout_nuisance_evaluation": {
            "has_disjoint_feature_and_category_levels": heldout_nuisance_levels,
            "development_slices": [
                [item.start, item.stop] for item in development_slices
            ],
            "evaluation_slices": [
                [item.start, item.stop] for item in evaluation_slices
            ],
            "representation_wise": heldout_selected_summary,
            "identity_frozen": heldout_frozen_summary,
            "development_pooled": heldout_development_pooled_summary,
            "development_pooled_minus_representation_wise_schema_risk": (
                float(heldout_development_pooled_summary["anova"]["total"])
                - float(heldout_selected_summary["anova"]["total"])
            ),
            "development_pooled_minus_representation_wise_brier": (
                float(heldout_development_pooled_summary["orbit_mean_brier"])
                - float(heldout_selected_summary["orbit_mean_brier"])
            ),
        },
        "selection_to_frozen_schema_risk_ratio": (
            selected_risk / frozen_risk if frozen_risk else None
        ),
        "selection_minus_frozen_schema_risk": selected_risk - frozen_risk,
        "selection_minus_frozen_orbit_mean_brier": (
            float(selected_summary["orbit_mean_brier"])
            - float(frozen_summary["orbit_mean_brier"])
        ),
        "paired_row_bootstrap": paired_row_bootstrap(
            frozen_predictions, selected_predictions, test_y
        ),
        "configuration_switch_decomposition": switch_decomposition(
            frozen_predictions, selected_predictions
        ),
        "orbit_average_minus_representation_wise_schema_risk": (
            float(orbit_average_summary["anova"]["total"]) - selected_risk
        ),
        "orbit_average_minus_representation_wise_orbit_mean_brier": (
            float(orbit_average_summary["orbit_mean_brier"])
            - float(selected_summary["orbit_mean_brier"])
        ),
    }
    output = args.output or HERE / f"selection_rule_{args.data_root.name}_{args.family}.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    if args.prediction_output is not None:
        np.savez_compressed(
            args.prediction_output,
            frozen_predictions=frozen_predictions,
            selected_predictions=selected_predictions,
            orbit_average_predictions=orbit_average_predictions,
            development_pooled_predictions=development_pooled_predictions,
            selected_configs=selected_configs,
            validation_losses=validation_losses,
            test_y=test_y,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
