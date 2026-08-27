"""One-factor-at-a-time invariance and selection audit for the Day 3 post.

The earlier product-orbit experiment deliberately mixed several nuisance
choices.  This extension separates four transformations and compares a frozen
identity-selected recipe with representation-wise validation selection.  It
also includes controls whose preprocessing should remove particular nuisances.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cross_model_orbit_gate import load_dataset, official_subsample
from orbit_anova import brier, risk_summary


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "invariance_matrix_config.json"


@dataclass(frozen=True)
class View:
    name: str
    feature_permutation: np.ndarray
    category_maps: dict[int, np.ndarray]
    class_permutation: np.ndarray
    unit_scale: np.ndarray
    unit_offset: np.ndarray


def read_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return json.loads(path.read_text())


def numerical_column_count(data_root: Path) -> int:
    path = data_root / "x_num.npy"
    return int(np.load(path, mmap_mode="r").shape[1]) if path.exists() else 0


def _identity_view(
    feature_count: int,
    class_count: int,
    numerical_count: int,
) -> View:
    return View(
        name="identity",
        feature_permutation=np.arange(feature_count),
        category_maps={},
        class_permutation=np.arange(class_count),
        unit_scale=np.ones(numerical_count),
        unit_offset=np.zeros(numerical_count),
    )


def build_views(
    factor: str,
    x: np.ndarray,
    y: np.ndarray,
    categorical_indices: tuple[int, ...],
    cardinalities: dict[int, int],
    numerical_count: int,
    nonidentity_count: int,
) -> list[View]:
    """Build a deterministic, one-factor menu whose first member is identity."""
    feature_count = x.shape[1]
    class_count = int(y.max() + 1)
    identity = _identity_view(feature_count, class_count, numerical_count)
    views = [identity]

    if factor == "feature_order":
        rng = np.random.default_rng(91_773)
        seen = {tuple(identity.feature_permutation)}
        while len(views) <= nonidentity_count:
            permutation = rng.permutation(feature_count)
            key = tuple(int(value) for value in permutation)
            if key in seen:
                continue
            seen.add(key)
            views.append(
                View(
                    name=f"feature_order_{len(views)}",
                    feature_permutation=permutation,
                    category_maps={},
                    class_permutation=identity.class_permutation,
                    unit_scale=identity.unit_scale,
                    unit_offset=identity.unit_offset,
                )
            )
    elif factor == "category_ids":
        if not categorical_indices:
            return views
        rng = np.random.default_rng(821_571)
        seen: set[tuple[tuple[int, ...], ...]] = set()
        while len(views) <= nonidentity_count:
            mappings = {
                column: rng.permutation(cardinalities[column])
                for column in categorical_indices
            }
            key = tuple(
                tuple(int(value) for value in mappings[column])
                for column in categorical_indices
            )
            if key in seen:
                continue
            seen.add(key)
            views.append(
                View(
                    name=f"category_ids_{len(views)}",
                    feature_permutation=identity.feature_permutation,
                    category_maps=mappings,
                    class_permutation=identity.class_permutation,
                    unit_scale=identity.unit_scale,
                    unit_offset=identity.unit_offset,
                )
            )
    elif factor == "class_ids":
        target = min(nonidentity_count + 1, math.factorial(class_count))
        rng = np.random.default_rng(430_921)
        seen = {tuple(identity.class_permutation)}
        while len(views) < target:
            permutation = rng.permutation(class_count)
            key = tuple(int(value) for value in permutation)
            if key in seen:
                continue
            seen.add(key)
            views.append(
                View(
                    name=f"class_ids_{len(views)}",
                    feature_permutation=identity.feature_permutation,
                    category_maps={},
                    class_permutation=permutation,
                    unit_scale=identity.unit_scale,
                    unit_offset=identity.unit_offset,
                )
            )
    elif factor == "numeric_units":
        if numerical_count == 0:
            return views
        finite_train = np.asarray(x[:, :numerical_count], dtype=np.float64)
        medians = np.nanmedian(finite_train, axis=0)
        q25 = np.nanquantile(finite_train, 0.25, axis=0)
        q75 = np.nanquantile(finite_train, 0.75, axis=0)
        spread = np.where(q75 > q25, q75 - q25, 1.0)
        rng = np.random.default_rng(612_031)
        for index in range(1, nonidentity_count + 1):
            scale = np.exp(rng.uniform(np.log(0.1), np.log(10.0), numerical_count))
            offset = scale * (medians + rng.uniform(-2.0, 2.0, numerical_count) * spread)
            views.append(
                View(
                    name=f"numeric_units_{index}",
                    feature_permutation=identity.feature_permutation,
                    category_maps={},
                    class_permutation=identity.class_permutation,
                    unit_scale=scale,
                    unit_offset=offset,
                )
            )
    else:
        raise ValueError(f"Unknown factor: {factor}")
    return views


def render_x(values: np.ndarray, view: View, numerical_count: int) -> np.ndarray:
    transformed = np.asarray(values, dtype=np.float64).copy()
    for column, mapping in view.category_maps.items():
        transformed[:, column] = mapping[values[:, column].astype(int)]
    if numerical_count:
        transformed[:, :numerical_count] = (
            transformed[:, :numerical_count] * view.unit_scale[None, :]
            + view.unit_offset[None, :]
        )
    return transformed[:, view.feature_permutation]


def transformed_categorical_indices(
    original: tuple[int, ...], view: View
) -> tuple[int, ...]:
    return tuple(
        new_index
        for new_index, old_index in enumerate(view.feature_permutation)
        if int(old_index) in original
    )


def preprocessing(
    categorical_indices: tuple[int, ...], feature_count: int
) -> ColumnTransformer:
    numerical_indices = [
        index for index in range(feature_count) if index not in categorical_indices
    ]
    return ColumnTransformer(
        [
            (
                "numerical",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                numerical_indices,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(categorical_indices),
            ),
        ],
        sparse_threshold=0.0,
    )


def fit_predict(
    family: str,
    spec: dict[str, Any],
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    categorical_indices: tuple[int, ...],
    random_state: int,
    forest_estimators: int,
    boosting_iterations: int,
) -> np.ndarray:
    if family == "onehot_logistic":
        model = make_pipeline(
            preprocessing(categorical_indices, train_x.shape[1]),
            LogisticRegression(
                C=float(spec["C"]),
                max_iter=2_000,
                tol=1e-11,
                random_state=random_state,
            ),
        )
    elif family in {"ordinal_forest_sqrt", "ordinal_forest_full"}:
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=forest_estimators,
                min_samples_leaf=int(spec["min_samples_leaf"]),
                max_features=("sqrt" if family.endswith("sqrt") else 1.0),
                n_jobs=4,
                random_state=random_state,
            ),
        )
    elif family == "onehot_forest_full":
        model = make_pipeline(
            preprocessing(categorical_indices, train_x.shape[1]),
            RandomForestClassifier(
                n_estimators=forest_estimators,
                min_samples_leaf=int(spec["min_samples_leaf"]),
                max_features=1.0,
                n_jobs=4,
                random_state=random_state,
            ),
        )
    elif family == "native_histgb":
        model = HistGradientBoostingClassifier(
            categorical_features=list(categorical_indices),
            learning_rate=0.08,
            max_iter=boosting_iterations,
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
            iterations=boosting_iterations,
            learning_rate=0.08,
            loss_function=("Logloss" if len(np.unique(train_y)) == 2 else "MultiClass"),
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
            thread_count=4,
            **spec,
        )
        model.fit(train_frame, train_y, cat_features=list(categorical_indices))
        return np.asarray(model.predict_proba(query_frame), dtype=np.float64)
    else:
        raise ValueError(family)
    model.fit(train_x, train_y)
    return np.asarray(model.predict_proba(query_x), dtype=np.float64)


def align_probabilities(raw: np.ndarray, class_permutation: np.ndarray) -> np.ndarray:
    """Map columns from rendered class IDs back to semantic class IDs."""
    return np.asarray(raw, dtype=np.float64)[:, class_permutation]


def maximum_identity_deviation(predictions: np.ndarray) -> float:
    return float(np.max(np.abs(predictions - predictions[0][None, ...])))


def candidate_validation_diagnostics(losses: np.ndarray) -> dict[str, Any]:
    ranges = np.ptp(losses, axis=0)
    return {
        "maximum_loss_range": float(ranges.max()),
        "loss_range_by_candidate": [float(value) for value in ranges],
        "mean_loss_by_candidate": [float(value) for value in losses.mean(axis=0)],
    }


def evaluate_cell(
    family: str,
    factor: str,
    views: list[View],
    specs: list[dict[str, Any]],
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_y: np.ndarray,
    test_y: np.ndarray,
    original_categorical: tuple[int, ...],
    numerical_count: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    design = config["design"]
    fit_x, validation_x, fit_y, validation_y = train_test_split(
        train_x,
        train_y,
        test_size=float(design["validation_fraction"]),
        random_state=int(design["split_seed"]),
        stratify=train_y,
    )
    validation_losses = np.empty((len(views), len(specs)), dtype=np.float64)
    rendered: list[tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...], np.ndarray]] = []

    for view_index, view in enumerate(views):
        categorical = transformed_categorical_indices(original_categorical, view)
        orbit_fit_x = render_x(fit_x, view, numerical_count)
        orbit_validation_x = render_x(validation_x, view, numerical_count)
        orbit_fit_y = view.class_permutation[fit_y]
        orbit_validation_y = view.class_permutation[validation_y]
        for config_index, spec in enumerate(specs):
            raw = fit_predict(
                family,
                spec,
                orbit_fit_x,
                orbit_fit_y,
                orbit_validation_x,
                categorical,
                int(design["fit_seed"]),
                int(design["forest_estimators"]),
                int(design["boosting_iterations"]),
            )
            validation_losses[view_index, config_index] = brier(
                validation_y,
                align_probabilities(raw, view.class_permutation),
            )
        rendered.append(
            (
                render_x(train_x, view, numerical_count),
                view.class_permutation[train_y],
                render_x(test_x, view, numerical_count),
                categorical,
                view.class_permutation,
            )
        )

    selected_configs = np.argmin(validation_losses, axis=1)
    frozen_config = int(selected_configs[0])
    frozen_predictions = np.empty((len(views), len(test_y), int(train_y.max() + 1)))
    selected_predictions = np.empty_like(frozen_predictions)
    per_candidate_predictions = np.empty(
        (len(specs), len(views), len(test_y), int(train_y.max() + 1)),
        dtype=np.float64,
    )

    for view_index, rendered_view in enumerate(rendered):
        full_x, full_y, query_x, categorical, alignment = rendered_view
        for config_index, spec in enumerate(specs):
            raw = fit_predict(
                family,
                spec,
                full_x,
                full_y,
                query_x,
                categorical,
                int(design["fit_seed"]),
                int(design["forest_estimators"]),
                int(design["boosting_iterations"]),
            )
            per_candidate_predictions[config_index, view_index] = align_probabilities(
                raw, alignment
            )
        frozen_predictions[view_index] = per_candidate_predictions[
            frozen_config, view_index
        ]
        selected_predictions[view_index] = per_candidate_predictions[
            selected_configs[view_index], view_index
        ]

    candidate_deviations = [
        maximum_identity_deviation(predictions)
        for predictions in per_candidate_predictions
    ]
    tolerance = float(design["invariance_tolerance"])
    frozen_summary = risk_summary(frozen_predictions, test_y, (factor,))
    selected_summary = risk_summary(selected_predictions, test_y, (factor,))
    return {
        "applicable": len(views) > 1,
        "view_names": [view.name for view in views],
        "view_count": len(views),
        "expected_behavior": config["expected_behavior"][family][factor],
        "candidate_specs": specs,
        "validation": candidate_validation_diagnostics(validation_losses),
        "identity_selected_config": frozen_config,
        "selected_configs": [int(value) for value in selected_configs],
        "selection_switch_fraction": float(np.mean(selected_configs != frozen_config)),
        "candidate_max_probability_deviation": candidate_deviations,
        "all_candidates_within_tolerance": bool(
            max(candidate_deviations, default=0.0) <= tolerance
        ),
        "frozen": {
            "max_probability_deviation_from_identity": maximum_identity_deviation(
                frozen_predictions
            ),
            "within_tolerance": bool(
                maximum_identity_deviation(frozen_predictions) <= tolerance
            ),
            "summary": frozen_summary,
        },
        "representation_wise_selected": {
            "max_probability_deviation_from_identity": maximum_identity_deviation(
                selected_predictions
            ),
            "within_tolerance": bool(
                maximum_identity_deviation(selected_predictions) <= tolerance
            ),
            "summary": selected_summary,
        },
        "selection_minus_frozen_schema_risk": float(
            selected_summary["anova"]["total"] - frozen_summary["anova"]["total"]
        ),
        "selection_minus_frozen_orbit_mean_brier": float(
            selected_summary["orbit_mean_brier"] - frozen_summary["orbit_mean_brier"]
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = read_config(args.config)
    design = config["design"]
    selected_datasets = args.datasets or config["datasets"]
    selected_families = args.families or config["families"]
    selected_factors = args.factors or config["factors"]
    output: dict[str, Any] = {
        "status": "completed_day3_invariance_matrix_extension",
        "config": config,
        "cells": {},
    }

    for dataset in selected_datasets:
        data_root = args.data_root / dataset
        x, y, categorical, cardinalities = load_dataset(data_root)
        numerical_count = numerical_column_count(data_root)
        train_x, test_x, train_y, test_y = official_subsample(
            data_root,
            x,
            y,
            int(design["train_size"]),
            int(design["test_size"]),
        )
        dataset_output: dict[str, Any] = {
            "train_size": len(train_y),
            "test_size": len(test_y),
            "feature_count": x.shape[1],
            "numerical_count": numerical_count,
            "categorical_count": len(categorical),
            "class_count": int(y.max() + 1),
            "families": {},
        }
        for family in selected_families:
            family_output = {}
            for factor in selected_factors:
                views = build_views(
                    factor,
                    train_x,
                    train_y,
                    categorical,
                    cardinalities,
                    numerical_count,
                    int(design["nonidentity_views_per_factor"]),
                )
                family_output[factor] = evaluate_cell(
                    family,
                    factor,
                    views,
                    config["candidate_grids"][family],
                    train_x,
                    test_x,
                    train_y,
                    test_y,
                    categorical,
                    numerical_count,
                    config,
                )
                print(
                    dataset,
                    family,
                    factor,
                    family_output[factor]["frozen"]["max_probability_deviation_from_identity"],
                    family_output[factor]["selection_switch_fraction"],
                    flush=True,
                )
            dataset_output["families"][family] = family_output
        output["cells"][dataset] = dataset_output
        if args.output is not None:
            args.output.write_text(json.dumps(output, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=HERE.parent / "road-to-iclr-day-01" / "data",
    )
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--families", nargs="+")
    parser.add_argument("--factors", nargs="+")
    parser.add_argument("--output", type=Path, default=HERE / "invariance_matrix_results.json")
    args = parser.parse_args()
    result = run(args)
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
