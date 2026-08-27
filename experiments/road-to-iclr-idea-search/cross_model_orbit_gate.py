"""Small cross-model OrbitANOVA gate on mixed-type binary datasets.

This is an exploratory gate, not a frozen benchmark.  It compares pipelines
under the same balanced product of feature-position, nominal-code, and class-ID
transformations.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from orbit_anova import (
    brier,
    budgeted_marginalization_frontier,
    log_orbit_summary,
    risk_summary,
    symmetrization_frontier,
)


def load_dataset(
    root: Path,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], dict[int, int]]:
    row_count = len(np.load(root / "y.npy"))
    numerical = (
        np.load(root / "x_num.npy").astype(np.float64)
        if (root / "x_num.npy").exists()
        else np.empty((row_count, 0), dtype=np.float64)
    )
    binary = (
        np.load(root / "x_bin.npy").astype(np.float64)
        if (root / "x_bin.npy").exists()
        else np.empty((row_count, 0), dtype=np.float64)
    )
    raw_categorical = (
        np.load(root / "x_cat.npy")
        if (root / "x_cat.npy").exists()
        else np.empty((row_count, 0), dtype=str)
    )
    encoded_columns = []
    cardinalities = []
    for column in range(raw_categorical.shape[1]):
        levels, encoded = np.unique(raw_categorical[:, column], return_inverse=True)
        encoded_columns.append(encoded)
        cardinalities.append(len(levels))
    categorical = (
        np.column_stack(encoded_columns).astype(np.float64)
        if encoded_columns
        else np.empty((row_count, 0), dtype=np.float64)
    )
    x = np.column_stack((numerical, binary, categorical))
    y = np.load(root / "y.npy").astype(int)
    categorical_indices = tuple(range(numerical.shape[1] + binary.shape[1], x.shape[1]))
    category_cardinalities = dict(zip(categorical_indices, cardinalities))
    return x, y, categorical_indices, category_cardinalities


def official_subsample(
    root: Path,
    x: np.ndarray,
    y: np.ndarray,
    train_size: int,
    test_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_indices = np.load(root / "splits/default/train.npy")
    test_indices = np.load(root / "splits/default/test.npy")
    rng = np.random.default_rng(20_260_826)

    def stratified(indices: np.ndarray, size: int) -> np.ndarray:
        selected = []
        for label in np.unique(y[indices]):
            candidates = indices[y[indices] == label]
            count = round(size * len(candidates) / len(indices))
            selected.extend(rng.choice(candidates, size=count, replace=False))
        selected = np.asarray(selected)
        if len(selected) > size:
            selected = rng.choice(selected, size=size, replace=False)
        elif len(selected) < size:
            remainder = np.setdiff1d(indices, selected, assume_unique=False)
            selected = np.concatenate(
                (selected, rng.choice(remainder, size=size - len(selected), replace=False))
            )
        return rng.permutation(selected)

    train_indices = stratified(train_indices, train_size)
    test_indices = stratified(test_indices, test_size)
    return x[train_indices], x[test_indices], y[train_indices], y[test_indices]


def preprocessing(
    categorical_indices: tuple[int, ...],
    feature_count: int,
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


def predict_model(
    name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    categorical_indices: tuple[int, ...],
    category_cardinalities: dict[int, int] | None = None,
    random_state: int = 77,
) -> np.ndarray:
    if name == "onehot_logistic":
        model = make_pipeline(
            preprocessing(categorical_indices, train_x.shape[1]),
            LogisticRegression(
                C=1.0,
                max_iter=1_000,
                tol=1e-11,
                random_state=random_state,
            ),
        )
    elif name == "ordinal_forest":
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=160,
                min_samples_leaf=3,
                max_features="sqrt",
                n_jobs=4,
                random_state=random_state,
            ),
        )
    elif name == "native_histgb":
        model = HistGradientBoostingClassifier(
            categorical_features=list(categorical_indices),
            learning_rate=0.08,
            max_iter=160,
            max_leaf_nodes=31,
            min_samples_leaf=15,
            l2_regularization=1.0,
            random_state=random_state,
        )
    elif name == "onehot_adam_mlp":
        model = make_pipeline(
            preprocessing(categorical_indices, train_x.shape[1]),
            MLPClassifier(
                hidden_layer_sizes=(96, 48),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=128,
                learning_rate_init=1e-3,
                max_iter=100,
                shuffle=True,
                random_state=random_state,
            ),
        )
    elif name == "lightgbm_native":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            n_estimators=160,
            learning_rate=0.08,
            num_leaves=31,
            min_child_samples=15,
            reg_lambda=1.0,
            random_state=random_state,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
            n_jobs=4,
        )
        model.fit(
            train_x,
            train_y,
            categorical_feature=list(categorical_indices),
        )
        return model.predict_proba(test_x)
    elif name in ("catboost_native", "xgboost_native"):
        train_frame = pd.DataFrame(train_x)
        test_frame = pd.DataFrame(test_x)
        for column in categorical_indices:
            train_frame[column] = train_frame[column].astype(int).astype(str)
            test_frame[column] = test_frame[column].astype(int).astype(str)
        if name == "catboost_native":
            from catboost import CatBoostClassifier

            model = CatBoostClassifier(
                iterations=160,
                depth=6,
                learning_rate=0.08,
                loss_function=("Logloss" if len(np.unique(train_y)) == 2 else "MultiClass"),
                random_seed=random_state,
                verbose=False,
                allow_writing_files=False,
                thread_count=4,
            )
            model.fit(train_frame, train_y, cat_features=list(categorical_indices))
        else:
            from xgboost import XGBClassifier

            for column in categorical_indices:
                if category_cardinalities is None:
                    raise ValueError("XGBoost requires declared category cardinalities")
                levels = [str(level) for level in range(category_cardinalities[column])]
                dtype = pd.CategoricalDtype(categories=levels)
                train_frame[column] = train_frame[column].astype(dtype)
                test_frame[column] = test_frame[column].astype(dtype)
            model = XGBClassifier(
                n_estimators=160,
                max_depth=6,
                learning_rate=0.08,
                min_child_weight=3.0,
                reg_lambda=1.0,
                subsample=1.0,
                colsample_bytree=1.0,
                tree_method="hist",
                enable_categorical=True,
                random_state=random_state,
                n_jobs=4,
            )
            model.fit(train_frame, train_y)
        return model.predict_proba(test_frame)
    else:
        raise ValueError(name)
    model.fit(train_x, train_y)
    return model.predict_proba(test_x)


def equal_compute_audit(
    joint_predictions: np.ndarray,
    y: np.ndarray,
    factor_names: tuple[str, ...],
    budgets: tuple[int, ...] = (2, 4, 8),
) -> dict[str, object]:
    """Compare label-free factor selection with seed and generic ensembles."""
    fixed_seed_predictions = joint_predictions[..., 0, :, :]
    frontier = symmetrization_frontier(fixed_seed_predictions, factor_names)
    reference = fixed_seed_predictions[(0,) * len(factor_names)]
    flat_schema = fixed_seed_predictions.reshape(
        (-1,) + fixed_seed_predictions.shape[-2:]
    )
    reference_seed_predictions = joint_predictions[(0,) * len(factor_names)]
    output: dict[str, object] = {"reference_brier": brier(y, reference), "budgets": {}}
    for budget in budgets:
        if budget > reference_seed_predictions.shape[0]:
            continue
        feasible = [item for item in frontier if item["full_orbit_cost"] <= budget]
        selected = max(
            feasible,
            key=lambda item: (item["removed_risk"], -item["full_orbit_cost"]),
        )
        selected_names = set(selected["factors"])
        selected_axes = tuple(
            index for index, name in enumerate(factor_names) if name in selected_names
        )
        member_index = tuple(
            slice(None) if index in selected_axes else 0
            for index in range(len(factor_names))
        ) + (slice(None), slice(None))
        selected_members = fixed_seed_predictions[member_index].reshape(
            (-1,) + fixed_seed_predictions.shape[-2:]
        )
        selected_ensemble = selected_members.mean(axis=0)

        seed_members = reference_seed_predictions[:budget]
        seed_ensemble = seed_members.mean(axis=0)

        rng = np.random.default_rng(904_731 + budget)
        if budget == 1:
            generic_indices = np.asarray([0])
        else:
            generic_indices = np.concatenate(
                (
                    np.asarray([0]),
                    rng.choice(
                        np.arange(1, len(flat_schema)),
                        size=min(budget - 1, len(flat_schema) - 1),
                        replace=False,
                    ),
                )
            )
        generic_members = flat_schema[generic_indices]
        generic_ensemble = generic_members.mean(axis=0)

        output["budgets"][str(budget)] = {
            "orbitcover_factors": selected["factors"],
            "orbitcover_actual_members": int(len(selected_members)),
            "orbitcover_removed_schema_fraction": selected["removed_fraction"],
            "orbitcover_member_mean_brier": float(
                np.mean([brier(y, member) for member in selected_members])
            ),
            "orbitcover_ensemble_brier": brier(y, selected_ensemble),
            "seed_ensemble_brier": brier(y, seed_ensemble),
            "generic_schema_indices": generic_indices.tolist(),
            "generic_schema_ensemble_brier": brier(y, generic_ensemble),
        }
    return output


def heldout_orbitcover_audit(
    predictions: np.ndarray,
    factor_names: tuple[str, ...],
    budgets: tuple[int, ...] = (2, 4, 8),
) -> dict[str, object]:
    """Select a label-free factor/MC action on rows disjoint from evaluation.

    An exact factor marginalization costs its level-product ``c_J``.  With
    budget ``B``, repeat it on ``floor(B/c_J)`` iid draws of the remaining
    factors; averaging those draws leaves expected residual
    ``SR(Q_J p) / floor(B/c_J)``.  The empty subset recovers generic iid schema
    averaging. Selection and evaluation use disjoint query rows and no labels.
    """
    rng = np.random.default_rng(61_027)
    permutation = rng.permutation(predictions.shape[-2])
    midpoint = len(permutation) // 2
    selection_rows = np.sort(permutation[:midpoint])
    evaluation_rows = np.sort(permutation[midpoint:])
    output: dict[str, object] = {
        "selection_rows": len(selection_rows),
        "evaluation_rows": len(evaluation_rows),
        "labels_used": False,
        "budgets": {},
    }
    for budget in budgets:
        selection_frontier = budgeted_marginalization_frontier(
            predictions[..., selection_rows, :], factor_names, budget
        )
        evaluation_frontier = budgeted_marginalization_frontier(
            predictions[..., evaluation_rows, :], factor_names, budget
        )
        evaluation_by_factors = {
            tuple(item["factors"]): item for item in evaluation_frontier
        }
        selection_total = selection_frontier[0]["residual_risk"]
        evaluation_total = evaluation_frontier[0]["residual_risk"]
        candidates = []
        for item in selection_frontier:
            evaluation = evaluation_by_factors[tuple(item["factors"])]
            candidates.append(
                {
                    "action": (
                        "generic_iid_schema_mean"
                        if not item["factors"]
                        else "factor_marginalization_then_iid_complement"
                    ),
                    "factors": item["factors"],
                    "cost": item["full_orbit_cost"],
                    "iid_repeats": item["conditional_draws"],
                    "realized_cost": item["realized_cost"],
                    "unused_budget": item["unused_budget"],
                    "selection_removed": item["expected_removed_risk"],
                    "evaluation_removed": evaluation["expected_removed_risk"],
                }
            )
        selected = max(candidates, key=lambda item: item["selection_removed"])
        oracle = max(candidates, key=lambda item: item["evaluation_removed"])
        regret = oracle["evaluation_removed"] - selected["evaluation_removed"]
        output["budgets"][str(budget)] = {
            "selected_action": selected["action"],
            "selected_factors": selected["factors"],
            "selected_cost": selected["cost"],
            "selected_realized_cost": selected["realized_cost"],
            "selected_unused_budget": selected["unused_budget"],
            "selected_iid_repeats": selected["iid_repeats"],
            "evaluation_removed_fraction": (
                selected["evaluation_removed"] / evaluation_total
                if evaluation_total
                else np.nan
            ),
            "evaluation_oracle_action": oracle["action"],
            "evaluation_oracle_factors": oracle["factors"],
            "evaluation_oracle_iid_repeats": oracle["iid_repeats"],
            "evaluation_oracle_removed_fraction": (
                oracle["evaluation_removed"] / evaluation_total
                if evaluation_total
                else np.nan
            ),
            "selection_regret_fraction_of_total": (
                regret / evaluation_total if evaluation_total else np.nan
            ),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("experiments/road-to-iclr-day-01/data/adult"),
    )
    parser.add_argument("--train-size", type=int, default=1_000)
    parser.add_argument("--test-size", type=int, default=1_000)
    parser.add_argument("--n-feature", type=int, default=4)
    parser.add_argument("--n-category", type=int, default=4)
    parser.add_argument(
        "--n-class",
        type=int,
        default=0,
        help="Class permutations; 0 uses all two binary swaps or four sampled multiclass maps.",
    )
    parser.add_argument("--n-seed", type=int, default=16)
    parser.add_argument(
        "--n-joint-seed",
        type=int,
        default=1,
        help="Seed levels crossed with the schema grid; 1 preserves the fast gate.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=(
            "onehot_logistic",
            "ordinal_forest",
            "native_histgb",
            "onehot_adam_mlp",
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    x, y, original_categorical, original_cardinalities = load_dataset(args.data_root)
    train_x, test_x, train_y, test_y = official_subsample(
        args.data_root,
        x,
        y,
        args.train_size,
        args.test_size,
    )
    feature_rng = np.random.default_rng(91_773)
    feature_permutations = [np.arange(x.shape[1])] + [
        feature_rng.permutation(x.shape[1]) for _ in range(args.n_feature - 1)
    ]
    category_rng = np.random.default_rng(821_571)
    category_maps = []
    category_level_count = args.n_category if original_categorical else 1
    for category_level in range(category_level_count):
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
    if args.n_class > 0:
        class_permutation_count = min(args.n_class, math.factorial(class_count))
    else:
        class_permutation_count = 2 if class_count == 2 else 4
    class_rng = np.random.default_rng(430_921)
    label_permutations = [np.arange(class_count)]
    while len(label_permutations) < class_permutation_count:
        candidate = class_rng.permutation(class_count)
        if not any(np.array_equal(candidate, existing) for existing in label_permutations):
            label_permutations.append(candidate)

    output: dict[str, object] = {
        "dataset": args.data_root.name,
        "train_size": len(train_y),
        "test_size": len(test_y),
        "factor_shape": [
            len(feature_permutations),
            len(category_maps),
            len(label_permutations),
        ],
        "models": {},
    }
    member_risk_grids: dict[str, np.ndarray] = {}
    prediction_grids: dict[str, np.ndarray] = {}
    for model_name in args.models:
        predictions = []
        for feature_permutation in feature_permutations:
            transformed_categorical = tuple(
                index
                for index, original_index in enumerate(feature_permutation)
                if original_index in original_categorical
            )
            transformed_cardinalities = {
                new_index: original_cardinalities[original_index]
                for new_index, original_index in enumerate(feature_permutation)
                if original_index in original_categorical
            }
            feature_predictions = []
            for mappings in category_maps:
                category_train = train_x.copy()
                category_test = test_x.copy()
                for column, mapping in mappings.items():
                    category_train[:, column] = mapping[train_x[:, column].astype(int)]
                    category_test[:, column] = mapping[test_x[:, column].astype(int)]
                label_predictions = []
                for label_permutation in label_permutations:
                    joint_seed_predictions = []
                    for joint_seed in range(args.n_joint_seed):
                        probabilities = predict_model(
                            model_name,
                            category_train[:, feature_permutation],
                            label_permutation[train_y],
                            category_test[:, feature_permutation],
                            transformed_categorical,
                            transformed_cardinalities,
                            random_state=77 + joint_seed,
                        )
                        joint_seed_predictions.append(
                            probabilities[:, label_permutation]
                        )
                    label_predictions.append(joint_seed_predictions)
                feature_predictions.append(label_predictions)
            predictions.append(feature_predictions)
        joint_prediction_array = np.asarray(predictions)
        prediction_array = joint_prediction_array[..., 0, :, :]
        prediction_grids[model_name] = prediction_array.astype(np.float64)
        flat_predictions = prediction_array.reshape(
            (-1,) + prediction_array.shape[-2:]
        )
        member_risk_grids[model_name] = np.asarray(
            [brier(test_y, member.astype(np.float64)) for member in flat_predictions]
        ).reshape(prediction_array.shape[:3])
        output["models"][model_name] = risk_summary(
            prediction_array,
            test_y,
            ("feature", "category", "class"),
        )
        output["models"][model_name]["log_loss_orbit"] = log_orbit_summary(
            prediction_array,
            3,
            test_y,
        )
        output["models"][model_name]["symmetrization_frontier"] = (
            symmetrization_frontier(
                prediction_array,
                ("feature", "category", "class"),
            )
        )
        output["models"][model_name]["heldout_orbitcover"] = (
            heldout_orbitcover_audit(
                prediction_array,
                ("feature", "category", "class"),
            )
        )
        if args.n_joint_seed > 1:
            joint_summary = risk_summary(
                joint_prediction_array,
                test_y,
                ("feature", "category", "class", "seed"),
            )
            joint_anova = joint_summary["anova"]
            component_items = {
                name: value
                for name, value in joint_anova.items()
                if name
                not in {
                    "total",
                    "component_sum_error",
                    "prediction_reconstruction_max_error",
                }
            }
            seed_main = component_items["seed"]
            schema_seed_interactions = sum(
                value
                for name, value in component_items.items()
                if "seed" in name.split(":") and name != "seed"
            )
            marginal_schema = sum(
                value
                for name, value in component_items.items()
                if "seed" not in name.split(":")
            )
            joint_summary["schema_seed_partition"] = {
                "marginal_schema_variance": marginal_schema,
                "seed_main_variance": seed_main,
                "schema_seed_interaction_variance": schema_seed_interactions,
                "expected_conditional_schema_variance": (
                    marginal_schema + schema_seed_interactions
                ),
                "expected_conditional_seed_variance": (
                    seed_main + schema_seed_interactions
                ),
            }
            output["models"][model_name]["joint_schema_seed_orbit"] = joint_summary
            output["models"][model_name]["equal_compute_audit"] = (
                equal_compute_audit(
                    joint_prediction_array,
                    test_y,
                    ("feature", "category", "class"),
                )
            )
        seed_predictions = np.asarray([
            predict_model(
                model_name,
                train_x,
                train_y,
                test_x,
                original_categorical,
                original_cardinalities,
                random_state=10_000 + seed,
            )
            for seed in range(args.n_seed)
        ])
        output["models"][model_name]["reference_seed_orbit"] = risk_summary(
            seed_predictions,
            test_y,
            ("seed",),
        )

    model_names = list(member_risk_grids)
    stacked_risks = np.stack([member_risk_grids[name] for name in model_names])
    ranks = np.argsort(np.argsort(stacked_risks, axis=0), axis=0) + 1
    ranking_audit: dict[str, object] = {
        "best_model_counts": {
            name: int(np.sum(np.argmin(stacked_risks, axis=0) == index))
            for index, name in enumerate(model_names)
        },
        "rank_ranges": {
            name: [int(ranks[index].min()), int(ranks[index].max())]
            for index, name in enumerate(model_names)
        },
        "pairwise": {},
    }
    for left_index, left_name in enumerate(model_names):
        for right_index in range(left_index + 1, len(model_names)):
            right_name = model_names[right_index]
            differences = stacked_risks[left_index] - stacked_risks[right_index]
            targets = np.eye(int(y.max() + 1))[test_y]
            left_losses = np.sum(
                (prediction_grids[left_name] - targets) ** 2, axis=-1
            )
            right_losses = np.sum(
                (prediction_grids[right_name] - targets) ** 2, axis=-1
            )
            paired_losses = left_losses - right_losses
            minimum_index = np.unravel_index(np.argmin(differences), differences.shape)
            maximum_index = np.unravel_index(np.argmax(differences), differences.shape)

            def paired_interval(index: tuple[int, ...]) -> list[float]:
                values = paired_losses[index]
                center = float(values.mean())
                half_width = float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))
                return [center - half_width, center + half_width]

            ranking_audit["pairwise"][f"{left_name}:{right_name}"] = {
                "left_better": int(np.sum(differences < -1e-12)),
                "right_better": int(np.sum(differences > 1e-12)),
                "ties": int(np.sum(np.abs(differences) <= 1e-12)),
                "minimum_left_minus_right": float(differences.min()),
                "maximum_left_minus_right": float(differences.max()),
                "minimum_gap_paired_95ci": paired_interval(minimum_index),
                "maximum_gap_paired_95ci": paired_interval(maximum_index),
            }
    output["ranking_audit"] = ranking_audit
    serialized = json.dumps(output, indent=2)
    if args.output is not None:
        args.output.write_text(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
