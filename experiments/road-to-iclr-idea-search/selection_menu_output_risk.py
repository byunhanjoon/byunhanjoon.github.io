"""Propagate balanced development-menu choices into held-out predictions.

The saved pilot contains several uniformly refit configuration paths.  Since
the full training set and model seed do not change with validation split, those
paths can be reused across split seeds.  Only two missing configuration orbits
are fitted here (Churn forest config 2 and Churn CatBoost config 2).
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from cross_model_orbit_gate import load_dataset, official_subsample
from orbit_anova import decompose, risk_summary
from selection_rule_orbit_pilot import (
    FACTOR_NAMES,
    build_orbit,
    candidate_specs,
    fit_predict,
)
from selection_split_confirmation import exact_binomial_sign_p, exact_sign_flip_p


HERE = Path(__file__).resolve().parent
ROOT = HERE / "selection_split_repeats"
DATA_ROOT = HERE.parent / "road-to-iclr-day-01" / "data"
SEEDS = tuple(range(20_260_827, 20_260_834))
CASES = (
    ("adult", "catboost_native"),
    ("churn", "ordinal_forest"),
    ("churn", "catboost_native"),
)


def capped_density_ratio_worst_mean(values: np.ndarray, cap: int) -> float:
    """Worst weighted mean for weights <= cap / n when n is cap-divisible."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) % cap:
        raise ValueError("This exact helper requires an evenly divisible menu")
    return float(np.sort(values)[-len(values) // cap :].mean())


def saved_uniform_predictions(
    dataset: str, family: str
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    paths: dict[int, np.ndarray] = {}
    test_y = None
    for seed in SEEDS:
        stem = f"{dataset}_{family}_s{seed}"
        record = json.loads((ROOT / f"{stem}.json").read_text())
        arrays = np.load(ROOT / f"{stem}.npz")
        candidates = (
            (
                record["selection_protocol"]["canonical_identity_selected_config"],
                arrays["frozen_predictions"],
            ),
            (
                record["selection_protocol"]["orbit_average_validation_selected_config"],
                arrays["orbit_average_predictions"],
            ),
            (
                record["selection_protocol"]["development_pooled_selected_config"],
                arrays["development_pooled_predictions"],
            ),
        )
        for config, predictions in candidates:
            config = int(config)
            if config in paths and not np.allclose(
                paths[config], predictions, atol=1e-14, rtol=0
            ):
                raise ValueError("Full refit changed across validation splits")
            paths[config] = predictions
        if test_y is None:
            test_y = arrays["test_y"]
        elif not np.array_equal(test_y, arrays["test_y"]):
            raise ValueError("Test rows changed across validation splits")
    assert test_y is not None
    return paths, test_y


def fit_config_orbit(dataset: str, family: str, config: int) -> np.ndarray:
    data_root = DATA_ROOT / dataset
    x, y, categorical, cardinalities = load_dataset(data_root)
    train_x, test_x, train_y, test_y = official_subsample(
        data_root, x, y, 3_000, 1_000
    )
    feature_permutations, category_maps, label_permutations = build_orbit(
        x, y, categorical, cardinalities, 4, 4, 2
    )
    shape = (
        len(feature_permutations),
        len(category_maps),
        len(label_permutations),
    )
    predictions = np.empty(shape + (len(test_y), int(y.max() + 1)))
    spec = candidate_specs(family)[config]
    for index in itertools.product(*(range(size) for size in shape)):
        feature_index, category_index, class_index = index
        permutation = feature_permutations[feature_index]
        mappings = category_maps[category_index]
        label_permutation = label_permutations[class_index]
        transformed_categorical = tuple(
            new_index
            for new_index, old_index in enumerate(permutation)
            if old_index in categorical
        )

        def render(values: np.ndarray) -> np.ndarray:
            transformed = values.copy()
            for column, mapping in mappings.items():
                transformed[:, column] = mapping[values[:, column].astype(int)]
            return transformed[:, permutation]

        raw = fit_predict(
            family,
            spec,
            render(train_x),
            label_permutation[train_y],
            render(test_x),
            transformed_categorical,
            77,
        )
        predictions[index] = raw[:, label_permutation]
    return predictions


def main() -> None:
    pairs = tuple(itertools.combinations(range(4), 2))
    output: dict[str, object] = {
        "status": "post_frozen_menu_output_risk_extension",
        "warning": (
            "the three cases were selected for baseline instability; balanced "
            "menus overlap, and split-level averages—not 252 menu/split cells—"
            "are the inferential replication units"
        ),
        "cases": {},
    }
    cache: dict[str, np.ndarray] = {}
    for dataset, family in CASES:
        case = f"{dataset}_{family}"
        config_predictions, test_y = saved_uniform_predictions(dataset, family)
        choices_by_split = []
        needed = set()
        selected_predictions_by_split = []
        for seed in SEEDS:
            arrays = np.load(ROOT / f"{case}_s{seed}.npz")
            losses = arrays["validation_losses"]
            selected_predictions_by_split.append(arrays["selected_predictions"])
            grid = np.empty((len(pairs), len(pairs)), dtype=int)
            for feature_index, feature in enumerate(pairs):
                for category_index, category in enumerate(pairs):
                    development = losses[
                        np.ix_(feature, category, range(2), range(4))
                    ].mean(axis=(0, 1, 2))
                    grid[feature_index, category_index] = int(
                        np.argmin(development)
                    )
            choices_by_split.append(grid)
            needed.update(int(value) for value in grid.flat)

        fitted = []
        for config in sorted(needed - set(config_predictions)):
            config_predictions[config] = fit_config_orbit(dataset, family, config)
            fitted.append(config)
        for config, predictions in config_predictions.items():
            if config in needed:
                cache[f"{case}_config_{config}"] = predictions

        heldout_records = []
        joint = np.empty((32, len(SEEDS), 36, len(test_y), 2))
        for split_index, (choices, selected_predictions) in enumerate(
            zip(choices_by_split, selected_predictions_by_split)
        ):
            menu_index = 0
            for feature_index, feature in enumerate(pairs):
                feature_evaluation = tuple(
                    level for level in range(4) if level not in feature
                )
                for category_index, category in enumerate(pairs):
                    category_evaluation = tuple(
                        level for level in range(4) if level not in category
                    )
                    config = int(choices[feature_index, category_index])
                    pooled = config_predictions[config]
                    joint[:, split_index, menu_index] = pooled.reshape(
                        32, len(test_y), 2
                    )
                    evaluation_index = np.ix_(
                        feature_evaluation,
                        category_evaluation,
                        range(2),
                    )
                    pooled_evaluation = pooled[evaluation_index]
                    selected_evaluation = selected_predictions[evaluation_index]
                    pooled_summary = risk_summary(
                        pooled_evaluation, test_y, FACTOR_NAMES
                    )
                    selected_summary = risk_summary(
                        selected_evaluation, test_y, FACTOR_NAMES
                    )
                    heldout_records.append(
                        {
                            "split_seed": SEEDS[split_index],
                            "menu_index": menu_index,
                            "selected_config": config,
                            "pooled_schema_risk": float(
                                pooled_summary["anova"]["total"]
                            ),
                            "selected_schema_risk": float(
                                selected_summary["anova"]["total"]
                            ),
                            "pooled_minus_selected_schema_risk": float(
                                pooled_summary["anova"]["total"]
                                - selected_summary["anova"]["total"]
                            ),
                            "pooled_brier": float(
                                pooled_summary["orbit_mean_brier"]
                            ),
                            "selected_brier": float(
                                selected_summary["orbit_mean_brier"]
                            ),
                            "pooled_minus_selected_brier": float(
                                pooled_summary["orbit_mean_brier"]
                                - selected_summary["orbit_mean_brier"]
                            ),
                        }
                    )
                    menu_index += 1

        split_records = []
        for seed in SEEDS:
            records = [r for r in heldout_records if r["split_seed"] == seed]
            risk = np.asarray(
                [r["pooled_minus_selected_schema_risk"] for r in records]
            )
            loss = np.asarray([r["pooled_minus_selected_brier"] for r in records])
            split_records.append(
                {
                    "split_seed": seed,
                    "mean_over_menus_schema_risk_difference": float(risk.mean()),
                    "lower_schema_risk_menus": int(np.sum(risk < 0)),
                    "schema_risk_difference_range": [
                        float(risk.min()),
                        float(risk.max()),
                    ],
                    "schema_risk_worst_mean_density_ratio_cap_2": (
                        capped_density_ratio_worst_mean(risk, 2)
                    ),
                    "schema_risk_worst_mean_density_ratio_cap_4": (
                        capped_density_ratio_worst_mean(risk, 4)
                    ),
                    "mean_over_menus_brier_difference": float(loss.mean()),
                    "lower_brier_menus": int(np.sum(loss < 0)),
                    "brier_difference_range": [
                        float(loss.min()),
                        float(loss.max()),
                    ],
                }
            )
        split_risk = np.asarray(
            [r["mean_over_menus_schema_risk_difference"] for r in split_records]
        )
        split_loss = np.asarray(
            [r["mean_over_menus_brier_difference"] for r in split_records]
        )
        robust_cap_2 = np.asarray(
            [
                r["schema_risk_worst_mean_density_ratio_cap_2"]
                for r in split_records
            ]
        )
        robust_cap_4 = np.asarray(
            [
                r["schema_risk_worst_mean_density_ratio_cap_4"]
                for r in split_records
            ]
        )
        all_risk_differences = np.asarray(
            [r["pooled_minus_selected_schema_risk"] for r in heldout_records]
        )
        selected_risks = np.asarray(
            [r["selected_schema_risk"] for r in heldout_records]
        )
        pooled_risks = np.asarray(
            [r["pooled_schema_risk"] for r in heldout_records]
        )
        joint_anova = decompose(joint, ("schema", "split", "menu"))
        menu_total = sum(
            value
            for name, value in joint_anova.items()
            if "menu" in name.split(":")
        )
        output["cases"][f"{dataset}/{family}"] = {
            "needed_configs": sorted(needed),
            "newly_fitted_configs": fitted,
            "split_summary": {
                "mean_menu_averaged_schema_risk_difference": float(
                    split_risk.mean()
                ),
                "mean_selected_schema_risk": float(selected_risks.mean()),
                "mean_pooled_schema_risk": float(pooled_risks.mean()),
                "relative_mean_schema_risk_reduction": float(
                    1 - pooled_risks.mean() / selected_risks.mean()
                ),
                "individual_menus_with_lower_schema_risk": int(
                    np.sum(all_risk_differences < 0)
                ),
                "individual_menu_count": len(all_risk_differences),
                "splits_with_lower_mean_schema_risk": int(np.sum(split_risk < 0)),
                "schema_risk_exact_sign_flip_p": exact_sign_flip_p(split_risk),
                "schema_risk_exact_binomial_sign_p": exact_binomial_sign_p(split_risk),
                "splits_robustly_lower_at_density_ratio_cap_2": int(
                    np.sum(robust_cap_2 < 0)
                ),
                "mean_worst_schema_risk_difference_density_ratio_cap_2": float(
                    robust_cap_2.mean()
                ),
                "splits_robustly_lower_at_density_ratio_cap_4": int(
                    np.sum(robust_cap_4 < 0)
                ),
                "mean_worst_schema_risk_difference_density_ratio_cap_4": float(
                    robust_cap_4.mean()
                ),
                "mean_menu_averaged_brier_difference": float(split_loss.mean()),
                "splits_with_lower_mean_brier": int(np.sum(split_loss < 0)),
                "brier_exact_sign_flip_p": exact_sign_flip_p(split_loss),
                "brier_exact_binomial_sign_p": exact_binomial_sign_p(split_loss),
            },
            "joint_output_anova": joint_anova,
            "joint_output_menu_total_fraction": float(
                menu_total / joint_anova["total"]
            ),
            "split_records": split_records,
        }

    np.savez_compressed(HERE / "selection_menu_config_predictions.npz", **cache)
    destination = HERE / "selection_menu_output_risk.json"
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
