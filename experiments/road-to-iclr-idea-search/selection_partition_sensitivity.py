"""Cross-fit pooled HPO decisions over every 2-by-2 nuisance partition.

This is a decision-level diagnostic over saved validation losses.  It does not
reuse test labels and does not claim independent-sample generalization: the
development and evaluation schema representatives share validation rows.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from orbit_anova import decompose


HERE = Path(__file__).resolve().parent
ROOT = HERE / "selection_split_repeats"
SEEDS = tuple(range(20_260_827, 20_260_834))
CASES = (
    ("adult", "catboost_native"),
    ("churn", "ordinal_forest"),
    ("churn", "catboost_native"),
)


def main() -> None:
    output: dict[str, object] = {
        "status": "post_frozen_nuisance_partition_sensitivity",
        "estimand": (
            "pooled configuration transfer across every disjoint 2-of-4 "
            "feature-level by 2-of-4 category-level partition"
        ),
        "warning": (
            "development and evaluation representatives share validation "
            "rows; this diagnoses nuisance-level decision sensitivity, not "
            "independent-sample or test-risk transfer"
        ),
        "split_seeds": list(SEEDS),
        "partitions_per_split": 36,
        "cases": {},
    }
    pairs = tuple(itertools.combinations(range(4), 2))
    for dataset, family in CASES:
        partition_records = []
        split_records = []
        decision_grid = np.zeros((len(pairs), len(pairs), len(SEEDS), 1, 4))
        for split_index, seed in enumerate(SEEDS):
            stem = f"{dataset}_{family}_s{seed}"
            losses = np.load(ROOT / f"{stem}.npz")["validation_losses"]
            full_choice = int(np.argmin(losses.mean(axis=(0, 1, 2))))
            identity_choice = int(np.argmin(losses[0, 0, 0]))
            choices = []
            for feature_index, feature_development in enumerate(pairs):
                feature_evaluation = tuple(
                    level for level in range(4) if level not in feature_development
                )
                for category_index, category_development in enumerate(pairs):
                    category_evaluation = tuple(
                        level
                        for level in range(4)
                        if level not in category_development
                    )
                    development = losses[
                        np.ix_(
                            feature_development,
                            category_development,
                            range(2),
                            range(losses.shape[-1]),
                        )
                    ].mean(axis=(0, 1, 2))
                    evaluation = losses[
                        np.ix_(
                            feature_evaluation,
                            category_evaluation,
                            range(2),
                            range(losses.shape[-1]),
                        )
                    ].mean(axis=(0, 1, 2))
                    choice = int(np.argmin(development))
                    decision_grid[
                        feature_index, category_index, split_index, 0, choice
                    ] = 1.0
                    choices.append(choice)
                    partition_records.append(
                        {
                            "split_seed": seed,
                            "feature_development": list(feature_development),
                            "category_development": list(category_development),
                            "selected_config": choice,
                            "full_menu_selected_config": full_choice,
                            "identity_selected_config": identity_choice,
                            "matches_full_menu_choice": choice == full_choice,
                            "heldout_validation_regret": float(
                                evaluation[choice] - evaluation.min()
                            ),
                            "heldout_minus_identity_validation_loss": float(
                                evaluation[choice] - evaluation[identity_choice]
                            ),
                        }
                    )
            values, counts = np.unique(choices, return_counts=True)
            split_records.append(
                {
                    "split_seed": seed,
                    "full_menu_selected_config": full_choice,
                    "identity_selected_config": identity_choice,
                    "number_of_partition_selected_configs": len(values),
                    "partition_selection_counts": {
                        str(int(value)): int(count)
                        for value, count in zip(values, counts)
                    },
                    "matches_full_menu_partitions": int(
                        np.sum(np.asarray(choices) == full_choice)
                    ),
                }
            )

        regrets = np.asarray(
            [record["heldout_validation_regret"] for record in partition_records]
        )
        identity_differences = np.asarray(
            [
                record["heldout_minus_identity_validation_loss"]
                for record in partition_records
            ]
        )
        matches = np.asarray(
            [record["matches_full_menu_choice"] for record in partition_records]
        )
        decision_anova = decompose(
            decision_grid, ("feature_menu", "category_menu", "split")
        )
        menu_main = sum(
            value
            for name, value in decision_anova.items()
            if name
            not in {
                "total",
                "split",
                "component_sum_error",
                "prediction_reconstruction_max_error",
            }
            and "split" not in name.split(":")
        )
        menu_split = sum(
            value
            for name, value in decision_anova.items()
            if "split" in name.split(":") and name != "split"
        )
        total = decision_anova["total"]
        output["cases"][f"{dataset}/{family}"] = {
            "summary": {
                "partition_count": len(partition_records),
                "fraction_matching_full_menu_choice": float(matches.mean()),
                "fraction_heldout_optimal": float(np.mean(regrets < 1e-15)),
                "mean_heldout_validation_regret": float(regrets.mean()),
                "maximum_heldout_validation_regret": float(regrets.max()),
                "fraction_no_worse_than_identity_on_heldout_validation": float(
                    np.mean(identity_differences <= 0)
                ),
                "mean_heldout_minus_identity_validation_loss": float(
                    identity_differences.mean()
                ),
                "splits_with_one_choice_across_all_partitions": int(
                    sum(
                        record["number_of_partition_selected_configs"] == 1
                        for record in split_records
                    )
                ),
                "decision_menu_main_fraction": float(menu_main / total),
                "decision_split_main_fraction": float(
                    decision_anova["split"] / total
                ),
                "decision_menu_by_split_fraction": float(menu_split / total),
            },
            "split_records": split_records,
            "decision_fanova": decision_anova,
        }

    destination = HERE / "selection_partition_sensitivity.json"
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
