"""Decision-only prospective split screen for the three stable Otto cells."""

from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from cross_model_orbit_gate import load_dataset, official_subsample
from orbit_anova import brier, decompose
from selection_rule_orbit_pilot import (
    FACTOR_NAMES,
    build_orbit,
    candidate_specs,
    entropy,
    fit_predict,
    selection_margin_diagnostic,
)


HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parent / "road-to-iclr-day-01" / "data" / "otto"
SEEDS = tuple(range(20_260_827, 20_260_834))
FAMILIES = ("ordinal_forest", "native_histgb", "catboost_native")


def main() -> None:
    x, y, categorical, cardinalities = load_dataset(DATA_ROOT)
    train_x, _, train_y, _ = official_subsample(DATA_ROOT, x, y, 3_000, 1_000)
    feature_permutations, category_maps, label_permutations = build_orbit(
        x, y, categorical, cardinalities, 4, 1, 4
    )
    shape = (4, 1, 4)
    output: dict[str, object] = {
        "status": "prospective_decision_only_screen_of_three_baseline_stable_otto_cells",
        "baseline_split_seed": 20_260_826,
        "prospective_split_seeds": list(SEEDS),
        "test_labels_or_predictions_used": False,
        "cases": {},
    }
    for family in FAMILIES:
        specs = candidate_specs(family)
        records = []
        for seed in SEEDS:
            fit_x, validation_x, fit_y, validation_y = train_test_split(
                train_x,
                train_y,
                test_size=0.25,
                random_state=seed,
                stratify=train_y,
            )
            losses = np.empty(shape + (len(specs),), dtype=np.float64)
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
                        transformed[:, column] = mapping[
                            values[:, column].astype(int)
                        ]
                    return transformed[:, permutation]

                orbit_fit_x = render(fit_x)
                orbit_validation_x = render(validation_x)
                orbit_fit_y = label_permutation[fit_y]
                orbit_validation_y = label_permutation[validation_y]
                for config, spec in enumerate(specs):
                    probabilities = fit_predict(
                        family,
                        spec,
                        orbit_fit_x,
                        orbit_fit_y,
                        orbit_validation_x,
                        transformed_categorical,
                        77,
                    )
                    losses[index + (config,)] = brier(
                        orbit_validation_y, probabilities
                    )
            selected = np.argmin(losses, axis=-1)
            identity = int(selected[0, 0, 0])
            counts = Counter(int(value) for value in selected.flat)
            records.append(
                {
                    "split_seed": seed,
                    "identity_config": identity,
                    "selection_counts": {
                        str(key): value for key, value in sorted(counts.items())
                    },
                    "fraction_different_from_identity": float(
                        np.mean(selected != identity)
                    ),
                    "selection_entropy_bits": entropy(counts, selected.size),
                    "selection_margin_diagnostic": selection_margin_diagnostic(
                        losses
                    ),
                    "decision_fanova": decompose(
                        np.eye(len(specs))[selected][..., None, :], FACTOR_NAMES
                    ),
                    "mean_validation_brier_by_config": losses.mean(
                        axis=(0, 1, 2)
                    ).tolist(),
                }
            )
        output["cases"][f"otto/{family}"] = {
            "records": records,
            "selection_unstable_splits": int(
                sum(record["fraction_different_from_identity"] > 0 for record in records)
            ),
            "certificate_holds_splits": int(
                sum(
                    record["selection_margin_diagnostic"][
                        "sufficient_certificate_holds"
                    ]
                    for record in records
                )
            ),
        }
    destination = HERE / "selection_otto_prospective_decisions.json"
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
