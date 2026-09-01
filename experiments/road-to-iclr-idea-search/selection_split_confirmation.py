"""Confirm selection-path effects across conditionally prospective split seeds."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from orbit_anova import decompose, risk_summary
from selection_rule_orbit_pilot import selection_margin_diagnostic


HERE = Path(__file__).resolve().parent
ROOT = HERE / "selection_split_repeats"
SEEDS = tuple(range(20_260_826, 20_260_834))
CASES = (
    ("adult", "catboost_native"),
    ("churn", "ordinal_forest"),
    ("churn", "catboost_native"),
)
SCHEMA_FACTORS = ("feature", "category", "class")


def exact_sign_flip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    observed = abs(float(values.mean()))
    null = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        null.append(abs(float(np.mean(values * signs))))
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def exact_binomial_sign_p(values: np.ndarray) -> float:
    """Distribution-free paired sign test after discarding exact ties."""
    values = np.asarray(values, dtype=np.float64)
    nonzero = values[values != 0]
    if not len(nonzero):
        return 1.0
    return float(binomtest(int(np.sum(nonzero > 0)), len(nonzero), 0.5).pvalue)


def partition(summary: dict[str, object]) -> dict[str, float]:
    anova = summary["anova"]
    ignored = {"total", "component_sum_error", "prediction_reconstruction_max_error"}
    components = {key: value for key, value in anova.items() if key not in ignored}
    split_main = float(components["split"])
    schema_split = float(
        sum(
            value
            for name, value in components.items()
            if "split" in name.split(":") and name != "split"
        )
    )
    persistent_schema = float(
        sum(value for name, value in components.items() if "split" not in name.split(":"))
    )
    return {
        "persistent_schema_variance": persistent_schema,
        "split_main_variance": split_main,
        "schema_split_interaction_variance": schema_split,
        "same_split_schema_variance": persistent_schema + schema_split,
        "schema_split_fraction_joint": schema_split / anova["total"] if anova["total"] else 0.0,
    }


def main() -> None:
    output = {
        "status": "conditional_confirmation_of_three_baseline_selected_cells",
        "baseline_split_seed": SEEDS[0],
        "prospective_split_seeds": list(SEEDS[1:]),
        "selection_warning": (
            "cases were selected for instability on the baseline split; confirmation "
            "tests use only the seven later split seeds"
        ),
        "cases": {},
    }
    for dataset, family in CASES:
        records = []
        frozen_grids = []
        selected_grids = []
        orbit_average_grids = []
        development_pooled_grids = []
        decision_grids = []
        test_y = None
        for seed in SEEDS:
            stem = f"{dataset}_{family}_s{seed}"
            record = json.loads((ROOT / f"{stem}.json").read_text())
            arrays = np.load(ROOT / f"{stem}.npz")
            frozen_grids.append(arrays["frozen_predictions"])
            selected_grids.append(arrays["selected_predictions"])
            orbit_average_grids.append(arrays["orbit_average_predictions"])
            development_pooled_grids.append(
                arrays["development_pooled_predictions"]
            )
            decision_grids.append(np.eye(4)[arrays["selected_configs"]])
            if test_y is None:
                test_y = arrays["test_y"]
            elif not np.array_equal(test_y, arrays["test_y"]):
                raise ValueError("Test rows changed across split seeds")
            records.append(
                {
                    "split_seed": seed,
                    "fraction_different_from_identity": record[
                        "fraction_different_from_identity_config"
                    ],
                    "selection_entropy_bits": record["selection_entropy_bits"],
                    "schema_risk_difference": record[
                        "selection_minus_frozen_schema_risk"
                    ],
                    "schema_risk_ratio": record[
                        "selection_to_frozen_schema_risk_ratio"
                    ],
                    "orbit_mean_brier_difference": record[
                        "selection_minus_frozen_orbit_mean_brier"
                    ],
                    "identity_config": record["selection_protocol"][
                        "canonical_identity_selected_config"
                    ],
                    "selection_counts": record["selection_counts"],
                    "selection_margin_diagnostic": selection_margin_diagnostic(
                        arrays["validation_losses"]
                    ),
                    "orbit_average_config": record["selection_protocol"][
                        "orbit_average_validation_selected_config"
                    ],
                    "development_pooled_config": record["selection_protocol"][
                        "development_pooled_selected_config"
                    ],
                    "orbit_average_minus_selected_schema_risk": record[
                        "orbit_average_minus_representation_wise_schema_risk"
                    ],
                    "orbit_average_minus_selected_brier": record[
                        "orbit_average_minus_representation_wise_orbit_mean_brier"
                    ],
                    "heldout_repair_schema_risk_difference": record[
                        "heldout_nuisance_evaluation"
                    ]["development_pooled_minus_representation_wise_schema_risk"],
                    "heldout_repair_brier_difference": record[
                        "heldout_nuisance_evaluation"
                    ]["development_pooled_minus_representation_wise_brier"],
                }
            )
        # Insert split immediately before query rows/classes.
        frozen_joint = np.stack(frozen_grids, axis=3)
        selected_joint = np.stack(selected_grids, axis=3)
        orbit_average_joint = np.stack(orbit_average_grids, axis=3)
        development_pooled_joint = np.stack(development_pooled_grids, axis=3)
        decision_joint = np.stack(decision_grids, axis=3)[..., None, :]
        factors = SCHEMA_FACTORS + ("split",)
        frozen_summary = risk_summary(frozen_joint, test_y, factors)
        selected_summary = risk_summary(selected_joint, test_y, factors)
        orbit_average_summary = risk_summary(orbit_average_joint, test_y, factors)
        # The frozen pilot uses levels 0:2 for development and 2:4 for
        # evaluation on feature/category; both class levels remain crossed.
        heldout_index = (slice(2, 4), slice(2, 4), slice(None), slice(None), slice(None), slice(None))
        heldout_selected_summary = risk_summary(
            selected_joint[heldout_index], test_y, factors
        )
        heldout_development_pooled_summary = risk_summary(
            development_pooled_joint[heldout_index], test_y, factors
        )
        decision_anova = decompose(decision_joint, factors)
        confirmation = records[1:]
        risk_differences = np.asarray(
            [record["schema_risk_difference"] for record in confirmation]
        )
        brier_differences = np.asarray(
            [record["orbit_mean_brier_difference"] for record in confirmation]
        )
        repair_risk_differences = np.asarray(
            [record["orbit_average_minus_selected_schema_risk"] for record in confirmation]
        )
        repair_brier_differences = np.asarray(
            [record["orbit_average_minus_selected_brier"] for record in confirmation]
        )
        heldout_repair_risk_differences = np.asarray(
            [record["heldout_repair_schema_risk_difference"] for record in confirmation]
        )
        heldout_repair_brier_differences = np.asarray(
            [record["heldout_repair_brier_difference"] for record in confirmation]
        )
        case = f"{dataset}/{family}"
        output["cases"][case] = {
            "split_records": records,
            "confirmation_summary": {
                "selection_unstable_splits": int(
                    sum(record["fraction_different_from_identity"] > 0 for record in confirmation)
                ),
                "higher_schema_risk_splits": int(np.sum(risk_differences > 0)),
                "lower_brier_splits": int(np.sum(brier_differences < 0)),
                "mean_schema_risk_difference": float(risk_differences.mean()),
                "schema_risk_difference_range": [
                    float(risk_differences.min()),
                    float(risk_differences.max()),
                ],
                "schema_risk_difference_exact_sign_flip_p": exact_sign_flip_p(
                    risk_differences
                ),
                "schema_risk_difference_exact_binomial_sign_p": exact_binomial_sign_p(
                    risk_differences
                ),
                "mean_orbit_brier_difference": float(brier_differences.mean()),
                "orbit_brier_difference_range": [
                    float(brier_differences.min()),
                    float(brier_differences.max()),
                ],
                "orbit_brier_difference_exact_sign_flip_p": exact_sign_flip_p(
                    brier_differences
                ),
                "orbit_brier_difference_exact_binomial_sign_p": exact_binomial_sign_p(
                    brier_differences
                ),
                "repair_lower_schema_risk_splits": int(
                    np.sum(repair_risk_differences < 0)
                ),
                "mean_repair_schema_risk_difference": float(
                    repair_risk_differences.mean()
                ),
                "repair_schema_risk_difference_range": [
                    float(repair_risk_differences.min()),
                    float(repair_risk_differences.max()),
                ],
                "repair_schema_risk_exact_sign_flip_p": exact_sign_flip_p(
                    repair_risk_differences
                ),
                "repair_schema_risk_exact_binomial_sign_p": exact_binomial_sign_p(
                    repair_risk_differences
                ),
                "repair_lower_brier_splits": int(np.sum(repair_brier_differences < 0)),
                "mean_repair_brier_difference": float(repair_brier_differences.mean()),
                "repair_brier_difference_range": [
                    float(repair_brier_differences.min()),
                    float(repair_brier_differences.max()),
                ],
                "repair_brier_exact_sign_flip_p": exact_sign_flip_p(
                    repair_brier_differences
                ),
                "repair_brier_exact_binomial_sign_p": exact_binomial_sign_p(
                    repair_brier_differences
                ),
                "heldout_repair_lower_schema_risk_splits": int(
                    np.sum(heldout_repair_risk_differences < 0)
                ),
                "mean_heldout_repair_schema_risk_difference": float(
                    heldout_repair_risk_differences.mean()
                ),
                "heldout_repair_schema_risk_difference_range": [
                    float(heldout_repair_risk_differences.min()),
                    float(heldout_repair_risk_differences.max()),
                ],
                "heldout_repair_schema_risk_exact_sign_flip_p": exact_sign_flip_p(
                    heldout_repair_risk_differences
                ),
                "heldout_repair_schema_risk_exact_binomial_sign_p": exact_binomial_sign_p(
                    heldout_repair_risk_differences
                ),
                "heldout_repair_lower_brier_splits": int(
                    np.sum(heldout_repair_brier_differences < 0)
                ),
                "mean_heldout_repair_brier_difference": float(
                    heldout_repair_brier_differences.mean()
                ),
                "heldout_repair_brier_difference_range": [
                    float(heldout_repair_brier_differences.min()),
                    float(heldout_repair_brier_differences.max()),
                ],
                "heldout_repair_brier_exact_sign_flip_p": exact_sign_flip_p(
                    heldout_repair_brier_differences
                ),
                "heldout_repair_brier_exact_binomial_sign_p": exact_binomial_sign_p(
                    heldout_repair_brier_differences
                ),
            },
            "frozen_joint_partition": partition(frozen_summary),
            "selected_joint_partition": partition(selected_summary),
            "orbit_average_joint_partition": partition(orbit_average_summary),
            "heldout_selected_joint_partition": partition(
                heldout_selected_summary
            ),
            "heldout_development_pooled_joint_partition": partition(
                heldout_development_pooled_summary
            ),
            "selection_decision_joint_anova": decision_anova,
        }
    destination = HERE / "selection_split_confirmation.json"
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
