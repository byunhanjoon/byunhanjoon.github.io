"""Summarize and diagnose the completed Day 3 invariance matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from cross_model_orbit_gate import load_dataset, official_subsample
from invariance_matrix import build_views, numerical_column_count, render_x


HERE = Path(__file__).resolve().parent
RAW = HERE / "invariance_matrix_results.json"


def flatten(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for dataset_name, dataset in data["cells"].items():
        for family_name, family in dataset["families"].items():
            for factor_name, cell in family.items():
                frozen = cell["frozen"]
                selected = cell["representation_wise_selected"]
                rows.append(
                    {
                        "dataset": dataset_name,
                        "family": family_name,
                        "factor": factor_name,
                        "applicable": cell["applicable"],
                        "view_count": cell["view_count"],
                        "expected_behavior": cell["expected_behavior"],
                        "identity_selected_config": cell["identity_selected_config"],
                        "selected_configs": ";".join(map(str, cell["selected_configs"])),
                        "selection_switch_fraction": cell["selection_switch_fraction"],
                        "validation_maximum_loss_range": cell["validation"]["maximum_loss_range"],
                        "frozen_max_probability_deviation": frozen[
                            "max_probability_deviation_from_identity"
                        ],
                        "frozen_schema_risk": frozen["summary"]["anova"]["total"],
                        "frozen_hard_label_flip_fraction": frozen["summary"][
                            "instance_audit"
                        ]["hard_label_flip_fraction"],
                        "selected_max_probability_deviation": selected[
                            "max_probability_deviation_from_identity"
                        ],
                        "selected_schema_risk": selected["summary"]["anova"]["total"],
                        "selected_hard_label_flip_fraction": selected["summary"][
                            "instance_audit"
                        ]["hard_label_flip_fraction"],
                        "selection_minus_frozen_schema_risk": cell[
                            "selection_minus_frozen_schema_risk"
                        ],
                        "selection_minus_frozen_orbit_mean_brier": cell[
                            "selection_minus_frozen_orbit_mean_brier"
                        ],
                    }
                )
    return rows


def affine_float32_diagnostic(
    data_root: Path,
    dataset_name: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    root = data_root / dataset_name
    x, y, categorical, cardinalities = load_dataset(root)
    numerical_count = numerical_column_count(root)
    train_x, test_x, train_y, _ = official_subsample(
        root,
        x,
        y,
        int(config["design"]["train_size"]),
        int(config["design"]["test_size"]),
    )
    views = build_views(
        "numeric_units",
        train_x,
        train_y,
        categorical,
        cardinalities,
        numerical_count,
        int(config["design"]["nonidentity_views_per_factor"]),
    )
    identity = render_x(train_x, views[0], numerical_count)
    identity_query = render_x(test_x, views[0], numerical_count)
    view_results = []
    for view in views[1:]:
        transformed = render_x(train_x, view, numerical_count)
        transformed_query = render_x(test_x, view, numerical_count)
        boundary_disagreements = 0
        train_unique_count_changes = 0
        query_interval_changes = 0
        for column in range(numerical_count):
            original = identity[:, column]
            changed = transformed[:, column]
            finite = np.isfinite(original) & np.isfinite(changed)
            order = np.argsort(original[finite], kind="mergesort")
            original32 = original[finite][order].astype(np.float32)
            changed32 = changed[finite][order].astype(np.float32)
            if len(original32) > 1:
                original_boundaries = original32[1:] != original32[:-1]
                changed_boundaries = changed32[1:] != changed32[:-1]
                boundary_disagreements += int(
                    np.sum(original_boundaries != changed_boundaries)
                )
            train_unique_count_changes += abs(
                len(np.unique(original32)) - len(np.unique(changed32))
            )

            original_train = np.sort(np.unique(original32))
            changed_train = np.sort(np.unique(changed32))
            original_query = identity_query[:, column].astype(np.float32)
            changed_query = transformed_query[:, column].astype(np.float32)
            original_bins = np.searchsorted(original_train, original_query, side="right")
            changed_bins = np.searchsorted(changed_train, changed_query, side="right")
            valid_query = np.isfinite(original_query) & np.isfinite(changed_query)
            query_interval_changes += int(
                np.sum(original_bins[valid_query] != changed_bins[valid_query])
            )
        view_results.append(
            {
                "view": view.name,
                "float32_training_boundary_disagreements": boundary_disagreements,
                "absolute_train_unique_count_change": train_unique_count_changes,
                "test_rows_with_changed_train_rank_interval_summed_over_columns": query_interval_changes,
            }
        )
    return {
        "numerical_columns": numerical_count,
        "views": view_results,
        "any_float32_training_boundary_change": any(
            item["float32_training_boundary_disagreements"] > 0
            for item in view_results
        ),
        "any_test_rank_interval_change": any(
            item["test_rows_with_changed_train_rank_interval_summed_over_columns"] > 0
            for item in view_results
        ),
    }


def aggregate(rows: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    applicable = [row for row in rows if row["applicable"]]
    exact = [row for row in applicable if row["frozen_max_probability_deviation"] <= tolerance]
    practical = [
        row for row in applicable if row["frozen_max_probability_deviation"] <= 1e-6
    ]
    switched = [row for row in applicable if row["selection_switch_fraction"] > 0]
    by_factor = {}
    for factor in sorted({row["factor"] for row in applicable}):
        factor_rows = [row for row in applicable if row["factor"] == factor]
        by_factor[factor] = {
            "cells": len(factor_rows),
            "within_preregistered_tolerance": sum(
                row["frozen_max_probability_deviation"] <= tolerance
                for row in factor_rows
            ),
            "within_practical_1e-6_tolerance": sum(
                row["frozen_max_probability_deviation"] <= 1e-6
                for row in factor_rows
            ),
            "selection_switch_cells": sum(
                row["selection_switch_fraction"] > 0 for row in factor_rows
            ),
        }
    return {
        "applicable_cells": len(applicable),
        "within_preregistered_tolerance": len(exact),
        "within_practical_1e-6_tolerance": len(practical),
        "selection_switch_cells": len(switched),
        "by_factor": by_factor,
        "switching_cells": [
            {
                "dataset": row["dataset"],
                "family": row["family"],
                "factor": row["factor"],
                "switch_fraction": row["selection_switch_fraction"],
            }
            for row in switched
        ],
    }


def markdown_report(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    float32: dict[str, Any],
    tolerance: float,
) -> str:
    family_labels = {
        "onehot_logistic": "one-hot logistic",
        "ordinal_forest_sqrt": "ordinal forest (sqrt)",
        "ordinal_forest_full": "ordinal forest (all)",
        "onehot_forest_full": "one-hot forest (all)",
        "native_histgb": "native HistGB",
        "catboost_native": "native CatBoost",
    }
    factor_labels = {
        "feature_order": "feature order",
        "category_ids": "category IDs",
        "class_ids": "class IDs",
        "numeric_units": "numeric units",
    }
    lines = [
        "# Day 3 invariance matrix report",
        "",
        "This extension varies one declared nuisance at a time, first with the",
        "identity-selected configuration frozen and then with validation selection",
        "rerun in every view. The preregistered numerical tolerance is",
        f"`{tolerance:g}`; `1e-6` is reported separately as a practical scale, not",
        "as a replacement threshold.",
        "",
        "## Frozen-fit maximum aligned probability deviations",
        "",
        "Each entry is the maximum absolute change from the identity view across",
        "evaluation rows and aligned output classes. A dash means the factor is not",
        "applicable (Otto has no categorical fields).",
        "",
    ]
    for dataset in ("adult", "churn", "otto"):
        lines.extend(
            [
                f"### {dataset.title()}",
                "",
                "| pipeline | feature order | category IDs | class IDs | numeric units |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for family in family_labels:
            values = []
            for factor in factor_labels:
                row = next(
                    item
                    for item in rows
                    if item["dataset"] == dataset
                    and item["family"] == family
                    and item["factor"] == factor
                )
                values.append(
                    f"{row['frozen_max_probability_deviation']:.3g}"
                    if row["applicable"]
                    else "—"
                )
            lines.append(f"| {family_labels[family]} | " + " | ".join(values) + " |")
        lines.append("")

    lines.extend(
        [
            "## Selection switches",
            "",
            "| dataset | pipeline | nuisance | fraction of views differing from identity choice |",
            "|---|---|---|---:|",
        ]
    )
    for item in summary["switching_cells"]:
        lines.append(
            f"| {item['dataset'].title()} | {family_labels[item['family']]} | "
            f"{factor_labels[item['factor']]} | {item['switch_fraction']:.0%} |"
        )
    if not summary["switching_cells"]:
        lines.append("| — | — | — | 0% |")

    lines.extend(
        [
            "",
            "## Main reading",
            "",
            "- Native CatBoost and HistGB are exactly invariant to the tested category-ID permutations on Adult and Churn. The ordinal-code forests are not.",
            "- One-hot logistic is invariant to machine precision on the binary tasks and within `6e-7` on multiclass Otto; it never changes its selected regularization.",
            "- Feature order changes seeded forests and CatBoost, and changes multiclass HistGB on Otto even when the chosen configuration remains fixed.",
            "- Binary class-ID reversal changes HistGB and CatBoost fits after output alignment, while the forests remain invariant. Multiclass Otto is invariant for both boosting pipelines in this menu.",
            "- HistGB is exactly invariant to every tested positive affine unit change. Standardized one-hot forests are also exact; raw forests are not, despite the tree hypothesis class admitting the same partitions.",
            "",
            "## Float32 unit diagnostic",
            "",
            "One possible mechanism for the raw-forest unit result is finite-precision",
            "split generation. The following diagnostic asks only whether casting the",
            "affine rewrite to float32 changes a training equality boundary or a",
            "test-to-training rank interval.",
            "",
            "| dataset | changed float32 training boundary | changed test rank interval |",
            "|---|---:|---:|",
        ]
    )
    for dataset, diagnostic in float32.items():
        lines.append(
            f"| {dataset.title()} | {str(diagnostic['any_float32_training_boundary_change']).lower()} | "
            f"{str(diagnostic['any_test_rank_interval_change']).lower()} |"
        )
    lines.extend(
        [
            "",
            "The diagnostic finds direct float32 rank changes only on Churn. It does",
            "not explain the Adult or Otto changes, so threshold arithmetic, tied split",
            "selection, and other implementation details remain hypotheses rather than",
            "established mechanisms.",
            "",
            "## Boundary of the claim",
            "",
            "The matrix is a controlled three-dataset audit, not an estimate of how",
            "often model families fail these invariances in the wild. The correct",
            "claim is transformation-specific: native categorical handling absorbed",
            "category renaming here, while other schema choices still reached fitting",
            "and selection.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=RAW)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=HERE.parent / "road-to-iclr-day-01" / "data",
    )
    parser.add_argument("--csv", type=Path, default=HERE / "invariance_matrix_summary.csv")
    parser.add_argument("--json", type=Path, default=HERE / "invariance_matrix_summary.json")
    parser.add_argument("--report", type=Path, default=HERE / "INVARIANCE_MATRIX_REPORT.md")
    args = parser.parse_args()

    data = json.loads(args.input.read_text())
    rows = flatten(data)
    tolerance = float(data["config"]["design"]["invariance_tolerance"])
    summary = aggregate(rows, tolerance)
    float32 = {
        dataset: affine_float32_diagnostic(args.data_root, dataset, data["config"])
        for dataset in data["config"]["datasets"]
    }
    result = {
        "status": "completed_day3_invariance_matrix_analysis",
        "source": args.input.name,
        "preregistered_tolerance": tolerance,
        "aggregate": summary,
        "affine_float32_diagnostic": float32,
    }
    with args.csv.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    args.report.write_text(markdown_report(rows, summary, float32, tolerance))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
