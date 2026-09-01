"""Put schema risk on interpretable relative, root, and seed-noise scales."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def read(name: str) -> dict:
    return json.loads((HERE / name).read_text())


def conventional_rows() -> list[dict]:
    rows = []
    for filename in (
        "cross_model_orbit_adult.json",
        "cross_model_orbit_churn.json",
        "cross_model_orbit_otto.json",
    ):
        data = read(filename)
        for model, result in data["models"].items():
            risk = float(result["anova"]["total"])
            mean_loss = float(result["mean_member_brier"])
            seed_risk = float(result["reference_seed_orbit"]["anova"]["total"])
            rows.append(
                {
                    "dataset": data["dataset"],
                    "pipeline": model,
                    "schema_risk": risk,
                    "root_schema_risk": float(np.sqrt(risk)),
                    "schema_risk_percent_mean_member_loss": 100 * risk / mean_loss,
                    "reference_seed_risk": seed_risk,
                    "schema_to_seed_risk_ratio": risk / seed_risk if seed_risk > 1e-20 else None,
                    "hard_label_flip_percent": 100
                    * float(result["instance_audit"]["hard_label_flip_fraction"]),
                    "material": risk > 1e-10,
                }
            )
    return rows


def neural_rows() -> list[dict]:
    adult_mlp = read("chart_orbit_adult_s32.json")["mlp"]["brier"]
    adult_resnet = read("adult_architecture_chart.json")["resnet_orbit"]
    diamond_mlp = read("chart_regression_diamond.json")["raw_adamw"]["summary"]
    diamond_resnet = read("diamond_architecture_chart.json")["resnet_orbit"]
    black_friday = read("chart_orbit_black_friday.json")["orbit"]
    records = [
        ("adult", "mlp", adult_mlp, "mean_member_brier", "brier_reduction_by_averaging"),
        ("adult", "resnet", adult_resnet, "mean_member_brier", "brier_reduction_by_averaging"),
        ("diamond", "mlp", diamond_mlp, "mean_member_mse_standardized", "mse_reduction_by_averaging"),
        ("diamond", "resnet", diamond_resnet, "mean_member_mse_standardized", "mse_reduction_by_averaging"),
        ("black-friday", "mlp", black_friday, "mean_member_mse_standardized", "mse_reduction_by_averaging"),
    ]
    coupling = {
        ("adult", "mlp"): read("chart_seed_coupling_s32.json"),
        ("diamond", "mlp"): read("chart_seed_coupling_diamond_mlp.json"),
        ("diamond", "resnet"): read("chart_seed_coupling_diamond_resnet.json"),
        ("black-friday", "mlp"): read("chart_seed_coupling_black_friday.json"),
    }
    rows = []
    for dataset, model, result, loss_key, total_key in records:
        anova = result["anova"]
        same_seed_schema = float(anova["chart"] + anova["chart:seed"])
        mean_loss = float(result[loss_key])
        persistent = None
        if (dataset, model) in coupling:
            source = coupling[(dataset, model)]
            # The files use a stable descriptive key but retain a fallback for
            # the earliest Adult artifact.
            persistent = source.get("persistent_mean_predictor_chart_risk")
            if persistent is None:
                persistent = source.get("persistent_chart_risk", {}).get("unbiased_u_statistic")
            if persistent is None:
                persistent = source.get("summary", {}).get("persistent_mean_predictor_chart_risk")
            if persistent is None:
                persistent = source.get("persistent_mean_schema_risk", {}).get(
                    "unbiased_u_statistic"
                )
        rows.append(
            {
                "dataset": dataset,
                "pipeline": model,
                "joint_schema_seed_risk": float(result[total_key]),
                "same_seed_schema_risk": same_seed_schema,
                "persistent_schema_risk": float(persistent) if persistent is not None else None,
                "root_same_seed_schema_risk": float(np.sqrt(same_seed_schema)),
                "same_seed_schema_percent_mean_member_loss": 100 * same_seed_schema / mean_loss,
                "joint_risk_percent_mean_member_loss": 100 * float(result[total_key]) / mean_loss,
                "schema_seed_interaction_fraction_joint": float(anova["chart:seed"] / anova["total"]),
                "hard_flip_percent": (
                    100 * float(result["instance_audit"]["hard_label_flip_fraction"])
                    if "instance_audit" in result
                    else None
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "effect_size_calibration.json")
    args = parser.parse_args()
    conventional = conventional_rows()
    neural = neural_rows()
    material = [row for row in conventional if row["material"]]
    relative = np.asarray([row["schema_risk_percent_mean_member_loss"] for row in material])
    neural_relative = np.asarray(
        [row["same_seed_schema_percent_mean_member_loss"] for row in neural]
    )
    result = {
        "conventional_summary": {
            "cells": len(conventional),
            "material_cells": len(material),
            "datasets_with_material_cell": len({row["dataset"] for row in material}),
            "relative_loss_tax_percent_median": float(np.median(relative)),
            "relative_loss_tax_percent_range": [float(relative.min()), float(relative.max())],
            "hard_flip_percent_range": [
                min(row["hard_label_flip_percent"] for row in material),
                max(row["hard_label_flip_percent"] for row in material),
            ],
        },
        "neural_summary": {
            "cells": len(neural),
            "same_seed_relative_loss_tax_percent_median": float(np.median(neural_relative)),
            "same_seed_relative_loss_tax_percent_range": [
                float(neural_relative.min()), float(neural_relative.max())
            ],
            "schema_seed_interaction_fraction_joint_range": [
                min(row["schema_seed_interaction_fraction_joint"] for row in neural),
                max(row["schema_seed_interaction_fraction_joint"] for row in neural),
            ],
        },
        "interpretation_warning": "relative tax is the exact proper-loss gain from quotient averaging versus the mean member, not excess risk versus Bayes and not an accuracy-loss estimate",
        "conventional": conventional,
        "neural": neural,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
