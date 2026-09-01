"""Analyze H3 all-row long-horizon closure and practical gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_semantic_arithmetic as h1_analysis

HERE = Path(__file__).resolve().parent


def analyze(input_dir: Path, output_dir: Path) -> dict:
    config = json.loads((HERE / "hypothesis_03_config.json").read_text())
    trajectory_rows, reference_rows, timing_rows = [], [], []
    artifacts = sorted(input_dir.glob("*.npz"))
    if not artifacts:
        raise FileNotFoundError(f"no H3 artifacts under {input_dir}")
    for artifact in artifacts:
        current, references = h1_analysis.rows_from_artifact(artifact)
        trajectory_rows.extend(current); reference_rows.extend(references)
        manifest = json.loads(artifact.with_suffix(".json").read_text())
        half = len(manifest["path_wall_seconds"]) // 2
        for index, seconds in enumerate(manifest["path_wall_seconds"]):
            timing_rows.append({
                "dataset": manifest["dataset"], "model": manifest["model"],
                "seed": manifest["seed"],
                "precision": "fp32" if index < half else "iea64",
                "seconds": float(seconds),
            })
    trajectories = pd.DataFrame(trajectory_rows)
    references = pd.DataFrame(reference_rows)
    timing = pd.DataFrame(timing_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories.to_csv(output_dir / "h3_trajectories.csv", index=False)
    references.to_csv(output_dir / "h3_references.csv", index=False)
    timing.to_csv(output_dir / "h3_timing.csv", index=False)

    final_checkpoint = int(trajectories.checkpoint.max())
    final = trajectories[trajectories.checkpoint == final_checkpoint]
    cells = final.groupby(["dataset", "model", "precision"], as_index=False).agg(
        mean_final_mse=("validation_prediction_mse", "mean"),
        maximum_final_mse=("validation_prediction_mse", "max"),
        paths=("action", "size"),
    )
    exact = trajectories.groupby(["dataset", "model", "precision"], as_index=False).agg(
        maximum_any_checkpoint_mse=("validation_prediction_mse", "max")
    )
    cells = cells.merge(exact, on=["dataset", "model", "precision"])
    cells.to_csv(output_dir / "h3_cells.csv", index=False)

    iea_cells = cells[cells.precision == "iea64"]
    exact_iea_cells = int((iea_cells.maximum_any_checkpoint_mse == 0).sum())
    ft_fp32 = cells[(cells.precision == "fp32") & (cells.model == "ft_transformer")]
    material_ft = int((ft_fp32.mean_final_mse > 1e-5).sum())
    stable = cells[
        (cells.precision == "fp32") & cells.model.isin(["mlp", "resnet"])
    ]
    stable_cells = int((stable.maximum_final_mse < 1e-8).sum())

    expected_iea_cells = len(config["datasets"]) * len(config["models"])
    failed_iea_cells = int((iea_cells.maximum_any_checkpoint_mse > 0).sum())
    maximum_possible_exact_iea_cells = expected_iea_cells - failed_iea_cells
    expected_stable_cells = len(config["datasets"]) * 2
    failed_stable_cells = int((stable.maximum_final_mse >= 1e-8).sum())
    maximum_possible_stable_cells = expected_stable_cells - failed_stable_cells
    successful_ft_datasets = set(
        ft_fp32.loc[ft_fp32.mean_final_mse > 1e-5, "dataset"]
    )
    ft_seed_counts = trajectories[
        (trajectories.precision == "fp32")
        & (trajectories.model == "ft_transformer")
    ].groupby("dataset").seed.nunique()
    completed_ft_datasets = {
        dataset for dataset, count in ft_seed_counts.items()
        if count == len(config["seeds"])
    }
    definitely_failed_ft = completed_ft_datasets - successful_ft_datasets
    maximum_possible_material_ft = len(config["datasets"]) - len(definitely_failed_ft)

    timing_summary = timing.groupby(["model", "precision"]).seconds.median().unstack()
    timing_ratios = {
        model: float(row.iea64 / row.fp32)
        for model, row in timing_summary.iterrows()
    }
    timing_models_pass = int(sum(value <= 1.25 for value in timing_ratios.values()))

    refs = references[references.checkpoint == final_checkpoint].pivot(
        index=["dataset", "model", "seed"], columns="precision", values="test_loss"
    ).reset_index()
    refs["relative_loss_change"] = (refs.iea64 - refs.fp32) / refs.fp32.clip(lower=1e-30)
    loss_by_dataset = refs.groupby("dataset").relative_loss_change.mean()
    equal_dataset_loss_change = float(loss_by_dataset.mean())
    refs.to_csv(output_dir / "h3_reference_loss_pairs.csv", index=False)
    complete = len(artifacts) == (
        len(config["datasets"]) * len(config["models"]) * len(config["seeds"])
    )
    gates = {
        "exact_iea64_cells": {
            "value": exact_iea_cells, "required": 8,
            "pass": exact_iea_cells >= 8,
        },
        "material_ft_fp32_datasets": {
            "value": material_ft, "required": 2,
            "pass": material_ft >= 2,
        },
        "stable_mlp_resnet_cells": {
            "value": stable_cells, "required": 5,
            "pass": stable_cells >= 5,
        },
        "timing_overhead": {
            "models_pass": timing_models_pass, "required": 3,
            "ratios_iea64_over_fp32": timing_ratios,
            "pass": timing_models_pass == 3,
        },
        "canonical_test_loss_change": {
            "value": equal_dataset_loss_change, "required_interval": [-0.01, 0.01],
            "pass": -0.01 <= equal_dataset_loss_change <= 0.01,
            "dataset_means": {key: float(value) for key, value in loss_by_dataset.items()},
        },
    }
    summary = {
        "status": "complete" if complete else "in_progress",
        "artifacts": len(artifacts), "final_checkpoint": final_checkpoint,
        "gates": gates,
        "logical_reachability": {
            "exact_iea64_cells": {
                "failed_cells_observed": failed_iea_cells,
                "maximum_possible_final_value": maximum_possible_exact_iea_cells,
                "required": 8,
                "reachable": maximum_possible_exact_iea_cells >= 8,
            },
            "stable_mlp_resnet_cells": {
                "failed_cells_observed": failed_stable_cells,
                "maximum_possible_final_value": maximum_possible_stable_cells,
                "required": 5,
                "reachable": maximum_possible_stable_cells >= 5,
            },
            "material_ft_fp32_datasets": {
                "successful_datasets_observed": len(successful_ft_datasets),
                "maximum_possible_final_value": maximum_possible_material_ft,
                "required": 2,
                "reachable": maximum_possible_material_ft >= 2,
            },
            "hypothesis_already_falsified": bool(
                maximum_possible_exact_iea_cells < 8
                or maximum_possible_stable_cells < 5
                or maximum_possible_material_ft < 2
            ),
            "note": "timing and equal-dataset loss gates are not monotone under partial bundles",
        },
        "hypothesis_supported": bool(complete and all(value["pass"] for value in gates.values())),
        "summed_fit_hours": float(timing.seconds.sum() / 3600),
    }
    (output_dir / "h3_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=HERE / "results" / "h3")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    analyze(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
