"""Analyze H1 paired semantic-arithmetic trajectories and frozen gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import semantic_arithmetic as saa

HERE = Path(__file__).resolve().parent


def rows_from_artifact(path: Path) -> tuple[list[dict], list[dict]]:
    bundle = np.load(path)
    labels = bundle["labels"]
    validation = bundle["validation_predictions"]
    test = bundle["test_predictions"]
    checkpoints = bundle["checkpoints"].astype(int)
    validation_y, test_y = bundle["validation_y"], bundle["test_y"]
    manifest = json.loads(path.with_suffix(".json").read_text())
    trajectory_rows: list[dict] = []
    reference_rows: list[dict] = []
    for precision in ("fp32", "iea64"):
        indices = np.flatnonzero(labels[:, 0] == precision)
        reference_index = next(index for index in indices if int(labels[index, 1]) == 0)
        reference_validation = validation[reference_index]
        reference_test = test[reference_index]
        for checkpoint_index, checkpoint in enumerate(checkpoints):
            reference_rows.append({
                "dataset": manifest["dataset"], "task": manifest["task"],
                "model": manifest["model"], "seed": manifest["seed"],
                "precision": precision, "checkpoint": checkpoint,
                "validation_loss": saa.proper_loss(
                    reference_validation[checkpoint_index], validation_y, manifest["task"]
                ),
                "test_loss": saa.proper_loss(
                    reference_test[checkpoint_index], test_y, manifest["task"]
                ),
            })
        for index in indices:
            action = int(labels[index, 1])
            if action == 0:
                continue
            for checkpoint_index, checkpoint in enumerate(checkpoints):
                validation_delta = (
                    validation[index, checkpoint_index] - reference_validation[checkpoint_index]
                ).astype(np.float64)
                test_delta = (
                    test[index, checkpoint_index] - reference_test[checkpoint_index]
                ).astype(np.float64)
                initial_delta = (
                    validation[index, 0] - reference_validation[0]
                ).astype(np.float64)
                initial_mse = float(np.mean(initial_delta**2))
                trajectory_rows.append({
                    "dataset": manifest["dataset"], "task": manifest["task"],
                    "model": manifest["model"], "seed": manifest["seed"],
                    "precision": precision, "action": action,
                    "feature_view": int(labels[index, 2]),
                    "category_view": int(labels[index, 3]),
                    "checkpoint": checkpoint,
                    "validation_prediction_mse": float(np.mean(validation_delta**2)),
                    "validation_max_gap": float(np.max(np.abs(validation_delta))),
                    "test_prediction_mse": float(np.mean(test_delta**2)),
                    "test_max_gap": float(np.max(np.abs(test_delta))),
                    "initial_validation_mse": initial_mse,
                    "amplification": float(
                        np.mean(validation_delta**2) / max(initial_mse, 1e-30)
                    ),
                    "validation_loss": saa.proper_loss(
                        validation[index, checkpoint_index], validation_y, manifest["task"]
                    ),
                    "reference_validation_loss": saa.proper_loss(
                        reference_validation[checkpoint_index], validation_y, manifest["task"]
                    ),
                    "test_loss": saa.proper_loss(
                        test[index, checkpoint_index], test_y, manifest["task"]
                    ),
                    "reference_test_loss": saa.proper_loss(
                        reference_test[checkpoint_index], test_y, manifest["task"]
                    ),
                })
    return trajectory_rows, reference_rows


def paired_bootstrap_log_ratio(cells: pd.DataFrame, draws: int = 10000) -> list[float]:
    datasets = sorted(cells["dataset"].unique())
    by_dataset = {
        dataset: cells[cells["dataset"] == dataset] for dataset in datasets
    }
    rng = np.random.default_rng(2026082861)
    estimates = []
    for _ in range(draws):
        sampled = rng.choice(datasets, len(datasets), replace=True)
        values = []
        for dataset in sampled:
            frame = by_dataset[dataset]
            ratio = (frame["iea64_mse"] + 1e-30) / (frame["fp32_mse"] + 1e-30)
            values.append(float(np.log(ratio).mean()))
        estimates.append(float(np.mean(values)))
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def analyze(input_dir: Path, output_dir: Path) -> dict:
    artifacts = sorted(input_dir.glob("*.npz"))
    if not artifacts:
        raise FileNotFoundError(f"no H1 artifacts under {input_dir}")
    trajectory_rows, reference_rows = [], []
    for artifact in artifacts:
        current, references = rows_from_artifact(artifact)
        trajectory_rows.extend(current)
        reference_rows.extend(references)
    trajectories = pd.DataFrame(trajectory_rows)
    references = pd.DataFrame(reference_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories.to_csv(output_dir / "h1_trajectories.csv", index=False)
    references.to_csv(output_dir / "h1_references.csv", index=False)

    final_checkpoint = int(trajectories["checkpoint"].max())
    final = trajectories[trajectories["checkpoint"] == final_checkpoint]
    means = final.groupby(
        ["dataset", "model", "seed", "precision"], as_index=False
    ).agg(
        prediction_mse=("validation_prediction_mse", "mean"),
        maximum_gap=("validation_max_gap", "max"),
        amplification=("amplification", "max"),
        test_loss=("test_loss", "mean"),
        reference_test_loss=("reference_test_loss", "mean"),
    )
    pivot = means.pivot(
        index=["dataset", "model", "seed"], columns="precision",
        values=["prediction_mse", "maximum_gap", "amplification", "test_loss", "reference_test_loss"],
    )
    pivot.columns = [f"{precision}_{metric}" for metric, precision in pivot.columns]
    paired = pivot.reset_index()
    paired["relative_reduction"] = 1.0 - (
        (paired["iea64_prediction_mse"] + 1e-30)
        / (paired["fp32_prediction_mse"] + 1e-30)
    )
    paired["iea64_relative_path_test_loss_change"] = (
        paired["iea64_test_loss"] - paired["iea64_reference_test_loss"]
    ) / paired["iea64_reference_test_loss"].clip(lower=1e-30)
    paired.to_csv(output_dir / "h1_seed_pairs.csv", index=False)

    cells = paired.groupby(["dataset", "model"], as_index=False).agg(
        fp32_mse=("fp32_prediction_mse", "mean"),
        iea64_mse=("iea64_prediction_mse", "mean"),
        mean_relative_reduction=("relative_reduction", "mean"),
        fp32_maximum_amplification=("fp32_amplification", "max"),
        maximum_abs_iea64_path_test_loss_change=(
            "iea64_relative_path_test_loss_change", lambda values: float(np.max(np.abs(values)))
        ),
        seeds=("seed", "nunique"),
    )
    cells["win"] = cells["iea64_mse"] < cells["fp32_mse"]
    cells.to_csv(output_dir / "h1_cells.csv", index=False)
    cell_ratio = (cells["iea64_mse"] + 1e-30) / (cells["fp32_mse"] + 1e-30)
    geometric_reduction = float(1.0 - np.exp(np.log(cell_ratio).mean()))
    config = json.loads((HERE / "hypothesis_01_config.json").read_text())
    gates = config["gates"]
    pilot_complete = all(
        set(config["pilot_seeds"]).issubset(
            set(paired[(paired.dataset == dataset) & (paired.model == model)].seed)
        )
        for dataset in config["datasets"] for model in config["models"]
    )
    cell_wins = int(cells["win"].sum())
    amplified_cells = int(
        (cells["fp32_maximum_amplification"] > float(gates["amplification_threshold"])).sum()
    )
    max_path_loss_change = float(cells["maximum_abs_iea64_path_test_loss_change"].max())
    gate_results = {
        "cell_wins": {
            "value": cell_wins, "required": int(gates["minimum_cell_wins"]),
            "pass": cell_wins >= int(gates["minimum_cell_wins"]),
        },
        "geometric_mean_reduction": {
            "value": geometric_reduction,
            "required": float(gates["minimum_geometric_mean_reduction"]),
            "pass": geometric_reduction >= float(gates["minimum_geometric_mean_reduction"]),
        },
        "maximum_iea64_path_test_loss_change": {
            "value": max_path_loss_change,
            "required": float(gates["maximum_relative_test_loss_change"]),
            "pass": max_path_loss_change <= float(gates["maximum_relative_test_loss_change"]),
        },
        "amplified_cells": {
            "value": amplified_cells, "required": int(gates["minimum_amplified_cells"]),
            "pass": amplified_cells >= int(gates["minimum_amplified_cells"]),
        },
    }
    confirmation_complete = all(
        set(config["confirmation_seeds"]).issubset(
            set(paired[(paired.dataset == dataset) & (paired.model == model)].seed)
        )
        for dataset in config["datasets"] for model in config["models"]
    )
    confirmation = paired[paired.seed.isin(config["confirmation_seeds"])]
    confirmation_cells = confirmation.groupby(["dataset", "model"], as_index=False).agg(
        fp32_mse=("fp32_prediction_mse", "mean"),
        iea64_mse=("iea64_prediction_mse", "mean"),
    ) if len(confirmation) else pd.DataFrame()
    confirmation_wins = int(
        (confirmation_cells.iea64_mse < confirmation_cells.fp32_mse).sum()
    ) if len(confirmation_cells) else 0
    confirmation_interval = (
        paired_bootstrap_log_ratio(confirmation_cells)
        if confirmation_complete else None
    )
    confirmation_pass = bool(
        confirmation_complete and confirmation_wins >= 7
        and confirmation_interval is not None and confirmation_interval[1] < 0
    )
    summary = {
        "status": (
            "confirmation_complete" if confirmation_complete
            else "pilot_complete" if pilot_complete else "pilot_in_progress"
        ),
        "artifacts": len(artifacts), "dataset_model_cells": len(cells),
        "seeds": sorted(int(value) for value in paired.seed.unique()),
        "final_checkpoint": final_checkpoint,
        "gate_results": gate_results,
        "all_frozen_pilot_gates_pass": bool(
            pilot_complete and all(item["pass"] for item in gate_results.values())
        ),
        "confirmation": {
            "complete": confirmation_complete,
            "cell_wins": confirmation_wins,
            "required_cell_wins": 7,
            "dataset_block_bootstrap_log_risk_ratio_95_interval": confirmation_interval,
            "pass": confirmation_pass,
        },
        "dataset_block_bootstrap_log_risk_ratio_95_interval": (
            paired_bootstrap_log_ratio(cells) if len(cells.dataset.unique()) == 3 else None
        ),
        "zero_iea64_cells": int((cells["iea64_mse"] == 0).sum()),
    }
    (output_dir / "h1_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=HERE / "results" / "h1")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    analyze(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
