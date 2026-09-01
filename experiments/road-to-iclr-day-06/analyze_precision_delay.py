"""Analyze H2 precision-delay hitting times and frozen gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent


def analyze(input_dir: Path, output_dir: Path) -> dict:
    config = json.loads((HERE / "hypothesis_02_config.json").read_text())
    threshold = float(config["hitting_threshold"])
    rows = []
    artifacts = sorted(input_dir.glob("*.npz"))
    if not artifacts:
        raise FileNotFoundError(f"no H2 artifacts under {input_dir}")
    for path in artifacts:
        bundle = np.load(path)
        manifest = json.loads(path.with_suffix(".json").read_text())
        labels, predictions = bundle["labels"], bundle["validation_predictions"]
        checkpoints = bundle["checkpoints"].astype(int)
        for precision in config["precisions"]:
            indices = np.flatnonzero(labels[:, 0] == precision)
            reference_index = next(index for index in indices if int(labels[index, 1]) == 0)
            reference = predictions[reference_index]
            for index in indices:
                action = int(labels[index, 1])
                if action == 0:
                    continue
                mse = np.mean(
                    (predictions[index].astype(np.float64) - reference.astype(np.float64)) ** 2,
                    axis=(1, 2),
                )
                hitting = next(
                    (int(epoch) for epoch, value in zip(checkpoints, mse) if value >= threshold),
                    int(checkpoints[-1]) + 1,
                )
                rows.append({
                    "dataset": manifest["dataset"], "model": manifest["model"],
                    "seed": manifest["seed"], "precision": precision,
                    "action": action, "initial_mse": float(mse[0]),
                    "final_mse": float(mse[-1]), "hitting_epoch": hitting,
                    "exact_all_checkpoints": bool(np.all(mse == 0)),
                })
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "h2_paths.csv", index=False)
    cells = frame.groupby(["dataset", "model", "precision"], as_index=False).agg(
        median_hitting_epoch=("hitting_epoch", "median"),
        mean_final_mse=("final_mse", "mean"),
        maximum_final_mse=("final_mse", "max"),
        exact_paths=("exact_all_checkpoints", "sum"),
        paths=("action", "size"),
    )
    cells.to_csv(output_dir / "h2_cells.csv", index=False)

    ft = cells[cells.model == "ft_transformer"]
    ordered_datasets = []
    order = config["precisions"]
    for dataset in config["datasets"]:
        current = ft[ft.dataset == dataset].set_index("precision").median_hitting_epoch
        if all(value in current.index for value in order):
            values = [float(current[value]) for value in order]
            if values[0] <= values[1] <= values[2] < values[3]:
                ordered_datasets.append(dataset)
    precision_medians = frame[frame.model == "ft_transformer"].groupby(
        "precision"
    ).hitting_epoch.median()
    available = [value for value in order if value in precision_medians.index]
    stability_rho = float(spearmanr(
        [-np.log2(float(config["unit_roundoff"][value])) for value in available],
        [float(precision_medians[value]) for value in available],
    ).statistic) if len(available) == 4 else float("nan")
    float64_cells = cells[cells.precision == "float64"]
    exact_float64_cells = int((float64_cells.maximum_final_mse == 0).sum())
    stable = cells[
        (cells.model.isin(["mlp", "resnet"])) & (cells.precision == "float32")
    ]
    stable_cells = int(
        (stable.maximum_final_mse < float(config["stable_boundary_threshold"])).sum()
    )
    complete = len(artifacts) == (
        len(config["datasets"]) * len(config["models"]) * len(config["seeds"])
    )
    gates = {
        "ordered_ft_datasets": {
            "value": len(ordered_datasets), "required": 2,
            "pass": len(ordered_datasets) >= 2,
            "datasets": ordered_datasets,
        },
        "precision_hitting_spearman": {
            "value": stability_rho, "required": 0.8,
            "pass": bool(stability_rho >= 0.8),
            "precision_median_hitting_epochs": {
                value: float(precision_medians[value]) for value in available
            },
        },
        "exact_float64_cells": {
            "value": exact_float64_cells, "required": 9,
            "pass": exact_float64_cells == 9,
        },
        "stable_fp32_control_cells": {
            "value": stable_cells, "required": 5,
            "pass": stable_cells >= 5,
        },
    }
    summary = {
        "status": "complete" if complete else "in_progress",
        "artifacts": len(artifacts), "paths": len(frame),
        "gates": gates,
        "hypothesis_supported": bool(complete and all(value["pass"] for value in gates.values())),
    }
    (output_dir / "h2_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=HERE / "results" / "h2")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    analyze(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
