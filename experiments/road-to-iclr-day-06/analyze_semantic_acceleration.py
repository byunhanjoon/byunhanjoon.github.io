"""Analyze the prospectively split H8 level-or-acceleration screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_semantic_arithmetic as h1_analysis
import audit_day6_integrity as integrity

HERE = Path(__file__).resolve().parent


def safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else float("nan")


def analyze(input_dir: Path, output_dir: Path) -> dict:
    h3 = json.loads((HERE / "hypothesis_03_config.json").read_text())
    config = json.loads((HERE / "hypothesis_08_config.json").read_text())
    h6_config = json.loads((HERE / "hypothesis_06_config.json").read_text())
    development = set(config["development_stems"])
    floor = float(config["log_floor"])
    threshold = float(config["material_mse"])
    rows = []
    for artifact in sorted(input_dir.glob("*.npz")):
        if artifact.stem in development:
            continue
        trajectories, _ = h1_analysis.rows_from_artifact(artifact)
        frame = pd.DataFrame(trajectories)
        frame = frame[frame.precision == "fp32"]
        curve = frame.groupby("checkpoint").validation_prediction_mse.mean()
        logs = {epoch: float(np.log10(curve[epoch] + floor)) for epoch in (5, 10, 20)}
        early_slope = (logs[10] - logs[5]) / 5.0
        late_slope = (logs[20] - logs[10]) / 10.0
        acceleration = late_slope - early_slope
        level_branch = logs[20] > float(config["level_log_threshold"])
        acceleration_branch = acceleration > float(config["acceleration_threshold"])
        final_mse = float(curve[int(config["final_checkpoint"])])
        h6_logs = np.asarray([
            np.log10(curve[epoch] + float(h6_config["log_floor"]))
            for epoch in h6_config["early_checkpoints"]
        ])
        h6_slope, h6_intercept = np.polyfit(
            np.asarray(h6_config["early_checkpoints"], dtype=float),
            h6_logs, deg=1,
        )
        h6_score = float(
            h6_intercept + float(h6_config["final_checkpoint"]) * h6_slope
        )
        manifest = json.loads(artifact.with_suffix(".json").read_text())
        rows.append({
            "stem": artifact.stem, "dataset": manifest["dataset"],
            "model": manifest["model"], "seed": int(manifest["seed"]),
            "log_mse_5": logs[5], "log_mse_10": logs[10],
            "log_mse_20": logs[20], "early_slope": early_slope,
            "late_slope": late_slope, "acceleration": acceleration,
            "level_branch": bool(level_branch),
            "acceleration_branch": bool(acceleration_branch),
            "predicted_material": bool(level_branch or acceleration_branch),
            "h6_predicted_material": bool(
                h6_score > np.log10(float(h6_config["material_final_mse"]))
            ),
            "h6_score": h6_score, "final_mse": final_mse,
            "material": bool(final_mse > threshold),
            "delayed_material": bool(
                logs[20] <= float(config["level_log_threshold"])
                and final_mse > threshold
            ),
        })
    columns = [
        "stem", "dataset", "model", "seed", "log_mse_5", "log_mse_10",
        "log_mse_20", "early_slope", "late_slope", "acceleration",
        "level_branch", "acceleration_branch", "predicted_material",
        "h6_predicted_material", "h6_score", "final_mse", "material",
        "delayed_material",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "h8_prospective_bundles.csv", index=False)

    expected = 36 - len(development)
    integrity_errors, integrity_summary = integrity.audit_family("h3", input_dir)
    complete = len(frame) == expected
    tp = int((frame.material & frame.predicted_material).sum())
    fn = int((frame.material & ~frame.predicted_material).sum())
    tn = int((~frame.material & ~frame.predicted_material).sum())
    fp = int((~frame.material & frame.predicted_material).sum())
    sensitivity = safe_rate(tp, tp + fn)
    specificity = safe_rate(tn, tn + fp)
    balanced_accuracy = (sensitivity + specificity) / 2.0
    accuracy = float((frame.material == frame.predicted_material).mean()) if len(frame) else float("nan")
    h6_accuracy = float((frame.material == frame.h6_predicted_material).mean()) if len(frame) else float("nan")
    delayed = frame[frame.delayed_material]
    delayed_recall = float(delayed.predicted_material.mean()) if len(delayed) else float("nan")
    dataset_accuracy = {
        dataset: float((current.material == current.predicted_material).mean())
        if len(current) else float("nan")
        for dataset in h3["datasets"]
        for current in [frame[frame.dataset == dataset]]
    }
    dataset_passes = int(sum(
        value >= float(config["minimum_dataset_accuracy"])
        for value in dataset_accuracy.values()
    ))
    improvement = accuracy - h6_accuracy
    gates = {
        "complete_and_integral": {
            "value": len(frame), "required": expected,
            "integrity_errors": integrity_errors,
            "integrity_bundles": integrity_summary["bundles"],
            "excluded_development_stems": sorted(development),
            "pass": bool(complete and not integrity_errors),
        },
        "sensitivity_specificity": {
            "sensitivity": sensitivity, "specificity": specificity,
            "required_sensitivity": config["minimum_sensitivity"],
            "required_specificity": config["minimum_specificity"],
            "confusion": {"tp": tp, "fn": fn, "tn": tn, "fp": fp},
            "pass": bool(sensitivity >= config["minimum_sensitivity"] and specificity >= config["minimum_specificity"]),
        },
        "balanced_accuracy": {
            "value": balanced_accuracy,
            "required": config["minimum_balanced_accuracy"],
            "pass": bool(balanced_accuracy >= config["minimum_balanced_accuracy"]),
        },
        "dataset_accuracy": {
            "value": dataset_passes, "required": config["minimum_dataset_passes"],
            "minimum_accuracy": config["minimum_dataset_accuracy"],
            "by_dataset": dataset_accuracy,
            "pass": dataset_passes >= int(config["minimum_dataset_passes"]),
        },
        "delayed_material_recall": {
            "value": delayed_recall, "required": config["minimum_delayed_recall"],
            "delayed_positives": len(delayed),
            "pass": bool(len(delayed) and delayed_recall >= config["minimum_delayed_recall"]),
        },
        "accuracy_improvement_over_h6": {
            "value": improvement,
            "required": config["minimum_accuracy_improvement_over_h6"],
            "h8_accuracy": accuracy, "h6_accuracy": h6_accuracy,
            "pass": bool(improvement >= config["minimum_accuracy_improvement_over_h6"]),
        },
    }
    summary = {
        "status": "complete" if complete else "in_progress",
        "artifacts": len(frame), "expected_artifacts": expected,
        "gates": gates,
        "hypothesis_supported": bool(complete and all(item["pass"] for item in gates.values())),
    }
    (output_dir / "h8_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
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
