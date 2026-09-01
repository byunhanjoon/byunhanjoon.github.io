"""Analyze prospective H7 material-survival delays for IEA64 versus FP32."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_semantic_arithmetic as h1_analysis
import audit_day6_integrity as integrity

HERE = Path(__file__).resolve().parent


def analyze(input_dir: Path, output_dir: Path) -> dict:
    h3 = json.loads((HERE / "hypothesis_03_config.json").read_text())
    config = json.loads((HERE / "hypothesis_07_config.json").read_text())
    development = set(config["development_stems"])
    threshold = float(config["material_mse"])
    censored = int(config["censored_hitting_epoch"])
    rows = []
    for artifact in sorted(input_dir.glob("*.npz")):
        if artifact.stem in development:
            continue
        trajectory_rows, _ = h1_analysis.rows_from_artifact(artifact)
        frame = pd.DataFrame(trajectory_rows)
        manifest = json.loads(artifact.with_suffix(".json").read_text())
        for action in sorted(frame.action.unique()):
            row = {
                "stem": artifact.stem, "dataset": manifest["dataset"],
                "model": manifest["model"], "seed": int(manifest["seed"]),
                "action": int(action),
            }
            for precision in ("fp32", "iea64"):
                current = frame[
                    (frame.action == action) & (frame.precision == precision)
                ].sort_values("checkpoint")
                hits = current.loc[
                    current.validation_prediction_mse > threshold, "checkpoint"
                ].astype(int)
                row[f"{precision}_hitting_epoch"] = int(hits.iloc[0]) if len(hits) else censored
                row[f"{precision}_final_mse"] = float(
                    current.loc[current.checkpoint == 200, "validation_prediction_mse"].iloc[0]
                )
                early = current[current.checkpoint <= int(config["early_survival_epoch"])]
                row[f"{precision}_exact_through_20"] = bool(
                    (early.validation_prediction_mse == 0).all()
                )
            row["paired_delay"] = row["iea64_hitting_epoch"] - row["fp32_hitting_epoch"]
            row["fp32_material"] = row["fp32_hitting_epoch"] <= 200
            row["iea64_material"] = row["iea64_hitting_epoch"] <= 200
            row["iea64_later"] = row["iea64_hitting_epoch"] > row["fp32_hitting_epoch"]
            row["iea64_final_win"] = row["iea64_final_mse"] < row["fp32_final_mse"]
            rows.append(row)
    columns = [
        "stem", "dataset", "model", "seed", "action",
        "fp32_hitting_epoch", "iea64_hitting_epoch", "fp32_final_mse",
        "iea64_final_mse", "fp32_exact_through_20", "iea64_exact_through_20",
        "paired_delay", "fp32_material", "iea64_material", "iea64_later",
        "iea64_final_win",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "h7_survival_pairs.csv", index=False)
    expected_bundles = 36 - len(development)
    expected_pairs = expected_bundles * 3
    integrity_errors, integrity_summary = integrity.audit_family("h3", input_dir)
    complete = len(frame) == expected_pairs
    eligible = frame[frame.fp32_material] if len(frame) else frame
    later_fraction = float(eligible.iea64_later.mean()) if len(eligible) else float("nan")
    final_win_fraction = float(eligible.iea64_final_win.mean()) if len(eligible) else float("nan")
    exact_early_fraction = float(frame.iea64_exact_through_20.mean()) if len(frame) else float("nan")
    iea_failure_fraction = float(frame.iea64_material.mean()) if len(frame) else float("nan")
    dataset_delays = {}
    dataset_rows = []
    for dataset in h3["datasets"]:
        current_all = frame[frame.dataset == dataset]
        current = eligible[eligible.dataset == dataset]
        median_delay = float(current.paired_delay.median()) if len(current) else float("nan")
        dataset_delays[dataset] = median_delay
        dataset_rows.append({
            "dataset": dataset,
            "pairs": len(current_all),
            "eligible_fp32_material_paths": len(current),
            "later_fraction": float(current.iea64_later.mean()) if len(current) else float("nan"),
            "median_paired_delay": median_delay,
            "exact_early_fraction": float(current_all.iea64_exact_through_20.mean())
            if len(current_all) else float("nan"),
            "final_win_fraction": float(current.iea64_final_win.mean())
            if len(current) else float("nan"),
            "iea64_material_failure_fraction": float(current_all.iea64_material.mean())
            if len(current_all) else float("nan"),
        })
    dataset_frame = pd.DataFrame(dataset_rows)
    dataset_frame.to_csv(output_dir / "h7_dataset_summary.csv", index=False)
    dataset_passes = int(sum(
        value >= float(config["minimum_dataset_delay"])
        for value in dataset_delays.values()
    ))
    gates = {
        "complete_and_integral": {
            "bundles": int(frame.stem.nunique()) if len(frame) else 0,
            "required_bundles": expected_bundles, "pairs": len(frame),
            "required_pairs": expected_pairs, "integrity_errors": integrity_errors,
            "integrity_bundles": integrity_summary["bundles"],
            "pass": bool(complete and not integrity_errors),
        },
        "later_material_hitting": {
            "value": later_fraction, "required": config["minimum_later_fraction"],
            "eligible_fp32_material_paths": len(eligible),
            "pass": bool(later_fraction >= config["minimum_later_fraction"]),
        },
        "dataset_median_delay": {
            "value": dataset_passes, "required": config["minimum_dataset_delay_passes"],
            "minimum_delay": config["minimum_dataset_delay"],
            "by_dataset": dataset_delays,
            "pass": dataset_passes >= int(config["minimum_dataset_delay_passes"]),
        },
        "exact_early_survival": {
            "value": exact_early_fraction, "required": config["minimum_exact_early_survival"],
            "pass": bool(exact_early_fraction >= config["minimum_exact_early_survival"]),
        },
        "final_mse_reduction": {
            "value": final_win_fraction,
            "required": config["minimum_final_reduction_win_fraction"],
            "pass": bool(final_win_fraction >= config["minimum_final_reduction_win_fraction"]),
        },
        "iea64_material_failure_fraction": {
            "value": iea_failure_fraction,
            "required_maximum": config["maximum_iea64_material_failure_fraction"],
            "pass": bool(iea_failure_fraction <= config["maximum_iea64_material_failure_fraction"]),
        },
    }
    summary = {
        "status": "complete" if complete else "in_progress",
        "prospective_bundles": int(frame.stem.nunique()) if len(frame) else 0,
        "expected_bundles": expected_bundles, "pairs": len(frame),
        "descriptive_by_dataset": {
            row["dataset"]: {
                key: value for key, value in row.items() if key != "dataset"
            }
            for row in dataset_rows
        },
        "gates": gates,
        "hypothesis_supported": bool(complete and all(item["pass"] for item in gates.values())),
    }
    (output_dir / "h7_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
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
