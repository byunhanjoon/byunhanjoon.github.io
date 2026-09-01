"""Analyze the prospectively split H9 post-breach attenuation hypothesis."""

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
    config = json.loads((HERE / "hypothesis_09_config.json").read_text())
    development = set(config["development_stems"])
    final_checkpoint = int(config["final_checkpoint"])
    threshold = float(config["material_mse"])
    floor = float(config["ratio_floor"])
    rows, loss_rows = [], []
    for artifact in sorted(input_dir.glob("*.npz")):
        if artifact.stem in development:
            continue
        trajectories, references = h1_analysis.rows_from_artifact(artifact)
        frame = pd.DataFrame(trajectories)
        final = frame[frame.checkpoint == final_checkpoint]
        pivot = final.pivot(index="action", columns="precision", values="validation_prediction_mse")
        manifest = json.loads(artifact.with_suffix(".json").read_text())
        for action, current in pivot.iterrows():
            fp32_mse, iea64_mse = float(current.fp32), float(current.iea64)
            ratio = (iea64_mse + floor) / (fp32_mse + floor)
            rows.append({
                "stem": artifact.stem, "dataset": manifest["dataset"],
                "model": manifest["model"], "seed": int(manifest["seed"]),
                "action": int(action), "fp32_final_mse": fp32_mse,
                "iea64_final_mse": iea64_mse, "final_ratio": ratio,
                "eligible_fp32_material": bool(fp32_mse > threshold),
                "iea64_final_win": bool(iea64_mse < fp32_mse),
                "material_rescue": bool(fp32_mse > threshold and iea64_mse <= threshold),
                "twofold_worsening": bool(ratio > 2.0),
                "iea64_exact_final": bool(iea64_mse == 0.0),
            })
        refs = pd.DataFrame(references)
        refs = refs[refs.checkpoint == final_checkpoint].set_index("precision")
        relative = (float(refs.loc["iea64", "test_loss"]) - float(refs.loc["fp32", "test_loss"])) / max(
            float(refs.loc["fp32", "test_loss"]), floor
        )
        loss_rows.append({
            "stem": artifact.stem, "dataset": manifest["dataset"],
            "model": manifest["model"], "seed": int(manifest["seed"]),
            "canonical_relative_test_loss_change": relative,
        })

    columns = [
        "stem", "dataset", "model", "seed", "action", "fp32_final_mse",
        "iea64_final_mse", "final_ratio", "eligible_fp32_material",
        "iea64_final_win", "material_rescue", "twofold_worsening",
        "iea64_exact_final",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    losses = pd.DataFrame(loss_rows, columns=[
        "stem", "dataset", "model", "seed", "canonical_relative_test_loss_change"
    ])
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "h9_prospective_pairs.csv", index=False)
    losses.to_csv(output_dir / "h9_canonical_loss_pairs.csv", index=False)

    eligible = frame[frame.eligible_fp32_material] if len(frame) else frame
    dataset_rows = []
    for dataset in h3["datasets"]:
        current = eligible[eligible.dataset == dataset]
        dataset_rows.append({
            "dataset": dataset, "eligible_pairs": len(current),
            "win_fraction": float(current.iea64_final_win.mean()) if len(current) else float("nan"),
            "median_final_ratio": float(current.final_ratio.median()) if len(current) else float("nan"),
            "rescue_fraction": float(current.material_rescue.mean()) if len(current) else float("nan"),
            "twofold_worsening_fraction": float(current.twofold_worsening.mean()) if len(current) else float("nan"),
            "exact_final_fraction": float(current.iea64_exact_final.mean()) if len(current) else float("nan"),
        })
    dataset_frame = pd.DataFrame(dataset_rows)
    dataset_frame.to_csv(output_dir / "h9_dataset_summary.csv", index=False)

    expected_bundles = len(h3["datasets"]) * len(h3["models"]) * len(h3["seeds"]) - len(development)
    expected_pairs = expected_bundles * int(h3["nonidentity_views"])
    integrity_errors, integrity_summary = integrity.audit_family("h3", input_dir)
    complete = len(frame) == expected_pairs
    win_fraction = float(eligible.iea64_final_win.mean()) if len(eligible) else float("nan")
    rescue_fraction = float(eligible.material_rescue.mean()) if len(eligible) else float("nan")
    worsening_fraction = float(eligible.twofold_worsening.mean()) if len(eligible) else float("nan")
    dataset_ratio_passes = int(sum(
        row["eligible_pairs"] > 0
        and row["median_final_ratio"] <= float(config["maximum_dataset_median_ratio"])
        for row in dataset_rows
    ))
    loss_by_dataset = losses.groupby("dataset").canonical_relative_test_loss_change.mean()
    equal_dataset_loss = float(loss_by_dataset.mean()) if len(loss_by_dataset) else float("nan")
    gates = {
        "complete_and_integral": {
            "bundles": int(frame.stem.nunique()) if len(frame) else 0,
            "required_bundles": expected_bundles, "pairs": len(frame),
            "required_pairs": expected_pairs, "integrity_errors": integrity_errors,
            "integrity_bundles": integrity_summary["bundles"],
            "excluded_development_stems": sorted(development),
            "pass": bool(complete and not integrity_errors),
        },
        "final_mse_win_fraction": {
            "value": win_fraction, "required": config["minimum_final_win_fraction"],
            "eligible_pairs": len(eligible),
            "pass": bool(win_fraction >= config["minimum_final_win_fraction"]),
        },
        "dataset_median_ratio": {
            "value": dataset_ratio_passes, "required": config["minimum_dataset_ratio_passes"],
            "required_maximum_ratio": config["maximum_dataset_median_ratio"],
            "by_dataset": {row["dataset"]: row["median_final_ratio"] for row in dataset_rows},
            "pass": dataset_ratio_passes >= int(config["minimum_dataset_ratio_passes"]),
        },
        "material_rescue_fraction": {
            "value": rescue_fraction, "required": config["minimum_material_rescue_fraction"],
            "pass": bool(rescue_fraction >= config["minimum_material_rescue_fraction"]),
        },
        "twofold_worsening_fraction": {
            "value": worsening_fraction,
            "required_maximum": config["maximum_twofold_worsening_fraction"],
            "pass": bool(worsening_fraction <= config["maximum_twofold_worsening_fraction"]),
        },
        "canonical_test_loss_change": {
            "value": equal_dataset_loss,
            "required_absolute_maximum": config["maximum_absolute_equal_dataset_canonical_loss_change"],
            "dataset_means": {key: float(value) for key, value in loss_by_dataset.items()},
            "pass": bool(abs(equal_dataset_loss) <= config["maximum_absolute_equal_dataset_canonical_loss_change"]),
        },
    }
    summary = {
        "status": "complete" if complete else "in_progress",
        "prospective_bundles": int(frame.stem.nunique()) if len(frame) else 0,
        "expected_bundles": expected_bundles, "pairs": len(frame),
        "exact_final_eligible_pairs": int(eligible.iea64_exact_final.sum()) if len(eligible) else 0,
        "descriptive_by_dataset": {
            row["dataset"]: {key: value for key, value in row.items() if key != "dataset"}
            for row in dataset_rows
        },
        "gates": gates,
        "hypothesis_supported": bool(complete and all(item["pass"] for item in gates.values())),
    }
    (output_dir / "h9_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
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
