"""Analyze the prospectively split H6 early orbit-growth screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

import analyze_semantic_arithmetic as h1_analysis
import audit_day6_integrity as integrity

HERE = Path(__file__).resolve().parent


def analyze(input_dir: Path, output_dir: Path) -> dict:
    h3 = json.loads((HERE / "hypothesis_03_config.json").read_text())
    config = json.loads((HERE / "hypothesis_06_config.json").read_text())
    development = set(config["development_stems"])
    all_artifacts = sorted(input_dir.glob("*.npz"))
    artifacts = [path for path in all_artifacts if path.stem not in development]
    rows = []
    early = np.asarray(config["early_checkpoints"], dtype=float)
    floor = float(config["log_floor"])
    for artifact in artifacts:
        trajectories, _ = h1_analysis.rows_from_artifact(artifact)
        frame = pd.DataFrame(trajectories)
        frame = frame[frame.precision == "fp32"]
        curve = frame.groupby("checkpoint").validation_prediction_mse.mean()
        if not set(early.astype(int)).issubset(curve.index):
            raise AssertionError(f"{artifact.stem}: missing early checkpoint")
        values = np.asarray([curve[int(epoch)] for epoch in early], dtype=float)
        slope, intercept = np.polyfit(early, np.log10(values + floor), deg=1)
        final_mse = float(curve[int(config["final_checkpoint"])])
        score = float(intercept + slope * int(config["final_checkpoint"]))
        manifest = json.loads(artifact.with_suffix(".json").read_text())
        rows.append({
            "stem": artifact.stem, "dataset": manifest["dataset"],
            "model": manifest["model"], "seed": int(manifest["seed"]),
            "early_slope": float(slope), "early_intercept": float(intercept),
            "epoch20_log_mse": float(np.log10(curve[20] + floor)),
            "extrapolated_epoch200_log_mse": score,
            "final_mse": final_mse,
            "final_log_mse": float(np.log10(final_mse + floor)),
            "material": bool(final_mse > float(config["material_final_mse"])),
            "predicted_material": bool(score > np.log10(config["material_final_mse"])),
        })
    frame = pd.DataFrame(rows, columns=[
        "stem", "dataset", "model", "seed", "early_slope", "early_intercept",
        "epoch20_log_mse", "extrapolated_epoch200_log_mse", "final_mse",
        "final_log_mse", "material", "predicted_material",
    ])
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "h6_prospective_bundles.csv", index=False)
    expected = len(h3["datasets"]) * len(h3["models"]) * len(h3["seeds"]) - len(development)
    integrity_errors, integrity_summary = integrity.audit_family("h3", input_dir)
    complete = len(frame) == expected
    both = len(frame) and frame.material.nunique() == 2
    auroc = float(roc_auc_score(
        frame.material, frame.extrapolated_epoch200_log_mse
    )) if both else float("nan")
    baseline_auroc = float(roc_auc_score(
        frame.material, frame.epoch20_log_mse
    )) if both else float("nan")
    dataset_auc = {}
    for dataset in h3["datasets"]:
        current = frame[frame.dataset == dataset]
        dataset_auc[dataset] = float(roc_auc_score(
            current.material, current.extrapolated_epoch200_log_mse
        )) if len(current) and current.material.nunique() == 2 else float("nan")
    dataset_passes = int(sum(value >= config["minimum_dataset_auroc"] for value in dataset_auc.values()))
    true_positive = int((frame.material & frame.predicted_material).sum()) if len(frame) else 0
    false_negative = int((frame.material & ~frame.predicted_material).sum()) if len(frame) else 0
    true_negative = int((~frame.material & ~frame.predicted_material).sum()) if len(frame) else 0
    false_positive = int((~frame.material & frame.predicted_material).sum()) if len(frame) else 0
    sensitivity = true_positive / (true_positive + false_negative) if true_positive + false_negative else float("nan")
    specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else float("nan")
    rank_parts = []
    for dataset, current in frame.groupby("dataset"):
        rank_parts.append(pd.DataFrame({
            "dataset": dataset,
            "score_rank": rankdata(current.extrapolated_epoch200_log_mse, method="average"),
            "target_rank": rankdata(current.final_log_mse, method="average"),
        }))
    ranks = pd.concat(rank_parts, ignore_index=True) if rank_parts else pd.DataFrame()
    rank_rho = float(spearmanr(
        ranks.score_rank, ranks.target_rank
    ).statistic) if len(ranks) else float("nan")
    improvement = auroc - baseline_auroc
    gates = {
        "complete_prospective_artifacts": {
            "value": len(frame), "required": expected,
            "integrity_errors": integrity_errors,
            "integrity_bundles": integrity_summary["bundles"],
            "pass": bool(complete and not integrity_errors),
            "excluded_development_stems": sorted(development),
        },
        "pooled_auroc": {
            "value": auroc, "required": config["minimum_pooled_auroc"],
            "material": int(frame.material.sum()) if len(frame) else 0,
            "total": len(frame), "pass": bool(auroc >= config["minimum_pooled_auroc"]),
        },
        "dataset_auroc": {
            "value": dataset_passes, "required": config["minimum_dataset_passes"],
            "by_dataset": dataset_auc,
            "pass": dataset_passes >= config["minimum_dataset_passes"],
        },
        "fixed_decision": {
            "sensitivity": sensitivity, "required_sensitivity": config["minimum_sensitivity"],
            "specificity": specificity, "required_specificity": config["minimum_specificity"],
            "confusion": {"tp": true_positive, "fn": false_negative, "tn": true_negative, "fp": false_positive},
            "pass": bool(sensitivity >= config["minimum_sensitivity"] and specificity >= config["minimum_specificity"]),
        },
        "rank_pooled_spearman": {
            "value": rank_rho, "required": config["minimum_rank_pooled_spearman"],
            "pass": bool(rank_rho >= config["minimum_rank_pooled_spearman"]),
        },
        "improvement_over_epoch20_level": {
            "value": improvement,
            "required": config["minimum_auroc_improvement_over_epoch20_level"],
            "extrapolated_auroc": auroc, "epoch20_level_auroc": baseline_auroc,
            "pass": bool(improvement >= config["minimum_auroc_improvement_over_epoch20_level"]),
        },
    }
    summary = {
        "status": "complete" if complete else "in_progress",
        "artifacts": len(frame), "expected_artifacts": expected,
        "gates": gates,
        "hypothesis_supported": bool(complete and all(item["pass"] for item in gates.values())),
    }
    (output_dir / "h6_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
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
