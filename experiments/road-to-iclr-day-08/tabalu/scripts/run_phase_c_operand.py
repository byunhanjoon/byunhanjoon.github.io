#!/usr/bin/env python3
"""Required raw/bounded/gated/unrestricted operand-estimator ablation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from tabalu.evaluation import regression_metrics
from tabalu.synthetic import generate_program_task, regenerate_split
from tabalu.training import train_operand_estimator


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(str(row["variant"]), str(row["split"]), float(row["noise_strength"]))].append(row)
    rng = np.random.default_rng(20260911)
    output: list[dict[str, Any]] = []
    for (variant, split, noise), rows in sorted(groups.items()):
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_task[str(row["task_id"])].append(row)

        def values(key: str) -> np.ndarray:
            return np.array(
                [np.mean([float(row[key]) for row in task_rows]) for task_rows in by_task.values()]
            )

        nrmse = values("nrmse")
        indices = rng.integers(0, len(nrmse), size=(2000, len(nrmse)))
        means = nrmse[indices].mean(axis=1)
        output.append(
            {
                "variant": variant,
                "split": split,
                "noise_strength": noise,
                "n_tasks": len(by_task),
                "n_observations": len(rows),
                "nrmse_mean": float(nrmse.mean()),
                "nrmse_std_across_tasks": float(nrmse.std(ddof=1)),
                "nrmse_ci95_low": float(np.quantile(means, 0.025)),
                "nrmse_ci95_high": float(np.quantile(means, 0.975)),
                "correction_rms_mean": float(values("normalized_correction_rms").mean()),
                "observed_confidence_mean": float(values("observed_confidence").mean()),
                "training_seconds_mean": float(values("training_seconds").mean()),
            }
        )
    return output


def gate(summary: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    lookup = {
        (str(row["variant"]), str(row["split"]), float(row["noise_strength"])): row
        for row in summary
    }
    level = float(thresholds["evaluation_noise"])
    raw = float(lookup[("raw", "iid", level)]["nrmse_mean"])
    bounded = float(lookup[("bounded_correction", "iid", level)]["nrmse_mean"])
    gated = float(lookup[("confidence_gated", "iid", level)]["nrmse_mean"])
    unrestricted = float(lookup[("unrestricted_encoder", "iid", level)]["nrmse_mean"])
    conservative = min(bounded, gated)
    conservative_name = "bounded_correction" if bounded <= gated else "confidence_gated"
    clean = float(lookup[(conservative_name, "iid", 0.0)]["nrmse_mean"])
    correction = float(lookup[(conservative_name, "iid", level)]["correction_rms_mean"])
    observed = {
        "best_conservative_variant": conservative_name,
        "conservative_vs_raw_ratio": conservative / max(raw, 1.0e-12),
        "conservative_vs_unrestricted_ratio": conservative / max(unrestricted, 1.0e-12),
        "clean_conservative_nrmse": clean,
        "normalized_correction_rms": correction,
    }
    checks = {
        "improves_over_raw": observed["conservative_vs_raw_ratio"]
        <= thresholds["conservative_vs_raw_ratio_max"],
        "competitive_with_unrestricted": observed["conservative_vs_unrestricted_ratio"]
        <= thresholds["conservative_vs_unrestricted_ratio_max"],
        "preserves_clean": clean <= thresholds["clean_conservative_nrmse_max"],
        "correction_is_bounded": correction <= thresholds["normalized_correction_rms_max"],
    }
    return {"passed": all(checks.values()), "checks": checks, "observed": observed, "thresholds": thresholds}


def plot(summary: list[dict[str, Any]], output: Path) -> None:
    iid = [row for row in summary if row["split"] == "iid"]
    figure, axis = plt.subplots(figsize=(7.5, 4.7), constrained_layout=True)
    for variant in sorted({str(row["variant"]) for row in iid}):
        rows = sorted(
            [row for row in iid if row["variant"] == variant], key=lambda row: float(row["noise_strength"])
        )
        axis.plot(
            [float(row["noise_strength"]) for row in rows],
            [max(float(row["nrmse_mean"]), 1.0e-6) for row in rows],
            marker="o",
            label=variant.replace("_", " "),
        )
    axis.set_yscale("log")
    axis.set_xlabel("Measurement-noise strength (fraction of feature SD)")
    axis.set_ylabel("IID normalized RMSE")
    axis.set_title("Phase C: conservative operand inference")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(output, dpi=200)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def run(config_path: Path) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = (PACKAGE_ROOT / config["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    repository = PACKAGE_ROOT.parents[1]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repository, text=True).strip())
    records: list[dict[str, Any]] = []
    for task_index in range(int(config["n_tasks"])):
        task = generate_program_task(
            int(config["task_seed_start"]) + task_index,
            n_features=int(config["n_features"]),
            depth_range=tuple(config["depth_range"]),
            operators=tuple(config["operators"]),
        )
        train_x, train_y = regenerate_split(task, "train", int(config["train_rows"]))
        validation_x, validation_y = regenerate_split(task, "validation", int(config["validation_rows"]))
        test_splits = {
            "iid": regenerate_split(task, "iid_test", int(config["test_rows"])),
            "ood": regenerate_split(
                task, "ood_test", int(config["test_rows"]), float(config["ood_multiplier"])
            ),
        }
        feature_scale = train_x.std(axis=0).clip(1.0e-6)
        for seed in config["seeds"]:
            for variant in config["variants"]:
                trained = train_operand_estimator(
                    variant,
                    task.program,
                    train_x,
                    train_y,
                    validation_x,
                    validation_y,
                    seed=int(seed),
                    noise_strength=float(config["train_noise_strength"]),
                    epochs=int(config["training"]["epochs"]),
                    learning_rate=float(config["training"]["learning_rate"]),
                    correction_weight=float(config["training"]["correction_weight"][variant]),
                    device=str(config["training"]["device"]),
                )
                estimator = trained.estimator
                device = next(estimator.buffers(), torch.empty(0)).device
                for split, (latent, targets) in test_splits.items():
                    for noise in config["test_noise_strengths"]:
                        rng = np.random.default_rng(
                            task.seed * 10007 + int(seed) * 101 + int(float(noise) * 1000) + (0 if split == "iid" else 1)
                        )
                        observed = latent + rng.normal(
                            0, float(noise) * feature_scale, latent.shape
                        ).astype(np.float32)
                        observed_t = torch.as_tensor(observed, dtype=torch.float32, device=device)
                        estimator.eval()
                        with torch.no_grad():
                            estimated_t = estimator(observed_t)
                            predictions = task.program(estimated_t).detach().cpu().numpy()
                            correction = float(
                                (((estimated_t - observed_t) / torch.as_tensor(feature_scale, device=device)) ** 2)
                                .mean()
                                .sqrt()
                            )
                            diagnostics = estimator.diagnostics(observed_t, estimated_t)
                            confidence = float(diagnostics.get("observed_confidence", torch.tensor(-1.0)))
                        row = {
                            "git_commit": commit,
                            "git_dirty": dirty,
                            "task_id": task.task_id,
                            "task_seed": task.seed,
                            "seed": seed,
                            "variant": variant,
                            "split": split,
                            "noise_strength": noise,
                            "train_noise_strength": config["train_noise_strength"],
                            "training_seconds": trained.training_seconds,
                            "validation_nrmse": trained.best_validation_nrmse,
                            "normalized_correction_rms": correction,
                            "observed_confidence": confidence,
                            **regression_metrics(targets, predictions),
                        }
                        records.append(row)
        write_csv(output / "records.csv", records)
        print(f"completed task {task_index + 1}/{config['n_tasks']}: {task.task_id}", flush=True)
    summary = summarize(records)
    write_csv(output / "summary.csv", summary)
    decision = gate(summary, config["gate"])
    expected = (
        int(config["n_tasks"])
        * len(config["seeds"])
        * len(config["variants"])
        * 2
        * len(config["test_noise_strengths"])
    )
    audit = {
        "git_commit": commit,
        "git_dirty": dirty,
        "expected_records": expected,
        "observed_records": len(records),
        "all_prediction_metrics_finite": all(
            math.isfinite(float(row[key])) for row in records for key in ("nrmse", "rmse", "mae")
        ),
        "confidence_interval_unit": "task (training seeds averaged within task)",
        "gate": decision,
    }
    audit["audit_passed"] = (
        expected == len(records) and audit["all_prediction_metrics_finite"] and decision["passed"]
    )
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    plot(summary, output / "noise_curve.png")
    print(json.dumps(audit, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
