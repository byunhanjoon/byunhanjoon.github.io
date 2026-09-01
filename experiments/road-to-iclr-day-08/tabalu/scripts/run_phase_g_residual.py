#!/usr/bin/env python3
"""Phase G neural residual continuum experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from tabalu.baselines import build_baseline
from tabalu.evaluation import regression_metrics
from tabalu.models.residual import ProgramResidualRegressor
from tabalu.synthetic import generate_residual_task, sample_residual_split


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(str(row["model"]), float(row["alpha"]), str(row["split"]))].append(row)
    rng = np.random.default_rng(20261130)
    output: list[dict[str, Any]] = []
    for (model, alpha, split), rows in sorted(groups.items()):
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_task[str(row["task_id"])].append(row)

        def task_values(key: str) -> np.ndarray:
            return np.array(
                [np.mean([float(row[key]) for row in task_rows]) for task_rows in by_task.values()]
            )

        nrmse = task_values("nrmse")
        indices = rng.integers(0, len(nrmse), size=(2000, len(nrmse)))
        bootstrap = nrmse[indices].mean(axis=1)
        output.append(
            {
                "model": model,
                "alpha": alpha,
                "split": split,
                "n_tasks": len(by_task),
                "n_observations": len(rows),
                "nrmse_mean": float(nrmse.mean()),
                "nrmse_std_across_tasks": float(nrmse.std(ddof=1)) if len(nrmse) > 1 else 0.0,
                "nrmse_ci95_low": float(np.quantile(bootstrap, 0.025)),
                "nrmse_ci95_high": float(np.quantile(bootstrap, 0.975)),
                "residual_usage_mean": float(task_values("residual_usage").mean()),
                "gate_mean": float(task_values("gate_mean").mean()),
                "training_seconds_mean": float(task_values("training_seconds").mean()),
            }
        )
    return output


def evaluate_gate(summary: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    lookup = {
        (row["model"], float(row["alpha"]), row["split"]): row
        for row in summary
    }
    scalar_rows = [lookup[("PenalizedScalarResidual", alpha, "iid")] for alpha in (0.0, 0.1, 0.25, 0.5, 1.0)]
    usages = np.array([float(row["residual_usage_mean"]) for row in scalar_rows])
    correlation = float(np.corrcoef(np.array((0.0, 0.1, 0.25, 0.5, 1.0)), usages)[0, 1])
    scalar_one = float(lookup[("PenalizedScalarResidual", 1.0, "iid")]["nrmse_mean"])
    adaptive_one = float(lookup[("AdaptiveResidual", 1.0, "iid")]["nrmse_mean"])
    program_one = float(lookup[("PureProgram", 1.0, "iid")]["nrmse_mean"])
    observed = {
        "alpha_zero_residual_usage": float(usages[0]),
        "alpha_one_residual_usage": float(usages[-1]),
        "usage_alpha_correlation": correlation,
        "penalized_vs_program_alpha_one_ratio": scalar_one / max(program_one, 1.0e-12),
        "adaptive_vs_scalar_alpha_one_ratio": adaptive_one / max(scalar_one, 1.0e-12),
        "alpha_zero_nrmse": float(lookup[("PenalizedScalarResidual", 0.0, "iid")]["nrmse_mean"]),
    }
    checks = {
        "stays_off_for_symbolic_target": observed["alpha_zero_residual_usage"] <= thresholds["alpha_zero_residual_usage_max"],
        "turns_on_for_nonsymbolic_target": observed["alpha_one_residual_usage"] >= thresholds["alpha_one_residual_usage_min"],
        "tracks_nonsymbolic_fraction": correlation >= thresholds["usage_alpha_correlation_min"],
        "improves_nonsymbolic_target": observed["penalized_vs_program_alpha_one_ratio"] <= thresholds["penalized_vs_program_alpha_one_ratio_max"],
        "adaptive_is_competitive": observed["adaptive_vs_scalar_alpha_one_ratio"] <= thresholds["adaptive_vs_scalar_alpha_one_ratio_max"],
        "preserves_symbolic_target": observed["alpha_zero_nrmse"] <= thresholds["alpha_zero_nrmse_max"],
    }
    return {"passed": all(checks.values()), "checks": checks, "observed": observed, "thresholds": thresholds}


def plot(summary: list[dict[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    models = ("PureProgram", "PureMLP", "UnpenalizedResidual", "PenalizedScalarResidual", "AdaptiveResidual")
    for model in models:
        rows = sorted(
            [row for row in summary if row["model"] == model and row["split"] == "iid"],
            key=lambda row: float(row["alpha"]),
        )
        axes[0].plot(
            [float(row["alpha"]) for row in rows],
            [max(float(row["nrmse_mean"]), 1.0e-6) for row in rows],
            marker="o",
            label=model,
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("True non-symbolic fraction α")
    axes[0].set_ylabel("IID normalized RMSE")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    for model in models:
        rows = sorted(
            [row for row in summary if row["model"] == model and row["split"] == "ood"],
            key=lambda row: float(row["alpha"]),
        )
        axes[1].plot(
            [float(row["alpha"]) for row in rows],
            [max(float(row["nrmse_mean"]), 1.0e-6) for row in rows],
            marker="o",
            label=model,
        )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("True non-symbolic fraction α")
    axes[1].set_ylabel("4× OOD normalized RMSE")
    axes[1].grid(alpha=0.25)
    for model in ("UnpenalizedResidual", "PenalizedScalarResidual", "AdaptiveResidual"):
        rows = sorted(
            [row for row in summary if row["model"] == model and row["split"] == "iid"],
            key=lambda row: float(row["alpha"]),
        )
        axes[2].plot(
            [float(row["alpha"]) for row in rows],
            [float(row["residual_usage_mean"]) for row in rows],
            marker="o",
            label=model,
        )
    axes[2].set_xlabel("True non-symbolic fraction α")
    axes[2].set_ylabel("IID residual contribution / target SD")
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8)
    figure.suptitle("Neural escape hatch: prediction and actual residual usage")
    figure.savefig(output, dpi=200)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def run(config_path: Path) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = (PACKAGE_ROOT / config["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "tasks").mkdir(exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    repository = PACKAGE_ROOT.parents[1]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repository, text=True).strip())
    records: list[dict[str, Any]] = []
    for task_index in range(int(config["n_tasks"])):
        task = generate_residual_task(int(config["task_seed_start"]) + task_index)
        (output / "tasks" / f"{task.task_id}.json").write_text(
            json.dumps(
                {
                    "task_id": task.task_id,
                    "symbolic": task.symbolic_task.to_dict(),
                    "residual_mean": task.residual_mean,
                    "residual_scale": task.residual_scale,
                    "symbolic_scale": task.symbolic_scale,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        for alpha in config["alphas"]:
            train_x, clean_train_y, _, _ = sample_residual_split(
                task, "train", int(config["train_rows"]), alpha=float(alpha)
            )
            validation_x, clean_validation_y, _, _ = sample_residual_split(
                task, "validation", int(config["validation_rows"]), alpha=float(alpha)
            )
            tests = {
                "iid": sample_residual_split(task, "iid_test", int(config["test_rows"]), alpha=float(alpha)),
                "ood": sample_residual_split(
                    task,
                    "ood",
                    int(config["test_rows"]),
                    alpha=float(alpha),
                    magnitude_multiplier=float(config["ood_multiplier"]),
                ),
            }
            for seed in config["seeds"]:
                rng = np.random.default_rng(task.symbolic_task.seed * 101 + int(seed) * 17 + int(float(alpha) * 1000))
                noise_scale = float(config["target_noise_relative_std"]) * max(float(clean_train_y.std()), 1.0e-6)
                train_y = clean_train_y + rng.normal(0, noise_scale, len(clean_train_y))
                validation_y = clean_validation_y + rng.normal(0, noise_scale, len(clean_validation_y))
                residual_specs = {
                    "UnpenalizedResidual": ("none", 0.0, 0.0),
                    "PenalizedScalarResidual": ("scalar", float(config["residual_penalty"]), float(config["gate_penalty"])),
                    "AdaptiveResidual": ("adaptive", float(config["residual_penalty"]), float(config["gate_penalty"])),
                }
                fitted: dict[str, tuple[Any, float]] = {}
                for model_name, (gate_mode, residual_penalty, gate_penalty) in residual_specs.items():
                    model = ProgramResidualRegressor(
                        task.symbolic_task.program,
                        seed=int(seed),
                        gate_mode=gate_mode,
                        residual_penalty=residual_penalty,
                        gate_penalty=gate_penalty,
                        epochs=int(config["neural_epochs"]),
                        device=str(config["device"]),
                    )
                    fit = model.fit(train_x, train_y, (validation_x, validation_y))
                    fitted[model_name] = (model, fit.training_seconds)
                mlp = build_baseline("MLP", int(seed), str(config["device"]))
                mlp.epochs = int(config["neural_epochs"])
                started = time.perf_counter()
                mlp.fit(train_x, train_y, (validation_x, validation_y))
                fitted["PureMLP"] = (mlp, time.perf_counter() - started)
                for split, (test_x, targets, _, _) in tests.items():
                    program_prediction = np.asarray(task.symbolic_task.program(test_x), dtype=np.float64)
                    model_outputs: list[tuple[str, np.ndarray, float, float, float]] = [
                        ("PureProgram", program_prediction, 0.0, 0.0, 0.0),
                    ]
                    for model_name, (model, elapsed) in fitted.items():
                        if isinstance(model, ProgramResidualRegressor):
                            prediction, contribution, gates = model.predict_with_usage(test_x)
                            usage = float(np.sqrt(np.mean(contribution**2)) / max(float(targets.std()), 1.0e-8))
                            gate_mean = float(gates.mean())
                        else:
                            prediction = model.predict(test_x)
                            usage = -1.0
                            gate_mean = -1.0
                        model_outputs.append((model_name, prediction, elapsed, usage, gate_mean))
                    for model_name, prediction, elapsed, usage, gate_mean in model_outputs:
                        records.append(
                            {
                                "git_commit": commit,
                                "git_dirty": dirty,
                                "task_id": task.task_id,
                                "task_seed": task.symbolic_task.seed,
                                "seed": seed,
                                "alpha": alpha,
                                "model": model_name,
                                "split": split,
                                "magnitude_multiplier": 1.0 if split == "iid" else config["ood_multiplier"],
                                "training_seconds": elapsed,
                                "residual_usage": usage,
                                "gate_mean": gate_mean,
                                **regression_metrics(targets, prediction),
                            }
                        )
                write_csv(output / "records.csv", records)
        print(f"completed task {task_index + 1}/{config['n_tasks']}: {task.task_id}", flush=True)
    summary = summarize(records)
    write_csv(output / "summary.csv", summary)
    decision = evaluate_gate(summary, config["gate"])
    expected = int(config["n_tasks"]) * len(config["seeds"]) * len(config["alphas"]) * 5 * 2
    audit = {
        "git_commit": commit,
        "git_dirty": dirty,
        "expected_records": expected,
        "observed_records": len(records),
        "all_finite": all(math.isfinite(float(row["nrmse"])) for row in records),
        "confidence_interval_unit": "task (training seeds averaged within task)",
        "gate": decision,
    }
    audit["audit_passed"] = expected == len(records) and audit["all_finite"] and decision["passed"]
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    plot(summary, output / "residual_continuum.png")
    print(json.dumps(audit, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
