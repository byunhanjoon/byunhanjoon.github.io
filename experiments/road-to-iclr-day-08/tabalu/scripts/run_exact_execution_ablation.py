#!/usr/bin/env python3
"""Direct exact-primitives versus neural-primitives causal ablation."""

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
from tabalu.models.neural_executor import train_neural_primitive_executor
from tabalu.synthetic import generate_program_task, regenerate_split


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(str(row["model"]), float(row["multiplier"]))].append(row)
    rng = np.random.default_rng(20261001)
    output: list[dict[str, Any]] = []
    for (model, multiplier), rows in sorted(groups.items()):
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_task[str(row["task_id"])].append(row)
        values = np.array(
            [np.mean([float(row["nrmse"]) for row in task_rows]) for task_rows in by_task.values()]
        )
        indices = rng.integers(0, len(values), size=(2000, len(values)))
        means = values[indices].mean(axis=1)
        output.append(
            {
                "model": model,
                "multiplier": multiplier,
                "n_tasks": len(by_task),
                "n_observations": len(rows),
                "nrmse_mean": float(values.mean()),
                "nrmse_std_across_tasks": float(values.std(ddof=1)),
                "nrmse_ci95_low": float(np.quantile(means, 0.025)),
                "nrmse_ci95_high": float(np.quantile(means, 0.975)),
                "training_seconds_mean": float(
                    np.mean([float(row["training_seconds"]) for row in rows])
                ),
            }
        )
    return output


def gate(summary: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    lookup = {(row["model"], float(row["multiplier"])): row for row in summary}
    neural_iid = float(lookup[("NeuralPrimitives", 1.0)]["nrmse_mean"])
    neural_ood = float(lookup[("NeuralPrimitives", 8.0)]["nrmse_mean"])
    exact_ood = float(lookup[("ExactPrimitives", 8.0)]["nrmse_mean"])
    observed = {
        "neural_primitive_iid_nrmse": neural_iid,
        "neural_primitive_extrapolation_growth": neural_ood / max(neural_iid, 1.0e-12),
        "exact_8x_nrmse": exact_ood,
    }
    checks = {
        "neural_primitives_interpolate": neural_iid <= thresholds["neural_primitive_iid_nrmse_max"],
        "neural_primitives_fail_to_extrapolate": observed["neural_primitive_extrapolation_growth"]
        >= thresholds["neural_primitive_extrapolation_growth_min"],
        "exact_primitives_extrapolate": exact_ood <= thresholds["exact_8x_nrmse_max"],
    }
    return {"passed": all(checks.values()), "checks": checks, "observed": observed, "thresholds": thresholds}


def plot(summary: list[dict[str, Any]], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    for model in sorted({str(row["model"]) for row in summary}):
        rows = sorted([row for row in summary if row["model"] == model], key=lambda row: float(row["multiplier"]))
        label = "ExactPrimitives (zero; shown at floor)" if model == "ExactPrimitives" else model
        axis.plot(
            [float(row["multiplier"]) for row in rows],
            [max(float(row["nrmse_mean"]), 1.0e-6) for row in rows],
            marker="o",
            label=label,
        )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xticks([1, 2, 4, 8])
    axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axis.set_xlabel("Magnitude extrapolation multiplier")
    axis.set_ylabel("Normalized RMSE")
    axis.set_title("Same oracle graph: exact vs neural primitive execution")
    axis.grid(alpha=0.25)
    axis.legend()
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
        task = generate_program_task(int(config["task_seed_start"]) + task_index)
        train_x, _ = regenerate_split(task, "train", int(config["train_rows"]))
        validation_x, _ = regenerate_split(task, "validation", int(config["validation_rows"]))
        evaluations = {
            float(multiplier): regenerate_split(
                task,
                "iid_test" if float(multiplier) == 1 else "ood_test",
                int(config["test_rows"]),
                float(multiplier),
            )
            for multiplier in config["multipliers"]
        }
        for seed in config["seeds"]:
            trained = train_neural_primitive_executor(
                task.program,
                train_x,
                validation_x,
                seed=int(seed),
                epochs=int(config["primitive_epochs"]),
                device=str(config["device"]),
            )
            device = next(trained.model.parameters()).device
            for multiplier, (features, targets) in evaluations.items():
                with torch.no_grad():
                    neural = (
                        trained.model(torch.as_tensor(features, dtype=torch.float32, device=device))
                        .cpu()
                        .numpy()
                    )
                for model, predictions, elapsed in (
                    ("ExactPrimitives", task.program(features), 0.0),
                    ("NeuralPrimitives", neural, trained.training_seconds),
                ):
                    records.append(
                        {
                            "git_commit": commit,
                            "git_dirty": dirty,
                            "task_id": task.task_id,
                            "task_seed": task.seed,
                            "seed": seed,
                            "model": model,
                            "multiplier": multiplier,
                            "training_seconds": elapsed,
                            "mean_node_validation_nrmse": float(np.mean(trained.node_validation_nrmse)),
                            **regression_metrics(targets, predictions),
                        }
                    )
        write_csv(output / "records.csv", records)
        print(f"completed task {task_index + 1}/{config['n_tasks']}: {task.task_id}", flush=True)
    source = PACKAGE_ROOT / config["source_phase_a_records"]
    with source.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            task_number = int(row["dataset"].split("-")[-1])
            if (
                row["model"] == "MLP"
                and int(config["task_seed_start"]) <= task_number < int(config["task_seed_start"]) + int(config["n_tasks"])
            ):
                records.append(
                    {
                        "git_commit": row["git_commit"],
                        "git_dirty": row["git_dirty"],
                        "task_id": row["dataset"],
                        "task_seed": row["task_generation_seed"],
                        "seed": row["seed"],
                        "model": "WholeMLP",
                        "multiplier": row["extrapolation_multiplier"],
                        "training_seconds": row["training_seconds"],
                        "mean_node_validation_nrmse": -1.0,
                        **{key: row[key] for key in ("mae", "rmse", "r2", "nrmse", "relative_error")},
                    }
                )
    write_csv(output / "records.csv", records)
    summary = summarize(records)
    write_csv(output / "summary.csv", summary)
    decision = gate(summary, config["gate"])
    expected = int(config["n_tasks"]) * len(config["seeds"]) * len(config["multipliers"]) * 3
    audit = {
        "git_commit": commit,
        "git_dirty": dirty,
        "expected_records": expected,
        "observed_records": len(records),
        "all_finite": all(math.isfinite(float(row["nrmse"])) for row in records),
        "whole_mlp_source": str(source),
        "confidence_interval_unit": "task (training seeds averaged within task)",
        "gate": decision,
    }
    audit["audit_passed"] = expected == len(records) and audit["all_finite"] and decision["passed"]
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    plot(summary, output / "exact_execution_ablation.png")
    print(json.dumps(audit, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
