#!/usr/bin/env python3
"""Program-discovery scaling at depths 2, 4, 6, and 8."""

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
from tabalu.evaluation import program_recovery_metrics, regression_metrics
from tabalu.models.discrete_search import beam_search_chain_program
from tabalu.synthetic import generate_program_task, regenerate_split


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(int(row["depth"]), str(row["model"]), str(row["split"]))].append(row)
    rng = np.random.default_rng(20261231)
    output: list[dict[str, Any]] = []
    for (depth, model, split), rows in sorted(groups.items()):
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_task[str(row["task_id"])].append(row)

        def task_values(key: str) -> np.ndarray:
            return np.array([np.mean([float(row[key]) for row in values]) for values in by_task.values()])

        nrmse = task_values("nrmse")
        indices = rng.integers(0, len(nrmse), size=(2000, len(nrmse)))
        boot = nrmse[indices].mean(axis=1)
        output.append(
            {
                "depth": depth,
                "model": model,
                "split": split,
                "n_tasks": len(by_task),
                "n_observations": len(rows),
                "nrmse_mean": float(nrmse.mean()),
                "nrmse_std_across_tasks": float(nrmse.std(ddof=1)),
                "nrmse_ci95_low": float(np.quantile(boot, 0.025)),
                "nrmse_ci95_high": float(np.quantile(boot, 0.975)),
                "functional_recovery_rate": float(task_values("functional_recovery").mean()),
                "exact_program_rate": float(task_values("exact_program").mean()),
                "operator_accuracy_mean": float(task_values("operator_accuracy").mean()),
                "training_seconds_mean": float(task_values("training_seconds").mean()),
            }
        )
    return output


def evaluate_gate(summary: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    lookup = {(int(row["depth"]), row["model"], row["split"]): row for row in summary}
    oracle_max = max(float(row["nrmse_mean"]) for row in summary if row["model"] == "OracleExact" and row["split"] == "ood")
    observed = {
        "oracle_ood_nrmse_max": oracle_max,
        "beam_depth2_ood_nrmse": float(lookup[(2, "BeamCompiled", "ood")]["nrmse_mean"]),
        "beam_depth8_ood_nrmse": float(lookup[(8, "BeamCompiled", "ood")]["nrmse_mean"]),
        "beam_depth8_functional_recovery": float(lookup[(8, "BeamCompiled", "ood")]["functional_recovery_rate"]),
    }
    checks = {
        "oracle_execution_scales": observed["oracle_ood_nrmse_max"] <= thresholds["oracle_ood_nrmse_max"],
        "beam_recovers_depth2": observed["beam_depth2_ood_nrmse"] <= thresholds["beam_depth2_ood_nrmse_max"],
        "beam_predicts_depth8": observed["beam_depth8_ood_nrmse"] <= thresholds["beam_depth8_ood_nrmse_max"],
        "beam_recovers_depth8": observed["beam_depth8_functional_recovery"] >= thresholds["beam_depth8_functional_recovery_min"],
    }
    return {"passed": all(checks.values()), "checks": checks, "observed": observed, "thresholds": thresholds}


def plot(summary: list[dict[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for model in ("OracleExact", "BeamCompiled", "MLP"):
        rows = sorted([row for row in summary if row["model"] == model and row["split"] == "ood"], key=lambda row: int(row["depth"]))
        axes[0].plot([int(row["depth"]) for row in rows], [max(float(row["nrmse_mean"]), 1.0e-6) for row in rows], marker="o", label=model)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Ground-truth program depth")
    axes[0].set_ylabel("4× OOD NRMSE")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    beam = sorted([row for row in summary if row["model"] == "BeamCompiled" and row["split"] == "ood"], key=lambda row: int(row["depth"]))
    axes[1].plot([int(row["depth"]) for row in beam], [float(row["functional_recovery_rate"]) for row in beam], marker="o", label="functional")
    axes[1].plot([int(row["depth"]) for row in beam], [float(row["exact_program_rate"]) for row in beam], marker="o", label="syntactic")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_xlabel("Ground-truth program depth")
    axes[1].set_ylabel("Recovery rate")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.suptitle("Depth scaling separates execution from discovery")
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
    task_counter = 0
    for depth in config["depths"]:
        for task_within_depth in range(int(config["tasks_per_depth"])):
            task_seed = int(config["task_seed_start"]) + int(depth) * 100 + task_within_depth
            task = generate_program_task(
                task_seed,
                n_features=4,
                depth_range=(int(depth), int(depth)),
                operators=tuple(config["operators"]),
                max_attempts=1000,
            )
            (output / "tasks" / f"{task.task_id}.json").write_text(json.dumps(task.to_dict(), indent=2) + "\n", encoding="utf-8")
            clean_train = regenerate_split(task, "train", int(config["train_rows"]))
            clean_validation = regenerate_split(task, "validation", int(config["validation_rows"]))
            tests = {
                "iid": regenerate_split(task, "iid_test", int(config["test_rows"])),
                "ood": regenerate_split(task, "ood_test", int(config["test_rows"]), float(config["ood_multiplier"])),
            }
            for seed in config["seeds"]:
                rng = np.random.default_rng(task.seed * 101 + int(seed))
                train_scale = float(config["target_noise_relative_std"]) * max(float(clean_train[1].std()), 1.0e-6)
                train_y = clean_train[1] + rng.normal(0, train_scale, len(clean_train[1]))
                val_y = clean_validation[1] + rng.normal(0, train_scale, len(clean_validation[1]))
                started = time.perf_counter()
                recovered = beam_search_chain_program(
                    clean_train[0],
                    train_y,
                    clean_validation[0],
                    val_y,
                    max_depth=int(depth),
                    operators=tuple(config["operators"]),
                    beam_width=int(config["beam_width"]),
                )
                beam_seconds = time.perf_counter() - started
                recovery = program_recovery_metrics(task.program, recovered)
                mlp = build_baseline("MLP", int(seed), str(config["device"]))
                mlp.epochs = int(config["neural_epochs"])
                started = time.perf_counter()
                mlp.fit(clean_train[0], train_y, (clean_validation[0], val_y))
                mlp_seconds = time.perf_counter() - started
                for split, (features, targets) in tests.items():
                    outputs = (
                        ("OracleExact", np.asarray(task.program(features)), 0.0, 1.0, 1.0, 1.0),
                        (
                            "BeamCompiled",
                            np.asarray(recovered(features)),
                            beam_seconds,
                            float(regression_metrics(targets, recovered(features))["nrmse"] < 1.0e-3),
                            float(recovery["exact_program"]),
                            float(recovery["operator_accuracy"]),
                        ),
                        ("MLP", mlp.predict(features), mlp_seconds, -1.0, -1.0, -1.0),
                    )
                    for model, prediction, elapsed, functional, exact, operator in outputs:
                        records.append(
                            {
                                "git_commit": commit,
                                "git_dirty": dirty,
                                "task_id": task.task_id,
                                "task_seed": task.seed,
                                "seed": seed,
                                "depth": depth,
                                "model": model,
                                "split": split,
                                "training_seconds": elapsed,
                                "functional_recovery": functional,
                                "exact_program": exact,
                                "operator_accuracy": operator,
                                **regression_metrics(targets, prediction),
                            }
                        )
                write_csv(output / "records.csv", records)
            task_counter += 1
            print(f"completed task {task_counter}/{len(config['depths']) * int(config['tasks_per_depth'])}: depth={depth}", flush=True)
    summary = summarize(records)
    write_csv(output / "summary.csv", summary)
    decision = evaluate_gate(summary, config["gate"])
    expected = len(config["depths"]) * int(config["tasks_per_depth"]) * len(config["seeds"]) * 3 * 2
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
    plot(summary, output / "depth_scaling.png")
    print(json.dumps(audit, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
