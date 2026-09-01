#!/usr/bin/env python3
"""Structural-recovery stress panel for the Phase-B go/no-go gate."""

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

from tabalu.evaluation import program_recovery_metrics, regression_metrics
from tabalu.models.discrete_search import search_chain_program
from tabalu.models.executor import ExecutableProgram, ProgramNode
from tabalu.synthetic import generate_program_task, regenerate_split


def git_state() -> tuple[str, bool]:
    repository = PACKAGE_ROOT.parents[1]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repository, text=True).strip())
    return commit, dirty


def rebase_program(program: ExecutableProgram, n_features: int) -> ExecutableProgram:
    """Increase the input width while preserving every node reference."""

    if n_features < program.n_features:
        raise ValueError("cannot rebase to fewer features")

    def rebase(reference: int) -> int:
        if reference < program.n_features:
            return reference
        return n_features + reference - program.n_features

    nodes = [
        ProgramNode(
            node.operator,
            rebase(node.left),
            rebase(node.right) if node.right is not None else None,
        )
        for node in program.nodes
    ]
    return ExecutableProgram(
        n_features,
        nodes,
        rebase(program.output),
        program.output_scale,
        program.output_bias,
        program.epsilon,
    )


def independent_features(
    rng: np.random.Generator, rows: int, count: int, multiplier: float
) -> np.ndarray:
    if multiplier <= 1:
        return rng.uniform(-2, 2, size=(rows, count)).astype(np.float32)
    magnitude = rng.uniform(2.25, 2 * multiplier, size=(rows, count))
    return (magnitude * rng.choice([-1.0, 1.0], size=(rows, count))).astype(np.float32)


def prepare_condition(
    condition: str,
    settings: dict[str, Any],
    seed: int,
    truth: ExecutableProgram,
    train: tuple[np.ndarray, np.ndarray],
    validation: tuple[np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray],
    multiplier: float,
) -> tuple[ExecutableProgram, tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(991 * seed + 17 * truth.n_features + 20260901)
    train_x, train_y = (train[0].copy(), train[1].copy())
    validation_x, validation_y = (validation[0].copy(), validation[1].copy())
    test_x, test_y = (test[0].copy(), test[1].copy())
    conditioned_truth = truth
    if condition == "clean":
        pass
    elif condition == "target_noise":
        strength = float(settings["relative_std"])
        train_y += rng.normal(0, strength * max(float(train_y.std()), 1.0e-6), len(train_y))
        validation_y += rng.normal(
            0, strength * max(float(validation_y.std()), 1.0e-6), len(validation_y)
        )
    elif condition == "measurement_noise":
        strength = float(settings["relative_std"])
        scale = train_x.std(axis=0, keepdims=True).clip(1.0e-6)
        train_x += rng.normal(0, strength * scale, train_x.shape)
        validation_x += rng.normal(0, strength * scale, validation_x.shape)
        test_x += rng.normal(0, strength * scale, test_x.shape)
    elif condition == "irrelevant_features":
        count = int(settings["count"])
        train_x = np.column_stack(
            [train_x, independent_features(rng, len(train_x), count, multiplier=1)]
        )
        validation_x = np.column_stack(
            [validation_x, independent_features(rng, len(validation_x), count, multiplier=1)]
        )
        test_x = np.column_stack(
            [test_x, independent_features(rng, len(test_x), count, multiplier=multiplier)]
        )
        conditioned_truth = rebase_program(truth, truth.n_features + count)
    elif condition == "correlated_features":
        count = int(settings["count"])
        strength = float(settings["relative_std"])

        def correlated(values: np.ndarray) -> np.ndarray:
            selected = values[:, np.arange(count) % truth.n_features]
            scale = train_x.std(axis=0)[np.arange(count) % truth.n_features]
            return selected + rng.normal(0, strength * scale, selected.shape)

        train_x = np.column_stack([train_x, correlated(train_x)])
        validation_x = np.column_stack([validation_x, correlated(validation_x)])
        test_x = np.column_stack([test_x, correlated(test_x)])
        conditioned_truth = rebase_program(truth, truth.n_features + count)
    else:
        raise KeyError(condition)
    return conditioned_truth, (train_x, train_y), (validation_x, validation_y), (test_x, test_y)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def task_cluster_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition"])].append(row)
    rng = np.random.default_rng(20260902)
    summary: list[dict[str, Any]] = []
    for condition, condition_rows in sorted(grouped.items()):
        task_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in condition_rows:
            task_rows[str(row["task_id"])].append(row)

        def task_values(metric: str) -> np.ndarray:
            return np.array(
                [np.mean([float(row[metric]) for row in values]) for values in task_rows.values()]
            )

        nrmse = task_values("nrmse")
        indices = rng.integers(0, len(nrmse), size=(2000, len(nrmse)))
        means = nrmse[indices].mean(axis=1)
        summary.append(
            {
                "condition": condition,
                "n_tasks": len(task_rows),
                "n_observations": len(condition_rows),
                "nrmse_mean": float(nrmse.mean()),
                "nrmse_std_across_tasks": float(nrmse.std(ddof=1)),
                "nrmse_ci95_low": float(np.quantile(means, 0.025)),
                "nrmse_ci95_high": float(np.quantile(means, 0.975)),
                "feature_f1_mean": float(task_values("feature_f1").mean()),
                "operator_accuracy_mean": float(task_values("operator_accuracy").mean()),
                "exact_program_fraction": float(task_values("exact_program").mean()),
                "operation_count_mean": float(task_values("operation_count").mean()),
                "search_seconds_mean": float(task_values("search_seconds").mean()),
            }
        )
    return summary


def gate(summary: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    rows = {str(row["condition"]): row for row in summary}
    checks = {
        "clean_nrmse": float(rows["clean"]["nrmse_mean"]) <= thresholds["clean_nrmse_max"],
        "clean_feature_f1": float(rows["clean"]["feature_f1_mean"])
        >= thresholds["clean_feature_f1_min"],
        "clean_exact_program": float(rows["clean"]["exact_program_fraction"])
        >= thresholds["clean_exact_program_min"],
        "target_noise_nrmse": float(rows["target_noise"]["nrmse_mean"])
        <= thresholds["target_noise_nrmse_max"],
        "measurement_noise_nrmse": float(rows["measurement_noise"]["nrmse_mean"])
        <= thresholds["measurement_noise_nrmse_max"],
        "irrelevant_features_nrmse": float(rows["irrelevant_features"]["nrmse_mean"])
        <= thresholds["irrelevant_features_nrmse_max"],
        "irrelevant_features_feature_f1": float(rows["irrelevant_features"]["feature_f1_mean"])
        >= thresholds["irrelevant_features_feature_f1_min"],
        "correlated_features_nrmse": float(rows["correlated_features"]["nrmse_mean"])
        <= thresholds["correlated_features_nrmse_max"],
    }
    return {"passed": all(checks.values()), "checks": checks, "thresholds": thresholds}


def plot(summary: list[dict[str, Any]], output: Path) -> None:
    labels = [str(row["condition"]).replace("_", "\n") for row in summary]
    nrmse = np.array([float(row["nrmse_mean"]) for row in summary])
    f1 = np.array([float(row["feature_f1_mean"]) for row in summary])
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    axes[0].bar(labels, nrmse, color="#31688e")
    axes[0].set_yscale("symlog", linthresh=1.0e-6)
    axes[0].set_ylabel("8× normalized RMSE")
    axes[0].set_title("Functional recovery")
    axes[1].bar(labels, f1, color="#35b779")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Feature-selection F1")
    axes[1].set_title("Structural recovery")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", labelsize=8)
    figure.savefig(output, dpi=200)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def run(config_path: Path) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = (PACKAGE_ROOT / config["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "programs").mkdir(exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    commit, dirty = git_state()
    records: list[dict[str, Any]] = []
    for task_index in range(int(config["n_tasks"])):
        task = generate_program_task(
            int(config["task_seed_start"]) + task_index,
            n_features=int(config["n_features"]),
            depth_range=tuple(config["depth_range"]),
            operators=tuple(config["operators"]),
        )
        train = regenerate_split(task, "train", int(config["train_rows"]))
        validation = regenerate_split(task, "validation", int(config["validation_rows"]))
        test = regenerate_split(
            task, "ood_test", int(config["test_rows"]), float(config["test_multiplier"])
        )
        for condition, settings in config["conditions"].items():
            seeds = settings.get("seeds", config["seeds"])
            for seed in seeds:
                truth, conditioned_train, conditioned_validation, conditioned_test = prepare_condition(
                    condition,
                    settings,
                    int(seed),
                    task.program,
                    train,
                    validation,
                    test,
                    float(config["test_multiplier"]),
                )
                started = time.perf_counter()
                recovered = search_chain_program(
                    conditioned_train[0],
                    conditioned_train[1],
                    conditioned_validation[0],
                    conditioned_validation[1],
                    max_depth=int(config["depth_range"][1]),
                    operators=tuple(config["operators"]),
                )
                elapsed = time.perf_counter() - started
                predictions = recovered(conditioned_test[0])
                row = {
                    "git_commit": commit,
                    "git_dirty": dirty,
                    "task_id": task.task_id,
                    "task_seed": task.seed,
                    "seed": seed,
                    "condition": condition,
                    "test_multiplier": config["test_multiplier"],
                    "search_seconds": elapsed,
                    "operation_count": recovered.operation_count,
                    "truth_expression": truth.expression(),
                    "recovered_expression": recovered.expression(),
                    **regression_metrics(conditioned_test[1], predictions),
                    **program_recovery_metrics(truth, recovered),
                }
                row["exact_program"] = int(bool(row["exact_program"]))
                records.append(row)
                artifact = {
                    "record": row,
                    "truth": truth.to_dict(),
                    "recovered": recovered.to_dict(),
                }
                name = f"{task.task_id}__{condition}__seed-{seed}.json"
                (output / "programs" / name).write_text(
                    json.dumps(artifact, indent=2) + "\n", encoding="utf-8"
                )
        write_csv(output / "records.csv", records)
        print(f"completed task {task_index + 1}/{config['n_tasks']}: {task.task_id}", flush=True)
    summary = task_cluster_summary(records)
    write_csv(output / "summary.csv", summary)
    decision = gate(summary, config["gate"])
    expected = int(config["n_tasks"]) * sum(
        len(settings.get("seeds", config["seeds"])) for settings in config["conditions"].values()
    )
    audit = {
        "git_commit": commit,
        "git_dirty": dirty,
        "expected_records": expected,
        "observed_records": len(records),
        "all_finite": all(
            math.isfinite(float(row[key]))
            for row in records
            for key in ("nrmse", "feature_f1", "operator_accuracy")
        ),
        "confidence_interval_unit": "task (corruption seeds averaged within task)",
        "gate": decision,
    }
    audit["audit_passed"] = (
        audit["expected_records"] == audit["observed_records"]
        and audit["all_finite"]
        and decision["passed"]
    )
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    plot(summary, output / "recovery_stress.png")
    print(json.dumps(audit, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
