#!/usr/bin/env python3
"""Scaling diagnostic for 1, 2, 4, and 8 categorical regimes."""

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
from scipy.optimize import linear_sum_assignment

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from tabalu.baselines import build_baseline
from tabalu.baselines.regressors import NeuralMixtureRegressor
from tabalu.evaluation import program_recovery_metrics, regression_metrics
from tabalu.models.discrete_search import search_chain_program
from tabalu.synthetic import generate_multi_regime_task, sample_multi_regime_split


OPERATORS = ("add", "subtract", "multiply", "safe_divide", "abs", "square")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aligned_accuracy(truth: np.ndarray, predicted: np.ndarray, n_regimes: int) -> float:
    contingency = np.zeros((n_regimes, n_regimes), dtype=int)
    for true_value, predicted_value in zip(truth, predicted):
        contingency[int(true_value), int(predicted_value)] += 1
    rows, columns = linear_sum_assignment(-contingency)
    return float(contingency[rows, columns].sum() / len(truth))


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(int(row["n_regimes"]), str(row["model"]), str(row["split"]))].append(row)
    rng = np.random.default_rng(20261331)
    output: list[dict[str, Any]] = []
    for (n_regimes, model, split), rows in sorted(groups.items()):
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
                "n_regimes": n_regimes,
                "model": model,
                "split": split,
                "n_tasks": len(by_task),
                "n_observations": len(rows),
                "nrmse_mean": float(nrmse.mean()),
                "nrmse_std_across_tasks": float(nrmse.std(ddof=1)),
                "nrmse_ci95_low": float(np.quantile(boot, 0.025)),
                "nrmse_ci95_high": float(np.quantile(boot, 0.975)),
                "functional_recovery_rate": float(task_values("functional_recovery").mean()),
                "operator_accuracy_mean": float(task_values("operator_accuracy").mean()),
                "regime_accuracy_mean": float(task_values("regime_accuracy").mean()),
                "training_seconds_mean": float(task_values("training_seconds").mean()),
            }
        )
    return output


def evaluate_gate(summary: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    lookup = {(int(row["n_regimes"]), row["model"], row["split"]): row for row in summary}
    hard = float(lookup[(8, "HardProgramRouter", "ood")]["nrmse_mean"])
    neural = float(lookup[(8, "NeuralMoE", "ood")]["nrmse_mean"])
    single = float(lookup[(8, "SingleProgram", "ood")]["nrmse_mean"])
    observed = {
        "hard_router_r8_ood_nrmse": hard,
        "hard_vs_neural_r8_ratio": hard / max(neural, 1.0e-12),
        "hard_vs_single_r8_ratio": hard / max(single, 1.0e-12),
        "hard_r8_operator_accuracy": float(lookup[(8, "HardProgramRouter", "ood")]["operator_accuracy_mean"]),
        "hard_r8_functional_recovery": float(lookup[(8, "HardProgramRouter", "ood")]["functional_recovery_rate"]),
    }
    checks = {
        "r8_predicts": observed["hard_router_r8_ood_nrmse"] <= thresholds["hard_router_r8_ood_nrmse_max"],
        "r8_beats_neural": observed["hard_vs_neural_r8_ratio"] <= thresholds["hard_vs_neural_r8_ratio_max"],
        "r8_beats_single": observed["hard_vs_single_r8_ratio"] <= thresholds["hard_vs_single_r8_ratio_max"],
        "r8_recovers_operators": observed["hard_r8_operator_accuracy"] >= thresholds["hard_r8_operator_accuracy_min"],
        "r8_recovers_functions": observed["hard_r8_functional_recovery"] >= thresholds["hard_r8_functional_recovery_min"],
    }
    return {"passed": all(checks.values()), "checks": checks, "observed": observed, "thresholds": thresholds}


def plot(summary: list[dict[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    models = ("OracleRouter", "HardProgramRouter", "SingleProgram", "MLP", "RandomForest", "NeuralMoE")
    for model in models:
        rows = sorted([row for row in summary if row["model"] == model and row["split"] == "ood"], key=lambda row: int(row["n_regimes"]))
        axes[0].plot([int(row["n_regimes"]) for row in rows], [max(float(row["nrmse_mean"]), 1.0e-6) for row in rows], marker="o", label=model)
    axes[0].set_yscale("log")
    axes[0].set_xticks((1, 2, 4, 8))
    axes[0].set_xlabel("Number of regimes")
    axes[0].set_ylabel("Shifted 4× NRMSE")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    hard = sorted([row for row in summary if row["model"] == "HardProgramRouter" and row["split"] == "ood"], key=lambda row: int(row["n_regimes"]))
    axes[1].plot([int(row["n_regimes"]) for row in hard], [float(row["functional_recovery_rate"]) for row in hard], marker="o", label="functional")
    axes[1].plot([int(row["n_regimes"]) for row in hard], [float(row["operator_accuracy_mean"]) for row in hard], marker="o", label="operator")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_xticks((1, 2, 4, 8))
    axes[1].set_xlabel("Number of regimes")
    axes[1].set_ylabel("Expert recovery")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.suptitle("Known categorical routing under regime-count scaling")
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
    total_tasks = len(config["regime_counts"]) * int(config["tasks_per_count"])
    for n_regimes in config["regime_counts"]:
        for task_within_count in range(int(config["tasks_per_count"])):
            task_seed = int(config["task_seed_start"]) + int(n_regimes) * 100 + task_within_count
            task = generate_multi_regime_task(task_seed, int(n_regimes))
            (output / "tasks" / f"{task.task_id}.json").write_text(
                json.dumps({"task_id": task.task_id, "seed": task.seed, "programs": [program.to_dict() for program in task.programs]}, indent=2) + "\n",
                encoding="utf-8",
            )
            for seed in config["seeds"]:
                train = sample_multi_regime_split(task, "train", int(config["train_rows"]), seed=int(seed))
                validation = sample_multi_regime_split(task, "validation", int(config["validation_rows"]), seed=int(seed))
                tests = {
                    "iid": sample_multi_regime_split(task, "iid", int(config["test_rows"]), seed=int(seed)),
                    "ood": sample_multi_regime_split(task, "ood", int(config["test_rows"]), seed=int(seed), magnitude_multiplier=float(config["ood_multiplier"])),
                }
                train_x, train_context, train_regime, clean_train_y = train
                val_x, val_context, val_regime, clean_val_y = validation
                rng = np.random.default_rng(task.seed * 101 + int(seed))
                noise_scale = float(config["target_noise_relative_std"]) * max(float(clean_train_y.std()), 1.0e-6)
                train_y = clean_train_y + rng.normal(0, noise_scale, len(clean_train_y))
                val_y = clean_val_y + rng.normal(0, noise_scale, len(clean_val_y))
                started = time.perf_counter()
                experts = []
                recoveries = []
                for regime in range(int(n_regimes)):
                    train_mask = train_regime == regime
                    val_mask = val_regime == regime
                    expert = search_chain_program(
                        train_x[train_mask], train_y[train_mask], val_x[val_mask], val_y[val_mask],
                        max_depth=2, operators=OPERATORS,
                    )
                    experts.append(expert)
                    recoveries.append(program_recovery_metrics(task.programs[regime], expert))
                hard_seconds = time.perf_counter() - started
                started = time.perf_counter()
                single = search_chain_program(train_x, train_y, val_x, val_y, max_depth=2, operators=OPERATORS)
                single_seconds = time.perf_counter() - started
                joined_train = np.column_stack((train_x, train_context))
                joined_val = np.column_stack((val_x, val_context))
                neural = NeuralMixtureRegressor(seed=int(seed), epochs=int(config["neural_epochs"]), device=str(config["device"]), n_experts=int(n_regimes))
                started = time.perf_counter()
                neural.fit(joined_train, train_y, (joined_val, val_y))
                neural_seconds = time.perf_counter() - started
                baselines = {}
                for name in ("MLP", "RandomForest"):
                    baseline = build_baseline(name, int(seed), str(config["device"]))
                    if name == "MLP":
                        baseline.epochs = int(config["neural_epochs"])
                    started = time.perf_counter()
                    baseline.fit(joined_train, train_y, (joined_val, val_y))
                    baselines[name] = (baseline, time.perf_counter() - started)
                for split, (features, context, regimes, targets) in tests.items():
                    joined = np.column_stack((features, context))
                    oracle_prediction = np.empty(len(features))
                    hard_prediction = np.empty(len(features))
                    for regime in range(int(n_regimes)):
                        mask = regimes == regime
                        oracle_prediction[mask] = np.asarray(task.programs[regime](features[mask]))
                        hard_prediction[mask] = np.asarray(experts[regime](features[mask]))
                    neural_prediction, probabilities = neural.predict_with_routing(joined)
                    neural_accuracy = aligned_accuracy(regimes, probabilities.argmax(axis=1), int(n_regimes))
                    mean_operator = float(np.mean([recovery["operator_accuracy"] for recovery in recoveries]))
                    functional = float(regression_metrics(targets, hard_prediction)["nrmse"] < 1.0e-3)
                    outputs = [
                        ("OracleRouter", oracle_prediction, 0.0, 1.0, 1.0, 1.0),
                        ("HardProgramRouter", hard_prediction, hard_seconds, 1.0, functional, mean_operator),
                        ("SingleProgram", np.asarray(single(features)), single_seconds, -1.0, -1.0, -1.0),
                        ("NeuralMoE", neural_prediction, neural_seconds, neural_accuracy, -1.0, -1.0),
                    ]
                    outputs.extend((name, model.predict(joined), elapsed, -1.0, -1.0, -1.0) for name, (model, elapsed) in baselines.items())
                    for model, prediction, elapsed, regime_accuracy, recovered_function, operator in outputs:
                        records.append(
                            {
                                "git_commit": commit,
                                "git_dirty": dirty,
                                "task_id": task.task_id,
                                "task_seed": task.seed,
                                "seed": seed,
                                "n_regimes": n_regimes,
                                "model": model,
                                "split": split,
                                "training_seconds": elapsed,
                                "regime_accuracy": regime_accuracy,
                                "functional_recovery": recovered_function,
                                "operator_accuracy": operator,
                                **regression_metrics(targets, prediction),
                            }
                        )
                write_csv(output / "records.csv", records)
            task_counter += 1
            print(f"completed task {task_counter}/{total_tasks}: regimes={n_regimes}", flush=True)
    summary = summarize(records)
    write_csv(output / "summary.csv", summary)
    decision = evaluate_gate(summary, config["gate"])
    expected = len(config["regime_counts"]) * int(config["tasks_per_count"]) * len(config["seeds"]) * 6 * 2
    audit = {
        "git_commit": commit,
        "git_dirty": dirty,
        "expected_records": expected,
        "observed_records": len(records),
        "all_finite": all(math.isfinite(float(row["nrmse"])) for row in records),
        "confidence_interval_unit": "task (training seeds averaged within task)",
        "known_categorical_regime_labels": True,
        "gate": decision,
    }
    audit["audit_passed"] = expected == len(records) and audit["all_finite"] and decision["passed"]
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    plot(summary, output / "regime_scaling.png")
    print(json.dumps(audit, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
