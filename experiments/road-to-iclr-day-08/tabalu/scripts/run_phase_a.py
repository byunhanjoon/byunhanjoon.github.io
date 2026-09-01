#!/usr/bin/env python3
"""Run the preregistered Phase-A arithmetic extrapolation panel."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from tabalu.baselines import build_baseline
from tabalu.evaluation import program_recovery_metrics, regression_metrics
from tabalu.models.discrete_search import search_chain_program
from tabalu.synthetic import generate_program_task, regenerate_split
from tabalu.training import ProgramTrainingConfig, train_program


def git_state() -> tuple[str, bool]:
    repository = PACKAGE_ROOT.parents[1]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=repository, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def predict_differentiable(
    model: torch.nn.Module, features: np.ndarray, temperature: float, hard: bool
) -> np.ndarray:
    device = next(model.parameters()).device
    values = torch.as_tensor(features, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        prediction, _ = model(values, temperature=temperature, hard=hard)
    return prediction.detach().cpu().numpy()


def base_record(
    *,
    commit: str,
    dirty: bool,
    task: Any,
    seed: int,
    model: str,
    split: str,
    multiplier: float,
    training_seconds: float,
    parameter_count: int,
    program_complexity: int | None,
    compiled_metric: float | None,
    hyperparameters: dict[str, Any],
) -> dict[str, Any]:
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "dataset": task.task_id,
        "task_generation_seed": task.seed,
        "seed": seed,
        "split": split,
        "model": model,
        "hyperparameters": json.dumps(hyperparameters, sort_keys=True),
        "training_seconds": training_seconds,
        "extrapolation_multiplier": multiplier,
        "parameter_count": parameter_count,
        "program_complexity": program_complexity,
        "regime_count": 1,
        "residual_usage": 0.0,
        "compiled_metric": compiled_metric,
        "status": "ok",
        "error": "",
    }


def attach_metrics(record: dict[str, Any], targets: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    record.update(regression_metrics(targets, predictions))
    return record


def bootstrap_interval(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) == 1:
        return float(values[0]), float(values[0])
    indices = rng.integers(0, len(values), size=(2000, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["status"] == "ok":
            groups[(record["model"], float(record["extrapolation_multiplier"]))].append(record)
    rng = np.random.default_rng(20260831)
    summary: list[dict[str, Any]] = []
    for (model, multiplier), rows in sorted(groups.items()):
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_task[str(row["dataset"])].append(row)
        values = np.array(
            [np.mean([float(row["nrmse"]) for row in task_rows]) for task_rows in by_task.values()]
        )
        low, high = bootstrap_interval(values, rng)
        summary.append(
            {
                "model": model,
                "extrapolation_multiplier": multiplier,
                "n_tasks": len(values),
                "n_observations": len(rows),
                "nrmse_mean": float(values.mean()),
                "nrmse_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "nrmse_ci95_low": low,
                "nrmse_ci95_high": high,
                "nrmse_median": float(np.median(values)),
                "rmse_mean": float(np.mean([float(row["rmse"]) for row in rows])),
                "r2_mean": float(np.mean([float(row["r2"]) for row in rows])),
                "training_seconds_mean": float(
                    np.mean([float(row["training_seconds"]) for row in rows])
                ),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(summary: list[dict[str, Any]], output: Path) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary:
        grouped[str(row["model"])].append(row)
    fig, axis = plt.subplots(figsize=(8.3, 5.2), constrained_layout=True)
    numerical_floor = 1.0e-5
    for model, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: float(row["extrapolation_multiplier"]))
        x = np.array([float(row["extrapolation_multiplier"]) for row in rows])
        y = np.maximum(
            np.array([float(row["nrmse_mean"]) for row in rows]), numerical_floor
        )
        low = np.maximum(
            np.array([float(row["nrmse_ci95_low"]) for row in rows]), numerical_floor
        )
        high = np.maximum(
            np.array([float(row["nrmse_ci95_high"]) for row in rows]), numerical_floor
        )
        axis.plot(x, y, marker="o", label=model)
        axis.fill_between(x, low, high, alpha=0.10)
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xticks(sorted({float(row["extrapolation_multiplier"]) for row in summary}))
    axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axis.set_xlabel("Extrapolation multiplier")
    axis.set_ylabel("Normalized RMSE (lower is better)")
    axis.set_title("Phase A: exact execution under magnitude shift")
    axis.annotate(
        "TabALU soft / hard / compiled: numerically exact\n(shown at plotting floor)",
        xy=(max(float(row["extrapolation_multiplier"]) for row in summary), numerical_floor),
        xytext=(0.48, 0.12),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "linewidth": 0.8},
        fontsize=8,
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def gate_decision(summary: list[dict[str, Any]], multipliers: list[float]) -> dict[str, Any]:
    lookup = {
        (str(row["model"]), float(row["extrapolation_multiplier"])): float(row["nrmse_mean"])
        for row in summary
    }
    extreme = float(max(multipliers))
    required = [("TabALU-compiled", extreme), ("TabALU-soft", 1.0), ("MLP", extreme), ("MLP", 1.0)]
    if any(key not in lookup for key in required):
        return {"passed": False, "reason": "missing required successful model cells", "operational_rule": {}}
    ood_ratio = lookup[("TabALU-compiled", extreme)] / max(lookup[("MLP", extreme)], 1.0e-12)
    iid_ratio = lookup[("TabALU-soft", 1.0)] / max(lookup[("MLP", 1.0)], 1.0e-12)
    compiled_retention = lookup[("TabALU-compiled", extreme)] / max(
        lookup[("TabALU-soft", extreme)], 1.0e-12
    )
    rule = {
        "extreme_ood_compiled_vs_mlp_nrmse_ratio_max": 0.50,
        "iid_soft_vs_mlp_nrmse_ratio_max": 1.25,
        "extreme_compiled_vs_soft_nrmse_ratio_max": 1.25,
    }
    passed = ood_ratio <= 0.50 and iid_ratio <= 1.25 and compiled_retention <= 1.25
    return {
        "passed": passed,
        "reason": "all preregistered pilot thresholds passed" if passed else "one or more pilot thresholds failed",
        "operational_rule": rule,
        "observed": {
            "extreme_ood_compiled_vs_mlp_nrmse_ratio": ood_ratio,
            "iid_soft_vs_mlp_nrmse_ratio": iid_ratio,
            "extreme_compiled_vs_soft_nrmse_ratio": compiled_retention,
        },
    }


def run(config_path: Path) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = (PACKAGE_ROOT / config["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "tasks").mkdir(exist_ok=True)
    (output / "programs").mkdir(exist_ok=True)
    (output / "histories").mkdir(exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    commit, dirty = git_state()
    multipliers = [float(value) for value in config["multipliers"]]
    training_config = ProgramTrainingConfig(
        n_nodes=int(config["program_search"]["n_nodes"]),
        operators=tuple(config["program_search"]["operators"]),
        selector=str(config["program_search"]["selector"]),
        epochs=int(config["program_search"]["epochs"]),
        learning_rate=float(config["program_search"]["learning_rate"]),
        entropy_weight=float(config["program_search"]["entropy_weight"]),
        patience=int(config["program_search"]["patience"]),
        device=str(config["device"]),
        discrete_warm_start=bool(config["program_search"].get("discrete_warm_start", True)),
    )
    records: list[dict[str, Any]] = []
    recovery: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for task_index in range(int(config["n_tasks"])):
        task_seed = int(config["task_seed_start"]) + task_index
        task = generate_program_task(
            task_seed,
            n_features=int(config["n_features"]),
            depth_range=tuple(config["depth_range"]),
            n_irrelevant=int(config["n_irrelevant"]),
            operators=tuple(config["generation_operators"]),
        )
        (output / "tasks" / f"{task.task_id}.json").write_text(
            json.dumps(task.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        train_x, train_y = regenerate_split(task, "train", int(config["train_rows"]))
        validation_x, validation_y = regenerate_split(
            task, "validation", int(config["validation_rows"])
        )
        evaluation = {
            multiplier: regenerate_split(
                task,
                "iid_test" if multiplier == 1.0 else "ood_test",
                int(config["test_rows"]),
                multiplier,
            )
            for multiplier in multipliers
        }
        warm_started_at = time.perf_counter()
        warm_start_program = search_chain_program(
            train_x,
            train_y,
            validation_x,
            validation_y,
            max_depth=training_config.n_nodes,
            operators=tuple(operator for operator in training_config.operators if operator != "identity"),
        )
        warm_start_seconds = time.perf_counter() - warm_started_at
        for seed in config["seeds"]:
            seed = int(seed)
            try:
                trained = train_program(
                    train_x,
                    train_y,
                    validation_x,
                    validation_y,
                    seed=seed,
                    config=training_config,
                    warm_start_program=warm_start_program,
                )
                compiled = trained.compiled
                (output / "programs" / f"{task.task_id}__seed-{seed}.json").write_text(
                    json.dumps(
                        {
                            "truth": task.program.to_dict(),
                            "truth_expression": task.program.expression(),
                            "compiled": compiled.to_dict(),
                            "compiled_expression": compiled.expression(),
                            "selector_probabilities": trained.model.selector_probabilities(),
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                (output / "histories" / f"{task.task_id}__seed-{seed}.json").write_text(
                    json.dumps(trained.history, indent=2) + "\n", encoding="utf-8"
                )
                recovered = {
                    "dataset": task.task_id,
                    "seed": seed,
                    "truth_expression": task.program.expression(),
                    "compiled_expression": compiled.expression(),
                    **program_recovery_metrics(task.program, compiled),
                }
                recovery.append(recovered)
                compiled_validation = regression_metrics(validation_y, compiled(validation_x))["nrmse"]
                parameter_count = sum(parameter.numel() for parameter in trained.model.parameters())
                for multiplier, (features, targets) in evaluation.items():
                    for model_name, predictions in (
                        (
                            "TabALU-soft",
                            predict_differentiable(
                                trained.model, features, training_config.end_temperature, hard=False
                            ),
                        ),
                        (
                            "TabALU-hard",
                            predict_differentiable(
                                trained.model, features, training_config.end_temperature, hard=True
                            ),
                        ),
                        ("TabALU-compiled", np.asarray(compiled(features))),
                    ):
                        record = base_record(
                            commit=commit,
                            dirty=dirty,
                            task=task,
                            seed=seed,
                            model=model_name,
                            split="iid_test" if multiplier == 1.0 else "ood_test",
                            multiplier=multiplier,
                            training_seconds=trained.training_seconds + warm_start_seconds,
                            parameter_count=parameter_count if model_name != "TabALU-compiled" else 0,
                            program_complexity=compiled.operation_count,
                            compiled_metric=float(compiled_validation),
                            hyperparameters=training_config.to_dict(),
                        )
                        records.append(attach_metrics(record, targets, predictions))
            except Exception as error:
                failure = {
                    "dataset": task.task_id,
                    "seed": seed,
                    "model": "TabALU",
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                print(json.dumps(failure), file=sys.stderr, flush=True)
            for baseline_name in config["baselines"]:
                started = time.perf_counter()
                try:
                    baseline = build_baseline(str(baseline_name), seed, device=str(config["device"]))
                    baseline.fit(train_x, train_y, (validation_x, validation_y))
                    elapsed = time.perf_counter() - started
                    for multiplier, (features, targets) in evaluation.items():
                        predictions = baseline.predict(features)
                        record = base_record(
                            commit=commit,
                            dirty=dirty,
                            task=task,
                            seed=seed,
                            model=str(baseline_name),
                            split="iid_test" if multiplier == 1.0 else "ood_test",
                            multiplier=multiplier,
                            training_seconds=elapsed,
                            parameter_count=-1,
                            program_complexity=None,
                            compiled_metric=None,
                            hyperparameters={"budget": "configs/baseline_search_spaces.yaml"},
                        )
                        records.append(attach_metrics(record, targets, predictions))
                except Exception as error:
                    failure = {
                        "dataset": task.task_id,
                        "seed": seed,
                        "model": baseline_name,
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                    }
                    failures.append(failure)
                    print(json.dumps(failure), file=sys.stderr, flush=True)
        write_csv(output / "records.csv", records)
        write_csv(output / "program_recovery.csv", recovery)
        (output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
        print(f"completed task {task_index + 1}/{config['n_tasks']}: {task.task_id}", flush=True)
    summary = summarize(records)
    write_csv(output / "summary.csv", summary)
    decision = gate_decision(summary, multipliers)
    audit = {
        "config": config,
        "git_commit": commit,
        "git_dirty": dirty,
        "record_count": len(records),
        "failure_count": len(failures),
        "task_count": int(config["n_tasks"]),
        "seed_count": len(config["seeds"]),
        "gate": decision,
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    plot_curves(summary, output / "extrapolation_curve.png")
    print(json.dumps(audit, indent=2), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
