#!/usr/bin/env python3
"""Temporal shared-structure versus changing-parameter experiment."""

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
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from tabalu.baselines import build_baseline
from tabalu.evaluation import program_recovery_metrics, regression_metrics
from tabalu.models.discrete_search import search_chain_program
from tabalu.models.shared_structure import ContextCoefficientModel, fit_regime_coefficients
from tabalu.synthetic import generate_temporal_task, sample_temporal_split


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def standardized_targets(
    train_targets: np.ndarray,
    train_regimes: np.ndarray,
    validation_targets: np.ndarray,
    validation_regimes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train_output = np.empty_like(train_targets)
    validation_output = np.empty_like(validation_targets)
    for regime in range(2):
        mask = train_regimes == regime
        mean = float(train_targets[mask].mean())
        scale = max(float(train_targets[mask].std()), 1.0e-6)
        train_output[mask] = (train_targets[mask] - mean) / scale
        validation_mask = validation_regimes == regime
        validation_output[validation_mask] = (validation_targets[validation_mask] - mean) / scale
    return train_output, validation_output


def train_context_coefficients(
    base_program: Any,
    train_x: np.ndarray,
    train_time: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_time: np.ndarray,
    validation_y: np.ndarray,
    seed: int,
    device: str,
) -> ContextCoefficientModel:
    torch.manual_seed(seed)
    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = ContextCoefficientModel().to(torch_device)
    base_train = torch.as_tensor(base_program(train_x), dtype=torch.float32, device=torch_device)
    time_train = torch.as_tensor(train_time, dtype=torch.float32, device=torch_device)
    y_train = torch.as_tensor(train_y, dtype=torch.float32, device=torch_device)
    base_validation = torch.as_tensor(
        base_program(validation_x), dtype=torch.float32, device=torch_device
    )
    time_validation = torch.as_tensor(validation_time, dtype=torch.float32, device=torch_device)
    y_validation = torch.as_tensor(validation_y, dtype=torch.float32, device=torch_device)
    variance = y_train.var(unbiased=False).clamp_min(1.0e-8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1.0e-5)
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best = float("inf")
    stale = 0
    for _ in range(800):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(base_train, time_train)
        loss = (prediction - y_train).square().mean() / variance
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = float((model(base_validation, time_validation) - y_validation).square().mean())
        if validation + 1.0e-7 < best:
            best = validation
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 120:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(str(row["model"]), str(row["split"]))].append(row)
    rng = np.random.default_rng(20260931)
    output: list[dict[str, Any]] = []
    for (model, split), rows in sorted(groups.items()):
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
                "model": model,
                "split": split,
                "n_tasks": len(by_task),
                "n_observations": len(rows),
                "nrmse_mean": float(nrmse.mean()),
                "nrmse_std_across_tasks": float(nrmse.std(ddof=1)),
                "nrmse_ci95_low": float(np.quantile(means, 0.025)),
                "nrmse_ci95_high": float(np.quantile(means, 0.975)),
                "operator_accuracy_mean": float(values("operator_accuracy").mean()),
                "post_change_train_rows_mean": float(values("post_change_train_rows").mean()),
                "training_seconds_mean": float(values("training_seconds").mean()),
            }
        )
    return output


def gate(summary: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    lookup = {(row["model"], row["split"]): row for row in summary}
    shared = float(lookup[("SharedStructureCoefficients", "future")]["nrmse_mean"])
    global_score = float(lookup[("GlobalProgram", "future")]["nrmse_mean"])
    independent = float(lookup[("IndependentPrograms", "future")]["nrmse_mean"])
    neural = float(lookup[("NeuralMoE", "future")]["nrmse_mean"])
    context = float(lookup[("ContextCoefficients", "future")]["nrmse_mean"])
    operator = float(lookup[("SharedStructureCoefficients", "future")]["operator_accuracy_mean"])
    observed = {
        "shared_vs_global_future_ratio": shared / max(global_score, 1.0e-12),
        "shared_vs_independent_future_ratio": shared / max(independent, 1.0e-12),
        "shared_vs_neural_moe_future_ratio": shared / max(neural, 1.0e-12),
        "context_vs_shared_future_ratio": context / max(shared, 1.0e-12),
        "shared_operator_accuracy": operator,
    }
    checks = {
        "beats_global": observed["shared_vs_global_future_ratio"]
        <= thresholds["shared_vs_global_future_ratio_max"],
        "competitive_independent": observed["shared_vs_independent_future_ratio"]
        <= thresholds["shared_vs_independent_future_ratio_max"],
        "beats_neural_moe": observed["shared_vs_neural_moe_future_ratio"]
        <= thresholds["shared_vs_neural_moe_future_ratio_max"],
        "context_coefficients_track": observed["context_vs_shared_future_ratio"]
        <= thresholds["context_vs_shared_future_ratio_max"],
        "recovers_structure": operator >= thresholds["shared_operator_accuracy_min"],
    }
    return {"passed": all(checks.values()), "checks": checks, "observed": observed, "thresholds": thresholds}


def plot(summary: list[dict[str, Any]], output: Path) -> None:
    models = sorted({str(row["model"]) for row in summary})
    lookup = {(row["model"], row["split"]): row for row in summary}
    x = np.arange(len(models))
    width = 0.36
    figure, axis = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    for offset, split in ((-width / 2, "iid"), (width / 2, "future")):
        axis.bar(
            x + offset,
            [max(float(lookup[(model, split)]["nrmse_mean"]), 1.0e-6) for model in models],
            width,
            label=split,
        )
    axis.set_yscale("log")
    axis.set_xticks(x, [model.replace("Coefficients", "\ncoefficients") for model in models], rotation=20, ha="right")
    axis.set_ylabel("Normalized RMSE")
    axis.set_title("Temporal shift: invariant graph, changing coefficients")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output, dpi=200)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def run(config_path: Path) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = (PACKAGE_ROOT / config["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "programs").mkdir(exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    repository = PACKAGE_ROOT.parents[1]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repository, text=True).strip())
    records: list[dict[str, Any]] = []
    for task_index in range(int(config["n_tasks"])):
        task = generate_temporal_task(
            int(config["task_seed_start"]) + task_index,
            int(config["n_features"]),
            float(config["change_point"]),
        )
        for seed in config["seeds"]:
            train = sample_temporal_split(task, "train", int(config["train_rows"]), seed=int(seed), magnitude_multiplier=1)
            validation = sample_temporal_split(
                task, "validation", int(config["validation_rows"]), seed=int(seed), magnitude_multiplier=1
            )
            tests = {
                "iid": sample_temporal_split(
                    task, "iid_test", int(config["test_rows"]), seed=int(seed), magnitude_multiplier=1
                ),
                "future": sample_temporal_split(
                    task,
                    "future_test",
                    int(config["test_rows"]),
                    seed=int(seed),
                    magnitude_multiplier=float(config["future_multiplier"]),
                ),
            }
            train_x, train_time, train_regime, clean_train_y = train
            val_x, val_time, val_regime, clean_val_y = validation
            rng = np.random.default_rng(task.seed * 101 + int(seed))
            train_y = clean_train_y + rng.normal(
                0,
                float(config["target_noise_relative_std"]) * max(float(clean_train_y.std()), 1.0e-6),
                len(clean_train_y),
            )
            val_y = clean_val_y + rng.normal(
                0,
                float(config["target_noise_relative_std"]) * max(float(clean_val_y.std()), 1.0e-6),
                len(clean_val_y),
            )
            operators = tuple(config["operators"])
            model_predictions: dict[str, Any] = {}
            model_times: dict[str, float] = {}
            model_operator: dict[str, float] = {}
            started = time.perf_counter()
            global_program = search_chain_program(train_x, train_y, val_x, val_y, max_depth=3, operators=operators)
            model_times["GlobalProgram"] = time.perf_counter() - started
            model_operator["GlobalProgram"] = float(
                program_recovery_metrics(task.program, global_program)["operator_accuracy"]
            )
            started = time.perf_counter()
            independent = []
            independent_recovery = []
            for regime in range(2):
                program = search_chain_program(
                    train_x[train_regime == regime],
                    train_y[train_regime == regime],
                    val_x[val_regime == regime],
                    val_y[val_regime == regime],
                    max_depth=3,
                    operators=operators,
                )
                independent.append(program)
                independent_recovery.append(program_recovery_metrics(task.program, program))
            model_times["IndependentPrograms"] = time.perf_counter() - started
            model_operator["IndependentPrograms"] = float(
                np.mean([score["operator_accuracy"] for score in independent_recovery])
            )
            normalized_train, normalized_validation = standardized_targets(
                train_y, train_regime, val_y, val_regime
            )
            started = time.perf_counter()
            shared_base = search_chain_program(
                train_x,
                normalized_train,
                val_x,
                normalized_validation,
                max_depth=3,
                operators=operators,
            )
            shared = fit_regime_coefficients(shared_base, train_x, train_y, train_regime, 2)
            model_times["SharedStructureCoefficients"] = time.perf_counter() - started
            shared_recovery = program_recovery_metrics(task.program, shared_base)
            model_operator["SharedStructureCoefficients"] = float(shared_recovery["operator_accuracy"])
            started = time.perf_counter()
            context_model = train_context_coefficients(
                shared_base,
                train_x,
                train_time,
                train_y,
                val_x,
                val_time,
                val_y,
                int(seed),
                str(config["device"]),
            )
            model_times["ContextCoefficients"] = time.perf_counter() - started
            model_operator["ContextCoefficients"] = float(shared_recovery["operator_accuracy"])
            combined_train = np.column_stack([train_x, train_time])
            combined_val = np.column_stack([val_x, val_time])
            baselines = {}
            for name in ("MLP", "NeuralMoE"):
                started = time.perf_counter()
                baseline = build_baseline(name, int(seed), device=str(config["device"]))
                baseline.fit(combined_train, train_y, (combined_val, val_y))
                baselines[name] = baseline
                model_times[name] = time.perf_counter() - started
                model_operator[name] = -1.0
            for split, (test_x, test_time, test_regime, test_y) in tests.items():
                model_predictions["GlobalProgram"] = global_program(test_x)
                model_predictions["IndependentPrograms"] = np.where(
                    test_regime == 0, independent[0](test_x), independent[1](test_x)
                )
                model_predictions["SharedStructureCoefficients"] = shared(test_x, test_regime)
                device = next(context_model.parameters()).device
                with torch.no_grad():
                    model_predictions["ContextCoefficients"] = (
                        context_model(
                            torch.as_tensor(shared_base(test_x), dtype=torch.float32, device=device),
                            torch.as_tensor(test_time, dtype=torch.float32, device=device),
                        )
                        .cpu()
                        .numpy()
                    )
                combined_test = np.column_stack([test_x, test_time])
                for name, baseline in baselines.items():
                    model_predictions[name] = baseline.predict(combined_test)
                for name in config["models"]:
                    records.append(
                        {
                            "git_commit": commit,
                            "git_dirty": dirty,
                            "task_id": task.task_id,
                            "task_seed": task.seed,
                            "seed": seed,
                            "model": name,
                            "split": split,
                            "training_seconds": model_times[name],
                            "operator_accuracy": model_operator[name],
                            "post_change_train_rows": int((train_regime == 1).sum()),
                            **regression_metrics(test_y, model_predictions[name]),
                        }
                    )
            (output / "programs" / f"{task.task_id}__seed-{seed}.json").write_text(
                json.dumps(
                    {
                        "truth": task.program.to_dict(),
                        "global": global_program.to_dict(),
                        "independent": [program.to_dict() for program in independent],
                        "shared": shared_base.to_dict(),
                        "shared_scales": shared.scales.tolist(),
                        "shared_biases": shared.biases.tolist(),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        write_csv(output / "records.csv", records)
        print(f"completed task {task_index + 1}/{config['n_tasks']}: {task.task_id}", flush=True)
    summary = summarize(records)
    write_csv(output / "summary.csv", summary)
    decision = gate(summary, config["gate"])
    expected = int(config["n_tasks"]) * len(config["seeds"]) * len(config["models"]) * 2
    audit = {
        "git_commit": commit,
        "git_dirty": dirty,
        "expected_records": expected,
        "observed_records": len(records),
        "all_finite": all(math.isfinite(float(row["nrmse"])) for row in records),
        "known_regime_labels_used_for_structure_parameter_ablation": True,
        "confidence_interval_unit": "task (training seeds averaged within task)",
        "gate": decision,
    }
    audit["audit_passed"] = expected == len(records) and audit["all_finite"] and decision["passed"]
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    plot(summary, output / "temporal_shared_structure.png")
    print(json.dumps(audit, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
