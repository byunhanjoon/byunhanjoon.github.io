#!/usr/bin/env python3
"""Phase F heterogeneous-type operator ablation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from tabalu.baselines import LearnedEmbeddingRegressor, ManualPreprocessingMLP
from tabalu.evaluation import regression_metrics
from tabalu.models.typed import SparseTypedProgram
from tabalu.synthetic import generate_heterogeneous_task, sample_heterogeneous_split


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(str(row["model"]), str(row["split"]))].append(row)
    rng = np.random.default_rng(20261031)
    output: list[dict[str, Any]] = []
    for (model, split), rows in sorted(groups.items()):
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_task[str(row["task_id"])].append(row)
        task_values = np.array(
            [np.mean([float(row["nrmse"]) for row in task_rows]) for task_rows in by_task.values()]
        )
        indices = rng.integers(0, len(task_values), size=(2000, len(task_values)))
        bootstrap = task_values[indices].mean(axis=1)
        output.append(
            {
                "model": model,
                "split": split,
                "n_tasks": len(by_task),
                "n_observations": len(rows),
                "nrmse_mean": float(task_values.mean()),
                "nrmse_std_across_tasks": float(task_values.std(ddof=1)),
                "nrmse_ci95_low": float(np.quantile(bootstrap, 0.025)),
                "nrmse_ci95_high": float(np.quantile(bootstrap, 0.975)),
                "training_seconds_mean": float(np.mean([float(row["training_seconds"]) for row in rows])),
                "operation_count_mean": float(np.mean([float(row["operation_count"]) for row in rows])),
            }
        )
    return output


def evaluate_gate(summary: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    lookup = {(row["model"], row["split"]): float(row["nrmse_mean"]) for row in summary}
    full = lookup[("TypedFull", "future")]
    ablations = {
        name: lookup[(name, "future")] / max(full, 1.0e-12)
        for name in ("TypedNoOrdinal", "TypedNoTime", "TypedNoCategoricalConditions")
    }
    observed = {
        "typed_future_nrmse": full,
        "typed_vs_embedding_future_ratio": full / max(lookup[("LearnedEmbeddings", "future")], 1.0e-12),
        "typed_vs_manual_future_ratio": full / max(lookup[("ManualPreprocessingMLP", "future")], 1.0e-12),
        "family_ablation_degradation": ablations,
    }
    checks = {
        "typed_future_accuracy": full <= thresholds["typed_future_nrmse_max"],
        "beats_learned_embeddings": observed["typed_vs_embedding_future_ratio"] <= thresholds["typed_vs_embedding_future_ratio_max"],
        "beats_manual_neural_preprocessing": observed["typed_vs_manual_future_ratio"] <= thresholds["typed_vs_manual_future_ratio_max"],
        **{
            f"{name}_matters": ratio >= thresholds["minimum_family_ablation_degradation"]
            for name, ratio in ablations.items()
        },
    }
    return {"passed": all(checks.values()), "checks": checks, "observed": observed, "thresholds": thresholds}


def plot(summary: list[dict[str, Any]], output: Path) -> None:
    order = [
        "TypedFull",
        "TypedNoOrdinal",
        "TypedNoTime",
        "TypedNoCategoricalConditions",
        "ManualPreprocessingMLP",
        "LearnedEmbeddings",
    ]
    lookup = {(row["model"], row["split"]): row for row in summary}
    x = np.arange(len(order))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for offset, split in ((-width / 2, "iid"), (width / 2, "future")):
        axis.bar(
            x + offset,
            [max(float(lookup[(model, split)]["nrmse_mean"]), 1.0e-6) for model in order],
            width,
            label=split,
        )
    labels = [
        "Typed\nfull",
        "Typed w/o\nordinal",
        "Typed w/o\ntime",
        "Typed w/o\ncategory conditions",
        "Manual features\n+ MLP",
        "Learned\nembeddings",
    ]
    axis.set_yscale("log")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Normalized RMSE")
    axis.set_title("Heterogeneous types: operator-family ablation")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(output, dpi=200)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def run(config_path: Path) -> Path:
    warnings.filterwarnings("ignore", message=".*linear dependence in the dictionary.*")
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
        task = generate_heterogeneous_task(int(config["task_seed_start"]) + task_index)
        (output / "tasks" / f"{task.task_id}.json").write_text(
            json.dumps(
                {
                    "task_id": task.task_id,
                    "seed": task.seed,
                    "active_feature_names": task.active_feature_names,
                    "coefficients": task.coefficients,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        for seed in config["seeds"]:
            train_batch, clean_train_y = sample_heterogeneous_split(
                task, "train", int(config["train_rows"]), seed=int(seed)
            )
            validation_batch, clean_validation_y = sample_heterogeneous_split(
                task, "validation", int(config["validation_rows"]), seed=int(seed)
            )
            rng = np.random.default_rng(task.seed * 101 + int(seed))
            noise_scale = float(config["target_noise_relative_std"]) * max(float(clean_train_y.std()), 1.0e-6)
            train_y = clean_train_y + rng.normal(0, noise_scale, len(clean_train_y))
            validation_y = clean_validation_y + rng.normal(0, noise_scale, len(clean_validation_y))
            tests = {
                "iid": sample_heterogeneous_split(task, "iid", int(config["test_rows"]), seed=int(seed)),
                "future": sample_heterogeneous_split(
                    task,
                    "future",
                    int(config["test_rows"]),
                    seed=int(seed),
                    magnitude_multiplier=float(config["future_magnitude_multiplier"]),
                ),
            }
            model_specs = {
                "TypedFull": SparseTypedProgram(),
                "TypedNoOrdinal": SparseTypedProgram(include_ordinal=False),
                "TypedNoTime": SparseTypedProgram(include_time=False),
                "TypedNoCategoricalConditions": SparseTypedProgram(include_categorical_conditions=False),
                "ManualPreprocessingMLP": ManualPreprocessingMLP(int(seed), str(config["device"]), int(config["neural_epochs"])),
                "LearnedEmbeddings": LearnedEmbeddingRegressor(int(seed), str(config["device"]), int(config["neural_epochs"])),
            }
            for model_name, model in model_specs.items():
                started = time.perf_counter()
                model.fit(train_batch, train_y, validation=(validation_batch, validation_y))
                elapsed = time.perf_counter() - started
                operation_count = model.operation_count if isinstance(model, SparseTypedProgram) else -1
                for split, (test_batch, targets) in tests.items():
                    predictions = model.predict(test_batch)
                    records.append(
                        {
                            "git_commit": commit,
                            "git_dirty": dirty,
                            "task_id": task.task_id,
                            "task_seed": task.seed,
                            "seed": seed,
                            "model": model_name,
                            "split": split,
                            "magnitude_multiplier": 1.0 if split == "iid" else config["future_magnitude_multiplier"],
                            "training_seconds": elapsed,
                            "operation_count": operation_count,
                            **regression_metrics(targets, predictions),
                        }
                    )
            write_csv(output / "records.csv", records)
        print(f"completed task {task_index + 1}/{config['n_tasks']}: {task.task_id}", flush=True)
    summary = summarize(records)
    write_csv(output / "summary.csv", summary)
    decision = evaluate_gate(summary, config["gate"])
    expected = int(config["n_tasks"]) * len(config["seeds"]) * 6 * 2
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
    plot(summary, output / "typed_ablation.png")
    print(json.dumps(audit, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
