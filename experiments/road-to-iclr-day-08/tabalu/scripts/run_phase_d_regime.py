#!/usr/bin/env python3
"""Categorical-regime executable-program MoE pilot and neural-MoE control."""

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
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from tabalu.baselines import build_baseline
from tabalu.evaluation import program_recovery_metrics, regression_metrics
from tabalu.models.discrete_search import search_chain_program
from tabalu.models.router import ProgramMixture, SparseProgramRouter
from tabalu.synthetic import generate_regime_task, sample_regime_split


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_router(context: np.ndarray, labels: np.ndarray, seed: int, device: str) -> SparseProgramRouter:
    torch.manual_seed(seed)
    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    router = SparseProgramRouter(context.shape[1], n_regimes=2).to(torch_device)
    context_t = torch.as_tensor(context, dtype=torch.float32, device=torch_device)
    labels_t = torch.as_tensor(labels, dtype=torch.long, device=torch_device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=0.02, weight_decay=1.0e-5)
    for _ in range(250):
        optimizer.zero_grad(set_to_none=True)
        logits = router(context_t)
        loss = torch.nn.functional.cross_entropy(logits, labels_t)
        loss.backward()
        optimizer.step()
    router.eval()
    return router


def aligned_regime_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    direct = float(np.mean(truth == predicted))
    flipped = float(np.mean(truth == (1 - predicted)))
    return {
        "regime_accuracy": max(direct, flipped),
        "regime_ari": float(adjusted_rand_score(truth, predicted)),
        "regime_nmi": float(normalized_mutual_info_score(truth, predicted)),
    }


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(str(row["model"]), str(row["split"]))].append(row)
    rng = np.random.default_rng(20260921)
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
                "regime_accuracy_mean": float(values("regime_accuracy").mean()),
                "regime_ari_mean": float(values("regime_ari").mean()),
                "regime_nmi_mean": float(values("regime_nmi").mean()),
                "router_entropy_mean": float(values("router_entropy").mean()),
                "operator_accuracy_mean": float(values("operator_accuracy").mean()),
                "training_seconds_mean": float(values("training_seconds").mean()),
            }
        )
    return output


def gate(summary: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    lookup = {(row["model"], row["split"]): row for row in summary}
    program_ood = float(lookup[("ProgramMoE-hard", "ood")]["nrmse_mean"])
    neural_ood = float(lookup[("NeuralMoE", "ood")]["nrmse_mean"])
    single_ood = float(lookup[("SingleProgram", "ood")]["nrmse_mean"])
    program_iid = float(lookup[("ProgramMoE-hard", "iid")]["nrmse_mean"])
    neural_iid = float(lookup[("NeuralMoE", "iid")]["nrmse_mean"])
    regime_accuracy = float(lookup[("ProgramMoE-hard", "ood")]["regime_accuracy_mean"])
    operator_accuracy = float(lookup[("ProgramMoE-hard", "ood")]["operator_accuracy_mean"])
    observed = {
        "program_moe_vs_neural_moe_ood_ratio": program_ood / max(neural_ood, 1.0e-12),
        "program_moe_vs_single_program_ood_ratio": program_ood / max(single_ood, 1.0e-12),
        "program_moe_vs_neural_moe_iid_ratio": program_iid / max(neural_iid, 1.0e-12),
        "program_moe_regime_accuracy": regime_accuracy,
        "program_moe_operator_accuracy": operator_accuracy,
    }
    checks = {
        "beats_neural_moe_ood": observed["program_moe_vs_neural_moe_ood_ratio"]
        <= thresholds["program_moe_vs_neural_moe_ood_ratio_max"],
        "beats_single_program_ood": observed["program_moe_vs_single_program_ood_ratio"]
        <= thresholds["program_moe_vs_single_program_ood_ratio_max"],
        "competitive_neural_moe_iid": observed["program_moe_vs_neural_moe_iid_ratio"]
        <= thresholds["program_moe_vs_neural_moe_iid_ratio_max"],
        "recovers_regimes": regime_accuracy >= thresholds["program_moe_regime_accuracy_min"],
        "recovers_operators": operator_accuracy >= thresholds["program_moe_operator_accuracy_min"],
    }
    return {"passed": all(checks.values()), "checks": checks, "observed": observed, "thresholds": thresholds}


def plot(summary: list[dict[str, Any]], output: Path) -> None:
    models = sorted({str(row["model"]) for row in summary})
    x = np.arange(len(models))
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for offset, split in ((-width / 2, "iid"), (width / 2, "ood")):
        lookup = {(row["model"], row["split"]): row for row in summary}
        axes[0].bar(
            x + offset,
            [max(float(lookup[(model, split)]["nrmse_mean"]), 1.0e-6) for model in models],
            width,
            label=split.upper(),
        )
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, [model.replace("ProgramMoE-", "Prog-") for model in models], rotation=25, ha="right")
    axes[0].set_ylabel("Normalized RMSE")
    axes[0].set_title("Prediction under regime-frequency + magnitude shift")
    routed = [model for model in models if model in {"ProgramMoE-hard", "ProgramMoE-soft", "NeuralMoE"}]
    lookup = {(row["model"], row["split"]): row for row in summary}
    axes[1].bar(
        [model.replace("ProgramMoE-", "Prog-") for model in routed],
        [float(lookup[(model, "ood")]["regime_accuracy_mean"]) for model in routed],
        color="#35b779",
    )
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Permutation-aligned regime accuracy")
    axes[1].set_title("Regime recovery")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend()
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
        task = generate_regime_task(int(config["task_seed_start"]) + task_index, int(config["n_features"]))
        for seed in config["seeds"]:
            train = sample_regime_split(
                task,
                "train",
                int(config["train_rows"]),
                seed=int(seed),
                magnitude_multiplier=1,
                regime_one_probability=float(config["train_regime_one_probability"]),
            )
            validation = sample_regime_split(
                task,
                "validation",
                int(config["validation_rows"]),
                seed=int(seed),
                magnitude_multiplier=1,
                regime_one_probability=float(config["train_regime_one_probability"]),
            )
            tests = {
                "iid": sample_regime_split(
                    task,
                    "iid_test",
                    int(config["test_rows"]),
                    seed=int(seed),
                    magnitude_multiplier=1,
                    regime_one_probability=float(config["train_regime_one_probability"]),
                ),
                "ood": sample_regime_split(
                    task,
                    "ood_test",
                    int(config["test_rows"]),
                    seed=int(seed),
                    magnitude_multiplier=float(config["ood_multiplier"]),
                    regime_one_probability=float(config["ood_regime_one_probability"]),
                ),
            }
            train_x, train_context, train_regime, train_y = train
            val_x, val_context, _, val_y = validation
            operators = tuple(config["operators"])
            started = time.perf_counter()
            clusters = KMeans(n_clusters=2, random_state=int(seed), n_init=10).fit(train_context)
            train_cluster = clusters.labels_
            val_cluster = clusters.predict(val_context)
            experts = []
            recovery_scores = []
            for cluster in range(2):
                expert = search_chain_program(
                    train_x[train_cluster == cluster],
                    train_y[train_cluster == cluster],
                    val_x[val_cluster == cluster],
                    val_y[val_cluster == cluster],
                    max_depth=3,
                    operators=operators,
                )
                experts.append(expert)
                true_label = int(np.bincount(train_regime[train_cluster == cluster], minlength=2).argmax())
                recovery_scores.append(program_recovery_metrics(task.programs[true_label], expert))
            router = train_router(train_context, train_cluster, int(seed), str(config["device"]))
            program_mixture = ProgramMixture(experts, router)
            program_training_seconds = time.perf_counter() - started
            started = time.perf_counter()
            single_program = search_chain_program(
                train_x, train_y, val_x, val_y, max_depth=3, operators=operators
            )
            single_seconds = time.perf_counter() - started
            combined_train = np.column_stack([train_x, train_context])
            combined_validation = np.column_stack([val_x, val_context])
            baselines = {}
            baseline_seconds = {}
            for name in ("MLP", "NeuralMoE", "RandomForest"):
                started = time.perf_counter()
                baseline = build_baseline(name, int(seed), device=str(config["device"]))
                baseline.fit(combined_train, train_y, (combined_validation, val_y))
                baselines[name] = baseline
                baseline_seconds[name] = time.perf_counter() - started
            for split, (test_x, test_context, test_regime, test_y) in tests.items():
                device = next(router.parameters()).device
                test_x_t = torch.as_tensor(test_x, dtype=torch.float32, device=device)
                test_context_t = torch.as_tensor(test_context, dtype=torch.float32, device=device)
                for hard, name in ((True, "ProgramMoE-hard"), (False, "ProgramMoE-soft")):
                    with torch.no_grad():
                        prediction_t, probability_t = program_mixture(test_x_t, test_context_t, hard=hard)
                    predictions = prediction_t.cpu().numpy()
                    probabilities = probability_t.cpu().numpy()
                    routing = aligned_regime_metrics(test_regime, probabilities.argmax(axis=1))
                    entropy = float(
                        np.mean(-np.sum(probabilities * np.log(np.clip(probabilities, 1.0e-12, 1)), axis=1))
                    )
                    records.append(
                        {
                            "git_commit": commit,
                            "git_dirty": dirty,
                            "task_id": task.task_id,
                            "task_seed": task.seed,
                            "seed": seed,
                            "model": name,
                            "split": split,
                            "training_seconds": program_training_seconds,
                            "router_entropy": entropy,
                            "operator_accuracy": float(
                                np.mean([score["operator_accuracy"] for score in recovery_scores])
                            ),
                            **routing,
                            **regression_metrics(test_y, predictions),
                        }
                    )
                records.append(
                    {
                        "git_commit": commit,
                        "git_dirty": dirty,
                        "task_id": task.task_id,
                        "task_seed": task.seed,
                        "seed": seed,
                        "model": "SingleProgram",
                        "split": split,
                        "training_seconds": single_seconds,
                        "router_entropy": -1.0,
                        "operator_accuracy": float(
                            max(
                                program_recovery_metrics(program, single_program)["operator_accuracy"]
                                for program in task.programs
                            )
                        ),
                        "regime_accuracy": -1.0,
                        "regime_ari": -1.0,
                        "regime_nmi": -1.0,
                        **regression_metrics(test_y, single_program(test_x)),
                    }
                )
                combined_test = np.column_stack([test_x, test_context])
                for name, baseline in baselines.items():
                    if name == "NeuralMoE":
                        predictions, probabilities = baseline.predict_with_routing(combined_test)
                        routing = aligned_regime_metrics(test_regime, probabilities.argmax(axis=1))
                        entropy = float(
                            np.mean(-np.sum(probabilities * np.log(np.clip(probabilities, 1.0e-12, 1)), axis=1))
                        )
                    else:
                        predictions = baseline.predict(combined_test)
                        routing = {"regime_accuracy": -1.0, "regime_ari": -1.0, "regime_nmi": -1.0}
                        entropy = -1.0
                    records.append(
                        {
                            "git_commit": commit,
                            "git_dirty": dirty,
                            "task_id": task.task_id,
                            "task_seed": task.seed,
                            "seed": seed,
                            "model": name,
                            "split": split,
                            "training_seconds": baseline_seconds[name],
                            "router_entropy": entropy,
                            "operator_accuracy": -1.0,
                            **routing,
                            **regression_metrics(test_y, predictions),
                        }
                    )
            (output / "programs" / f"{task.task_id}__seed-{seed}.json").write_text(
                json.dumps(
                    {
                        "truth": [program.to_dict() for program in task.programs],
                        "experts": [program.to_dict() for program in experts],
                        "single_program": single_program.to_dict(),
                        "recovery": recovery_scores,
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
        "all_finite": all(
            math.isfinite(float(row[key])) for row in records for key in ("nrmse", "regime_accuracy")
        ),
        "confidence_interval_unit": "task (training seeds averaged within task)",
        "gate": decision,
    }
    audit["audit_passed"] = expected == len(records) and audit["all_finite"] and decision["passed"]
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    plot(summary, output / "regime_pilot.png")
    print(json.dumps(audit, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
