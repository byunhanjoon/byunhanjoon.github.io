#!/usr/bin/env python3
"""Frozen 300-vs-600 epoch convergence check for representative MPE cells."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from neural_benchmark import TRAINING_SEEDS, fit_validation, selected_bandwidth
from representations import load_task, representation_tables, split_row_indices
from ridge_benchmark import ordinary_design


HERE = Path(__file__).resolve().parent
RUNNABLE = ("acs_occupation", "tlc_pickup_zone")
FROZEN_UNAVAILABLE = {
    "amazon_leaf_category": (
        "NOT RUN — the frozen Amazon file has zero rows with both a positive "
        "price and nonempty category hierarchy"
    )
}


def atomic_json(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def run_task(task_name: str, device_name: str) -> dict[str, object]:
    split = 0
    setting = "full_table"
    backbone = "mlp"
    representation = "mpe"
    base_path = (
        HERE / "raw" / "neural_cells" /
        f"{task_name}__split{split}__{setting}__{backbone}__{representation}.json"
    )
    if not base_path.exists():
        raise FileNotFoundError(f"complete the frozen 300-epoch cell first: {base_path}")
    base = json.loads(base_path.read_text())
    if base.get("status") != "complete" or base.get("mpe_implementation_version") != 2:
        raise RuntimeError(f"invalid or legacy base cell: {base_path}")

    task = load_task(task_name)
    parts = split_row_indices(task, split)
    bandwidth = selected_bandwidth(task_name, split, setting)
    tables, _ = representation_tables(task, split, bandwidth)
    table = tables[representation]
    state_indices = task.row_state_indices()
    representation_design = sparse.csr_matrix(table[state_indices], dtype=np.float32)
    ordinary = ordinary_design(task, parts["train"])
    design = sparse.hstack([ordinary, representation_design], format="csr")
    raw_target = pd.to_numeric(task.rows["target"], errors="raise").to_numpy(np.float64)
    target_scale = float(raw_target[parts["train"]].std()) or 1.0
    target = (raw_target - float(raw_target[parts["train"]].mean())) / target_scale
    states = task.rows["field_state"].astype(str).to_numpy()
    selected_config = dict(base["selected_config"])
    device = torch.device(device_name)
    training_design: sparse.csr_matrix | torch.Tensor
    if device.type == "cuda":
        training_design = torch.from_numpy(design.toarray().astype(np.float32, copy=False)).to(
            device, non_blocking=True
        )
    else:
        training_design = design

    repeated = []
    for seed in TRAINING_SEEDS:
        state, telemetry = fit_validation(
            training_design,
            target,
            states,
            parts["train"],
            parts["validation"],
            backbone,
            selected_config,
            seed,
            device,
            max_epochs=600,
            patience=30,
            tokenized_representation_size=int(table.shape[1]),
            token_dimension=32,
        )
        del state
        repeated.append({"seed": seed, **telemetry})
        print(
            f"convergence {task_name} seed={seed} "
            f"val={telemetry['validation_score']:.6f} "
            f"epoch={telemetry['best_epoch']}/{telemetry['stop_epoch']}",
            flush=True,
        )

    original_scores = [float(row["validation_score"]) for row in base["final_fit_telemetry"]]
    repeated_scores = [float(row["validation_score"]) for row in repeated]
    original_mean = float(np.mean(original_scores))
    repeated_mean = float(np.mean(repeated_scores))
    relative_improvement = 100.0 * (original_mean - repeated_mean) / original_mean
    return {
        "status": "complete",
        "task": task_name,
        "split": split,
        "setting": setting,
        "backbone": backbone,
        "representation": representation,
        "selected_config": selected_config,
        "training_seeds": TRAINING_SEEDS,
        "base_max_epochs": 300,
        "repeat_max_epochs": 600,
        "patience": 30,
        "base_validation_scores": original_scores,
        "repeat_validation_scores": repeated_scores,
        "base_mean_validation": original_mean,
        "repeat_mean_validation": repeated_mean,
        "relative_validation_improvement_percent": relative_improvement,
        "material_threshold_percent": 1.0,
        "material_improvement": bool(relative_improvement > 1.0),
        "repeat_telemetry": repeated,
        "test_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=[*RUNNABLE, "all"], default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=HERE / "raw" / "convergence_results.json")
    args = parser.parse_args()
    previous = json.loads(args.output.read_text()) if args.output.exists() else {"results": []}
    completed = {row["task"]: row for row in previous.get("results", []) if row.get("status") == "complete"}
    requested = RUNNABLE if args.task == "all" else (args.task,)
    for task_name in requested:
        if task_name not in completed:
            completed[task_name] = run_task(task_name, args.device)
            atomic_json(
                {
                    "protocol": "frozen representative 300-vs-600 convergence check",
                    "results": [*completed.values()],
                    "unavailable": FROZEN_UNAVAILABLE,
                },
                args.output,
            )
        else:
            print(f"resume convergence {task_name}", flush=True)
    atomic_json(
        {
            "protocol": "frozen representative 300-vs-600 convergence check",
            "results": [completed[key] for key in sorted(completed)],
            "unavailable": FROZEN_UNAVAILABLE,
        },
        args.output,
    )


if __name__ == "__main__":
    main()
