#!/usr/bin/env python3
"""Five-fold, training-state-only representation router."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
MFT_ROOT = HERE.parent / "metric-field-transport"
if str(MFT_ROOT) not in sys.path:
    sys.path.insert(0, str(MFT_ROOT))

from successor_experiments import (  # noqa: E402
    ALL_TASKS,
    RIDGE_ALPHAS,
    balanced_mse,
    compose_design,
    load_task,
    ordinary_design_subset,
    representation_tables,
    ridge_prediction,
    rows_for_states,
    sealed_raw_target,
    split_row_indices,
    split_state_indices,
    standardized_target,
)


PROTOCOL_PATH = HERE / "EXPLORATORY_PROTOCOL.md"
REPRESENTATIONS = ["weights_m32", "distance_m32"]


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def stable_hash(*parts: object) -> int:
    payload = " | ".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def state_folds(task: Any, split_index: int) -> list[np.ndarray]:
    training_states = split_state_indices(task, split_index)["train"]
    ordered = sorted(
        training_states.tolist(),
        key=lambda state: (
            stable_hash("mtr-fivefold", task.name, split_index, task.state_ids[state]),
            task.state_ids[state],
        ),
    )
    folds = [np.asarray(sorted(ordered[offset::5]), dtype=np.int64) for offset in range(5)]
    if any(len(fold) == 0 for fold in folds):
        raise ValueError(f"{task.name}: fewer than five outer-training states")
    return folds


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def run_cell(task_name: str, split_index: int, output_root: Path) -> dict[str, Any]:
    cell_id = f"{task_name}__split{split_index}"
    path = output_root / "router_cells" / f"{cell_id}.json"
    protocol_hash = sha256_path(PROTOCOL_PATH)
    if path.exists():
        payload = json.loads(path.read_text())
        if payload.get("status") == "complete" and payload.get("protocol_sha256") == protocol_hash:
            print(f"resume {cell_id}", flush=True)
            return payload

    started = time.perf_counter()
    task = load_task(task_name)
    outer_parts = split_state_indices(task, split_index)
    row_parts = split_row_indices(task, split_index)
    raw_target = sealed_raw_target(task, row_parts["test"])
    folds = state_folds(task, split_index)
    outer_train_set = set(outer_parts["train"].tolist())
    if set(np.concatenate(folds).tolist()) != outer_train_set:
        raise AssertionError("router folds do not partition outer training states")

    fold_results = []
    scores: dict[str, dict[float, list[float]]] = {
        representation: {float(alpha): [] for alpha in RIDGE_ALPHAS}
        for representation in REPRESENTATIONS
    }
    for fold_index, validation_states in enumerate(folds):
        training_states = np.asarray(
            sorted(outer_train_set - set(validation_states.tolist())), dtype=np.int64
        )
        training_rows = rows_for_states(task, training_states)
        validation_rows = rows_for_states(task, validation_states)
        if np.intersect1d(np.concatenate([training_rows, validation_rows]), row_parts["test"]).size:
            raise AssertionError("sealed test row entered router fold")
        output_rows = np.concatenate([training_rows, validation_rows])
        training_local = np.arange(len(training_rows), dtype=np.int64)
        validation_local = np.arange(len(training_rows), len(output_rows), dtype=np.int64)
        target_global, _, _ = standardized_target(raw_target, training_rows)
        target = target_global[output_rows]
        labels = task.rows["field_state"].astype(str).to_numpy()[output_rows]
        tables, metadata = representation_tables(task, training_states)
        ordinary = ordinary_design_subset(task, training_rows, output_rows)
        representation_rows = []
        for representation in REPRESENTATIONS:
            design = compose_design(task, tables[representation], output_rows, ordinary)
            trials = []
            for alpha in RIDGE_ALPHAS:
                prediction = ridge_prediction(
                    design,
                    target,
                    labels,
                    training_local,
                    validation_local,
                    alpha,
                )
                score = balanced_mse(
                    target[validation_local], prediction, labels[validation_local]
                )
                scores[representation][float(alpha)].append(score)
                trials.append(
                    {"alpha": float(alpha), "state_balanced_standardized_mse": score}
                )
            representation_rows.append({"representation": representation, "trials": trials})
        fold_results.append(
            {
                "fold": fold_index,
                "training_states": int(len(training_states)),
                "validation_states": int(len(validation_states)),
                "training_rows": int(len(training_rows)),
                "validation_rows": int(len(validation_rows)),
                "validation_state_ids": [task.state_ids[state] for state in validation_states],
                "representation_metadata": metadata,
                "representations": representation_rows,
            }
        )

    aggregate = []
    selected_scores: dict[str, float] = {}
    selected_alphas: dict[str, float] = {}
    for representation in REPRESENTATIONS:
        trials = [
            {
                "alpha": float(alpha),
                "mean_fivefold_state_balanced_standardized_mse": float(
                    np.mean(scores[representation][float(alpha)])
                ),
                "fold_scores": scores[representation][float(alpha)],
            }
            for alpha in RIDGE_ALPHAS
        ]
        winner = min(
            trials,
            key=lambda row: (row["mean_fivefold_state_balanced_standardized_mse"], row["alpha"]),
        )
        selected_scores[representation] = winner[
            "mean_fivefold_state_balanced_standardized_mse"
        ]
        selected_alphas[representation] = winner["alpha"]
        aggregate.append(
            {
                "representation": representation,
                "selected_alpha": winner["alpha"],
                "selected_mean_fivefold_mse": winner[
                    "mean_fivefold_state_balanced_standardized_mse"
                ],
                "trials": trials,
            }
        )
    decision = (
        "distance_m32"
        if selected_scores["distance_m32"] < selected_scores["weights_m32"]
        else "weights_m32"
    )
    payload = {
        "status": "complete",
        "stage": "fivefold_metric_trust_router",
        "cell_id": cell_id,
        "task": task_name,
        "source_unit": task.manifest["source_unit"],
        "split": split_index,
        "setting": "full_table",
        "protocol_sha256": protocol_hash,
        "scientific_status": "post_outcome_exploratory_only",
        "sealed_original_test": True,
        "test_target_evaluations": 0,
        "outer_training_states": int(len(outer_parts["train"])),
        "sealed_outer_validation_states": int(len(outer_parts["validation"])),
        "sealed_original_test_states": int(len(outer_parts["test"])),
        "ridge_alphas": RIDGE_ALPHAS,
        "fold_results": fold_results,
        "aggregate": aggregate,
        "selected_scores": selected_scores,
        "selected_alphas": selected_alphas,
        "decision": decision,
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_json(payload, path)
    relative = (
        selected_scores["weights_m32"] - selected_scores["distance_m32"]
    ) / selected_scores["weights_m32"]
    print(f"{cell_id} decision={decision} cv_raw_gain={relative:+.3%}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all")
    parser.add_argument("--split", default="all")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    observed = sha256_path(PROTOCOL_PATH)
    expected = (HERE / "PROTOCOL_SHA256.txt").read_text().split()[0]
    if observed != expected:
        raise RuntimeError(f"protocol hash mismatch: {observed} != {expected}")
    tasks = ALL_TASKS if args.task == "all" else [args.task]
    splits = range(5) if args.split == "all" else [int(args.split)]
    for task_name in tasks:
        if task_name not in ALL_TASKS:
            raise ValueError(task_name)
        for split_index in splits:
            run_cell(task_name, split_index, args.output)


if __name__ == "__main__":
    main()
