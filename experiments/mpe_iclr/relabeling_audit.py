#!/usr/bin/env python3
"""Real-field storage-code relabeling audit with transported metric metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from control_suite import evaluate_tables
from mpe import codebook_permutation, farthest_point_landmarks, kernel_affinity, state_weight_table
from representations import (
    candidate_bandwidths,
    categorical_unknown_table,
    load_task,
    piecewise_linear_table,
    split_row_indices,
    split_state_indices,
)
from ridge_benchmark import DEFAULT_TASKS


HERE = Path(__file__).resolve().parent


def stable_seed(task: str, relabeling: int) -> int:
    digest = hashlib.sha256(f"20260829|real-codebook|{task}|{relabeling}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def atomic_json(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def code_ple_tables(task, training: np.ndarray, code_for_semantic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    row_semantic = task.row_state_indices()
    train_rows = np.isin(row_semantic, training)
    train_codes = code_for_semantic[row_semantic[train_rows]].astype(np.float64)
    quantile = np.quantile(train_codes, np.linspace(0.0, 1.0, 33))
    uniform = np.linspace(train_codes.min(), train_codes.max(), 33)
    semantic_codes = code_for_semantic.astype(np.float64)
    return piecewise_linear_table(semantic_codes, quantile, 32), piecewise_linear_table(semantic_codes, uniform, 32)


def code_rbf_table(training: np.ndarray, code_for_semantic: np.ndarray, semantic_landmarks: np.ndarray) -> np.ndarray:
    codes = code_for_semantic.astype(np.float64)
    landmark_codes = codes[semantic_landmarks]
    train_codes = np.sort(codes[training])
    gaps = np.diff(train_codes)
    positive = gaps[gaps > 0]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0
    return np.exp(-0.5 * ((codes[:, None] - landmark_codes[None, :]) / bandwidth) ** 2).astype(np.float32)


def run_task(task_name: str, output: Path) -> None:
    path = output / f"{task_name}.json"
    state_path = output / f"{task_name}__state_metrics.parquet"
    if path.exists() and state_path.exists() and json.loads(path.read_text()).get("status") == "complete":
        print(f"resume relabeling {task_name}", flush=True)
        return
    task = load_task(task_name)
    split = 0
    row_parts = split_row_indices(task, split)
    training = split_state_indices(task, split)["train"]
    ridge_path = HERE / "raw" / "ridge_cells" / f"{task_name}__split0__isolated_field.json"
    if ridge_path.exists():
        bandwidth = float(json.loads(ridge_path.read_text())["selected_bandwidth"])
    else:
        candidates = candidate_bandwidths(task, split)
        bandwidth = float(candidates[len(candidates) // 2])
    landmarks = farthest_point_landmarks(task.distance, training, 32, state_ids=task.state_ids)
    reference_mpe = state_weight_table(task.distance, landmarks, bandwidth).astype(np.float32)
    reference_similarity = kernel_affinity(task.distance[:, landmarks] / bandwidth, "gaussian").astype(np.float32)
    reference_lookup = categorical_unknown_table(len(task.states), training)
    identity = np.arange(len(task.states), dtype=np.int64)
    reference_q, reference_uniform = code_ple_tables(task, training, identity)
    reference_code_rbf = code_rbf_table(training, identity, landmarks)

    tables: dict[str, np.ndarray] = {
        "reference_mpe": reference_mpe,
        "reference_similarity": reference_similarity,
        "reference_lookup": reference_lookup,
        "reference_q_ple": reference_q,
        "reference_uniform_ple": reference_uniform,
        "reference_code_rbf": reference_code_rbf,
    }
    feature_audit: list[dict[str, Any]] = []
    for index in range(8):
        code_for_semantic = codebook_permutation(len(task.states), stable_seed(task_name, index))
        semantic_for_code = np.argsort(code_for_semantic)
        transported_distance = task.distance[np.ix_(semantic_for_code, semantic_for_code)]
        transported_landmarks = code_for_semantic[landmarks]
        transported_training = code_for_semantic[training]
        mpe_by_code = state_weight_table(transported_distance, transported_landmarks, bandwidth).astype(np.float32)
        sim_by_code = kernel_affinity(
            transported_distance[:, transported_landmarks] / bandwidth, "gaussian"
        ).astype(np.float32)
        lookup_by_code = categorical_unknown_table(len(task.states), transported_training)
        semantic_mpe = mpe_by_code[code_for_semantic]
        semantic_similarity = sim_by_code[code_for_semantic]
        semantic_lookup = lookup_by_code[code_for_semantic]
        q_ple, uniform_ple = code_ple_tables(task, training, code_for_semantic)
        code_rbf = code_rbf_table(training, code_for_semantic, landmarks)
        tables[f"relabel_{index}_mpe"] = semantic_mpe
        tables[f"relabel_{index}_similarity"] = semantic_similarity
        tables[f"relabel_{index}_lookup"] = semantic_lookup
        tables[f"relabel_{index}_q_ple"] = q_ple
        tables[f"relabel_{index}_uniform_ple"] = uniform_ple
        tables[f"relabel_{index}_code_rbf"] = code_rbf
        feature_audit.append(
            {
                "relabeling": index,
                "bijection_valid": bool(np.array_equal(np.sort(code_for_semantic), identity)),
                "mpe_max_abs_difference": float(np.max(np.abs(semantic_mpe - reference_mpe))),
                "similarity_max_abs_difference": float(np.max(np.abs(semantic_similarity - reference_similarity))),
                "lookup_max_abs_difference": float(np.max(np.abs(semantic_lookup - reference_lookup))),
                "q_ple_max_abs_difference": float(np.max(np.abs(q_ple - reference_q))),
                "uniform_ple_max_abs_difference": float(np.max(np.abs(uniform_ple - reference_uniform))),
                "code_rbf_max_abs_difference": float(np.max(np.abs(code_rbf - reference_code_rbf))),
            }
        )
    if max(row["mpe_max_abs_difference"] for row in feature_audit) != 0.0:
        raise AssertionError("transported MPE was not exactly codebook invariant")
    results, states = evaluate_tables(task, row_parts, "isolated_field", tables)
    states.assign(task=task_name, split=0, setting="isolated_field").to_parquet(
        state_path, index=False, compression="zstd"
    )
    payload = {
        "status": "complete", "task": task_name, "source_unit": task.manifest["source_unit"],
        "split": 0, "setting": "isolated_field", "relabelings": 8,
        "bandwidth": bandwidth, "landmark_state_ids": [task.state_ids[value] for value in landmarks],
        "feature_audit": feature_audit, "results": results,
        "test_evaluations_per_representation": 1,
    }
    atomic_json(payload, path)
    print(f"complete relabeling {task_name}", flush=True)


def consolidate(output: Path) -> None:
    feature_rows = []
    result_rows = []
    for path in sorted(output.glob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") != "complete":
            continue
        feature_rows.extend({"task": payload["task"], "source_unit": payload["source_unit"], **row} for row in payload["feature_audit"])
        result_rows.extend(
            {
                "task": payload["task"], "source_unit": payload["source_unit"],
                **{key: value for key, value in row.items() if key != "validation_trials"},
            }
            for row in payload["results"]
        )
    if feature_rows:
        pd.DataFrame(feature_rows).to_parquet(
            HERE / "raw" / "relabeling_feature_audit.parquet", index=False, compression="zstd"
        )
        pd.DataFrame(feature_rows).to_csv(HERE / "raw" / "relabeling_feature_audit.csv", index=False)
    if result_rows:
        pd.DataFrame(result_rows).to_parquet(
            HERE / "raw" / "relabeling_results.parquet", index=False, compression="zstd"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=DEFAULT_TASKS + ["all"], default="all")
    parser.add_argument("--consolidate-only", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "raw" / "relabeling_cells")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.consolidate_only:
        tasks = DEFAULT_TASKS if args.task == "all" else [args.task]
        for task in tasks:
            run_task(task, args.output)
    consolidate(args.output)


if __name__ == "__main__":
    main()
