#!/usr/bin/env python3
"""Prospectively declared MPE ablations on the frozen representative panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler, Levenshtein

from control_suite import evaluate_tables
from mpe import (
    bandwidth_grid,
    corrupt_state_association,
    farthest_point_landmarks,
    kernel_affinity,
    partial_permutation,
    state_weight_table,
)
from representations import load_task, split_row_indices, split_state_indices


HERE = Path(__file__).resolve().parent
REPRESENTATIVE_TASKS = [
    "acs_occupation",
    "tlc_pickup_zone",
    "citibike_start_station",
    "medical_charges",
]
PARTIAL_CORRUPTION_TASKS = [
    "acs_occupation",
    "tlc_pickup_zone",
    "citibike_start_station",
    "airline_origin_airport",
    "medical_charges",
]
SETTINGS = ("isolated_field", "full_table")


def stable_seed(*items: object) -> int:
    digest = hashlib.sha256("|".join(map(str, (20260829, *items))).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def atomic_json(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def kmedoids_landmarks(task, training: np.ndarray, count: int) -> np.ndarray:
    """Deterministic alternate minimization for the training-state k-medoids objective."""
    count = min(count, len(training))
    medoids = farthest_point_landmarks(task.distance, training, count, state_ids=task.state_ids)
    training = np.asarray(training, dtype=np.int64)
    for _ in range(20):
        assignment = np.argmin(task.distance[np.ix_(training, medoids)], axis=1)
        updated = []
        for cluster_index, old_medoid in enumerate(medoids):
            cluster = training[assignment == cluster_index]
            if len(cluster) == 0:
                updated.append(int(old_medoid))
                continue
            objective = task.distance[np.ix_(cluster, cluster)].sum(axis=1)
            best = np.flatnonzero(np.isclose(objective, objective.min(), rtol=0.0, atol=1e-12))
            chosen = min(best, key=lambda index: str(task.state_ids[int(cluster[index])]))
            updated.append(int(cluster[chosen]))
        updated_array = np.asarray(updated, dtype=np.int64)
        if np.array_equal(updated_array, medoids):
            break
        medoids = updated_array
    if len(np.unique(medoids)) != len(medoids):
        raise AssertionError("k-medoids produced duplicate landmarks")
    return medoids


def select_landmarks(task, split: int, method: str, count: int) -> np.ndarray:
    training = split_state_indices(task, split)["train"]
    count = min(count, len(training))
    if method == "farthest_point":
        return farthest_point_landmarks(task.distance, training, count, state_ids=task.state_ids)
    if method == "k_medoids":
        return kmedoids_landmarks(task, training, count)
    if method == "uniform_random":
        rng = np.random.default_rng(stable_seed(task.name, split, method, count))
        return np.sort(rng.choice(training, size=count, replace=False)).astype(np.int64)
    if method == "frequency":
        row_states = task.row_state_indices()
        counts = {state: int(np.sum(row_states == state)) for state in training}
        ordered = sorted(training.tolist(), key=lambda state: (-counts[state], str(task.state_ids[state])))
        return np.asarray(ordered[:count], dtype=np.int64)
    raise KeyError(method)


def string_distance(states: list[str], metric: str) -> np.ndarray:
    result = np.zeros((len(states), len(states)), dtype=np.float64)
    function = JaroWinkler.normalized_distance if metric == "jaro_winkler" else Levenshtein.normalized_distance
    for left in range(len(states)):
        for right in range(left + 1, len(states)):
            result[left, right] = result[right, left] = float(function(states[left], states[right]))
    return result


def build_tables(task, split: int, primary_bandwidth: float) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    tables: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    training = split_state_indices(task, split)["train"]

    # Landmark selection at the three frozen comparison budgets.
    for method in ("farthest_point", "k_medoids", "uniform_random", "frequency"):
        for count in (16, 32, 64):
            landmarks = select_landmarks(task, split, method, count)
            name = f"landmark_{method}_m{len(landmarks)}"
            tables[name] = state_weight_table(task.distance, landmarks, primary_bandwidth).astype(np.float32)
            metadata[name] = {"family": "landmark_selection", "method": method, "m": len(landmarks), "bandwidth": primary_bandwidth}

    # Wider budget curve, including all declared values that cardinality permits.
    for count in (8, 16, 32, 64, 128, 256):
        landmarks = select_landmarks(task, split, "farthest_point", count)
        name = f"budget_farthest_point_m{len(landmarks)}"
        tables[name] = state_weight_table(task.distance, landmarks, primary_bandwidth).astype(np.float32)
        cover = float(np.max(np.min(task.distance[np.ix_(training, landmarks)], axis=1)))
        metadata[name] = {"family": "landmark_budget", "method": "farthest_point", "m": len(landmarks), "cover_radius": cover, "bandwidth": primary_bandwidth}

    landmarks = select_landmarks(task, split, "farthest_point", 32)
    for kernel in ("gaussian", "laplacian", "triangular", "inverse_distance"):
        bandwidth = primary_bandwidth
        bandwidth_rule = "primary_validation_selected"
        if kernel == "triangular":
            # A compact kernel has a legitimate zero partition outside its
            # support.  Use the smallest target-independent radius covering
            # every declared state so the requested ablation is well-defined.
            required = float(np.max(np.min(task.distance[:, landmarks], axis=1)))
            bandwidth = max(bandwidth, np.nextafter(required, np.inf))
            bandwidth_rule = "max(primary, all_state_cover_radius_nextafter)"
        name = f"kernel_{kernel}"
        tables[name] = state_weight_table(
            task.distance, landmarks, bandwidth, kernel=kernel, normalization="partition"
        ).astype(np.float32)
        metadata[name] = {"family": "kernel", "kernel": kernel, "bandwidth": bandwidth, "bandwidth_rule": bandwidth_rule}

    for normalization in ("partition", "unnormalized", "softmax_distance"):
        name = f"normalization_{normalization}"
        tables[name] = state_weight_table(
            task.distance, landmarks, primary_bandwidth,
            kernel="gaussian", normalization=normalization,
        ).astype(np.float32)
        metadata[name] = {"family": "normalization", "normalization": normalization, "bandwidth": primary_bandwidth}

    # With a linear head, learned token dimensions 16/32/64 are algebraically
    # unidentifiable after collapsing V and the head.  Preserve the exact alias
    # in raw output instead of fabricating different fixed token matrices.
    primary = state_weight_table(task.distance, landmarks, primary_bandwidth).astype(np.float32)
    for dimension in (16, 32, 64):
        name = f"token_dimension_D{dimension}_linear_equivalence"
        tables[name] = primary.copy()
        metadata[name] = {"family": "token_dimension", "D": dimension, "linear_head_equivalence": True}

    for index, bandwidth in enumerate(bandwidth_grid(task.distance, training)):
        name = f"bandwidth_candidate_{index}"
        tables[name] = state_weight_table(task.distance, landmarks, float(bandwidth)).astype(np.float32)
        metadata[name] = {"family": "bandwidth", "candidate": index, "bandwidth": float(bandwidth)}

    for scale in (0.5, 1.0, 2.0):
        name = f"metric_scale_{scale:g}"
        tables[name] = state_weight_table(task.distance * scale, landmarks, primary_bandwidth).astype(np.float32)
        metadata[name] = {"family": "metric_scale", "scale": scale, "bandwidth": primary_bandwidth}

    if task.name in PARTIAL_CORRUPTION_TASKS:
        for fraction in (0.10, 0.25, 0.50, 1.00):
            permutation = partial_permutation(
                len(task.states), fraction, stable_seed(task.name, split, "partial", fraction)
            )
            corrupted = corrupt_state_association(task.distance, permutation)
            corrupt_landmarks = farthest_point_landmarks(
                corrupted, training, 32, state_ids=task.state_ids
            )
            name = f"partial_corruption_{fraction:.2f}"
            tables[name] = state_weight_table(corrupted, corrupt_landmarks, primary_bandwidth).astype(np.float32)
            metadata[name] = {"family": "partial_corruption", "fraction": fraction, "bandwidth": primary_bandwidth}

    if task.manifest["source_unit"] == "STRING_BENCHMARK":
        for metric in ("jaro_winkler", "levenshtein"):
            distance = string_distance(task.state_ids, metric)
            alt_landmarks = farthest_point_landmarks(distance, training, 32, state_ids=task.state_ids)
            candidates = bandwidth_grid(distance, training)
            bandwidth = float(candidates[len(candidates) // 2])
            mpe_name = f"string_{metric}_mpe"
            sim_name = f"string_{metric}_similarity"
            tables[mpe_name] = state_weight_table(distance, alt_landmarks, bandwidth).astype(np.float32)
            tables[sim_name] = kernel_affinity(distance[:, alt_landmarks] / bandwidth, "gaussian").astype(np.float32)
            metadata[mpe_name] = {"family": "secondary_string_metric", "metric": metric, "form": "mpe", "bandwidth_rule": "median_frozen_candidate", "bandwidth": bandwidth}
            metadata[sim_name] = {"family": "secondary_string_metric", "metric": metric, "form": "similarity", "bandwidth_rule": "median_frozen_candidate", "bandwidth": bandwidth}

    return tables, metadata


def run_cell(task_name: str, split: int, setting: str, output: Path) -> None:
    cell = f"{task_name}__split{split}__{setting}"
    path = output / f"{cell}.json"
    state_path = output / f"{cell}__state_metrics.parquet"
    if path.exists() and state_path.exists() and json.loads(path.read_text()).get("status") == "complete":
        print(f"resume ablation {cell}", flush=True)
        return
    ridge_path = HERE / "raw" / "ridge_cells" / f"{cell}.json"
    if not ridge_path.exists():
        raise FileNotFoundError(f"primary ridge cell must finish first: {ridge_path}")
    primary = json.loads(ridge_path.read_text())
    if primary.get("status") != "complete":
        raise RuntimeError(f"primary ridge cell incomplete: {ridge_path}")
    task = load_task(task_name)
    row_parts = split_row_indices(task, split)
    bandwidth = float(primary["selected_bandwidth"])
    tables, metadata = build_tables(task, split, bandwidth)
    results, states = evaluate_tables(task, row_parts, setting, tables)
    states.assign(task=task_name, split=split, setting=setting).to_parquet(
        state_path, index=False, compression="zstd"
    )
    payload = {
        "status": "complete", "task": task_name, "source_unit": task.manifest["source_unit"],
        "split": split, "setting": setting, "primary_bandwidth": bandwidth,
        "ablation_metadata": metadata, "results": results,
        "test_evaluations_per_representation": 1,
    }
    atomic_json(payload, path)
    print(f"complete ablation {cell}", flush=True)


def consolidate(output: Path) -> None:
    rows = []
    states = []
    for path in sorted(output.glob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") != "complete":
            continue
        metadata = payload["ablation_metadata"]
        for result in payload["results"]:
            name = result["representation"]
            rows.append(
                {
                    "task": payload["task"], "source_unit": payload["source_unit"],
                    "split": payload["split"], "setting": payload["setting"],
                    **metadata[name],
                    **{key: value for key, value in result.items() if key != "validation_trials"},
                }
            )
        state_path = path.with_name(path.stem + "__state_metrics.parquet")
        if state_path.exists():
            states.append(pd.read_parquet(state_path))
    if rows:
        frame = pd.DataFrame(rows)
        frame.to_parquet(HERE / "raw" / "ablation_results.parquet", index=False, compression="zstd")
        frame.to_csv(HERE / "raw" / "ablation_results.csv", index=False)
    if states:
        pd.concat(states, ignore_index=True).to_parquet(
            HERE / "raw" / "ablation_state_results.parquet", index=False, compression="zstd"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=REPRESENTATIVE_TASKS + ["airline_origin_airport", "all"], default="all")
    parser.add_argument("--split", type=int, choices=range(5))
    parser.add_argument("--setting", choices=[*SETTINGS, "all"], default="all")
    parser.add_argument("--consolidate-only", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "raw" / "ablation_cells")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.consolidate_only:
        tasks = REPRESENTATIVE_TASKS if args.task == "all" else [args.task]
        splits = range(5) if args.split is None else [args.split]
        settings = SETTINGS if args.setting == "all" else [args.setting]
        for task in tasks:
            for split in splits:
                for setting in settings:
                    run_cell(task, split, setting, args.output)
    consolidate(args.output)


if __name__ == "__main__":
    main()
