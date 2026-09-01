#!/usr/bin/env python3
"""Label-free hard subtree and spatial-block cold-state evaluations."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from control_suite import evaluate_tables, select_bandwidth_custom, task_with_state_parts
from mpe import make_state_partition
from representations import load_task, representation_tables
from ridge_benchmark import DEFAULT_TASKS, ordinary_design


HERE = Path(__file__).resolve().parent
HIERARCHY = {"acs_occupation", "acs_industry"}
GEOGRAPHY = {
    "tlc_pickup_zone", "tlc_dropoff_zone", "citibike_start_station",
    "airline_origin_airport", "airline_destination_airport",
}
SEEDS = [20261001, 20261002, 20261003, 20261004, 20261005]


def atomic_json(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def hierarchy_parts(task, split: int) -> tuple[dict[str, list[str]], dict[str, Any]]:
    branches = {
        state: str(json.loads(path)[1])
        for state, path in zip(task.state_ids, task.states["path_json"].astype(str))
    }
    branch_parts = make_state_partition(sorted(set(branches.values())), SEEDS[split])
    state_parts = {
        part: [state for state in task.state_ids if branches[state] in set(map(str, values))]
        for part, values in branch_parts.items()
    }
    return state_parts, {
        "kind": "complete_subtree_below_root",
        "branch_column": "path_json[1]",
        "branch_counts": {part: len(values) for part, values in branch_parts.items()},
    }


def spatial_parts(task, split: int) -> tuple[dict[str, list[str]], dict[str, Any]]:
    coordinates = np.asarray(task.arrays["coordinates"], dtype=np.float64)
    clusters = min(15, len(task.states))
    labels = KMeans(n_clusters=clusters, random_state=20261601, n_init=20).fit_predict(coordinates)
    cluster_names = [f"block-{index:02d}" for index in range(clusters)]
    cluster_parts = make_state_partition(cluster_names, SEEDS[split])
    state_parts = {
        part: [
            task.state_ids[index]
            for index, label in enumerate(labels)
            if cluster_names[int(label)] in set(map(str, values))
        ]
        for part, values in cluster_parts.items()
    }
    return state_parts, {
        "kind": "coordinate_kmeans_blocks",
        "clusters": clusters,
        "kmeans_random_state": 20261601,
        "block_counts": {part: len(values) for part, values in cluster_parts.items()},
    }


def run_cell(task_name: str, split: int, setting: str, output: Path) -> None:
    cell = f"{task_name}__split{split}__{setting}"
    path = output / f"{cell}.json"
    state_path = output / f"{cell}__state_metrics.parquet"
    if path.exists() and state_path.exists() and json.loads(path.read_text()).get("status") == "complete":
        print(f"resume hard {cell}", flush=True)
        return
    original = load_task(task_name)
    state_parts, split_audit = (
        hierarchy_parts(original, split) if task_name in HIERARCHY else spatial_parts(original, split)
    )
    task = task_with_state_parts(original, state_parts)
    state_values = task.rows["field_state"].astype(str).to_numpy()
    row_parts = {
        part: np.flatnonzero(np.isin(state_values, states))
        for part, states in state_parts.items()
    }
    if any(len(rows) == 0 for rows in row_parts.values()):
        raise AssertionError("hard split contains an empty row partition")
    raw = pd.to_numeric(task.rows["target"], errors="raise").to_numpy(np.float64)
    target = (raw - raw[row_parts["train"]].mean()) / (raw[row_parts["train"]].std() or 1.0)
    ordinary = ordinary_design(task, row_parts["train"]) if setting == "full_table" else None
    bandwidth, bandwidth_trials = select_bandwidth_custom(
        task, row_parts, setting, target, task.row_state_indices(), ordinary
    )
    all_tables, metadata = representation_tables(task, 0, bandwidth)
    keep = {
        "mpe", "similarity_unnormalized", "nystrom", "unknown_embedding",
        "q_ple", "uniform_ple", "ancestor_multihot", "path_to_root", "wu_palmer",
        "lch_path", "laplacian", "node2vec", "raw_coordinates", "raw_latlon",
        "coordinate_fourier", "spatial_rbf", "graph_laplacian",
    }
    tables = {name: table for name, table in all_tables.items() if name in keep}
    results, states = evaluate_tables(task, row_parts, setting, tables)
    states.assign(task=task_name, split=split, setting=setting).to_parquet(
        state_path, index=False, compression="zstd"
    )
    payload = {
        "status": "complete", "task": task_name, "source_unit": task.manifest["source_unit"],
        "split": split, "seed": SEEDS[split], "setting": setting,
        "hard_split": split_audit,
        "states": {part: len(values) for part, values in state_parts.items()},
        "rows": {part: len(values) for part, values in row_parts.items()},
        "selected_bandwidth": bandwidth, "bandwidth_trials": bandwidth_trials,
        "representation_metadata": metadata, "results": results,
        "test_evaluations_per_representation": 1,
    }
    atomic_json(payload, path)
    print(f"complete hard {cell}", flush=True)


def consolidate(output: Path) -> None:
    rows, states = [], []
    for path in sorted(output.glob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") != "complete":
            continue
        rows.extend(
            {
                "task": payload["task"], "source_unit": payload["source_unit"],
                "split": payload["split"], "setting": payload["setting"],
                "hard_split_kind": payload["hard_split"]["kind"],
                **{key: value for key, value in result.items() if key != "validation_trials"},
            }
            for result in payload["results"]
        )
        state_path = path.with_name(path.stem + "__state_metrics.parquet")
        if state_path.exists():
            states.append(pd.read_parquet(state_path))
    if rows:
        frame = pd.DataFrame(rows)
        frame.to_parquet(HERE / "raw" / "hard_split_results.parquet", index=False, compression="zstd")
        frame.to_csv(HERE / "raw" / "hard_split_results.csv", index=False)
    if states:
        pd.concat(states, ignore_index=True).to_parquet(
            HERE / "raw" / "hard_split_state_results.parquet", index=False, compression="zstd"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    choices = sorted(HIERARCHY | GEOGRAPHY)
    parser.add_argument("--task", choices=choices + ["all"], default="all")
    parser.add_argument("--split", type=int, choices=range(5))
    parser.add_argument("--setting", choices=["isolated_field", "full_table", "all"], default="all")
    parser.add_argument("--consolidate-only", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "raw" / "hard_split_cells")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.consolidate_only:
        tasks = choices if args.task == "all" else [args.task]
        splits = range(5) if args.split is None else [args.split]
        settings = ("isolated_field", "full_table") if args.setting == "all" else (args.setting,)
        for task in tasks:
            for split in splits:
                for setting in settings:
                    run_cell(task, split, setting, args.output)
    consolidate(args.output)


if __name__ == "__main__":
    main()
