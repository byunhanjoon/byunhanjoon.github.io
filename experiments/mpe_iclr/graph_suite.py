#!/usr/bin/env python3
"""Training-row route-network metric comparisons for geography tasks."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components, shortest_path

from control_suite import evaluate_tables, select_bandwidth_custom
from representations import load_task, representation_tables, split_row_indices
from ridge_benchmark import ordinary_design


HERE = Path(__file__).resolve().parent
TASKS = ["citibike_start_station", "airline_origin_airport", "airline_destination_airport"]


def atomic_json(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def route_adjacency(task, split: int) -> np.ndarray:
    other = "end_station_id" if task.name == "citibike_start_station" else "other_endpoint_airport"
    train_rows = split_row_indices(task, split)["train"]
    lookup = task.state_to_index
    adjacency = np.zeros((len(task.states), len(task.states)), dtype=np.float32)
    frame = task.rows.iloc[train_rows]
    for left, right in zip(frame["field_state"].astype(str), frame[other].astype(str)):
        if right not in lookup:
            continue
        i, j = lookup[left], lookup[right]
        if i != j:
            adjacency[i, j] = adjacency[j, i] = 1.0
    return adjacency


def run_cell(task_name: str, split: int, setting: str, output: Path) -> None:
    cell = f"{task_name}__split{split}__{setting}"
    path = output / f"{cell}.json"
    state_path = output / f"{cell}__state_metrics.parquet"
    if path.exists() and json.loads(path.read_text()).get("status") in {"complete", "NOT RUN"}:
        print(f"resume graph {cell}", flush=True)
        return
    original = load_task(task_name)
    adjacency = route_adjacency(original, split)
    components, labels = connected_components(sparse.csr_matrix(adjacency), directed=False)
    sizes = np.bincount(labels)
    audit = {
        "nodes": len(original.states), "edges": int(adjacency.sum() // 2),
        "components": int(components), "component_sizes": sorted(map(int, sizes), reverse=True),
        "construction": "unweighted endpoint graph from training rows only; no target labels",
    }
    if components != 1:
        atomic_json(
            {
                "status": "NOT RUN", "task": task_name, "source_unit": original.manifest["source_unit"],
                "split": split, "setting": setting,
                "reason": "training-row route graph is disconnected, so all-state shortest path is not a finite metric",
                "graph_audit": audit,
            },
            path,
        )
        print(f"NOT RUN graph {cell}: {components} components", flush=True)
        return
    distance = np.asarray(shortest_path(sparse.csr_matrix(adjacency), directed=False, unweighted=True), dtype=np.float64)
    if not np.isfinite(distance).all():
        raise AssertionError("connected graph produced non-finite shortest paths")
    task = replace(original, distance=distance, arrays={**original.arrays, "adjacency": adjacency})
    row_parts = split_row_indices(task, split)
    raw = pd.to_numeric(task.rows["target"], errors="raise").to_numpy(np.float64)
    target = (raw - raw[row_parts["train"]].mean()) / (raw[row_parts["train"]].std() or 1.0)
    ordinary = ordinary_design(task, row_parts["train"]) if setting == "full_table" else None
    # select_bandwidth_custom expects its custom split at key zero.
    if split != 0:
        task = replace(task, splits={"0": task.splits[str(split)]})
    bandwidth, trials = select_bandwidth_custom(
        task, row_parts, setting, target, task.row_state_indices(), ordinary
    )
    route_tables, metadata = representation_tables(task, 0, bandwidth)
    primary_ridge = HERE / "raw" / "ridge_cells" / f"{cell}.json"
    if not primary_ridge.exists():
        print(f"defer graph {cell}: primary ridge bandwidth not ready", flush=True)
        return
    primary_bandwidth = float(json.loads(primary_ridge.read_text())["selected_bandwidth"])
    primary_tables, _ = representation_tables(original, split, primary_bandwidth)
    tables = {
        "route_mpe": route_tables["mpe"],
        "route_similarity": route_tables["similarity_unnormalized"],
        "route_nystrom": route_tables["nystrom"],
        "route_graph_laplacian": route_tables["graph_laplacian"],
        "route_node2vec": route_tables["node2vec"],
        "geodesic_mpe": primary_tables["mpe"],
        "raw_coordinates": primary_tables["raw_coordinates"],
        "coordinate_fourier": primary_tables["coordinate_fourier"],
    }
    results, states = evaluate_tables(task, row_parts, setting, tables)
    states.assign(task=task_name, split=split, setting=setting).to_parquet(
        state_path, index=False, compression="zstd"
    )
    payload = {
        "status": "complete", "task": task_name, "source_unit": original.manifest["source_unit"],
        "split": split, "setting": setting, "graph_audit": audit,
        "route_bandwidth": bandwidth, "route_bandwidth_trials": trials,
        "representation_metadata": metadata, "results": results,
        "test_evaluations_per_representation": 1,
    }
    atomic_json(payload, path)
    print(f"complete graph {cell}", flush=True)


def consolidate(output: Path) -> None:
    rows, audits = [], []
    states = []
    for path in sorted(output.glob("*.json")):
        payload = json.loads(path.read_text())
        audits.append(
            {
                "task": payload["task"], "source_unit": payload["source_unit"],
                "split": payload["split"], "setting": payload["setting"],
                "status": payload["status"], "reason": payload.get("reason"), **payload["graph_audit"],
            }
        )
        if payload["status"] == "complete":
            rows.extend(
                {
                    "task": payload["task"], "source_unit": payload["source_unit"],
                    "split": payload["split"], "setting": payload["setting"],
                    **{key: value for key, value in result.items() if key != "validation_trials"},
                }
                for result in payload["results"]
            )
            state_path = path.with_name(path.stem + "__state_metrics.parquet")
            if state_path.exists():
                states.append(pd.read_parquet(state_path))
    if audits:
        pd.DataFrame(audits).to_json(HERE / "raw" / "graph_metric_audit.json", orient="records", indent=2)
    if rows:
        frame = pd.DataFrame(rows)
        frame.to_parquet(HERE / "raw" / "graph_results.parquet", index=False, compression="zstd")
        frame.to_csv(HERE / "raw" / "graph_results.csv", index=False)
    if states:
        pd.concat(states, ignore_index=True).to_parquet(
            HERE / "raw" / "graph_state_results.parquet", index=False, compression="zstd"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS + ["all"], default="all")
    parser.add_argument("--split", type=int, choices=range(5))
    parser.add_argument("--setting", choices=["isolated_field", "full_table", "all"], default="all")
    parser.add_argument("--consolidate-only", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "raw" / "graph_cells")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.consolidate_only:
        tasks = TASKS if args.task == "all" else [args.task]
        splits = range(5) if args.split is None else [args.split]
        settings = ("isolated_field", "full_table") if args.setting == "all" else (args.setting,)
        for task in tasks:
            for split in splits:
                for setting in settings:
                    run_cell(task, split, setting, args.output)
    consolidate(args.output)


if __name__ == "__main__":
    main()
