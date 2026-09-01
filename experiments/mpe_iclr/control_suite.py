#!/usr/bin/env python3
"""Frozen boundary controls for the final MPE program.

This runner covers three controls that are deliberately separate from the
state-disjoint main matrix:

* Citi Bike's natural new-station time split;
* seen-state, row-disjoint controls for every runnable task; and
* equality-metric nominal fields from ACS, TLC, and BTS.

Every model choice is made on validation rows.  A completed cell is atomic and
resume-safe, and test rows are evaluated once per selected configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse

from mpe import (
    bandwidth_grid,
    equality_distance,
    farthest_point_landmarks,
    make_state_partition,
    state_balanced_mean,
    state_weight_table,
)
from representations import (
    TaskData,
    candidate_bandwidths,
    corrupted_mpe_table,
    load_task,
    representation_tables,
)
from ridge_benchmark import (
    ALPHAS,
    DEFAULT_TASKS,
    fit_ridge,
    metrics,
    ordinary_design,
    row_representation,
)


HERE = Path(__file__).resolve().parent
SEEDS = [20261001, 20261002, 20261003, 20261004, 20261005]
SETTINGS = ("isolated_field", "full_table")


def atomic_json(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def stable_key(seed: int, value: object) -> bytes:
    return hashlib.sha256(f"20260829|{seed}|{value}".encode()).digest()


def seen_row_partition(task: TaskData, seed: int) -> dict[str, np.ndarray]:
    """Target-independent 60/20/20 rows within every eligible state."""
    state_values = task.rows["field_state"].astype(str).to_numpy()
    row_ids = task.rows["row_id"].astype(str).to_numpy()
    parts: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    for state in task.state_ids:
        indices = np.flatnonzero(state_values == state)
        ordered = np.asarray(
            sorted(indices.tolist(), key=lambda index: stable_key(seed, row_ids[index])),
            dtype=np.int64,
        )
        n_train = max(1, int(np.floor(0.6 * len(ordered))))
        n_validation = max(1, int(np.floor(0.2 * len(ordered))))
        if n_train + n_validation >= len(ordered):
            n_train = len(ordered) - 2
            n_validation = 1
        parts["train"].extend(ordered[:n_train].tolist())
        parts["validation"].extend(ordered[n_train : n_train + n_validation].tolist())
        parts["test"].extend(ordered[n_train + n_validation :].tolist())
    result = {name: np.asarray(sorted(values), dtype=np.int64) for name, values in parts.items()}
    row_sets = {name: set(values.tolist()) for name, values in result.items()}
    if row_sets["train"] & row_sets["validation"] or row_sets["train"] & row_sets["test"] or row_sets["validation"] & row_sets["test"]:
        raise AssertionError("seen-state row split overlaps")
    for name, indices in result.items():
        if set(state_values[indices]) != set(task.state_ids):
            raise AssertionError(f"seen-state {name} rows do not cover every state")
    return result


def task_with_state_parts(task: TaskData, parts: dict[str, Iterable[str]]) -> TaskData:
    payload = {
        "seed": "custom-control",
        **{name: [str(value) for value in values] for name, values in parts.items()},
    }
    return replace(task, splits={"0": payload})


def select_bandwidth_custom(
    task: TaskData,
    row_parts: dict[str, np.ndarray],
    setting: str,
    target: np.ndarray,
    row_state_indices: np.ndarray,
    ordinary: sparse.csr_matrix | None,
) -> tuple[float, list[dict[str, float]]]:
    trials: list[dict[str, float]] = []
    for bandwidth in candidate_bandwidths(task, 0):
        tables, _ = representation_tables(task, 0, float(bandwidth))
        representation = row_representation(task, tables["mpe"])
        design = representation if ordinary is None else sparse.hstack([ordinary, representation], format="csr")
        prediction, elapsed = fit_ridge(
            design, target, row_state_indices, row_parts["train"], row_parts["validation"], 1.0
        )
        score = state_balanced_mean(
            (prediction - target[row_parts["validation"]]) ** 2,
            task.rows["field_state"].astype(str).to_numpy()[row_parts["validation"]],
        )
        trials.append({"bandwidth": float(bandwidth), "validation_score": float(score), "seconds": elapsed})
    winner = min(trials, key=lambda row: (row["validation_score"], row["bandwidth"]))
    return float(winner["bandwidth"]), trials


def evaluate_tables(
    task: TaskData,
    row_parts: dict[str, np.ndarray],
    setting: str,
    tables: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    row_states_index = task.row_state_indices()
    row_states = task.rows["field_state"].astype(str).to_numpy()
    raw_target = pd.to_numeric(task.rows["target"], errors="raise").to_numpy(np.float64)
    target_mean = float(raw_target[row_parts["train"]].mean())
    target_scale = float(raw_target[row_parts["train"]].std()) or 1.0
    target = (raw_target - target_mean) / target_scale
    ordinary = ordinary_design(task, row_parts["train"]) if setting == "full_table" else None
    results: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    digest_cache: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for name, table in tables.items():
        digest = hashlib.sha256(np.ascontiguousarray(table).view(np.uint8)).hexdigest()
        if digest in digest_cache:
            summary, per_state = digest_cache[digest]
            alias_of = next(row["representation"] for row in results if row["feature_digest"] == digest)
            trials: list[dict[str, float]] = []
            selected_alpha = next(row["selected_alpha"] for row in results if row["feature_digest"] == digest)
        else:
            representation = row_representation(task, table)
            design = representation if ordinary is None else sparse.hstack([ordinary, representation], format="csr")
            trials = []
            for alpha in ALPHAS:
                prediction, elapsed = fit_ridge(
                    design, target, row_states_index, row_parts["train"], row_parts["validation"], alpha
                )
                score = state_balanced_mean(
                    (prediction - target[row_parts["validation"]]) ** 2,
                    row_states[row_parts["validation"]],
                )
                trials.append({"alpha": alpha, "score": float(score), "seconds": elapsed})
            winner = min(trials, key=lambda row: (row["score"], row["alpha"]))
            selected_alpha = float(winner["alpha"])
            prediction, _ = fit_ridge(
                design, target, row_states_index, row_parts["train"], row_parts["test"], selected_alpha
            )
            summary, per_state = metrics(
                target[row_parts["test"]], prediction, raw_target[row_parts["test"]],
                row_states[row_parts["test"]], target_scale,
            )
            digest_cache[digest] = (summary, per_state)
            alias_of = None
        results.append(
            {
                "representation": name,
                "feature_dimension": int(table.shape[1]),
                "feature_digest": digest,
                "alias_of": alias_of,
                "selected_alpha": selected_alpha,
                "validation_trials": trials,
                **summary,
            }
        )
        state_rows.extend({"representation": name, **row} for row in per_state)
    return results, pd.DataFrame(state_rows)


def run_seen_cell(task_name: str, split: int, setting: str, output: Path) -> None:
    cell = f"{task_name}__split{split}__{setting}"
    path = output / f"{cell}.json"
    state_path = output / f"{cell}__state_metrics.parquet"
    if path.exists() and state_path.exists() and json.loads(path.read_text()).get("status") == "complete":
        print(f"resume seen {cell}", flush=True)
        return
    original = load_task(task_name)
    all_states = original.state_ids
    task = task_with_state_parts(
        original, {"train": all_states, "validation": all_states, "test": all_states}
    )
    row_parts = seen_row_partition(task, SEEDS[split])
    raw_target = pd.to_numeric(task.rows["target"], errors="raise").to_numpy(np.float64)
    target = (raw_target - raw_target[row_parts["train"]].mean()) / (raw_target[row_parts["train"]].std() or 1.0)
    ordinary = ordinary_design(task, row_parts["train"]) if setting == "full_table" else None
    bandwidth, bandwidth_trials = select_bandwidth_custom(
        task, row_parts, setting, target, task.row_state_indices(), ordinary
    )
    all_tables, metadata = representation_tables(task, 0, bandwidth)
    keep = {
        "mpe", "similarity_unnormalized", "nystrom", "unknown_embedding",
        "q_ple", "uniform_ple", "ancestor_multihot", "path_to_root",
        "laplacian", "node2vec", "raw_coordinates", "raw_latlon",
        "coordinate_fourier", "spatial_rbf", "graph_laplacian",
        "character_3gram_hash",
    }
    tables = {name: table for name, table in all_tables.items() if name in keep}
    results, states = evaluate_tables(task, row_parts, setting, tables)
    states.assign(task=task_name, split=split, setting=setting, control="seen_state").to_parquet(
        state_path, index=False, compression="zstd"
    )
    payload = {
        "status": "complete", "control": "seen_state", "task": task_name,
        "source_unit": task.manifest["source_unit"], "split": split, "seed": SEEDS[split],
        "setting": setting, "rows": {name: len(value) for name, value in row_parts.items()},
        "states_in_every_part": len(task.state_ids), "selected_bandwidth": bandwidth,
        "bandwidth_trials": bandwidth_trials, "representation_metadata": metadata,
        "results": results, "test_evaluations_per_representation": 1,
    }
    atomic_json(payload, path)
    print(f"complete seen {cell}", flush=True)


def natural_citi_parts(task: TaskData) -> tuple[TaskData, dict[str, np.ndarray], dict[str, Any]]:
    payload = json.loads((HERE / "processed" / task.name / "natural_split.json").read_text())
    custom = task_with_state_parts(
        task,
        {"train": payload["train"], "validation": payload["validation"], "test": payload["test"]},
    )
    state = custom.rows["field_state"].astype(str).to_numpy()
    period = custom.rows["period"].astype(str).to_numpy()
    row_parts = {
        part: np.flatnonzero(
            (period == payload["row_periods"][part]) & np.isin(state, payload[part])
        )
        for part in ("train", "validation", "test")
    }
    if any(len(value) == 0 for value in row_parts.values()):
        raise AssertionError("natural Citi split has an empty row part")
    return custom, row_parts, payload


def run_natural_citi(setting: str, output: Path) -> None:
    cell = f"citibike_start_station__natural__{setting}"
    path = output / f"{cell}.json"
    state_path = output / f"{cell}__state_metrics.parquet"
    if path.exists() and state_path.exists() and json.loads(path.read_text()).get("status") == "complete":
        print(f"resume {cell}", flush=True)
        return
    task, row_parts, natural = natural_citi_parts(load_task("citibike_start_station"))
    raw_target = pd.to_numeric(task.rows["target"], errors="raise").to_numpy(np.float64)
    target = (raw_target - raw_target[row_parts["train"]].mean()) / (raw_target[row_parts["train"]].std() or 1.0)
    ordinary = ordinary_design(task, row_parts["train"]) if setting == "full_table" else None
    bandwidth, bandwidth_trials = select_bandwidth_custom(
        task, row_parts, setting, target, task.row_state_indices(), ordinary
    )
    all_tables, metadata = representation_tables(task, 0, bandwidth)
    keep = {
        "mpe", "similarity_unnormalized", "nystrom", "unknown_embedding",
        "q_ple", "uniform_ple", "mpe_equality", "raw_coordinates", "raw_latlon",
        "coordinate_fourier", "spatial_rbf",
    }
    tables = {name: table for name, table in all_tables.items() if name in keep}
    for corruption in range(10):
        tables[f"mpe_corrupt_{corruption}"] = corrupted_mpe_table(task, 0, bandwidth, corruption)
    results, states = evaluate_tables(task, row_parts, setting, tables)
    states.assign(task=task.name, split="natural", setting=setting, control="natural_time").to_parquet(
        state_path, index=False, compression="zstd"
    )
    payload = {
        "status": "complete", "control": "natural_time", "task": task.name,
        "source_unit": task.manifest["source_unit"], "setting": setting,
        "natural_rule": natural["rule"],
        "rows": {name: len(value) for name, value in row_parts.items()},
        "states": {name: len(natural[name]) for name in ("train", "validation", "test")},
        "selected_bandwidth": bandwidth, "bandwidth_trials": bandwidth_trials,
        "representation_metadata": metadata, "results": results,
        "test_evaluations_per_representation": 1,
    }
    atomic_json(payload, path)
    print(f"complete {cell}", flush=True)


def small_state_partition(states: list[str], seed: int) -> dict[str, list[str]]:
    unique = sorted(set(states), key=str)
    if len(unique) >= 15:
        return {name: list(map(str, values)) for name, values in make_state_partition(unique, seed).items()}
    if len(unique) < 3:
        raise ValueError("nominal control requires at least three eligible states")
    ordered = sorted(unique, key=lambda value: stable_key(seed, value))
    n_train = max(1, int(np.floor(0.6 * len(ordered))))
    n_validation = max(1, int(np.floor(0.2 * len(ordered))))
    if n_train + n_validation >= len(ordered):
        n_train = len(ordered) - 2
        n_validation = 1
    return {
        "train": ordered[:n_train],
        "validation": ordered[n_train : n_train + n_validation],
        "test": ordered[n_train + n_validation :],
    }


def nominal_task(name: str) -> TaskData:
    specifications = {
        "acs_class_of_worker": ("acs_occupation", "COW", "ACS"),
        "tlc_payment_type": ("tlc_pickup_zone", "payment_type", "NYC_TLC"),
        "bts_reporting_airline": ("airline_origin_airport", "reporting_airline", "BTS"),
    }
    base_name, field, source = specifications[name]
    base = load_task(base_name)
    frame = base.rows.copy()
    frame["original_metric_field"] = frame["field_state"].astype(str)
    frame["field_state"] = frame[field].astype("string").fillna("__MISSING__").astype(str)
    counts = frame["field_state"].value_counts()
    states = sorted(counts[counts >= 50].index.astype(str), key=str)
    frame = frame[frame["field_state"].isin(states)].reset_index(drop=True)
    ordinary = [column for column in base.manifest["ordinary_covariates"] if column != field]
    if "original_metric_field" not in ordinary:
        ordinary.append("original_metric_field")
    splits = {
        str(index): {"seed": seed, **small_state_partition(states, seed)}
        for index, seed in enumerate(SEEDS)
    }
    manifest = {
        "status": "RUN", "source_unit": source, "ordinary_covariates": ordinary,
        "metric": "equality distance", "metric_target_independent": True,
        "control_field": field, "base_task": base_name,
    }
    return TaskData(
        name=name, rows=frame, states=pd.DataFrame({"state_id": states}),
        distance=equality_distance(len(states)), splits=splits, manifest=manifest, arrays={},
    )


def nominal_random_geometry(task: TaskData, split: int) -> np.ndarray:
    rng = np.random.default_rng(20262000 + split + int.from_bytes(hashlib.sha256(task.name.encode()).digest()[:2], "little"))
    coordinates = rng.normal(size=(len(task.states), 2))
    distance = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2)
    lookup = task.state_to_index
    training = np.asarray([lookup[state] for state in task.splits[str(split)]["train"]], dtype=np.int64)
    landmarks = farthest_point_landmarks(distance, training, 32, state_ids=task.state_ids)
    candidates = bandwidth_grid(distance, training)
    bandwidth = float(candidates[len(candidates) // 2])
    return state_weight_table(distance, landmarks, bandwidth).astype(np.float32)


def run_nominal_cell(name: str, split: int, setting: str, output: Path) -> None:
    cell = f"{name}__split{split}__{setting}"
    path = output / f"{cell}.json"
    state_path = output / f"{cell}__state_metrics.parquet"
    if path.exists() and state_path.exists() and json.loads(path.read_text()).get("status") == "complete":
        print(f"resume nominal {cell}", flush=True)
        return
    task = nominal_task(name)
    lookup = task.state_to_index
    split_payload = task.splits[str(split)]
    state_values = task.rows["field_state"].astype(str).to_numpy()
    row_parts = {
        part: np.flatnonzero(np.isin(state_values, split_payload[part]))
        for part in ("train", "validation", "test")
    }
    custom = task_with_state_parts(
        task, {part: split_payload[part] for part in ("train", "validation", "test")}
    )
    training = np.asarray([lookup[state] for state in split_payload["train"]], dtype=np.int64)
    landmarks = farthest_point_landmarks(custom.distance, training, 32, state_ids=custom.state_ids)
    equality_mpe = state_weight_table(custom.distance, landmarks, 1.0).astype(np.float32)
    all_tables, _ = representation_tables(custom, 0, 1.0)
    tables = {
        "mpe_equality": equality_mpe,
        "lookup_unknown": all_tables["unknown_embedding"],
        "support_complete_onehot": all_tables["support_complete_categorical"],
        "uniform_ple": all_tables["uniform_ple"],
        "random_geometry_mpe": nominal_random_geometry(task, split),
    }
    results, states = evaluate_tables(custom, row_parts, setting, tables)
    states.assign(task=name, split=split, setting=setting, control="nominal_equality").to_parquet(
        state_path, index=False, compression="zstd"
    )
    payload = {
        "status": "complete", "control": "nominal_equality", "task": name,
        "source_unit": custom.manifest["source_unit"], "base_task": custom.manifest["base_task"],
        "field": custom.manifest["control_field"], "split": split, "seed": SEEDS[split],
        "setting": setting, "eligible_states": len(custom.states),
        "states": {part: len(split_payload[part]) for part in ("train", "validation", "test")},
        "rows": {part: len(row_parts[part]) for part in row_parts}, "results": results,
        "test_evaluations_per_representation": 1,
    }
    atomic_json(payload, path)
    print(f"complete nominal {cell}", flush=True)


def consolidate(folder: Path, stem: str) -> None:
    rows = []
    state_frames = []
    for path in sorted(folder.glob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") != "complete":
            continue
        for result in payload.get("results", []):
            rows.append(
                {
                    "control": payload["control"], "task": payload["task"],
                    "source_unit": payload["source_unit"], "split": payload.get("split", "natural"),
                    "setting": payload["setting"],
                    **{key: value for key, value in result.items() if key != "validation_trials"},
                }
            )
        state_path = path.with_name(path.stem + "__state_metrics.parquet")
        if state_path.exists():
            state_frames.append(pd.read_parquet(state_path))
    if rows:
        frame = pd.DataFrame(rows)
        frame.to_parquet(HERE / "raw" / f"{stem}_results.parquet", index=False, compression="zstd")
        frame.to_csv(HERE / "raw" / f"{stem}_results.csv", index=False)
    if state_frames:
        states = pd.concat(state_frames, ignore_index=True)
        states.to_parquet(HERE / "raw" / f"{stem}_state_results.parquet", index=False, compression="zstd")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["seen", "natural", "nominal", "all"], default="all")
    parser.add_argument("--task", choices=DEFAULT_TASKS + ["all"], default="all")
    parser.add_argument("--split", type=int, choices=range(5))
    parser.add_argument("--setting", choices=[*SETTINGS, "all"], default="all")
    parser.add_argument("--consolidate-only", action="store_true")
    args = parser.parse_args()
    settings = SETTINGS if args.setting == "all" else (args.setting,)
    splits = range(5) if args.split is None else (args.split,)

    seen_output = HERE / "raw" / "seen_cells"
    natural_output = HERE / "raw" / "natural_cells"
    nominal_output = HERE / "raw" / "nominal_cells"
    for folder in (seen_output, natural_output, nominal_output):
        folder.mkdir(parents=True, exist_ok=True)

    if not args.consolidate_only and args.suite in {"seen", "all"}:
        tasks = DEFAULT_TASKS if args.task == "all" else [args.task]
        for task in tasks:
            for split in splits:
                for setting in settings:
                    run_seen_cell(task, split, setting, seen_output)
    if not args.consolidate_only and args.suite in {"natural", "all"}:
        for setting in settings:
            run_natural_citi(setting, natural_output)
    if not args.consolidate_only and args.suite in {"nominal", "all"}:
        for task in ("acs_class_of_worker", "tlc_payment_type", "bts_reporting_airline"):
            for split in splits:
                for setting in settings:
                    run_nominal_cell(task, split, setting, nominal_output)

    consolidate(seen_output, "seen")
    consolidate(natural_output, "natural")
    consolidate(nominal_output, "nominal")


if __name__ == "__main__":
    main()
