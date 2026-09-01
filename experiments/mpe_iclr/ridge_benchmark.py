#!/usr/bin/env python3
"""State-disjoint ridge mechanism benchmark for the frozen MPE panel.

Validation is the only source of model/bandwidth choices.  A cell predicts the
sealed test rows exactly once, after its regularization value is selected.
Results are append-safe and keyed by task/split/setting/representation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mpe import (  # noqa: E402
    farthest_point_landmarks,
    nearest_support_distance,
    state_balanced_mean,
    state_loss_table,
    state_weight_table,
    weighted_metric_radius,
)
from representations import (  # noqa: E402
    TaskData,
    candidate_bandwidths,
    corrupted_mpe_table,
    load_task,
    representation_tables,
    split_row_indices,
    split_state_indices,
)


ALPHAS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
PARTS = ("train", "validation", "test")
DEFAULT_TASKS = [
    "acs_occupation",
    "acs_industry",
    "tlc_pickup_zone",
    "tlc_dropoff_zone",
    "citibike_start_station",
    "airline_origin_airport",
    "airline_destination_airport",
    "employee_salaries",
    "medical_charges",
]


def atomic_json(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def ordinary_design(task: TaskData, train_rows: np.ndarray) -> sparse.csr_matrix:
    columns = task.manifest["ordinary_covariates"]
    frame = task.rows[columns].copy()
    numeric_columns = [column for column in columns if pd.api.types.is_numeric_dtype(frame[column])]
    categorical_columns = [column for column in columns if column not in numeric_columns]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in categorical_columns:
        frame[column] = frame[column].astype("string").fillna("__MISSING__").astype(str)
    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32),
                ),
                categorical_columns,
            )
        )
    transformer = ColumnTransformer(transformers, sparse_threshold=1.0)
    transformer.fit(frame.iloc[train_rows])
    result = transformer.transform(frame)
    return sparse.csr_matrix(result, dtype=np.float32)


def row_representation(task: TaskData, table: np.ndarray) -> sparse.csr_matrix:
    return sparse.csr_matrix(np.asarray(table, dtype=np.float32)[task.row_state_indices()])


def state_balanced_training_weights(states: np.ndarray) -> np.ndarray:
    unique, counts = np.unique(states, return_counts=True)
    lookup = dict(zip(unique.tolist(), counts.tolist()))
    weight = np.asarray([1.0 / lookup[state] for state in states], dtype=np.float64)
    return weight * len(weight) / weight.sum()


def metrics(
    target_standard: np.ndarray,
    prediction_standard: np.ndarray,
    raw_target: np.ndarray,
    states: np.ndarray,
    target_scale: float,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    error = prediction_standard - target_standard
    losses = error**2
    state_mse = state_loss_table(losses, states)
    ordered = np.asarray(sorted(state_mse.values()), dtype=np.float64)
    count = len(ordered)
    summary = {
        "state_balanced_standardized_mse": state_balanced_mean(losses, states),
        "row_weighted_standardized_mse": float(np.mean(losses)),
        "rmse": float(np.sqrt(np.mean(losses)) * target_scale),
        "mae": float(np.mean(np.abs(error)) * target_scale),
        "worst_quartile_state_mse": float(np.mean(ordered[-max(1, math.ceil(0.25 * count)) :])),
        "worst_decile_state_mse": float(np.mean(ordered[-max(1, math.ceil(0.10 * count)) :])),
        "rows": int(len(states)),
        "states": int(count),
        "raw_target_mean": float(np.mean(raw_target)),
    }
    state_rows = [
        {
            "state_id": str(state),
            "rows": int(np.sum(states == state)),
            "standardized_mse": float(state_mse[state]),
            "standardized_mae": float(np.mean(np.abs(error[states == state]))),
        }
        for state in sorted(state_mse, key=str)
    ]
    return summary, state_rows


def fit_ridge(
    design: sparse.csr_matrix,
    target: np.ndarray,
    state_labels: np.ndarray,
    train_rows: np.ndarray,
    evaluation_rows: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    model = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr", tol=1e-5, max_iter=3000)
    sample_weight = state_balanced_training_weights(state_labels[train_rows])
    model.fit(design[train_rows], target[train_rows], sample_weight=sample_weight)
    prediction = np.asarray(model.predict(design[evaluation_rows]), dtype=np.float64)
    return prediction, time.perf_counter() - started


def select_bandwidth(
    task: TaskData,
    split_index: int,
    setting: str,
    ordinary: sparse.csr_matrix | None,
    row_parts: dict[str, np.ndarray],
    target: np.ndarray,
    row_states: np.ndarray,
) -> tuple[float, list[dict[str, float]]]:
    candidates = candidate_bandwidths(task, split_index)
    trials = []
    for bandwidth in candidates:
        tables, _ = representation_tables(task, split_index, float(bandwidth))
        representation = row_representation(task, tables["mpe"])
        design = representation if setting == "isolated_field" else sparse.hstack([ordinary, representation], format="csr")
        prediction, elapsed = fit_ridge(
            design, target, row_states, row_parts["train"], row_parts["validation"], 1.0
        )
        score = state_balanced_mean(
            (prediction - target[row_parts["validation"]]) ** 2,
            row_states[row_parts["validation"]],
        )
        trials.append({"bandwidth": float(bandwidth), "validation_score": score, "fit_seconds": elapsed})
        del tables, representation, design
    winner = min(trials, key=lambda item: (item["validation_score"], item["bandwidth"]))
    return float(winner["bandwidth"]), trials


def feature_digest(table: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(table.shape).encode())
    digest.update(np.ascontiguousarray(table).view(np.uint8))
    return digest.hexdigest()


def knn_state_predictions(
    task: TaskData,
    split_index: int,
    target: np.ndarray,
    row_states: np.ndarray,
    k: int,
) -> np.ndarray:
    parts = split_state_indices(task, split_index)
    train_rows = np.isin(row_states, parts["train"])
    means = {
        state: float(np.mean(target[train_rows & (row_states == state)]))
        for state in parts["train"]
    }
    prediction = np.empty(len(task.states), dtype=np.float64)
    for state in range(len(task.states)):
        distances = task.distance[state, parts["train"]]
        order = np.argsort(distances, kind="stable")[: min(k, len(order := distances))]
        neighbor_states = parts["train"][order]
        neighbor_distance = distances[order]
        zero = neighbor_distance <= 1e-12
        if zero.any():
            prediction[state] = np.mean([means[value] for value in neighbor_states[zero]])
        else:
            weights = 1.0 / neighbor_distance
            prediction[state] = np.average([means[value] for value in neighbor_states], weights=weights)
    return prediction


def support_values(task: TaskData, split_index: int, bandwidth: float) -> dict[str, tuple[float, float]]:
    parts = split_state_indices(task, split_index)
    landmarks = farthest_point_landmarks(task.distance, parts["train"], 32, state_ids=task.state_ids)
    weights = state_weight_table(task.distance, landmarks, bandwidth)
    nearest = nearest_support_distance(task.distance, np.arange(len(task.states)), parts["train"])
    weighted = weighted_metric_radius(weights, task.distance[:, landmarks])
    return {
        task.state_ids[index]: (float(nearest[index]), float(weighted[index]))
        for index in range(len(task.states))
    }


def run_cell(
    task: TaskData,
    split_index: int,
    setting: str,
    output: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    cell_id = f"{task.name}__split{split_index}__{setting}"
    cell_path = output / f"{cell_id}.json"
    state_path = output / f"{cell_id}__state_metrics.parquet"
    if cell_path.exists() and state_path.exists() and not force:
        payload = json.loads(cell_path.read_text())
        if payload.get("status") == "complete":
            print(f"resume {cell_id}", flush=True)
            return payload

    started = time.perf_counter()
    row_parts = split_row_indices(task, split_index)
    row_state_indices = task.row_state_indices()
    row_state_ids = task.rows["field_state"].astype(str).to_numpy()
    raw_target = pd.to_numeric(task.rows["target"], errors="raise").to_numpy(np.float64)
    target_mean = float(raw_target[row_parts["train"]].mean())
    target_scale = float(raw_target[row_parts["train"]].std()) or 1.0
    target = (raw_target - target_mean) / target_scale
    ordinary = ordinary_design(task, row_parts["train"]) if setting == "full_table" else None

    bandwidth, bandwidth_trials = select_bandwidth(
        task, split_index, setting, ordinary, row_parts, target, row_state_indices
    )
    tables, table_metadata = representation_tables(task, split_index, bandwidth)
    for corruption_index in range(10):
        name = f"mpe_corrupt_{corruption_index}"
        tables[name] = corrupted_mpe_table(task, split_index, bandwidth, corruption_index)

    support = support_values(task, split_index, bandwidth)
    results: list[dict[str, Any]] = []
    state_results: list[dict[str, Any]] = []
    cache: dict[str, tuple[float, list[dict[str, Any]], dict[str, float], list[dict[str, Any]], float]] = {}
    for representation_name, table in tables.items():
        digest = feature_digest(table)
        cache_key = f"{setting}|{digest}"
        if cache_key in cache:
            best_alpha, validation_trials, test_summary, per_state, fit_seconds = cache[cache_key]
            alias_of = next(item["representation"] for item in results if item.get("feature_digest") == digest)
        else:
            representation = row_representation(task, table)
            design = representation if ordinary is None else sparse.hstack([ordinary, representation], format="csr")
            validation_trials = []
            for alpha in ALPHAS:
                prediction, elapsed = fit_ridge(
                    design, target, row_state_indices,
                    row_parts["train"], row_parts["validation"], alpha,
                )
                score = state_balanced_mean(
                    (prediction - target[row_parts["validation"]]) ** 2,
                    row_state_ids[row_parts["validation"]],
                )
                validation_trials.append({"alpha": alpha, "score": score, "seconds": elapsed})
            winner = min(validation_trials, key=lambda item: (item["score"], item["alpha"]))
            best_alpha = float(winner["alpha"])
            test_prediction, test_seconds = fit_ridge(
                design, target, row_state_indices,
                row_parts["train"], row_parts["test"], best_alpha,
            )
            test_summary, per_state = metrics(
                target[row_parts["test"]], test_prediction,
                raw_target[row_parts["test"]], row_state_ids[row_parts["test"]], target_scale,
            )
            fit_seconds = float(sum(item["seconds"] for item in validation_trials) + test_seconds)
            cache[cache_key] = (best_alpha, validation_trials, test_summary, per_state, fit_seconds)
            alias_of = None
            del representation, design, test_prediction
        result = {
            "task": task.name,
            "source_unit": task.manifest["source_unit"],
            "split": split_index,
            "setting": setting,
            "backbone": "ridge",
            "representation": representation_name,
            "bandwidth": bandwidth,
            "feature_dimension": int(table.shape[1]),
            "feature_digest": digest,
            "alias_of": alias_of,
            "selected_alpha": best_alpha,
            "validation_trials": validation_trials,
            "fit_seconds": fit_seconds,
            **test_summary,
        }
        results.append(result)
        for row in per_state:
            nearest, weighted = support[row["state_id"]]
            state_results.append(
                {
                    "task": task.name,
                    "source_unit": task.manifest["source_unit"],
                    "split": split_index,
                    "setting": setting,
                    "backbone": "ridge",
                    "representation": representation_name,
                    "support_distance": nearest,
                    "weighted_landmark_radius": weighted,
                    **row,
                }
            )
        print(
            f"{cell_id} {representation_name} val={min(x['score'] for x in validation_trials):.5f} "
            f"test={test_summary['state_balanced_standardized_mse']:.5f}",
            flush=True,
        )
        gc.collect()

    # Direct metric kNN is selected on validation and is intentionally not
    # folded into a learned representation in the isolated mechanism view.
    knn_trials = []
    for k in (1, 3, 5, 10):
        state_prediction = knn_state_predictions(task, split_index, target, row_state_indices, k)
        prediction = state_prediction[row_state_indices[row_parts["validation"]]]
        score = state_balanced_mean(
            (prediction - target[row_parts["validation"]]) ** 2,
            row_state_ids[row_parts["validation"]],
        )
        knn_trials.append({"k": k, "score": score})
    best_k = min(knn_trials, key=lambda item: (item["score"], item["k"]))["k"]
    state_prediction = knn_state_predictions(task, split_index, target, row_state_indices, best_k)
    if setting == "isolated_field":
        test_prediction = state_prediction[row_state_indices[row_parts["test"]]]
    else:
        appended = sparse.csr_matrix(state_prediction[row_state_indices, None], dtype=np.float32)
        design = sparse.hstack([ordinary, appended], format="csr")
        alpha_trials = []
        for alpha in ALPHAS:
            prediction, elapsed = fit_ridge(design, target, row_state_indices, row_parts["train"], row_parts["validation"], alpha)
            score = state_balanced_mean((prediction - target[row_parts["validation"]]) ** 2, row_state_ids[row_parts["validation"]])
            alpha_trials.append({"alpha": alpha, "score": score, "seconds": elapsed})
        alpha_winner = min(alpha_trials, key=lambda item: (item["score"], item["alpha"]))
        test_prediction, _ = fit_ridge(design, target, row_state_indices, row_parts["train"], row_parts["test"], alpha_winner["alpha"])
    test_summary, per_state = metrics(
        target[row_parts["test"]], test_prediction, raw_target[row_parts["test"]],
        row_state_ids[row_parts["test"]], target_scale,
    )
    results.append(
        {
            "task": task.name, "source_unit": task.manifest["source_unit"], "split": split_index,
            "setting": setting, "backbone": "ridge", "representation": "knn_metric",
            "bandwidth": bandwidth, "feature_dimension": 1, "feature_digest": None, "alias_of": None,
            "selected_k": best_k, "validation_trials": knn_trials, "fit_seconds": 0.0, **test_summary,
        }
    )
    for row in per_state:
        nearest, weighted = support[row["state_id"]]
        state_results.append(
            {
                "task": task.name, "source_unit": task.manifest["source_unit"], "split": split_index,
                "setting": setting, "backbone": "ridge", "representation": "knn_metric",
                "support_distance": nearest, "weighted_landmark_radius": weighted, **row,
            }
        )

    state_frame = pd.DataFrame(state_results)
    state_frame.to_parquet(state_path, index=False, compression="zstd")
    payload = {
        "status": "complete",
        "cell_id": cell_id,
        "task": task.name,
        "source_unit": task.manifest["source_unit"],
        "split": split_index,
        "split_seed": task.splits[str(split_index)]["seed"],
        "setting": setting,
        "backbone": "ridge",
        "rows": {part: int(len(row_parts[part])) for part in PARTS},
        "states": {part: int(len(split_state_indices(task, split_index)[part])) for part in PARTS},
        "target_train_mean": target_mean,
        "target_train_scale": target_scale,
        "selected_bandwidth": bandwidth,
        "bandwidth_trials": bandwidth_trials,
        "representation_metadata": table_metadata,
        "alpha_grid": ALPHAS,
        "test_evaluations_per_representation": 1,
        "results": results,
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_json(payload, cell_path)
    return payload


def consolidate(output: Path) -> None:
    result_rows, state_paths = [], []
    for path in sorted(output.glob("*__*.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") == "complete" and "results" in payload:
            for result in payload["results"]:
                flattened = {key: value for key, value in result.items() if key != "validation_trials"}
                result_rows.append(flattened)
            state_path = path.with_name(path.stem + "__state_metrics.parquet")
            if state_path.exists():
                state_paths.append(state_path)
    if result_rows:
        frame = pd.DataFrame(result_rows)
        frame.to_parquet(output.parent / "ridge_results.parquet", index=False, compression="zstd")
        frame.to_csv(output.parent / "ridge_results.csv", index=False)
    if state_paths:
        state = pd.concat([pd.read_parquet(path) for path in state_paths], ignore_index=True)
        state.to_parquet(output.parent / "ridge_state_results.parquet", index=False, compression="zstd")
        state.to_csv(output.parent / "ridge_state_results.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=DEFAULT_TASKS + ["all"], default="all")
    parser.add_argument("--split", type=int, choices=range(5))
    parser.add_argument("--setting", choices=["isolated_field", "full_table", "all"], default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--consolidate-only", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "raw" / "ridge_cells")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.consolidate_only:
        tasks = DEFAULT_TASKS if args.task == "all" else [args.task]
        splits = range(5) if args.split is None else [args.split]
        settings = ["isolated_field", "full_table"] if args.setting == "all" else [args.setting]
        for task_name in tasks:
            folder = HERE / "processed" / task_name
            if not folder.exists():
                print(f"skip unavailable {task_name}", flush=True)
                continue
            manifest = json.loads((folder / "manifest.json").read_text())
            if manifest["status"] != "RUN":
                print(f"skip {task_name}: {manifest['status']}", flush=True)
                continue
            task = load_task(task_name)
            for split_index in splits:
                for setting in settings:
                    run_cell(task, split_index, setting, args.output, force=args.force)
    consolidate(args.output)


if __name__ == "__main__":
    main()
