#!/usr/bin/env python3
"""Validation-selected CatBoost and LightGBM cold-state baselines."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import catboost
import lightgbm as lgb
import numpy as np
import pandas as pd

from mpe import state_balanced_mean, state_loss_table
from representations import load_task, representation_tables, split_row_indices
from ridge_benchmark import DEFAULT_TASKS


HERE = Path(__file__).resolve().parent
TRIALS = [
    {"learning_rate": 0.03, "depth": 6, "l2": 1.0, "leaves": 31, "min_child": 20},
    {"learning_rate": 0.05, "depth": 8, "l2": 3.0, "leaves": 63, "min_child": 20},
    {"learning_rate": 0.10, "depth": 6, "l2": 3.0, "leaves": 31, "min_child": 50},
    {"learning_rate": 0.03, "depth": 8, "l2": 10.0, "leaves": 127, "min_child": 50},
    {"learning_rate": 0.05, "depth": 10, "l2": 1.0, "leaves": 127, "min_child": 100},
    {"learning_rate": 0.10, "depth": 8, "l2": 10.0, "leaves": 63, "min_child": 100},
    {"learning_rate": 0.02, "depth": 10, "l2": 3.0, "leaves": 255, "min_child": 50},
    {"learning_rate": 0.05, "depth": 6, "l2": 10.0, "leaves": 63, "min_child": 20},
]


def state_weights(states: np.ndarray) -> np.ndarray:
    unique, counts = np.unique(states, return_counts=True)
    lookup = dict(zip(unique.tolist(), counts.tolist()))
    values = np.asarray([1.0 / lookup[state] for state in states])
    return values * len(values) / values.sum()


def prepare_frame(task, setting: str, with_mpe: bool, split: int, bandwidth: float, train_rows: np.ndarray):
    columns = [] if setting == "isolated_field" else list(task.manifest["ordinary_covariates"])
    frame = task.rows[columns].copy()
    frame.insert(0, "metric_field", task.rows["field_state"].astype(str))
    categorical = []
    for column in frame:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = frame[column].astype("string").fillna("__MISSING__").astype(str)
            categorical.append(column)
        else:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame[column] = frame[column].fillna(frame.iloc[train_rows][column].median())
    if with_mpe:
        table, _ = representation_tables(task, split, bandwidth)
        weights = table["mpe"][task.row_state_indices()]
        for index in range(weights.shape[1]):
            frame[f"mpe_{index:02d}"] = weights[:, index]
    return frame, categorical


def encode_lightgbm(frame: pd.DataFrame, categorical: list[str], train_rows: np.ndarray) -> pd.DataFrame:
    result = frame.copy()
    for column in categorical:
        levels = sorted(result.iloc[train_rows][column].astype(str).unique())
        mapping = {value: index for index, value in enumerate(levels)}
        result[column] = pd.Categorical(
            result[column].astype(str).map(mapping).fillna(-1).astype(int),
            categories=range(-1, len(levels)),
        )
    return result


def summary(target: np.ndarray, prediction: np.ndarray, states: np.ndarray, scale: float):
    loss = (prediction - target) ** 2
    return {
        "state_balanced_standardized_mse": state_balanced_mean(loss, states),
        "row_weighted_standardized_mse": float(loss.mean()),
        "rmse": float(np.sqrt(loss.mean()) * scale),
        "mae": float(np.abs(prediction - target).mean() * scale),
    }


def run_cell(task_name: str, split: int, setting: str, model_name: str, with_mpe: bool, output: Path):
    suffix = "mpe" if with_mpe else "native"
    cell = f"{task_name}__split{split}__{setting}__{model_name}__{suffix}"
    path = output / f"{cell}.json"
    state_path = output / f"{cell}__state_metrics.parquet"
    if path.exists() and state_path.exists() and json.loads(path.read_text()).get("status") == "complete":
        print(f"resume {cell}", flush=True)
        return
    task = load_task(task_name)
    parts = split_row_indices(task, split)
    ridge_path = HERE / "raw" / "ridge_cells" / f"{task_name}__split{split}__{setting}.json"
    if not ridge_path.exists():
        raise FileNotFoundError(f"run ridge bandwidth selection first: {ridge_path}")
    bandwidth = float(json.loads(ridge_path.read_text())["selected_bandwidth"])
    frame, categorical = prepare_frame(task, setting, with_mpe, split, bandwidth, parts["train"])
    raw = pd.to_numeric(task.rows["target"], errors="raise").to_numpy(float)
    mean = float(raw[parts["train"]].mean())
    scale = float(raw[parts["train"]].std()) or 1.0
    target = (raw - mean) / scale
    states = task.rows["field_state"].astype(str).to_numpy()
    weight = state_weights(states[parts["train"]])
    validation_trials = []
    models = []
    started = time.perf_counter()
    if model_name == "lightgbm":
        design = encode_lightgbm(frame, categorical, parts["train"])
    else:
        design = frame
    for trial_index, trial in enumerate(TRIALS):
        if model_name == "catboost":
            model = catboost.CatBoostRegressor(
                iterations=1500, learning_rate=trial["learning_rate"], depth=trial["depth"],
                l2_leaf_reg=trial["l2"], loss_function="RMSE", random_seed=20261101,
                verbose=False, allow_writing_files=False, thread_count=8,
            )
            model.fit(
                design.iloc[parts["train"]], target[parts["train"]],
                cat_features=categorical, sample_weight=weight,
                eval_set=(design.iloc[parts["validation"]], target[parts["validation"]]),
                early_stopping_rounds=50, verbose=False,
            )
        else:
            model = lgb.LGBMRegressor(
                n_estimators=1500, learning_rate=trial["learning_rate"], num_leaves=trial["leaves"],
                max_depth=trial["depth"], reg_lambda=trial["l2"], min_child_samples=trial["min_child"],
                random_state=20261101, n_jobs=8, verbosity=-1,
            )
            model.fit(
                design.iloc[parts["train"]], target[parts["train"]], sample_weight=weight,
                categorical_feature=categorical,
                eval_set=[(design.iloc[parts["validation"]], target[parts["validation"]])],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
        prediction = model.predict(design.iloc[parts["validation"]])
        score = state_balanced_mean(
            (prediction - target[parts["validation"]]) ** 2, states[parts["validation"]]
        )
        validation_trials.append({"trial": trial_index, "config": trial, "score": score})
        models.append(model)
        print(f"{cell} trial={trial_index} val={score:.6f}", flush=True)
    winner = min(validation_trials, key=lambda row: (row["score"], row["trial"]))
    model = models[int(winner["trial"])]
    prediction = model.predict(design.iloc[parts["test"]])
    metrics = summary(target[parts["test"]], prediction, states[parts["test"]], scale)
    losses = (prediction - target[parts["test"]]) ** 2
    per_state = state_loss_table(losses, states[parts["test"]])
    pd.DataFrame(
        [{"state_id": state, "standardized_mse": value} for state, value in per_state.items()]
    ).assign(task=task_name, split=split, setting=setting, model=model_name, with_mpe=with_mpe).to_parquet(
        state_path, index=False, compression="zstd"
    )
    payload = {
        "status": "complete", "cell": cell, "task": task_name,
        "source_unit": task.manifest["source_unit"], "split": split, "setting": setting,
        "model": model_name, "with_mpe": with_mpe, "bandwidth": bandwidth,
        "validation_trials": validation_trials, "selected_trial": winner["trial"],
        "test_evaluations": 1, "metrics": metrics, "wall_seconds": time.perf_counter() - started,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=DEFAULT_TASKS)
    parser.add_argument("--split", required=True, type=int, choices=range(5))
    parser.add_argument("--setting", required=True, choices=["isolated_field", "full_table"])
    parser.add_argument("--model", required=True, choices=["catboost", "lightgbm"])
    parser.add_argument("--with-mpe", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "raw" / "tree_cells")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run_cell(args.task, args.split, args.setting, args.model, args.with_mpe, args.output)


if __name__ == "__main__":
    main()
