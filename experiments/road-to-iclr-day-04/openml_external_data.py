#!/usr/bin/env python3
"""Prospective OpenML loader with deterministic train/validation/test splits."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from semantic_multiview_pilot import PARTS, SplitData, _encode_categories


HERE = Path(__file__).resolve().parent


def _indices(target: np.ndarray, task: str, seed: int) -> dict[str, np.ndarray]:
    all_rows = np.arange(len(target))
    stratify = target if task == "classification" else None
    train_val, test = train_test_split(
        all_rows,
        test_size=0.2,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    second_stratify = target[train_val] if task == "classification" else None
    train, val = train_test_split(
        train_val,
        test_size=0.25,
        random_state=seed + 1,
        shuffle=True,
        stratify=second_stratify,
    )
    return {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}


def _categorical_frame(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = np.empty((len(frame), len(columns)), dtype=object)
    for column, name in enumerate(columns):
        series = frame[name].astype("string").fillna("__MISSING__")
        values[:, column] = series.to_numpy(dtype=str)
    return values


def load_openml(name: str, config: dict) -> SplitData:
    data_id = int(config["dataset_ids"][name])
    task = config["dataset_tasks"][name]
    bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    frame = bunch.data.copy()
    numeric_columns = [
        column
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column].dtype)
    ]
    categorical_columns = [
        column for column in frame.columns if column not in numeric_columns
    ]
    if not numeric_columns:
        raise ValueError(f"{name} has no numerical fields")
    numeric = frame[numeric_columns].to_numpy(dtype=np.float64)
    medians = np.nanmedian(numeric, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    bad = ~np.isfinite(numeric)
    if bad.any():
        numeric[bad] = medians[np.where(bad)[1]]

    raw_target = np.asarray(bunch.target)
    if task == "classification":
        levels, target = np.unique(raw_target.astype(str), return_inverse=True)
        if len(levels) != 2:
            raise ValueError(f"{name} has {len(levels)} classes")
        target = target.astype(np.float32)
    else:
        target = raw_target.astype(np.float64)
    indices = _indices(target, task, int(config["split_seed"]))
    x_num = {
        part: np.ascontiguousarray(numeric[index], dtype=np.float32)
        for part, index in indices.items()
    }
    if categorical_columns:
        raw_categorical = _categorical_frame(frame, categorical_columns)
        categorical_parts = {
            part: raw_categorical[index] for part, index in indices.items()
        }
        x_cat, cardinalities = _encode_categories(categorical_parts)
    else:
        x_cat, cardinalities = None, []
    target_parts = {part: target[index] for part, index in indices.items()}
    if task == "regression":
        y_mean = float(np.mean(target_parts["train"]))
        y_scale = float(np.std(target_parts["train"])) or 1.0
        y = {
            part: np.ascontiguousarray(
                (values - y_mean) / y_scale, dtype=np.float32
            )
            for part, values in target_parts.items()
        }
    else:
        y_mean, y_scale = 0.0, 1.0
        y = {
            part: np.ascontiguousarray(values, dtype=np.float32)
            for part, values in target_parts.items()
        }
    metadata = {
        "openml_data_id": data_id,
        "openml_name": bunch.details.get("name"),
        "task": task,
        "rows": len(frame),
        "numeric_fields": numeric_columns,
        "categorical_fields": categorical_columns,
        "split_sizes": {part: len(index) for part, index in indices.items()},
    }
    metadata_path = HERE / "results" / f"{name}.prospective_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return SplitData(
        x_num=x_num,
        x_bin=None,
        x_cat=x_cat,
        y=y,
        y_mean=y_mean,
        y_scale=y_scale,
        category_cardinalities=cardinalities,
        cyclic_columns=[],
        cyclic_names=[],
        cyclic_periods=[],
        cyclic_origins=[],
        split_sizes_full={part: len(index) for part, index in indices.items()},
    )
