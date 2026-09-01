"""Exact feature-order x target-ID x seed panel on prospective OpenML data."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


HERE = Path(__file__).resolve().parent
DAY4 = HERE.parent / "road-to-iclr-day-04"
sys.path.insert(0, str(DAY4))

from openml_external_data import load_openml  # noqa: E402
import tier1_orbit as TIER1  # noqa: E402


def adapt(name: str, config: dict) -> TIER1.Dataset:
    if config["dataset_tasks"][name] == "multiclass":
        return adapt_multiclass(name, config)
    try:
        source = load_openml(name, config)
    except ValueError as error:
        if str(error) == f"{name} has no numerical fields":
            return adapt_direct_classification(name, config, task="binclass")
        raise
    task = "binclass" if config["dataset_tasks"][name] == "classification" else "regression"
    target_dtype = int if task == "binclass" else np.float64
    def categorical(part: str) -> np.ndarray:
        if source.x_cat is None:
            return np.empty((len(source.y[part]), 0), dtype=object)
        return np.asarray(source.x_cat[part], dtype=object)
    return TIER1.Dataset(
        name=name, task=task,
        train_n=np.asarray(source.x_num["train"], dtype=np.float64),
        validation_n=np.asarray(source.x_num["val"], dtype=np.float64),
        test_n=np.asarray(source.x_num["test"], dtype=np.float64),
        train_c=categorical("train"), validation_c=categorical("val"), test_c=categorical("test"),
        train_y=np.asarray(source.y["train"], dtype=target_dtype),
        validation_y=np.asarray(source.y["val"], dtype=target_dtype),
        test_y=np.asarray(source.y["test"], dtype=target_dtype),
    )


def adapt_multiclass(name: str, config: dict) -> TIER1.Dataset:
    return adapt_direct_classification(name, config, task="multiclass")


def adapt_direct_classification(name: str, config: dict, task: str) -> TIER1.Dataset:
    """Load classification data directly, including categorical-only tables."""
    bunch = fetch_openml(data_id=int(config["dataset_ids"][name]), as_frame=True, parser="auto")
    frame = bunch.data.copy()
    numeric_columns = [column for column in frame if pd.api.types.is_numeric_dtype(frame[column].dtype)]
    categorical_columns = [column for column in frame if column not in numeric_columns]
    numeric = frame[numeric_columns].to_numpy(dtype=np.float64)
    if numeric.shape[1]:
        medians = np.nanmedian(numeric, axis=0)
        bad = ~np.isfinite(numeric)
        if bad.any():
            numeric[bad] = np.where(np.isfinite(medians), medians, 0.0)[np.where(bad)[1]]
    if categorical_columns:
        categorical = np.column_stack([
            frame[column].astype("string").fillna("__MISSING__").to_numpy(dtype=str)
            for column in categorical_columns
        ]).astype(object)
    else:
        categorical = np.empty((len(frame), 0), dtype=object)
    _, target = np.unique(np.asarray(bunch.target).astype(str), return_inverse=True)
    all_rows = np.arange(len(target))
    train_val, test = train_test_split(
        all_rows, test_size=.2, random_state=int(config["split_seed"]), stratify=target
    )
    train, validation = train_test_split(
        train_val, test_size=.25, random_state=int(config["split_seed"]) + 1,
        stratify=target[train_val],
    )
    train, validation, test = map(np.sort, (train, validation, test))
    return TIER1.Dataset(
        name=name, task=task,
        train_n=numeric[train], validation_n=numeric[validation], test_n=numeric[test],
        train_c=categorical[train], validation_c=categorical[validation], test_c=categorical[test],
        train_y=target[train].astype(int), validation_y=target[validation].astype(int),
        test_y=target[test].astype(int),
    )


def main() -> None:
    run_started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "openml_external_cover_config.json")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "openml_external_cover")
    parser.add_argument("--split-seed", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.split_seed is not None:
        config["split_seed"] = int(args.split_seed)
    if args.dataset not in config["datasets"] or args.model not in config["models"]:
        raise ValueError("cell outside frozen panel")
    data = adapt(args.dataset, config)
    encoded, cardinalities = TIER1.encode_categories(data)
    views = TIER1.make_views(data, config, cardinalities)
    shape = (len(views["feature"]), len(views["category"]), len(views["class"]), len(config["seeds"]))
    output_width = len(np.unique(data.train_y)) if data.task in {"binclass", "multiclass"} else 1
    validation = np.empty(shape + (len(data.validation_y), output_width), dtype=np.float32)
    test = np.empty(shape + (len(data.test_y), output_width), dtype=np.float32)
    completed = 0
    fit_started = time.perf_counter()
    for fi, feature in enumerate(views["feature"]):
        for ci, maps in enumerate(views["category"]):
            train_x, categorical = TIER1.render(data.train_n, encoded["train"], feature, maps)
            validation_x, _ = TIER1.render(data.validation_n, encoded["validation"], feature, maps)
            test_x, _ = TIER1.render(data.test_n, encoded["test"], feature, maps)
            query = np.concatenate((validation_x, test_x), axis=0)
            for li, class_map in enumerate(views["class"]):
                for si, seed in enumerate(config["seeds"]):
                    train_target = class_map[data.train_y] if data.task in {"binclass", "multiclass"} else data.train_y
                    raw = TIER1.fit_predict(
                        args.model, data.task, int(seed), train_x, train_target,
                        query, categorical, config,
                    )
                    if data.task in {"binclass", "multiclass"}:
                        raw = raw[:, class_map]
                    validation[fi, ci, li, si] = raw[: len(validation_x)]
                    test[fi, ci, li, si] = raw[len(validation_x):]
                    completed += 1
    fit_loop_seconds = time.perf_counter() - fit_started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.dataset}__{args.model}"
    np.savez_compressed(
        args.output_dir / f"{stem}.npz", validation_predictions=validation,
        test_predictions=test, validation_y=data.validation_y, test_y=data.test_y,
    )
    end_to_end_seconds = time.perf_counter() - run_started
    manifest = {
        "status": "complete", "dataset": args.dataset, "model": args.model,
        "task": data.task, "shape": list(shape), "fits": completed,
        "rows": {"train": len(data.train_y), "validation": len(data.validation_y), "test": len(data.test_y)},
        "full_product_verified": completed == math.prod(shape),
        "post_failure_schema_adapter": bool(
            args.dataset in config.get("post_failure_schema_adapter", [])
        ),
        "effective_split_seed": int(config["split_seed"]),
        "timing": {
            "setup_seconds": fit_started - run_started,
            "fit_loop_seconds": fit_loop_seconds,
            "end_to_end_seconds": end_to_end_seconds,
            "fits_per_fit_loop_second": completed / fit_loop_seconds,
        },
    }
    (args.output_dir / f"{stem}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
