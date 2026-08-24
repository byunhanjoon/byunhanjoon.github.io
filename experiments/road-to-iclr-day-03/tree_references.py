"""Frozen CatBoost, LightGBM, and XGBoost references on TabArena splits."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor


HERE = Path(__file__).resolve().parent
DEFAULT_JPLE = Path("/home/byunhanjoon/2027ICLR/projects/multi_ple/jple_tabarena")
sys.path.insert(0, str(DEFAULT_JPLE))

from src.data import load_official_split  # type: ignore[import-not-found]  # noqa: E402
from src.tabarena_adapter import (  # type: ignore[import-not-found]  # noqa: E402
    METRIC_BY_PROBLEM,
    metric_error,
    official_metric,
)


DEFAULT_DATASETS = [
    "wine_quality",
    "miami_housing",
    "Food_Delivery_Time",
    "seismic-bumps",
    "heloc",
    "credit_card_clients_default",
]


def frame(split, subset: str) -> tuple[pd.DataFrame, list[int]]:
    numerical = getattr(split, f"x_num_{subset}")
    categorical = getattr(split, f"x_cat_{subset}")
    result = pd.DataFrame(
        numerical, columns=[f"numerical_{i}" for i in range(numerical.shape[1])]
    )
    indices: list[int] = []
    for column in range(categorical.shape[1]):
        result[f"categorical_{column}"] = pd.Categorical(
            categorical[:, column], categories=range(split.cardinalities[column])
        )
        indices.append(numerical.shape[1] + column)
    return result, indices


def target(split, subset: str) -> np.ndarray:
    values = getattr(split, f"y_{subset}")
    if split.problem_type == "regression" and subset != "train":
        return ((values - split.y_mean) / split.y_std).astype(np.float32)
    return values


def prediction_for_metric(split, prediction: np.ndarray) -> np.ndarray:
    if split.problem_type == "regression":
        return prediction * split.y_std + split.y_mean
    return np.column_stack((1.0 - prediction, prediction))


def fit_reference(split, model_name: str, seed: int, threads: int) -> dict[str, object]:
    train_x, categorical_indices = frame(split, "train")
    val_x, _ = frame(split, "val")
    test_x, _ = frame(split, "test")
    train_y, val_y = target(split, "train"), target(split, "val")
    regression = split.problem_type == "regression"
    if model_name == "catboost":
        common = dict(
            iterations=2_000,
            depth=8,
            learning_rate=0.05,
            l2_leaf_reg=3.0,
            random_seed=seed,
            thread_count=threads,
            allow_writing_files=False,
            verbose=False,
        )
        model = (
            CatBoostRegressor(loss_function="RMSE", **common)
            if regression
            else CatBoostClassifier(loss_function="Logloss", eval_metric="AUC", **common)
        )
        start = time.perf_counter()
        model.fit(
            train_x,
            train_y,
            cat_features=categorical_indices,
            eval_set=(val_x, val_y),
            early_stopping_rounds=100,
            use_best_model=True,
        )
        train_seconds = time.perf_counter() - start
        start = time.perf_counter()
        raw_prediction = (
            model.predict(test_x)
            if regression
            else model.predict_proba(test_x)[:, 1]
        )
        inference_seconds = time.perf_counter() - start
        best_iteration = int(model.get_best_iteration())
    elif model_name == "lightgbm":
        cls = lgb.LGBMRegressor if regression else lgb.LGBMClassifier
        model = cls(
            n_estimators=2_000,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=threads,
            verbosity=-1,
        )
        start = time.perf_counter()
        model.fit(
            train_x,
            train_y,
            eval_set=[(val_x, val_y)],
            callbacks=[lgb.early_stopping(100, verbose=False)],
            categorical_feature=[f"categorical_{i}" for i in range(len(categorical_indices))],
        )
        train_seconds = time.perf_counter() - start
        start = time.perf_counter()
        raw_prediction = (
            model.predict(test_x)
            if regression
            else model.predict_proba(test_x)[:, 1]
        )
        inference_seconds = time.perf_counter() - start
        best_iteration = int(model.best_iteration_)
    elif model_name == "xgboost":
        cls = xgb.XGBRegressor if regression else xgb.XGBClassifier
        model = cls(
            n_estimators=2_000,
            learning_rate=0.03,
            max_depth=8,
            min_child_weight=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=threads,
            tree_method="hist",
            enable_categorical=True,
            eval_metric="rmse" if regression else "auc",
            early_stopping_rounds=100,
        )
        start = time.perf_counter()
        model.fit(train_x, train_y, eval_set=[(val_x, val_y)], verbose=False)
        train_seconds = time.perf_counter() - start
        start = time.perf_counter()
        raw_prediction = (
            model.predict(test_x)
            if regression
            else model.predict_proba(test_x)[:, 1]
        )
        inference_seconds = time.perf_counter() - start
        best_iteration = int(model.best_iteration)
    else:
        raise ValueError(model_name)

    metric_prediction = prediction_for_metric(split, np.asarray(raw_prediction))
    metric = official_metric(split.problem_type, split.y_test, metric_prediction)
    return {
        "test_metric": metric,
        "test_error": metric_error(split.problem_type, metric),
        "best_iteration": best_iteration,
        "training_wall_time_s": train_seconds,
        "inference_time_s": inference_seconds,
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jple-root", type=Path, default=DEFAULT_JPLE)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--folds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--models", nargs="+", choices=["catboost", "lightgbm", "xgboost"],
        default=["catboost", "lightgbm", "xgboost"],
    )
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--bins", type=int, default=128)
    parser.add_argument(
        "--output", type=Path, default=HERE / "results" / "tree_references.csv"
    )
    args = parser.parse_args()
    specs = json.loads(
        (args.jple_root / "configs" / "stage1_datasets.json").read_text()
    )["datasets"]
    by_name = {spec["dataset"]: spec for spec in specs}
    rows: list[dict[str, object]] = []
    if args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    completed = {
        (row["dataset"], int(row["fold"]), row["model"]) for row in rows
    }
    for dataset_name in args.datasets:
        spec = by_name[dataset_name]
        if spec["problem_type"] not in ("regression", "binary"):
            continue
        for fold in args.folds:
            split = load_official_split(
                spec,
                repeat=0,
                fold=fold,
                n_bins=args.bins,
                validation_fraction=0.2,
                seed=17,
                cache_dir=args.jple_root / "data_cache" / "openml",
            )
            for model_name in args.models:
                if (dataset_name, fold, model_name) in completed:
                    continue
                result = fit_reference(split, model_name, args.seed + 1009 * fold, args.threads)
                row = {
                    "dataset": dataset_name,
                    "problem_type": split.problem_type,
                    "fold": fold,
                    "split_hash": split.split_hash,
                    "model": model_name,
                    "seed": args.seed + 1009 * fold,
                    "official_metric": METRIC_BY_PROBLEM[split.problem_type],
                    "train_rows": len(split.y_train),
                    **result,
                }
                rows.append(row)
                completed.add((dataset_name, fold, model_name))
                write_rows(args.output, rows)
                print(
                    f"{dataset_name} fold={fold} {model_name}: "
                    f"error={float(result['test_error']):.6f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
