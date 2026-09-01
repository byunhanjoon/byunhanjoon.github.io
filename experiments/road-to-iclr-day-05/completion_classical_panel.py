"""Frozen classical-model completion panel including XGBoost and LightGBM."""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from completion_neural_panel import HERE, PARTS, digest, prepare, render, views


CONFIG = HERE / "completion_classical_config.json"


def read_config(path: Path) -> dict[str, Any]:
    specific = json.loads(path.read_text())
    base = json.loads((path.parent / specific["base_config"]).read_text())
    return {**base, **{key: value for key, value in specific.items() if key != "base_config"}}


def render_ordinal(data, part: str, feature: np.ndarray, category: list[np.ndarray]) -> tuple[np.ndarray, tuple[int, ...]]:
    n_num = data.x_num[part].shape[1]
    columns = []
    categorical = []
    for position, field in enumerate(feature):
        field = int(field)
        if field < n_num:
            columns.append(data.x_num[part][:, field : field + 1])
        else:
            cat = field - n_num
            mapping = category[cat]
            values = np.full(len(data.x_cat[part]), -1, dtype=np.float32)
            known = data.x_cat[part][:, cat] >= 0
            values[known] = mapping[data.x_cat[part][known, cat]]
            columns.append(values[:, None])
            categorical.append(position)
    return np.ascontiguousarray(np.concatenate(columns, axis=1), dtype=np.float32), tuple(categorical)


def fit_predict(
    name: str,
    task: str,
    seed: int,
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    categorical: tuple[int, ...],
    iterations: int,
) -> np.ndarray:
    classification = task == "classification"
    if name == "onehot_linear":
        estimator = LogisticRegression(C=1.0, max_iter=2000, tol=1e-10, random_state=seed) if classification else Ridge(alpha=1.0)
    elif name == "native_histgb":
        cls = HistGradientBoostingClassifier if classification else HistGradientBoostingRegressor
        valid_categorical = [
            column for column in categorical
            if len(np.unique(train_x[train_x[:, column] >= 0, column])) <= 255
        ]
        estimator = cls(
            categorical_features=valid_categorical, max_iter=iterations, learning_rate=0.08,
            max_leaf_nodes=31, min_samples_leaf=15, l2_regularization=1.0, random_state=seed,
        )
    elif name == "catboost_native":
        from catboost import CatBoostClassifier, CatBoostRegressor
        cls = CatBoostClassifier if classification else CatBoostRegressor
        train_frame = pd.DataFrame(train_x)
        query_frame = pd.DataFrame(query_x)
        for column in categorical:
            train_frame[column] = train_frame[column].fillna(-1).astype(int).astype(str)
            query_frame[column] = query_frame[column].fillna(-1).astype(int).astype(str)
        kwargs = {
            "iterations": iterations, "depth": 6, "learning_rate": 0.08,
            "random_seed": seed, "verbose": False, "allow_writing_files": False,
            "thread_count": 1, "loss_function": "Logloss" if classification else "RMSE",
        }
        estimator = cls(**kwargs)
        estimator.fit(train_frame, train_y, cat_features=list(categorical))
        raw = estimator.predict_proba(query_frame) if classification else estimator.predict(query_frame)[:, None]
        return np.asarray(raw, dtype=np.float64)
    elif name == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor
        cls = XGBClassifier if classification else XGBRegressor
        estimator = cls(
            n_estimators=iterations, max_depth=6, learning_rate=0.08,
            subsample=1.0, colsample_bytree=1.0, reg_lambda=1.0,
            random_state=seed, n_jobs=1, tree_method="hist",
            objective="binary:logistic" if classification else "reg:squarederror",
            eval_metric="logloss" if classification else "rmse",
        )
    elif name == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor
        cls = LGBMClassifier if classification else LGBMRegressor
        estimator = cls(
            n_estimators=iterations, max_depth=-1, num_leaves=31, learning_rate=0.08,
            subsample=1.0, colsample_bytree=1.0, reg_lambda=1.0,
            random_state=seed, n_jobs=1, verbosity=-1,
        )
    else:
        raise ValueError(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if name == "lightgbm" and categorical:
            estimator.fit(train_x, train_y, categorical_feature=list(categorical))
        else:
            estimator.fit(train_x, train_y)
    raw = estimator.predict_proba(query_x) if classification else estimator.predict(query_x)[:, None]
    return np.asarray(raw, dtype=np.float64)


def run(dataset: str, model: str, config: dict[str, Any], output: Path) -> None:
    data = prepare(dataset, int(config["split_seed"]), config)
    design = views(data, config)
    actions = list(np.ndindex(
        len(design["feature"]), len(design["category"]), len(design["class"]), len(config["seeds"])
    ))
    output_dim = 2 if data.task == "classification" else 1
    validation = np.empty((len(actions), len(data.y["validation"]), output_dim), dtype=np.float32)
    test = np.empty((len(actions), len(data.y["test"]), output_dim), dtype=np.float32)
    telemetry = []
    for index, (fi, ci, li, si) in enumerate(actions):
        class_map = design["class"][li]
        rendered = {}; categorical = None
        for part in PARTS:
            if model == "onehot_linear":
                rendered[part], _ = render(data, part, design["feature"][fi], design["category"][ci])
                current = ()
            else:
                rendered[part], current = render_ordinal(data, part, design["feature"][fi], design["category"][ci])
            categorical = current if categorical is None else categorical
            if categorical != current:
                raise AssertionError("categorical positions changed")
        train_y = class_map[data.y["train"]] if data.task == "classification" else data.y["train"]
        query = np.concatenate((rendered["validation"], rendered["test"]), axis=0)
        started = time.perf_counter()
        raw = fit_predict(
            model, data.task, int(config["seeds"][si]), rendered["train"], train_y,
            query, categorical, int(config["iterations"]),
        )
        elapsed = time.perf_counter() - started
        if data.task == "classification":
            raw = raw[:, class_map]
        validation[index] = raw[: len(data.y["validation"])]
        test[index] = raw[len(data.y["validation"]) :]
        telemetry.append({
            "action": index, "feature": fi, "category": ci, "class": li,
            "seed": si, "wall_seconds": elapsed,
        })
        print(f"{dataset} {model} {index + 1}/{len(actions)}", flush=True)
    stem = f"{dataset}__{model}"
    np.savez_compressed(
        output / f"{stem}.npz", validation_predictions=validation,
        test_predictions=test, validation_y=data.y["validation"], test_y=data.y["test"],
        actions=np.asarray(actions, dtype=np.int16), y_mean=data.y_mean, y_scale=data.y_scale,
    )
    manifest = {
        "status": "complete", "dataset": dataset, "task": data.task, "model": model,
        "actions": len(actions), "represented_fits": len(actions),
        "split_seed": config["split_seed"], "wall_seconds": float(sum(x["wall_seconds"] for x in telemetry)),
        "device": "cpu", "threads_per_fit": int(config["threads_per_fit"]),
        "protocol_sha256": config["protocol_sha256"], "telemetry": telemetry,
    }
    (output / f"{stem}.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "completion_classical")
    args = parser.parse_args()
    config = read_config(args.config)
    if digest(HERE / config["protocol"]) != config["protocol_sha256"]:
        raise AssertionError("completion protocol hash mismatch")
    if args.dataset not in config["datasets"] or args.model not in config["models"]:
        raise ValueError("cell outside frozen config")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run(args.dataset, args.model, config, args.output_dir)


if __name__ == "__main__":
    main()
