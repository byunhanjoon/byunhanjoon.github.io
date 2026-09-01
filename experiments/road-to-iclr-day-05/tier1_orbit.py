"""Frozen Day-5 Tier-1 schema-orbit experiment.

Each invocation runs one dataset/model cell and saves compact float32 aligned
validation/test predictions. Analysis is deliberately separate so outcomes
cannot change the transformation menu or fit recipes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "tier1_config.json"


@dataclass(frozen=True)
class Dataset:
    name: str
    task: str
    train_n: np.ndarray
    validation_n: np.ndarray
    test_n: np.ndarray
    train_c: np.ndarray
    validation_c: np.ndarray
    test_c: np.ndarray
    train_y: np.ndarray
    validation_y: np.ndarray
    test_y: np.ndarray


def read_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_optional(path: Path, rows: int) -> np.ndarray:
    if not path.exists():
        return np.empty((rows, 0), dtype=object)
    return np.asarray(np.load(path, allow_pickle=True), dtype=object)


def _subsample_indices(y: np.ndarray, maximum: int, seed: int, task: str) -> np.ndarray:
    if len(y) <= maximum:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    if task in {"binclass", "multiclass"}:
        chosen = []
        for label in np.unique(y):
            candidates = np.flatnonzero(y == label)
            count = max(1, round(maximum * len(candidates) / len(y)))
            chosen.extend(rng.choice(candidates, min(count, len(candidates)), replace=False))
        chosen = np.asarray(chosen, dtype=int)
        if len(chosen) > maximum:
            chosen = rng.choice(chosen, maximum, replace=False)
        elif len(chosen) < maximum:
            remainder = np.setdiff1d(np.arange(len(y)), chosen, assume_unique=False)
            chosen = np.concatenate((chosen, rng.choice(remainder, maximum - len(chosen), replace=False)))
        return np.sort(chosen)
    return np.sort(rng.choice(len(y), maximum, replace=False))


def load_dataset(root: Path, config: dict[str, Any]) -> Dataset:
    info = json.loads((root / "info.json").read_text())
    task = str(info["task_type"])
    if task not in {"binclass", "multiclass", "regression"}:
        raise ValueError(f"unsupported task {task}")
    arrays: dict[str, np.ndarray] = {}
    caps = config["subsample"]
    for split, maximum, offset in (
        ("train", int(caps["max_train_rows"]), 0),
        ("val", int(caps["max_validation_rows"]), 1),
        ("test", int(caps["max_test_rows"]), 2),
    ):
        y = np.asarray(np.load(root / f"y_{split}.npy"))
        n = np.asarray(np.load(root / f"N_{split}.npy"), dtype=np.float64)
        c = _load_optional(root / f"C_{split}.npy", len(y))
        indices = _subsample_indices(y, maximum, int(caps["seed"]) + offset, task)
        arrays[f"{split}_y"] = y[indices]
        arrays[f"{split}_n"] = n[indices]
        arrays[f"{split}_c"] = c[indices]
    if task == "regression":
        mean = float(arrays["train_y"].mean())
        scale = float(arrays["train_y"].std())
        if not scale:
            scale = 1.0
        for split in ("train", "val", "test"):
            arrays[f"{split}_y"] = (arrays[f"{split}_y"].astype(float) - mean) / scale
    else:
        labels = np.unique(arrays["train_y"])
        mapping = {value: index for index, value in enumerate(labels)}
        for split in ("train", "val", "test"):
            arrays[f"{split}_y"] = np.asarray([mapping[value] for value in arrays[f"{split}_y"]], dtype=int)
    return Dataset(
        name=root.name,
        task=task,
        train_n=arrays["train_n"], validation_n=arrays["val_n"], test_n=arrays["test_n"],
        train_c=arrays["train_c"], validation_c=arrays["val_c"], test_c=arrays["test_c"],
        train_y=arrays["train_y"], validation_y=arrays["val_y"], test_y=arrays["test_y"],
    )


def encode_categories(data: Dataset) -> tuple[dict[str, np.ndarray], list[int]]:
    encoded: dict[str, np.ndarray] = {}
    cardinalities: list[int] = []
    for column in range(data.train_c.shape[1]):
        values = pd.Series(data.train_c[:, column]).fillna("<MISSING>").astype(str)
        levels = sorted(values.unique().tolist())
        mapping = {value: index for index, value in enumerate(levels)}
        cardinalities.append(len(levels))
        for split, raw in (("train", data.train_c), ("validation", data.validation_c), ("test", data.test_c)):
            current = pd.Series(raw[:, column]).fillna("<MISSING>").astype(str)
            column_values = np.asarray([mapping.get(value, -1) for value in current], dtype=float)
            encoded.setdefault(split, np.empty((len(raw), data.train_c.shape[1]), dtype=float))[:, column] = column_values
    if data.train_c.shape[1] == 0:
        for split, raw in (("train", data.train_c), ("validation", data.validation_c), ("test", data.test_c)):
            encoded[split] = np.empty((len(raw), 0), dtype=float)
    return encoded, cardinalities


def make_views(data: Dataset, config: dict[str, Any], cardinalities: list[int]) -> dict[str, list[Any]]:
    level_config = config["factor_levels"]
    total_features = data.train_n.shape[1] + data.train_c.shape[1]
    view_seeds = config.get("view_seeds", {})
    feature_rng = np.random.default_rng(int(view_seeds.get("feature", 91773)))
    features = [np.arange(total_features)]
    while len(features) < int(level_config["feature"]):
        candidate = feature_rng.permutation(total_features)
        if not any(np.array_equal(candidate, existing) for existing in features):
            features.append(candidate)
    category_rng = np.random.default_rng(int(view_seeds.get("category", 821571)))
    categories: list[list[np.ndarray]] = [[np.arange(size) for size in cardinalities]]
    available_category_maps = math.prod(math.factorial(size) for size in cardinalities)
    target_categories = min(int(level_config["category"]), available_category_maps) if cardinalities else 1
    while len(categories) < target_categories:
        candidate = [category_rng.permutation(size) for size in cardinalities]
        if not any(all(np.array_equal(a, b) for a, b in zip(candidate, existing)) for existing in categories):
            categories.append(candidate)
    if data.task in {"binclass", "multiclass"}:
        class_count = len(np.unique(data.train_y))
        class_rng = np.random.default_rng(int(view_seeds.get("class", 194027)))
        classes = [np.arange(class_count)]
        target_classes = min(int(level_config["class"]), math.factorial(class_count))
        while len(classes) < target_classes:
            candidate = class_rng.permutation(class_count)
            if not any(np.array_equal(candidate, existing) for existing in classes):
                classes.append(candidate)
    else:
        classes = [np.asarray([0])]
    classes = classes[: int(level_config["class"])]
    return {"feature": features, "category": categories, "class": classes}


def render(
    numerical: np.ndarray,
    categorical: np.ndarray,
    feature_permutation: np.ndarray,
    category_maps: list[np.ndarray],
) -> tuple[np.ndarray, tuple[int, ...]]:
    categorical = categorical.copy()
    for column, mapping in enumerate(category_maps):
        known = categorical[:, column] >= 0
        categorical[known, column] = mapping[categorical[known, column].astype(int)]
    combined = np.concatenate((numerical, categorical), axis=1)
    original_categorical = set(range(numerical.shape[1], combined.shape[1]))
    transformed_categorical = tuple(
        new for new, old in enumerate(feature_permutation) if int(old) in original_categorical
    )
    return combined[:, feature_permutation], transformed_categorical


def preprocessing(categorical: tuple[int, ...], feature_count: int) -> ColumnTransformer:
    numerical = [index for index in range(feature_count) if index not in categorical]
    return ColumnTransformer(
        [
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numerical),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(categorical)),
        ],
        sparse_threshold=0.0,
    )


def fit_predict(
    model_name: str,
    task: str,
    seed: int,
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    categorical: tuple[int, ...],
    config: dict[str, Any],
) -> np.ndarray:
    training = config["training"]
    threads = int(training["threads_per_fit"])
    classification = task in {"binclass", "multiclass"}
    if model_name == "onehot_linear":
        estimator = (
            LogisticRegression(C=1.0, max_iter=2000, tol=1e-10, random_state=seed)
            if classification else Ridge(alpha=1.0)
        )
        model = make_pipeline(preprocessing(categorical, train_x.shape[1]), estimator)
    elif model_name == "ordinal_forest":
        estimator_class = RandomForestClassifier if classification else RandomForestRegressor
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            estimator_class(
                n_estimators=int(training["forest_estimators"]),
                min_samples_leaf=3,
                max_features="sqrt",
                n_jobs=threads,
                random_state=seed,
            ),
        )
    elif model_name == "native_histgb":
        estimator_class = HistGradientBoostingClassifier if classification else HistGradientBoostingRegressor
        hist_categorical = list(categorical)
        if training.get("histgb_high_cardinality_policy") == "ordinal_fallback_above_255":
            hist_categorical = [
                column for column in categorical
                if len(np.unique(train_x[train_x[:, column] >= 0, column])) <= 255
            ]
        model = estimator_class(
            categorical_features=hist_categorical, learning_rate=0.08,
            max_iter=int(training["boosting_iterations"]), max_leaf_nodes=31,
            min_samples_leaf=15, l2_regularization=1.0, random_state=seed,
        )
    elif model_name == "catboost_native":
        from catboost import CatBoostClassifier, CatBoostRegressor

        train_frame = pd.DataFrame(train_x)
        query_frame = pd.DataFrame(query_x)
        for column in categorical:
            train_frame[column] = train_frame[column].fillna(-1).astype(int).astype(str)
            query_frame[column] = query_frame[column].fillna(-1).astype(int).astype(str)
        cls = CatBoostClassifier if classification else CatBoostRegressor
        kwargs: dict[str, Any] = {
            "iterations": int(training["boosting_iterations"]), "depth": 6,
            "learning_rate": 0.08, "random_seed": seed, "verbose": False,
            "allow_writing_files": False, "thread_count": threads,
        }
        if classification:
            kwargs["loss_function"] = "Logloss"
        else:
            kwargs["loss_function"] = "RMSE"
        model = cls(**kwargs)
        model.fit(train_frame, train_y, cat_features=list(categorical))
        raw = model.predict_proba(query_frame) if classification else model.predict(query_frame)[:, None]
        return np.asarray(raw, dtype=np.float64)
    elif model_name == "onehot_adam_mlp":
        cls = MLPClassifier if classification else MLPRegressor
        model = make_pipeline(
            preprocessing(categorical, train_x.shape[1]),
            cls(
                hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
                alpha=1e-4, batch_size=256, learning_rate_init=1e-3,
                max_iter=int(training["mlp_max_epochs"]), early_stopping=True,
                validation_fraction=0.15, n_iter_no_change=10, random_state=seed,
            ),
        )
    else:
        raise ValueError(model_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(train_x, train_y)
    if classification:
        return np.asarray(model.predict_proba(query_x), dtype=np.float64)
    return np.asarray(model.predict(query_x), dtype=np.float64)[:, None]


def run_cell(dataset_name: str, model_name: str, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    data_root = Path(config["data_root"]) / dataset_name
    data = load_dataset(data_root, config)
    encoded, cardinalities = encode_categories(data)
    views = make_views(data, config, cardinalities)
    shape = (len(views["feature"]), len(views["category"]), len(views["class"]), len(config["seeds"]))
    output_dim = len(np.unique(data.train_y)) if data.task in {"binclass", "multiclass"} else 1
    validation_predictions = np.empty(shape + (len(data.validation_y), output_dim), dtype=np.float32)
    test_predictions = np.empty(shape + (len(data.test_y), output_dim), dtype=np.float32)
    total = math.prod(shape)
    completed = 0
    for fi, feature in enumerate(views["feature"]):
        for ci, category in enumerate(views["category"]):
            train_x, categorical = render(data.train_n, encoded["train"], feature, category)
            validation_x, _ = render(data.validation_n, encoded["validation"], feature, category)
            test_x, _ = render(data.test_n, encoded["test"], feature, category)
            for li, class_map in enumerate(views["class"]):
                train_y = class_map[data.train_y] if data.task in {"binclass", "multiclass"} else data.train_y
                for si, seed in enumerate(config["seeds"]):
                    joined = np.concatenate((validation_x, test_x), axis=0)
                    raw = fit_predict(model_name, data.task, int(seed), train_x, train_y, joined, categorical, config)
                    if data.task in {"binclass", "multiclass"}:
                        raw = raw[:, class_map]
                    validation_predictions[fi, ci, li, si] = raw[: len(validation_x)]
                    test_predictions[fi, ci, li, si] = raw[len(validation_x) :]
                    completed += 1
                    print(f"{dataset_name} {model_name} {completed}/{total}", flush=True)
    manifest = {
        "status": "complete",
        "dataset": dataset_name,
        "model": model_name,
        "task": data.task,
        "factor_names": ["feature", "category", "class"],
        "factor_shape": list(shape[:-1]),
        "seeds": config["seeds"],
        "rows": {"train": len(data.train_y), "validation": len(data.validation_y), "test": len(data.test_y)},
        "features": {"numerical": data.train_n.shape[1], "categorical": data.train_c.shape[1]},
        "histgb_high_cardinality_policy": config["training"].get("histgb_high_cardinality_policy"),
        "source_hashes": {
            name: sha256(data_root / name)
            for name in ("N_train.npy", "N_val.npy", "N_test.npy", "y_train.npy", "y_val.npy", "y_test.npy")
        },
    }
    arrays = {
        "validation_predictions": validation_predictions,
        "test_predictions": test_predictions,
        "validation_y": data.validation_y,
        "test_y": data.test_y,
    }
    return manifest, arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "tier1")
    args = parser.parse_args()
    config = read_config(args.config)
    if args.dataset not in config["datasets"] or args.model not in config["models"]:
        raise ValueError("dataset/model not in frozen config")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.dataset}__{args.model}"
    manifest, arrays = run_cell(args.dataset, args.model, config)
    np.savez_compressed(args.output_dir / f"{stem}.npz", **arrays)
    (args.output_dir / f"{stem}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
