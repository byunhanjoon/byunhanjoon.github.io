"""Thin, explicit adapters for current TFMs and conventional controls."""

from __future__ import annotations

import functools
import gc
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TABPFN_CLASSIFIER = Path("/home/byunhanjoon/.cache/tabpfn/tabpfn-v2.5-classifier-v2.5_default.ckpt")
TABPFN_REGRESSOR = Path("/home/byunhanjoon/.cache/tabpfn/tabpfn-v2.5-regressor-v2.5_default.ckpt")
MITRA_PYTHON = Path(os.environ.get("REPARAM_MITRA_PYTHON", "/data/byunhanjoon/reparam_mitra_env/bin/python"))
MITRA_WORKER = Path(__file__).resolve().parents[2] / "scripts" / "mitra_worker.py"
MITRA_JOB_ROOT = Path(os.environ.get("REPARAM_MITRA_JOB_ROOT", "/data/byunhanjoon/reparam_mitra_jobs"))
MITRA_HF_HOME = Path(os.environ.get("REPARAM_MITRA_HF_HOME", "/data/byunhanjoon/huggingface"))
PYTABKIT_WORKER = Path(__file__).resolve().parents[2] / "scripts" / "pytabkit_worker.py"
PYTABKIT_JOB_ROOT = Path(os.environ.get("REPARAM_PYTABKIT_JOB_ROOT", "/data/byunhanjoon/reparam_pytabkit_jobs"))


@dataclass
class FitPrediction:
    prediction: np.ndarray
    telemetry: dict[str, Any]


@functools.lru_cache(maxsize=None)
def available_model(name: str, problem_type: str) -> tuple[bool, str]:
    modules = {
        "tabpfn_v25_single": "tabpfn",
        "tabpfn_v25_default": "tabpfn",
        "tabicl_v2_single": "tabicl",
        "tabicl_v2_default": "tabicl",
        "xgboost": "xgboost",
        "catboost": "catboost",
        "lightgbm": "lightgbm",
        "random_forest": "sklearn",
        "linear": "sklearn",
        "tabm_default": "pytabkit",
        "realmlp_default": "pytabkit",
    }
    if name in {"mitra", "mitra_default", "mitra_icl_single"}:
        if not MITRA_PYTHON.exists():
            return False, f"isolated Mitra interpreter is missing: {MITRA_PYTHON}"
        if not MITRA_WORKER.exists():
            return False, f"Mitra worker is missing: {MITRA_WORKER}"
        probe = subprocess.run(
            [
                str(MITRA_PYTHON),
                "-c",
                "import torch; from autogluon.tabular.models.mitra.mitra_model import MitraModel; "
                "assert torch.cuda.is_available(), 'CUDA is unavailable in Mitra environment'",
            ],
            capture_output=True,
            text=True,
        )
        if probe.returncode:
            detail = (probe.stderr or probe.stdout).strip().splitlines()
            return False, f"Mitra environment probe failed: {detail[-1] if detail else 'unknown error'}"
        return True, "available in isolated AutoGluon environment"
    if name not in modules:
        return False, f"unknown model adapter: {name}"
    if importlib.util.find_spec(modules[name]) is None:
        return False, f"Python module {modules[name]} is not installed"
    if name in {"tabm_default", "realmlp_default"} and not PYTABKIT_WORKER.exists():
        return False, f"pytabkit worker is missing: {PYTABKIT_WORKER}"
    if name.startswith("tabpfn"):
        checkpoint = TABPFN_REGRESSOR if problem_type == "regression" else TABPFN_CLASSIFIER
        if not checkpoint.exists():
            return False, f"checkpoint is missing: {checkpoint}"
    return True, "available"


@functools.lru_cache(maxsize=None)
def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"checkpoint": str(path), "checkpoint_sha256": digest, "checkpoint_bytes": path.stat().st_size}


def _shared_query_predictions(
    model: Any,
    X_queries: dict[str, pd.DataFrame],
    prediction_fn: Any,
    fit_seconds: float,
    base_telemetry: dict[str, Any] | None = None,
) -> dict[str, FitPrediction]:
    if not X_queries:
        raise ValueError("at least one query representation is required")
    shared_fit_id = f"{type(model).__name__}_{time.perf_counter_ns()}"
    outcomes: dict[str, FitPrediction] = {}
    for key, query in X_queries.items():
        started = time.perf_counter()
        prediction = prediction_fn(query)
        predict_seconds = time.perf_counter() - started
        telemetry = {
            **(base_telemetry or {}),
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "shared_fit_id": shared_fit_id,
            "shared_fit_query_count": len(X_queries),
        }
        outcomes[key] = FitPrediction(np.asarray(prediction), telemetry)
    return outcomes


def _encode_for_tree(
    train: pd.DataFrame, query: pd.DataFrame, categorical_columns: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    train_blocks, query_blocks = [], []
    categorical = set(categorical_columns)
    for name in train.columns:
        if name in categorical:
            values = train[name].astype("string").fillna("__MISSING__")
            categories = {value: index for index, value in enumerate(sorted(values.unique().tolist()))}
            train_blocks.append(values.map(categories).fillna(-1).to_numpy(dtype=np.float64)[:, None])
            query_blocks.append(
                query[name].astype("string").fillna("__MISSING__").map(categories).fillna(-1).to_numpy(dtype=np.float64)[:, None]
            )
        else:
            train_blocks.append(pd.to_numeric(train[name], errors="coerce").to_numpy(dtype=np.float64)[:, None])
            query_blocks.append(pd.to_numeric(query[name], errors="coerce").to_numpy(dtype=np.float64)[:, None])
    return np.concatenate(train_blocks, axis=1), np.concatenate(query_blocks, axis=1)


def _fit_xgboost(
    problem_type: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_query: pd.DataFrame,
    categorical_columns: list[str],
    seed: int,
) -> FitPrediction:
    import xgboost as xgb

    train, query = _encode_for_tree(X_train, X_query, categorical_columns)
    common = dict(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1.0,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=8,
        tree_method="hist",
    )
    if problem_type == "regression":
        model = xgb.XGBRegressor(objective="reg:squarederror", **common)
    else:
        n_classes = len(np.unique(y_train))
        objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
        extra = {} if n_classes == 2 else {"num_class": n_classes}
        model = xgb.XGBClassifier(objective=objective, **extra, **common)
    started = time.perf_counter()
    model.fit(train, y_train)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    prediction = model.predict(query) if problem_type == "regression" else model.predict_proba(query)
    predict_seconds = time.perf_counter() - started
    return FitPrediction(np.asarray(prediction), {"fit_seconds": fit_seconds, "predict_seconds": predict_seconds})


def _fit_xgboost_many(
    problem_type: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_queries: dict[str, pd.DataFrame],
    categorical_columns: list[str],
    seed: int,
) -> dict[str, FitPrediction]:
    import xgboost as xgb

    first_query = next(iter(X_queries.values()))
    train, _ = _encode_for_tree(X_train, first_query, categorical_columns)
    common = dict(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1.0,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=8,
        tree_method="hist",
    )
    if problem_type == "regression":
        model = xgb.XGBRegressor(objective="reg:squarederror", **common)
    else:
        n_classes = len(np.unique(y_train))
        objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
        extra = {} if n_classes == 2 else {"num_class": n_classes}
        model = xgb.XGBClassifier(objective=objective, **extra, **common)
    started = time.perf_counter()
    model.fit(train, y_train)
    fit_seconds = time.perf_counter() - started

    def predict(query_frame: pd.DataFrame) -> np.ndarray:
        _, query = _encode_for_tree(X_train, query_frame, categorical_columns)
        return model.predict(query) if problem_type == "regression" else model.predict_proba(query)

    return _shared_query_predictions(model, X_queries, predict, fit_seconds)


def _fit_catboost(
    problem_type: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_query: pd.DataFrame,
    categorical_columns: list[str],
    seed: int,
) -> FitPrediction:
    from catboost import CatBoostClassifier, CatBoostRegressor

    train, query = X_train.copy(), X_query.copy()
    for name in categorical_columns:
        train[name] = train[name].astype("string").fillna("__MISSING__")
        query[name] = query[name].astype("string").fillna("__MISSING__")
    common = dict(
        iterations=400,
        depth=7,
        learning_rate=0.05,
        random_seed=seed,
        thread_count=8,
        allow_writing_files=False,
        verbose=False,
    )
    model = CatBoostRegressor(loss_function="RMSE", **common) if problem_type == "regression" else CatBoostClassifier(loss_function="MultiClass", **common)
    started = time.perf_counter()
    model.fit(train, y_train, cat_features=categorical_columns)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    prediction = model.predict(query).reshape(-1) if problem_type == "regression" else model.predict_proba(query)
    predict_seconds = time.perf_counter() - started
    return FitPrediction(np.asarray(prediction), {"fit_seconds": fit_seconds, "predict_seconds": predict_seconds})


def _fit_catboost_many(
    problem_type: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_queries: dict[str, pd.DataFrame],
    categorical_columns: list[str],
    seed: int,
) -> dict[str, FitPrediction]:
    from catboost import CatBoostClassifier, CatBoostRegressor

    train = X_train.copy()
    for column in categorical_columns:
        train[column] = train[column].astype("string").fillna("__MISSING__")
    common = dict(
        iterations=400,
        depth=7,
        learning_rate=0.05,
        random_seed=seed,
        thread_count=8,
        allow_writing_files=False,
        verbose=False,
    )
    model = (
        CatBoostRegressor(loss_function="RMSE", **common)
        if problem_type == "regression"
        else CatBoostClassifier(loss_function="MultiClass", **common)
    )
    started = time.perf_counter()
    model.fit(train, y_train, cat_features=categorical_columns)
    fit_seconds = time.perf_counter() - started

    def predict(query_frame: pd.DataFrame) -> np.ndarray:
        query = query_frame.copy()
        for column in categorical_columns:
            query[column] = query[column].astype("string").fillna("__MISSING__")
        return model.predict(query).reshape(-1) if problem_type == "regression" else model.predict_proba(query)

    return _shared_query_predictions(model, X_queries, predict, fit_seconds)


def _fit_lightgbm(
    problem_type: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_query: pd.DataFrame,
    categorical_columns: list[str],
    seed: int,
) -> FitPrediction:
    import lightgbm as lgb

    train, query = _encode_for_tree(X_train, X_query, categorical_columns)
    cls = lgb.LGBMRegressor if problem_type == "regression" else lgb.LGBMClassifier
    model = cls(
        n_estimators=400,
        learning_rate=0.04,
        num_leaves=31,
        random_state=seed,
        n_jobs=8,
        verbosity=-1,
    )
    started = time.perf_counter()
    model.fit(train, y_train)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    prediction = model.predict(query) if problem_type == "regression" else model.predict_proba(query)
    predict_seconds = time.perf_counter() - started
    return FitPrediction(np.asarray(prediction), {"fit_seconds": fit_seconds, "predict_seconds": predict_seconds})


def _fit_lightgbm_many(
    problem_type: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_queries: dict[str, pd.DataFrame],
    categorical_columns: list[str],
    seed: int,
) -> dict[str, FitPrediction]:
    import lightgbm as lgb

    first_query = next(iter(X_queries.values()))
    train, _ = _encode_for_tree(X_train, first_query, categorical_columns)
    cls = lgb.LGBMRegressor if problem_type == "regression" else lgb.LGBMClassifier
    model = cls(
        n_estimators=400,
        learning_rate=0.04,
        num_leaves=31,
        random_state=seed,
        n_jobs=8,
        verbosity=-1,
    )
    started = time.perf_counter()
    model.fit(train, y_train)
    fit_seconds = time.perf_counter() - started

    def predict(query_frame: pd.DataFrame) -> np.ndarray:
        _, query = _encode_for_tree(X_train, query_frame, categorical_columns)
        return model.predict(query) if problem_type == "regression" else model.predict_proba(query)

    return _shared_query_predictions(model, X_queries, predict, fit_seconds)


def _fit_sklearn_many(
    name: str,
    problem_type: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_queries: dict[str, pd.DataFrame],
    categorical_columns: list[str],
    seed: int,
) -> dict[str, FitPrediction]:
    """Fit conventional controls with training-fitted, explicit preprocessing."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric_columns = [column for column in X_train.columns if column not in set(categorical_columns)]
    numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    categorical = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=True),
    )
    preprocessing = ColumnTransformer(
        [("numeric", numeric, numeric_columns), ("categorical", categorical, categorical_columns)],
        remainder="drop",
    )
    if name == "random_forest":
        estimator = (
            RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=seed, n_jobs=8)
            if problem_type == "regression"
            else RandomForestClassifier(n_estimators=500, min_samples_leaf=2, random_state=seed, n_jobs=8)
        )
        # Scaling is harmless for trees and keeps missing/categorical handling identical across controls.
    elif name == "linear":
        estimator = Ridge(alpha=1.0) if problem_type == "regression" else LogisticRegression(max_iter=2_000, random_state=seed)
    else:
        raise ValueError(f"unknown sklearn control: {name}")
    model = make_pipeline(preprocessing, estimator)
    started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started

    def predict(query_frame: pd.DataFrame) -> np.ndarray:
        return model.predict(query_frame) if problem_type == "regression" else model.predict_proba(query_frame)

    return _shared_query_predictions(
        model,
        X_queries,
        predict,
        fit_seconds,
        {
            "preprocessing_policy": "train_median_standardize_and_train_one_hot",
            "estimator_policy": "fixed_phase2_control",
        },
    )


def _fit_pytabkit_many(
    name: str,
    problem_type: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_queries: dict[str, pd.DataFrame],
    seed: int,
    device: str,
) -> dict[str, FitPrediction]:
    """Run pytabkit out of process to isolate Lightning teardown state."""
    if not X_queries or any(not key.replace("_", "").isalnum() for key in X_queries):
        raise ValueError(f"invalid pytabkit query names: {sorted(X_queries)!r}")
    PYTABKIT_JOB_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pytabkit_", dir=PYTABKIT_JOB_ROOT) as temporary:
        job_dir = Path(temporary)
        X_train.to_parquet(job_dir / "train.parquet", index=False)
        for key, query in X_queries.items():
            query.to_parquet(job_dir / f"query__{key}.parquet", index=False)
        np.save(job_dir / "y_train.npy", np.asarray(y_train), allow_pickle=False)
        request = {
            "model": name,
            "problem_type": problem_type,
            "seed": int(seed),
            "device": device,
            "query_names": list(X_queries),
        }
        request_path = job_dir / "request.json"
        request_path.write_text(json.dumps(request, sort_keys=True) + "\n")
        started = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, str(PYTABKIT_WORKER), "--request", str(request_path)],
            capture_output=True,
            text=True,
        )
        subprocess_seconds = time.perf_counter() - started
        artifacts_complete = (job_dir / "telemetry.json").exists() and all(
            (job_dir / f"prediction__{key}.npy").exists() for key in X_queries
        )
        # RealMLP/pytabkit 1.7.3 can segfault during interpreter teardown after
        # successful fit, prediction, and complete artifact writes. Keep that
        # teardown defect isolated and accept only a fully materialized job.
        if completed.returncode and not (completed.returncode == -11 and artifacts_complete):
            output = "\n".join((completed.stdout + "\n" + completed.stderr).splitlines()[-80:])
            raise RuntimeError(f"isolated pytabkit worker failed with exit {completed.returncode}:\n{output}")
        telemetry = json.loads((job_dir / "telemetry.json").read_text())
        telemetry["subprocess_seconds"] = subprocess_seconds
        telemetry["worker_exit_code"] = completed.returncode
        predict_seconds = telemetry.pop("predict_seconds_by_query")
        outcomes = {}
        for key in X_queries:
            prediction = np.load(job_dir / f"prediction__{key}.npy", allow_pickle=False)
            outcomes[key] = FitPrediction(
                prediction,
                {**telemetry, "predict_seconds": predict_seconds[key], "shared_fit_id": job_dir.name},
            )
        return outcomes


def _fit_mitra(
    name: str,
    problem_type: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_query: pd.DataFrame,
    seed: int,
    device: str,
) -> FitPrediction:
    return _fit_mitra_many(
        name,
        problem_type,
        X_train,
        y_train,
        {"query": X_query},
        seed,
        device,
    )["query"]


def _fit_mitra_many(
    name: str,
    problem_type: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_queries: dict[str, pd.DataFrame],
    seed: int,
    device: str,
) -> dict[str, FitPrediction]:
    """Delegate Mitra to its isolated AutoGluon environment."""
    if not X_queries or any(not key.replace("_", "").isalnum() for key in X_queries):
        raise ValueError(f"invalid Mitra query names: {sorted(X_queries)!r}")
    MITRA_JOB_ROOT.mkdir(parents=True, exist_ok=True)
    MITRA_HF_HOME.mkdir(parents=True, exist_ok=True)
    fine_tune = name in {"mitra", "mitra_default"}
    with tempfile.TemporaryDirectory(prefix="mitra_", dir=MITRA_JOB_ROOT) as temporary:
        job_dir = Path(temporary)
        X_train.to_parquet(job_dir / "train.parquet", index=False)
        for key, query in X_queries.items():
            query.to_parquet(job_dir / f"query__{key}.parquet", index=False)
        np.save(job_dir / "y_train.npy", np.asarray(y_train), allow_pickle=False)
        request = {
            "problem_type": problem_type,
            "seed": int(seed),
            "fine_tune": fine_tune,
            "num_cpus": 8,
            "query_names": list(X_queries),
        }
        (job_dir / "request.json").write_text(json.dumps(request, sort_keys=True) + "\n")
        environment = os.environ.copy()
        environment["HF_HOME"] = str(MITRA_HF_HOME)
        environment["PYTHONHASHSEED"] = str(seed)
        if device.startswith("cuda:"):
            environment["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
        started = time.perf_counter()
        completed = subprocess.run(
            [str(MITRA_PYTHON), str(MITRA_WORKER), "--request", str(job_dir / "request.json")],
            capture_output=True,
            text=True,
            env=environment,
        )
        subprocess_seconds = time.perf_counter() - started
        if completed.returncode:
            output = "\n".join((completed.stdout + "\n" + completed.stderr).splitlines()[-80:])
            raise RuntimeError(f"isolated Mitra worker failed with exit {completed.returncode}:\n{output}")
        telemetry = json.loads((job_dir / "telemetry.json").read_text())
        telemetry["subprocess_seconds"] = subprocess_seconds
        telemetry["adapter"] = "isolated_autogluon_worker"
        predict_seconds = telemetry.pop("predict_seconds_by_query")
        outcomes = {}
        for key in X_queries:
            prediction = np.load(job_dir / f"prediction__{key}.npy", allow_pickle=False)
            query_telemetry = {**telemetry, "predict_seconds": predict_seconds[key], "shared_fit_id": str(job_dir.name)}
            outcomes[key] = FitPrediction(np.asarray(prediction), query_telemetry)
        return outcomes


def _tabpfn_inference_config(problem_type: str, minimal: bool) -> Any:
    if not minimal:
        return None
    from dataclasses import asdict
    from tabpfn.constants import ModelVersion
    from tabpfn.inference_config import InferenceConfig

    task = "regression" if problem_type == "regression" else "multiclass"
    config = asdict(InferenceConfig.get_default(task, ModelVersion.V2_5))
    config["PREPROCESS_TRANSFORMS"] = [
        {
            "name": "none",
            "categorical_name": "ordinal_very_common_categories_shuffled",
            "append_original": False,
            "max_features_per_estimator": 500,
            "global_transformer_name": None,
            "differentiable": False,
        }
    ]
    config["FEATURE_SHIFT_METHOD"] = None
    config["CLASS_SHIFT_METHOD"] = None
    if problem_type == "regression":
        config["REGRESSION_Y_PREPROCESS_TRANSFORMS"] = (None,)
    return config


def _fit_tabpfn(
    name: str,
    problem_type: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_query: pd.DataFrame,
    categorical_indices: list[int],
    seed: int,
    device: str,
) -> FitPrediction:
    from tabpfn import TabPFNClassifier, TabPFNRegressor

    minimal = name.endswith("single")
    checkpoint = TABPFN_REGRESSOR if problem_type == "regression" else TABPFN_CLASSIFIER
    cls = TabPFNRegressor if problem_type == "regression" else TabPFNClassifier
    model = cls(
        n_estimators=1 if minimal else 8,
        categorical_features_indices=categorical_indices or None,
        model_path=checkpoint,
        device=device,
        random_state=seed,
        inference_precision="autocast",
        fit_mode="fit_preprocessors",
        inference_config=_tabpfn_inference_config(problem_type, minimal),
    )
    started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    prediction = model.predict(X_query) if problem_type == "regression" else model.predict_proba(X_query)
    predict_seconds = time.perf_counter() - started
    telemetry = {
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "n_estimators": 1 if minimal else 8,
        "preprocessing_policy": "none_single" if minimal else "recommended_default",
        **_checkpoint_metadata(checkpoint),
    }
    del model
    gc.collect()
    return FitPrediction(np.asarray(prediction), telemetry)


def _fit_tabpfn_many(
    name: str,
    problem_type: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_queries: dict[str, pd.DataFrame],
    categorical_indices: list[int],
    seed: int,
    device: str,
) -> dict[str, FitPrediction]:
    from tabpfn import TabPFNClassifier, TabPFNRegressor

    minimal = name.endswith("single")
    checkpoint = TABPFN_REGRESSOR if problem_type == "regression" else TABPFN_CLASSIFIER
    cls = TabPFNRegressor if problem_type == "regression" else TabPFNClassifier
    model = cls(
        n_estimators=1 if minimal else 8,
        categorical_features_indices=categorical_indices or None,
        model_path=checkpoint,
        device=device,
        random_state=seed,
        inference_precision="autocast",
        fit_mode="fit_preprocessors",
        inference_config=_tabpfn_inference_config(problem_type, minimal),
    )
    started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started

    def predict(query_frame: pd.DataFrame) -> np.ndarray:
        return model.predict(query_frame) if problem_type == "regression" else model.predict_proba(query_frame)

    outcomes = _shared_query_predictions(
        model,
        X_queries,
        predict,
        fit_seconds,
        {
            "n_estimators": 1 if minimal else 8,
            "preprocessing_policy": "none_single" if minimal else "recommended_default",
            **_checkpoint_metadata(checkpoint),
        },
    )
    del model
    gc.collect()
    return outcomes


def _fit_tabicl(
    name: str,
    problem_type: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_query: pd.DataFrame,
    seed: int,
    device: str,
) -> FitPrediction:
    from tabicl import TabICLClassifier, TabICLRegressor

    minimal = name.endswith("single")
    cls = TabICLRegressor if problem_type == "regression" else TabICLClassifier
    kwargs: dict[str, Any] = {
        "n_estimators": 1 if minimal else 8,
        "norm_methods": "none" if minimal else None,
        "device": device,
        "use_amp": True,
        "random_state": seed,
        "batch_size": 1 if minimal else 8,
        "allow_auto_download": True,
    }
    if problem_type != "regression" and minimal:
        kwargs.update({"feat_shuffle_method": "none", "class_shuffle_method": "none"})
    elif minimal:
        kwargs.update({"feat_shuffle_method": "none"})
    model = cls(**kwargs)
    started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    prediction = model.predict(X_query) if problem_type == "regression" else model.predict_proba(X_query)
    predict_seconds = time.perf_counter() - started
    model_path = Path(model.model_path_)
    telemetry = {
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "n_estimators": 1 if minimal else 8,
        "preprocessing_policy": "none_single" if minimal else "recommended_default",
        "checkpoint_version": model.checkpoint_version,
        **_checkpoint_metadata(model_path),
    }
    del model
    gc.collect()
    return FitPrediction(np.asarray(prediction), telemetry)


def _fit_tabicl_many(
    name: str,
    problem_type: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_queries: dict[str, pd.DataFrame],
    seed: int,
    device: str,
) -> dict[str, FitPrediction]:
    from tabicl import TabICLClassifier, TabICLRegressor

    minimal = name.endswith("single")
    cls = TabICLRegressor if problem_type == "regression" else TabICLClassifier
    kwargs: dict[str, Any] = {
        "n_estimators": 1 if minimal else 8,
        "norm_methods": "none" if minimal else None,
        "device": device,
        "use_amp": True,
        "random_state": seed,
        "batch_size": 1 if minimal else 8,
        "allow_auto_download": True,
    }
    if problem_type != "regression" and minimal:
        kwargs.update({"feat_shuffle_method": "none", "class_shuffle_method": "none"})
    elif minimal:
        kwargs.update({"feat_shuffle_method": "none"})
    model = cls(**kwargs)
    started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started
    model_path = Path(model.model_path_)

    def predict(query_frame: pd.DataFrame) -> np.ndarray:
        return model.predict(query_frame) if problem_type == "regression" else model.predict_proba(query_frame)

    outcomes = _shared_query_predictions(
        model,
        X_queries,
        predict,
        fit_seconds,
        {
            "n_estimators": 1 if minimal else 8,
            "preprocessing_policy": "none_single" if minimal else "recommended_default",
            "checkpoint_version": model.checkpoint_version,
            **_checkpoint_metadata(model_path),
        },
    )
    del model
    gc.collect()
    return outcomes


def fit_predict(
    name: str,
    problem_type: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_query: pd.DataFrame,
    *,
    categorical_columns: list[str],
    categorical_indices: list[int],
    seed: int,
    device: str,
) -> FitPrediction:
    """Fit/in-context-condition on train and predict a query with explicit settings."""
    return fit_predict_many(
        name,
        problem_type,
        X_train,
        y_train,
        {"query": X_query},
        categorical_columns=categorical_columns,
        categorical_indices=categorical_indices,
        seed=seed,
        device=device,
    )["query"]


def fit_predict_many(
    name: str,
    problem_type: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_queries: dict[str, pd.DataFrame],
    *,
    categorical_columns: list[str],
    categorical_indices: list[int],
    seed: int,
    device: str,
) -> dict[str, FitPrediction]:
    """Fit once and predict several query representations when supported."""
    if not X_queries:
        raise ValueError("at least one query representation is required")
    if name == "xgboost":
        return _fit_xgboost_many(problem_type, X_train, y_train, X_queries, categorical_columns, seed)
    if name == "catboost":
        return _fit_catboost_many(problem_type, X_train, y_train, X_queries, categorical_columns, seed)
    if name == "lightgbm":
        return _fit_lightgbm_many(problem_type, X_train, y_train, X_queries, categorical_columns, seed)
    if name in {"random_forest", "linear"}:
        return _fit_sklearn_many(name, problem_type, X_train, y_train, X_queries, categorical_columns, seed)
    if name in {"tabm_default", "realmlp_default"}:
        return _fit_pytabkit_many(name, problem_type, X_train, y_train, X_queries, seed, device)
    if name in {"mitra", "mitra_default", "mitra_icl_single"}:
        return _fit_mitra_many(name, problem_type, X_train, y_train, X_queries, seed, device)
    if name.startswith("tabpfn_v25"):
        return _fit_tabpfn_many(name, problem_type, X_train, y_train, X_queries, categorical_indices, seed, device)
    if name.startswith("tabicl_v2"):
        return _fit_tabicl_many(name, problem_type, X_train, y_train, X_queries, seed, device)
    raise ValueError(f"unknown model: {name}")
