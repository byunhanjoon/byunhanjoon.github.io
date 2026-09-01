"""Shared, deterministic utilities for the ICLR 2027 projectivity benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.special import ndtr


HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text())
CACHE = Path(CONFIG["cache_root"])
EPS = 1e-10


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    source_id: str
    target: str
    categorical: list[bool]


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_array(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    return sha256_bytes(arr.view(np.uint8).tobytes())


def slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def configure_openml() -> None:
    import openml

    directory = CACHE / "openml"
    directory.mkdir(parents=True, exist_ok=True)
    openml.config.set_root_cache_directory(directory)


def _clean_dataset(
    name: str,
    X: pd.DataFrame,
    y: Iterable[Any],
    source_id: str,
    target: str,
    categorical: Iterable[bool] | None,
) -> Dataset:
    frame = X.copy().reset_index(drop=True)
    target_values = pd.to_numeric(pd.Series(np.asarray(y)), errors="coerce").to_numpy(dtype=np.float64)
    keep = np.isfinite(target_values)
    frame = frame.loc[keep].reset_index(drop=True)
    target_values = target_values[keep]
    if categorical is None:
        cat = [not pd.api.types.is_numeric_dtype(frame[column]) for column in frame]
    else:
        cat = list(categorical)
    if len(cat) != frame.shape[1]:
        raise ValueError(f"categorical metadata mismatch for {name}")
    for column, is_cat in zip(frame, cat):
        if is_cat:
            frame[column] = frame[column].astype("category")
    if len(frame) < 400 or frame.shape[1] < 1:
        raise ValueError(f"invalid dataset {name}: {frame.shape}")
    return Dataset(name, frame, target_values, source_id, target, cat)


def load_openml_task(task_id: int) -> tuple[Dataset, Any]:
    configure_openml()
    import openml

    task = openml.tasks.get_task(int(task_id), download_splits=True)
    dataset = openml.datasets.get_dataset(task.dataset_id, download_data=True)
    X, y, categorical, _ = dataset.get_data(
        target=task.target_name, dataset_format="dataframe"
    )
    clean = _clean_dataset(
        dataset.name,
        X,
        y,
        source_id=f"openml-task-{task_id}-dataset-{task.dataset_id}-v{dataset.version}",
        target=task.target_name,
        categorical=categorical,
    )
    if len(clean.y) != len(y):
        raise ValueError(
            f"task {task_id} has missing/non-numeric targets; official split indices would change"
        )
    return clean, task


def load_openml_dataset(spec: dict[str, Any]) -> Dataset:
    configure_openml()
    import openml

    dataset = openml.datasets.get_dataset(int(spec["openml_id"]), download_data=True)
    target = dataset.default_target_attribute
    if target is None:
        raise ValueError(f"no default target for {dataset.name}")
    X, y, categorical, _ = dataset.get_data(target=target, dataset_format="dataframe")
    return _clean_dataset(
        spec["name"],
        X,
        y,
        source_id=f"openml-dataset-{dataset.dataset_id}-v{dataset.version}",
        target=target,
        categorical=categorical,
    )


def local_data_root() -> Path:
    config = json.loads((HERE.parent / "final_closure" / "final_closure_config.json").read_text())
    return Path(config["data_root"])


def load_local_dataset(spec: dict[str, Any]) -> Dataset:
    root = local_data_root() / spec["name"]
    X = np.concatenate([np.load(root / f"N_{part}.npy") for part in ("train", "val", "test")])
    y = np.concatenate([np.load(root / f"y_{part}.npy") for part in ("train", "val", "test")])
    frame = pd.DataFrame(X, columns=[f"x{j}" for j in range(X.shape[1])])
    return _clean_dataset(spec["name"], frame, y, f"local:{root}", "y", [False] * X.shape[1])


def load_spec(spec: dict[str, Any]) -> Dataset:
    return load_local_dataset(spec) if spec.get("source") == "local" else load_openml_dataset(spec)


def metric_affine(y_train_pool: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(y_train_pool))
    scale = float(np.std(y_train_pool))
    if not np.isfinite(scale) or scale < EPS:
        scale = 1.0
    return mean, scale


def native_frame(frame: pd.DataFrame, indices: np.ndarray) -> pd.DataFrame:
    """Select rows without losing pandas categorical metadata."""
    return frame.iloc[np.asarray(indices, dtype=np.int64)].reset_index(drop=True)


def numeric_encode(
    context: pd.DataFrame, query: pd.DataFrame, categorical: list[bool]
) -> tuple[np.ndarray, np.ndarray]:
    """Context-only numeric encoding used by classical baselines and raw kernels."""
    train_cols: list[np.ndarray] = []
    test_cols: list[np.ndarray] = []
    for column, is_cat in zip(context.columns, categorical):
        if is_cat:
            values = context[column].astype("string").fillna("<NA>")
            categories = {value: i for i, value in enumerate(pd.unique(values))}
            train = values.map(categories).fillna(-1).to_numpy(dtype=np.float64)
            test = (
                query[column].astype("string").fillna("<NA>").map(categories).fillna(-1).to_numpy(dtype=np.float64)
            )
        else:
            train = pd.to_numeric(context[column], errors="coerce").to_numpy(dtype=np.float64)
            test = pd.to_numeric(query[column], errors="coerce").to_numpy(dtype=np.float64)
            finite = np.isfinite(train)
            median = float(np.median(train[finite])) if finite.any() else 0.0
            train = np.where(np.isfinite(train), train, median)
            test = np.where(np.isfinite(test), test, median)
        center = float(np.mean(train))
        scale = float(np.std(train))
        if not np.isfinite(scale) or scale < 1e-7:
            scale = 1.0
        train_cols.append(np.clip((train - center) / scale, -10.0, 10.0))
        test_cols.append(np.clip((test - center) / scale, -10.0, 10.0))
    return (
        np.stack(train_cols, axis=1).astype(np.float32),
        np.stack(test_cols, axis=1).astype(np.float32),
    )


def make_coefficients(seed: int, groups: int | None = None, q: int | None = None) -> tuple[list[str], np.ndarray]:
    groups = int(groups or CONFIG["query_groups"])
    q = int(q or CONFIG["query_size"])
    n = groups * q
    rng = np.random.default_rng(seed)
    result: dict[str, np.ndarray] = {}

    point = np.zeros((groups, n), dtype=np.float64)
    for group in range(groups):
        point[group, group * q + int(rng.integers(q))] = 1.0
    result["point"] = point

    subset_mean = np.zeros((groups, n), dtype=np.float64)
    subset_total = np.zeros((groups, n), dtype=np.float64)
    pair = np.zeros((groups, n), dtype=np.float64)
    contrast = np.zeros((groups, n), dtype=np.float64)
    dense_signed = np.zeros((groups, n), dtype=np.float64)
    dense_positive = np.zeros((groups, n), dtype=np.float64)
    scaled = np.zeros((groups, n), dtype=np.float64)

    for group in range(groups):
        offset = group * q
        k = int(rng.integers(2, q))
        subset = rng.choice(q, size=k, replace=False)
        subset_mean[group, offset + subset] = 1.0 / k
        subset_total[group, offset + subset] = 1.0 / math.sqrt(k)

        i, j = rng.choice(q, size=2, replace=False)
        pair[group, offset + i] = 1.0 / math.sqrt(2.0)
        pair[group, offset + j] = -1.0 / math.sqrt(2.0)

        perm = rng.permutation(q)
        k_group = q // 2
        contrast[group, offset + perm[:k_group]] = 1.0 / math.sqrt(2.0 * k_group)
        contrast[group, offset + perm[k_group : 2 * k_group]] = -1.0 / math.sqrt(2.0 * k_group)

        signed = rng.normal(size=q)
        signed /= np.linalg.norm(signed)
        dense_signed[group, offset : offset + q] = signed
        positive = np.abs(rng.normal(size=q))
        positive /= np.linalg.norm(positive)
        dense_positive[group, offset : offset + q] = positive
        scaled[group, offset : offset + q] = signed * float(rng.uniform(1.75, 3.0))

    result["subset_mean"] = subset_mean
    result["subset_total_l2"] = subset_total
    result["pair_difference"] = pair
    result["group_contrast"] = contrast
    result["dense_signed"] = dense_signed
    result["dense_positive"] = dense_positive
    result["scaled_dense"] = scaled
    families = list(CONFIG["query_families"])
    return families, np.stack([result[family] for family in families]).astype(np.float32)


def gaussian_scores(target: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> dict[str, np.ndarray]:
    variance = np.maximum(np.asarray(variance, dtype=np.float64), 1e-9)
    target = np.asarray(target, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    sd = np.sqrt(variance)
    z = (target - mean) / sd
    nll = 0.5 * (np.log(2.0 * np.pi * variance) + z**2)
    crps = sd * (z * (2.0 * ndtr(z) - 1.0) + 2.0 * np.exp(-0.5 * z**2) / math.sqrt(2.0 * np.pi) - 1.0 / math.sqrt(np.pi))
    return {
        "nll": nll,
        "crps": crps,
        "squared_error": (target - mean) ** 2,
        "abs_error": np.abs(target - mean),
        "z_abs": np.abs(z),
        "sd": sd,
    }


def atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name, suffix=".npz", delete=False) as handle:
        temp = Path(handle.name)
    try:
        np.savez_compressed(temp, **arrays)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name, suffix=".json", delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp = Path(handle.name)
    os.replace(temp, path)


def environment_record() -> dict[str, Any]:
    import scipy
    import sklearn
    import torch
    import tabicl

    try:
        import tabpfn
        tabpfn_version = tabpfn.__version__
    except Exception as exc:  # pragma: no cover - diagnostic only
        tabpfn_version = f"unavailable:{exc}"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "tabicl": getattr(tabicl, "__version__", "unknown"),
        "tabpfn": tabpfn_version,
        "protocol_sha256": sha256_bytes((HERE / "PROTOCOL.md").read_bytes()),
        "config_sha256": sha256_bytes((HERE / "config.json").read_bytes()),
    }
