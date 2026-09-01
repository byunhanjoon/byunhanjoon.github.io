"""Prospective compatibility x candidate-reliability pilot.

The protocol is frozen in PROSPECTIVE_RISK_PROTOCOL.md.  This file deliberately
keeps the Day-8 compact models fixed and changes only candidate scoring at
inference.  Candidate uncertainty is estimated from training-only OOF outputs.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import openml
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch import Tensor, nn

from day8_core import (
    ArrayData,
    ModernNCAModel,
    TabRModel,
    build_model,
    evaluate_predictions,
    make_synthetic,
    predict_model,
    seed_all,
    train_model,
)


HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
PROCESSED = RAW / "prospective_processed"
REAL_OUTPUT = RAW / "prospective_risk"
SYNTH_OUTPUT = RAW / "prospective_risk_synthetic"
CHECKPOINTS = RAW / "prospective_risk_checkpoints"

DATASETS: tuple[tuple[str, int, str], ...] = (
    ("bank-marketing", 1461, "classification"),
    ("credit-g", 31, "classification"),
    ("electricity", 151, "classification"),
    ("jannis", 41168, "classification"),
    ("covertype", 1596, "classification"),
    ("MagicTelescope", 1120, "classification"),
    ("abalone", 42726, "regression"),
    ("cpu_act", 573, "regression"),
    ("elevators", 216, "regression"),
    ("Bike_Sharing_Demand", 42712, "regression"),
    ("sulfur", 23515, "regression"),
    ("superconduct", 43174, "regression"),
)
SPLIT_SEEDS = (20260901, 20260902, 20260903)
MODEL_SEEDS = (20260911, 20260912, 20260913)
SYNTH_SEEDS = tuple(range(20261001, 20261009))
LAMBDAS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)


def slug(value: str) -> str:
    return value.lower().replace("_", "-").replace(" ", "-")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True))
    os.replace(temp, path)


def capped(indices: np.ndarray, y: np.ndarray, cap: int, task: str, seed: int) -> np.ndarray:
    if len(indices) <= cap:
        return np.asarray(indices)
    stratify = y[indices] if task == "classification" else None
    chosen, _ = train_test_split(
        indices,
        train_size=cap,
        random_state=seed,
        stratify=stratify,
    )
    return np.asarray(chosen)


def processed_path(name: str, split_seed: int) -> Path:
    return PROCESSED / f"{slug(name)}__split-{split_seed}.npz"


def prepare_dataset(name: str, did: int, task: str, split_seed: int) -> Path:
    """Download by immutable OpenML ID, split, preprocess, and cache arrays."""

    path = processed_path(name, split_seed)
    if path.exists():
        return path
    dataset = openml.datasets.get_dataset(did, download_data=True)
    x, raw_y, categorical, attributes = dataset.get_data(
        dataset_format="dataframe",
        target=dataset.default_target_attribute,
    )
    if raw_y is None:
        raise RuntimeError(f"OpenML {did} has no default target")
    frame = pd.DataFrame(x).replace([np.inf, -np.inf], np.nan)
    raw_y = pd.Series(raw_y)
    valid = raw_y.notna().to_numpy()
    frame = frame.loc[valid].reset_index(drop=True)
    raw_y = raw_y.loc[valid].reset_index(drop=True)
    categorical = np.asarray(categorical, dtype=bool)

    if task == "classification":
        labels = LabelEncoder().fit_transform(raw_y.astype(str)).astype(np.int64)
    else:
        labels = pd.to_numeric(raw_y, errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(labels)
        frame = frame.loc[finite].reset_index(drop=True)
        labels = labels[finite]

    rows = np.arange(len(labels))
    stratify = labels if task == "classification" else None
    train_val, test = train_test_split(
        rows,
        test_size=0.2,
        random_state=split_seed,
        stratify=stratify,
    )
    stratify_tv = labels[train_val] if task == "classification" else None
    train, validation = train_test_split(
        train_val,
        test_size=0.25,
        random_state=split_seed + 1,
        stratify=stratify_tv,
    )
    train = capped(train, labels, 8192, task, split_seed + 11)
    validation = capped(validation, labels, 2048, task, split_seed + 12)
    test = capped(test, labels, 2048, task, split_seed + 13)
    parts = {"train": train, "validation": validation, "test": test}

    numeric_columns = [column for column, is_cat in zip(attributes, categorical) if not is_cat]
    categorical_columns = [column for column, is_cat in zip(attributes, categorical) if is_cat]
    if numeric_columns:
        numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        medians = np.nanmedian(numeric[train], axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        bad = ~np.isfinite(numeric)
        if bad.any():
            numeric[bad] = medians[np.where(bad)[1]]
        scaler = StandardScaler().fit(numeric[train])
        x_num = {
            part: np.ascontiguousarray(scaler.transform(numeric[index]), dtype=np.float32)
            for part, index in parts.items()
        }
    else:
        x_num = {part: np.empty((len(index), 0), dtype=np.float32) for part, index in parts.items()}

    if categorical_columns:
        categories = frame[categorical_columns].astype("string").fillna("__MISSING__")
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        encoder.fit(categories.iloc[train])
        x_cat = {
            part: np.ascontiguousarray(encoder.transform(categories.iloc[index]), dtype=np.float32)
            for part, index in parts.items()
        }
    else:
        x_cat = {part: np.empty((len(index), 0), dtype=np.float32) for part, index in parts.items()}

    if task == "regression":
        y_mean = float(labels[train].mean())
        y_scale = float(labels[train].std()) or 1.0
        y = {
            part: np.ascontiguousarray((labels[index] - y_mean) / y_scale, dtype=np.float32)
            for part, index in parts.items()
        }
        n_classes = 1
    else:
        y_mean, y_scale = 0.0, 1.0
        y = {part: np.ascontiguousarray(labels[index], dtype=np.int64) for part, index in parts.items()}
        n_classes = int(labels.max() + 1)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        name=np.asarray(name),
        did=np.asarray(did),
        task=np.asarray(task),
        n_classes=np.asarray(n_classes),
        y_mean=np.asarray(y_mean),
        y_scale=np.asarray(y_scale),
        attributes=np.asarray(attributes, dtype=str),
        **{f"x_num_{part}": value for part, value in x_num.items()},
        **{f"x_cat_{part}": value for part, value in x_cat.items()},
        **{f"y_{part}": value for part, value in y.items()},
    )
    atomic_json(
        path.with_suffix(".json"),
        {
            "status": "complete",
            "name": name,
            "openml_id": did,
            "openml_version": int(dataset.version),
            "openml_url": dataset.url,
            "license": dataset.licence,
            "task": task,
            "split_seed": split_seed,
            "rows_total": len(labels),
            "rows": {part: len(index) for part, index in parts.items()},
            "n_num": x_num["train"].shape[1],
            "n_cat_onehot": x_cat["train"].shape[1],
            "n_classes": n_classes,
        },
    )
    return path


def load_processed(name: str, split_seed: int) -> ArrayData:
    payload = np.load(processed_path(name, split_seed), allow_pickle=False)
    task = str(payload["task"])
    return ArrayData(
        name=str(payload["name"]),
        task=task,  # type: ignore[arg-type]
        n_classes=int(payload["n_classes"]),
        x_num={part: payload[f"x_num_{part}"] for part in ("train", "validation", "test")},
        x_cat={part: payload[f"x_cat_{part}"] for part in ("train", "validation", "test")},
        y={part: payload[f"y_{part}"] for part in ("train", "validation", "test")},
        y_mean=float(payload["y_mean"]),
        y_scale=float(payload["y_scale"]),
    )


def cross_fitted_proxy(data: ArrayData, seed: int, cache: Path | None = None) -> dict[str, np.ndarray]:
    if cache is not None and cache.exists():
        return dict(np.load(cache))
    x_train = np.concatenate((data.x_num["train"], data.x_cat["train"]), axis=1)
    x_validation = np.concatenate((data.x_num["validation"], data.x_cat["validation"]), axis=1)
    x_test = np.concatenate((data.x_num["test"], data.x_cat["test"]), axis=1)
    y = data.y["train"]
    workers = min(4, max(1, (os.cpu_count() or 2) // 4))
    if data.task == "regression":
        mean_model = ExtraTreesRegressor(
            n_estimators=200,
            min_samples_leaf=8,
            max_features=0.8,
            n_jobs=workers,
            random_state=seed,
        )
        mean_cv = KFold(5, shuffle=True, random_state=seed)
        m_train = cross_val_predict(mean_model, x_train, y, cv=mean_cv, n_jobs=1, method="predict")
        mean_model.fit(x_train, y)
        m_validation = mean_model.predict(x_validation)
        m_test = mean_model.predict(x_test)
        residual2 = np.square(y - m_train)
        noise_model = ExtraTreesRegressor(
            n_estimators=160,
            min_samples_leaf=20,
            max_features=0.8,
            n_jobs=workers,
            random_state=seed + 1,
        )
        noise_cv = KFold(5, shuffle=True, random_state=seed + 2)
        sigma_train = np.maximum(
            cross_val_predict(noise_model, x_train, residual2, cv=noise_cv, n_jobs=1, method="predict"),
            1e-7,
        )
    else:
        mean_model = ExtraTreesClassifier(
            n_estimators=200,
            min_samples_leaf=8,
            max_features=0.8,
            n_jobs=workers,
            random_state=seed,
        )
        mean_cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        m_train = cross_val_predict(mean_model, x_train, y, cv=mean_cv, n_jobs=1, method="predict_proba")
        mean_model.fit(x_train, y)
        m_validation = mean_model.predict_proba(x_validation)
        m_test = mean_model.predict_proba(x_test)
        if m_train.shape[1] != data.n_classes:
            raise AssertionError("OOF probability width does not match n_classes")
        sigma_train = np.maximum(1.0 - np.square(m_train).sum(axis=1), 1e-7)
    output = {
        "m_train": np.asarray(m_train, dtype=np.float32),
        "m_validation": np.asarray(m_validation, dtype=np.float32),
        "m_test": np.asarray(m_test, dtype=np.float32),
        "sigma_train": np.asarray(sigma_train, dtype=np.float32),
    }
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **output)
    return output


def rank01(cost: np.ndarray) -> np.ndarray:
    cost = np.asarray(cost, dtype=np.float64)
    if len(cost) <= 1:
        return np.zeros_like(cost, dtype=np.float32)
    return np.asarray((rankdata(cost, method="average") - 1.0) / (len(cost) - 1.0), dtype=np.float32)


def squared_distances(q: Tensor, c: Tensor) -> Tensor:
    qf, cf = q.float(), c.float()
    return (
        qf.square().sum(dim=1, keepdim=True)
        - 2.0 * qf @ cf.T
        + cf.square().sum(dim=1)[None]
    ).clamp_min(0.0)


@torch.no_grad()
def predict_with_cost(
    model: nn.Module,
    data: ArrayData,
    part: str,
    device: torch.device,
    candidate_cost: np.ndarray,
    lambda_value: float,
    query_limit: int | None = None,
    return_full_score: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Predict after adding the directed candidate term to retrieval scores."""

    if not isinstance(model, (TabRModel, ModernNCAModel)):
        raise TypeError(type(model))
    model.eval()
    q_num = torch.as_tensor(data.x_num[part], device=device)
    q_cat = torch.as_tensor(data.x_cat[part], device=device)
    if query_limit is not None:
        q_num, q_cat = q_num[:query_limit], q_cat[:query_limit]
    c_num = torch.as_tensor(data.x_num["train"], device=device)
    c_cat = torch.as_tensor(data.x_cat["train"], device=device)
    c_y = torch.as_tensor(data.y["train"], device=device)
    cost = torch.as_tensor(candidate_cost, device=device, dtype=torch.float32)
    if len(cost) != len(c_y):
        raise ValueError("candidate cost length mismatch")
    c_key = model.keys(c_num, c_cat)
    outputs: list[Tensor] = []
    indices: list[Tensor] = []
    scores: list[Tensor] = []
    for start in range(0, len(q_num), 256):
        qn, qc = q_num[start : start + 256], q_cat[start : start + 256]
        q_key = model.keys(qn, qc)
        distance = squared_distances(q_key, c_key)
        scale = torch.quantile(distance, 0.5, dim=1, keepdim=True).clamp_min(1e-6)
        score = distance + float(lambda_value) * scale * cost[None]
        idx = torch.topk(score, k=min(16, len(c_y)), largest=False).indices
        if isinstance(model, TabRModel):
            selected_key = c_key[idx]
            selected_y = c_y[idx]
            selected_score = torch.gather(score, 1, idx)
            attention = torch.softmax(-selected_score / math.sqrt(model.width), dim=1)
            if model.task == "regression":
                label = model.label(selected_y.float()[..., None])
            else:
                label = model.label(selected_y.long())
            values = label + model.value(q_key[:, None] - selected_key)
            mixed = model.pred(qn, qc) + torch.bmm(attention[:, None], values).squeeze(1)
            output = model.head(mixed)
        else:
            temperature = model.log_temperature.exp().clamp(0.05, 5.0)
            weights = torch.softmax(-score / temperature, dim=1)
            if model.task == "regression":
                output = weights @ c_y.float()[:, None]
            else:
                output = weights @ nn.functional.one_hot(c_y.long(), model.n_classes).float()
        outputs.append(output.cpu())
        indices.append(idx.cpu())
        if return_full_score:
            scores.append(score.cpu())
    raw = torch.cat(outputs).numpy()
    if data.task == "classification" and isinstance(model, TabRModel):
        raw = torch.softmax(torch.from_numpy(raw), dim=1).numpy()
    return raw, torch.cat(indices).numpy(), torch.cat(scores).numpy() if scores else None


def mismatch_vector(m_train: np.ndarray, m_query: np.ndarray) -> np.ndarray:
    if m_train.ndim == 1:
        return np.square(m_train - float(m_query))
    return np.square(m_train - m_query[None]).sum(axis=1)


def mechanism_diagnostics(
    model: nn.Module,
    data: ArrayData,
    device: torch.device,
    candidate_cost: np.ndarray,
    lambda_value: float,
    proxy: dict[str, np.ndarray],
    part: str = "test",
    limit: int = 128,
) -> dict[str, float]:
    _, indices, scores = predict_with_cost(
        model,
        data,
        part,
        device,
        candidate_cost,
        lambda_value,
        query_limit=limit,
        return_full_score=True,
    )
    assert scores is not None
    m_train = proxy["m_train"]
    m_query = proxy[f"m_{part}"][: len(indices)]
    sigma = proxy["sigma_train"]
    rng = np.random.default_rng(20260931)
    rhos: list[float] = []
    risks: list[float] = []
    mismatches: list[float] = []
    noises: list[float] = []
    overlaps: list[float] = []
    for q, selected in enumerate(indices):
        mismatch = mismatch_vector(m_train, m_query[q])
        risk = mismatch + sigma
        sample = rng.choice(len(risk), size=min(512, len(risk)), replace=False)
        rho = spearmanr(scores[q, sample], risk[sample]).statistic
        rhos.append(float(rho) if np.isfinite(rho) else 0.0)
        risks.append(float(risk[selected].mean()))
        mismatches.append(float(mismatch[selected].mean()))
        noises.append(float(sigma[selected].mean()))
        oracle = np.argpartition(risk, min(len(selected) - 1, len(risk) - 1))[: len(selected)]
        overlaps.append(len(set(selected.tolist()) & set(oracle.tolist())) / len(selected))
    return {
        "risk_spearman": float(np.mean(rhos)),
        "topk_proxy_risk": float(np.mean(risks)),
        "topk_target_mismatch": float(np.mean(mismatches)),
        "topk_candidate_noise": float(np.mean(noises)),
        "oracle_topk_overlap": float(np.mean(overlaps)),
    }


def tune_method(
    model: nn.Module,
    data: ArrayData,
    device: torch.device,
    cost: np.ndarray,
    proxy: dict[str, np.ndarray],
    method: str,
    lambda_grid: tuple[float, ...] = LAMBDAS,
) -> dict[str, Any]:
    validation: list[dict[str, float]] = []
    for value in lambda_grid:
        pred, _, _ = predict_with_cost(model, data, "validation", device, cost, value)
        metrics = evaluate_predictions(pred, data.y["validation"], data)
        validation.append({"lambda": value, "loss": metrics["loss"], "metric": metrics["metric"]})
    chosen = min(validation, key=lambda row: (row["loss"], row["lambda"]))
    pred, _, _ = predict_with_cost(model, data, "test", device, cost, chosen["lambda"])
    test = evaluate_predictions(pred, data.y["test"], data)
    mechanism = mechanism_diagnostics(model, data, device, cost, chosen["lambda"], proxy)
    return {
        "method": method,
        "lambda": float(chosen["lambda"]),
        "validation_loss": float(chosen["loss"]),
        "validation_grid": validation,
        **test,
        **mechanism,
    }


def checkpoint_paths(dataset: str, split_seed: int, model_name: str, model_seed: int) -> tuple[Path, Path]:
    stem = f"{slug(dataset)}__split-{split_seed}__{model_name}__seed-{model_seed}"
    return CHECKPOINTS / f"{stem}.pt", CHECKPOINTS / f"{stem}.json"


def fit_or_load(
    data: ArrayData,
    dataset: str,
    split_seed: int,
    model_name: str,
    model_seed: int,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint, metadata = checkpoint_paths(dataset, split_seed, model_name, model_seed)
    model = build_model(data, model_name, "raw", "raw", "standard").to(device)
    if checkpoint.exists() and metadata.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        return model, json.loads(metadata.read_text())
    started = time.perf_counter()
    model, metrics = train_model(
        data,
        model_name,
        "raw",
        "raw",
        model_seed,
        device,
        "standard",
        max_epochs=48,
    )
    metrics = {**metrics, "fit_wall_seconds": time.perf_counter() - started}
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temp = checkpoint.with_suffix(f".tmp-{os.getpid()}.pt")
    torch.save(model.state_dict(), temp)
    os.replace(temp, checkpoint)
    atomic_json(metadata, metrics)
    return model, metrics


def real_cell(
    data: ArrayData,
    split_seed: int,
    model_name: str,
    model_seed: int,
    device: torch.device,
    proxy: dict[str, np.ndarray],
) -> dict[str, Any]:
    output = REAL_OUTPUT / (
        f"{slug(data.name)}__split-{split_seed}__{model_name}__seed-{model_seed}.json"
    )
    if output.exists():
        payload = json.loads(output.read_text())
        if payload.get("status") == "complete":
            return payload
    model, fit_metrics = fit_or_load(data, data.name, split_seed, model_name, model_seed, device)
    true_cost = rank01(proxy["sigma_train"])
    rng = np.random.default_rng(model_seed + split_seed)
    permuted_cost = true_cost[rng.permutation(len(true_cost))]
    rows = [
        tune_method(model, data, device, np.zeros_like(true_cost), proxy, "distance", (0.0,)),
        tune_method(model, data, device, true_cost, proxy, "oof_reliability"),
        tune_method(model, data, device, permuted_cost, proxy, "permuted_reliability"),
    ]
    payload = {
        "status": "complete",
        "dataset": data.name,
        "task": data.task,
        "split_seed": split_seed,
        "model": model_name,
        "model_seed": model_seed,
        "n_train": len(data.y["train"]),
        "n_validation": len(data.y["validation"]),
        "n_test": len(data.y["test"]),
        "n_num": data.n_num,
        "n_cat_onehot": data.n_cat,
        "uncertainty_mean": float(np.mean(proxy["sigma_train"])),
        "uncertainty_sd": float(np.std(proxy["sigma_train"])),
        "uncertainty_iqr": float(np.subtract(*np.percentile(proxy["sigma_train"], [75, 25]))),
        "fit": fit_metrics,
        "methods": rows,
    }
    atomic_json(output, payload)
    return payload


def run_prepare() -> None:
    for name, did, task in DATASETS:
        for split_seed in SPLIT_SEEDS:
            print(f"prepare {name} split={split_seed}", flush=True)
            prepare_dataset(name, did, task, split_seed)


def run_real(device: torch.device, shard: int, n_shards: int) -> None:
    for dataset_index, (name, _, _) in enumerate(DATASETS):
        if dataset_index % n_shards != shard:
            continue
        for split_seed in SPLIT_SEEDS:
            data = load_processed(name, split_seed)
            proxy_path = PROCESSED / f"{slug(name)}__split-{split_seed}__proxy.npz"
            print(f"real shard={shard} proxy {name} split={split_seed}", flush=True)
            proxy = cross_fitted_proxy(data, split_seed, proxy_path)
            for model_name in ("TabR", "ModernNCA"):
                for model_seed in MODEL_SEEDS:
                    print(
                        f"real shard={shard} {name} split={split_seed} model={model_name} seed={model_seed}",
                        flush=True,
                    )
                    real_cell(data, split_seed, model_name, model_seed, device, proxy)


def exact_synthetic_proxy(data: ArrayData, meta: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "m_train": np.asarray(meta["m"]["train"], dtype=np.float32),
        "m_validation": np.asarray(meta["m"]["validation"], dtype=np.float32),
        "m_test": np.asarray(meta["m"]["test"], dtype=np.float32),
        "sigma_train": np.square(np.asarray(meta["sigma"]["train"], dtype=np.float32)),
    }


def run_synthetic(device: torch.device, shard: int, n_shards: int) -> None:
    tasks = ("S1_rotating", "S2_global", "S3_noise", "S4_warp")
    cells = [(task, seed, model) for task in tasks for seed in SYNTH_SEEDS for model in ("TabR", "ModernNCA")]
    for cell_index, (task, seed, model_name) in enumerate(cells):
        if cell_index % n_shards != shard:
            continue
        output = SYNTH_OUTPUT / f"{task}__{model_name}__seed-{seed}.json"
        if output.exists() and json.loads(output.read_text()).get("status") == "complete":
            continue
        print(f"synthetic shard={shard} {task} {model_name} seed={seed}", flush=True)
        data, meta = make_synthetic(task, seed, n_train=4096, n_val=1024, n_test=1024)
        model, fit = train_model(
            data,
            model_name,
            "raw",
            "raw",
            seed,
            device,
            "standard",
            max_epochs=48,
        )
        exact = exact_synthetic_proxy(data, meta)
        estimated = cross_fitted_proxy(data, seed)
        exact_cost = rank01(exact["sigma_train"])
        estimated_cost = rank01(estimated["sigma_train"])
        rng = np.random.default_rng(seed + 71)
        permuted_cost = exact_cost[rng.permutation(len(exact_cost))]
        rows = [
            tune_method(model, data, device, np.zeros_like(exact_cost), exact, "distance", (0.0,)),
            tune_method(model, data, device, exact_cost, exact, "exact_reliability"),
            tune_method(model, data, device, estimated_cost, exact, "estimated_reliability"),
            tune_method(model, data, device, permuted_cost, exact, "permuted_exact_reliability"),
        ]
        atomic_json(
            output,
            {
                "status": "complete",
                "task": task,
                "seed": seed,
                "model": model_name,
                "n_train": len(data.y["train"]),
                "fit": fit,
                "estimated_exact_uncertainty_spearman": float(
                    spearmanr(estimated["sigma_train"], exact["sigma_train"]).statistic
                ),
                "methods": rows,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "real", "synthetic"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(2)
    random.seed(20260901)
    np.random.seed(20260901)
    if args.stage == "prepare":
        run_prepare()
    elif args.stage == "real":
        run_real(torch.device(args.device), args.shard, args.n_shards)
    else:
        run_synthetic(torch.device(args.device), args.shard, args.n_shards)


if __name__ == "__main__":
    main()
