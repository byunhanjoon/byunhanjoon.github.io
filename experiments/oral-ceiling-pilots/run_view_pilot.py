"""Pilot A: lossless-view regret for longitudinal prediction."""

from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.fft import dct, idct
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNRegressor


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "phasecover-confirmation" / "raw" / "data"
OUT = HERE / "view"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = (20261221, 20261222, 20261223)
DATASETS = ("JenaWeather", "Electricity", "Traffic")
VIEWS = ("levels", "differences", "dct", "reverse")
MODELS = ("lightgbm", "mlp", "tabpfn")
PROTOCOL_SHA256 = "538f14851b6a1cf54737c3b9bc8df3cf3b227c1b33cf2ef53469f261317b3164"


def make_examples(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    payload = np.load(SOURCE / f"{name}.npz")
    values = payload["values"]
    train_end = int(payload["train_end"])
    validation_end = int(payload["validation_end"])

    def build(first: int, last: int, n_starts: int) -> tuple[np.ndarray, np.ndarray]:
        available = np.arange(max(32, first), last)
        positions = np.rint(np.linspace(0, len(available) - 1, n_starts)).astype(np.int64)
        starts = available[positions]
        histories = np.stack([values[start - 32 : start] for start in starts])
        x = histories.transpose(0, 2, 1).reshape(-1, 32)
        y = values[starts].reshape(-1)
        return np.ascontiguousarray(x), np.ascontiguousarray(y)

    x_train, y_train = build(32, train_end, 512)
    x_test, y_test = build(validation_end, len(values), 256)
    return x_train, y_train, x_test, y_test


def transform(x: np.ndarray, view: str) -> np.ndarray:
    if view == "levels":
        return x.copy()
    if view == "differences":
        return np.concatenate([x[:, :1], np.diff(x, axis=1)], axis=1)
    if view == "dct":
        return dct(x, type=2, axis=1, norm="ortho")
    if view == "reverse":
        return x[:, ::-1].copy()
    raise ValueError(view)


def inverse(x: np.ndarray, view: str) -> np.ndarray:
    if view == "levels":
        return x.copy()
    if view == "differences":
        return np.cumsum(x, axis=1)
    if view == "dct":
        return idct(x, type=2, axis=1, norm="ortho")
    if view == "reverse":
        return x[:, ::-1].copy()
    raise ValueError(view)


def fit_predict(
    model_name: str,
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> np.ndarray:
    if model_name == "lightgbm":
        model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            n_jobs=4,
            verbosity=-1,
        )
    elif model_name == "mlp":
        model = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(128, 128),
                early_stopping=True,
                max_iter=200,
                batch_size=256,
                random_state=seed,
            ),
        )
    elif model_name == "tabpfn":
        model = TabPFNRegressor(
            n_estimators=4,
            device="cuda:0",
            random_state=seed,
            fit_mode="fit_preprocessors",
        )
    else:
        raise ValueError(model_name)
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_test), dtype=np.float32)


def main() -> None:
    started = time.perf_counter()
    rows = []
    predictions: dict[str, np.ndarray] = {}
    integrity: dict[str, float] = {}
    for dataset in DATASETS:
        x_train, y_train, x_test, y_test = make_examples(dataset)
        for view in VIEWS:
            train_view = transform(x_train, view)
            test_view = transform(x_test, view)
            integrity[f"{dataset}/{view}"] = float(
                max(np.max(np.abs(inverse(train_view, view) - x_train)), np.max(np.abs(inverse(test_view, view) - x_test)))
            )
            for model_name in MODELS:
                for seed in SEEDS:
                    cell_started = time.perf_counter()
                    prediction = fit_predict(model_name, seed, train_view, y_train, test_view)
                    if not np.isfinite(prediction).all():
                        raise FloatingPointError((dataset, view, model_name, seed))
                    key = f"{dataset}__{view}__{model_name}__{seed}"
                    predictions[key] = prediction
                    rows.append({
                        "dataset": dataset,
                        "view": view,
                        "model": model_name,
                        "seed": seed,
                        "rmse": float(np.sqrt(mean_squared_error(y_test, prediction))),
                        "fit_predict_seconds": time.perf_counter() - cell_started,
                    })
                    if time.perf_counter() - started > 30 * 60:
                        raise TimeoutError("view pilot exceeded 30 minutes")
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "cells.csv", index=False)
    np.savez_compressed(OUT / "predictions.npz", **predictions)
    summary_rows = []
    for dataset in DATASETS:
        _, _, _, y_test = make_examples(dataset)
        for model_name in MODELS:
            for seed in SEEDS:
                group = frame[(frame.dataset == dataset) & (frame.model == model_name) & (frame.seed == seed)]
                rmses = group.set_index("view").loc[list(VIEWS)].rmse.to_numpy()
                view_predictions = np.stack([
                    predictions[f"{dataset}__{view}__{model_name}__{seed}"] for view in VIEWS
                ])
                mean_prediction = view_predictions.mean(axis=0)
                dispersion = float(np.sqrt(np.mean(np.square(view_predictions - mean_prediction[None]))))
                canonical_rmse = float(group[group.view == "levels"].rmse.iloc[0])
                summary_rows.append({
                    "dataset": dataset,
                    "model": model_name,
                    "seed": seed,
                    "best_rmse": float(rmses.min()),
                    "worst_rmse": float(rmses.max()),
                    "relative_rmse_spread": float((rmses.max() - rmses.min()) / rmses.min()),
                    "prediction_dispersion": dispersion,
                    "relative_prediction_dispersion": dispersion / canonical_rmse,
                    "view_ensemble_rmse": float(np.sqrt(mean_squared_error(y_test, mean_prediction))),
                })
    by_seed = pd.DataFrame(summary_rows)
    by_seed.to_csv(OUT / "comparisons_by_seed.csv", index=False)
    summary = by_seed.groupby(["dataset", "model"], as_index=False).mean(numeric_only=True)
    summary.to_csv(OUT / "summary.csv", index=False)
    qualifying = summary[(summary.relative_rmse_spread >= 0.10) & (summary.relative_prediction_dispersion >= 0.10)]
    mean_rmse = frame.groupby(["dataset", "view", "model"], as_index=False).rmse.mean()
    rank_flips = {}
    for dataset in DATASETS:
        winners = []
        for view in VIEWS:
            group = mean_rmse[(mean_rmse.dataset == dataset) & (mean_rmse.view == view)]
            winners.append(str(group.loc[group.rmse.idxmin(), "model"]))
        rank_flips[dataset] = len(set(winners)) > 1
    audit = {
        "status": "complete",
        "protocol_sha256": PROTOCOL_SHA256,
        "wall_seconds": time.perf_counter() - started,
        "cells": len(frame),
        "maximum_roundtrip_error": max(integrity.values()),
        "qualifying_models": int(qualifying.model.nunique()),
        "qualifying_datasets": int(qualifying.dataset.nunique()),
        "qualifying_pairs": len(qualifying),
        "rank_flips": rank_flips,
        "rank_flip_datasets": sum(rank_flips.values()),
    }
    audit["passed"] = bool(
        audit["qualifying_models"] >= 2
        and audit["qualifying_datasets"] >= 2
        and audit["rank_flip_datasets"] >= 2
    )
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
