#!/usr/bin/env python3
"""Source-pinned real temporal pilot on UCI Bike Sharing."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from tabalu.baselines.regressors import NeuralRegressor
from tabalu.data.bike_sharing import (
    ARCHIVE_SHA256,
    SOURCE_DOI,
    SOURCE_LICENSE,
    SOURCE_URL,
    load_hourly_bike_sharing,
    temporal_bike_split,
)
from tabalu.evaluation import regression_metrics
from tabalu.models.bike_typed import SeasonRoutedBikeProgram, SparseBikeProgram, bike_typed_design


CATEGORICAL = ["season", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]
NUMERICAL = ["temp", "atemp", "hum", "windspeed"]
RAW_FEATURES = NUMERICAL + CATEGORICAL


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fit_predict_models(
    split: dict[str, Any], seed: int, config: dict[str, Any]
) -> dict[str, tuple[dict[str, np.ndarray], float, int]]:
    train, validation = split["train"], split["validation"]
    train_y = train["cnt"].to_numpy(dtype=np.float64)
    validation_y = validation["cnt"].to_numpy(dtype=np.float64)
    tests = {name: frame for name, frame in split.items() if name in {"iid", "future"}}
    output: dict[str, tuple[dict[str, np.ndarray], float, int]] = {}

    for name, model in (("TabALU-Global", SparseBikeProgram()), ("TabALU-SeasonRouter", SeasonRoutedBikeProgram())):
        started = time.perf_counter()
        model.fit(train, train_y, (validation, validation_y))
        elapsed = time.perf_counter() - started
        output[name] = ({key: model.predict(frame) for key, frame in tests.items()}, elapsed, model.operation_count)

    train_design, _ = bike_typed_design(train)
    validation_design, _ = bike_typed_design(validation)
    mlp = NeuralRegressor("MLP", seed=seed, epochs=int(config["neural_epochs"]), device=str(config["device"]))
    started = time.perf_counter()
    mlp.fit(train_design, train_y, (validation_design, validation_y))
    elapsed = time.perf_counter() - started
    output["TypedFeatures-MLP"] = (
        {key: mlp.predict(bike_typed_design(frame)[0]) for key, frame in tests.items()},
        elapsed,
        -1,
    )

    encoder = ColumnTransformer(
        (("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL), ("numeric", "passthrough", NUMERICAL)),
        sparse_threshold=0.0,
    )
    train_encoded = encoder.fit_transform(train[RAW_FEATURES])
    validation_encoded = encoder.transform(validation[RAW_FEATURES])
    tree_models = {
        "XGBoost": XGBRegressor(
            n_estimators=800,
            max_depth=8,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=seed,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            max_features=0.8,
            n_jobs=-1,
            random_state=seed,
        ),
    }
    for name, model in tree_models.items():
        started = time.perf_counter()
        if name == "XGBoost":
            model.fit(train_encoded, train_y, eval_set=[(validation_encoded, validation_y)], verbose=False)
        else:
            model.fit(train_encoded, train_y)
        elapsed = time.perf_counter() - started
        output[name] = (
            {key: np.asarray(model.predict(encoder.transform(frame[RAW_FEATURES]))) for key, frame in tests.items()},
            elapsed,
            -1,
        )

    cat_train = train[RAW_FEATURES].copy()
    cat_validation = validation[RAW_FEATURES].copy()
    for column in CATEGORICAL:
        cat_train[column] = cat_train[column].astype(str)
        cat_validation[column] = cat_validation[column].astype(str)
    catboost = CatBoostRegressor(
        iterations=800,
        depth=8,
        learning_rate=0.04,
        loss_function="RMSE",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )
    started = time.perf_counter()
    catboost.fit(
        cat_train,
        train_y,
        cat_features=CATEGORICAL,
        eval_set=(cat_validation, validation_y),
        early_stopping_rounds=80,
        verbose=False,
    )
    elapsed = time.perf_counter() - started
    cat_predictions: dict[str, np.ndarray] = {}
    for key, frame in tests.items():
        values = frame[RAW_FEATURES].copy()
        for column in CATEGORICAL:
            values[column] = values[column].astype(str)
        cat_predictions[key] = np.asarray(catboost.predict(values))
    output["CatBoost"] = (cat_predictions, elapsed, -1)
    return output


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(str(row["model"]), str(row["split"]))].append(row)
    output: list[dict[str, Any]] = []
    for (model, split), rows in sorted(groups.items()):
        values = np.array([float(row["nrmse"]) for row in rows])
        output.append(
            {
                "model": model,
                "split": split,
                "n_seeds": len(rows),
                "nrmse_mean": float(values.mean()),
                "nrmse_std_across_seeds": float(values.std(ddof=1)),
                "rmse_mean": float(np.mean([float(row["rmse"]) for row in rows])),
                "mae_mean": float(np.mean([float(row["mae"]) for row in rows])),
                "r2_mean": float(np.mean([float(row["r2"]) for row in rows])),
                "training_seconds_mean": float(np.mean([float(row["training_seconds"]) for row in rows])),
                "operation_count_mean": float(np.mean([float(row["operation_count"]) for row in rows])),
            }
        )
    return output


def evaluate_gate(summary: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    lookup = {(row["model"], row["split"]): float(row["nrmse_mean"]) for row in summary}
    router_future = lookup[("TabALU-SeasonRouter", "future")]
    global_future = lookup[("TabALU-Global", "future")]
    best_tree = min(lookup[(name, "future")] for name in ("CatBoost", "XGBoost", "RandomForest"))
    degradation = router_future / max(lookup[("TabALU-SeasonRouter", "iid")], 1.0e-12)
    observed = {
        "router_vs_global_future_ratio": router_future / max(global_future, 1.0e-12),
        "router_vs_best_tree_future_ratio": router_future / max(best_tree, 1.0e-12),
        "router_temporal_degradation_ratio": degradation,
    }
    checks = {
        "router_beats_global": observed["router_vs_global_future_ratio"]
        <= thresholds["router_vs_global_future_ratio_max"],
        "router_competes_with_best_tree": observed["router_vs_best_tree_future_ratio"]
        <= thresholds["router_vs_best_tree_future_ratio_max"],
        "router_temporal_degradation_is_bounded": observed["router_temporal_degradation_ratio"]
        <= thresholds["router_temporal_degradation_ratio_max"],
    }
    return {"passed": all(checks.values()), "checks": checks, "observed": observed, "thresholds": thresholds}


def plot(summary: list[dict[str, Any]], output: Path) -> None:
    models = sorted({str(row["model"]) for row in summary})
    lookup = {(row["model"], row["split"]): row for row in summary}
    x = np.arange(len(models))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for offset, split in ((-width / 2, "iid"), (width / 2, "future")):
        axis.bar(
            x + offset,
            [float(lookup[(model, split)]["nrmse_mean"]) for model in models],
            width,
            label=split,
        )
    axis.set_xticks(x, [model.replace("-", "\n", 1) for model in models], rotation=15, ha="right")
    axis.set_yscale("log")
    axis.set_ylabel("Normalized RMSE")
    axis.set_title("UCI Bike Sharing: 2011 IID holdout vs all of 2012")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(output, dpi=200)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def run(config_path: Path) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = (PACKAGE_ROOT / config["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    frame = load_hourly_bike_sharing((PACKAGE_ROOT / config["cache_dir"]).resolve())
    repository = PACKAGE_ROOT.parents[1]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repository, text=True).strip())
    records: list[dict[str, Any]] = []
    split_sizes: dict[str, int] = {}
    for seed in config["seeds"]:
        split = temporal_bike_split(frame, int(seed))
        split_sizes = {name: len(values) for name, values in split.items()}
        models = fit_predict_models(split, int(seed), config)
        for model_name, (predictions, elapsed, operation_count) in models.items():
            for split_name, prediction in predictions.items():
                targets = split[split_name]["cnt"].to_numpy(dtype=np.float64)
                records.append(
                    {
                        "git_commit": commit,
                        "git_dirty": dirty,
                        "dataset": "UCI Bike Sharing (hourly)",
                        "seed": seed,
                        "model": model_name,
                        "split": split_name,
                        "training_seconds": elapsed,
                        "operation_count": operation_count,
                        **regression_metrics(targets, prediction),
                    }
                )
        write_csv(output / "records.csv", records)
        print(f"completed seed {int(seed) + 1}/{len(config['seeds'])}", flush=True)
    summary = summarize(records)
    write_csv(output / "summary.csv", summary)
    decision = evaluate_gate(summary, config["gate"])
    expected = len(config["seeds"]) * 6 * 2
    audit = {
        "git_commit": commit,
        "git_dirty": dirty,
        "source_url": SOURCE_URL,
        "source_doi": SOURCE_DOI,
        "source_license": SOURCE_LICENSE,
        "source_sha256": ARCHIVE_SHA256,
        "leakage_columns_excluded": ["casual", "registered", "cnt", "instant"],
        "split_definition": "70/15/15 random train/validation/IID within 2011; all 2012 future",
        "split_sizes_last_seed": split_sizes,
        "expected_records": expected,
        "observed_records": len(records),
        "all_finite": all(math.isfinite(float(row["nrmse"])) for row in records),
        "inference_unit": "one dataset; seed dispersion only, no dataset-level superiority inference",
        "gate": decision,
    }
    audit["audit_passed"] = expected == len(records) and audit["all_finite"] and decision["passed"]
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    plot(summary, output / "bike_temporal.png")
    print(json.dumps(audit, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
