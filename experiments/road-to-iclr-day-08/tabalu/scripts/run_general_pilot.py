#!/usr/bin/env python3
"""Small real general-tabular regression/classification boundary pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from tabalu.evaluation import regression_metrics
from tabalu.models.generic import SparseExactClassifier, SparseExactRegressor


def datasets() -> dict[str, tuple[np.ndarray, np.ndarray, str, str]]:
    diabetes = load_diabetes()
    cancer = load_breast_cancer()
    wine = load_wine()
    return {
        "diabetes": (diabetes.data.astype(float), diabetes.target.astype(float), "regression", "sklearn load_diabetes; Efron et al. 2004"),
        "breast_cancer": (cancer.data.astype(float), cancer.target.astype(int), "classification", "UCI Wisconsin Diagnostic Breast Cancer via sklearn"),
        "wine_binary": (wine.data.astype(float), (wine.target == 0).astype(int), "classification", "UCI Wine Recognition via sklearn; class 0 versus rest"),
    }


def split_data(features: np.ndarray, targets: np.ndarray, task: str, seed: int) -> tuple[np.ndarray, ...]:
    stratify = targets if task == "classification" else None
    train_x, remainder_x, train_y, remainder_y = train_test_split(
        features,
        targets,
        test_size=0.4,
        random_state=seed,
        stratify=stratify,
    )
    remainder_stratify = remainder_y if task == "classification" else None
    validation_x, test_x, validation_y, test_y = train_test_split(
        remainder_x,
        remainder_y,
        test_size=0.5,
        random_state=seed + 100,
        stratify=remainder_stratify,
    )
    return train_x, validation_x, test_x, train_y, validation_y, test_y


def build_models(task: str, seed: int) -> dict[str, Any]:
    if task == "regression":
        return {
            "TabALU-SparseExact": SparseExactRegressor(),
            "Linear": make_pipeline(StandardScaler(), LinearRegression()),
            "RandomForest": RandomForestRegressor(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=seed),
            "XGBoost": XGBRegressor(n_estimators=700, max_depth=5, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=seed),
            "MLP": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), early_stopping=True, max_iter=800, random_state=seed)),
            "CatBoost": CatBoostRegressor(iterations=700, depth=6, learning_rate=0.04, verbose=False, allow_writing_files=False, random_seed=seed),
        }
    return {
        "TabALU-SparseExact": SparseExactClassifier(),
        "Linear": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=seed)),
        "RandomForest": RandomForestClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=seed),
        "XGBoost": XGBClassifier(n_estimators=700, max_depth=5, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=seed),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 64), early_stopping=True, max_iter=800, random_state=seed)),
        "CatBoost": CatBoostClassifier(iterations=700, depth=6, learning_rate=0.04, verbose=False, allow_writing_files=False, random_seed=seed),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(str(row["dataset"]), str(row["task"]), str(row["model"]))].append(row)
    output: list[dict[str, Any]] = []
    for (dataset, task, model), rows in sorted(groups.items()):
        errors = np.array([float(row["primary_error"]) for row in rows])
        output.append(
            {
                "dataset": dataset,
                "task": task,
                "model": model,
                "n_seeds": len(rows),
                "primary_metric": "nrmse" if task == "regression" else "log_loss",
                "primary_error_mean": float(errors.mean()),
                "primary_error_std": float(errors.std(ddof=1)),
                "score_mean": float(np.mean([float(row["score"]) for row in rows])),
                "training_seconds_mean": float(np.mean([float(row["training_seconds"]) for row in rows])),
                "operation_count_mean": float(np.mean([float(row["operation_count"]) for row in rows])),
            }
        )
    return output


def evaluate_gate(summary: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary:
        by_dataset[str(row["dataset"])].append(row)
    ratios: dict[str, float] = {}
    for dataset, rows in by_dataset.items():
        tabalu = float(next(row for row in rows if row["model"] == "TabALU-SparseExact")["primary_error_mean"])
        best = min(float(row["primary_error_mean"]) for row in rows if row["model"] != "TabALU-SparseExact")
        ratios[dataset] = tabalu / max(best, 1.0e-12)
    competitive = sum(ratio <= thresholds["competitive_error_ratio_max"] for ratio in ratios.values())
    observed = {
        "dataset_error_ratios_vs_best_baseline": ratios,
        "competitive_dataset_count": competitive,
        "maximum_error_ratio": max(ratios.values()),
    }
    checks = {
        "competitive_on_required_datasets": competitive >= thresholds["competitive_dataset_count_min"],
        "no_catastrophic_general_failure": max(ratios.values()) <= thresholds["catastrophic_error_ratio_max"],
    }
    return {"passed": all(checks.values()), "checks": checks, "observed": observed, "thresholds": thresholds}


def plot(summary: list[dict[str, Any]], output: Path) -> None:
    datasets_order = sorted({str(row["dataset"]) for row in summary})
    models = sorted({str(row["model"]) for row in summary})
    lookup = {(row["dataset"], row["model"]): float(row["primary_error_mean"]) for row in summary}
    figure, axes = plt.subplots(1, len(datasets_order), figsize=(14, 4.2), constrained_layout=True)
    for axis, dataset in zip(axes, datasets_order):
        values = [lookup[(dataset, model)] for model in models]
        axis.bar(np.arange(len(models)), values)
        axis.set_xticks(np.arange(len(models)), [model.replace("-", "\n", 1) for model in models], rotation=30, ha="right", fontsize=8)
        metric = next(row["primary_metric"] for row in summary if row["dataset"] == dataset)
        axis.set_ylabel(metric)
        axis.set_title(dataset)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("General numeric pilot: lower is better")
    figure.savefig(output, dpi=200)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def run(config_path: Path) -> Path:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = (PACKAGE_ROOT / config["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    repository = PACKAGE_ROOT.parents[1]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repository, text=True).strip())
    catalog = datasets()
    records: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    for dataset_name in config["datasets"]:
        features, targets, task, source = catalog[dataset_name]
        metadata[dataset_name] = {"rows": len(features), "features": features.shape[1], "task": task, "source": source}
        for seed in config["seeds"]:
            train_x, val_x, test_x, train_y, val_y, test_y = split_data(features, targets, task, int(seed))
            for model_name, model in build_models(task, int(seed)).items():
                started = time.perf_counter()
                if isinstance(model, (SparseExactRegressor, SparseExactClassifier)):
                    model.fit(train_x, train_y, (val_x, val_y))
                elif model_name == "CatBoost":
                    model.fit(train_x, train_y, eval_set=(val_x, val_y), early_stopping_rounds=80, verbose=False)
                else:
                    model.fit(train_x, train_y)
                elapsed = time.perf_counter() - started
                operation_count = model.operation_count if isinstance(model, (SparseExactRegressor, SparseExactClassifier)) else -1
                if task == "regression":
                    prediction = np.asarray(model.predict(test_x), dtype=float)
                    metrics = regression_metrics(test_y, prediction)
                    primary_error = metrics["nrmse"]
                    score = metrics["r2"]
                    extra = {"accuracy": -1.0, "log_loss": -1.0, "brier": -1.0, **metrics}
                else:
                    probability = np.asarray(model.predict_proba(test_x), dtype=float)[:, 1]
                    prediction = (probability >= 0.5).astype(int)
                    loss = float(log_loss(test_y, np.column_stack((1 - probability, probability)), labels=[0, 1]))
                    auc = float(roc_auc_score(test_y, probability))
                    primary_error = loss
                    score = auc
                    extra = {
                        "accuracy": float(accuracy_score(test_y, prediction)),
                        "log_loss": loss,
                        "brier": float(brier_score_loss(test_y, probability)),
                        "mae": -1.0,
                        "rmse": -1.0,
                        "r2": -1.0,
                        "nrmse": -1.0,
                        "relative_error": -1.0,
                    }
                records.append(
                    {
                        "git_commit": commit,
                        "git_dirty": dirty,
                        "dataset": dataset_name,
                        "task": task,
                        "seed": seed,
                        "model": model_name,
                        "training_seconds": elapsed,
                        "operation_count": operation_count,
                        "primary_error": primary_error,
                        "score": score,
                        **extra,
                    }
                )
            write_csv(output / "records.csv", records)
        print(f"completed dataset {dataset_name}", flush=True)
    summary = summarize(records)
    write_csv(output / "summary.csv", summary)
    decision = evaluate_gate(summary, config["gate"])
    expected = len(config["datasets"]) * len(config["seeds"]) * 6
    audit = {
        "git_commit": commit,
        "git_dirty": dirty,
        "sklearn_version": sklearn.__version__,
        "dataset_metadata": metadata,
        "split_definition": "60/20/20 random; stratified for classification",
        "expected_records": expected,
        "observed_records": len(records),
        "all_finite": all(math.isfinite(float(row["primary_error"])) for row in records),
        "inference_unit": "dataset; only three pilot datasets, no formal rank claim",
        "gate": decision,
    }
    audit["audit_passed"] = expected == len(records) and audit["all_finite"] and decision["passed"]
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    plot(summary, output / "general_pilot.png")
    print(json.dumps(audit, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
