"""Leakage-safe nested residual target-encoding diagnostic (Intervention C)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold

from .core import PARTS, base_schema, combine, geometry, load_dataset, make_prepared, quantile_numeric, train_model


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "day3" / "residual_te.csv"
CURVES = ROOT / "results" / "day3" / "curves_residual_te.csv"


def read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def splitter(task: str, y: np.ndarray, seed: int, folds: int = 5):
    split = StratifiedKFold(folds, shuffle=True, random_state=seed) if task == "binclass" else KFold(folds, shuffle=True, random_state=seed)
    return list(split.split(np.zeros(len(y)), y if task == "binclass" else None))


def predict_numeric(task: str, x_fit: np.ndarray, y_fit: np.ndarray, x_query: np.ndarray) -> np.ndarray:
    if task == "binclass":
        model = LogisticRegression(C=1.0, max_iter=300, solver="lbfgs").fit(x_fit, y_fit)
        return model.predict_proba(x_query)[:, 1]
    return Ridge(alpha=1.0).fit(x_fit, y_fit).predict(x_query)


def oof_numeric(task: str, x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    prediction = np.zeros(len(y), dtype=np.float64)
    for fit, holdout in splitter(task, y, seed):
        prediction[holdout] = predict_numeric(task, x[fit], y[fit], x[holdout])
    return prediction


def map_values(train_values: np.ndarray, statistic: np.ndarray, query: np.ndarray, smoothing: float, prior: float) -> np.ndarray:
    levels, inverse, counts = np.unique(train_values.astype(str), return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=statistic)
    means = (sums + smoothing * prior) / (counts + smoothing)
    lookup = dict(zip(levels.tolist(), means.tolist()))
    return np.asarray([lookup.get(str(value), prior) for value in query], dtype=np.float64)


def nested_features(dataset, smoothing: float = 20.0) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    assert dataset.x_num is not None and dataset.x_cat is not None
    numeric = quantile_numeric(dataset.x_num)
    y = dataset.y["train"].astype(np.float64)
    standard_train = np.zeros((len(y), dataset.x_cat["train"].shape[1]), dtype=np.float64)
    residual_train = np.zeros_like(standard_train)
    # Each outer holdout is encoded by maps whose numeric residuals are themselves
    # inner-OOF within the outer fitting partition.
    for outer_number, (fit, holdout) in enumerate(splitter(dataset.task, y, 71000)):
        inner_prediction = oof_numeric(dataset.task, numeric["train"][fit], y[fit], 72000 + outer_number)
        residual = y[fit] - inner_prediction
        for j in range(standard_train.shape[1]):
            standard_train[holdout, j] = map_values(dataset.x_cat["train"][fit, j], y[fit], dataset.x_cat["train"][holdout, j], smoothing, float(y[fit].mean()))
            residual_train[holdout, j] = map_values(dataset.x_cat["train"][fit, j], residual, dataset.x_cat["train"][holdout, j], smoothing, 0.0)

    full_prediction = oof_numeric(dataset.task, numeric["train"], y, 73000)
    full_residual = y - full_prediction
    standard = {"train": standard_train}
    residual = {"train": residual_train}
    for part in ("val", "test"):
        standard[part] = np.column_stack([
            map_values(dataset.x_cat["train"][:, j], y, dataset.x_cat[part][:, j], smoothing, float(y.mean()))
            for j in range(standard_train.shape[1])
        ])
        residual[part] = np.column_stack([
            map_values(dataset.x_cat["train"][:, j], full_residual, dataset.x_cat[part][:, j], smoothing, 0.0)
            for j in range(standard_train.shape[1])
        ])
    return standard, residual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["adult", "diamond"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    rows, curves_rows = read(OUT), read(CURVES)
    complete = {(r["dataset"], int(r["seed"]), r["representation"]) for r in rows}
    for name in args.datasets:
        dataset = load_dataset(name)
        schema = base_schema(dataset)
        standard, residual = nested_features(dataset)
        variants = {"plain_contrast": schema, "standard_target_encoding": combine([schema, standard]), "residual_target_encoding": combine([schema, residual])}
        for representation, x in variants.items():
            for seed in args.seeds:
                if (name, seed, representation) in complete:
                    continue
                fit, curves = train_model(make_prepared(dataset, x, {}), seed, args.device)
                row = {"experiment": "residual_te", "intervention_class": "C", "dataset": name, "task": dataset.task, "model": "mlp", "optimizer": "AdamW", "weight_decay": 1e-4, "seed": seed, "representation": representation, "regularizer": "standard", "nested_outer_folds": 5, "nested_inner_folds": 5, "smoothing": 20.0, "split_fingerprint": dataset.split_fingerprint, **geometry(x["train"]), **fit}
                rows.append(row)
                write(OUT, rows)
                curves_rows.extend({"experiment": "residual_te", "dataset": name, "seed": seed, "representation": representation, **curve} for curve in curves)
                write(CURVES, curves_rows)
                print(f"residual_te {name} s{seed} {representation} metric={fit['test_metric']:.6f}", flush=True)


if __name__ == "__main__":
    main()
