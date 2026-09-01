#!/usr/bin/env python3
"""State-cardinality and real-row scaling measurements for MPE."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

from mpe import farthest_point_landmarks, kernel_affinity, nystrom_features, state_balanced_mean, state_weight_table
from representations import load_task, representation_tables, split_row_indices
from ridge_benchmark import fit_ridge


HERE = Path(__file__).resolve().parent


def matrix_bytes(value: np.ndarray | sparse.spmatrix) -> int:
    if sparse.issparse(value):
        value = sparse.csr_matrix(value)
        return int(value.data.nbytes + value.indices.nbytes + value.indptr.nbytes)
    return int(np.asarray(value).nbytes)


def state_order(count: int, seed: int) -> np.ndarray:
    keys = []
    for index in range(count):
        digest = hashlib.sha256(f"20260829|scaling|{seed}|{index}".encode()).digest()
        keys.append((digest, index))
    return np.asarray([index for _, index in sorted(keys)], dtype=np.int64)


def fit_state_ridge(table: np.ndarray, target: np.ndarray, train: np.ndarray, test: np.ndarray) -> tuple[float, float]:
    started = time.perf_counter()
    model = Ridge(alpha=1e-3, solver="lsqr", tol=1e-6, max_iter=3000)
    model.fit(table[train], target[train])
    prediction = model.predict(table[test])
    return float(np.mean((prediction - target[test]) ** 2)), time.perf_counter() - started


def synthetic_cardinality_suite() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for count in (100, 1000, 10000):
        coordinate = np.linspace(0.0, 1.0, count, dtype=np.float64)
        distance_started = time.perf_counter()
        distance = np.abs(coordinate[:, None] - coordinate[None, :]).astype(np.float32)
        distance_seconds = time.perf_counter() - distance_started
        order = state_order(count, 20261501)
        n_train = int(0.6 * count)
        n_validation = int(0.2 * count)
        train = order[:n_train]
        test = order[n_train + n_validation :]
        target = np.sin(2.0 * np.pi * coordinate) + 0.2 * np.cos(6.0 * np.pi * coordinate)
        within_training = distance[np.ix_(train, train)].copy()
        np.fill_diagonal(within_training, np.inf)
        nearest = np.min(within_training, axis=1)
        nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
        bandwidth = float(np.median(nearest)) if len(nearest) else 1.0 / count

        full_started = time.perf_counter()
        full_similarity = kernel_affinity(distance[:, train] / bandwidth, "gaussian").astype(np.float32)
        full_precompute = time.perf_counter() - full_started
        score, fit_seconds = fit_state_ridge(full_similarity, target, train, test)
        rows.append(
            {
                "states": count, "method": "full_similarity", "landmarks": len(train),
                "sparse_k": None, "feature_dimension": len(train), "precompute_seconds": full_precompute,
                "fit_seconds": fit_seconds, "state_metric_seconds": distance_seconds,
                "representation_bytes": matrix_bytes(full_similarity), "standardized_mse": score,
            }
        )
        del full_similarity
        gc.collect()

        # The unseen lookup state is a shared constant; store it sparsely rather
        # than allocating the deliberately huge dense one-hot matrix.
        lookup_score = float(np.mean((target[test] - target[train].mean()) ** 2))
        rows.append(
            {
                "states": count, "method": "lookup_unknown", "landmarks": len(train),
                "sparse_k": 1, "feature_dimension": len(train) + 1, "precompute_seconds": 0.0,
                "fit_seconds": 0.0, "state_metric_seconds": 0.0,
                "representation_bytes": int((count + 1) * (4 + 4) + (count + 1) * 4),
                "standardized_mse": lookup_score,
            }
        )

        for budget in (16, 32, 64, 128, 256):
            if budget > len(train):
                continue
            landmark_started = time.perf_counter()
            landmarks = farthest_point_landmarks(distance, train, budget)
            landmark_seconds = time.perf_counter() - landmark_started
            for method, sparse_k in (("mpe_dense", None), ("mpe_sparse", 8)):
                started = time.perf_counter()
                table = state_weight_table(distance, landmarks, bandwidth, sparse_k=sparse_k).astype(np.float32)
                precompute = time.perf_counter() - started
                if sparse_k is not None:
                    storage = sparse.csr_matrix(table)
                else:
                    storage = table
                score, fit_seconds = fit_state_ridge(table, target, train, test)
                rows.append(
                    {
                        "states": count, "method": method, "landmarks": budget,
                        "sparse_k": sparse_k, "feature_dimension": budget,
                        "precompute_seconds": precompute + landmark_seconds, "fit_seconds": fit_seconds,
                        "state_metric_seconds": distance_seconds, "representation_bytes": matrix_bytes(storage),
                        "standardized_mse": score,
                    }
                )
                del table, storage
            started = time.perf_counter()
            table, rank = nystrom_features(distance, np.arange(count), landmarks, bandwidth)
            precompute = time.perf_counter() - started
            score, fit_seconds = fit_state_ridge(table, target, train, test)
            rows.append(
                {
                    "states": count, "method": "nystrom", "landmarks": budget,
                    "sparse_k": None, "feature_dimension": int(rank),
                    "precompute_seconds": precompute + landmark_seconds, "fit_seconds": fit_seconds,
                    "state_metric_seconds": distance_seconds, "representation_bytes": matrix_bytes(table),
                    "standardized_mse": score,
                }
            )
            del table
            gc.collect()
        del distance
        gc.collect()
    return pd.DataFrame(rows)


def deterministic_subset(indices: np.ndarray, row_ids: np.ndarray, count: int, salt: str) -> np.ndarray:
    if len(indices) <= count:
        return np.asarray(indices, dtype=np.int64)
    ordered = sorted(
        indices.tolist(),
        key=lambda index: hashlib.sha256(f"20260829|{salt}|{row_ids[index]}".encode()).digest(),
    )
    return np.asarray(sorted(ordered[:count]), dtype=np.int64)


def real_row_suite() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    unavailable = [
        {
            "task": "amazon_leaf_category", "status": "NOT RUN",
            "reason": "frozen Amazon hierarchy schema unavailable; see preparation manifest",
        }
    ]
    for task_name in ("acs_occupation", "tlc_pickup_zone"):
        task = load_task(task_name)
        parts = split_row_indices(task, 0)
        ridge_path = HERE / "raw" / "ridge_cells" / f"{task_name}__split0__isolated_field.json"
        if not ridge_path.exists():
            raise FileNotFoundError(ridge_path)
        bandwidth = float(json.loads(ridge_path.read_text())["selected_bandwidth"])
        tables, _ = representation_tables(task, 0, bandwidth)
        row_ids = task.rows["row_id"].astype(str).to_numpy()
        raw = pd.to_numeric(task.rows["target"], errors="raise").to_numpy(np.float64)
        state_indices = task.row_state_indices()
        state_ids = task.rows["field_state"].astype(str).to_numpy()
        total = len(task.rows)
        for requested in (10000, 50000, total):
            cap = min(requested, total)
            counts = {
                name: max(1, int(round(cap * len(indices) / total)))
                for name, indices in parts.items()
            }
            selected = {
                name: deterministic_subset(indices, row_ids, counts[name], f"{task_name}|{cap}|{name}")
                for name, indices in parts.items()
            }
            mean = float(raw[selected["train"]].mean())
            scale = float(raw[selected["train"]].std()) or 1.0
            target = (raw - mean) / scale
            for method in ("mpe", "similarity_unnormalized", "nystrom", "unknown_embedding"):
                table = tables[method]
                design = sparse.csr_matrix(table[state_indices], dtype=np.float32)
                started = time.perf_counter()
                prediction, fit_seconds = fit_ridge(
                    design, target, state_indices, selected["train"], selected["test"], alpha=1.0
                )
                inference_seconds = time.perf_counter() - started - fit_seconds
                loss = (prediction - target[selected["test"]]) ** 2
                rows.append(
                    {
                        "task": task_name, "source_unit": task.manifest["source_unit"],
                        "requested_rows": str(requested) if requested != total else "full_cap",
                        "actual_rows": int(sum(map(len, selected.values()))), "method": method,
                        "feature_dimension": int(table.shape[1]),
                        "representation_bytes": matrix_bytes(table), "fit_seconds": fit_seconds,
                        "inference_seconds": max(0.0, inference_seconds),
                        "state_balanced_standardized_mse": state_balanced_mean(loss, state_ids[selected["test"]]),
                        "row_weighted_standardized_mse": float(loss.mean()),
                    }
                )
    return pd.DataFrame(rows), unavailable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["state", "rows", "all"], default="all")
    args = parser.parse_args()
    (HERE / "raw").mkdir(exist_ok=True)
    if args.suite in {"state", "all"}:
        state = synthetic_cardinality_suite()
        state.to_parquet(HERE / "raw" / "scalability_state_results.parquet", index=False, compression="zstd")
        state.to_csv(HERE / "raw" / "scalability_state_results.csv", index=False)
    if args.suite in {"rows", "all"}:
        rows, unavailable = real_row_suite()
        rows.to_parquet(HERE / "raw" / "scalability_row_results.parquet", index=False, compression="zstd")
        rows.to_csv(HERE / "raw" / "scalability_row_results.csv", index=False)
        (HERE / "raw" / "scalability_unavailable.json").write_text(json.dumps(unavailable, indent=2) + "\n")


if __name__ == "__main__":
    main()
