#!/usr/bin/env python3
"""Validation-only E0/E1 experiments for the post-MPE successor.

The original MPE test rows are deliberately never evaluated.  This runner uses
the frozen training states for fitting and the frozen validation states as a
development-only outer evaluation set.  E1a uses a deterministic inner split
of the training states for ridge regularization selection.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn


HERE = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = HERE.parents[1]
MPE_ROOT = EXPERIMENTS_ROOT / "mpe_iclr"
if str(MPE_ROOT) not in sys.path:
    sys.path.insert(0, str(MPE_ROOT))

from mpe import farthest_point_landmarks, kernel_affinity, state_balanced_mean, state_weight_table  # noqa: E402
from representations import TaskData, load_task, split_row_indices, split_state_indices  # noqa: E402
from ridge_benchmark import state_balanced_training_weights  # noqa: E402


ALL_TASKS = [
    "acs_occupation",
    "acs_industry",
    "tlc_pickup_zone",
    "tlc_dropoff_zone",
    "citibike_start_station",
    "airline_origin_airport",
    "airline_destination_airport",
    "employee_salaries",
    "medical_charges",
]
E0_TASKS = ["acs_occupation", "tlc_dropoff_zone", "medical_charges"]
E1B_TASKS = ["acs_occupation", "tlc_dropoff_zone", "citibike_start_station", "medical_charges"]
E0_CONDITIONS = [
    "weights_direct",
    "factor_random_learned",
    "factor_identity_learned",
    "factor_orthogonal_frozen",
    "factor_rezero",
]
E1_REPRESENTATIONS = [
    "weights_m32",
    "affinity_m32",
    "distance_m32",
    "distance_m64",
    "distance_m128",
    "distance_all",
    "distance_plus_weights_m128",
]
E1B_REPRESENTATIONS = [name for name in E1_REPRESENTATIONS if name != "affinity_m32"]
NEURAL_SEEDS = [20262101, 20262102, 20262103]
RIDGE_ALPHAS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
PROTOCOL_PATH = HERE / "DEVELOPMENT_PROTOCOL.md"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def stable_hash(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def inner_state_split(task: TaskData, split_index: int) -> tuple[np.ndarray, np.ndarray]:
    outer_train = split_state_indices(task, split_index)["train"]
    ordered = sorted(
        outer_train.tolist(),
        key=lambda state: (stable_hash("mft-inner", task.name, split_index, task.state_ids[state]), task.state_ids[state]),
    )
    validation_count = max(1, int(round(0.2 * len(ordered))))
    inner_validation = np.asarray(sorted(ordered[:validation_count]), dtype=np.int64)
    inner_train = np.asarray(sorted(ordered[validation_count:]), dtype=np.int64)
    if len(inner_train) == 0:
        raise ValueError(f"{task.name}: inner state split has no training states")
    return inner_train, inner_validation


def rows_for_states(task: TaskData, state_indices: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.isin(task.row_state_indices(), np.asarray(state_indices, dtype=np.int64)))


def train_metric_scale(distance: np.ndarray, training_states: np.ndarray) -> float:
    states = np.asarray(training_states, dtype=np.int64)
    within = np.asarray(distance, dtype=np.float64)[np.ix_(states, states)].copy()
    np.fill_diagonal(within, np.inf)
    nearest = np.min(within, axis=1)
    valid = nearest[np.isfinite(nearest) & (nearest > 1e-12)]
    if len(valid):
        return float(np.median(valid))
    upper = within[np.triu_indices(len(states), 1)]
    valid = upper[np.isfinite(upper) & (upper > 1e-12)]
    return float(np.median(valid)) if len(valid) else 1.0


def landmark_table(
    task: TaskData,
    training_states: np.ndarray,
    budget: int,
) -> tuple[np.ndarray, np.ndarray]:
    landmarks = farthest_point_landmarks(
        task.distance,
        training_states,
        min(int(budget), len(training_states)),
        state_ids=task.state_ids,
    )
    if not set(landmarks.tolist()).issubset(set(np.asarray(training_states, dtype=np.int64).tolist())):
        raise AssertionError("landmark outside development-training states")
    return np.asarray(task.distance[:, landmarks], dtype=np.float64), landmarks


def standardize_landmark_coordinates(
    raw: np.ndarray,
    training_states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Condition coordinates without using targets or held-out-state moments."""
    training = np.asarray(raw, dtype=np.float64)[np.asarray(training_states, dtype=np.int64)]
    center = np.mean(training, axis=0)
    scale = np.std(training, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    standardized = (np.asarray(raw, dtype=np.float64) - center) / scale
    if not np.isfinite(standardized).all():
        raise FloatingPointError("non-finite standardized landmark coordinate")
    return standardized.astype(np.float32), center, scale


def representation_tables(
    task: TaskData,
    training_states: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    scale = train_metric_scale(task.distance, training_states)
    raw32, landmarks32 = landmark_table(task, training_states, 32)
    raw64, landmarks64 = landmark_table(task, training_states, 64)
    raw128, landmarks128 = landmark_table(task, training_states, 128)
    raw_all, landmarks_all = landmark_table(task, training_states, min(256, len(training_states)))
    distance32, center32, coordinate_scale32 = standardize_landmark_coordinates(raw32, training_states)
    distance64, center64, coordinate_scale64 = standardize_landmark_coordinates(raw64, training_states)
    distance128, center128, coordinate_scale128 = standardize_landmark_coordinates(raw128, training_states)
    distance_all, center_all, coordinate_scale_all = standardize_landmark_coordinates(raw_all, training_states)
    weights = state_weight_table(
        task.distance,
        landmarks32,
        scale,
        kernel="gaussian",
        normalization="partition",
    )
    affinity = kernel_affinity(task.distance[:, landmarks32] / scale, "gaussian")
    tables = {
        "weights_m32": weights.astype(np.float32),
        "affinity_m32": affinity.astype(np.float32),
        "distance_m32": distance32,
        "distance_m64": distance64,
        "distance_m128": distance128,
        "distance_all": distance_all,
        "distance_plus_weights_m128": np.concatenate([distance128, weights], axis=1).astype(np.float32),
    }
    metadata = {
        "scale": scale,
        "landmark_indices": {
            "m32": landmarks32.tolist(),
            "m64": landmarks64.tolist(),
            "m128": landmarks128.tolist(),
            "all_capped_256": landmarks_all.tolist(),
        },
        "landmark_state_ids": {
            "m32": [task.state_ids[index] for index in landmarks32],
            "m64": [task.state_ids[index] for index in landmarks64],
            "m128": [task.state_ids[index] for index in landmarks128],
            "all_capped_256": [task.state_ids[index] for index in landmarks_all],
        },
        "dimensions": {name: int(table.shape[1]) for name, table in tables.items()},
        "coordinate_standardization": {
            "fit_states": "development_training_only",
            "constant_coordinate_scale": 1.0,
            "m32": {"center": center32.tolist(), "scale": coordinate_scale32.tolist()},
            "m64": {"center": center64.tolist(), "scale": coordinate_scale64.tolist()},
            "m128": {"center": center128.tolist(), "scale": coordinate_scale128.tolist()},
            "all_capped_256": {"center": center_all.tolist(), "scale": coordinate_scale_all.tolist()},
        },
    }
    return tables, metadata


def ordinary_design_subset(
    task: TaskData,
    fit_rows: np.ndarray,
    output_rows: np.ndarray,
) -> sparse.csr_matrix:
    columns = task.manifest["ordinary_covariates"]
    frame = task.rows[columns].copy()
    numeric_columns = [column for column in columns if pd.api.types.is_numeric_dtype(frame[column])]
    categorical_columns = [column for column in columns if column not in numeric_columns]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in categorical_columns:
        frame[column] = frame[column].astype("string").fillna("__MISSING__").astype(str)
    transformers = []
    if numeric_columns:
        transformers.append(
            ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric_columns)
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32),
                ),
                categorical_columns,
            )
        )
    transformer = ColumnTransformer(transformers, sparse_threshold=1.0)
    transformer.fit(frame.iloc[np.asarray(fit_rows, dtype=np.int64)])
    transformed = transformer.transform(frame.iloc[np.asarray(output_rows, dtype=np.int64)])
    return sparse.csr_matrix(transformed, dtype=np.float32)


def compose_design(
    task: TaskData,
    table: np.ndarray,
    output_rows: np.ndarray,
    ordinary: sparse.csr_matrix | None,
) -> sparse.csr_matrix:
    row_states = task.row_state_indices()[output_rows]
    representation = sparse.csr_matrix(np.asarray(table[row_states], dtype=np.float32))
    return representation if ordinary is None else sparse.hstack([ordinary, representation], format="csr")


def sealed_raw_target(task: TaskData, test_rows: np.ndarray) -> np.ndarray:
    target = pd.to_numeric(task.rows["target"], errors="raise").to_numpy(np.float64, copy=True)
    # Make accidental downstream evaluation of sealed targets fail noisily.
    target[np.asarray(test_rows, dtype=np.int64)] = np.nan
    return target


def standardized_target(raw_target: np.ndarray, train_rows: np.ndarray) -> tuple[np.ndarray, float, float]:
    values = raw_target[np.asarray(train_rows, dtype=np.int64)]
    if not np.isfinite(values).all():
        raise AssertionError("non-finite development-training target")
    mean = float(np.mean(values))
    scale = float(np.std(values)) or 1.0
    return (raw_target - mean) / scale, mean, scale


def ridge_prediction(
    design: sparse.csr_matrix,
    target: np.ndarray,
    states: np.ndarray,
    train_rows: np.ndarray,
    evaluation_rows: np.ndarray,
    alpha: float,
) -> np.ndarray:
    model = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr", tol=1e-5, max_iter=3000)
    weights = state_balanced_training_weights(states[train_rows])
    model.fit(design[train_rows], target[train_rows], sample_weight=weights)
    return np.asarray(model.predict(design[evaluation_rows]), dtype=np.float64)


def balanced_mse(target: np.ndarray, prediction: np.ndarray, states: np.ndarray) -> float:
    return float(state_balanced_mean((np.asarray(prediction) - np.asarray(target)) ** 2, states))


def run_e1a_cell(task_name: str, split_index: int, setting: str, output_root: Path) -> dict[str, Any]:
    cell_id = f"{task_name}__split{split_index}__{setting}"
    path = output_root / "e1a_cells" / f"{cell_id}.json"
    if path.exists():
        payload = json.loads(path.read_text())
        if payload.get("status") == "complete" and payload.get("protocol_sha256") == sha256_path(PROTOCOL_PATH):
            print(f"resume {cell_id}", flush=True)
            return payload

    started = time.perf_counter()
    task = load_task(task_name)
    state_parts = split_state_indices(task, split_index)
    row_parts = split_row_indices(task, split_index)
    inner_train_states, inner_validation_states = inner_state_split(task, split_index)
    inner_train_rows = rows_for_states(task, inner_train_states)
    inner_validation_rows = rows_for_states(task, inner_validation_states)
    if np.intersect1d(inner_train_rows, row_parts["test"]).size or np.intersect1d(
        inner_validation_rows, row_parts["test"]
    ).size:
        raise AssertionError("sealed test row entered inner development split")
    raw_target = sealed_raw_target(task, row_parts["test"])

    inner_output_rows = np.concatenate([inner_train_rows, inner_validation_rows])
    inner_train_local = np.arange(len(inner_train_rows), dtype=np.int64)
    inner_validation_local = np.arange(len(inner_train_rows), len(inner_output_rows), dtype=np.int64)
    inner_target_global, _, _ = standardized_target(raw_target, inner_train_rows)
    inner_target = inner_target_global[inner_output_rows]
    inner_states = task.rows["field_state"].astype(str).to_numpy()[inner_output_rows]
    inner_tables, inner_metadata = representation_tables(task, inner_train_states)
    inner_ordinary = (
        ordinary_design_subset(task, inner_train_rows, inner_output_rows) if setting == "full_table" else None
    )

    selected_alphas: dict[str, float] = {}
    inner_trials: dict[str, list[dict[str, float]]] = {}
    for representation_name in E1_REPRESENTATIONS:
        design = compose_design(task, inner_tables[representation_name], inner_output_rows, inner_ordinary)
        trials = []
        for alpha in RIDGE_ALPHAS:
            prediction = ridge_prediction(
                design,
                inner_target,
                inner_states,
                inner_train_local,
                inner_validation_local,
                alpha,
            )
            score = balanced_mse(
                inner_target[inner_validation_local], prediction, inner_states[inner_validation_local]
            )
            trials.append({"alpha": float(alpha), "state_balanced_standardized_mse": score})
        winner = min(trials, key=lambda item: (item["state_balanced_standardized_mse"], item["alpha"]))
        selected_alphas[representation_name] = float(winner["alpha"])
        inner_trials[representation_name] = trials
        del design

    outer_train_rows = row_parts["train"]
    outer_validation_rows = row_parts["validation"]
    outer_output_rows = np.concatenate([outer_train_rows, outer_validation_rows])
    outer_train_local = np.arange(len(outer_train_rows), dtype=np.int64)
    outer_validation_local = np.arange(len(outer_train_rows), len(outer_output_rows), dtype=np.int64)
    outer_target_global, target_mean, target_scale = standardized_target(raw_target, outer_train_rows)
    outer_target = outer_target_global[outer_output_rows]
    outer_states = task.rows["field_state"].astype(str).to_numpy()[outer_output_rows]
    outer_tables, outer_metadata = representation_tables(task, state_parts["train"])
    outer_ordinary = (
        ordinary_design_subset(task, outer_train_rows, outer_output_rows) if setting == "full_table" else None
    )

    results = []
    for representation_name in E1_REPRESENTATIONS:
        design = compose_design(task, outer_tables[representation_name], outer_output_rows, outer_ordinary)
        prediction = ridge_prediction(
            design,
            outer_target,
            outer_states,
            outer_train_local,
            outer_validation_local,
            selected_alphas[representation_name],
        )
        score = balanced_mse(
            outer_target[outer_validation_local], prediction, outer_states[outer_validation_local]
        )
        results.append(
            {
                "representation": representation_name,
                "selected_alpha": selected_alphas[representation_name],
                "feature_dimension": int(outer_tables[representation_name].shape[1]),
                "validation_state_balanced_standardized_mse": score,
            }
        )
        print(f"{cell_id} {representation_name} dev={score:.6f}", flush=True)
        del design, prediction

    payload = {
        "status": "complete",
        "stage": "e1a",
        "cell_id": cell_id,
        "task": task_name,
        "source_unit": task.manifest["source_unit"],
        "split": split_index,
        "setting": setting,
        "protocol_sha256": sha256_path(PROTOCOL_PATH),
        "sealed_original_test": True,
        "test_target_evaluations": 0,
        "states": {
            "inner_train": int(len(inner_train_states)),
            "inner_validation": int(len(inner_validation_states)),
            "outer_train": int(len(state_parts["train"])),
            "outer_development_validation": int(len(state_parts["validation"])),
            "sealed_test": int(len(state_parts["test"])),
        },
        "rows": {
            "inner_train": int(len(inner_train_rows)),
            "inner_validation": int(len(inner_validation_rows)),
            "outer_train": int(len(outer_train_rows)),
            "outer_development_validation": int(len(outer_validation_rows)),
            "sealed_test": int(len(row_parts["test"])),
        },
        "target_outer_train_mean": target_mean,
        "target_outer_train_scale": target_scale,
        "ridge_alphas": RIDGE_ALPHAS,
        "inner_trials": inner_trials,
        "inner_representation_metadata": inner_metadata,
        "outer_representation_metadata": outer_metadata,
        "results": results,
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_json(payload, path)
    return payload


class DenseMLP(nn.Module):
    def __init__(self, input_size: int, width: int = 128, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ControlledFactorMLP(nn.Module):
    def __init__(self, input_size: int, representation_size: int, condition: str):
        super().__init__()
        self.representation_size = int(representation_size)
        self.condition = condition
        # Construct the backbone first so paired seeds give every condition the
        # exact same downstream initialization.
        self.backbone = DenseMLP(input_size)
        self.tokenizer: nn.Linear | None = None
        self.delta: nn.Linear | None = None
        self.gamma: nn.Parameter | None = None
        if condition in {"factor_random_learned", "factor_identity_learned", "factor_orthogonal_frozen"}:
            self.tokenizer = nn.Linear(representation_size, representation_size, bias=False)
            if condition == "factor_identity_learned":
                with torch.no_grad():
                    self.tokenizer.weight.copy_(torch.eye(representation_size))
            elif condition == "factor_orthogonal_frozen":
                rng = np.random.default_rng(20262100)
                q, _ = np.linalg.qr(rng.normal(size=(representation_size, representation_size)))
                with torch.no_grad():
                    self.tokenizer.weight.copy_(torch.from_numpy(q.astype(np.float32)))
                self.tokenizer.weight.requires_grad_(False)
        elif condition == "factor_rezero":
            self.delta = nn.Linear(representation_size, representation_size, bias=False)
            self.gamma = nn.Parameter(torch.zeros(()))
        elif condition != "weights_direct":
            raise KeyError(condition)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.condition == "weights_direct":
            return x
        ordinary = x[:, : x.shape[1] - self.representation_size]
        representation = x[:, x.shape[1] - self.representation_size :]
        if self.tokenizer is not None:
            transformed = self.tokenizer(representation)
        elif self.delta is not None and self.gamma is not None:
            transformed = representation + self.gamma * self.delta(representation)
        else:
            raise AssertionError("missing tokenizer")
        return torch.cat([ordinary, transformed], dim=1) if ordinary.shape[1] else transformed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.transform(x))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dense_batch(
    design: sparse.csr_matrix | torch.Tensor,
    indices: np.ndarray | torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(design, torch.Tensor):
        index = indices if isinstance(indices, torch.Tensor) else torch.as_tensor(indices, device=device)
        return design.index_select(0, index)
    if isinstance(indices, torch.Tensor):
        numpy_indices = indices.detach().cpu().numpy()
    else:
        numpy_indices = np.asarray(indices)
    values = design[numpy_indices].toarray().astype(np.float32, copy=False)
    return torch.from_numpy(values).to(device, non_blocking=True)


def maybe_device_design(
    design: sparse.csr_matrix,
    device: torch.device,
    maximum_bytes: int = 6_000_000_000,
) -> sparse.csr_matrix | torch.Tensor:
    required = int(design.shape[0]) * int(design.shape[1]) * 4
    if device.type == "cuda" and required <= maximum_bytes:
        return torch.from_numpy(design.toarray().astype(np.float32, copy=False)).to(device, non_blocking=True)
    return design


def predict_neural(
    model: nn.Module,
    design: sparse.csr_matrix | torch.Tensor,
    rows: np.ndarray,
    device: torch.device,
    batch_size: int = 8192,
) -> np.ndarray:
    model.eval()
    values = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            chosen = rows[start : start + batch_size]
            values.append(model(dense_batch(design, chosen, device)).reshape(-1).float().cpu().numpy())
    return np.concatenate(values).astype(np.float64)


def fit_neural_validation(
    design: sparse.csr_matrix,
    target: np.ndarray,
    states: np.ndarray,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    representation_size: int,
    seed: int,
    device: torch.device,
    factor_condition: str | None,
) -> dict[str, Any]:
    seed_everything(seed)
    if factor_condition is None:
        model: nn.Module = DenseMLP(design.shape[1]).to(device)
    else:
        model = ControlledFactorMLP(design.shape[1], representation_size, factor_condition).to(device)
    device_design = maybe_device_design(design, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=160, eta_min=5e-5)
    weights = np.zeros(len(states), dtype=np.float32)
    weights[train_rows] = state_balanced_training_weights(states[train_rows]).astype(np.float32)
    target_tensor = torch.from_numpy(target.astype(np.float32, copy=False)).to(device)
    weight_tensor = torch.from_numpy(weights).to(device)
    generator = np.random.default_rng(seed + 991)
    initial_prediction = predict_neural(model, device_design, validation_rows, device)
    initial_score = balanced_mse(target[validation_rows], initial_prediction, states[validation_rows])
    best_score = float("inf")
    best_epoch = -1
    stale = 0
    curve = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(160):
        model.train()
        order = generator.permutation(train_rows)
        for start in range(0, len(order), 2048):
            chosen = order[start : start + 2048]
            chosen_tensor = torch.as_tensor(chosen, device=device)
            x = dense_batch(device_design, chosen_tensor, device)
            y = target_tensor.index_select(0, chosen_tensor)
            row_weight = weight_tensor.index_select(0, chosen_tensor)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x).reshape(-1)
            loss_rows = (prediction - y) ** 2
            loss = (loss_rows * row_weight).sum() / row_weight.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
        scheduler.step()
        prediction = predict_neural(model, device_design, validation_rows, device)
        score = balanced_mse(target[validation_rows], prediction, states[validation_rows])
        curve.append({"epoch": epoch + 1, "validation_state_balanced_standardized_mse": score})
        if score < best_score - 1e-8:
            best_score = score
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
        if stale >= 20:
            break
    gamma = None
    if isinstance(model, ControlledFactorMLP) and model.gamma is not None:
        gamma = float(model.gamma.detach().cpu())
    result = {
        "initial_validation_state_balanced_standardized_mse": initial_score,
        "validation_state_balanced_standardized_mse": best_score,
        "best_epoch": best_epoch,
        "stop_epoch": len(curve),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        "gamma_final": gamma,
        "wall_seconds": time.perf_counter() - started,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "curve": curve,
    }
    del model, device_design, target_tensor, weight_tensor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def neural_dev_design(
    task: TaskData,
    split_index: int,
    representation_name: str,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    state_parts = split_state_indices(task, split_index)
    row_parts = split_row_indices(task, split_index)
    train_rows = row_parts["train"]
    validation_rows = row_parts["validation"]
    if np.intersect1d(np.concatenate([train_rows, validation_rows]), row_parts["test"]).size:
        raise AssertionError("sealed test row entered neural development design")
    output_rows = np.concatenate([train_rows, validation_rows])
    train_local = np.arange(len(train_rows), dtype=np.int64)
    validation_local = np.arange(len(train_rows), len(output_rows), dtype=np.int64)
    raw_target = sealed_raw_target(task, row_parts["test"])
    target_global, target_mean, target_scale = standardized_target(raw_target, train_rows)
    target = target_global[output_rows]
    states = task.rows["field_state"].astype(str).to_numpy()[output_rows]
    tables, metadata = representation_tables(task, state_parts["train"])
    ordinary = ordinary_design_subset(task, train_rows, output_rows)
    design = compose_design(task, tables[representation_name], output_rows, ordinary)
    metadata = {
        **metadata,
        "target_train_mean": target_mean,
        "target_train_scale": target_scale,
        "development_rows": {"train": int(len(train_rows)), "validation": int(len(validation_rows))},
        "sealed_test_rows": int(len(row_parts["test"])),
    }
    return design, target, states, train_local, validation_local, metadata


def run_neural_cell(
    stage: str,
    task_name: str,
    split_index: int,
    condition: str,
    seed: int,
    device_name: str,
    output_root: Path,
) -> dict[str, Any]:
    if stage == "e0":
        representation_name = "weights_m32"
        factor_condition = condition
    elif stage == "e1b":
        representation_name = condition
        factor_condition = None
    else:
        raise KeyError(stage)
    cell_id = f"{task_name}__split{split_index}__{condition}__seed{seed}"
    path = output_root / f"{stage}_cells" / f"{cell_id}.json"
    if path.exists():
        payload = json.loads(path.read_text())
        if payload.get("status") == "complete" and payload.get("protocol_sha256") == sha256_path(PROTOCOL_PATH):
            print(f"resume {stage} {cell_id}", flush=True)
            return payload
    if stage == "e1b" and condition == "weights_m32" and task_name in E0_TASKS:
        source_path = (
            output_root
            / "e0_cells"
            / f"{task_name}__split{split_index}__weights_direct__seed{seed}.json"
        )
        source = json.loads(source_path.read_text())
        payload = {
            **source,
            "stage": "e1b",
            "cell_id": cell_id,
            "condition": "weights_m32",
            "representation": "weights_m32",
            "reused_without_retraining": True,
            "reused_from": str(source_path.relative_to(HERE)),
        }
        atomic_json(payload, path)
        print(f"e1b {cell_id} reused=e0_weights_direct", flush=True)
        return payload
    task = load_task(task_name)
    design, target, states, train_rows, validation_rows, metadata = neural_dev_design(
        task, split_index, representation_name
    )
    started = time.perf_counter()
    result = fit_neural_validation(
        design,
        target,
        states,
        train_rows,
        validation_rows,
        metadata["dimensions"][representation_name],
        seed,
        torch.device(device_name),
        factor_condition,
    )
    payload = {
        "status": "complete",
        "stage": stage,
        "cell_id": cell_id,
        "task": task_name,
        "source_unit": task.manifest["source_unit"],
        "split": split_index,
        "setting": "full_table",
        "condition": condition,
        "representation": representation_name,
        "seed": seed,
        "device": device_name,
        "protocol_sha256": sha256_path(PROTOCOL_PATH),
        "sealed_original_test": True,
        "test_target_evaluations": 0,
        "feature_dimension": int(design.shape[1]),
        "representation_dimension": int(metadata["dimensions"][representation_name]),
        "representation_metadata": metadata,
        "optimization": {
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "width": 128,
            "depth": 2,
            "dropout": 0.1,
            "batch_size": 2048,
            "max_epochs": 160,
            "patience": 20,
        },
        "result": result,
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_json(payload, path)
    print(
        f"{stage} {cell_id} dev={result['validation_state_balanced_standardized_mse']:.6f} "
        f"epoch={result['best_epoch']}",
        flush=True,
    )
    del design
    gc.collect()
    return payload


def selected(values: Iterable[Any], requested: Any | None) -> list[Any]:
    values = list(values)
    return values if requested is None or requested == "all" else [requested]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["e0", "e1a", "e1b"], required=True)
    parser.add_argument("--task", default="all")
    parser.add_argument("--split", default="all")
    parser.add_argument("--setting", choices=["isolated_field", "full_table", "all"], default="all")
    parser.add_argument("--condition", default="all")
    parser.add_argument("--seed", default="all")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()

    protocol_hash = sha256_path(PROTOCOL_PATH)
    expected_hash_file = HERE / "PROTOCOL_SHA256.txt"
    expected = expected_hash_file.read_text().split()[0] if expected_hash_file.exists() else ""
    if protocol_hash != expected:
        raise RuntimeError(
            f"protocol hash mismatch: current={protocol_hash}, expected={expected}; update the freeze before running"
        )

    if args.stage == "e1a":
        tasks = ALL_TASKS if args.task == "all" else [args.task]
        splits = list(range(5)) if args.split == "all" else [int(args.split)]
        settings = ["isolated_field", "full_table"] if args.setting == "all" else [args.setting]
        for task_name in tasks:
            if task_name not in ALL_TASKS:
                raise ValueError(task_name)
            for split_index in splits:
                for setting in settings:
                    run_e1a_cell(task_name, split_index, setting, args.output)
        return

    task_menu = E0_TASKS if args.stage == "e0" else E1B_TASKS
    condition_menu = E0_CONDITIONS if args.stage == "e0" else E1B_REPRESENTATIONS
    tasks = task_menu if args.task == "all" else [args.task]
    splits = [0, 1] if args.split == "all" else [int(args.split)]
    conditions = condition_menu if args.condition == "all" else [args.condition]
    seeds = NEURAL_SEEDS if args.seed == "all" else [int(args.seed)]
    for task_name in tasks:
        if task_name not in task_menu:
            raise ValueError(f"{task_name} not in {args.stage} menu")
        for split_index in splits:
            for condition in conditions:
                if condition not in condition_menu:
                    raise ValueError(f"{condition} not in {args.stage} menu")
                for seed in seeds:
                    if seed not in NEURAL_SEEDS:
                        raise ValueError(f"seed {seed} not frozen")
                    run_neural_cell(
                        args.stage,
                        task_name,
                        split_index,
                        condition,
                        seed,
                        args.device,
                        args.output,
                    )


if __name__ == "__main__":
    main()
