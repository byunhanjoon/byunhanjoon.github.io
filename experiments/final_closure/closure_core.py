"""Shared, restartable infrastructure for the frozen final-closure program."""

from __future__ import annotations

import hashlib
import copy
import json
import os
import random
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DAY5 = HERE.parent / "road-to-iclr-day-05"
sys.path.insert(0, str(DAY5))

import completion_neural_panel as completion  # noqa: E402


CONFIG_PATH = HERE / "final_closure_config.json"
HASH_PATH = HERE / "PROTOCOL_HASH.txt"
REGISTRY_PATH = HERE / "fit_registry.sqlite3"
RAW = HERE / "raw"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text())
    expected: dict[str, str] = {}
    for line in HASH_PATH.read_text().splitlines():
        name, digest = line.split()
        expected[name] = digest
    for name in ("FINAL_CLOSURE_PROTOCOL.md", "final_closure_config.json"):
        actual = sha256(HERE / name)
        if actual != expected.get(name):
            raise AssertionError(f"frozen {name} hash mismatch: {actual}")
    if config["status"] != "frozen_before_final_closure_outcomes":
        raise AssertionError("final closure config is not frozen")
    return config


CONFIG = load_config()


def stable_hash(*parts: Any) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode()
    return hashlib.sha256(payload).hexdigest()


def stable_seed(*parts: Any) -> int:
    root = int(CONFIG["rng"]["root_seed"])
    value = int(stable_hash(CONFIG["rng"]["domain_prefix"], root, *parts)[:16], 16)
    value &= (1 << int(CONFIG["rng"]["bits"])) - 1
    return value or 1


def derive_subseeds(master_seed: int) -> dict[str, int]:
    output = {}
    for domain in CONFIG["rng"]["domains"]:
        value = int(stable_hash(CONFIG["rng"]["domain_prefix"], int(master_seed), domain)[:16], 16)
        output[domain] = (value & ((1 << int(CONFIG["rng"]["bits"])) - 1)) or 1
    return output


def completion_config() -> dict[str, Any]:
    source = json.loads((DAY5 / "completion_config.json").read_text())
    source["datasets"] = list(CONFIG["all_datasets"])
    source["dataset_tasks"] = dict(CONFIG["dataset_tasks"])
    source["openml_ids"] = dict(CONFIG["openml_ids"])
    source["models"] = list(CONFIG["primary_models"])
    source["split_seeds"] = list(CONFIG["split_seeds"])
    source["view_seed"] = int(CONFIG["view_seed"])
    source["factor_levels"] = dict(CONFIG["factor_levels"])
    source["training"] = dict(CONFIG["training"])
    source["training"]["epochs"] = int(CONFIG["experiment_a"]["epochs"])
    source["subsample"] = {
        "train": int(CONFIG["experiment_a"]["train_rows"]),
        "validation": int(CONFIG["experiment_a"]["validation_rows"]),
        "test": int(CONFIG["experiment_a"]["test_rows"]),
    }
    return source


def full_split_indices(
    target: np.ndarray, task: str, split_seed: int,
    validation_cap: int = 512, test_cap: int = 512,
) -> dict[str, np.ndarray]:
    """Return the full training partition and completion-compatible eval sets."""

    # Preserve sklearn's original training-index order here.  Sorting it before
    # reapplying completion.capped changes train_test_split's deterministic
    # 2,048-row choice even with the same seed.
    rows = np.arange(len(target))
    stratify = target if task == "classification" else None
    train_validation, test = completion.train_test_split(
        rows, test_size=0.2, random_state=split_seed, stratify=stratify
    )
    second = target[train_validation] if task == "classification" else None
    train, validation = completion.train_test_split(
        train_validation, test_size=0.25, random_state=split_seed + 1,
        stratify=second,
    )
    return {
        "train": train,
        "validation": completion.capped(
            validation, target, validation_cap, task, split_seed + 12
        ),
        "test": completion.capped(test, target, test_cap, task, split_seed + 13),
    }


def stratified_remaining_order(
    indices: np.ndarray, target: np.ndarray, task: str, seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if task != "classification":
        return rng.permutation(indices)
    levels, counts = np.unique(target[indices], return_counts=True)
    queues = {int(level): list(rng.permutation(indices[target[indices] == level])) for level in levels}
    used = {int(level): 0 for level in levels}
    proportions = {int(level): count / len(indices) for level, count in zip(levels, counts)}
    output = []
    for position in range(len(indices)):
        available = [int(level) for level in levels if used[int(level)] < len(queues[int(level)])]
        chosen = max(
            available,
            key=lambda level: ((position + 1) * proportions[level] - used[level], -level),
        )
        output.append(queues[chosen][used[chosen]])
        used[chosen] += 1
    return np.asarray(output, dtype=np.int64)


def nested_training_indices(
    full_train: np.ndarray, target: np.ndarray, task: str, split_seed: int,
) -> np.ndarray:
    """Place the exact completion 2,048 subset first, then add stratified rows."""

    small = completion.capped(full_train, target, 2048, task, split_seed + 11)
    remaining = np.setdiff1d(full_train, small, assume_unique=True)
    ordered_remaining = stratified_remaining_order(
        remaining, target, task, stable_seed("B", split_seed, "nested-order") % (2**32)
    )
    return np.concatenate((small, ordered_remaining))


def prepare_with_indices(
    name: str, indices: dict[str, np.ndarray], config: dict[str, Any],
    numeric: np.ndarray | None = None, categorical: np.ndarray | None = None,
    raw_target: np.ndarray | None = None,
) -> completion.Prepared:
    """Apply the completion preprocessing recipe to caller-supplied partitions."""

    task = config["dataset_tasks"][name]
    if numeric is None or categorical is None or raw_target is None:
        numeric, categorical, raw_target = (
            completion.raw_openml(name, config)
            if name.startswith("openml-")
            else completion.raw_local(name, config)
        )
    assert numeric is not None and categorical is not None and raw_target is not None
    if task == "classification":
        _, target = np.unique(raw_target.astype(str), return_inverse=True)
        target = target.astype(np.int64)
    else:
        target = raw_target.astype(np.float64)
    train_numeric = numeric[indices["train"]]
    medians = np.nanmedian(train_numeric, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    imputed: dict[str, np.ndarray] = {}
    for part in completion.PARTS:
        values = numeric[indices[part]].copy()
        bad = ~np.isfinite(values)
        if bad.any():
            values[bad] = medians[np.where(bad)[1]]
        imputed[part] = values
    means = imputed["train"].mean(axis=0)
    scales = imputed["train"].std(axis=0)
    scales = np.where(scales > 0, scales, 1.0)
    x_num = {
        part: np.ascontiguousarray((imputed[part] - means) / scales, dtype=np.float32)
        for part in completion.PARTS
    }
    x_cat = {
        part: np.empty((len(indices[part]), categorical.shape[1]), dtype=np.int64)
        for part in completion.PARTS
    }
    cardinalities = []
    for column in range(categorical.shape[1]):
        training_values = (
            pd.Series(categorical[indices["train"], column])
            .fillna("__MISSING__").astype(str)
        )
        levels = sorted(training_values.unique().tolist())
        mapping = {value: index for index, value in enumerate(levels)}
        cardinalities.append(len(levels))
        for part in completion.PARTS:
            values = (
                pd.Series(categorical[indices[part], column])
                .fillna("__MISSING__").astype(str)
            )
            x_cat[part][:, column] = np.asarray(
                [mapping.get(value, -1) for value in values], dtype=np.int64
            )
    if task == "regression":
        y_mean = float(target[indices["train"]].mean())
        y_scale = float(target[indices["train"]].std()) or 1.0
        y = {
            part: np.ascontiguousarray(
                (target[indices[part]] - y_mean) / y_scale, dtype=np.float32
            )
            for part in completion.PARTS
        }
    else:
        y_mean, y_scale = 0.0, 1.0
        y = {
            part: np.ascontiguousarray(target[indices[part]], dtype=np.int64)
            for part in completion.PARTS
        }
    return completion.Prepared(name, task, x_num, x_cat, y, cardinalities, y_mean, y_scale)


def b_prepared_datasets(
    name: str, split_seed: int, config: dict[str, Any]
) -> tuple[dict[str, completion.Prepared], dict[str, np.ndarray]]:
    """Create every distinct frozen nested training-size condition."""

    numeric, categorical, raw_target = (
        completion.raw_openml(name, config)
        if name.startswith("openml-")
        else completion.raw_local(name, config)
    )
    task = config["dataset_tasks"][name]
    if task == "classification":
        _, target = np.unique(raw_target.astype(str), return_inverse=True)
    else:
        target = raw_target.astype(np.float64)
    base = full_split_indices(target, task, split_seed)
    ordering = nested_training_indices(base["train"], target, task, split_seed)
    requested = [int(value) for value in CONFIG["experiment_b"]["training_sizes"] if value != "full"]
    sizes = [size for size in requested if size < len(ordering)] + [len(ordering)]
    sizes = sorted(set(sizes))
    prepared: dict[str, completion.Prepared] = {}
    raw_indices: dict[str, np.ndarray] = {}
    for size in sizes:
        label = "full" if size == len(ordering) else str(size)
        current = {
            "train": np.sort(ordering[:size]),
            "validation": base["validation"],
            "test": base["test"],
        }
        prepared[label] = prepare_with_indices(
            name, current, config, numeric, categorical, raw_target
        )
        raw_indices[label] = current["train"]
    return prepared, raw_indices


def model_config_hash(model: str, training: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps({"model": model, "training": training}, sort_keys=True).encode()
    ).hexdigest()


def schema_hash(dataset: str, action: Iterable[int]) -> str:
    return stable_hash("schema", dataset, *[int(value) for value in action])


def fit_key(
    *, dataset: str, split: int, model: str, model_hash: str,
    schema_digest: str, master_seed: int | None, training_size: int,
    training_budget: str, matched_arm: str = "ordinary",
    finite_init_seed: int | None = None, finite_order_seed: int | None = None,
) -> str:
    return stable_hash(
        dataset, split, model, model_hash, schema_digest, master_seed,
        training_size, training_budget, matched_arm, finite_init_seed,
        finite_order_seed,
    )


def registry_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(REGISTRY_PATH, timeout=120)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS fits (
          fit_key TEXT PRIMARY KEY,
          status TEXT NOT NULL,
          experiment TEXT NOT NULL,
          dataset TEXT NOT NULL,
          split_seed INTEGER NOT NULL,
          model TEXT NOT NULL,
          master_seed INTEGER,
          artifact TEXT NOT NULL,
          array_index TEXT NOT NULL,
          prediction_sha256 TEXT NOT NULL,
          wall_seconds REAL NOT NULL,
          peak_device_bytes INTEGER NOT NULL,
          recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.commit()
    return connection


def prediction_digest(validation: np.ndarray, test: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (validation, test):
        current = np.ascontiguousarray(array)
        digest.update(str(current.shape).encode())
        digest.update(current.dtype.str.encode())
        digest.update(current.tobytes())
    return digest.hexdigest()


def register_fit(
    *, key: str, experiment: str, dataset: str, split_seed: int, model: str,
    master_seed: int | None, artifact: Path, array_index: str,
    validation: np.ndarray, test: np.ndarray, wall_seconds: float,
    peak_device_bytes: int,
) -> None:
    with registry_connection() as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO fits
            (fit_key,status,experiment,dataset,split_seed,model,master_seed,
             artifact,array_index,prediction_sha256,wall_seconds,peak_device_bytes)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                key, "complete", experiment, dataset, int(split_seed), model,
                master_seed, str(artifact.relative_to(REPO)), array_index,
                prediction_digest(validation, test), float(wall_seconds),
                int(peak_device_bytes),
            ),
        )


def open_memmap(path: Path, shape: tuple[int, ...], dtype: np.dtype[Any]) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        array = np.lib.format.open_memmap(path, mode="r+")
        if array.shape != shape or array.dtype != np.dtype(dtype):
            raise AssertionError(f"resume array mismatch at {path}: {array.shape}/{array.dtype}")
        return array
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def initialize_model(
    model_name: str, input_width: int, output_width: int,
    init_seed: int, config: dict[str, Any], device: torch.device,
) -> torch.nn.Module:
    random.seed(int(init_seed))
    np.random.seed(int(init_seed) % (2**32))
    torch.manual_seed(int(init_seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(init_seed))
    return completion.make_model(model_name, input_width, output_width, config).to(device)


def evaluate_loss(
    model: torch.nn.Module, x: np.ndarray, y: np.ndarray, task: str,
    model_name: str, device: torch.device, batch_size: int = 2048,
) -> float:
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(device)
            if task == "classification":
                yb = torch.from_numpy(y[start : start + batch_size].astype(np.int64)).to(device)
            else:
                yb = torch.from_numpy(y[start : start + batch_size].astype(np.float32)).to(device)
            loss = completion.loss_value(completion.forward(model, xb, model_name), yb, task, model_name)
            total += float(loss.detach().cpu()) * len(xb)
    return total / len(x)


def fit_fixed(
    model: torch.nn.Module, x: np.ndarray, y: np.ndarray, task: str,
    model_name: str, subseeds: dict[str, int], training: dict[str, Any],
    device: torch.device,
) -> tuple[float, int]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    batch = int(training["batch_size"])
    order_rng = np.random.default_rng(int(subseeds["dataloader"]))
    torch.manual_seed(int(subseeds["dropout"]))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(subseeds["dropout"]))
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(int(training["epochs"])):
        model.train()
        order = order_rng.permutation(len(x))
        for start in range(0, len(x), batch):
            chosen = order[start : start + batch]
            xb = torch.from_numpy(x[chosen]).to(device)
            if task == "classification":
                yb = torch.from_numpy(y[chosen].astype(np.int64)).to(device)
            else:
                yb = torch.from_numpy(y[chosen].astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = completion.loss_value(
                completion.forward(model, xb, model_name), yb, task, model_name
            )
            loss.backward()
            optimizer.step()
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return elapsed, peak


def fit_diagnostic(
    model: torch.nn.Module, x_train: np.ndarray, y_train: np.ndarray,
    x_validation: np.ndarray, y_validation: np.ndarray, task: str,
    model_name: str, subseeds: dict[str, int], training: dict[str, Any],
    device: torch.device, budget: int | str,
) -> tuple[float, int, list[dict[str, float]], int, int]:
    """Train a fixed or early-stopped condition and return epoch telemetry."""

    convergence = CONFIG["experiment_b"]["convergence"]
    converged = budget == "convergence"
    maximum_epochs = int(convergence["maximum_epochs"] if converged else budget)
    patience = int(convergence["patience"])
    relative_minimum = float(convergence["relative_minimum_improvement"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    batch = int(training["batch_size"])
    order_rng = np.random.default_rng(int(subseeds["dataloader"]))
    torch.manual_seed(int(subseeds["dropout"]))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(subseeds["dropout"]))
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    trajectory: list[dict[str, float]] = []
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        order = order_rng.permutation(len(x_train))
        total_loss = 0.0
        total_grad_squared = 0.0
        before = [parameter.detach().clone() for parameter in model.parameters()]
        batches = 0
        for start in range(0, len(x_train), batch):
            chosen = order[start : start + batch]
            xb = torch.from_numpy(x_train[chosen]).to(device)
            if task == "classification":
                yb = torch.from_numpy(y_train[chosen].astype(np.int64)).to(device)
            else:
                yb = torch.from_numpy(y_train[chosen].astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = completion.loss_value(
                completion.forward(model, xb, model_name), yb, task, model_name
            )
            loss.backward()
            grad_squared = 0.0
            for parameter in model.parameters():
                if parameter.grad is not None:
                    grad_squared += float(torch.sum(parameter.grad.detach() ** 2).cpu())
            total_grad_squared += grad_squared
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(xb)
            batches += 1
        update_squared = 0.0
        for previous, current in zip(before, model.parameters()):
            update_squared += float(torch.sum((current.detach() - previous) ** 2).cpu())
        validation_loss = evaluate_loss(
            model, x_validation, y_validation, task, model_name, device
        )
        trajectory.append(
            {
                "epoch": float(epoch),
                "training_loss": total_loss / len(x_train),
                "validation_loss": validation_loss,
                "gradient_norm": (total_grad_squared / max(batches, 1)) ** 0.5,
                "parameter_update_norm": update_squared ** 0.5,
            }
        )
        threshold = best_loss * (1.0 - relative_minimum)
        if validation_loss < threshold:
            best_loss = validation_loss
            best_epoch = epoch
            stale = 0
            if converged:
                best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if converged and stale >= patience:
            break
    stopped_epoch = len(trajectory)
    if converged and bool(convergence["restore_best"]) and best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return elapsed, peak, trajectory, best_epoch, stopped_epoch


def train_predict(
    *, data: completion.Prepared, design: dict[str, list[Any]], schema_action: tuple[int, int, int],
    model_name: str, master_seed: int, config: dict[str, Any], device: torch.device,
    training: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, float, int, dict[str, int]]:
    fi, ci, li = [int(value) for value in schema_action]
    class_map = design["class"][li]
    rendered = {
        part: completion.render(data, part, design["feature"][fi], design["category"][ci])[0]
        for part in completion.PARTS
    }
    output_width = 2 if data.task == "classification" else 1
    current_training = dict(config["training"] if training is None else training)
    subseeds = derive_subseeds(int(master_seed))
    model = initialize_model(
        model_name, rendered["train"].shape[1], output_width,
        subseeds["initialization"], config, device,
    )
    transformed_y = class_map[data.y["train"]] if data.task == "classification" else data.y["train"]
    elapsed, peak = fit_fixed(
        model, rendered["train"], transformed_y, data.task, model_name,
        subseeds, current_training, device,
    )
    validation = completion.predict(
        model, rendered["validation"], data.task, model_name, class_map, device
    )
    test = completion.predict(model, rendered["test"], data.task, model_name, class_map, device)
    if not np.isfinite(validation).all() or not np.isfinite(test).all():
        raise AssertionError("non-finite prediction")
    return validation, test, elapsed, peak, subseeds


def schema_actions(data: completion.Prepared, design: dict[str, list[Any]]) -> list[tuple[int, int, int]]:
    return list(np.ndindex(
        len(design["feature"]), len(design["category"]), len(design["class"])
    ))


def action_cards(data: completion.Prepared, design: dict[str, list[Any]]) -> tuple[int, int, int]:
    return (len(design["feature"]), len(design["category"]), len(design["class"]))


def save_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def validate_probabilities(predictions: np.ndarray, task: str) -> None:
    if not np.isfinite(predictions).all():
        raise AssertionError("non-finite predictions")
    if task == "classification":
        if predictions.min() < -1e-6 or predictions.max() > 1.0 + 1e-6:
            raise AssertionError("classification probabilities outside [0,1]")
        error = np.max(np.abs(predictions.sum(axis=-1) - 1.0))
        if error > 2e-5:
            raise AssertionError(f"classification probability sum error {error}")


def target_loss(prediction: np.ndarray, target: np.ndarray, task: str) -> float:
    if task == "classification":
        onehot = np.eye(prediction.shape[-1], dtype=np.float64)[target.astype(int)]
        return float(np.mean(np.sum((prediction - onehot) ** 2, axis=-1)))
    return float(np.mean((prediction.reshape(-1) - target.reshape(-1)) ** 2))


def prediction_residual(prediction: np.ndarray, reference: np.ndarray) -> float:
    return float(np.mean(np.sum((prediction - reference) ** 2, axis=-1)))
