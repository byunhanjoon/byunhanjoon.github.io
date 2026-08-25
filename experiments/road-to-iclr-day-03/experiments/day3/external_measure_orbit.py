"""Untouched external, exactly update-matched Selective Measure-Orbit test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .core import Dataset, PARTS, Prepared, loss_numpy, make_prepared, metric
from .measure_orbit import (
    MeasureViewTabM,
    VIEW_NAMES,
    _member_tensor,
    _predict,
    config as method_config,
)
from .orbit_ensemble import _ensemble_numpy
from .selective_measure_orbit import build_views


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    ROOT / "experiments/day3/configs/external_measure_orbit_preregistered.json"
)
RESULTS = ROOT / "results/day3/external_measure_orbit"
SOURCE_PATHS = {
    "measure_orbit_config_sha256": ROOT
    / "experiments/day3/configs/measure_orbit_preregistered.json",
    "measure_orbit_code_sha256": ROOT / "experiments/day3/measure_orbit.py",
    "selective_measure_orbit_code_sha256": ROOT
    / "experiments/day3/selective_measure_orbit.py",
}


def config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_locked_sources() -> None:
    expected = config()["locked_method"]["source_hashes"]
    observed = {name: _sha256(path) for name, path in SOURCE_PATHS.items()}
    if observed != expected:
        raise RuntimeError(
            f"Locked Measure-Orbit sources changed after freeze: {observed} != {expected}"
        )


def _indices(length: int, limit: int, seed: int) -> np.ndarray:
    values = np.arange(length, dtype=np.int64)
    if length <= limit:
        return values
    return np.sort(np.random.default_rng(seed).choice(values, limit, replace=False))


def load_external_dataset(name: str) -> Dataset:
    cfg = config()
    if name not in cfg["datasets"]:
        raise KeyError(f"Dataset {name!r} is absent from the frozen external panel")
    spec = cfg["datasets"][name]
    directory = Path(spec["path"])
    task = str(spec["task"])
    if task not in ("binclass", "multiclass", "regression"):
        raise ValueError(f"Unsupported task {task!r}")

    indices: dict[str, np.ndarray] = {}
    for offset, part in enumerate(PARTS):
        y_path = directory / f"y_{part}.npy"
        length = len(np.load(y_path, mmap_mode="r", allow_pickle=True))
        limit = int(
            cfg["data"]["max_train_rows"]
            if part == "train"
            else cfg["data"]["max_eval_rows"]
        )
        indices[part] = _indices(
            length, limit, int(cfg["data"]["sample_seed"]) + offset
        )

    def arrays(stem: str | None, *, allow_pickle: bool = False):
        if stem is None or not (directory / f"{stem}_train.npy").exists():
            return None
        return {
            part: np.asarray(
                np.load(
                    directory / f"{stem}_{part}.npy",
                    mmap_mode=None if allow_pickle else "r",
                    allow_pickle=allow_pickle,
                )[indices[part]]
            )
            for part in PARTS
        }

    x_num = arrays(spec["numeric_stem"])
    x_cat = arrays(spec["categorical_stem"], allow_pickle=True)
    y_raw = arrays("y", allow_pickle=True)
    assert y_raw is not None
    if task == "regression":
        y = {part: value.astype(np.float32) for part, value in y_raw.items()}
        n_classes = 1
    else:
        classes = sorted(set(y_raw["train"].tolist()), key=str)
        lookup = {value: index for index, value in enumerate(classes)}
        y = {}
        for part, values in y_raw.items():
            if any(value not in lookup for value in values.tolist()):
                raise ValueError(f"Unseen target class in {name}/{part}")
            y[part] = np.asarray(
                [lookup[value] for value in values.tolist()], dtype=np.int64
            )
        n_classes = 2 if task == "binclass" else len(classes)

    digest = hashlib.sha256()
    digest.update(str(directory.resolve()).encode())
    for part in PARTS:
        digest.update(indices[part].tobytes())
    return Dataset(
        name=name,
        task=task,
        x_num=x_num,
        x_bin=None,
        x_cat=x_cat,
        y=y,
        n_classes=n_classes,
        split_fingerprint=digest.hexdigest()[:16],
    )


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _upsert(
    path: Path,
    rows: list[dict[str, object]],
    row: dict[str, object],
) -> None:
    key = (str(row["dataset"]), int(row["seed"]), str(row["arm"]))
    rows[:] = [
        old
        for old in rows
        if (str(old["dataset"]), int(old["seed"]), str(old["arm"])) != key
    ]
    rows.append(row)
    _write(path, rows)


def prediction_path(dataset: str, seed: int, arm: str) -> Path:
    return RESULTS / "predictions" / f"{dataset}__s{seed}__{arm}.npz"


def _save_predictions(
    path: Path, validation: np.ndarray, test: np.ndarray
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            val=np.asarray(validation, dtype=np.float32),
            test=np.asarray(test, dtype=np.float32),
        )
    temporary.replace(path)


def train_arm(
    prepared: Prepared,
    views: dict[str, dict[str, np.ndarray]],
    arm: str,
    seed: int,
    device: str,
    *,
    forced_epochs: int | None = None,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cfg = method_config()["training"]
    resolved = torch.device(device)
    assignment = (
        ["baseline_fixed_ple"] * len(method_config()["member_assignment"])
        if arm in ("baseline_anchor", "baseline_seedmate_update_matched")
        else list(method_config()["member_assignment"])
    )
    model = MeasureViewTabM(
        prepared.x["train"].shape[1],
        prepared.n_classes if prepared.task == "multiclass" else 1,
    ).to(resolved)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    batch_size = int(
        cfg["large_batch_size"]
        if len(prepared.y["train"]) >= int(cfg["large_dataset_threshold"])
        else cfg["batch_size"]
    )
    tensors = [torch.from_numpy(views[name]["train"]) for name in VIEW_NAMES]
    loader = DataLoader(
        TensorDataset(*tensors, torch.from_numpy(prepared.y["train"])),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 70000),
        pin_memory=resolved.type == "cuda",
    )
    maximum = int(forced_epochs or cfg["max_epochs"])
    best_loss, best_epoch, stale, best_state = math.inf, 0, 0, None
    started = time.perf_counter()
    epochs_executed = 0
    for epoch in range(1, maximum + 1):
        epochs_executed = epoch
        model.train()
        for *feature_values, target_values in loader:
            batch = {
                name: value.to(resolved, non_blocking=True)
                for name, value in zip(VIEW_NAMES, feature_values)
            }
            target_values = target_values.to(resolved, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            members = model.forward_members(_member_tensor(batch, assignment))
            if prepared.task == "binclass":
                target_binary = target_values.float()
                member_logits = members.squeeze(-1)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    member_logits,
                    target_binary[:, None].expand_as(member_logits),
                )
            elif prepared.task == "multiclass":
                expanded = target_values.long()[:, None].expand(-1, members.shape[1])
                loss = nn.functional.cross_entropy(
                    members.flatten(0, 1), expanded.flatten()
                )
            else:
                target_continuous = target_values.float()
                member_values = members.squeeze(-1)
                loss = nn.functional.mse_loss(
                    member_values,
                    target_continuous[:, None].expand_as(member_values),
                )
            loss.backward()
            optimizer.step()

        val_members = _predict(
            model,
            {name: value["val"] for name, value in views.items()},
            assignment,
            resolved,
            batch_size * 2,
        )
        val_prediction = _ensemble_numpy(val_members, prepared.task)
        val_loss = loss_numpy(prepared.task, val_prediction, prepared.y["val"])
        if not np.isfinite(val_loss):
            raise FloatingPointError(f"Non-finite validation loss for {arm}")
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        if forced_epochs is None and stale > int(cfg["patience"]):
            break

    if best_state is None:
        raise RuntimeError("No finite checkpoint")
    model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    predictions: dict[str, np.ndarray] = {}
    result: dict[str, object] = {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "epochs_trained": epochs_executed,
        "batches_per_epoch": len(loader),
        "gradient_updates": epochs_executed * len(loader),
        "forced_epoch_budget": forced_epochs if forced_epochs is not None else "",
        "train_seconds": elapsed,
    }
    for part in ("val", "test"):
        members = _predict(
            model,
            {name: value[part] for name, value in views.items()},
            assignment,
            resolved,
            batch_size * 2,
        )
        prediction = _ensemble_numpy(members, prepared.task)
        predictions[part] = prediction
        result[f"{part}_proper_loss"] = loss_numpy(
            prepared.task, prediction, prepared.y[part]
        )
        result[f"{part}_metric"] = metric(
            prepared, prediction, prepared.y[part]
        )
    return result, predictions


def _existing(
    rows: list[dict[str, object]], dataset: str, seed: int, arm: str
) -> dict[str, object] | None:
    for row in rows:
        if (
            str(row["dataset"]) == dataset
            and int(row["seed"]) == seed
            and str(row["arm"]) == arm
            and prediction_path(dataset, seed, arm).exists()
        ):
            return row
    return None


def run(args: argparse.Namespace) -> None:
    verify_locked_sources()
    cfg = config()
    datasets = args.datasets or list(cfg["datasets"])
    seeds = args.seeds or list(cfg["seeds"])
    rows: list[dict[str, object]] = list(_read(args.output))
    selected = [
        name
        for index, name in enumerate(datasets)
        if index % args.num_shards == args.shard
    ]
    for dataset_name in selected:
        dataset = load_external_dataset(dataset_name)
        for seed_value in seeds:
            seed = int(seed_value)
            views, metadata = build_views(dataset, seed)
            prepared = make_prepared(dataset, views["baseline_fixed_ple"], {})
            specifications = [
                ("baseline_anchor", seed, None),
                ("measure_orbit", seed, None),
            ]
            results: dict[str, dict[str, object]] = {}
            for arm, training_seed, forced in specifications:
                prior = _existing(rows, dataset_name, seed, arm)
                if prior is not None:
                    results[arm] = prior
                    continue
                result, predictions = train_arm(
                    prepared,
                    views,
                    arm,
                    training_seed,
                    args.device,
                    forced_epochs=forced,
                )
                _save_predictions(
                    prediction_path(dataset_name, seed, arm),
                    predictions["val"],
                    predictions["test"],
                )
                row = {
                    "hypothesis": "external_selective_measure_orbit",
                    "dataset": dataset_name,
                    "task": dataset.task,
                    "seed": seed,
                    "training_seed": training_seed,
                    "arm": arm,
                    "budget_per_numeric": metadata["budget"],
                    "input_features": prepared.x["train"].shape[1],
                    "train_rows": len(prepared.y["train"]),
                    "val_rows": len(prepared.y["val"]),
                    "test_rows": len(prepared.y["test"]),
                    "split_fingerprint": dataset.split_fingerprint,
                    "failure": "",
                    **result,
                }
                _upsert(args.output, rows, row)
                results[arm] = row
                print(
                    f"{dataset_name} s{seed} {arm} "
                    f"epochs={result['epochs_trained']} "
                    f"val={float(result['val_proper_loss']):.6f} "
                    f"test={float(result['test_proper_loss']):.6f}",
                    flush=True,
                )

            orbit_epochs = int(results["measure_orbit"]["epochs_trained"])
            arm = "baseline_seedmate_update_matched"
            prior = _existing(rows, dataset_name, seed, arm)
            if prior is None:
                training_seed = seed + int(cfg["seedmate_offset"])
                result, predictions = train_arm(
                    prepared,
                    views,
                    arm,
                    training_seed,
                    args.device,
                    forced_epochs=orbit_epochs,
                )
                _save_predictions(
                    prediction_path(dataset_name, seed, arm),
                    predictions["val"],
                    predictions["test"],
                )
                row = {
                    "hypothesis": "external_selective_measure_orbit",
                    "dataset": dataset_name,
                    "task": dataset.task,
                    "seed": seed,
                    "training_seed": training_seed,
                    "arm": arm,
                    "budget_per_numeric": metadata["budget"],
                    "input_features": prepared.x["train"].shape[1],
                    "train_rows": len(prepared.y["train"]),
                    "val_rows": len(prepared.y["val"]),
                    "test_rows": len(prepared.y["test"]),
                    "split_fingerprint": dataset.split_fingerprint,
                    "failure": "",
                    **result,
                }
                _upsert(args.output, rows, row)
                print(
                    f"{dataset_name} s{seed} {arm} "
                    f"epochs={result['epochs_trained']} "
                    f"val={float(result['val_proper_loss']):.6f} "
                    f"test={float(result['test_proper_loss']):.6f}",
                    flush=True,
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, default=RESULTS / "runs.csv")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
