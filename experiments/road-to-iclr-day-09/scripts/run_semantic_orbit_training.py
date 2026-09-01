#!/usr/bin/env python3
"""Conditional training-time ablations on the strongest T6 effect.

The panel uses condition-one (orthogonal) basis changes so robustness cannot be attributed
to suppressing ill-conditioning.  Results live outside the frozen primary verdict grid.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
import yaml


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from src.semantic_orbits import (  # noqa: E402
    build_representations,
    disagreement_metrics,
    jsonable,
    load_dataset,
    prediction_metrics,
    sha256_file,
)


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class MLP(nn.Module):
    def __init__(self, n_features: int, width: int, layers: int, dropout: float):
        super().__init__()
        blocks: list[nn.Module] = []
        size = n_features
        for _ in range(layers):
            blocks.extend([nn.Linear(size, width), nn.GELU(), nn.Dropout(dropout)])
            size = width
        blocks.append(nn.Linear(size, 1))
        self.network = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def impute_before_orbit(split: Any) -> dict[str, Any]:
    details = {}
    for column in split.numerical_columns:
        train = pd.to_numeric(split.X_train_numeric[column], errors="coerce")
        fill = float(train.median())
        count = 0
        for frame in (split.X_train_numeric, split.X_validation_numeric, split.X_test_numeric):
            count += int(frame[column].isna().sum())
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(fill)
        if count:
            details[column] = {"count": count, "train_median": fill}
    return details


def get_views(split: Any, orbit_members: int) -> tuple[list[Any], dict[str, np.ndarray]]:
    reps = build_representations(split, "numeric_code", orbit_members)
    members = sorted(
        [rep for rep in reps if rep.family == "T6" and rep.variant == "orthogonal"],
        key=lambda rep: rep.member,
    )
    if len(members) != orbit_members:
        raise RuntimeError(f"expected {orbit_members} orthogonal members, found {len(members)}")
    reference_id = members[0].reference_id
    reference = next(rep for rep in reps if rep.representation_id == reference_id)
    ordered = [reference, *members]
    arrays: dict[str, np.ndarray] = {}
    for split_name in ("train", "validation", "test"):
        frames = [getattr(rep, f"X_{split_name}") for rep in ordered]
        arrays[split_name] = np.stack([frame.to_numpy(dtype=np.float32) for frame in frames])
    # Algebraic audit: invert A phi(x) and confirm exact recovery of the RBF reference block.
    feature = str(reference.metadata["feature"])
    rbf_columns = [f"{feature}__rbf{index}" for index in range(8)]
    positions = [reference.X_train.columns.get_loc(column) for column in rbf_columns]
    max_delta = 0.0
    for rep_index, rep in enumerate(members, start=1):
        matrix = np.asarray(rep.metadata["matrix"], dtype=float)
        recovered = arrays["train"][rep_index][:, positions] @ np.linalg.inv(matrix.T)
        delta = float(np.max(np.abs(recovered - arrays["train"][0][:, positions])))
        max_delta = max(max_delta, delta)
    if max_delta > 2e-5:
        raise RuntimeError(f"basis inverse audit failed: {max_delta}")
    return ordered, {**arrays, "inverse_audit_max_delta": np.asarray(max_delta)}


def train_method(
    method: str,
    views: dict[str, np.ndarray],
    y_train: np.ndarray,
    y_validation: np.ndarray,
    seed: int,
    device: str,
    cfg: dict[str, Any],
) -> tuple[nn.Module, dict[str, Any], np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    train = views["train"].astype(np.float32)
    validation = views["validation"].astype(np.float32)
    mean = train[0].mean(axis=0)
    scale = train[0].std(axis=0)
    scale[scale < 1e-6] = 1.0
    train = (train - mean[None, None, :]) / scale[None, None, :]
    validation = (validation - mean[None, None, :]) / scale[None, None, :]
    y_mean, y_scale = float(np.mean(y_train)), max(float(np.std(y_train)), 1e-8)
    target = ((np.asarray(y_train) - y_mean) / y_scale).astype(np.float32)
    val_target = ((np.asarray(y_validation) - y_mean) / y_scale).astype(np.float32)
    dual = method == "dual_view"
    model = MLP(
        train.shape[-1] * (2 if dual else 1), int(cfg["hidden_width"]),
        int(cfg["hidden_layers"]), float(cfg["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])
    )
    loss_fn = nn.MSELoss()
    batch_size = int(cfg["batch_size"])
    best_state = None
    best_loss = float("inf")
    best_epoch = -1
    no_improvement = 0
    lambda_value = float(method.rsplit("_", 1)[1]) if method.startswith("consistency_") else 0.0
    started = time.perf_counter()
    for epoch in range(int(cfg["max_epochs"])):
        model.train()
        order = rng.permutation(len(target))
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            chosen = rng.integers(1, train.shape[0], size=len(idx))
            base_np = train[0, idx]
            if method in ("orbit_augmentation", "dual_view"):
                raw_np = train[chosen, idx]
            else:
                raw_np = base_np
            x = torch.as_tensor(raw_np, device=device)
            if method == "generic_noise":
                x = x + float(cfg["generic_noise_std"]) * torch.randn_like(x)
            if dual:
                x = torch.cat([x, torch.as_tensor(base_np, device=device)], dim=1)
            y = torch.as_tensor(target[idx], device=device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            if method.startswith("consistency_"):
                transformed = torch.as_tensor(train[chosen, idx], device=device)
                loss = loss + lambda_value * loss_fn(pred, model(transformed))
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            losses = []
            validation_view_ids = range(validation.shape[0]) if method in (
                "orbit_augmentation", "dual_view", "consistency_0.1", "consistency_1.0"
            ) else (0,)
            yv = torch.as_tensor(val_target, device=device)
            for view_id in validation_view_ids:
                if method == "canonical_only":
                    value = validation[0]
                elif dual:
                    value = np.concatenate([validation[view_id], validation[0]], axis=1)
                else:
                    value = validation[view_id]
                losses.append(float(loss_fn(model(torch.as_tensor(value, device=device)), yv).item()))
            validation_loss = float(np.mean(losses))
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= int(cfg["patience"]):
                break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "fit_seconds": time.perf_counter() - started,
        "best_epoch": best_epoch,
        "best_validation_mse_normalized": best_loss,
        "lambda": lambda_value,
    }, mean, scale


def predict_views(
    model: nn.Module, method: str, views: np.ndarray, mean: np.ndarray, scale: np.ndarray,
    y_train: np.ndarray, device: str,
) -> np.ndarray:
    standardized = (views.astype(np.float32) - mean[None, None, :]) / scale[None, None, :]
    y_mean, y_scale = float(np.mean(y_train)), max(float(np.std(y_train)), 1e-8)
    predictions = []
    model.eval()
    with torch.no_grad():
        for view_id in range(len(standardized)):
            if method == "canonical_only":
                value = standardized[0]
            elif method == "dual_view":
                value = np.concatenate([standardized[view_id], standardized[0]], axis=1)
            else:
                value = standardized[view_id]
            pred = model(torch.as_tensor(value, device=device)).cpu().numpy() * y_scale + y_mean
            predictions.append(pred)
    return np.stack(predictions)


def metadata_environment() -> dict[str, Any]:
    packages = {}
    for package in ("numpy", "pandas", "torch", "scikit-learn"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": platform.python_version(), "platform": platform.platform(), "packages": packages,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def run_bundle(config: dict[str, Any], config_path: Path, config_hash: str, spec: dict[str, Any], seed: int, device: str) -> str:
    destination = ROOT / "results" / "semantic_orbits" / "training_ablations" / spec["name"] / f"seed_{seed}"
    if (destination / "metadata.json").exists():
        metadata = json.loads((destination / "metadata.json").read_text())
        if metadata.get("config_sha256") != config_hash:
            raise RuntimeError(f"config drift at {destination}")
        print(f"[cached] {spec['name']} seed={seed}", flush=True)
        return "cached"
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite incomplete immutable bundle: {destination}")
    split = load_dataset(spec, config)
    imputation = impute_before_orbit(split)
    reps, views = get_views(split, int(config["orbit_members"]))
    prediction_parts = []
    metric_rows = []
    method_telemetry = {}
    started = time.time()
    for method in config["methods"]:
        model, telemetry, mean, scale = train_method(
            method, views, split.y_train, split.y_validation, seed, device, config["model"]
        )
        predictions = predict_views(model, method, views["test"], mean, scale, split.y_train, device)
        method_telemetry[method] = telemetry
        reference_prediction = predictions[0]
        for view_id, rep in enumerate(reps):
            task = prediction_metrics("regression", split.y_test, predictions[view_id])
            disagreement = disagreement_metrics("regression", split.y_test, reference_prediction, predictions[view_id])
            metric_rows.append({
                "dataset": split.name, "model": "mlp", "model_seed": seed, "method": method,
                "family": rep.family, "variant": rep.variant, "scope": rep.scope,
                "representation_id": rep.representation_id, "reference_id": reps[0].representation_id,
                "member": rep.member, "is_reference": rep.is_reference,
                **task, **disagreement, **telemetry,
            })
            prediction_parts.append(pd.DataFrame({
                "dataset": split.name, "model": "mlp", "model_seed": seed, "method": method,
                "representation_id": rep.representation_id, "reference_id": reps[0].representation_id,
                "member": rep.member, "is_reference": rep.is_reference,
                "test_row_id": split.test_indices, "target": split.y_test, "prediction": predictions[view_id],
            }))
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"[{split.name} seed={seed}] {method}: {telemetry}", flush=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        pd.DataFrame(metric_rows).to_csv(temporary / "metrics.csv", index=False)
        pd.concat(prediction_parts, ignore_index=True).to_csv(
            temporary / "predictions.csv.gz", index=False, compression="gzip"
        )
        files = {}
        for filename in ("metrics.csv", "predictions.csv.gz"):
            path = temporary / filename
            files[filename] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        git = {
            "commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, check=True, capture_output=True, text=True
            ).stdout.strip(),
            "day09_status": subprocess.run(
                ["git", "status", "--short", "--", str(ROOT.relative_to(REPOSITORY))],
                cwd=REPOSITORY, check=True, capture_output=True, text=True,
            ).stdout.strip(),
        }
        metadata = {
            "status": "complete", "experiment": config["experiment"], "protocol_version": config["protocol_version"],
            "config": str(config_path.relative_to(ROOT)), "config_sha256": config_hash,
            "dataset_spec": spec, "model_seed": seed, "device": device,
            "methods": list(config["methods"]), "method_telemetry": method_telemetry,
            "representation_count_per_method": len(reps), "wall_seconds": time.time() - started,
            "preprocessing": {"continuous_missing": "training median before orbit construction", "imputed_columns": imputation},
            "inverse_audit_max_delta": float(views["inverse_audit_max_delta"]),
            "split_audit": {
                "train_validation_disjoint": not bool(set(split.train_indices) & set(split.validation_indices)),
                "train_test_disjoint": not bool(set(split.train_indices) & set(split.test_indices)),
                "validation_test_disjoint": not bool(set(split.validation_indices) & set(split.test_indices)),
                "test_row_order_sha256": digest_bytes(split.test_indices.tobytes()),
                "target_sha256": digest_bytes(np.ascontiguousarray(split.y_test).view(np.uint8)),
            },
            "environment": metadata_environment(), "git": git, "files": files,
        }
        (temporary / "metadata.json").write_text(json.dumps(jsonable(metadata), indent=2, sort_keys=True) + "\n")
        os.rename(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return "complete"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "semantic_orbit_training.yaml")
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    config_hash = digest_bytes(config_bytes)
    datasets = [spec for spec in config["datasets"] if not args.dataset or spec["name"] in set(args.dataset)]
    seeds = [int(seed) for seed in config["model_seeds"] if not args.seed or int(seed) in set(args.seed)]
    counts = {"complete": 0, "cached": 0, "failed": 0}
    failures = []
    for spec in datasets:
        for seed in seeds:
            try:
                status = run_bundle(config, args.config.resolve(), config_hash, spec, seed, args.device)
                counts[status] += 1
            except Exception as error:
                counts["failed"] += 1
                failures.append({"dataset": spec["name"], "seed": seed, "error": repr(error), "traceback": traceback.format_exc()})
                print(json.dumps(failures[-1]), flush=True)
    print(json.dumps({"counts": counts, "failures": failures}, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
