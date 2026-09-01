#!/usr/bin/env python3
"""Run Experiment B with paired minibatches and function-matched initialization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.basis_dependence import (  # noqa: E402
    build_primary_representations, build_rbf_feature_matrix, disagreement_metrics,
    environment_metadata, jsonable, load_dataset, prediction_metrics, sha256_file,
)
from src.mechanism import (  # noqa: E402
    function_matched_copy, make_controlled_mlp, make_optimizer, max_logit_difference,
    minibatch_orders, optimizer_conditions, order_sha256,
)


CONFIG_PATH = ROOT / "configs" / "development_protocol.yaml"
PANEL_PATH = ROOT / "configs" / "dataset_panel.json"
CHECKPOINT_EPOCHS = {0, 1, 2, 5, 10, 20}


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def predictions(model: Any, X: np.ndarray, problem_type: str, y_mean: float, y_scale: float, device: str) -> np.ndarray:
    import torch

    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)).squeeze(-1)
        if problem_type == "regression":
            return logits.cpu().numpy() * y_scale + y_mean
        return torch.softmax(logits, dim=1).cpu().numpy()


def prediction_rows(
    dataset: str, seed: int, member: int, condition: str, epoch: int | str, split: str,
    row_ids: np.ndarray, target: np.ndarray, side: str, values: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame({
        "dataset": dataset, "model_seed": seed, "orbit_member": member,
        "condition": condition, "epoch": str(epoch), "split": split, "side": side,
        "row_id": row_ids, "target": target,
    })
    values = np.asarray(values)
    if values.ndim == 1:
        frame["prediction"] = values
    else:
        for class_index in range(values.shape[1]):
            frame[f"prediction_{class_index}"] = values[:, class_index]
    return frame


def run_bundle(
    config: dict[str, Any], config_hash: str, panel_hash: str, spec: dict[str, Any],
    seed: int, member: int, condition_name: str, device: str,
) -> str:
    if spec["panel"] != "development":
        raise RuntimeError("mechanism runner refuses prospective datasets")
    destination = (ROOT / "results" / "raw" / "development" / "mechanism" / spec["key"] /
                   f"seed_{seed}" / f"member_{member}" / condition_name)
    if (destination / "metadata.json").exists():
        metadata = json.loads((destination / "metadata.json").read_text())
        if metadata["config_sha256"] != config_hash or metadata["dataset_panel_sha256"] != panel_hash:
            raise RuntimeError(f"frozen config drift at {destination}")
        return "cached"
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite incomplete bundle: {destination}")

    data = load_dataset(spec, config)
    blocks = build_rbf_feature_matrix(data, config)
    reps = build_primary_representations(blocks, 4)
    reference = reps[0]
    transformed = next(
        rep for rep in reps if rep.variant == "orthogonal_all" and rep.member == member
    )
    model_config = config["models"]["controlled_mlp"]
    conditions = {item.name: item for item in optimizer_conditions(float(model_config["weight_decay"]))}
    condition = conditions[condition_name]
    output_dimension = 1 if data.problem_type == "regression" else int(np.max(data.y_train)) + 1
    reference_model = make_controlled_mlp(reference.X_train.shape[1], output_dimension, model_config, seed, device)
    if condition.function_matched:
        transformed_model = function_matched_copy(reference_model, transformed, device)
    else:
        transformed_model = make_controlled_mlp(transformed.X_train.shape[1], output_dimension, model_config, seed, device)
    initial_difference = max_logit_difference(
        reference_model, transformed_model, reference.X_train, transformed.X_train, device
    )
    if condition.function_matched and initial_difference >= 1e-5:
        raise RuntimeError(f"function matching failed: {initial_difference}")

    import torch
    from torch import nn

    if data.problem_type == "regression":
        y_mean = float(np.mean(data.y_train))
        y_scale = max(float(np.std(data.y_train)), 1e-8)
        target = ((data.y_train - y_mean) / y_scale).astype(np.float32)
        target_tensor = torch.as_tensor(target, device=device)
        loss_fn: Any = nn.MSELoss()
    else:
        y_mean, y_scale = 0.0, 1.0
        target_tensor = torch.as_tensor(data.y_train.astype(np.int64), device=device)
        loss_fn = nn.CrossEntropyLoss()
    reference_train = torch.as_tensor(reference.X_train.astype(np.float32), device=device)
    transformed_train = torch.as_tensor(transformed.X_train.astype(np.float32), device=device)
    reference_optimizer = make_optimizer(reference_model, condition, float(model_config["learning_rate"]))
    transformed_optimizer = make_optimizer(transformed_model, condition, float(model_config["learning_rate"]))
    final_epoch = int(model_config["max_epochs"])
    orders = minibatch_orders(len(data.y_train), final_epoch, seed)
    order_hashes = [order_sha256(order) for order in orders]
    batch_size = int(model_config["batch_size"])
    metric_records: list[dict[str, Any]] = []
    prediction_parts: list[pd.DataFrame] = []

    def record(epoch: int | str) -> None:
        for split, row_ids, y, X_ref, X_alt in (
            ("validation", data.validation_indices, data.y_validation, reference.X_validation, transformed.X_validation),
            ("test", data.test_indices, data.y_test, reference.X_test, transformed.X_test),
        ):
            p_ref = predictions(reference_model, X_ref, data.problem_type, y_mean, y_scale, device)
            p_alt = predictions(transformed_model, X_alt, data.problem_type, y_mean, y_scale, device)
            metric_records.append({
                "dataset": data.key, "problem_type": data.problem_type, "model": "controlled_mlp",
                "model_seed": seed, "orbit_member": member, "condition": condition.name,
                "function_matched": condition.function_matched, "optimizer": condition.optimizer,
                "momentum": condition.momentum, "weight_decay": condition.weight_decay,
                "epoch": str(epoch), "split": split, "initial_max_logit_difference": initial_difference,
                **{f"reference_{key}": value for key, value in prediction_metrics(data.problem_type, y, p_ref).items()},
                **{f"transformed_{key}": value for key, value in prediction_metrics(data.problem_type, y, p_alt).items()},
                **disagreement_metrics(data.problem_type, y, p_ref, p_alt),
            })
            prediction_parts.extend([
                prediction_rows(data.key, seed, member, condition.name, epoch, split, row_ids, y, "reference", p_ref),
                prediction_rows(data.key, seed, member, condition.name, epoch, split, row_ids, y, "transformed", p_alt),
            ])

    started = time.time()
    record(0)
    for epoch, order in enumerate(orders, start=1):
        reference_model.train()
        transformed_model.train()
        for start in range(0, len(order), batch_size):
            indices = torch.as_tensor(order[start:start + batch_size], device=device)
            reference_optimizer.zero_grad(set_to_none=True)
            transformed_optimizer.zero_grad(set_to_none=True)
            reference_loss = loss_fn(reference_model(reference_train[indices]).squeeze(-1), target_tensor[indices])
            transformed_loss = loss_fn(transformed_model(transformed_train[indices]).squeeze(-1), target_tensor[indices])
            reference_loss.backward()
            transformed_loss.backward()
            reference_optimizer.step()
            transformed_optimizer.step()
        if epoch in CHECKPOINT_EPOCHS:
            record(epoch)
    record("final")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{condition.name}.", dir=destination.parent))
    try:
        pd.DataFrame(metric_records).to_csv(temporary / "metrics.csv", index=False)
        pd.concat(prediction_parts, ignore_index=True).to_csv(
            temporary / "predictions.csv.gz", index=False, compression="gzip"
        )
        metadata = {
            "status": "complete", "stage": "development_mechanism", "dataset_spec": spec,
            "model_seed": seed, "orbit_member": member, "condition": condition.__dict__,
            "config_sha256": config_hash, "dataset_panel_sha256": panel_hash,
            "function_match_orientation": "W_prime = W @ inv(A).T",
            "initial_max_logit_difference": initial_difference,
            "same_minibatch_order": True, "minibatch_order_sha256_by_epoch": order_hashes,
            "checkpoint_epochs": [0, 1, 2, 5, 10, 20, "final"],
            "wall_seconds": time.time() - started, "environment": environment_metadata(),
        }
        for filename in ("metrics.csv", "predictions.csv.gz"):
            metadata.setdefault("files", {})[filename] = {
                "sha256": sha256_file(temporary / filename), "bytes": (temporary / filename).stat().st_size,
            }
        (temporary / "metadata.json").write_text(json.dumps(jsonable(metadata), indent=2, sort_keys=True) + "\n")
        os.rename(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return "complete"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--member", action="append", type=int)
    parser.add_argument("--condition", action="append")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config_bytes, panel_bytes = CONFIG_PATH.read_bytes(), PANEL_PATH.read_bytes()
    config, panel = yaml.safe_load(config_bytes), json.loads(panel_bytes)
    selected = set(args.dataset)
    specs = [spec for spec in panel["datasets"] if spec["key"] in selected]
    if len(specs) != len(selected):
        raise SystemExit(f"unknown datasets: {selected - {spec['key'] for spec in specs}}")
    conditions = [item.name for item in optimizer_conditions(float(config["models"]["controlled_mlp"]["weight_decay"]))]
    seeds = args.seed or list(map(int, config["model_seeds"]))
    members = args.member or list(range(4))
    conditions = args.condition or conditions
    failures = []
    for spec in specs:
        for seed in seeds:
            for member in members:
                for condition in conditions:
                    print(f"=== {spec['key']} seed={seed} member={member} {condition} ===", flush=True)
                    try:
                        print(run_bundle(config, digest(config_bytes), digest(panel_bytes), spec, seed, member, condition, args.device), flush=True)
                    except Exception as error:
                        failures.append({"dataset": spec["key"], "seed": seed, "member": member,
                                         "condition": condition, "error": repr(error), "traceback": traceback.format_exc()})
                        print(json.dumps(failures[-1]), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
