#!/usr/bin/env python3
"""Train the controlled MLP with the frozen orbit-consistency penalty (D5)."""

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
sys.path.insert(0, str(ROOT / "scripts"))

from run_replication import prediction_frame  # noqa: E402
from src.basis_dependence import (  # noqa: E402
    build_primary_representations, build_rbf_feature_matrix, disagreement_metrics,
    environment_metadata, jsonable, load_dataset, orthogonal_matrix, prediction_metrics,
    sha256_file, stable_seed,
)
from src.mechanism import make_controlled_mlp, minibatch_orders, order_sha256  # noqa: E402


CONFIG_PATH = ROOT / "configs" / "development_protocol.yaml"
PANEL_PATH = ROOT / "configs" / "dataset_panel.json"


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def run_bundle(
    config: dict[str, Any], config_hash: str, panel_hash: str, spec: dict[str, Any],
    seed: int, device: str,
) -> str:
    if spec["panel"] != "development":
        raise RuntimeError("consistency runner refuses prospective datasets")
    destination = ROOT / "results" / "raw" / "development" / "consistency" / "controlled_mlp" / spec["key"] / f"seed_{seed}"
    if (destination / "metadata.json").exists():
        metadata = json.loads((destination / "metadata.json").read_text())
        if metadata["config_sha256"] != config_hash or metadata["dataset_panel_sha256"] != panel_hash:
            raise RuntimeError(f"frozen config drift at {destination}")
        return "cached"
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite incomplete bundle: {destination}")

    data = load_dataset(spec, config)
    blocks = build_rbf_feature_matrix(data, config)
    orbit = build_primary_representations(blocks, int(config["orbit_members"]))
    reference = orbit[0]
    evaluated = [reference, *[rep for rep in orbit if rep.variant == "orthogonal_all"]]
    model_config = config["models"]["controlled_mlp"]
    output_dimension = 1 if data.problem_type == "regression" else int(np.max(data.y_train)) + 1

    import torch
    from torch import nn

    model = make_controlled_mlp(reference.X_train.shape[1], output_dimension, model_config, seed, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    X_train = torch.as_tensor(reference.X_train.astype(np.float32), device=device)
    X_validation = torch.as_tensor(reference.X_validation.astype(np.float32), device=device)
    if data.problem_type == "regression":
        y_mean = float(np.mean(data.y_train))
        y_scale = max(float(np.std(data.y_train)), 1e-8)
        train_target = torch.as_tensor(((data.y_train - y_mean) / y_scale).astype(np.float32), device=device)
        validation_target = torch.as_tensor(
            ((data.y_validation - y_mean) / y_scale).astype(np.float32), device=device
        )
        task_loss: Any = nn.MSELoss()
    else:
        y_mean, y_scale = 0.0, 1.0
        train_target = torch.as_tensor(data.y_train.astype(np.int64), device=device)
        validation_target = torch.as_tensor(data.y_validation.astype(np.int64), device=device)
        task_loss = nn.CrossEntropyLoss()
    pool_size = int(config["orbit_members"])
    pools = {
        feature: [
            torch.as_tensor(
                orthogonal_matrix(len(indices), stable_seed(data.key, "D5", feature, member)),
                dtype=torch.float32, device=device,
            )
            for member in range(pool_size)
        ]
        for feature, indices in reference.feature_blocks.items()
    }
    lambda_consistency = float(config["remedies"]["orbit_consistency_lambda"])
    orders = minibatch_orders(len(data.y_train), int(model_config["max_epochs"]), seed)
    augmentation_rng = np.random.default_rng(stable_seed(data.key, seed, "D5-augmentation-order"))
    augmentation_members: list[list[int]] = []
    best_state = None
    best_validation = float("inf")
    best_epoch = -1
    stale = 0
    started = time.time()
    for epoch, order in enumerate(orders, start=1):
        model.train()
        epoch_members = []
        for start in range(0, len(order), int(model_config["batch_size"])):
            indices = torch.as_tensor(order[start:start + int(model_config["batch_size"])], device=device)
            raw = X_train[indices]
            augmented = raw.clone()
            selected = []
            for feature, block_indices in reference.feature_blocks.items():
                member = int(augmentation_rng.integers(pool_size))
                selected.append(member)
                augmented[:, block_indices] = raw[:, block_indices] @ pools[feature][member]
            epoch_members.extend(selected)
            optimizer.zero_grad(set_to_none=True)
            raw_logits = model(raw).squeeze(-1)
            augmented_logits = model(augmented).squeeze(-1)
            task = task_loss(raw_logits, train_target[indices])
            if data.problem_type == "regression":
                consistency = torch.mean((raw_logits - augmented_logits) ** 2)
            else:
                p = torch.softmax(raw_logits, dim=1)
                q = torch.softmax(augmented_logits, dim=1)
                midpoint = 0.5 * (p + q)
                consistency = 0.5 * (
                    torch.sum(p * (torch.log(p + 1e-8) - torch.log(midpoint + 1e-8)), dim=1).mean()
                    + torch.sum(q * (torch.log(q + 1e-8) - torch.log(midpoint + 1e-8)), dim=1).mean()
                )
            (task + lambda_consistency * consistency).backward()
            optimizer.step()
        augmentation_members.append(epoch_members)
        model.eval()
        with torch.no_grad():
            validation_loss = float(task_loss(model(X_validation).squeeze(-1), validation_target).item())
        if validation_loss < best_validation - 1e-7:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(model_config["patience"]):
                break
    if best_state is None:
        raise RuntimeError("consistency MLP produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()

    prediction_parts = []
    predictions = {}
    metrics = []
    for rep in evaluated:
        for split, row_ids, y, X in (
            ("validation", data.validation_indices, data.y_validation, rep.X_validation),
            ("test", data.test_indices, data.y_test, rep.X_test),
        ):
            with torch.no_grad():
                logits = model(torch.as_tensor(X.astype(np.float32), device=device)).squeeze(-1)
                prediction = (
                    logits.cpu().numpy() * y_scale + y_mean if data.problem_type == "regression"
                    else torch.softmax(logits, dim=1).cpu().numpy()
                )
            predictions[(rep.representation_id, split)] = prediction
            frame = prediction_frame(data, "controlled_mlp", seed, rep, split, row_ids, y, prediction)
            frame["repair"] = "orbit_consistency_lambda_1"
            prediction_parts.append(frame)
    for rep in evaluated:
        for split, y in (("validation", data.y_validation), ("test", data.y_test)):
            prediction = predictions[(rep.representation_id, split)]
            metrics.append({
                "dataset": data.key, "problem_type": data.problem_type, "model": "controlled_mlp",
                "model_seed": seed, "split": split, "repair": "orbit_consistency_lambda_1",
                "representation_id": rep.representation_id, "variant": rep.variant,
                "member": rep.member, "is_reference": rep.is_reference,
                **prediction_metrics(data.problem_type, y, prediction),
                **disagreement_metrics(
                    data.problem_type, y, predictions[(reference.representation_id, split)], prediction
                ),
            })

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        pd.DataFrame(metrics).to_csv(temporary / "metrics.csv", index=False)
        pd.concat(prediction_parts, ignore_index=True).to_csv(
            temporary / "predictions.csv.gz", index=False, compression="gzip"
        )
        metadata = {
            "status": "complete", "stage": "development_consistency", "dataset_spec": spec,
            "model": "controlled_mlp", "model_seed": seed, "device": device,
            "config_sha256": config_hash, "dataset_panel_sha256": panel_hash,
            "lambda": lambda_consistency, "pool_size_per_feature": pool_size,
            "fresh_transform_per_feature_per_batch": True,
            "augmentation_member_indices": augmentation_members,
            "minibatch_order_sha256_by_epoch": [order_sha256(order) for order in orders[:len(augmentation_members)]],
            "best_epoch": best_epoch, "best_validation_loss": best_validation,
            "epochs_run": len(augmentation_members), "wall_seconds": time.time() - started,
            "environment": environment_metadata(), "files": {},
        }
        for filename in ("metrics.csv", "predictions.csv.gz"):
            metadata["files"][filename] = {
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
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config_bytes, panel_bytes = CONFIG_PATH.read_bytes(), PANEL_PATH.read_bytes()
    config, panel = yaml.safe_load(config_bytes), json.loads(panel_bytes)
    specs = [spec for spec in panel["datasets"] if spec["panel"] == "development"
             and (not args.dataset or spec["key"] in set(args.dataset))]
    seeds = [int(seed) for seed in config["model_seeds"] if not args.seed or int(seed) in set(args.seed)]
    failures = []
    for spec in specs:
        for seed in seeds:
            print(f"=== {spec['key']} controlled_mlp seed={seed} ===", flush=True)
            try:
                print(run_bundle(config, digest(config_bytes), digest(panel_bytes), spec, seed, args.device), flush=True)
            except Exception as error:
                failures.append({"dataset": spec["key"], "seed": seed, "error": repr(error),
                                 "traceback": traceback.format_exc()})
                print(json.dumps(failures[-1]), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
