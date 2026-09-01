#!/usr/bin/env python3
"""Matched-function optimizer equivariance audit on two development datasets."""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tournament.common import (  # noqa: E402
    development_specs,
    disagreement,
    environment_metadata,
    load_blocks,
    load_protocol,
    orthogonal_all_orbit,
    task_error,
    write_json,
)
from tournament.models import (  # noqa: E402
    _forward,
    _loss,
    _prediction_from_logits,
    build_model,
)
from tournament.optimizers import make_optimizers, step, zero_grad  # noqa: E402


METHODS = {
    "AdamW": "adamw",
    "BlockScalarAdam": "block_scalar_adam",
    "BlockAdam": "block_adam",
    "MatrixAdam": "matrix_adam",
    "SGD": "sgd_control",
}


@torch.no_grad()
def match_first_layer(first_weight: torch.nn.Parameter, transforms: dict[str, np.ndarray], blocks: dict[str, list[int]]) -> None:
    for feature, matrix in transforms.items():
        if feature not in blocks:
            continue
        indices = torch.as_tensor(blocks[feature], dtype=torch.long, device=first_weight.device)
        q = torch.as_tensor(matrix, dtype=first_weight.dtype, device=first_weight.device)
        matched = first_weight[:, indices] @ q
        first_weight.index_copy_(1, indices, matched)


def run_pair(spec: dict[str, Any], seed: int, device: str) -> list[dict[str, Any]]:
    protocol = load_protocol()
    blocks = load_blocks(spec, protocol)
    orbit = orthogonal_all_orbit(blocks, protocol)
    reference, rotated = orbit[0], orbit[1]
    problem_type = blocks.dataset.problem_type
    output_dim = 1 if problem_type == "regression" else int(np.max(blocks.dataset.y_train)) + 1
    model_config = protocol["models"]["controlled_mlp"]
    train_reference = torch.as_tensor(reference.X_train.astype(np.float32), device=device)
    train_rotated = torch.as_tensor(rotated.X_train.astype(np.float32), device=device)
    test_reference = torch.as_tensor(reference.X_test.astype(np.float32), device=device)
    test_rotated = torch.as_tensor(rotated.X_test.astype(np.float32), device=device)
    if problem_type == "regression":
        y_mean = float(np.mean(blocks.dataset.y_train))
        y_scale = max(float(np.std(blocks.dataset.y_train)), 1e-8)
        train_target = torch.as_tensor(
            ((blocks.dataset.y_train - y_mean) / y_scale).astype(np.float32), device=device
        )
    else:
        y_mean, y_scale = 0.0, 1.0
        train_target = torch.as_tensor(blocks.dataset.y_train.astype(np.int64), device=device)
    records: list[dict[str, Any]] = []
    checkpoints = set(int(value) for value in protocol["matched_audit_epochs"])
    for label, optimizer_method in METHODS.items():
        model_a, first_a, _ = build_model(
            "controlled_mlp", reference.X_train.shape[1], output_dim, model_config, seed, device
        )
        model_b, first_b, _ = build_model(
            "controlled_mlp", rotated.X_train.shape[1], output_dim, model_config, seed, device
        )
        model_b.load_state_dict(copy.deepcopy(model_a.state_dict()))
        match_first_layer(first_b, rotated.transforms, rotated.feature_blocks)
        learning_rate = (
            float(protocol["optimizer_methods"]["sgd_control"]["learning_rate"])
            if optimizer_method == "sgd_control"
            else float(model_config["learning_rate"])
        )
        optimizer_a = make_optimizers(
            model_a,
            first_a,
            reference.feature_blocks,
            method=optimizer_method,
            lr=learning_rate,
            weight_decay=float(model_config["weight_decay"]),
        )
        optimizer_b = make_optimizers(
            model_b,
            first_b,
            rotated.feature_blocks,
            method=optimizer_method,
            lr=learning_rate,
            weight_decay=float(model_config["weight_decay"]),
        )
        rng = np.random.default_rng(seed)
        started = time.perf_counter()

        def record(epoch: int) -> None:
            model_a.eval()
            model_b.eval()
            with torch.no_grad():
                first_prediction = _prediction_from_logits(
                    "controlled_mlp",
                    problem_type,
                    _forward("controlled_mlp", model_a, test_reference),
                    y_mean,
                    y_scale,
                ).cpu().numpy()
                second_prediction = _prediction_from_logits(
                    "controlled_mlp",
                    problem_type,
                    _forward("controlled_mlp", model_b, test_rotated),
                    y_mean,
                    y_scale,
                ).cpu().numpy()
            records.append(
                {
                    "dataset": blocks.dataset.key,
                    "problem_type": problem_type,
                    "model": "controlled_mlp",
                    "seed": seed,
                    "method": label,
                    "optimizer": optimizer_method,
                    "epoch": epoch,
                    "disagreement": disagreement(
                        problem_type, blocks.dataset.y_test, first_prediction, second_prediction
                    ),
                    "max_prediction_absolute_difference": float(
                        np.max(np.abs(first_prediction - second_prediction))
                    ),
                    "reference_task_error": task_error(
                        problem_type, blocks.dataset.y_test, first_prediction
                    ),
                    "rotated_task_error": task_error(
                        problem_type, blocks.dataset.y_test, second_prediction
                    ),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )

        record(0)
        for epoch in range(1, max(checkpoints) + 1):
            order = rng.permutation(len(blocks.dataset.y_train))
            model_a.train()
            model_b.train()
            batch_size = int(model_config["batch_size"])
            for start in range(0, len(order), batch_size):
                indices = torch.as_tensor(order[start : start + batch_size], device=device)
                zero_grad(optimizer_a)
                zero_grad(optimizer_b)
                loss_a = _loss(
                    "controlled_mlp",
                    problem_type,
                    _forward("controlled_mlp", model_a, train_reference[indices]),
                    train_target[indices],
                )
                loss_b = _loss(
                    "controlled_mlp",
                    problem_type,
                    _forward("controlled_mlp", model_b, train_rotated[indices]),
                    train_target[indices],
                )
                loss_a.backward()
                loss_b.backward()
                step(optimizer_a)
                step(optimizer_b)
            if epoch in checkpoints:
                record(epoch)
        print(f"[audit] {blocks.dataset.key} seed={seed} {label}", flush=True)
        del model_a, model_b, optimizer_a, optimizer_b
        torch.cuda.empty_cache() if str(device).startswith("cuda") else None
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=["all", "california_housing", "phoneme"], default="all"
    )
    parser.add_argument("--seed", default="all")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    protocol = load_protocol()
    seeds = protocol["model_seeds"] if args.seed == "all" else [int(args.seed)]
    if not set(seeds).issubset(set(protocol["model_seeds"])):
        raise RuntimeError("seed not in frozen protocol")
    wanted = {"california_housing", "phoneme"} if args.dataset == "all" else {args.dataset}
    specs = [spec for spec in development_specs(protocol) if spec["key"] in wanted]
    destination = ROOT / "results" / "processed" / "mechanism"
    destination.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        for seed in seeds:
            output = destination / f"{spec['key']}__seed_{seed}.csv"
            if output.exists():
                print(f"[cached] {output}")
                continue
            records = run_pair(spec, int(seed), args.device)
            pd.DataFrame(records).to_csv(output, index=False)
            write_json(
                output.with_suffix(".json"),
                {"dataset": spec["key"], "seed": seed, "environment": environment_metadata()},
            )


if __name__ == "__main__":
    main()
