"""Frozen semantic training-row-order control for the Day-5 completion panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from completion_neural_panel import (
    CONFIG,
    HERE,
    PARTS,
    digest,
    fit,
    initialize,
    predict,
    prepare,
    render,
    views,
)


DATASETS = ("australian_credit_approval", "bank_marketing_subscription",
            "fremtpl_claim_count", "kdd17_stock_return")
MODELS = ("mlp", "resnet", "ft_transformer", "tabm")
ROW_ORDERS = 4


def row_permutations(rows: int, seed: int) -> list[np.ndarray]:
    """Return a canonical order and frozen, distinct semantic row permutations."""
    rng = np.random.default_rng(seed)
    orders = [np.arange(rows)]
    while len(orders) < ROW_ORDERS:
        candidate = rng.permutation(rows)
        if not any(np.array_equal(candidate, old) for old in orders):
            orders.append(candidate)
    return orders


def run(dataset: str, model_name: str, device_name: str, config: dict, output: Path) -> None:
    split_seed = int(config["split_seeds"][0])
    data = prepare(dataset, split_seed, config)
    design = views(data, config)
    feature, category, class_map = design["feature"][0], design["category"][0], design["class"][0]
    rendered = {part: render(data, part, feature, category)[0] for part in PARTS}
    orders = row_permutations(len(data.y["train"]), int(config["view_seed"]) + 9000 + sum(dataset.encode()))
    output_dim = 2 if data.task == "classification" else 1
    predictions = np.empty((ROW_ORDERS, len(data.y["test"]), output_dim), dtype=np.float32)
    device = torch.device(device_name)
    telemetry = []
    init_seed = int(config["init_seeds"][0])
    minibatch_seed = int(config["order_seeds"][0])
    transformed_y = class_map[data.y["train"]] if data.task == "classification" else data.y["train"]
    for index, order in enumerate(orders):
        model = initialize(model_name, rendered["train"].shape[1], output_dim, init_seed, config, device)
        elapsed, peak = fit(
            model,
            np.ascontiguousarray(rendered["train"][order]),
            np.ascontiguousarray(transformed_y[order]),
            data.task,
            model_name,
            minibatch_seed,
            config,
            device,
        )
        predictions[index] = predict(model, rendered["test"], data.task, model_name, class_map, device)
        telemetry.append({"row_order": index, "wall_seconds": elapsed, "peak_device_bytes": peak})
        print(f"row-order {dataset} {model_name} {index + 1}/{ROW_ORDERS}", flush=True)
    stem = f"{dataset}__{model_name}__row_order"
    np.savez_compressed(
        output / f"{stem}.npz",
        test_predictions=predictions,
        test_y=data.y["test"],
        row_orders=np.asarray(orders, dtype=np.int32),
    )
    movement = np.mean(np.sum((predictions - predictions[0:1]) ** 2, axis=-1), axis=1)
    manifest = {
        "status": "complete",
        "dataset": dataset,
        "task": data.task,
        "model": model_name,
        "row_orders": ROW_ORDERS,
        "represented_fits": ROW_ORDERS,
        "initialization_seed_held_fixed": init_seed,
        "minibatch_order_seed_held_fixed": minibatch_seed,
        "mean_squared_prediction_movement_by_order": movement.tolist(),
        "protocol_sha256": config["protocol_sha256"],
        "device": device_name,
        "wall_seconds": float(sum(item["wall_seconds"] for item in telemetry)),
        "maximum_peak_device_bytes": max(item["peak_device_bytes"] for item in telemetry),
        "telemetry": telemetry,
    }
    (output / f"{stem}.json").write_text(json.dumps(manifest, indent=2) + "\n")


def analyze(output: Path) -> None:
    rows = []
    for dataset in DATASETS:
        for model in MODELS:
            stem = f"{dataset}__{model}__row_order"
            archive = np.load(output / f"{stem}.npz")
            predictions = archive["test_predictions"].astype(np.float64)
            center = predictions.mean(axis=0, keepdims=True)
            rows.append({
                "dataset": dataset,
                "model": model,
                "row_order_risk": float(np.mean(np.sum((predictions - center) ** 2, axis=-1))),
                "canonical_to_permuted_movement": float(
                    np.mean(np.sum((predictions[1:] - predictions[0]) ** 2, axis=-1))
                ),
            })
    import pandas as pd

    frame = pd.DataFrame(rows)
    frame.to_csv(HERE / "results" / "completion_row_order_cells.csv", index=False)
    summary = {
        "status": "complete",
        "cells": len(frame),
        "mean_row_order_risk": float(frame.row_order_risk.mean()),
        "median_row_order_risk": float(frame.row_order_risk.median()),
        "by_model": frame.groupby("model").row_order_risk.mean().to_dict(),
    }
    (HERE / "results" / "completion_row_order_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "completion_row_order")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.analyze:
        analyze(args.output_dir)
        return
    if args.dataset is None or args.model is None:
        raise ValueError("--dataset and --model are required for fitting")
    config = json.loads(args.config.read_text())
    if digest(HERE / config["protocol"]) != config["protocol_sha256"]:
        raise AssertionError("completion protocol hash mismatch")
    torch.set_num_threads(1)
    run(args.dataset, args.model, args.device, config, args.output_dir)


if __name__ == "__main__":
    main()
