"""Nested finite-schema-menu approximation experiment for modern neural models."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from completion_neural_panel import (
    CONFIG, HERE, PARTS, digest, fit, initialize, predict, prepare, render,
)


DATASETS = ("australian_credit_approval", "fremtpl_claim_count")
MODELS = ("mlp", "resnet", "ft_transformer", "tabm")
STATES = 256


def schema_states(data, seed: int) -> list[tuple[np.ndarray, list[np.ndarray], np.ndarray]]:
    rng = np.random.default_rng(seed)
    fields = data.x_num["train"].shape[1] + data.x_cat["train"].shape[1]
    output = []
    signatures = set()
    while len(output) < STATES:
        feature = rng.permutation(fields)
        category = [rng.permutation(size) for size in data.cardinalities]
        classes = rng.permutation(2) if data.task == "classification" else np.asarray([0])
        signature = (tuple(feature), tuple(tuple(value) for value in category), tuple(classes))
        if signature not in signatures:
            signatures.add(signature)
            output.append((feature, category, classes))
    return output


def run(dataset: str, model_name: str, device_name: str, config: dict, output: Path) -> None:
    data = prepare(dataset, int(config["split_seeds"][0]), config)
    states = schema_states(data, int(config["view_seed"]) + 5000 + sum(dataset.encode()))
    canonical, _ = render(data, "train", states[0][0], states[0][1])
    output_dim = 2 if data.task == "classification" else 1
    validation = np.empty((STATES, len(data.y["validation"]), output_dim), dtype=np.float32)
    test = np.empty((STATES, len(data.y["test"]), output_dim), dtype=np.float32)
    device = torch.device(device_name)
    telemetry = []
    for index, (feature, category, class_map) in enumerate(states):
        rendered = {part: render(data, part, feature, category)[0] for part in PARTS}
        model = initialize(model_name, canonical.shape[1], output_dim, int(config["init_seeds"][0]), config, device)
        target = class_map[data.y["train"]] if data.task == "classification" else data.y["train"]
        elapsed, peak = fit(
            model, rendered["train"], target, data.task, model_name,
            int(config["order_seeds"][0]), config, device,
        )
        validation[index] = predict(model, rendered["validation"], data.task, model_name, class_map, device)
        test[index] = predict(model, rendered["test"], data.task, model_name, class_map, device)
        telemetry.append({"state": index, "wall_seconds": elapsed, "peak_device_bytes": peak})
        print(f"menu {dataset} {model_name} {index + 1}/{STATES}", flush=True)
    stem = f"{dataset}__{model_name}"
    np.savez_compressed(
        output / f"{stem}.npz", validation_predictions=validation, test_predictions=test,
        validation_y=data.y["validation"], test_y=data.y["test"],
    )
    manifest = {
        "status": "complete", "dataset": dataset, "task": data.task, "model": model_name,
        "schema_states": STATES, "represented_fits": STATES,
        "device": device_name,
        "wall_seconds": float(sum(row["wall_seconds"] for row in telemetry)),
        "maximum_peak_device_bytes": max(row["peak_device_bytes"] for row in telemetry),
        "protocol_sha256": config["protocol_sha256"], "telemetry": telemetry,
    }
    (output / f"{stem}.json").write_text(json.dumps(manifest, indent=2) + "\n")


def analyze(output: Path) -> None:
    rows = []
    rng = np.random.default_rng(2026082857)
    for dataset in DATASETS:
        for model in MODELS:
            archive = np.load(output / f"{dataset}__{model}.npz")
            predictions = archive["test_predictions"].astype(np.float64)
            reference = predictions.mean(0)
            for menu in (4, 8, 16, 32, 64):
                target = predictions[:menu].mean(0)
                movement = float(np.mean(np.sum((target - reference) ** 2, axis=-1)))
                budget = min(16, menu)
                errors = []
                for _ in range(2048):
                    chosen = rng.choice(menu, budget, replace=False)
                    estimate = predictions[chosen].mean(0)
                    errors.append(np.mean(np.sum((estimate - target) ** 2, axis=-1)))
                rows.append({
                    "dataset": dataset, "model": model, "menu_size": menu,
                    "reference_states": STATES, "within_menu_budget": budget,
                    "menu_target_movement": movement,
                    "within_menu_sampling_error": float(np.mean(errors)),
                })
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "completion_menu_size_cells.csv", index=False)
    summary = {
        "status": "complete", "datasets": len(DATASETS), "models": len(MODELS),
        "reference_states": STATES,
        "mean_target_movement_by_menu": frame.groupby("menu_size").menu_target_movement.mean().to_dict(),
        "mean_sampling_error_by_menu": frame.groupby("menu_size").within_menu_sampling_error.mean().to_dict(),
    }
    (RESULTS / "completion_menu_size_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


RESULTS = HERE / "results"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "completion_menu_size")
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
