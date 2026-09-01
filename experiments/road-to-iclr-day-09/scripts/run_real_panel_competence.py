#!/usr/bin/env python3
"""Run frozen numeric real-panel competence transfer without real-data tuning."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DAY8 = ROOT.parent / "road-to-iclr-day-08"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(DAY8) not in sys.path:
    sys.path.insert(1, str(DAY8))

from day8_core import load_real_dataset
from src.methods import (
    EXPERTS,
    competence_weights,
    cross_validated_expert_losses,
    fit_predict_experts,
    prediction_loss,
    weighted_prediction,
)
from src.runio import append_manifest, git_commit, package_versions, sha256_file, write_json_exclusive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/real_panel_competence.yaml")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sample_indices(y: np.ndarray, size: int, seed: int, classification: bool) -> np.ndarray:
    rows = np.arange(len(y))
    if size > len(rows):
        raise ValueError("sample size exceeds split")
    chosen, _ = train_test_split(
        rows,
        train_size=size,
        random_state=seed,
        stratify=y if classification else None,
    )
    return np.sort(np.asarray(chosen, dtype=np.int64))


def tuning_metadata() -> dict:
    paths = sorted(ROOT.glob("results/raw/fallback_loss_router_*_development.metadata.json"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one loss-router development metadata file, found {paths}")
    return json.loads(paths[0].read_text())["tuning"]


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    tuning = tuning_metadata()
    datasets = [
        (task_type, name)
        for task_type, names in config["datasets"].items()
        for name in names
    ]
    if args.smoke:
        datasets = datasets[:1]
    repeats = 1 if args.smoke else int(config["repeats_per_dataset"])
    context_size = int(config["context_size"])
    query_size = int(config["query_size"])
    base_seed = int(config["episode_seed"])
    started = time.perf_counter()
    records = []
    bundle = {
        "dataset": [], "task_type": [], "repeat": [], "feature_count": [],
        "context_index": [], "query_index": [], "cv_expert_loss": [],
        "query_expert_loss": [], "expert_prediction": [], "query_y": [],
    }
    data_hashes = {}
    fit_count = 0
    for dataset_index, (task_type, name) in enumerate(datasets):
        data = load_real_dataset(name, split_seed=int(config["source_split_seed"]))
        expected = "classification" if task_type == "classification" else "regression"
        if data.task != expected or (task_type == "classification" and data.n_classes != 2):
            raise AssertionError(f"task mismatch for {name}: {data.task}/{data.n_classes}")
        train_x = np.asarray(data.x_num["train"], dtype=float)
        test_x = np.asarray(data.x_num["test"], dtype=float)
        train_y = np.asarray(data.y["train"])
        test_y = np.asarray(data.y["test"])
        if train_x.shape[1] == 0 or not np.isfinite(train_x).all() or not np.isfinite(test_x).all():
            raise AssertionError(f"invalid numeric data for {name}")
        source = DAY8.parent / "road-to-iclr-day-01" / "data" / name
        data_hashes[name] = {
            "info": sha256_file(source / "info.json"),
            "x_num": sha256_file(source / "x_num.npy"),
            "y": sha256_file(source / "y.npy"),
        }
        dataset_losses = []
        for repeat in range(repeats):
            seed = base_seed + dataset_index * 100_000 + repeat * 10
            classification = task_type == "classification"
            context_index = sample_indices(train_y, context_size, seed + 1, classification)
            query_index = sample_indices(test_y, query_size, seed + 2, classification)
            context_x, context_y = train_x[context_index], train_y[context_index]
            query_x, query_y = test_x[query_index], test_y[query_index]
            predictions = fit_predict_experts(context_x, context_y, query_x, task_type, seed + 700)
            cv_loss = cross_validated_expert_losses(
                context_x, context_y, task_type, seed + 500, int(config["cv_folds"])
            )
            query_loss = np.asarray([
                prediction_loss(query_y, prediction, task_type) for prediction in predictions
            ])
            soft = competence_weights(
                cv_loss,
                float(tuning[task_type]["temperature"]),
                float(tuning[task_type]["uniform_shrinkage"]),
            )
            weights = {
                "uniform": np.full(len(EXPERTS), 1 / len(EXPERTS)),
                "fixed": np.asarray(tuning[task_type]["fixed_weights"], dtype=float),
                "competence": soft,
                "hard_cv": np.eye(len(EXPERTS))[int(np.argmin(cv_loss))],
            }
            losses = {
                method: prediction_loss(query_y, weighted_prediction(predictions, weight), task_type)
                for method, weight in weights.items()
            }
            losses["best_individual_oracle"] = float(query_loss.min())
            dataset_losses.append(losses["competence"])
            episode_index = len(bundle["dataset"])
            for method, loss in losses.items():
                records.append({
                    "episode_index": episode_index,
                    "dataset": name,
                    "task_type": task_type,
                    "repeat": repeat,
                    "feature_count": train_x.shape[1],
                    "method": method,
                    "loss": loss,
                })
            bundle["dataset"].append(name)
            bundle["task_type"].append(task_type)
            bundle["repeat"].append(repeat)
            bundle["feature_count"].append(train_x.shape[1])
            bundle["context_index"].append(context_index)
            bundle["query_index"].append(query_index)
            bundle["cv_expert_loss"].append(cv_loss)
            bundle["query_expert_loss"].append(query_loss)
            bundle["expert_prediction"].append(predictions.astype(np.float32))
            bundle["query_y"].append(query_y.astype(np.float32))
            fit_count += len(EXPERTS) * (int(config["cv_folds"]) + 1)
        print(
            f"dataset={name} task={task_type} d={train_x.shape[1]} repeats={repeats} "
            f"mean_competence_loss={np.mean(dataset_losses):.6f}", flush=True
        )
    if args.smoke:
        print("smoke_ok")
        return

    arrays = {key: np.asarray(value) for key, value in bundle.items()}
    config_hash = sha256_file(config_path)
    run_key = f"real_panel_competence_{config_hash[:10]}"
    raw_path = ROOT / "results/raw" / f"{run_key}.npz"
    cells_path = ROOT / "results/processed" / f"{run_key}_cells.csv"
    metadata_path = ROOT / "results/raw" / f"{run_key}.metadata.json"
    for output in (raw_path, cells_path, metadata_path):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
    np.savez_compressed(raw_path, **arrays)
    pd.DataFrame(records).to_csv(cells_path, index=False)
    metadata = {
        "run_key": run_key,
        "experiment": config["experiment"],
        "protocol": "REAL_PANEL_COMPETENCE_PROTOCOL.md",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": config_hash,
        "git_commit": git_commit(ROOT),
        "package_versions": package_versions(),
        "episodes": len(arrays["dataset"]),
        "expert_fits": fit_count,
        "wall_clock_seconds": time.perf_counter() - started,
        "synthetic_tuning": tuning,
        "data_hashes": data_hashes,
        "raw_bundle": str(raw_path.relative_to(ROOT)),
        "processed_summary": str(cells_path.relative_to(ROOT)),
    }
    write_json_exclusive(metadata_path, metadata)
    append_manifest(ROOT / "results/MANIFEST.jsonl", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
