#!/usr/bin/env python3
"""Run frozen nested-context scaling on the confirmed real regression panel."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_openml_breadth_competence import load_numeric_task, tuning_metadata
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
    parser.add_argument("--config", type=Path, default=ROOT / "configs/real_context_scaling.yaml")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    tuning = tuning_metadata()["regression"]
    specs = config["datasets"][:1] if args.smoke else config["datasets"]
    repeats = 1 if args.smoke else int(config["repeats_per_dataset"])
    context_sizes = config["context_sizes"][:1] if args.smoke else [int(x) for x in config["context_sizes"]]
    max_context = max(context_sizes)
    started = time.perf_counter()
    records = []
    bundle = {
        "dataset": [], "repeat": [], "context_size": [], "feature_count": [],
        "context_index": [], "query_index": [], "cv_expert_loss": [],
        "query_expert_loss": [], "expert_prediction": [], "query_y": [],
    }
    audits, fits = {}, 0
    for dataset_index, spec in enumerate(specs):
        train_x, train_y, test_x, test_y, train_rows, test_rows, audit = load_numeric_task(
            spec, "regression", Path(config["openml_cache"]), int(config["max_numeric_features"])
        )
        name = str(spec["name"]); audits[name] = audit
        dataset_gains = []
        for repeat in range(repeats):
            seed = int(config["episode_seed"]) + dataset_index * 100_000 + repeat * 10
            rng = np.random.default_rng(seed + 1)
            context_order = rng.choice(len(train_y), size=max_context, replace=False)
            query_local = np.sort(np.random.default_rng(seed + 2).choice(
                len(test_y), size=int(config["query_size"]), replace=False
            ))
            query_x, query_y = test_x[query_local], test_y[query_local]
            for context_size in context_sizes:
                context_local = context_order[:context_size]
                context_x, context_y = train_x[context_local], train_y[context_local]
                predictions = fit_predict_experts(
                    context_x, context_y, query_x, "regression", seed + 700 + context_size
                )
                cv_loss = cross_validated_expert_losses(
                    context_x, context_y, "regression", seed + 500 + context_size,
                    int(config["cv_folds"]),
                )
                query_loss = np.asarray([
                    prediction_loss(query_y, prediction, "regression") for prediction in predictions
                ])
                soft = competence_weights(
                    cv_loss, float(tuning["temperature"]), float(tuning["uniform_shrinkage"])
                )
                weights = {
                    "uniform": np.full(len(EXPERTS), 1 / len(EXPERTS)),
                    "fixed": np.asarray(tuning["fixed_weights"], dtype=float),
                    "competence": soft,
                    "hard_cv": np.eye(len(EXPERTS))[int(np.argmin(cv_loss))],
                }
                losses = {
                    method: prediction_loss(
                        query_y, weighted_prediction(predictions, weight), "regression"
                    ) for method, weight in weights.items()
                }
                losses["best_individual_oracle"] = float(query_loss.min())
                dataset_gains.append(losses["fixed"] - losses["competence"])
                episode_index = len(bundle["dataset"])
                for method, value in losses.items():
                    records.append({
                        "episode_index": episode_index, "dataset": name, "repeat": repeat,
                        "context_size": context_size, "feature_count": train_x.shape[1],
                        "method": method, "loss": value,
                    })
                padded = np.full(max_context, -1, dtype=np.int64)
                padded[:context_size] = train_rows[context_local]
                bundle["dataset"].append(name); bundle["repeat"].append(repeat)
                bundle["context_size"].append(context_size)
                bundle["feature_count"].append(train_x.shape[1])
                bundle["context_index"].append(padded)
                bundle["query_index"].append(test_rows[query_local])
                bundle["cv_expert_loss"].append(cv_loss)
                bundle["query_expert_loss"].append(query_loss)
                bundle["expert_prediction"].append(predictions.astype(np.float32))
                bundle["query_y"].append(query_y.astype(np.float32))
                fits += len(EXPERTS) * (int(config["cv_folds"]) + 1)
        print(
            f"dataset={name} repeats={repeats} contexts={context_sizes} "
            f"mean_gain={np.mean(dataset_gains):.6f}", flush=True
        )
    if args.smoke:
        print("smoke_ok"); return
    arrays = {key: np.asarray(value) for key, value in bundle.items()}
    config_hash = sha256_file(config_path)
    run_key = f"real_context_scaling_{config_hash[:10]}"
    raw_path = ROOT / "results/raw" / f"{run_key}.npz"
    cells_path = ROOT / "results/processed" / f"{run_key}_cells.csv"
    metadata_path = ROOT / "results/raw" / f"{run_key}.metadata.json"
    for output in (raw_path, cells_path, metadata_path):
        if output.exists(): raise FileExistsError(f"refusing to overwrite {output}")
    np.savez_compressed(raw_path, **arrays); pd.DataFrame(records).to_csv(cells_path, index=False)
    metadata = {
        "run_key": run_key, "experiment": config["experiment"], "protocol": config["protocol"],
        "config": str(config_path.relative_to(ROOT)), "config_sha256": config_hash,
        "git_commit": git_commit(ROOT), "package_versions": package_versions(),
        "episodes": len(arrays["dataset"]), "expert_fits": fits,
        "wall_clock_seconds": time.perf_counter() - started, "dataset_audit": audits,
        "synthetic_tuning": tuning, "raw_bundle": str(raw_path.relative_to(ROOT)),
        "processed_summary": str(cells_path.relative_to(ROOT)),
    }
    write_json_exclusive(metadata_path, metadata)
    append_manifest(ROOT / "results/MANIFEST.jsonl", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
