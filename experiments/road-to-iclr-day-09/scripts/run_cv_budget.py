#!/usr/bin/env python3
"""Run frozen 2/3/5-fold competence compute-budget comparison."""

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
    if str(path) not in sys.path: sys.path.insert(0, str(path))

from run_openml_breadth_competence import load_numeric_task, tuning_metadata
from src.methods import (
    EXPERTS, competence_weights, cross_validated_expert_losses,
    fit_predict_experts, prediction_loss, weighted_prediction,
)
from src.runio import append_manifest, git_commit, package_versions, sha256_file, write_json_exclusive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/cv_budget.yaml")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args(); config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text()); tuning = tuning_metadata()["regression"]
    specs = config["datasets"][:1] if args.smoke else config["datasets"]
    repeats = 1 if args.smoke else int(config["repeats_per_dataset"])
    folds = [int(x) for x in config["cv_folds"]]
    records, audits, fits = [], {}, 0; started = time.perf_counter()
    bundle = {
        "dataset": [], "repeat": [], "feature_count": [], "context_index": [],
        "query_index": [], "query_y": [], "expert_prediction": [],
        **{f"cv_loss_{fold}": [] for fold in folds},
    }
    for dataset_index, spec in enumerate(specs):
        train_x, train_y, test_x, test_y, train_rows, test_rows, audit = load_numeric_task(
            spec, "regression", Path(config["openml_cache"]), int(config["max_numeric_features"])
        )
        name = str(spec["name"]); audits[name] = audit; log_losses = []
        for repeat in range(repeats):
            seed = int(config["episode_seed"]) + dataset_index * 100_000 + repeat * 10
            rng = np.random.default_rng(seed)
            context_local = np.sort(rng.choice(
                len(train_y), size=int(config["context_size"]), replace=False
            ))
            query_local = np.sort(np.random.default_rng(seed + 1).choice(
                len(test_y), size=int(config["query_size"]), replace=False
            ))
            context_x, context_y = train_x[context_local], train_y[context_local]
            query_x, query_y = test_x[query_local], test_y[query_local]
            predictions = fit_predict_experts(context_x, context_y, query_x, "regression", seed + 700)
            fixed = prediction_loss(
                query_y, weighted_prediction(predictions, np.asarray(tuning["fixed_weights"])),
                "regression",
            )
            losses = {"fixed": fixed}
            cv_losses = {}
            for fold in folds:
                cv = cross_validated_expert_losses(
                    context_x, context_y, "regression", seed + 500 + fold * 100, fold
                )
                weight = competence_weights(
                    cv, float(tuning["temperature"]), float(tuning["uniform_shrinkage"])
                )
                losses[f"competence_{fold}fold"] = prediction_loss(
                    query_y, weighted_prediction(predictions, weight), "regression"
                )
                cv_losses[fold] = cv; fits += len(EXPERTS) * fold
            fits += len(EXPERTS); log_losses.append(losses["competence_2fold"])
            episode_index = len(bundle["dataset"])
            for method, value in losses.items():
                records.append({
                    "episode_index": episode_index, "dataset": name, "repeat": repeat,
                    "feature_count": train_x.shape[1], "method": method, "loss": value,
                })
            bundle["dataset"].append(name); bundle["repeat"].append(repeat)
            bundle["feature_count"].append(train_x.shape[1])
            bundle["context_index"].append(train_rows[context_local])
            bundle["query_index"].append(test_rows[query_local]); bundle["query_y"].append(query_y)
            bundle["expert_prediction"].append(predictions.astype(np.float32))
            for fold in folds: bundle[f"cv_loss_{fold}"].append(cv_losses[fold])
        print(f"dataset={name} repeats={repeats} mean_2fold_loss={np.mean(log_losses):.6f}", flush=True)
    if args.smoke: print("smoke_ok"); return
    arrays = {key: np.asarray(value) for key, value in bundle.items()}
    config_hash = sha256_file(config_path); run_key = f"real_cv_budget_{config_hash[:10]}"
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
        "episodes": len(arrays["dataset"]), "expert_fits_all_fold_arms": fits,
        "standalone_fits_per_episode": {str(fold): len(EXPERTS) * (fold + 1) for fold in folds},
        "wall_clock_seconds": time.perf_counter() - started, "dataset_audit": audits,
        "synthetic_tuning": tuning, "raw_bundle": str(raw_path.relative_to(ROOT)),
        "processed_summary": str(cells_path.relative_to(ROOT)),
    }
    write_json_exclusive(metadata_path, metadata); append_manifest(ROOT / "results/MANIFEST.jsonl", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__": main()
