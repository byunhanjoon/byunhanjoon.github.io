#!/usr/bin/env python3
"""Run the frozen unseen-identity OpenML competence breadth panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import openml
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    parser.add_argument("--config", type=Path, default=ROOT / "configs/openml_breadth_competence.yaml")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def tuning_metadata() -> dict:
    paths = sorted(ROOT.glob("results/raw/fallback_loss_router_*_development.metadata.json"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one development metadata file, found {paths}")
    return json.loads(paths[0].read_text())["tuning"]


def sample_indices(y: np.ndarray, size: int, seed: int, classification: bool) -> np.ndarray:
    rows = np.arange(len(y))
    chosen, _ = train_test_split(
        rows, train_size=size, random_state=seed,
        stratify=y if classification else None,
    )
    return np.sort(np.asarray(chosen, dtype=np.int64))


def split_hash(train: np.ndarray, test: np.ndarray) -> str:
    digest = hashlib.sha256()
    for values in (train, test):
        array = np.ascontiguousarray(values, dtype=np.int64)
        digest.update(array.view(np.uint8))
    return digest.hexdigest()


def load_numeric_task(spec: dict, task_type: str, cache: Path, max_features: int) -> tuple:
    openml.config.set_root_cache_directory(str(cache.resolve()))
    task = openml.tasks.get_task(int(spec["task_id"]), download_data=True)
    dataset = task.get_dataset()
    X, target, categorical, names = dataset.get_data(
        target=task.target_name, dataset_format="dataframe"
    )
    train_rows, test_rows = task.get_train_test_split_indices(repeat=0, fold=0)
    train_rows = np.asarray(train_rows, dtype=np.int64)
    test_rows = np.asarray(test_rows, dtype=np.int64)
    numeric_names = [
        str(name) for name, is_categorical in zip(names, categorical) if not is_categorical
    ]
    numeric = X[numeric_names].apply(pd.to_numeric, errors="coerce")
    valid_names = [name for name in numeric_names if numeric.iloc[train_rows][name].notna().any()]
    selected = valid_names[:max_features]
    if not selected:
        raise RuntimeError(f"no usable numeric features for {spec['name']}")
    train_frame = numeric.iloc[train_rows][selected].to_numpy(dtype=float)
    test_frame = numeric.iloc[test_rows][selected].to_numpy(dtype=float)
    medians = np.nanmedian(train_frame, axis=0)
    train_frame = np.where(np.isfinite(train_frame), train_frame, medians)
    test_frame = np.where(np.isfinite(test_frame), test_frame, medians)
    scaler = StandardScaler().fit(train_frame)
    train_x = scaler.transform(train_frame)
    test_x = scaler.transform(test_frame)
    raw_y = np.asarray(target)
    if task_type == "classification":
        encoder = LabelEncoder().fit(raw_y[train_rows])
        if len(encoder.classes_) != 2:
            raise AssertionError(f"nonbinary task {spec['name']}: {encoder.classes_}")
        train_y = encoder.transform(raw_y[train_rows]).astype(np.int64)
        test_y = encoder.transform(raw_y[test_rows]).astype(np.int64)
    else:
        numeric_y = pd.to_numeric(pd.Series(raw_y), errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(numeric_y[train_rows]).all() or not np.isfinite(numeric_y[test_rows]).all():
            raise AssertionError(f"nonfinite target for {spec['name']}")
        mean = float(numeric_y[train_rows].mean())
        scale = float(numeric_y[train_rows].std()) or 1.0
        train_y = (numeric_y[train_rows] - mean) / scale
        test_y = (numeric_y[test_rows] - mean) / scale
    audit = {
        "task_id": int(spec["task_id"]),
        "dataset_id": int(dataset.dataset_id),
        "dataset_md5": dataset.md5_checksum,
        "outer_train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "numeric_features_available": len(valid_names),
        "selected_features": selected,
        "split_sha256": split_hash(train_rows, test_rows),
    }
    return train_x, train_y, test_x, test_y, train_rows, test_rows, audit


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    tuning = tuning_metadata()
    datasets = [
        (task_type, spec)
        for task_type, specs in config["datasets"].items()
        for spec in specs
    ]
    if args.smoke:
        datasets = datasets[:1]
    repeats = 1 if args.smoke else int(config["repeats_per_dataset"])
    started = time.perf_counter()
    records = []
    bundle = {
        "dataset": [], "task_type": [], "repeat": [], "feature_count": [],
        "context_index": [], "query_index": [], "cv_expert_loss": [],
        "query_expert_loss": [], "expert_prediction": [], "query_y": [],
    }
    dataset_audit = {}
    fits = 0
    for dataset_index, (task_type, spec) in enumerate(datasets):
        train_x, train_y, test_x, test_y, train_rows, test_rows, source_audit = load_numeric_task(
            spec, task_type, Path(config["openml_cache"]), int(config["max_numeric_features"])
        )
        name = str(spec["name"])
        dataset_audit[name] = source_audit
        losses_for_log = []
        for repeat in range(repeats):
            seed = int(config["episode_seed"]) + dataset_index * 100_000 + repeat * 10
            classification = task_type == "classification"
            c_local = sample_indices(train_y, int(config["context_size"]), seed + 1, classification)
            q_local = sample_indices(test_y, int(config["query_size"]), seed + 2, classification)
            context_x, context_y = train_x[c_local], train_y[c_local]
            query_x, query_y = test_x[q_local], test_y[q_local]
            if bool(config.get("episode_rescale", False)):
                episode_scaler = StandardScaler().fit(context_x)
                context_x = episode_scaler.transform(context_x)
                query_x = episode_scaler.transform(query_x)
                if task_type == "regression":
                    episode_mean = float(context_y.mean())
                    episode_scale = float(context_y.std()) or 1.0
                    context_y = (context_y - episode_mean) / episode_scale
                    query_y = (query_y - episode_mean) / episode_scale
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
            method_loss = {
                method: prediction_loss(query_y, weighted_prediction(predictions, weight), task_type)
                for method, weight in weights.items()
            }
            method_loss["best_individual_oracle"] = float(query_loss.min())
            losses_for_log.append(method_loss["competence"])
            episode_index = len(bundle["dataset"])
            for method, loss in method_loss.items():
                records.append({
                    "episode_index": episode_index, "dataset": name,
                    "task_type": task_type, "repeat": repeat,
                    "feature_count": train_x.shape[1], "method": method, "loss": loss,
                })
            bundle["dataset"].append(name)
            bundle["task_type"].append(task_type)
            bundle["repeat"].append(repeat)
            bundle["feature_count"].append(train_x.shape[1])
            bundle["context_index"].append(train_rows[c_local])
            bundle["query_index"].append(test_rows[q_local])
            bundle["cv_expert_loss"].append(cv_loss)
            bundle["query_expert_loss"].append(query_loss)
            bundle["expert_prediction"].append(predictions.astype(np.float32))
            bundle["query_y"].append(query_y.astype(np.float32))
            fits += len(EXPERTS) * (int(config["cv_folds"]) + 1)
        print(
            f"dataset={name} task={task_type} d={train_x.shape[1]} repeats={repeats} "
            f"mean_competence_loss={np.mean(losses_for_log):.6f}", flush=True
        )
    if args.smoke:
        print("smoke_ok")
        return
    arrays = {key: np.asarray(value) for key, value in bundle.items()}
    config_hash = sha256_file(config_path)
    run_key = f"{config.get('artifact_prefix', 'openml_breadth_competence')}_{config_hash[:10]}"
    raw_path = ROOT / "results/raw" / f"{run_key}.npz"
    cells_path = ROOT / "results/processed" / f"{run_key}_cells.csv"
    metadata_path = ROOT / "results/raw" / f"{run_key}.metadata.json"
    for output in (raw_path, cells_path, metadata_path):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
    np.savez_compressed(raw_path, **arrays)
    pd.DataFrame(records).to_csv(cells_path, index=False)
    metadata = {
        "run_key": run_key, "experiment": config["experiment"],
        "protocol": config.get("protocol", "OPENML_BREADTH_COMPETENCE_PROTOCOL.md"),
        "config": str(config_path.relative_to(ROOT)), "config_sha256": config_hash,
        "git_commit": git_commit(ROOT), "package_versions": package_versions(),
        "episodes": len(arrays["dataset"]), "expert_fits": fits,
        "wall_clock_seconds": time.perf_counter() - started,
        "synthetic_tuning": tuning, "dataset_audit": dataset_audit,
        "raw_bundle": str(raw_path.relative_to(ROOT)),
        "processed_summary": str(cells_path.relative_to(ROOT)),
    }
    write_json_exclusive(metadata_path, metadata)
    append_manifest(ROOT / "results/MANIFEST.jsonl", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
