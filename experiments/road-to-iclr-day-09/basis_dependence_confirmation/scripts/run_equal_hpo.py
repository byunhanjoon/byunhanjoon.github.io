#!/usr/bin/env python3
"""Run the frozen nine-trial equal-HPO control on development datasets."""

from __future__ import annotations

import argparse
import copy
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
    environment_metadata, fit_predict, jsonable, load_dataset, prediction_metrics, sha256_file,
)


CONFIG_PATH = ROOT / "configs" / "development_protocol.yaml"
PANEL_PATH = ROOT / "configs" / "dataset_panel.json"


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def prediction_frame(
    data: Any, model: str, seed: int, representation: str, split: str,
    row_ids: np.ndarray, target: np.ndarray, prediction: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame({
        "dataset": data.key, "model": model, "model_seed": seed,
        "representation": representation, "split": split, "row_id": row_ids, "target": target,
    })
    values = np.asarray(prediction)
    if values.ndim == 1:
        frame["prediction"] = values
    else:
        for class_index in range(values.shape[1]):
            frame[f"prediction_{class_index}"] = values[:, class_index]
    return frame


def run_bundle(
    config: dict[str, Any], config_hash: str, panel_hash: str, spec: dict[str, Any],
    model_name: str, seed: int, device: str,
) -> str:
    if spec["panel"] != "development":
        raise RuntimeError("equal-HPO runner refuses prospective datasets")
    if model_name not in {"controlled_mlp", "tabm_d"}:
        raise ValueError("equal HPO is specified only for MLP/TabM")
    destination = ROOT / "results" / "raw" / "development" / "equal_hpo" / model_name / spec["key"] / f"seed_{seed}"
    if (destination / "metadata.json").exists():
        metadata = json.loads((destination / "metadata.json").read_text())
        if metadata["config_sha256"] != config_hash or metadata["dataset_panel_sha256"] != panel_hash:
            raise RuntimeError(f"frozen config drift at {destination}")
        return "cached"
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite incomplete bundle: {destination}")

    data = load_dataset(spec, config)
    blocks = build_rbf_feature_matrix(data, config)
    primary = build_primary_representations(blocks, 1)
    representations = [primary[0], next(rep for rep in primary if rep.variant == "orthogonal_all")]
    learning_rates = list(map(float, config["hpo"]["learning_rates"]))
    weight_decays = list(map(float, config["hpo"]["weight_decays"]))
    if len(learning_rates) * len(weight_decays) != int(config["hpo"]["maximum_trials_per_representation"]):
        raise RuntimeError("HPO grid no longer matches frozen nine-trial budget")
    trial_records: list[dict[str, Any]] = []
    trial_predictions: dict[tuple[str, float, float, str], np.ndarray] = {}
    started = time.time()
    for rep in representations:
        for learning_rate in learning_rates:
            for weight_decay in weight_decays:
                trial_config = copy.deepcopy(config)
                trial_config["models"][model_name]["learning_rate"] = learning_rate
                trial_config["models"][model_name]["weight_decay"] = weight_decay
                validation, test, telemetry = fit_predict(
                    model_name, data.problem_type, rep, data.y_train, data.y_validation, seed, device, trial_config
                )
                trial_predictions[(rep.representation_id, learning_rate, weight_decay, "validation")] = validation
                trial_predictions[(rep.representation_id, learning_rate, weight_decay, "test")] = test
                for split, y, prediction in (
                    ("validation", data.y_validation, validation), ("test", data.y_test, test)
                ):
                    trial_records.append({
                        "dataset": data.key, "problem_type": data.problem_type, "model": model_name,
                        "model_seed": seed, "representation_id": rep.representation_id,
                        "variant": rep.variant, "learning_rate": learning_rate,
                        "weight_decay": weight_decay, "split": split,
                        **prediction_metrics(data.problem_type, y, prediction), **telemetry,
                    })
                print(f"{data.key} {model_name} seed={seed} {rep.representation_id} lr={learning_rate} wd={weight_decay}", flush=True)

    trials = pd.DataFrame(trial_records)
    validation_metric = "log_loss" if data.problem_type == "classification" else "rmse"
    selections: dict[str, dict[str, Any]] = {}
    selected_predictions: dict[tuple[str, str], np.ndarray] = {}
    for rep in representations:
        candidates = trials[(trials["representation_id"] == rep.representation_id) & (trials["split"] == "validation")]
        best = candidates.sort_values(
            [validation_metric, "learning_rate", "weight_decay"], kind="mergesort"
        ).iloc[0]
        learning_rate, weight_decay = float(best.learning_rate), float(best.weight_decay)
        selections[rep.representation_id] = {
            "selection_split": "validation", "selection_metric": validation_metric,
            "learning_rate": learning_rate, "weight_decay": weight_decay,
            "validation_metric_value": float(best[validation_metric]),
        }
        for split in ("validation", "test"):
            selected_predictions[(rep.representation_id, split)] = trial_predictions[
                (rep.representation_id, learning_rate, weight_decay, split)
            ]
    reference, transformed = representations
    comparison_records = []
    prediction_parts = []
    for split, row_ids, y in (
        ("validation", data.validation_indices, data.y_validation),
        ("test", data.test_indices, data.y_test),
    ):
        p_ref = selected_predictions[(reference.representation_id, split)]
        p_alt = selected_predictions[(transformed.representation_id, split)]
        comparison_records.append({
            "dataset": data.key, "problem_type": data.problem_type, "model": model_name,
            "model_seed": seed, "split": split,
            **{f"reference_{key}": value for key, value in prediction_metrics(data.problem_type, y, p_ref).items()},
            **{f"transformed_{key}": value for key, value in prediction_metrics(data.problem_type, y, p_alt).items()},
            **disagreement_metrics(data.problem_type, y, p_ref, p_alt),
        })
        prediction_parts.extend([
            prediction_frame(data, model_name, seed, "reference", split, row_ids, y, p_ref),
            prediction_frame(data, model_name, seed, "transformed", split, row_ids, y, p_alt),
        ])

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        trials.to_csv(temporary / "trials.csv", index=False)
        pd.DataFrame(comparison_records).to_csv(temporary / "selected_comparison.csv", index=False)
        pd.concat(prediction_parts, ignore_index=True).to_csv(
            temporary / "selected_predictions.csv.gz", index=False, compression="gzip"
        )
        (temporary / "selection.json").write_text(json.dumps(jsonable(selections), indent=2, sort_keys=True) + "\n")
        metadata = {
            "status": "complete", "stage": "development_equal_hpo", "dataset_spec": spec,
            "model": model_name, "model_seed": seed, "device": device,
            "config_sha256": config_hash, "dataset_panel_sha256": panel_hash,
            "representation_rule": "reference and fixed orthogonal_all member 0",
            "selection_uses_validation_only": True, "trials_per_representation": 9,
            "grid": {"learning_rates": learning_rates, "weight_decays": weight_decays},
            "wall_seconds": time.time() - started, "environment": environment_metadata(), "files": {},
        }
        for filename in ("trials.csv", "selected_comparison.csv", "selected_predictions.csv.gz", "selection.json"):
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
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--model", action="append", choices=["controlled_mlp", "tabm_d"])
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config_bytes, panel_bytes = CONFIG_PATH.read_bytes(), PANEL_PATH.read_bytes()
    config, panel = yaml.safe_load(config_bytes), json.loads(panel_bytes)
    selected = set(args.dataset)
    specs = [spec for spec in panel["datasets"] if spec["key"] in selected]
    if len(specs) != len(selected):
        raise SystemExit(f"unknown datasets: {selected - {spec['key'] for spec in specs}}")
    models = args.model or ["controlled_mlp", "tabm_d"]
    seeds = args.seed or list(map(int, config["model_seeds"]))
    failures = []
    for spec in specs:
        for model in models:
            for seed in seeds:
                print(f"=== {spec['key']} {model} seed={seed} ===", flush=True)
                try:
                    print(run_bundle(config, digest(config_bytes), digest(panel_bytes), spec, model, seed, args.device), flush=True)
                except Exception as error:
                    failures.append({"dataset": spec["key"], "model": model, "seed": seed,
                                     "error": repr(error), "traceback": traceback.format_exc()})
                    print(json.dumps(failures[-1]), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
