#!/usr/bin/env python3
"""Run the frozen M0--M5 development/test oracle-headroom kill test."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.model_selection import StratifiedKFold, cross_val_predict


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.methods import (
    context_gate_descriptor,
    episode_loss,
    featurewise_pooled_gate_descriptor,
    fit_knn_views,
    mixture_loss_curve,
)
from src.priors import PriorDial, balanced_coupling_schedule
from src.runio import append_manifest, git_commit, package_versions, sha256_file, write_json_exclusive


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/e3_method_kill.yaml")
    return parser.parse_args()


def choose_fixed_alpha(indices, labels, raw, rank, task_types, alphas, clip):
    risk = []
    for alpha in alphas:
        risk.append(np.mean([
            episode_loss(labels[index], alpha * raw[index] + (1 - alpha) * rank[index], task_types[index], clip)
            for index in indices
        ]))
    best = int(np.argmin(risk))
    return float(alphas[best]), float(risk[best])


def mean_gate_risk(indices, beta, fixed_alpha, proposed_alpha, labels, raw, rank, task_types, clip):
    return float(np.mean([
        episode_loss(
            labels[index],
            (fixed_alpha + beta * (proposed_alpha[index] - fixed_alpha)) * raw[index]
            + (1 - fixed_alpha - beta * (proposed_alpha[index] - fixed_alpha)) * rank[index],
            task_types[index], clip,
        )
        for index in indices
    ]))


def logit_calibrate(values, temperature, bias):
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-5, 1 - 1e-5)
    logits = np.log(clipped / (1 - clipped))
    calibrated = 1 / (1 + np.exp(-(logits + bias) / temperature))
    return calibrated


def main() -> None:
    args = arguments()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    config_hash = sha256_file(config_path)
    run_key = f"e3_method_kill_{config_hash[:10]}"
    raw_path = ROOT / "results/raw" / f"{run_key}.npz"
    metadata_path = ROOT / "results/raw" / f"{run_key}.metadata.json"
    processed_path = ROOT / "results/processed" / f"{run_key}_episodes.csv"
    if any(path.exists() for path in (raw_path, metadata_path, processed_path)):
        raise FileExistsError(run_key)

    backbone = config["backbone"]
    augmentation = config["augmentation"]
    clip = float(config["classification_probability_clip"])
    oracle_grid = np.linspace(0, 1, int(config["oracle_alpha_grid_points"]))
    fixed_grid = np.linspace(0, 1, int(config["fixed_alpha_grid_points"]))
    started = time.perf_counter()
    descriptor_function = (
        featurewise_pooled_gate_descriptor
        if config.get("gate_descriptor", "aggregate_v1") == "featurewise_pooled_v1"
        else context_gate_descriptor
    )

    records: list[dict] = []
    labels: list[np.ndarray] = []
    descriptors: list[np.ndarray] = []
    predictions = {name: [] for name in ("raw", "robust", "rank", "augmentation")}
    oracle_alphas: list[float] = []
    task_types: list[str] = []
    total = sum(int(config["split_tasks_per_cell"][split]) for split in config["split_tasks_per_cell"])
    total *= len(config["rhos"]) * len(config["task_types"])

    for split_index, split in enumerate(("train", "development", "test")):
        tasks = int(config["split_tasks_per_cell"][split])
        if tasks % 6:
            raise ValueError("each split cell must be divisible by six mechanisms")
        base_seed = int(config["split_generator_seeds"][split])
        for task_index, task_type in enumerate(config["task_types"]):
            for rho_index, rho in enumerate(config["rhos"]):
                cell_seed = base_seed + task_index * 10_000_000 + rho_index * 100_000
                schedule = balanced_coupling_schedule(tasks, float(rho), np.random.default_rng(cell_seed))
                for episode_index, (mechanism, warp, coupled) in enumerate(schedule):
                    episode_seed = cell_seed + episode_index + 1
                    episode = PriorDial(
                        seed=episode_seed,
                        n_context=int(config["context_size"]), n_query=int(config["query_size"]),
                        n_features=int(config["feature_count"]), task_type=task_type,
                        informative_fraction=float(config["informative_fraction"]),
                        correlation=float(config["correlation"]), label_noise=float(config["label_noise"]),
                        classification_logit_scale=float(config["classification_logit_scale"]),
                    ).generate(mechanism, warp, coupled)
                    view = fit_knn_views(
                        episode,
                        neighbors=int(backbone["neighbors"]), metric_p=int(backbone["metric_p"]),
                        augmentation_seed=episode_seed + 70_000,
                        augmentation_knots=int(augmentation["n_knots"]),
                        augmentation_slope_sigma=float(augmentation["slope_sigma"]),
                    )
                    curve = mixture_loss_curve(
                        episode.query_y, view["raw"], view["rank"], task_type, oracle_grid, clip
                    )
                    oracle_index = int(np.argmin(curve))
                    record = {
                        "split": split, "task_type": task_type, "rho": float(rho),
                        "effective_rho": float(np.mean([x[2] for x in schedule])),
                        "mechanism": mechanism, "warp": warp, "coupled": bool(coupled),
                        "episode_seed": episode_seed,
                        "m0_raw_loss": episode_loss(episode.query_y, view["raw"], task_type, clip),
                        "m1_robust_loss": episode_loss(episode.query_y, view["robust"], task_type, clip),
                        "m2_rank_loss": episode_loss(episode.query_y, view["rank"], task_type, clip),
                        "m3_augmentation_loss": episode_loss(episode.query_y, view["augmentation"], task_type, clip),
                        "m4_50_loss": episode_loss(episode.query_y, 0.5 * (view["raw"] + view["rank"]), task_type, clip),
                        "oracle_alpha": float(oracle_grid[oracle_index]),
                        "oracle_loss": float(curve[oracle_index]),
                    }
                    records.append(record)
                    labels.append(np.asarray(episode.query_y))
                    descriptors.append(descriptor_function(episode.context_x, episode.context_y))
                    for name in predictions:
                        predictions[name].append(view[name])
                    oracle_alphas.append(float(oracle_grid[oracle_index]))
                    task_types.append(task_type)
                    if len(records) % 120 == 0 or len(records) == total:
                        print(f"[{len(records)}/{total}] split={split} task={task_type} rho={rho}", flush=True)

    descriptor_array = np.asarray(descriptors, dtype=np.float64)
    label_array = np.asarray(labels)
    prediction_arrays = {key: np.asarray(value, dtype=np.float64) for key, value in predictions.items()}
    oracle_alpha_array = np.asarray(oracle_alphas)
    task_type_array = np.asarray(task_types)
    split_array = np.asarray([record["split"] for record in records])
    mechanism_array = np.asarray([record["mechanism"] for record in records])
    warp_array = np.asarray([record["warp"] for record in records])
    learned_alpha = np.empty(len(records), dtype=np.float64)
    selections: dict[str, dict] = {}

    gate_input = descriptor_array
    auxiliary_config = config.get("gate_auxiliary")
    auxiliary_details: dict[str, dict] = {}
    if auxiliary_config:
        auxiliary_probability = np.zeros((len(records), 12), dtype=np.float64)
        for task_offset, task_type in enumerate(config["task_types"]):
            train = np.flatnonzero((split_array == "train") & (task_type_array == task_type))
            nontrain = np.flatnonzero((split_array != "train") & (task_type_array == task_type))
            details = {}
            for target_offset, (target_name, target) in enumerate(
                (("mechanism", mechanism_array), ("warp", warp_array))
            ):
                class_names = sorted(np.unique(target[train]).tolist())
                class_index = {name: index for index, name in enumerate(class_names)}
                encoded = np.asarray([class_index[value] for value in target[train]])
                seed = int(auxiliary_config["random_state"]) + task_offset * 10 + target_offset
                folds = StratifiedKFold(
                    n_splits=int(auxiliary_config["folds"]), shuffle=True, random_state=seed
                )
                model = ExtraTreesClassifier(
                    n_estimators=int(auxiliary_config["n_estimators"]),
                    min_samples_leaf=int(auxiliary_config["min_samples_leaf"]),
                    max_features=auxiliary_config["max_features"],
                    random_state=seed, n_jobs=int(auxiliary_config["n_jobs"]),
                )
                train_probability = cross_val_predict(
                    model, descriptor_array[train], encoded, cv=folds,
                    method="predict_proba", n_jobs=1,
                )
                model.fit(descriptor_array[train], encoded)
                nontrain_probability = model.predict_proba(descriptor_array[nontrain])
                start = 0 if target_name == "mechanism" else 6
                auxiliary_probability[train, start:start + 6] = train_probability
                auxiliary_probability[nontrain, start:start + 6] = nontrain_probability
                details[f"{target_name}_train_oof_accuracy"] = float(
                    np.mean(train_probability.argmax(axis=1) == encoded)
                )
            auxiliary_details[task_type] = details
        gate_input = np.c_[descriptor_array, auxiliary_probability]

    gate_config = config["gate"]
    for task_offset, task_type in enumerate(config["task_types"]):
        train = np.flatnonzero((split_array == "train") & (task_type_array == task_type))
        development = np.flatnonzero((split_array == "development") & (task_type_array == task_type))
        all_task = np.flatnonzero(task_type_array == task_type)
        fixed_alpha, fixed_development_risk = choose_fixed_alpha(
            development, label_array, prediction_arrays["raw"], prediction_arrays["rank"],
            task_type_array, fixed_grid, clip,
        )
        gate = ExtraTreesRegressor(
            n_estimators=int(gate_config["n_estimators"]),
            min_samples_leaf=int(gate_config["min_samples_leaf"]),
            max_features=gate_config["max_features"],
            random_state=int(gate_config["random_state"]) + task_offset,
            n_jobs=int(gate_config["n_jobs"]),
        )
        gate.fit(gate_input[train], oracle_alpha_array[train])
        proposed = np.clip(gate.predict(gate_input), 0, 1)
        calibration_config = config.get("gate_calibration")
        temperatures = calibration_config["temperature_grid"] if calibration_config else [1.0]
        biases = calibration_config["bias_grid"] if calibration_config else [0.0]
        calibration_risks = {}
        calibrated_candidates = {}
        for temperature in temperatures:
            for bias in biases:
                calibrated = logit_calibrate(proposed, float(temperature), float(bias))
                for beta in gate_config["shrinkage_grid"]:
                    key = (float(temperature), float(bias), float(beta))
                    calibration_risks[key] = mean_gate_risk(
                        development, float(beta), fixed_alpha, calibrated,
                        label_array, prediction_arrays["raw"], prediction_arrays["rank"], task_type_array, clip,
                    )
                    calibrated_candidates[key] = calibrated
        selected_key = min(calibration_risks, key=calibration_risks.get)
        selected_temperature, selected_bias, selected_beta = selected_key
        selected_proposed = calibrated_candidates[selected_key]
        learned_alpha[all_task] = np.clip(
            fixed_alpha + selected_beta * (selected_proposed[all_task] - fixed_alpha), 0, 1
        )
        selections[task_type] = {
            "fixed_alpha": fixed_alpha,
            "fixed_development_risk": fixed_development_risk,
            "selected_gate_shrinkage": selected_beta,
            "selected_temperature": selected_temperature,
            "selected_logit_bias": selected_bias,
            "selected_gate_development_risk": calibration_risks[selected_key],
            "development_risk_by_calibration": {
                f"temperature={key[0]},bias={key[1]},shrinkage={key[2]}": value
                for key, value in calibration_risks.items()
            },
            "gate_train_episodes": len(train),
            "gate_development_episodes": len(development),
        }

    for index, record in enumerate(records):
        task_type = task_type_array[index]
        fixed_alpha = selections[task_type]["fixed_alpha"]
        raw = prediction_arrays["raw"][index]
        rank = prediction_arrays["rank"][index]
        record["m4_fixed_alpha"] = fixed_alpha
        record["m4_fixed_loss"] = episode_loss(
            label_array[index], fixed_alpha * raw + (1 - fixed_alpha) * rank, task_type, clip
        )
        record["m5_gate_alpha"] = learned_alpha[index]
        record["m5_gate_loss"] = episode_loss(
            label_array[index], learned_alpha[index] * raw + (1 - learned_alpha[index]) * rank,
            task_type, clip,
        )

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            y_query=label_array.astype(np.float32),
            gate_descriptor=descriptor_array.astype(np.float32),
            gate_input=gate_input.astype(np.float32),
            prediction_raw=prediction_arrays["raw"].astype(np.float32),
            prediction_robust=prediction_arrays["robust"].astype(np.float32),
            prediction_rank=prediction_arrays["rank"].astype(np.float32),
            prediction_augmentation=prediction_arrays["augmentation"].astype(np.float32),
            oracle_alpha=oracle_alpha_array.astype(np.float32),
            learned_alpha=learned_alpha.astype(np.float32),
            split=split_array, task_type=task_type_array,
            rho=np.asarray([record["rho"] for record in records]),
            mechanism=mechanism_array,
            warp=warp_array,
            episode_seed=np.asarray([record["episode_seed"] for record in records]),
        )
    pd.DataFrame(records).to_csv(processed_path, index=False, mode="x")
    elapsed = time.perf_counter() - started
    metadata = {
        "run_key": run_key, "experiment": "E3", "config": str(config_path.relative_to(ROOT)),
        "config_sha256": config_hash, "git_commit": git_commit(ROOT),
        "package_versions": package_versions(), "episodes": len(records),
        "expert_fits": len(records) * 4, "wall_clock_seconds": elapsed,
        "selections": selections, "auxiliary_details": auxiliary_details,
        "raw_bundle": str(raw_path.relative_to(ROOT)),
        "processed_episodes": str(processed_path.relative_to(ROOT)),
    }
    write_json_exclusive(metadata_path, metadata)
    append_manifest(ROOT / "results/MANIFEST.jsonl", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
