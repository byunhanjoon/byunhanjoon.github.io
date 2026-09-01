#!/usr/bin/env python3
"""Small, immutable current-TFM phase diagram on PriorDial."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import log_loss, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import jensen_shannon, total_variation
from src.priors import PriorDial, balanced_coupling_schedule
from src.runio import append_manifest, git_commit, package_versions, sha256_file, write_json_exclusive
from src.transforms import MonotonePWLTransform, audit_transform


MECHANISMS = ("linear", "additive", "threshold", "interaction", "partition", "periodic")
WARPS = ("identity", "affine", "signed_power", "asinh", "pwl", "sinh")


def load_day8_adapters():
    path = ROOT.parent / "road-to-iclr-day-08/src/models/adapters.py"
    spec = importlib.util.spec_from_file_location("day8_model_adapters", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/e2_tfm_phase.yaml")
    parser.add_argument("--model", required=True, choices=["tabicl_v2_single", "tabicl_v2_default", "mitra_default"])
    parser.add_argument("--task-type", required=True, choices=["classification", "regression"])
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def nuisance_view(context: np.ndarray, query: np.ndarray, seed: int):
    transformed_context = np.empty_like(context)
    transformed_query = np.empty_like(query)
    states, audits = [], []
    for feature in range(context.shape[1]):
        transform = MonotonePWLTransform(seed=seed + feature, n_knots=7, slope_sigma=0.7).fit(context[:, feature])
        transformed_context[:, feature] = transform.transform(context[:, feature])
        transformed_query[:, feature] = transform.transform(query[:, feature])
        states.append(transform.state_dict())
        audits.append(audit_transform(transform, np.r_[context[:, feature], query[:, feature]]))
    if not all(
        audit["strictly_increasing"] and audit["ties_preserved"] and audit["all_finite"]
        and audit["inverse_max_scaled_error"] < 1e-8
        for audit in audits
    ):
        raise AssertionError("nuisance transform audit failed")
    return transformed_context, transformed_query, states


def fit_once(adapters, model, task_type, context_x, context_y, query_x, seed, device):
    columns = [f"x{index}" for index in range(context_x.shape[1])]
    outcomes = adapters.fit_predict_many(
        model,
        "binary" if task_type == "classification" else "regression",
        pd.DataFrame(context_x, columns=columns),
        context_y,
        {"query": pd.DataFrame(query_x, columns=columns)},
        categorical_columns=[], categorical_indices=[], seed=seed, device=device,
    )
    return outcomes["query"]


def cell_metrics(task_type, y, clean, matched, identity, context_y):
    if task_type == "classification":
        clean_loss = float(log_loss(y, clean, labels=[0, 1]))
        matched_loss = float(log_loss(y, matched, labels=[0, 1]))
        identity_disagreement = float(total_variation(clean, identity).mean())
        disagreement = float(total_variation(clean, matched).mean())
        return {
            "clean_loss": clean_loss,
            "matched_loss": matched_loss,
            "matched_loss_gap": matched_loss - clean_loss,
            "matched_disagreement": disagreement,
            "identity_disagreement": identity_disagreement,
            "excess_disagreement": disagreement - identity_disagreement,
            "matched_js": float(jensen_shannon(clean, matched).mean()),
            "identity_js": float(jensen_shannon(clean, identity).mean()),
            "matched_flip_rate": float(np.mean(clean.argmax(1) != matched.argmax(1))),
        }
    clean_loss = float(mean_squared_error(y, clean))
    matched_loss = float(mean_squared_error(y, matched))
    scale = max(float(np.std(context_y)), 1e-12)
    disagreement = float(np.mean(np.abs(clean - matched)) / scale)
    identity_disagreement = float(np.mean(np.abs(clean - identity)) / scale)
    return {
        "clean_loss": clean_loss,
        "matched_loss": matched_loss,
        "matched_loss_gap": matched_loss - clean_loss,
        "matched_disagreement": disagreement,
        "identity_disagreement": identity_disagreement,
        "excess_disagreement": disagreement - identity_disagreement,
        "matched_js": np.nan,
        "identity_js": np.nan,
        "matched_flip_rate": np.nan,
    }


def main() -> None:
    args = arguments()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    tasks = int(config["tasks_per_cell"])
    if tasks % len(MECHANISMS):
        raise ValueError("tasks_per_cell must be a multiple of six")
    config_hash = sha256_file(config_path)
    run_key = f"e2_{args.model}_{args.task_type}_{config_hash[:10]}"
    raw_path = ROOT / "results/raw" / f"{run_key}.npz"
    metadata_path = ROOT / "results/raw" / f"{run_key}.metadata.json"
    metrics_path = ROOT / "results/processed" / f"{run_key}_metrics.csv"
    if raw_path.exists() or metadata_path.exists() or metrics_path.exists():
        raise FileExistsError(run_key)
    adapters = load_day8_adapters()
    started = time.perf_counter()
    rows, predictions, labels = [], {"clean": [], "matched": [], "identity": []}, []
    rho_values, mechanism_values, warp_values = [], [], []
    checkpoint = None
    transform_states = []
    for rho_index, rho in enumerate(config["rhos"]):
        cell_seed = int(config["generator_seed"]) + rho_index * 10_000 + (0 if args.task_type == "classification" else 1_000_000)
        schedule = balanced_coupling_schedule(tasks, float(rho), np.random.default_rng(cell_seed))
        for episode_index, (mechanism, warp, coupled) in enumerate(schedule):
            episode_seed = cell_seed + episode_index + 1
            episode = PriorDial(
                seed=episode_seed,
                n_context=int(config["context_size"]), n_query=int(config["query_size"]),
                n_features=int(config["feature_count"]), task_type=args.task_type,
                informative_fraction=float(config["informative_fraction"]),
                correlation=float(config["correlation"]), label_noise=float(config["label_noise"]),
                classification_logit_scale=float(config["classification_logit_scale"]),
            ).generate(mechanism, warp, coupled)
            nuisance_context, nuisance_query, states = nuisance_view(
                episode.context_x, episode.query_x, episode_seed + 50_000
            )
            model_seed = episode_seed + 90_000
            clean = fit_once(adapters, args.model, args.task_type, episode.context_x, episode.context_y, episode.query_x, model_seed, args.device)
            matched = fit_once(adapters, args.model, args.task_type, nuisance_context, episode.context_y, nuisance_query, model_seed, args.device)
            identity = fit_once(adapters, args.model, args.task_type, episode.context_x, episode.context_y, episode.query_x, model_seed, args.device)
            checkpoint = checkpoint or clean.telemetry.get("checkpoint")
            metric = cell_metrics(
                args.task_type, episode.query_y, clean.prediction, matched.prediction,
                identity.prediction, episode.context_y,
            )
            rows.append({
                "model": args.model, "task_type": args.task_type, "rho": float(rho),
                "episode_seed": episode_seed, "mechanism": mechanism, "warp": warp,
                "coupled": coupled, **metric,
                "fit_seconds": sum(item.telemetry.get("fit_seconds", 0.0) for item in (clean, matched, identity)),
                "predict_seconds": sum(item.telemetry.get("predict_seconds", 0.0) for item in (clean, matched, identity)),
            })
            for key, outcome in (("clean", clean), ("matched", matched), ("identity", identity)):
                predictions[key].append(outcome.prediction)
            labels.append(episode.query_y)
            rho_values.append(float(rho)); mechanism_values.append(mechanism); warp_values.append(warp)
            transform_states.append(json.dumps(states, sort_keys=True))
            print(
                f"[{len(rows)}/{len(config['rhos']) * tasks}] model={args.model} task={args.task_type} "
                f"rho={rho} mechanism={mechanism} loss={metric['clean_loss']:.4f} "
                f"excess_disagreement={metric['excess_disagreement']:.4f}", flush=True,
            )
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            prediction_clean=np.asarray(predictions["clean"]),
            prediction_matched=np.asarray(predictions["matched"]),
            prediction_identity=np.asarray(predictions["identity"]),
            y_query=np.asarray(labels), rho=np.asarray(rho_values),
            mechanism=np.asarray(mechanism_values), warp=np.asarray(warp_values),
            transform_states=np.asarray(transform_states),
        )
    pd.DataFrame(rows).to_csv(metrics_path, index=False, mode="x")
    elapsed = time.perf_counter() - started
    metadata = {
        "run_key": run_key, "experiment": "E2", "model": args.model,
        "task_type": args.task_type, "config": str(config_path.relative_to(ROOT)),
        "config_sha256": config_hash, "git_commit": git_commit(ROOT),
        "package_versions": package_versions(), "checkpoint": checkpoint,
        "device": args.device, "episodes": len(rows), "wall_clock_seconds": elapsed,
        "raw_bundle": str(raw_path.relative_to(ROOT)),
        "processed_metrics": str(metrics_path.relative_to(ROOT)),
    }
    write_json_exclusive(metadata_path, metadata)
    append_manifest(ROOT / "results/MANIFEST.jsonl", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

