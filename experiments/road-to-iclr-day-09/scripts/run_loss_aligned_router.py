#!/usr/bin/env python3
"""Run the frozen context-only loss-aligned routing development/test phases."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesClassifier

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
from src.priors import MECHANISMS, WARPS, PriorDial, balanced_coupling_schedule
from src.representations import marginal_descriptors
from src.runio import append_manifest, git_commit, package_versions, sha256_file, write_json_exclusive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/fallback_loss_aligned_router.yaml")
    parser.add_argument("--phase", choices=("development", "test"), required=True)
    parser.add_argument("--smoke", action="store_true", help="run six tasks in the first cell without writing")
    return parser.parse_args()


def task_loss(y: np.ndarray, prediction: np.ndarray, task_type: str) -> float:
    return prediction_loss(y, prediction, task_type)


def mean_query_loss(y: np.ndarray, prediction: np.ndarray, task_type: str) -> float:
    if task_type == "classification":
        p = np.clip(prediction, 1e-6, 1 - 1e-6)
        return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))
    return float(np.mean((y - prediction) ** 2))


def fit_fixed_weights(predictions: np.ndarray, y: np.ndarray, task_type: str) -> np.ndarray:
    def objective(weights: np.ndarray) -> float:
        mixed = np.einsum("e,neq->nq", weights, predictions)
        return mean_query_loss(y, mixed, task_type)

    starts = [np.full(len(EXPERTS), 1 / len(EXPERTS))]
    starts.extend(np.eye(len(EXPERTS)))
    best = None
    for start in starts:
        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * len(EXPERTS),
            constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
            options={"maxiter": 300, "ftol": 1e-12},
        )
        if result.success and (best is None or result.fun < best.fun):
            best = result
    if best is None:
        raise RuntimeError("fixed-mixture optimization failed")
    weights = np.clip(best.x, 0.0, 1.0)
    weights /= weights.sum()
    return weights


def tune_competence(
    predictions: np.ndarray,
    y: np.ndarray,
    cv_losses: np.ndarray,
    task_type: str,
    temperatures: list[float],
    shrinkages: list[float],
) -> tuple[float, float, float]:
    best = None
    for temperature in temperatures:
        for shrinkage in shrinkages:
            weights = np.asarray([
                competence_weights(loss, temperature, shrinkage) for loss in cv_losses
            ])
            mixed = np.einsum("ne,neq->nq", weights, predictions)
            loss = mean_query_loss(y, mixed, task_type)
            candidate = (loss, temperature, shrinkage)
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    return best


def collect_phase(config: dict, phase: str, smoke: bool) -> tuple[dict[str, np.ndarray], float, int]:
    base_seed = int(config[f"{phase}_seed"])
    tasks = 6 if smoke else int(config[f"{phase}_tasks_per_cell"])
    query_size = min(32, int(config["query_size"])) if smoke else int(config["query_size"])
    regimes = config["regimes"][:1] if smoke else config["regimes"]
    rhos = config["rhos"][:1] if smoke else config["rhos"]
    task_types = config["task_types"]
    folds = int(config["cv_folds"])
    started = time.perf_counter()
    fit_count = 0
    bundle: dict[str, list] = {
        "split": [], "task_type": [], "rho": [], "effective_rho": [],
        "context_size": [], "feature_count": [], "mechanism": [], "warp": [],
        "descriptor": [], "cv_expert_loss": [], "query_expert_loss": [],
        "expert_prediction": [], "query_y": [],
    }
    for task_index, task_type in enumerate(task_types):
        for regime_index, regime in enumerate(regimes):
            n_context = int(regime["context_size"])
            n_features = int(regime["feature_count"])
            for rho_index, rho in enumerate(rhos):
                cell_seed = base_seed + task_index * 10_000_000 + regime_index * 100_000 + rho_index * 1_000
                schedule = balanced_coupling_schedule(tasks, float(rho), np.random.default_rng(cell_seed))
                cell_losses = []
                for episode_index, (mechanism, warp, coupled) in enumerate(schedule):
                    episode_seed = cell_seed + episode_index + 1
                    episode = PriorDial(
                        seed=episode_seed,
                        n_context=n_context,
                        n_query=query_size,
                        n_features=n_features,
                        task_type=task_type,
                        informative_fraction=0.75,
                        correlation=0.0,
                        label_noise=0.05,
                        classification_logit_scale=float(config["classification_logit_scale"]),
                    ).generate(mechanism, warp, coupled)
                    predictions = fit_predict_experts(
                        episode.context_x, episode.context_y, episode.query_x,
                        task_type, episode_seed + 700,
                    )
                    cv_loss = cross_validated_expert_losses(
                        episode.context_x, episode.context_y, task_type,
                        episode_seed + 500, folds,
                    )
                    query_loss = np.asarray([
                        task_loss(episode.query_y, prediction, task_type) for prediction in predictions
                    ])
                    cell_losses.append(query_loss.mean())
                    fit_count += len(EXPERTS) * (folds + 1)
                    bundle["split"].append(phase)
                    bundle["task_type"].append(task_type)
                    bundle["rho"].append(float(rho))
                    bundle["effective_rho"].append(float(np.mean([item[2] for item in schedule])))
                    bundle["context_size"].append(n_context)
                    bundle["feature_count"].append(n_features)
                    bundle["mechanism"].append(MECHANISMS.index(mechanism))
                    bundle["warp"].append(WARPS.index(warp))
                    bundle["descriptor"].append(marginal_descriptors(episode.context_x))
                    bundle["cv_expert_loss"].append(cv_loss)
                    bundle["query_expert_loss"].append(query_loss)
                    bundle["expert_prediction"].append(predictions.astype(np.float32))
                    bundle["query_y"].append(np.asarray(episode.query_y, dtype=np.float32))
                print(
                    f"phase={phase} task={task_type} n={n_context} d={n_features} rho={rho:.2f} "
                    f"episodes={tasks} mean_uniform_expert_loss={np.mean(cell_losses):.5f}",
                    flush=True,
                )
    arrays = {key: np.asarray(value) for key, value in bundle.items()}
    return arrays, time.perf_counter() - started, fit_count


def development_tuning(arrays: dict[str, np.ndarray], config: dict) -> dict:
    tuning = {}
    for task_type in config["task_types"]:
        mask = arrays["task_type"].astype(str) == task_type
        predictions = arrays["expert_prediction"][mask].astype(float)
        y = arrays["query_y"][mask].astype(float)
        cv_losses = arrays["cv_expert_loss"][mask].astype(float)
        fixed = fit_fixed_weights(predictions, y, task_type)
        competence = tune_competence(
            predictions, y, cv_losses, task_type,
            [float(x) for x in config["temperatures"]],
            [float(x) for x in config["uniform_shrinkages"]],
        )
        tuning[task_type] = {
            "fixed_weights": fixed.tolist(),
            "competence_development_loss": competence[0],
            "temperature": competence[1],
            "uniform_shrinkage": competence[2],
        }
    return tuning


def method_losses(
    arrays: dict[str, np.ndarray],
    config: dict,
    tuning: dict,
    development_arrays: dict[str, np.ndarray] | None,
) -> pd.DataFrame:
    records = []
    shape_models = {}
    if development_arrays is not None:
        for task_type in config["task_types"]:
            train_mask = development_arrays["task_type"].astype(str) == task_type
            shape_models[task_type] = ExtraTreesClassifier(
                n_estimators=400, min_samples_leaf=3, max_features="sqrt",
                random_state=17 if task_type == "classification" else 29, n_jobs=8,
            ).fit(
                development_arrays["descriptor"][train_mask],
                development_arrays["mechanism"][train_mask],
            )
    for index in range(len(arrays["rho"])):
        task_type = str(arrays["task_type"][index])
        predictions = arrays["expert_prediction"][index].astype(float)
        y = arrays["query_y"][index].astype(float)
        fixed_weights = np.asarray(tuning[task_type]["fixed_weights"], dtype=float)
        competence = competence_weights(
            arrays["cv_expert_loss"][index],
            float(tuning[task_type]["temperature"]),
            float(tuning[task_type]["uniform_shrinkage"]),
        )
        weights = {
            "uniform": np.full(len(EXPERTS), 1 / len(EXPERTS)),
            "fixed": fixed_weights,
            "competence": competence,
            "matched_family": np.eye(len(EXPERTS))[int(arrays["mechanism"][index])],
        }
        if shape_models:
            weights["shape_family"] = shape_models[task_type].predict_proba(
                arrays["descriptor"][index:index + 1]
            )[0]
        losses = {
            method: task_loss(y, weighted_prediction(predictions, weight), task_type)
            for method, weight in weights.items()
        }
        losses["best_individual_oracle"] = float(np.min(arrays["query_expert_loss"][index]))
        for method, loss in losses.items():
            records.append({
                "episode_index": index,
                "split": str(arrays["split"][index]),
                "task_type": task_type,
                "context_size": int(arrays["context_size"][index]),
                "feature_count": int(arrays["feature_count"][index]),
                "rho": float(arrays["rho"][index]),
                "mechanism": EXPERTS[int(arrays["mechanism"][index])],
                "method": method,
                "loss": loss,
                "competence_argmin": EXPERTS[int(np.argmin(arrays["cv_expert_loss"][index]))],
                "query_best_expert": EXPERTS[int(np.argmin(arrays["query_expert_loss"][index]))],
            })
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    arrays, elapsed, fit_count = collect_phase(config, args.phase, args.smoke)
    if args.smoke:
        tuning = development_tuning(arrays, config)
        print(json.dumps({"smoke": True, "episodes": len(arrays["rho"]), "tuning": tuning}, indent=2))
        return

    config_hash = sha256_file(config_path)
    run_key = f"fallback_loss_router_{config_hash[:10]}_{args.phase}"
    raw_path = ROOT / "results/raw" / f"{run_key}.npz"
    metadata_path = ROOT / "results/raw" / f"{run_key}.metadata.json"
    cells_path = ROOT / "results/processed" / f"{run_key}_cells.csv"
    tuning_path = ROOT / "results/processed" / f"fallback_loss_router_{config_hash[:10]}_tuning.json"
    for path in (raw_path, metadata_path, cells_path):
        if path.exists():
            raise FileExistsError(f"immutable output exists: {path}")

    if args.phase == "development":
        if tuning_path.exists():
            raise FileExistsError(f"immutable tuning exists: {tuning_path}")
        tuning = development_tuning(arrays, config)
        write_json_exclusive(tuning_path, tuning)
        frame = method_losses(arrays, config, tuning, None)
    else:
        if not tuning_path.exists():
            raise FileNotFoundError("development tuning must exist before test")
        tuning = json.loads(tuning_path.read_text())
        development_path = ROOT / "results/raw" / f"fallback_loss_router_{config_hash[:10]}_development.npz"
        if not development_path.exists():
            raise FileNotFoundError(development_path)
        development_arrays = dict(np.load(development_path, allow_pickle=False))
        frame = method_losses(arrays, config, tuning, development_arrays)

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    cells_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    frame.to_csv(cells_path, index=False, mode="x")
    metadata = {
        "run_key": run_key,
        "experiment": "fallback_loss_aligned_router",
        "phase": args.phase,
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": config_hash,
        "git_commit": git_commit(ROOT),
        "package_versions": package_versions(),
        "episodes": int(len(arrays["rho"])),
        "expert_fits": int(fit_count),
        "wall_clock_seconds": elapsed,
        "raw_bundle": str(raw_path.relative_to(ROOT)),
        "processed_summary": str(cells_path.relative_to(ROOT)),
        "tuning": tuning,
    }
    write_json_exclusive(metadata_path, metadata)
    append_manifest(ROOT / "results/MANIFEST.jsonl", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
