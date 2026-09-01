#!/usr/bin/env python3
"""Validate that PriorDial controls mechanism information before method training."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.priors import PriorDial, balanced_coupling_schedule
from src.representations import TieAwareECDF, marginal_descriptors
from src.runio import append_manifest, git_commit, package_versions, sha256_file, write_json_exclusive
from src.stats import mean_interval


MECHANISMS = ("linear", "additive", "threshold", "interaction", "partition", "periodic")
WARPS = ("identity", "affine", "signed_power", "asinh", "pwl", "sinh")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/prior_dial_dev.yaml")
    parser.add_argument("--tasks-per-cell", type=int)
    parser.add_argument("--context-sizes", type=str)
    parser.add_argument("--run-tag", default="e1")
    return parser.parse_args()


def simple_losses(episode) -> tuple[float, float]:
    ranker = TieAwareECDF().fit(episode.context_x)
    rank_context = ranker.transform(episode.context_x)
    rank_query = ranker.transform(episode.query_x)
    if episode.metadata["task_type"] == "classification":
        if np.unique(episode.context_y).size == 1:
            p = np.clip(float(episode.context_y[0]), 1e-6, 1 - 1e-6)
            raw_prediction = rank_prediction = np.full(episode.query_y.size, p)
        else:
            raw = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=300))
            rank = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=300))
            raw.fit(episode.context_x, episode.context_y)
            rank.fit(rank_context, episode.context_y)
            raw_prediction = raw.predict_proba(episode.query_x)[:, 1]
            rank_prediction = rank.predict_proba(rank_query)[:, 1]
        return (
            float(log_loss(episode.query_y, raw_prediction, labels=[0, 1])),
            float(log_loss(episode.query_y, rank_prediction, labels=[0, 1])),
        )
    raw = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    rank = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    raw.fit(episode.context_x, episode.context_y)
    rank.fit(rank_context, episode.context_y)
    return (
        float(mean_squared_error(episode.query_y, raw.predict(episode.query_x))),
        float(mean_squared_error(episode.query_y, rank.predict(rank_query))),
    )


def stable_descriptor(episode) -> np.ndarray:
    """Context-label summary using only the strictly increasing invariant channel."""
    rank = TieAwareECDF().fit(episode.context_x).transform(episode.context_x)
    y = np.asarray(episode.context_y, dtype=float)
    y_centered = y - y.mean()
    correlations = []
    for col in rank.T:
        col_centered = col - col.mean()
        denominator = np.sqrt(np.sum(col_centered**2) * np.sum(y_centered**2))
        correlations.append(0.0 if denominator <= 1e-12 else float(col_centered @ y_centered / denominator))
    correlations = np.asarray(correlations)
    return np.r_[
        np.sort(correlations),
        np.sort(np.abs(correlations)),
        y.mean(),
        y.std(),
        episode.context_x.shape[0],
        episode.context_x.shape[1],
    ]


def _periodic_features(x: np.ndarray) -> np.ndarray:
    frequencies = (0.75, 1.5, 2.5, 4.0)
    return np.concatenate(
        [function(frequency * x) for frequency in frequencies for function in (np.sin, np.cos)],
        axis=1,
    )


def expert_predictions(episode, seed: int) -> np.ndarray:
    """Six cheap, predeclared mechanism-family experts fit on context labels only."""
    x_context, x_query = episode.context_x, episode.query_x
    y = episode.context_y
    classification = episode.metadata["task_type"] == "classification"
    if classification and np.unique(y).size == 1:
        return np.full((len(MECHANISMS), episode.query_y.size), float(y[0]))

    linear = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=300) if classification else Ridge(alpha=1.0),
    )
    additive = make_pipeline(
        SplineTransformer(n_knots=4, degree=2, include_bias=False),
        StandardScaler(),
        LogisticRegression(C=0.5, max_iter=300) if classification else Ridge(alpha=2.0),
    )
    threshold = (
        RandomForestClassifier(n_estimators=40, min_samples_leaf=3, max_features="sqrt", random_state=seed, n_jobs=1)
        if classification
        else RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features="sqrt", random_state=seed, n_jobs=1)
    )
    interaction = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        StandardScaler(),
        LogisticRegression(C=0.25, max_iter=300) if classification else Ridge(alpha=4.0),
    )
    partition = (
        DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=seed)
        if classification
        else DecisionTreeRegressor(max_depth=3, min_samples_leaf=3, random_state=seed)
    )
    periodic = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.5, max_iter=300) if classification else Ridge(alpha=2.0),
    )
    models = (linear, additive, threshold, interaction, partition, periodic)
    predictions = []
    for index, model in enumerate(models):
        train_x = _periodic_features(x_context) if index == 5 else x_context
        query_x = _periodic_features(x_query) if index == 5 else x_query
        model.fit(train_x, y)
        prediction = model.predict_proba(query_x)[:, 1] if classification else model.predict(query_x)
        if classification:
            # Predeclared empirical-Bayes smoothing prevents tiny-context tree experts
            # from turning a selector error into arbitrarily large log loss.
            prior_strength = 20.0
            prediction = (
                len(y) * prediction + prior_strength * float(np.mean(y))
            ) / (len(y) + prior_strength)
        predictions.append(prediction)
    return np.asarray(predictions)


def mixture_loss(y: np.ndarray, expert_prediction: np.ndarray, weights: np.ndarray, task_type: str) -> float:
    prediction = np.sum(expert_prediction * weights[:, None], axis=0)
    if task_type == "classification":
        return float(log_loss(y, np.clip(prediction, 1e-6, 1 - 1e-6), labels=[0, 1]))
    return float(mean_squared_error(y, prediction))


def empirical_mi(mechanism: np.ndarray, warp: np.ndarray) -> float:
    table = np.zeros((len(MECHANISMS), len(WARPS)), dtype=float)
    for c, w in zip(mechanism, warp, strict=True):
        table[c, w] += 1
    joint = table / table.sum()
    pc = joint.sum(axis=1, keepdims=True)
    pw = joint.sum(axis=0, keepdims=True)
    positive = joint > 0
    return float(np.sum(joint[positive] * np.log(joint[positive] / (pc @ pw)[positive])))


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    tasks = int(args.tasks_per_cell or config["tasks_per_cell"])
    if tasks % len(MECHANISMS):
        raise ValueError(f"tasks-per-cell must be divisible by {len(MECHANISMS)}")
    contexts = (
        [int(x) for x in args.context_sizes.split(",")]
        if args.context_sizes
        else [int(x) for x in config["context_sizes"]]
    )
    rhos = [float(x) for x in config["rhos"]]
    task_types = list(config["task_types"])
    seed = int(config.get("generator_seed", config.get("seed")))
    n_features = int(config.get("primary_feature_counts", config["feature_counts"])[0])
    n_query = int(config["query_size"])
    draws = int(config.get("bootstrap_draws", 2000))

    run_key = f"{args.run_tag}_{sha256_file(config_path)[:10]}_t{tasks}_n{'-'.join(map(str, contexts))}"
    raw_path = ROOT / "results/raw" / f"{run_key}.npz"
    metadata_path = ROOT / "results/raw" / f"{run_key}.metadata.json"
    processed_path = ROOT / "results/processed" / f"{run_key}_summary.csv"
    if raw_path.exists() or metadata_path.exists() or processed_path.exists():
        raise FileExistsError(f"immutable run already exists: {run_key}")

    started = time.perf_counter()
    records: list[dict] = []
    bundle: dict[str, list] = {
        "descriptor": [], "stable_descriptor": [], "mechanism": [], "warp": [], "rho": [], "effective_rho": [],
        "context_size": [], "task_type": [], "raw_loss": [], "rank_loss": [],
        "oof_probability": [], "stable_oof_probability": [], "combined_oof_probability": [],
        "uniform_expert_loss": [], "stable_expert_loss": [], "combined_expert_loss": [],
        "oracle_expert_loss": [],
    }
    for task_index, task_type in enumerate(task_types):
        for context_index, n_context in enumerate(contexts):
            for rho_index, rho in enumerate(rhos):
                cell_seed = seed + task_index * 10_000_000 + context_index * 100_000 + rho_index * 1_000
                schedule_rng = np.random.default_rng(cell_seed)
                schedule = balanced_coupling_schedule(tasks, rho, schedule_rng)
                descriptors, stable_descriptors, mechanisms, warps = [], [], [], []
                raw_losses, rank_losses, all_expert_predictions, all_query_y = [], [], [], []
                for episode_index, (mechanism, warp, coupled) in enumerate(schedule):
                    generator = PriorDial(
                        seed=cell_seed + episode_index + 1,
                        n_context=n_context,
                        n_query=n_query,
                        n_features=n_features,
                        task_type=task_type,
                        informative_fraction=0.75,
                        correlation=0.0,
                        label_noise=0.05,
                        classification_logit_scale=float(config.get("classification_logit_scale", 2.5)),
                    )
                    episode = generator.generate(mechanism, warp, coupled)
                    descriptors.append(marginal_descriptors(episode.context_x))
                    stable_descriptors.append(stable_descriptor(episode))
                    mechanisms.append(MECHANISMS.index(mechanism))
                    warps.append(WARPS.index(warp))
                    raw_loss, rank_loss = simple_losses(episode)
                    raw_losses.append(raw_loss)
                    rank_losses.append(rank_loss)
                    all_expert_predictions.append(expert_predictions(episode, cell_seed + episode_index))
                    all_query_y.append(episode.query_y)
                x = np.asarray(descriptors)
                stable_x = np.asarray(stable_descriptors)
                y = np.asarray(mechanisms)
                w = np.asarray(warps)
                folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=cell_seed)
                model = ExtraTreesClassifier(
                    n_estimators=300,
                    min_samples_leaf=3,
                    max_features="sqrt",
                    random_state=cell_seed,
                    n_jobs=8,
                )
                probability = cross_val_predict(model, x, y, cv=folds, method="predict_proba")
                stable_model = ExtraTreesClassifier(
                    n_estimators=300, min_samples_leaf=3, max_features="sqrt",
                    random_state=cell_seed + 17, n_jobs=8,
                )
                combined_model = ExtraTreesClassifier(
                    n_estimators=300, min_samples_leaf=3, max_features="sqrt",
                    random_state=cell_seed + 23, n_jobs=8,
                )
                stable_probability = cross_val_predict(
                    stable_model, stable_x, y, cv=folds, method="predict_proba"
                )
                combined_probability = cross_val_predict(
                    combined_model, np.c_[stable_x, x], y, cv=folds, method="predict_proba"
                )
                correct = (probability.argmax(axis=1) == y).astype(float)
                accuracy, accuracy_low, accuracy_high = mean_interval(correct, draws, cell_seed)
                raw_array, rank_array = np.asarray(raw_losses), np.asarray(rank_losses)
                advantage, adv_low, adv_high = mean_interval(rank_array - raw_array, draws, cell_seed + 1)
                uniform_losses, stable_losses, combined_losses, oracle_losses = [], [], [], []
                for index, (predictions, query_y) in enumerate(
                    zip(all_expert_predictions, all_query_y, strict=True)
                ):
                    uniform_losses.append(
                        mixture_loss(query_y, predictions, np.full(len(MECHANISMS), 1 / len(MECHANISMS)), task_type)
                    )
                    stable_losses.append(
                        mixture_loss(query_y, predictions, stable_probability[index], task_type)
                    )
                    combined_losses.append(
                        mixture_loss(query_y, predictions, combined_probability[index], task_type)
                    )
                    oracle_weight = np.eye(len(MECHANISMS))[y[index]]
                    oracle_losses.append(mixture_loss(query_y, predictions, oracle_weight, task_type))
                uniform_losses = np.asarray(uniform_losses)
                stable_losses = np.asarray(stable_losses)
                combined_losses = np.asarray(combined_losses)
                oracle_losses = np.asarray(oracle_losses)
                marginal_utility, utility_low, utility_high = mean_interval(
                    stable_losses - combined_losses, draws, cell_seed + 2
                )
                effective_rho = float(np.mean([item[2] for item in schedule]))
                records.append({
                    "task_type": task_type,
                    "context_size": n_context,
                    "rho": rho,
                    "effective_rho": effective_rho,
                    "tasks": tasks,
                    "mechanism_accuracy": accuracy,
                    "mechanism_accuracy_ci_low": accuracy_low,
                    "mechanism_accuracy_ci_high": accuracy_high,
                    "mechanism_log_loss": float(log_loss(y, probability, labels=np.arange(len(MECHANISMS)))),
                    "stable_mechanism_accuracy": float(np.mean(stable_probability.argmax(axis=1) == y)),
                    "combined_mechanism_accuracy": float(np.mean(combined_probability.argmax(axis=1) == y)),
                    "empirical_mi_c_w_nats": empirical_mi(y, w),
                    "raw_mean_loss": float(raw_array.mean()),
                    "rank_mean_loss": float(rank_array.mean()),
                    "rank_minus_raw_loss": advantage,
                    "rank_minus_raw_ci_low": adv_low,
                    "rank_minus_raw_ci_high": adv_high,
                    "uniform_expert_loss": float(uniform_losses.mean()),
                    "stable_expert_loss": float(stable_losses.mean()),
                    "combined_expert_loss": float(combined_losses.mean()),
                    "oracle_expert_loss": float(oracle_losses.mean()),
                    "marginal_query_utility": marginal_utility,
                    "marginal_query_utility_ci_low": utility_low,
                    "marginal_query_utility_ci_high": utility_high,
                })
                for key, values in {
                    "descriptor": x,
                    "stable_descriptor": stable_x,
                    "mechanism": y,
                    "warp": w,
                    "raw_loss": raw_array,
                    "rank_loss": rank_array,
                    "oof_probability": probability,
                    "stable_oof_probability": stable_probability,
                    "combined_oof_probability": combined_probability,
                    "uniform_expert_loss": uniform_losses,
                    "stable_expert_loss": stable_losses,
                    "combined_expert_loss": combined_losses,
                    "oracle_expert_loss": oracle_losses,
                }.items():
                    bundle[key].extend(values)
                bundle["rho"].extend([rho] * tasks)
                bundle["effective_rho"].extend([effective_rho] * tasks)
                bundle["context_size"].extend([n_context] * tasks)
                bundle["task_type"].extend([task_type] * tasks)
                print(
                    f"cell task={task_type} n={n_context} rho={rho:.2f} "
                    f"acc={accuracy:.3f} MI={records[-1]['empirical_mi_c_w_nats']:.3f} "
                    f"rank-raw={advantage:+.4f} marginal-utility={marginal_utility:+.4f}",
                    flush=True,
                )

    frame = pd.DataFrame(records)
    monotonic_checks = []
    for (task_type, context), cell in frame.groupby(["task_type", "context_size"]):
        ordered = cell.sort_values("rho")
        corr = float(spearmanr(ordered["rho"], ordered["mechanism_accuracy"]).statistic)
        delta = float(ordered["mechanism_accuracy"].iloc[-1] - ordered["mechanism_accuracy"].iloc[0])
        monotonic_checks.append({"task_type": task_type, "context_size": int(context), "spearman": corr, "endpoint_gain": delta})
    gate_pass = bool(
        all(x["spearman"] >= 0.75 and x["endpoint_gain"] >= 0.25 for x in monotonic_checks)
    )
    utility_checks = []
    for (task_type, context), cell in frame.groupby(["task_type", "context_size"]):
        ordered = cell.sort_values("rho")
        low_rho = ordered.iloc[0]
        high_rho = ordered.iloc[-1]
        utility_checks.append({
            "task_type": task_type,
            "context_size": int(context),
            "rho0_utility": float(low_rho["marginal_query_utility"]),
            "rho1_utility": float(high_rho["marginal_query_utility"]),
            "rho1_ci_low": float(high_rho["marginal_query_utility_ci_low"]),
            "passes": bool(
                high_rho["marginal_query_utility_ci_low"] > 0
                and high_rho["marginal_query_utility"] > low_rho["marginal_query_utility"]
            ),
        })
    predictive_gate_pass = bool(all(item["passes"] for item in utility_checks))

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("xb") as handle:
        np.savez_compressed(handle, **{key: np.asarray(value) for key, value in bundle.items()})
    frame.to_csv(processed_path, index=False, mode="x")
    elapsed = time.perf_counter() - started
    metadata = {
        "run_key": run_key,
        "experiment": "E1",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256_file(config_path),
        "git_commit": git_commit(ROOT),
        "package_versions": package_versions(),
        "tasks_per_cell": tasks,
        "context_sizes": contexts,
        "feature_count": n_features,
        "query_size": n_query,
        "wall_clock_seconds": elapsed,
        "primary_shape_information_gate_pass": gate_pass,
        "predictive_utility_gate_pass": predictive_gate_pass,
        "monotonic_checks": monotonic_checks,
        "utility_checks": utility_checks,
        "raw_bundle": str(raw_path.relative_to(ROOT)),
        "processed_summary": str(processed_path.relative_to(ROOT)),
        "note": "Gate concerns label-free mechanism information only; predictive crossover is separately reported and not inferred from this gate.",
    }
    write_json_exclusive(metadata_path, metadata)
    append_manifest(ROOT / "results/MANIFEST.jsonl", metadata)
    print(json.dumps({
        "run_key": run_key,
        "shape_information_gate_pass": gate_pass,
        "predictive_utility_gate_pass": predictive_gate_pass,
        "wall_clock_seconds": elapsed,
    }, indent=2))


if __name__ == "__main__":
    main()
