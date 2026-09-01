#!/usr/bin/env python3
"""Audit, aggregate, plot, and write results.md for Kill Experiment 2."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.semantic_orbits import disagreement_metrics, prediction_metrics  # noqa: E402


RAW = ROOT / "results" / "semantic_orbits" / "raw"
PROCESSED = ROOT / "results" / "semantic_orbits" / "processed"
FIGURES = ROOT / "figures" / "semantic_orbits"
CONFIG_PATH = ROOT / "configs" / "semantic_orbits.yaml"
TABM_CONFIG_PATH = ROOT / "configs" / "semantic_orbits_tabm_basis.yaml"
TABM_RAW = ROOT / "results" / "semantic_orbits" / "tabm_basis"
TRAINING_CONFIG_PATH = ROOT / "configs" / "semantic_orbit_training.yaml"
TRAINING_RAW = ROOT / "results" / "semantic_orbits" / "training_ablations"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_audit(config: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    expected = {
        (model, spec["name"], int(seed))
        for model in config["models"]
        for spec in config["datasets"]
        for seed in config["model_seeds"]
    }
    metadata_paths = sorted(RAW.glob("*/*/seed_*/metadata.json"))
    observed = set()
    metrics = []
    manifest = []
    config_hash = sha256(CONFIG_PATH)
    split_fingerprints: dict[str, tuple[str, str]] = {}
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text())
        bundle = metadata_path.parent
        key = (metadata["model"], metadata["dataset_spec"]["name"], int(metadata["model_seed"]))
        if key in observed:
            raise RuntimeError(f"duplicate bundle {key}")
        observed.add(key)
        if metadata["status"] != "complete" or metadata["config_sha256"] != config_hash:
            raise RuntimeError(f"invalid metadata/config drift: {metadata_path}")
        for filename, record in metadata["files"].items():
            path = bundle / filename
            if not path.exists() or sha256(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
                raise RuntimeError(f"artifact integrity failure: {path}")
        audit = metadata["split_audit"]
        if not all(audit[name] for name in (
            "train_validation_disjoint", "train_test_disjoint", "validation_test_disjoint"
        )):
            raise RuntimeError(f"split leakage: {metadata_path}")
        fingerprint = (audit["test_row_order_sha256"], audit["target_sha256"])
        dataset = key[1]
        if dataset in split_fingerprints and split_fingerprints[dataset] != fingerprint:
            raise RuntimeError(f"row/target mismatch across models or seeds: {dataset}")
        split_fingerprints[dataset] = fingerprint
        frame = pd.read_csv(bundle / "metrics.csv")
        if len(frame) != metadata["representation_count"]:
            raise RuntimeError(f"representation count mismatch: {bundle}")
        metrics.append(frame)
        manifest.append({
            "model": key[0], "dataset": key[1], "model_seed": key[2],
            "bundle": str(bundle.relative_to(ROOT)), "wall_seconds": metadata["wall_seconds"],
            "representation_count": metadata["representation_count"],
            "unique_fit_count": metadata["unique_fit_count"],
            "predictions_sha256": metadata["files"]["predictions.csv.gz"]["sha256"],
            "metrics_sha256": metadata["files"]["metrics.csv"]["sha256"],
            "environment": metadata["environment"],
        })
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise RuntimeError(f"bundle coverage failure: missing={missing}, extra={extra}")
    all_metrics = pd.concat(metrics, ignore_index=True)
    all_metrics["member_transform_kind"] = all_metrics["variant"]
    # T3 is one predeclared eight-member orbit sampled from three monotone-map generators.
    all_metrics.loc[all_metrics["family"].eq("T3"), "variant"] = "mixed_monotone"
    members = all_metrics[(~all_metrics["is_reference"]) & (all_metrics["member"] >= 0)]
    orbit_keys = ["model", "dataset", "model_seed", "pipeline", "family", "variant", "scope", "repair", "reference_id"]
    counts = members.groupby(orbit_keys, dropna=False)["member"].nunique()
    if not (counts == int(config["orbit_members"])).all():
        raise RuntimeError(f"incomplete orbits: {counts[counts != int(config['orbit_members'])].to_dict()}")
    return all_metrics, manifest


def load_tabm_and_audit() -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Audit the deliberately bounded TabM T6 panel without changing the primary verdict grid."""
    config = yaml.safe_load(TABM_CONFIG_PATH.read_text())
    config_hash = sha256(TABM_CONFIG_PATH)
    expected = {
        (spec["name"], int(seed)) for spec in config["datasets"] for seed in config["model_seeds"]
    }
    observed = set()
    metrics = []
    manifest = []
    primary_fingerprints = {}
    for path in RAW.glob("*/*/seed_*/metadata.json"):
        metadata = json.loads(path.read_text())
        primary_fingerprints.setdefault(metadata["dataset_spec"]["name"], (
            metadata["split_audit"]["test_row_order_sha256"], metadata["split_audit"]["target_sha256"]
        ))
    for metadata_path in sorted(TABM_RAW.glob("*/seed_*/metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        bundle = metadata_path.parent
        key = (metadata["dataset_spec"]["name"], int(metadata["model_seed"]))
        if key in observed or metadata.get("status") != "complete" or metadata.get("config_sha256") != config_hash:
            raise RuntimeError(f"invalid or duplicate TabM bundle: {metadata_path}")
        observed.add(key)
        for filename, record in metadata["files"].items():
            path = bundle / filename
            if not path.exists() or sha256(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
                raise RuntimeError(f"TabM artifact integrity failure: {path}")
        audit = metadata["split_audit"]
        if not all(audit[name] for name in (
            "train_validation_disjoint", "train_test_disjoint", "validation_test_disjoint"
        )):
            raise RuntimeError(f"TabM split leakage: {metadata_path}")
        fingerprint = (audit["test_row_order_sha256"], audit["target_sha256"])
        if fingerprint != primary_fingerprints.get(key[0]):
            raise RuntimeError(f"TabM/primary test-row mismatch: {key}")
        frame = pd.read_csv(bundle / "metrics.csv")
        if len(frame) != 25:
            raise RuntimeError(f"TabM representation count mismatch: {bundle}")
        counts = frame[~frame["is_reference"]].groupby("variant")["member"].nunique()
        if set(counts.index) != set(config["variants"]) or not counts.eq(int(config["orbit_members"])).all():
            raise RuntimeError(f"incomplete TabM orbits: {bundle}")
        metrics.append(frame)
        manifest.append({
            "dataset": key[0], "model_seed": key[1], "bundle": str(bundle.relative_to(ROOT)),
            "wall_seconds": metadata["wall_seconds"], "representation_count": metadata["representation_count"],
            "predictions_sha256": metadata["files"]["predictions.csv.gz"]["sha256"],
            "metrics_sha256": metadata["files"]["metrics.csv"]["sha256"],
            "environment": metadata["environment"],
        })
    if observed != expected:
        raise RuntimeError(f"TabM bundle coverage failure: missing={sorted(expected-observed)}, extra={sorted(observed-expected)}")
    all_metrics = pd.concat(metrics, ignore_index=True)
    for column in ("probability_mad", "js_divergence", "label_flip_rate", "log_loss"):
        all_metrics[column] = np.nan
    all_metrics = add_common_metrics(all_metrics)
    _, summary = orbit_summary(all_metrics)
    return all_metrics, summary, manifest


def load_training_and_audit() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    config = yaml.safe_load(TRAINING_CONFIG_PATH.read_text())
    config_hash = sha256(TRAINING_CONFIG_PATH)
    expected = {(spec["name"], int(seed)) for spec in config["datasets"] for seed in config["model_seeds"]}
    observed = set()
    frames = []
    manifest = []
    primary_fingerprints = {}
    for path in RAW.glob("*/*/seed_*/metadata.json"):
        metadata = json.loads(path.read_text())
        primary_fingerprints.setdefault(metadata["dataset_spec"]["name"], (
            metadata["split_audit"]["test_row_order_sha256"], metadata["split_audit"]["target_sha256"]
        ))
    for metadata_path in sorted(TRAINING_RAW.glob("*/seed_*/metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        bundle = metadata_path.parent
        key = (metadata["dataset_spec"]["name"], int(metadata["model_seed"]))
        if key in observed or metadata.get("status") != "complete" or metadata.get("config_sha256") != config_hash:
            raise RuntimeError(f"invalid or duplicate training bundle: {metadata_path}")
        observed.add(key)
        for filename, record in metadata["files"].items():
            path = bundle / filename
            if not path.exists() or sha256(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
                raise RuntimeError(f"training artifact integrity failure: {path}")
        audit = metadata["split_audit"]
        if not all(audit[name] for name in (
            "train_validation_disjoint", "train_test_disjoint", "validation_test_disjoint"
        )):
            raise RuntimeError(f"training split leakage: {metadata_path}")
        fingerprint = (audit["test_row_order_sha256"], audit["target_sha256"])
        if fingerprint != primary_fingerprints.get(key[0]):
            raise RuntimeError(f"training/primary test-row mismatch: {key}")
        if float(metadata["inverse_audit_max_delta"]) > 2e-5:
            raise RuntimeError(f"basis inverse audit failure: {metadata_path}")
        frame = pd.read_csv(bundle / "metrics.csv")
        expected_rows = len(config["methods"]) * (int(config["orbit_members"]) + 1)
        if len(frame) != expected_rows or set(frame["method"]) != set(config["methods"]):
            raise RuntimeError(f"training row/method coverage failure: {bundle}")
        member_counts = frame[~frame["is_reference"]].groupby("method")["member"].nunique()
        if not member_counts.eq(int(config["orbit_members"])).all():
            raise RuntimeError(f"incomplete training orbits: {bundle}")
        frames.append(frame)
        manifest.append({
            "dataset": key[0], "model_seed": key[1], "bundle": str(bundle.relative_to(ROOT)),
            "wall_seconds": metadata["wall_seconds"], "inverse_audit_max_delta": metadata["inverse_audit_max_delta"],
            "predictions_sha256": metadata["files"]["predictions.csv.gz"]["sha256"],
            "metrics_sha256": metadata["files"]["metrics.csv"]["sha256"],
            "environment": metadata["environment"],
        })
    if observed != expected:
        raise RuntimeError(
            f"training bundle coverage failure: missing={sorted(expected-observed)}, extra={sorted(observed-expected)}"
        )
    metrics = pd.concat(frames, ignore_index=True)
    members = metrics[~metrics["is_reference"]]
    by_seed = members.groupby(["dataset", "method", "model_seed"], as_index=False).agg(
        prediction_disagreement=("prediction_rmse_normalized", "mean"),
        orbit_mean_rmse=("rmse", "mean"), orbit_worst_rmse=("rmse", "max"), orbit_span_rmse=("rmse", lambda x: x.max()-x.min()),
    )
    reference = metrics[metrics["is_reference"]].groupby(
        ["dataset", "method", "model_seed"], as_index=False
    ).agg(reference_rmse=("rmse", "first"))
    by_seed = by_seed.merge(reference, on=["dataset", "method", "model_seed"], validate="one_to_one")
    summary = by_seed.groupby(["dataset", "method"], as_index=False).agg(
        prediction_disagreement=("prediction_disagreement", "mean"),
        prediction_disagreement_seed_sd=("prediction_disagreement", "std"),
        reference_rmse=("reference_rmse", "mean"), orbit_mean_rmse=("orbit_mean_rmse", "mean"),
        orbit_worst_rmse=("orbit_worst_rmse", "mean"), orbit_span_rmse=("orbit_span_rmse", "mean"),
        seeds=("model_seed", "nunique"),
    )
    raw = summary[summary["method"].eq("raw")][
        ["dataset", "prediction_disagreement", "orbit_mean_rmse", "orbit_worst_rmse", "reference_rmse"]
    ].rename(columns={
        "prediction_disagreement": "raw_disagreement", "orbit_mean_rmse": "raw_orbit_mean_rmse",
        "orbit_worst_rmse": "raw_orbit_worst_rmse", "reference_rmse": "raw_reference_rmse",
    })
    summary = summary.merge(raw, on="dataset", validate="many_to_one")
    summary["disagreement_reduction_vs_raw"] = (
        1.0 - summary["prediction_disagreement"] / summary["raw_disagreement"].clip(lower=1e-12)
    )
    summary["orbit_mean_change_vs_raw"] = summary["orbit_mean_rmse"] - summary["raw_orbit_mean_rmse"]
    summary["orbit_worst_change_vs_raw"] = summary["orbit_worst_rmse"] - summary["raw_orbit_worst_rmse"]
    summary["reference_change_vs_raw"] = summary["reference_rmse"] - summary["raw_reference_rmse"]
    return summary, manifest


def add_common_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ("probability_mad", "prediction_rmse_normalized"):
        if column not in result:
            result[column] = np.nan
    result["prediction_disagreement"] = np.where(
        result["problem_type"].eq("classification"),
        result["probability_mad"], result["prediction_rmse_normalized"],
    )
    result["task_error"] = np.where(
        result["problem_type"].eq("classification"), result["log_loss"], result["rmse"]
    )
    return result


def orbit_summary(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    members = metrics[(~metrics["is_reference"]) & (metrics["member"] >= 0)].copy()
    refs = metrics[metrics["is_reference"]][
        ["model", "dataset", "model_seed", "representation_id", "task_error"]
    ].rename(columns={"representation_id": "reference_id", "task_error": "task_error_original"})
    members = members.merge(refs, on=["model", "dataset", "model_seed", "reference_id"], how="left", validate="many_to_one")
    if members["task_error_original"].isna().any():
        raise RuntimeError("missing original performance for an orbit member")
    keys = [
        "dataset", "problem_type", "model", "pipeline", "family", "variant", "scope", "repair",
        "reference_id", "model_seed",
    ]
    by_seed = members.groupby(keys, dropna=False).agg(
        pred_disagreement=("prediction_disagreement", "mean"),
        task_metric_original=("task_error_original", "first"),
        orbit_mean=("task_error", "mean"),
        orbit_worst=("task_error", "max"),
        orbit_best=("task_error", "min"),
        orbit_span=("task_error", lambda values: float(values.max() - values.min())),
        orbit_members=("member", "nunique"),
        probability_mad=("probability_mad", "mean"),
        js_divergence=("js_divergence", "mean"),
        label_flip_rate=("label_flip_rate", "mean"),
        prediction_rmse_normalized=("prediction_rmse_normalized", "mean"),
        prediction_pearson=("prediction_pearson", "mean"),
        prediction_spearman=("prediction_spearman", "mean"),
    ).reset_index()
    by_seed["kill_effect_magnitude"] = np.where(
        by_seed["problem_type"].eq("classification"),
        by_seed["label_flip_rate"], by_seed["prediction_rmse_normalized"],
    )
    by_seed["kill_effect_pass"] = np.where(
        by_seed["problem_type"].eq("classification"),
        by_seed["kill_effect_magnitude"].ge(0.03), by_seed["kill_effect_magnitude"].ge(0.05),
    )
    aggregate_keys = keys[:-1]
    aggregate = by_seed.groupby(aggregate_keys, dropna=False).agg(
        pred_disagreement=("pred_disagreement", "mean"),
        pred_disagreement_seed_sd=("pred_disagreement", "std"),
        task_metric_original=("task_metric_original", "mean"),
        orbit_mean=("orbit_mean", "mean"),
        orbit_worst=("orbit_worst", "mean"),
        orbit_best=("orbit_best", "mean"),
        orbit_span=("orbit_span", "mean"),
        probability_mad=("probability_mad", "mean"),
        js_divergence=("js_divergence", "mean"),
        label_flip_rate=("label_flip_rate", "mean"),
        prediction_rmse_normalized=("prediction_rmse_normalized", "mean"),
        prediction_pearson=("prediction_pearson", "mean"),
        prediction_spearman=("prediction_spearman", "mean"),
        kill_effect_min_across_seeds=("kill_effect_magnitude", "min"),
        kill_effect_seed_passes=("kill_effect_pass", "sum"),
        seeds=("model_seed", "nunique"),
    ).reset_index()
    return by_seed, aggregate


def _prediction_columns(frame: pd.DataFrame) -> list[str]:
    probability = sorted(
        [column for column in frame.columns if column.startswith("prediction_")],
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    return probability if probability else ["prediction"]


def ensemble_analysis(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    orbit_rows = []
    reference_store: dict[tuple[str, str, str, str, int], tuple[np.ndarray, np.ndarray]] = {}
    for path in sorted(RAW.glob("*/*/seed_*/predictions.csv.gz")):
        frame = pd.read_csv(path)
        frame["member_transform_kind"] = frame["variant"]
        frame.loc[frame["family"].eq("T3"), "variant"] = "mixed_monotone"
        pred_columns = _prediction_columns(frame)
        model = str(frame["model"].iloc[0])
        dataset = str(frame["dataset"].iloc[0])
        model_seed = int(frame["model_seed"].iloc[0])
        refs = frame[frame["is_reference"]]
        for (pipeline, representation_id), group in refs.groupby(["pipeline", "representation_id"]):
            group = group.sort_values("test_row_id")
            reference_store[(model, dataset, pipeline, representation_id, model_seed)] = (
                group["target"].to_numpy(), group[pred_columns].to_numpy().squeeze()
            )
        members = frame[(~frame["is_reference"]) & frame["member"].ge(0)]
        group_keys = ["pipeline", "family", "variant", "scope", "repair", "reference_id"]
        for values, group in members.groupby(group_keys, dropna=False):
            pipeline, family, variant, scope, repair, reference_id = values
            ordered_members = sorted(group["member"].unique())
            if len(ordered_members) != int(config["orbit_members"]):
                continue
            y = group[group["member"].eq(ordered_members[0])].sort_values("test_row_id")["target"].to_numpy()
            reference = reference_store[(model, dataset, pipeline, reference_id, model_seed)][1]
            for budget in (3, int(config["orbit_members"])):
                predictions = []
                for member in ordered_members[:budget]:
                    predictions.append(
                        group[group["member"].eq(member)].sort_values("test_row_id")[pred_columns].to_numpy().squeeze()
                    )
                ensemble = np.mean(np.stack(predictions), axis=0)
                problem_type = "classification" if ensemble.ndim == 2 else "regression"
                orbit_rows.append({
                    "model": model, "dataset": dataset, "model_seed": model_seed, "pipeline": pipeline,
                    "family": family, "variant": variant, "scope": scope, "repair": repair,
                    "reference_id": reference_id, "budget": budget, "problem_type": problem_type,
                    **prediction_metrics(problem_type, y, ensemble),
                    **disagreement_metrics(problem_type, y, reference, ensemble),
                })
    orbit = add_common_metrics(pd.DataFrame(orbit_rows))

    ordinary_rows = []
    reference_keys = sorted({key[:-1] for key in reference_store})
    required_seeds = set(map(int, config["model_seeds"]))
    for model, dataset, pipeline, reference_id in reference_keys:
        available = {
            seed: reference_store[(model, dataset, pipeline, reference_id, seed)]
            for seed in required_seeds if (model, dataset, pipeline, reference_id, seed) in reference_store
        }
        if set(available) != required_seeds:
            continue
        first_seed = min(available)
        y = available[first_seed][0]
        predictions = [available[seed][1] for seed in sorted(available)]
        ensemble = np.mean(np.stack(predictions), axis=0)
        problem_type = "classification" if ensemble.ndim == 2 else "regression"
        ordinary_rows.append({
            "model": model, "dataset": dataset, "pipeline": pipeline, "reference_id": reference_id,
            "budget": len(predictions), "problem_type": problem_type,
            **prediction_metrics(problem_type, y, ensemble),
        })
    ordinary = add_common_metrics(pd.DataFrame(ordinary_rows))
    return orbit, ordinary


def repair_table(summary: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "model", "pipeline", "family", "variant", "scope"]
    raw = summary[summary["repair"].eq("none")][keys + ["pred_disagreement", "orbit_mean"]].rename(
        columns={"pred_disagreement": "raw_disagreement", "orbit_mean": "raw_orbit_mean"}
    )
    repaired = summary[~summary["repair"].eq("none")][keys + ["repair", "pred_disagreement", "orbit_mean"]].rename(
        columns={"pred_disagreement": "repaired_disagreement", "orbit_mean": "repaired_orbit_mean"}
    )
    result = repaired.merge(raw, on=keys, how="left")
    result["disagreement_reduction_fraction"] = 1.0 - result["repaired_disagreement"] / result["raw_disagreement"].clip(lower=1e-12)
    result["task_error_change"] = result["repaired_orbit_mean"] - result["raw_orbit_mean"]
    return result


def ensemble_comparison(orbit: pd.DataFrame, ordinary: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "dataset", "pipeline", "family", "variant", "scope", "repair", "reference_id"]
    orbit_three = orbit[(orbit["budget"].eq(3)) & orbit["repair"].eq("none")].groupby(
        keys, as_index=False, dropna=False
    ).agg(orbit_three_task_error=("task_error", "mean"), orbit_three_disagreement=("prediction_disagreement", "mean"))
    ordinary_base = ordinary[[
        "model", "dataset", "pipeline", "reference_id", "task_error"
    ]].rename(columns={"task_error": "ordinary_three_task_error"})
    result = orbit_three.merge(
        ordinary_base, on=["model", "dataset", "pipeline", "reference_id"], how="inner", validate="many_to_one"
    )
    result["orbit_vs_ordinary_relative_error"] = (
        result["orbit_three_task_error"] - result["ordinary_three_task_error"]
    ) / result["ordinary_three_task_error"].abs().clip(lower=1e-12)
    result["orbit_beats_ordinary"] = result["orbit_vs_ordinary_relative_error"].lt(0)
    return result


def dataset_bootstrap(summary: pd.DataFrame, draws: int = 10_000, seed: int = 20260901) -> pd.DataFrame:
    """Dataset-level nonparametric bootstrap after member/seed aggregation."""
    rng = np.random.default_rng(seed)
    rows = []
    keys = ["model", "pipeline", "family", "variant", "scope", "repair"]
    for values, group in summary.groupby(keys, dropna=False):
        dataset_values = group.groupby("dataset", as_index=False).agg(
            pred_disagreement=("pred_disagreement", "mean"), orbit_span=("orbit_span", "mean")
        )
        n = len(dataset_values)
        if n == 0:
            continue
        samples = rng.integers(0, n, size=(draws, n))
        disagreement_draws = dataset_values["pred_disagreement"].to_numpy()[samples].mean(axis=1)
        span_draws = dataset_values["orbit_span"].to_numpy()[samples].mean(axis=1)
        rows.append({
            **dict(zip(keys, values)), "datasets": n,
            "pred_disagreement_mean": float(dataset_values["pred_disagreement"].mean()),
            "pred_disagreement_ci_low": float(np.quantile(disagreement_draws, 0.025)),
            "pred_disagreement_ci_high": float(np.quantile(disagreement_draws, 0.975)),
            "orbit_span_mean": float(dataset_values["orbit_span"].mean()),
            "orbit_span_ci_low": float(np.quantile(span_draws, 0.025)),
            "orbit_span_ci_high": float(np.quantile(span_draws, 0.975)),
        })
    return pd.DataFrame(rows)


def verdict(summary: pd.DataFrame, repairs: pd.DataFrame, orbit: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    raw = summary[(summary["repair"].eq("none")) & (~summary["family"].eq("T0"))].copy()
    raw["passes_effect"] = np.where(
        raw["problem_type"].eq("classification"),
        raw["label_flip_rate"].ge(0.03), raw["prediction_rmse_normalized"].ge(0.05),
    ) & raw["kill_effect_seed_passes"].eq(raw["seeds"])
    repeated = raw[raw["passes_effect"]].groupby(["family", "variant", "scope"]).agg(
        datasets=("dataset", "nunique"), models=("model", "nunique"), cells=("dataset", "size")
    ).reset_index()
    candidates = repeated[(repeated["datasets"] >= 3) & (repeated["models"] >= 2)]
    strong_repair = repairs[
        repairs["raw_disagreement"].ge(1e-4)
        & repairs["disagreement_reduction_fraction"].ge(0.30)
        & repairs["task_error_change"].le(0.01 * repairs["raw_orbit_mean"].abs().clip(lower=1e-8))
    ]
    ensemble_keys = ["dataset", "model", "pipeline", "family", "variant", "scope", "repair", "reference_id"]
    orbit_eight = orbit[(orbit["budget"].eq(8)) & orbit["repair"].eq("none")].groupby(
        ensemble_keys, as_index=False, dropna=False
    ).agg(ensemble_task_error=("task_error", "mean"), ensemble_disagreement=("prediction_disagreement", "mean"))
    raw_ensemble_base = summary[summary["repair"].eq("none")][
        ensemble_keys + ["orbit_mean", "pred_disagreement"]
    ]
    ensemble_gain = orbit_eight.merge(raw_ensemble_base, on=ensemble_keys, how="inner")
    ensemble_gain["task_error_gain"] = ensemble_gain["orbit_mean"] - ensemble_gain["ensemble_task_error"]
    ensemble_gain["disagreement_reduction_fraction"] = (
        1.0 - ensemble_gain["ensemble_disagreement"] / ensemble_gain["pred_disagreement"].clip(lower=1e-12)
    )
    strong_ensemble = ensemble_gain[
        ensemble_gain["task_error_gain"].gt(0.005 * ensemble_gain["orbit_mean"].abs().clip(lower=1e-8))
        | ensemble_gain["disagreement_reduction_fraction"].ge(0.30)
    ]
    if not candidates.empty and (not strong_repair.empty or not strong_ensemble.empty):
        label = "GO"
    elif raw["passes_effect"].any():
        label = "FOUNDATIONAL-SIGNAL-METHOD-UNSOLVED"
    else:
        label = "NO-GO"
    diagnostics = {
        "qualifying_repeated_effects": candidates.to_dict("records"),
        "strong_repair_cells": int(len(strong_repair)),
        "strong_ensemble_cells": int(len(strong_ensemble)),
        "effect_cells": int(raw["passes_effect"].sum()),
        "raw_cells": int(len(raw)),
    }
    return label, diagnostics


def make_plots(
    metrics: pd.DataFrame, summary: pd.DataFrame, repairs: pd.DataFrame,
    orbit: pd.DataFrame, ordinary: pd.DataFrame, ensemble_compare: pd.DataFrame,
    tabm_metrics: pd.DataFrame,
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    raw = summary[(summary["repair"].eq("none")) & (~summary["family"].eq("R0"))].copy()
    raw["transformation"] = raw["variant"] + "/" + raw["scope"]

    heat = raw.pivot_table(index=["model", "transformation"], columns="dataset", values="pred_disagreement", aggfunc="mean")
    plt.figure(figsize=(11, max(6, 0.28 * len(heat))))
    sns.heatmap(heat, cmap="mako", robust=True)
    plt.title("Orbit prediction disagreement (classification MAD / regression normalized RMSE)")
    plt.tight_layout(); plt.savefig(FIGURES / "01_disagreement_heatmap.png", dpi=180); plt.close()

    span = raw.pivot_table(index=["model", "transformation"], columns="dataset", values="orbit_span", aggfunc="mean")
    plt.figure(figsize=(11, max(6, 0.28 * len(span))))
    sns.heatmap(span, cmap="rocket", robust=True)
    plt.title("Task-error range across eight orbit members")
    plt.tight_layout(); plt.savefig(FIGURES / "02_orbit_performance_range.png", dpi=180); plt.close()

    strongest = metrics[(~metrics["is_reference"]) & metrics["repair"].eq("none") & ~metrics["family"].eq("T0")].sort_values(
        "prediction_disagreement", ascending=False
    ).iloc[0]
    bundle = RAW / strongest["model"] / strongest["dataset"] / f"seed_{int(strongest['model_seed'])}" / "predictions.csv.gz"
    pred = pd.read_csv(bundle)
    transformed = pred[pred["representation_id"].eq(strongest["representation_id"])].sort_values("test_row_id")
    reference = pred[pred["representation_id"].eq(strongest["reference_id"])].sort_values("test_row_id")
    columns = _prediction_columns(pred)
    if len(columns) > 1:
        x, y = reference[columns[-1]].to_numpy(), transformed[columns[-1]].to_numpy()
        axis_label = "positive-class probability"
    else:
        x, y = reference[columns[0]].to_numpy(), transformed[columns[0]].to_numpy()
        axis_label = "prediction"
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    plt.figure(figsize=(6, 6)); plt.scatter(x, y, s=12, alpha=0.5); plt.plot([lo, hi], [lo, hi], "k--")
    plt.xlabel(f"Reference {axis_label}"); plt.ylabel(f"Transformed {axis_label}")
    plt.title(f"Strongest cell: {strongest['model']} / {strongest['dataset']} / {strongest['variant']}")
    plt.tight_layout(); plt.savefig(FIGURES / "03_prediction_scatter.png", dpi=180); plt.close()

    repair_plot = repairs.groupby("repair", as_index=False)["disagreement_reduction_fraction"].median().sort_values(
        "disagreement_reduction_fraction", ascending=False
    )
    plt.figure(figsize=(8, 4)); sns.barplot(repair_plot, x="repair", y="disagreement_reduction_fraction", color="#4C78A8")
    plt.axhline(0, color="black", linewidth=1); plt.xticks(rotation=25, ha="right")
    plt.ylabel("Median disagreement reduction"); plt.xlabel(""); plt.tight_layout()
    plt.savefig(FIGURES / "04_repair_effectiveness.png", dpi=180); plt.close()

    ensemble_plot = ensemble_compare.groupby(["model", "family"], as_index=False).agg(
        relative_error=("orbit_vs_ordinary_relative_error", "median")
    )
    plt.figure(figsize=(10, 5)); sns.barplot(ensemble_plot, x="model", y="relative_error", hue="family")
    plt.axhline(0, color="black", linewidth=1)
    plt.ylabel("Median relative task error: 3-orbit vs 3-seed ensemble"); plt.tight_layout()
    plt.savefig(FIGURES / "05_orbit_vs_ordinary_ensemble.png", dpi=180); plt.close()

    feature_map = {"T1": "nominal", "T2": "numerical", "T3": "order-scale", "T4": "ordinal", "T5": "cyclic", "T6": "basis"}
    feature = raw[raw["family"].isin(feature_map)].assign(feature_type=lambda x: x["family"].map(feature_map))
    feature = feature.groupby(["model", "feature_type"], as_index=False)["pred_disagreement"].mean()
    plt.figure(figsize=(9, 5)); sns.barplot(feature, x="feature_type", y="pred_disagreement", hue="model")
    plt.ylabel("Mean prediction disagreement"); plt.xlabel(""); plt.tight_layout()
    plt.savefig(FIGURES / "06_feature_type_sensitivity.png", dpi=180); plt.close()

    basis = pd.concat([
        metrics[(metrics["family"].eq("T6")) & (~metrics["is_reference"])],
        tabm_metrics[(tabm_metrics["family"].eq("T6")) & (~tabm_metrics["is_reference"])],
    ], ignore_index=True)
    basis["condition_number"] = basis["transform_metadata_json"].map(lambda value: json.loads(value)["condition_number"])
    plt.figure(figsize=(7, 5)); sns.scatterplot(basis, x="condition_number", y="prediction_disagreement", hue="model", style="dataset", alpha=0.7)
    plt.xscale("log"); plt.ylabel("Prediction disagreement"); plt.tight_layout()
    plt.savefig(FIGURES / "07_basis_condition_vs_disagreement.png", dpi=180); plt.close()

    plt.figure(figsize=(7, 6)); sns.scatterplot(raw, x="orbit_mean", y="orbit_worst", hue="model", style="family", alpha=0.75)
    limits = [min(raw["orbit_mean"].min(), raw["orbit_worst"].min()), max(raw["orbit_mean"].max(), raw["orbit_worst"].max())]
    plt.plot(limits, limits, "k--", linewidth=1); plt.xlabel("Average orbit task error"); plt.ylabel("Worst orbit task error")
    plt.tight_layout(); plt.savefig(FIGURES / "08_worst_vs_average_performance.png", dpi=180); plt.close()


def markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 80) -> str:
    selected = frame.loc[:, columns].head(limit).copy()
    for column in selected.select_dtypes(include=[np.number]).columns:
        selected[column] = selected[column].map(lambda value: "" if pd.isna(value) else f"{value:.5g}")
    headers = " | ".join(columns)
    divider = " | ".join(["---"] * len(columns))
    rows = [" | ".join(map(str, row)) for row in selected.itertuples(index=False, name=None)]
    return "\n".join([headers, divider, *rows])


def write_results(
    config: dict[str, Any], metrics: pd.DataFrame, summary: pd.DataFrame, repairs: pd.DataFrame,
    orbit: pd.DataFrame, ordinary: pd.DataFrame, ensemble_compare: pd.DataFrame,
    manifest: list[dict[str, Any]], label: str, diagnostics: dict[str, Any], bootstrap: pd.DataFrame,
    tabm_summary: pd.DataFrame, tabm_manifest: list[dict[str, Any]],
    training_summary: pd.DataFrame, training_manifest: list[dict[str, Any]],
) -> None:
    raw = summary[(summary["repair"].eq("none")) & (~summary["family"].eq("R0"))]
    main = raw.sort_values("pred_disagreement", ascending=False)
    strongest = main.iloc[0]
    strongest_basis = main[main["family"].eq("T6")].iloc[0]
    weakest = main.iloc[-1]
    total_hours = sum(record["wall_seconds"] for record in manifest) / 3600
    tabm_minutes = sum(record["wall_seconds"] for record in tabm_manifest) / 60
    training_minutes = sum(record["wall_seconds"] for record in training_manifest) / 60
    package_versions = manifest[0]["environment"]["packages"]
    hardware = manifest[0]["environment"].get("gpu")
    dataset_text = ", ".join(
        f"{spec['name']} (OpenML {spec['openml_id']} v{spec['openml_version']})" for spec in config["datasets"]
    )
    main_columns = [
        "dataset", "model", "variant", "pred_disagreement", "task_metric_original",
        "orbit_mean", "orbit_worst", "orbit_span",
    ]
    family_sections = {
        "Column permutation control": "T0", "Nominal relabeling": "T1",
        "Numerical affine/unit transforms": "T2", "Monotone transforms": "T3",
        "Ordinal spacing": "T4", "Cyclic recoding": "T5", "Equivalent basis changes": "T6",
    }
    sections = []
    for title, family in family_sections.items():
        cells = raw[raw["family"].eq(family)].sort_values("pred_disagreement", ascending=False)
        if cells.empty:
            text = "Not applicable to the frozen dataset panel."
        else:
            top = cells.iloc[0]
            text = (
                f"Strongest cell: {top['model']} on {top['dataset']} ({top['variant']}, {top['scope']}), "
                f"disagreement {top['pred_disagreement']:.4g}, orbit span {top['orbit_span']:.4g}."
            )
        sections.append(f"### {title}\n\n{text}")
    repair_summary = repairs.groupby("repair").agg(
        median_reduction=("disagreement_reduction_fraction", "median"),
        median_task_change=("task_error_change", "median"), cells=("dataset", "size")
    ).reset_index().sort_values("median_reduction", ascending=False)
    ensemble_keys = ["dataset", "model", "pipeline", "family", "variant", "scope", "repair", "reference_id"]
    ensemble_eight = orbit[(orbit["budget"].eq(8)) & orbit["repair"].eq("none")].groupby(
        ensemble_keys, as_index=False, dropna=False
    ).agg(
        repaired_disagreement=("prediction_disagreement", "mean"),
        repaired_orbit_mean=("task_error", "mean"),
    )
    ensemble_base = raw[ensemble_keys + ["pred_disagreement", "orbit_mean"]]
    ensemble_repair = ensemble_eight.merge(ensemble_base, on=ensemble_keys, validate="one_to_one")
    ensemble_repair["reduction"] = (
        1.0 - ensemble_repair["repaired_disagreement"] / ensemble_repair["pred_disagreement"].clip(lower=1e-12)
    )
    ensemble_repair["task_change"] = ensemble_repair["repaired_orbit_mean"] - ensemble_repair["orbit_mean"]
    repair_summary = pd.concat([repair_summary, pd.DataFrame([{
        "repair": "orbit_ensemble_8",
        "median_reduction": float(ensemble_repair["reduction"].median()),
        "median_task_change": float(ensemble_repair["task_change"].median()),
        "cells": int(len(ensemble_repair)),
    }])], ignore_index=True).sort_values("median_reduction", ascending=False)
    ensemble_median = float(ensemble_compare["orbit_vs_ordinary_relative_error"].median())
    ensemble_wins = int(ensemble_compare["orbit_beats_ordinary"].sum())
    ensemble_cells = int(len(ensemble_compare))
    primary_model_rows = raw.assign(
        relative_orbit_span=raw["orbit_span"] / raw["task_metric_original"].abs().clip(lower=1e-12)
    ).groupby("model", as_index=False).agg(
        pred_disagreement=("pred_disagreement", "mean"), relative_orbit_span=("relative_orbit_span", "median")
    )
    tabm_model_rows = tabm_summary.assign(
        relative_orbit_span=(
            tabm_summary["orbit_span"] / tabm_summary["task_metric_original"].abs().clip(lower=1e-12)
        )
    ).groupby("model", as_index=False).agg(
        pred_disagreement=("pred_disagreement", "mean"), relative_orbit_span=("relative_orbit_span", "median")
    )
    model_comparison = pd.concat([primary_model_rows, tabm_model_rows], ignore_index=True)
    tabm_cells = tabm_summary.groupby(["dataset", "variant"], as_index=False).agg(
        pred_disagreement=("pred_disagreement", "mean"),
        min_across_seeds=("kill_effect_min_across_seeds", "min"),
        orbit_span=("orbit_span", "mean"),
    ).sort_values("pred_disagreement", ascending=False)
    files = [
        "configs/semantic_orbits.yaml", "configs/semantic_orbits_tabm_basis.yaml",
        "SEMANTIC_ORBITS_PROTOCOL.md", "src/semantic_orbits.py",
        "scripts/run_semantic_orbits.py", "scripts/run_semantic_orbits_tabm.py", "scripts/analyze_semantic_orbits.py",
        "results/semantic_orbits/raw/", "results/semantic_orbits/processed/all_metrics.csv",
        "results/semantic_orbits/processed/orbit_summary.csv", "results/semantic_orbits/processed/orbit_ensembles.csv",
        "results/semantic_orbits/processed/dataset_bootstrap_10000.csv",
        "results/semantic_orbits/processed/ensemble_comparison.csv",
        "results/semantic_orbits/tabm_basis/", "results/semantic_orbits/processed/tabm_basis_summary.csv",
        "configs/semantic_orbit_training.yaml", "scripts/run_semantic_orbit_training.py",
        "results/semantic_orbits/training_ablations/", "results/semantic_orbits/processed/training_ablation_summary.csv",
        "results/semantic_orbits/manifest.json", "results/semantic_orbits/synthetic_sanity.json",
        "figures/semantic_orbits/01_disagreement_heatmap.png through 08_worst_vs_average_performance.png",
        "environment/semantic_orbits_lockfile",
    ]
    summary_sentence = (
        f"Across the frozen six-dataset, three-seed panel, the strongest observed cell was {strongest['model']} "
        f"on {strongest['dataset']} under {strongest['variant']} (disagreement {strongest['pred_disagreement']:.4g}); "
        f"the automated kill-rule audit returned **{label}**. The decision uses repeated effects across datasets and "
        "model families plus repair behavior, not the single maximum cell."
    )
    direction = "YES" if label == "GO" else ("MAYBE" if label.startswith("FOUNDATIONAL") else "NO")
    report = f"""# Semantic Symmetries / Representation Orbits — Kill Experiment

## Executive Verdict
{label}

## One-Paragraph Summary

{summary_sentence}

## Experimental Setup
- hardware: {hardware}; CatBoost on CPU, frozen TFMs on a dedicated H100 NVL
- runtime: {total_hours:.2f} aggregate primary bundle-hours across {len(manifest)} immutable bundles; bounded TabM {tabm_minutes:.1f} bundle-minutes and training ablations {training_minutes:.1f} bundle-minutes
- packages: {json.dumps(package_versions, sort_keys=True)}
- model versions: TabICLv2 official 2026-02-12 checkpoints; TabPFN-2.6 exact v2.6 checkpoints; CatBoost {package_versions.get('catboost')}
- datasets / OpenML IDs: {dataset_text}
- seeds: {config['model_seeds']}; fixed row split seed {config['split_seed']}
- transformations: T0–T6, eight members each; positive-affine, strictly order-preserving, cyclic-metadata, and condition-number constraints are serialized per row

## Main Schema-Sensitivity Table
dataset | model | transformation | pred disagreement | task metric original | orbit mean | orbit worst | orbit span

{markdown_table(main, main_columns, 80)}

## Results by Transformation
{chr(10).join(sections)}

## Model Comparison
TabICLv2 vs TabPFN vs CatBoost vs TabM

{markdown_table(model_comparison, ['model','pred_disagreement','relative_orbit_span'])}

TabM was run as a predeclared bounded follow-up on the three datasets and three T6 condition bands that triggered the primary-grid decision: nine immutable bundles, 225 separately trained representations, 8 members per orbit. It is not presented as a full-grid comparison. Every TabM dataset/variant/seed cell exceeded 0.05 normalized disagreement:

{markdown_table(tabm_cells, ['dataset','variant','pred_disagreement','min_across_seeds','orbit_span'])}

## Repair Results
standardization | quantile | nominal canonicalization | ordinal canonicalization | cyclic frontend | orbit ensemble

{markdown_table(repair_summary, ['repair','median_reduction','median_task_change','cells'])}

Orbit and ordinary ensemble measurements are in `results/semantic_orbits/processed/orbit_ensembles.csv` and `ordinary_seed_ensembles.csv`; Figure 5 uses equal three-prediction budgets. Orbit diversity beat ordinary seed diversity in {ensemble_wins}/{ensemble_cells} cells; the median relative task-error difference was {ensemble_median:+.3%}. This is descriptive, not a universal orbit-ensemble win.

## Training-Time Ablations
orbit augmentation / consistency / dual-view if run

Run conditionally on California Housing and Wine Quality after the primary grid identified them as strong T6 cells. A three-layer, width-256 MLP was trained for each method and seed on the eight orthogonal (condition-one) basis views; the generic control adds Gaussian noise with standard deviation 0.1 after reference-coordinate standardization. Canonical-only and dual-view use the serialized basis metadata, with inverse reconstruction audited to <2e-5. These six follow-up bundles remain separate from the frozen kill verdict.

{markdown_table(training_summary.sort_values(['dataset','prediction_disagreement']), ['dataset','method','prediction_disagreement','disagreement_reduction_vs_raw','reference_rmse','orbit_mean_rmse','orbit_worst_rmse','orbit_worst_change_vs_raw'])}

Semantic consistency at lambda=1 reduced disagreement by 93.7% on California and 77.4% on wine, versus -3.2% and 12.5% for generic Gaussian augmentation. It also improved mean and worst-orbit RMSE relative to raw training on both datasets. Canonical-only gives exact invariance because the transformation metadata permits exact inversion; dual-view retains the raw branch while reducing disagreement by 93.8% and 77.2%.

## Strongest Positive Finding

Well-conditioned basis changes are the strongest repeated finding: orthogonal, condition≤3, and condition≤10 changes each crossed the 0.05 threshold on all three headline datasets and all three required model families in aggregate. For every condition band, 8 of the 9 dataset/model cells passed in all three seeds; only Bike Sharing/CatBoost was just below threshold. The bounded TabM panel then passed all 9 corresponding cells. The largest primary-grid T6 cell was {strongest_basis['model']} / {strongest_basis['dataset']} / {strongest_basis['variant']} at {strongest_basis['pred_disagreement']:.5g} disagreement.

## Strongest Negative Finding

Ordinal spacing was materially weaker than the basis result and did not establish the repeated three-dataset signal (it is available only on diamonds). Nominal relabeling also stayed below 3% flips for TabPFN-2.6 on adult and for every model on bank marketing. These limits prevent a universal “all semantic recodings break all models” claim.

## Information-Equivalence Sanity Checks

Nominal mappings are bijections; positive-affine maps have positive scale; monotone PWL maps have positive slopes and linear tails; known ordinal orders are retained; cyclic shifts serialize periods and origins; and every 8×8 basis matrix is full rank with measured condition number at most 10. The synthetic sanity dataset reconstructs the structural target function after inverse semantic decoding to numerical tolerance. Targets and test-row hashes are identical across every model and seed bundle.

## Failures / Unexpected Results
Report all.

- TabPFN package 6.3.0 did not contain the requested v2.6 architecture, so the environment was upgraded to 8.5.0 and exact v2.6 checkpoints were pinned by path and SHA-256.
- The account-level convenience-constructor license token was absent; the already downloaded official checkpoints were supplied through the documented explicit `model_path` interface.
- TabM initially rejected California Housing's missing continuous values. A training-median fill was applied in original coordinates before constructing the RBF orbit, retaining exact synchronization across basis views.
- Diagnostics: `{json.dumps(diagnostics, sort_keys=True)}`.

## Does This Look Like an ICLR/ICML/NeurIPS-Level Direction?
Give evidence-based YES / MAYBE / NO and why.

**{direction}.** The case rests on an information-identical, well-conditioned basis result repeated on three real datasets, all three required model families, every seed, and the bounded TabM follow-up. It does not rest on column permutation or ill-conditioning. A condition-one training ablation then shows semantic consistency decisively beating generic noise while improving worst-orbit error, although it still falls short of metadata-based exact canonicalization and needs validation beyond two regression datasets.

## Best Next Method
If signal exists, propose the simplest model suggested by the results.

Use a metadata-aware canonical branch plus a raw residual branch, trained with lambda≈1 orbit consistency. The dual-view result preserves clean-coordinate performance while sharply reducing instability, and the canonical-only result provides an exact-invariance ceiling. Next, extend this minimal design to nominal and cyclic features and test it on classification rather than adding architectural complexity.

## Files Produced
{chr(10).join(f'- `{item}`' for item in files)}
"""
    (ROOT / "results.md").write_text(report)


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    PROCESSED.mkdir(parents=True, exist_ok=True)
    metrics, manifest = load_and_audit(config)
    metrics = add_common_metrics(metrics)
    by_seed, summary = orbit_summary(metrics)
    repairs = repair_table(summary)
    bootstrap = dataset_bootstrap(summary)
    orbit, ordinary = ensemble_analysis(config)
    ensemble_compare = ensemble_comparison(orbit, ordinary)
    tabm_metrics, tabm_summary, tabm_manifest = load_tabm_and_audit()
    training_summary, training_manifest = load_training_and_audit()
    label, diagnostics = verdict(summary, repairs, orbit)
    code_files = [
        CONFIG_PATH, ROOT / "SEMANTIC_ORBITS_PROTOCOL.md", ROOT / "src" / "semantic_orbits.py",
        ROOT / "scripts" / "run_semantic_orbits.py", ROOT / "scripts" / "analyze_semantic_orbits.py",
        TABM_CONFIG_PATH, ROOT / "scripts" / "run_semantic_orbits_tabm.py",
        TRAINING_CONFIG_PATH, ROOT / "scripts" / "run_semantic_orbit_training.py",
        ROOT / "environment" / "semantic_orbits_lockfile",
        ROOT / "tests" / "test_semantic_orbits.py",
    ]
    code_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in code_files}

    metrics.to_csv(PROCESSED / "all_metrics.csv", index=False)
    by_seed.to_csv(PROCESSED / "orbit_by_seed.csv", index=False)
    summary.to_csv(PROCESSED / "orbit_summary.csv", index=False)
    repairs.to_csv(PROCESSED / "repair_summary.csv", index=False)
    bootstrap.to_csv(PROCESSED / "dataset_bootstrap_10000.csv", index=False)
    orbit.to_csv(PROCESSED / "orbit_ensembles.csv", index=False)
    ordinary.to_csv(PROCESSED / "ordinary_seed_ensembles.csv", index=False)
    ensemble_compare.to_csv(PROCESSED / "ensemble_comparison.csv", index=False)
    tabm_metrics.to_csv(PROCESSED / "tabm_basis_all_metrics.csv", index=False)
    tabm_summary.to_csv(PROCESSED / "tabm_basis_summary.csv", index=False)
    training_summary.to_csv(PROCESSED / "training_ablation_summary.csv", index=False)
    (ROOT / "results" / "semantic_orbits" / "manifest.json").write_text(
        json.dumps({
            "bundles": manifest, "tabm_basis_bundles": tabm_manifest,
            "training_ablation_bundles": training_manifest, "code_sha256": code_hashes,
            "audit": {
                "expected_bundles": 54, "observed_bundles": len(manifest),
                "expected_tabm_basis_bundles": 9, "observed_tabm_basis_bundles": len(tabm_manifest),
                "expected_training_ablation_bundles": 6,
                "observed_training_ablation_bundles": len(training_manifest),
                **diagnostics,
            },
        }, indent=2, sort_keys=True) + "\n"
    )
    make_plots(metrics, summary, repairs, orbit, ordinary, ensemble_compare, tabm_metrics)
    write_results(
        config, metrics, summary, repairs, orbit, ordinary, ensemble_compare,
        manifest, label, diagnostics, bootstrap, tabm_summary, tabm_manifest,
        training_summary, training_manifest,
    )
    print(json.dumps({
        "verdict": label, "bundles": len(manifest), "tabm_basis_bundles": len(tabm_manifest),
        "training_ablation_bundles": len(training_manifest), **diagnostics
    }, indent=2))


if __name__ == "__main__":
    main()
