#!/usr/bin/env python3
"""Audit and aggregate Experiment A plus validation-selected basis analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw" / "development" / "replication"
PROCESSED = ROOT / "results" / "processed"
CONFIG_PATH = ROOT / "configs" / "development_protocol.yaml"
PANEL_PATH = ROOT / "configs" / "dataset_panel.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_audit(allow_partial: bool) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    panel = json.loads(PANEL_PATH.read_text())
    datasets = [spec["key"] for spec in panel["datasets"] if spec["panel"] == "development"]
    expected = {(model, dataset, int(seed)) for model in config["models"] for dataset in datasets for seed in config["model_seeds"]}
    observed = set()
    frames = []
    manifest = []
    fingerprints = {}
    for metadata_path in sorted(RAW.glob("*/*/seed_*/metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        bundle = metadata_path.parent
        key = (metadata["model"], metadata["dataset_spec"]["key"], int(metadata["model_seed"]))
        if key in observed or metadata["status"] != "complete":
            raise RuntimeError(f"invalid or duplicate bundle: {metadata_path}")
        observed.add(key)
        if metadata["config_sha256"] != sha256(CONFIG_PATH) or metadata["dataset_panel_sha256"] != sha256(PANEL_PATH):
            raise RuntimeError(f"config drift: {metadata_path}")
        for filename, record in metadata["files"].items():
            path = bundle / filename
            if not path.exists() or sha256(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
                raise RuntimeError(f"artifact integrity failure: {path}")
        audit = metadata["split_audit"]
        if not all(audit[name] for name in (
            "train_validation_disjoint", "train_test_disjoint", "validation_test_disjoint"
        )):
            raise RuntimeError(f"split leakage: {metadata_path}")
        fingerprint = (
            audit["validation_row_order_sha256"], audit["test_row_order_sha256"],
            audit["validation_target_sha256"], audit["test_target_sha256"],
        )
        dataset = key[1]
        if dataset in fingerprints and fingerprints[dataset] != fingerprint:
            raise RuntimeError(f"row/target mismatch across model/seed: {dataset}")
        fingerprints[dataset] = fingerprint
        frame = pd.read_csv(bundle / "metrics.csv")
        if len(frame) != 50 or frame["representation_id"].nunique() != 25:
            raise RuntimeError(f"representation/metric count mismatch: {bundle}")
        member_counts = frame[(~frame["is_reference"]) & frame["split"].eq("test")].groupby("variant")["member"].nunique()
        if set(member_counts.index) != {"orthogonal_one", "orthogonal_all", "condition_le_3_all"} or not member_counts.eq(8).all():
            raise RuntimeError(f"incomplete orbits: {bundle}")
        if frame["max_reconstruction_error"].max() >= 1e-6 or frame.loc[
            frame["variant"].str.startswith("orthogonal"), "max_orthogonality_error"
        ].max() >= 1e-6:
            raise RuntimeError(f"equivalence failure: {bundle}")
        frames.append(frame)
        manifest.append({
            "model": key[0], "dataset": key[1], "model_seed": key[2],
            "bundle": str(bundle.relative_to(ROOT)), "wall_seconds": metadata["wall_seconds"],
            "metrics_sha256": metadata["files"]["metrics.csv"]["sha256"],
            "predictions_sha256": metadata["files"]["predictions.csv.gz"]["sha256"],
            "environment": metadata["environment"],
        })
    missing, extra = expected - observed, observed - expected
    if extra or (missing and not allow_partial):
        raise RuntimeError(f"coverage failure: missing={sorted(missing)}, extra={sorted(extra)}")
    if not frames:
        raise RuntimeError("no replication bundles")
    return pd.concat(frames, ignore_index=True), manifest


def add_common(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ("probability_rmse", "label_flip_rate", "prediction_rmse_normalized", "log_loss", "rmse"):
        if column not in result:
            result[column] = np.nan
    result["prediction_disagreement"] = np.where(
        result["problem_type"].eq("classification"), result["probability_rmse"], result["prediction_rmse_normalized"]
    )
    result["task_error"] = np.where(result["problem_type"].eq("classification"), result["log_loss"], result["rmse"])
    return result


def aggregate(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = metrics[metrics["split"].eq("test")]
    refs = test[test["is_reference"]][
        ["dataset", "model", "model_seed", "task_error"]
    ].rename(columns={"task_error": "task_original"})
    members = test[~test["is_reference"]].merge(refs, on=["dataset", "model", "model_seed"], validate="many_to_one")
    keys = ["dataset", "problem_type", "model", "model_seed", "variant", "scope", "selected_feature"]
    by_seed = members.groupby(keys, as_index=False).agg(
        mean_disagreement=("prediction_disagreement", "mean"),
        max_disagreement=("prediction_disagreement", "max"),
        label_flip_rate=("label_flip_rate", "mean"), task_original=("task_original", "first"),
        orbit_mean=("task_error", "mean"), orbit_worst=("task_error", "max"),
        orbit_best=("task_error", "min"), orbit_span=("task_error", lambda values: values.max()-values.min()),
        members=("member", "nunique"),
    )
    summary_keys = ["dataset", "problem_type", "model", "variant", "scope", "selected_feature"]
    summary = by_seed.groupby(summary_keys, as_index=False).agg(
        mean_disagreement=("mean_disagreement", "mean"),
        mean_disagreement_seed_sd=("mean_disagreement", "std"),
        max_disagreement=("max_disagreement", "mean"), label_flip_rate=("label_flip_rate", "mean"),
        task_original=("task_original", "mean"), orbit_mean=("orbit_mean", "mean"),
        orbit_worst=("orbit_worst", "mean"), orbit_best=("orbit_best", "mean"),
        orbit_span=("orbit_span", "mean"), seeds=("model_seed", "nunique"),
    )
    summary["meaningful"] = np.where(
        summary["problem_type"].eq("classification"), summary["label_flip_rate"].ge(0.03),
        summary["mean_disagreement"].ge(0.05),
    )
    return by_seed, summary


def basis_selection(metrics: pd.DataFrame) -> pd.DataFrame:
    values = metrics.copy()
    rows = []
    keys = ["dataset", "problem_type", "model", "model_seed", "variant", "scope"]
    for group_values, group in values.groupby(keys):
        if group_values[4] == "reference":
            continue
        validation = group[(group["split"].eq("validation")) & (~group["is_reference"])]
        test = group[(group["split"].eq("test")) & (~group["is_reference"])]
        reference = values[
            values["dataset"].eq(group_values[0]) & values["model"].eq(group_values[2])
            & values["model_seed"].eq(group_values[3]) & values["split"].eq("test") & values["is_reference"]
        ]
        if validation.empty or test.empty or len(reference) != 1:
            continue
        selected_id = validation.sort_values(["task_error", "representation_id"]).iloc[0]["representation_id"]
        selected_test = float(test[test["representation_id"].eq(selected_id)]["task_error"].iloc[0])
        test_best = float(test["task_error"].min())
        original = float(reference["task_error"].iloc[0])
        rows.append({
            **dict(zip(keys, group_values)), "task_original": original,
            "oracle_best_basis_task": test_best, "validation_selected_basis_task": selected_test,
            "oracle_best_relative_gain": (original-test_best)/max(abs(original),1e-12),
            "validation_selected_relative_gain": (original-selected_test)/max(abs(original),1e-12),
            "validation_selected_representation_id": selected_id,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    metrics, manifest = load_and_audit(args.allow_partial)
    metrics = add_common(metrics)
    by_seed, summary = aggregate(metrics)
    selection = basis_selection(metrics)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(PROCESSED / "replication_all_metrics.csv", index=False)
    by_seed.to_csv(PROCESSED / "replication_by_seed.csv", index=False)
    summary.to_csv(PROCESSED / "replication_summary.csv", index=False)
    selection.to_csv(PROCESSED / "basis_selection.csv", index=False)
    (ROOT / "results" / "replication_manifest.json").write_text(json.dumps({
        "complete_bundles": len(manifest), "bundles": manifest,
        "coverage_complete": len(manifest) == 165,
    }, indent=2, sort_keys=True) + "\n")
    print(f"bundles={len(manifest)}")
    print(summary.groupby(["variant", "model"]).agg(
        datasets=("dataset", "nunique"), median_disagreement=("mean_disagreement", "median"),
        meaningful_rate=("meaningful", "mean")
    ).to_string())


if __name__ == "__main__":
    main()
