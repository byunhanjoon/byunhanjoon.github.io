#!/usr/bin/env python3
"""Integrity-first analysis for the frozen E0 reproduction panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stats import mean_interval


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="v1")
    return parser.parse_args()


def exclusive_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, mode="x")


def main() -> None:
    args = parse_args()
    metadata_paths = sorted((ROOT / "results/raw/e0_reproduction").glob("*/*.json"))
    records = [json.loads(path.read_text()) for path in metadata_paths]
    if len(records) != 120:
        raise AssertionError(f"expected 120 E0 records, found {len(records)}")
    status = pd.DataFrame(
        [{"model": x["model"], "dataset": x["dataset"], "status": x["status"], "failure": x.get("failure")} for x in records]
    )
    expected = {"complete": 72, "unavailable": 48}
    if status.status.value_counts().to_dict() != expected:
        raise AssertionError(f"unexpected status counts: {status.status.value_counts().to_dict()}")

    completed = [x for x in records if x["status"] == "complete"]
    row_hashes: dict[str, set[str]] = {}
    rows = []
    max_inverse_error = 0.0
    for record in completed:
        result_path = Path(record["result_path"])
        if sha256(result_path) != record["result_sha256"]:
            raise AssertionError(f"checksum mismatch: {result_path}")
        audit = record["transform_audit"]
        if audit["strict_order_violations"] or not audit["missing_mask_preserved"]:
            raise AssertionError(f"transform audit failed: {record['job_key']}")
        if not audit["all_finite_inputs_have_finite_outputs"]:
            raise AssertionError(f"nonfinite transform: {record['job_key']}")
        max_inverse_error = max(max_inverse_error, float(audit["max_rel_reconstruction_error"]))
        telemetry = record["telemetry"]
        if telemetry["clean"]["shared_fit_id"] != telemetry["query_only"]["shared_fit_id"]:
            raise AssertionError("original-context queries do not share a fit")
        if telemetry["matched"]["shared_fit_id"] != telemetry["context_only"]["shared_fit_id"]:
            raise AssertionError("transformed-context queries do not share a fit")
        with np.load(result_path, allow_pickle=False) as bundle:
            row_hash = hashlib.sha256(bundle["test_row_ids"].tobytes()).hexdigest()
            row_hashes.setdefault(record["dataset"], set()).add(row_hash)
            if any(bundle[f"prediction__{condition}"].shape[0] != bundle["y_test"].shape[0] for condition in ("clean", "matched", "context_only", "query_only")):
                raise AssertionError("prediction/row alignment failed")
        matched = record["metrics"]["matched"]
        disagreement_key = "normalized_absolute_disagreement" if record["problem_type"] == "regression" else "total_variation"
        rows.append({
            "job_key": record["job_key"],
            "model": record["model"],
            "dataset": record["dataset"],
            "problem_type": record["problem_type"],
            "transform": record["transformation"]["name"],
            "seed": record["seed"],
            "matched_disagreement": matched[disagreement_key],
            "matched_js": matched.get("js_divergence", np.nan),
            "matched_flip_rate": matched.get("argmax_flip_rate", np.nan),
            "matched_loss_gap": matched["normalized_isomorphism_gap"],
            "clean_loss": record["metrics"]["clean"]["loss"],
            "wall_clock_seconds": record["wall_clock_seconds"],
            "peak_gpu_memory_bytes": record["peak_gpu_memory_bytes"],
        })
    if any(len(value) != 1 for value in row_hashes.values()):
        raise AssertionError(f"row identities differ within dataset: {row_hashes}")

    frame = pd.DataFrame(rows)
    identity = frame[frame["transform"] == "identity"].set_index(["model", "dataset", "seed"])
    nonidentity = frame[frame["transform"] != "identity"].copy()
    keys = pd.MultiIndex.from_frame(nonidentity[["model", "dataset", "seed"]])
    nonidentity["identity_noise"] = identity.loc[keys, "matched_disagreement"].to_numpy()
    nonidentity["excess_disagreement"] = nonidentity.matched_disagreement - nonidentity.identity_noise

    summaries = []
    for (model, dataset, problem_type), group in nonidentity.groupby(["model", "dataset", "problem_type"]):
        excess = mean_interval(group.excess_disagreement.to_numpy(), draws=10000, seed=17)
        gap = mean_interval(group.matched_loss_gap.to_numpy(), draws=10000, seed=19)
        summaries.append({
            "model": model,
            "dataset": dataset,
            "problem_type": problem_type,
            "cells": len(group),
            "identity_noise_mean": float(group.identity_noise.mean()),
            "excess_disagreement_mean": excess[0],
            "excess_disagreement_ci_low": excess[1],
            "excess_disagreement_ci_high": excess[2],
            "matched_loss_gap_mean": gap[0],
            "matched_loss_gap_ci_low": gap[1],
            "matched_loss_gap_ci_high": gap[2],
            "argmax_flip_rate_mean": float(group.matched_flip_rate.mean()),
            "matched_js_mean": float(group.matched_js.mean()),
            "wall_clock_seconds": float(group.wall_clock_seconds.sum()),
        })
    summary = pd.DataFrame(summaries)
    out = ROOT / "results/processed"
    exclusive_csv(status, out / f"e0_availability_{args.tag}.csv")
    exclusive_csv(nonidentity, out / f"e0_cells_{args.tag}.csv")
    exclusive_csv(summary, out / f"e0_model_dataset_summary_{args.tag}.csv")
    integrity = {
        "records": len(records),
        "complete": len(completed),
        "unavailable": sum(x["status"] == "unavailable" for x in records),
        "checksums_verified": len(completed),
        "row_alignment_verified": True,
        "shared_fit_pairing_verified": True,
        "transform_audits_verified": True,
        "max_relative_inverse_error": max_inverse_error,
        "row_hashes_per_dataset": {key: len(value) for key, value in row_hashes.items()},
    }
    integrity_path = out / f"e0_integrity_{args.tag}.json"
    with integrity_path.open("x", encoding="utf-8") as handle:
        json.dump(integrity, handle, indent=2, sort_keys=True)
        handle.write("\n")

    models = ["tabicl_v2_single", "tabicl_v2_default", "mitra_default"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1))
    for axis, (dataset, label) in zip(axes, (("churn", "TV minus identity noise"), ("wine_quality", "normalized disagreement minus identity noise")), strict=True):
        cell = summary[summary.dataset == dataset].set_index("model").reindex(models)
        axis.bar(np.arange(len(models)), cell.excess_disagreement_mean, color=["#2667ff", "#4cc9f0", "#ef476f"])
        axis.errorbar(
            np.arange(len(models)), cell.excess_disagreement_mean,
            yerr=np.vstack([
                cell.excess_disagreement_mean - cell.excess_disagreement_ci_low,
                cell.excess_disagreement_ci_high - cell.excess_disagreement_mean,
            ]), fmt="none", color="black", capsize=3,
        )
        axis.axhline(0, color="black", lw=1)
        axis.set_xticks(np.arange(len(models)), ["TabICL\nsingle", "TabICL\ndefault", "Mitra"], rotation=0)
        axis.set(title=dataset, ylabel=label)
    fig.suptitle("E0 matched reparameterization disagreement (three transforms × three seeds)")
    fig.tight_layout()
    figure_path = ROOT / f"figures/e0_reproduction_{args.tag}.png"
    if figure_path.exists():
        raise FileExistsError(figure_path)
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(summary.to_string(index=False))
    print(json.dumps(integrity, indent=2))


if __name__ == "__main__":
    main()
