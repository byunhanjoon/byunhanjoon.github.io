#!/usr/bin/env python3
"""Analyze and audit the frozen Census CBP hierarchy addendum."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


HERE = Path(__file__).resolve().parent
RAW = HERE / "raw" / "hierarchy_gap"
TABLE = HERE / "tables"
FIG = HERE / "figures"


def stats(frame: pd.DataFrame) -> dict:
    predicted = frame.predicted_delta.to_numpy(float)
    actual = frame.actual_delta.to_numpy(float)
    return {
        "n": len(frame),
        "pearson": float(pearsonr(predicted, actual).statistic),
        "spearman": float(spearmanr(predicted, actual).statistic),
        "mae": float(np.mean(np.abs(predicted - actual))),
        "sign_accuracy": float(np.mean(np.sign(predicted) == np.sign(actual))),
        "calibration_slope": float(np.polyfit(predicted, actual, 1)[0]),
        "harmful": int(np.sum(actual < 0)),
    }


def valid_seals() -> tuple[bool, list[dict]]:
    details = []
    for path in sorted((RAW / "sealed_predictions").glob("*.json")):
        value = json.loads(path.read_text())
        claimed = value.pop("payload_sha256")
        observed = hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        details.append(
            {
                "file": path.name,
                "hash_valid": claimed == observed,
                "disjoint": not (set(value["train_state_ids"]) & set(value["test_state_ids"])),
                "outcomes_accessed": value["outer_test_outcomes_accessed"],
            }
        )
    return (
        len(details) == 3
        and all(x["hash_valid"] and x["disjoint"] and not x["outcomes_accessed"] for x in details),
        details,
    )


def loso(frame: pd.DataFrame, predictor: str) -> dict:
    prediction = np.zeros(len(frame))
    for source in frame.source.unique():
        train = frame.source != source
        test = ~train
        x = frame.loc[train, predictor]
        y = frame.loc[train, "actual_delta"]
        if np.std(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            prediction[test] = intercept + slope * frame.loc[test, predictor]
        else:
            prediction[test] = y.mean()
    scored = frame[["actual_delta"]].copy()
    scored["predicted_delta"] = prediction
    return stats(scored)


def main() -> None:
    cells = pd.read_csv(RAW / "cells.csv")
    aggregate = cells.groupby(["source", "family", "operator"], as_index=False).agg(
        predicted_delta=("predicted_delta", "mean"),
        actual_delta=("actual_delta", "mean"),
        predicted_se=("predicted_delta", "std"),
        support_distance=("support_distance", "mean"),
        cover_radius=("cover_radius", "mean"),
        raw_smoothness=("raw_smoothness", "mean"),
    )
    old = pd.read_csv(TABLE / "prospective_results.csv")
    combined = pd.concat([old, aggregate], ignore_index=True)
    combined_heuristics = pd.DataFrame(
        [{"predictor": predictor, **loso(combined, predictor)} for predictor in
         ["predicted_delta", "support_distance", "cover_radius", "raw_smoothness"]]
    )
    seals_ok, seal_details = valid_seals()
    protocol_hash = hashlib.sha256((HERE / "GAP_PROTOCOL.md").read_bytes()).hexdigest()
    config_hash = hashlib.sha256((HERE / "GAP_CONFIG.json").read_bytes()).hexdigest()
    hashes_ok = protocol_hash == "36b20009bb3716ba293126c64443604fdf00768da5086f7e3ca53fd66abb2e28" and config_hash == "3bdec9c0184dba80e43b856350479efd2453f71638b7776c91034e843068e7fd"
    cell_stats = stats(cells)
    aggregate_stats = stats(aggregate)
    combined_stats = stats(combined)
    criteria = {
        "G1_integrity": bool(seals_ok and hashes_ok and len(cells) == 9 and not cells.isna().any().any()),
        "G2_split_cell_spearman_at_least_050": cell_stats["spearman"] >= 0.50,
        "G3_split_cell_sign_accuracy_at_least_075": cell_stats["sign_accuracy"] >= 0.75,
        "G4_two_aggregate_signs_correct": int(np.sum(np.sign(aggregate.predicted_delta) == np.sign(aggregate.actual_delta))) >= 2,
        "G5_combined_spearman_and_sign": combined_stats["spearman"] >= 0.60 and combined_stats["sign_accuracy"] >= 0.75,
    }
    criteria["passed_all"] = all(criteria.values())
    summary = {
        "status": "PASS" if criteria["passed_all"] else "FAIL",
        "source": {"rows": 37783, "states": 922},
        "protocol_hash": protocol_hash,
        "config_hash": config_hash,
        "hashes_valid": hashes_ok,
        "seals_valid": seals_ok,
        "seal_details": seal_details,
        "split_cells": cell_stats,
        "source_operator_aggregates": aggregate_stats,
        "combined_prospective_aggregates": combined_stats,
        "combined_leave_one_source_out": combined_heuristics.set_index("predictor").to_dict("index"),
        "criteria": criteria,
    }
    (RAW / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    aggregate.to_csv(TABLE / "hierarchy_gap_results.csv", index=False)
    combined.to_csv(TABLE / "prospective_results_with_gap.csv", index=False)
    combined_heuristics.to_csv(TABLE / "prospective_heuristics_with_gap.csv", index=False)

    plt.figure(figsize=(6.2, 5.2))
    for source, group in combined.groupby("source"):
        plt.scatter(group.predicted_delta, group.actual_delta, s=48, label=source)
    lo = combined[["predicted_delta", "actual_delta"]].min().min()
    hi = combined[["predicted_delta", "actual_delta"]].max().max()
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    plt.axhline(0, color="gray", linewidth=0.7)
    plt.axvline(0, color="gray", linewidth=0.7)
    plt.xlabel("Nested state-CV predicted Δ")
    plt.ylabel("Sealed outer-state actual Δ")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG / "figure_12_hierarchy_gap.png", dpi=180)
    plt.savefig(FIG / "figure_12_hierarchy_gap.pdf")
    plt.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
