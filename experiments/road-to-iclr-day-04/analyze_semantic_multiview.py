"""Aggregate the frozen neural-only semantic multi-view screen."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RAW_PATTERNS = (
    "semantic_multiview_weather_*.csv",
    "semantic_multiview_cooking_*.csv",
)


def read_raw() -> list[dict[str, str]]:
    paths: list[Path] = []
    for pattern in RAW_PATTERNS:
        paths.extend(RESULTS.glob(pattern))
    # Do not accidentally ingest derived files if the naming scheme grows.
    paths = [path for path in paths if path.name not in {
        "semantic_multiview_comparisons.csv",
        "semantic_multiview_screen.csv",
    }]
    rows: list[dict[str, str]] = []
    for path in sorted(paths):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    keys = [(row["dataset"], row["model"], row["seed"], row["method"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate semantic multi-view cells")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty table")
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    raw = read_raw()
    raw = sorted(raw, key=lambda row: (
        row["dataset"], row["model"], int(row["seed"]), row["method"]
    ))
    by_key = {
        (row["dataset"], row["model"], int(row["seed"]), row["method"]): row
        for row in raw
    }
    comparisons: list[dict[str, object]] = []
    for row in raw:
        if row["method"] == "ple":
            continue
        key = (row["dataset"], row["model"], int(row["seed"]), "ple")
        baseline = by_key[key]
        baseline_rmse = float(baseline["test_rmse"])
        method_rmse = float(row["test_rmse"])
        comparisons.append({
            "dataset": row["dataset"],
            "model": row["model"],
            "seed": int(row["seed"]),
            "method": row["method"],
            "ple_test_rmse": baseline_rmse,
            "method_test_rmse": method_rmse,
            "gain_vs_ple_pct": 100.0 * (baseline_rmse - method_rmse) / baseline_rmse,
            "val_gain_vs_ple_pct": 100.0
            * (float(baseline["val_rmse"]) - float(row["val_rmse"]))
            / float(baseline["val_rmse"]),
            "parameter_ratio_vs_ple": float(row["parameters"]) / float(baseline["parameters"]),
        })
    write_csv(RESULTS / "semantic_multiview_comparisons.csv", comparisons)

    methods: list[dict[str, object]] = []
    for method in ("topology", "topology_wrong", "multiview_noalign", "multiview_vicreg", "multiview_wrong"):
        selected = [row for row in comparisons if row["method"] == method]
        if not selected:
            continue
        gains = np.asarray([float(row["gain_vs_ple_pct"]) for row in selected])
        methods.append({
            "method": method,
            "pairs": len(selected),
            "wins_vs_ple": int(np.sum(gains > 0)),
            "mean_gain_vs_ple_pct": float(gains.mean()),
            "median_gain_vs_ple_pct": float(np.median(gains)),
        })

    mechanism = []
    for dataset in ("weather", "cooking-time"):
        for model in ("mlp", "resnet", "ft_transformer"):
            correct = by_key[(dataset, model, 20260827, "multiview_vicreg")]
            wrong = by_key[(dataset, model, 20260827, "multiview_wrong")]
            correct_rmse, wrong_rmse = float(correct["test_rmse"]), float(wrong["test_rmse"])
            mechanism.append({
                "dataset": dataset,
                "model": model,
                "correct_test_rmse": correct_rmse,
                "wrong_test_rmse": wrong_rmse,
                "correct_gain_pct": 100.0 * (wrong_rmse - correct_rmse) / wrong_rmse,
            })
    mechanism_gains = np.asarray([row["correct_gain_pct"] for row in mechanism])
    summary = {
        "status": "falsified_as_broad_default",
        "protocol": {
            "datasets": ["weather", "cooking-time"],
            "split": "official temporal partitions with fixed train-only subsampling",
            "models": ["mlp", "resnet", "ft_transformer"],
            "seed": 20260827,
            "rows": {"train": 50000, "validation": 15000, "test": 15000},
            "baseline": "16-bin quantile PLE",
            "alignment": "VICReg-style same-row latent alignment, weight 0.01",
            "tree_or_anchor_dependency": False,
        },
        "method_summary": methods,
        "correct_vs_wrong_geometry": {
            "pairs": len(mechanism),
            "correct_wins": int(np.sum(mechanism_gains > 0)),
            "mean_correct_gain_pct": float(mechanism_gains.mean()),
            "cells": mechanism,
        },
        "headline": (
            "Semantic multi-view VICReg beats PLE in only 1/6 architecture-dataset "
            "cells (-0.69% mean RMSE gain). Correct cyclic geometry beats the "
            "permuted control in 4/6 cells, but by only +0.03% on average."
        ),
        "decision": (
            "Do not promote this full-row alignment as a default representation. "
            "If continued, align only declared semantic field tokens and gate the "
            "auxiliary loss without using test performance."
        ),
    }
    (RESULTS / "semantic_multiview_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(summary["headline"])


if __name__ == "__main__":
    main()
