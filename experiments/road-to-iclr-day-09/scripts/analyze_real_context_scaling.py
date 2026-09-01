#!/usr/bin/env python3
"""Frozen hierarchical context-scaling analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def unique(pattern: str) -> Path:
    paths = sorted(ROOT.glob(pattern))
    if len(paths) != 1: raise RuntimeError(f"expected one {pattern}, found {paths}")
    return paths[0]


def hierarchical(values: dict[str, np.ndarray], draws: int, seed: int) -> dict:
    names = sorted(values); rng = np.random.default_rng(seed)
    observed = float(np.mean([values[name].mean() for name in names]))
    samples = np.empty(draws)
    for draw in range(draws):
        chosen = rng.choice(names, len(names), replace=True)
        samples[draw] = np.mean([
            np.mean(rng.choice(values[str(name)], len(values[str(name)]), replace=True))
            for name in chosen
        ])
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"value": observed, "ci_low": float(low), "ci_high": float(high)}


def main() -> None:
    cells = pd.read_csv(unique("results/processed/real_context_scaling_*_cells.csv"))
    metadata = json.loads(unique("results/raw/real_context_scaling_*.metadata.json").read_text())
    config = yaml.safe_load((ROOT / "configs/real_context_scaling.yaml").read_text())
    draws = int(config["bootstrap_draws"])
    pivot = cells.pivot(
        index=["dataset", "repeat", "context_size"], columns="method", values="loss"
    ).reset_index()
    pivot["gain"] = pivot["fixed"] - pivot["competence"]
    gain_by_context = {}
    for index, context_size in enumerate(config["context_sizes"]):
        part = pivot[pivot["context_size"] == context_size]
        values = {name: group["gain"].to_numpy() for name, group in part.groupby("dataset")}
        gain_by_context[str(context_size)] = hierarchical(values, draws, 25_000 + index)
    slopes = {}
    x = np.log2(np.asarray(config["context_sizes"], dtype=float))
    for dataset, group in pivot.groupby("dataset", sort=True):
        rows = []
        for _, repeat in group.groupby("repeat", sort=True):
            ordered = repeat.set_index("context_size").loc[config["context_sizes"]]
            rows.append(float(np.polyfit(x, ordered["gain"].to_numpy(), 1)[0]))
        slopes[str(dataset)] = np.asarray(rows)
    slope = hierarchical(slopes, draws, 25_100)
    earliest = next(
        (int(size) for size in config["context_sizes"] if gain_by_context[str(size)]["ci_low"] > 0),
        None,
    )
    audit = {
        "protocol": "REAL_CONTEXT_SCALING_PROTOCOL.md", "metadata": metadata,
        "gain_by_context": gain_by_context, "log2_context_gain_slope": slope,
        "positive_slope_pass": slope["ci_low"] > 0,
        "earliest_positive_context": earliest,
    }
    audit_path = ROOT / "results/processed/real_context_scaling_audit_v1.json"
    summary_path = ROOT / "results/processed/real_context_scaling_summary_v1.csv"
    for output in (audit_path, summary_path):
        if output.exists(): raise FileExistsError(f"refusing to overwrite {output}")
    cells.groupby(["dataset", "context_size", "method"], as_index=False).agg(
        mean_loss=("loss", "mean"), episodes=("episode_index", "nunique")
    ).to_csv(summary_path, index=False)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
