#!/usr/bin/env python3
"""Frozen hierarchical analysis for independent regression confirmation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def unique(pattern: str) -> Path:
    paths = sorted(ROOT.glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(f"expected one {pattern}, found {paths}")
    return paths[0]


def paired(frame: pd.DataFrame, left: str, right: str) -> dict[str, np.ndarray]:
    result = {}
    for dataset, group in frame.groupby("dataset", sort=True):
        pivot = group.pivot(index="episode_index", columns="method", values="loss")
        result[str(dataset)] = (pivot[left] - pivot[right]).to_numpy()
    return result


def hierarchical(values: dict[str, np.ndarray], draws: int, seed: int) -> dict:
    names = sorted(values)
    observed = float(np.mean([values[name].mean() for name in names]))
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for draw in range(draws):
        selected = rng.choice(names, len(names), replace=True)
        samples[draw] = np.mean([
            np.mean(rng.choice(values[str(name)], len(values[str(name)]), replace=True))
            for name in selected
        ])
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"gain": observed, "ci_low": float(low), "ci_high": float(high)}


def episode_ci(values: np.ndarray, draws: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(values), size=(draws, len(values)))
    return np.quantile(values[index].mean(axis=1), [0.025, 0.975]).tolist()


def main() -> None:
    cells_path = unique("results/processed/regression_confirmation_*_cells.csv")
    metadata_path = unique("results/raw/regression_confirmation_*.metadata.json")
    config = yaml.safe_load((ROOT / "configs/regression_confirmation.yaml").read_text())
    frame = pd.read_csv(cells_path)
    metadata = json.loads(metadata_path.read_text())
    if set(frame["task_type"]) != {"regression"} or frame["dataset"].nunique() != 5:
        raise AssertionError("confirmation panel is incomplete")
    draws = int(config["bootstrap_draws"])
    comparisons = {}
    for index, (label, left, right) in enumerate((
        ("competence_vs_fixed", "fixed", "competence"),
        ("competence_vs_uniform", "uniform", "competence"),
        ("competence_vs_hard", "hard_cv", "competence"),
        ("fixed_to_best_individual", "fixed", "best_individual_oracle"),
    )):
        comparisons[label] = hierarchical(paired(frame, left, right), draws, 23_000 + index)
    primary = paired(frame, "fixed", "competence")
    per_dataset = {
        name: {"gain": float(values.mean()), "ci": episode_ci(values, draws, 24_000 + index)}
        for index, (name, values) in enumerate(sorted(primary.items()))
    }
    audit = {
        "protocol": "REGRESSION_CONFIRMATION_PROTOCOL.md",
        "metadata": metadata,
        "comparisons": comparisons,
        "per_dataset_competence_vs_fixed": per_dataset,
        "mean_losses": frame.groupby("method")["loss"].mean().to_dict(),
        "positive_dataset_count": int(sum(item["gain"] > 0 for item in per_dataset.values())),
        "confirmation_pass": comparisons["competence_vs_fixed"]["ci_low"] > 0,
    }
    audit_path = ROOT / "results/processed/regression_confirmation_audit_v1.json"
    summary_path = ROOT / "results/processed/regression_confirmation_summary_v1.csv"
    for output in (audit_path, summary_path):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
    frame.groupby(["dataset", "feature_count", "method"], as_index=False).agg(
        mean_loss=("loss", "mean"), episodes=("episode_index", "nunique")
    ).to_csv(summary_path, index=False)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
