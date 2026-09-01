#!/usr/bin/env python3
"""Frozen hierarchical analysis for numeric real-panel competence transfer."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def unique(pattern: str) -> Path:
    paths = sorted(ROOT.glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(f"expected one {pattern}, found {paths}")
    return paths[0]


def paired_values(frame: pd.DataFrame, left: str, right: str) -> dict[str, np.ndarray]:
    result = {}
    for dataset, group in frame.groupby("dataset", sort=True):
        pivot = group.pivot(index="episode_index", columns="method", values="loss")
        result[str(dataset)] = (pivot[left] - pivot[right]).to_numpy()
    return result


def hierarchical_bootstrap(values: dict[str, np.ndarray], draws: int, seed: int) -> dict:
    names = sorted(values)
    observed = float(np.mean([values[name].mean() for name in names]))
    rng = np.random.default_rng(seed)
    sampled = np.empty(draws)
    for draw in range(draws):
        chosen = rng.choice(names, size=len(names), replace=True)
        dataset_means = []
        for name in chosen:
            row = values[str(name)]
            dataset_means.append(float(np.mean(rng.choice(row, size=len(row), replace=True))))
        sampled[draw] = np.mean(dataset_means)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return {"gain": observed, "ci_low": float(low), "ci_high": float(high)}


def paired_episode_bootstrap(values: np.ndarray, draws: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    return np.quantile(values[indices].mean(axis=1), [0.025, 0.975]).tolist()


def main() -> None:
    cells_path = unique("results/processed/real_panel_competence_*_cells.csv")
    metadata_path = unique("results/raw/real_panel_competence_*.metadata.json")
    config = yaml.safe_load((ROOT / "configs/real_panel_competence.yaml").read_text())
    frame = pd.read_csv(cells_path)
    metadata = json.loads(metadata_path.read_text())
    draws = int(config["bootstrap_draws"])
    summary_path = ROOT / "results/processed/real_panel_competence_summary_v1.csv"
    audit_path = ROOT / "results/processed/real_panel_competence_audit_v1.json"
    figure_path = ROOT / "figures/real_panel_competence_v1.png"
    for output in (summary_path, audit_path, figure_path):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")

    summary = frame.groupby(["task_type", "dataset", "feature_count", "method"], as_index=False).agg(
        mean_loss=("loss", "mean"), episodes=("episode_index", "nunique")
    )
    summary.to_csv(summary_path, index=False)
    audit = {"protocol": "REAL_PANEL_COMPETENCE_PROTOCOL.md", "metadata": metadata, "tasks": {}}
    passing = []
    no_harm = {}
    for task_index, task_type in enumerate(("classification", "regression")):
        task = frame[frame["task_type"] == task_type]
        comparisons = {}
        for comparison_index, (label, left, right) in enumerate((
            ("competence_vs_fixed", "fixed", "competence"),
            ("competence_vs_uniform", "uniform", "competence"),
            ("competence_vs_hard", "hard_cv", "competence"),
            ("fixed_to_best_individual", "fixed", "best_individual_oracle"),
        )):
            values = paired_values(task, left, right)
            comparisons[label] = hierarchical_bootstrap(
                values, draws, 18_000 + task_index * 100 + comparison_index
            )
        primary_values = paired_values(task, "fixed", "competence")
        per_dataset = {
            name: {
                "gain": float(values.mean()),
                "ci": paired_episode_bootstrap(values, draws, 19_000 + index),
            }
            for index, (name, values) in enumerate(sorted(primary_values.items()))
        }
        threshold = float(config["material_harm"][task_type])
        primary = comparisons["competence_vs_fixed"]
        if primary["ci_low"] > 0:
            passing.append(task_type)
        no_harm[task_type] = primary["ci_high"] >= -threshold
        audit["tasks"][task_type] = {
            "comparisons": comparisons,
            "per_dataset_competence_vs_fixed": per_dataset,
            "mean_losses": task.groupby("method")["loss"].mean().to_dict(),
            "datasets": sorted(task["dataset"].unique().tolist()),
        }
    audit["passing_task_types"] = passing
    audit["no_material_harm"] = no_harm
    audit["strong_transfer_pass"] = len(passing) == 2
    audit["scoped_transfer_pass"] = bool(passing and all(no_harm.values()))
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")

    methods = ["fixed", "competence", "hard_cv", "best_individual_oracle"]
    pivot = summary[summary["method"].isin(methods)].pivot(
        index=["task_type", "dataset"], columns="method", values="mean_loss"
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, task_type in zip(axes, ("classification", "regression")):
        cell = pivot.loc[task_type]
        x = np.arange(len(cell))
        width = 0.2
        for offset, method in enumerate(methods):
            axis.bar(x + (offset - 1.5) * width, cell[method], width, label=method)
        axis.set_xticks(x, cell.index, rotation=25, ha="right")
        axis.set_ylabel("Log loss" if task_type == "classification" else "Standardized MSE")
        axis.set_title(task_type.capitalize())
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Real numeric panel: synthetic-tuned competence transfer")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
