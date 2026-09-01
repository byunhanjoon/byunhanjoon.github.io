#!/usr/bin/env python3
"""Test whether mechanism identification is aligned with predictive expert routing."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def unique_glob(pattern: str) -> Path:
    matches = sorted(ROOT.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one match for {pattern!r}, found {matches}")
    return matches[0]


INPUTS = {
    "development": ROOT / "results/raw/e1_primary_b779842a24_t420_n64.npz",
    "independent_replication": unique_glob("results/raw/fallback_dial_replication_*.npz"),
}


def interval(values: np.ndarray, seed: int, draws: int = 10_000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = np.empty(draws)
    for start in range(0, draws, 500):
        stop = min(start + 500, draws)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        sampled[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def main() -> None:
    output_csv = ROOT / "results/processed/fallback_routing_alignment_v1.csv"
    output_json = ROOT / "results/processed/fallback_routing_alignment_audit_v1.json"
    output_figure = ROOT / "figures/fallback_routing_alignment_v1.png"
    for path in (output_csv, output_json, output_figure):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    records = []
    for run_index, (run, path) in enumerate(INPUTS.items()):
        raw = np.load(path, allow_pickle=False)
        rho = raw["rho"].astype(float)
        tasks = raw["task_type"].astype(str)
        mechanism = raw["mechanism"].astype(int)
        selector = raw["combined_oof_probability"].astype(float).argmax(axis=1)
        routing_gain = raw["stable_expert_loss"] - raw["combined_expert_loss"]
        matched_family_gain = raw["stable_expert_loss"] - raw["oracle_expert_loss"]
        for task_index, task in enumerate(("classification", "regression")):
            for rho_index, rho_value in enumerate(sorted(np.unique(rho[tasks == task]))):
                mask = (tasks == task) & np.isclose(rho, rho_value)
                routing = interval(routing_gain[mask], 6100 + run_index * 100 + task_index * 20 + rho_index)
                matched = interval(matched_family_gain[mask], 7100 + run_index * 100 + task_index * 20 + rho_index)
                records.append({
                    "run": run,
                    "task_type": task,
                    "rho": float(rho_value),
                    "tasks": int(mask.sum()),
                    "mechanism_selector_accuracy": float(np.mean(selector[mask] == mechanism[mask])),
                    "shape_routing_gain": routing[0],
                    "shape_routing_ci_low": routing[1],
                    "shape_routing_ci_high": routing[2],
                    "matched_family_gain": matched[0],
                    "matched_family_ci_low": matched[1],
                    "matched_family_ci_high": matched[2],
                })
    frame = pd.DataFrame(records)
    frame.to_csv(output_csv, index=False)

    correlations = []
    for (run, task), cell in frame.groupby(["run", "task_type"]):
        correlations.append({
            "run": run,
            "task_type": task,
            "selector_accuracy_routing_gain_pearson": float(
                np.corrcoef(cell["mechanism_selector_accuracy"], cell["shape_routing_gain"])[0, 1]
            ),
            "rho1_selector_accuracy": float(cell.loc[cell["rho"].idxmax(), "mechanism_selector_accuracy"]),
            "rho1_routing_gain": float(cell.loc[cell["rho"].idxmax(), "shape_routing_gain"]),
            "rho1_matched_family_gain": float(cell.loc[cell["rho"].idxmax(), "matched_family_gain"]),
        })
    audit = {
        "interpretation": (
            "Positive gains mean lower predictive loss than the stable-selector mixture. "
            "The matched-family expert is not a loss oracle; it routes to the generator-family-labelled expert."
        ),
        "inputs": {key: str(value.relative_to(ROOT)) for key, value in INPUTS.items()},
        "correlations": correlations,
    }
    output_json.write_text(json.dumps(audit, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    colors = {"development": "#3366cc", "independent_replication": "#d95f02"}
    markers = {"classification": "o", "regression": "s"}
    for axis, metric, title in (
        (axes[0], "shape_routing_gain", "Shape-informed routing"),
        (axes[1], "matched_family_gain", "One-hot matched-family routing"),
    ):
        for (run, task), cell in frame.groupby(["run", "task_type"]):
            cell = cell.sort_values("mechanism_selector_accuracy")
            axis.plot(
                cell["mechanism_selector_accuracy"], cell[metric], marker=markers[task],
                color=colors[run], label=f"{run}, {task}" if metric == "shape_routing_gain" else None,
            )
        axis.axhline(0, color="black", linewidth=0.8, linestyle="--")
        axis.set_xlabel("Mechanism selector accuracy")
        axis.set_ylabel("Stable-mixture loss minus routed loss")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Task-family identification is not sufficient for predictive routing")
    fig.tight_layout()
    fig.savefig(output_figure, dpi=180)
    plt.close(fig)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
