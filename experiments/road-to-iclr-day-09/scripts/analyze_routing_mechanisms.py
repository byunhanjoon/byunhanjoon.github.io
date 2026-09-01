#!/usr/bin/env python3
"""Decompose rho=1 routing alignment over all frozen mechanism families."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MECHANISMS = ("linear", "additive", "threshold", "interaction", "partition", "periodic")


def unique(pattern: str) -> Path:
    paths = sorted(ROOT.glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(f"expected one {pattern}, found {paths}")
    return paths[0]


INPUTS = {
    (64, 8): ROOT / "results/raw/e1_primary_b779842a24_t420_n64.npz",
    (96, 8): unique("results/raw/fallback_routing_axis_n96_d8_*.npz"),
    (64, 12): unique("results/raw/fallback_routing_axis_n64_d12_*.npz"),
    (96, 12): unique("results/raw/fallback_dial_replication_*.npz"),
}


def mean_interval(values: np.ndarray, rng: np.random.Generator, draws: int = 10_000) -> tuple[float, float, float]:
    indices = rng.integers(0, values.size, size=(draws, values.size))
    means = values[indices].mean(axis=1)
    return float(values.mean()), *np.quantile(means, [0.025, 0.975]).tolist()


def main() -> None:
    csv_path = ROOT / "results/processed/fallback_routing_mechanisms_v1.csv"
    audit_path = ROOT / "results/processed/fallback_routing_mechanisms_audit_v1.json"
    figure_path = ROOT / "figures/fallback_routing_mechanisms_v1.png"
    for path in (csv_path, audit_path, figure_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    arrays = {}
    records = []
    for cell_index, ((n_context, n_features), path) in enumerate(INPUTS.items()):
        raw = np.load(path, allow_pickle=False)
        rho = raw["rho"].astype(float)
        task = raw["task_type"].astype(str)
        mechanism = raw["mechanism"].astype(int)
        selector = raw["combined_oof_probability"].argmax(axis=1)
        routing = raw["stable_expert_loss"] - raw["combined_expert_loss"]
        matched = raw["stable_expert_loss"] - raw["oracle_expert_loss"]
        for task_index, task_type in enumerate(("classification", "regression")):
            for mechanism_index, mechanism_name in enumerate(MECHANISMS):
                mask = (task == task_type) & np.isclose(rho, 1.0) & (mechanism == mechanism_index)
                key = (task_type, mechanism_name, n_context, n_features)
                arrays[key] = routing[mask].astype(float)
                rng = np.random.default_rng(10_000 + cell_index * 100 + task_index * 10 + mechanism_index)
                route_stats = mean_interval(routing[mask], rng)
                matched_stats = mean_interval(matched[mask], rng)
                records.append({
                    "task_type": task_type,
                    "mechanism": mechanism_name,
                    "n_context": n_context,
                    "n_features": n_features,
                    "tasks": int(mask.sum()),
                    "selector_accuracy": float(np.mean(selector[mask] == mechanism[mask])),
                    "shape_routing_gain": route_stats[0],
                    "shape_routing_ci_low": route_stats[1],
                    "shape_routing_ci_high": route_stats[2],
                    "matched_family_gain": matched_stats[0],
                    "matched_family_ci_low": matched_stats[1],
                    "matched_family_ci_high": matched_stats[2],
                })
    frame = pd.DataFrame(records)
    frame.to_csv(csv_path, index=False)

    contrasts = []
    for task_index, task_type in enumerate(("classification", "regression")):
        for mechanism_index, mechanism_name in enumerate(MECHANISMS):
            low = arrays[(task_type, mechanism_name, 64, 8)]
            high = arrays[(task_type, mechanism_name, 64, 12)]
            rng = np.random.default_rng(12_000 + task_index * 100 + mechanism_index)
            low_draw = low[rng.integers(0, low.size, size=(10_000, low.size))].mean(axis=1)
            high_draw = high[rng.integers(0, high.size, size=(10_000, high.size))].mean(axis=1)
            difference = high_draw - low_draw
            contrasts.append({
                "task_type": task_type,
                "mechanism": mechanism_name,
                "d12_minus_d8_routing_gain_at_n64": float(high.mean() - low.mean()),
                "ci_low": float(np.quantile(difference, 0.025)),
                "ci_high": float(np.quantile(difference, 0.975)),
            })
    audit = {
        "scope": "post-hoc all-family decomposition at rho=1",
        "dimension_contrasts": contrasts,
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False)
    x = np.arange(len(MECHANISMS))
    width = 0.19
    cells = list(INPUTS)
    for axis, task_type in zip(axes, ("classification", "regression"), strict=True):
        for cell_index, (n_context, n_features) in enumerate(cells):
            cell = frame[
                (frame["task_type"] == task_type)
                & (frame["n_context"] == n_context)
                & (frame["n_features"] == n_features)
            ].set_index("mechanism").loc[list(MECHANISMS)]
            axis.bar(
                x + (cell_index - 1.5) * width, cell["shape_routing_gain"], width,
                label=f"n={n_context}, d={n_features}",
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(x, MECHANISMS, rotation=30, ha="right")
        axis.set_ylabel("Stable-mixture loss minus shape-routed loss")
        axis.set_title(task_type.capitalize())
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("rho=1 routing gain by frozen mechanism family")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
