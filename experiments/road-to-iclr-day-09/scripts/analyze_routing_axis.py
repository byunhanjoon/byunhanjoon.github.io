#!/usr/bin/env python3
"""Post-hoc 2x2 context-size/feature-count routing diagnostic at rho=1."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def unique(pattern: str) -> Path:
    paths = sorted(ROOT.glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(f"expected one {pattern}, found {paths}")
    return paths[0]


def bootstrap(samples: dict[tuple[int, int], np.ndarray], draws: int, seed: int) -> dict[str, tuple[float, float, float]]:
    rng = np.random.default_rng(seed)
    keys = ((64, 8), (96, 8), (64, 12), (96, 12))
    means = {key: np.empty(draws) for key in keys}
    for start in range(0, draws, 500):
        stop = min(start + 500, draws)
        for key in keys:
            values = samples[key]
            indices = rng.integers(0, values.size, size=(stop - start, values.size))
            means[key][start:stop] = values[indices].mean(axis=1)
    contrasts = {
        "context_effect_at_d8": means[(96, 8)] - means[(64, 8)],
        "feature_effect_at_n64": means[(64, 12)] - means[(64, 8)],
        "corner_interaction": means[(96, 12)] - means[(96, 8)] - means[(64, 12)] + means[(64, 8)],
    }
    return {
        name: (float(values.mean()), *np.quantile(values, [0.025, 0.975]).tolist())
        for name, values in contrasts.items()
    }


def main() -> None:
    inputs = {
        (64, 8): ROOT / "results/raw/e1_primary_b779842a24_t420_n64.npz",
        (96, 8): unique("results/raw/fallback_routing_axis_n96_d8_*.npz"),
        (64, 12): unique("results/raw/fallback_routing_axis_n64_d12_*.npz"),
        (96, 12): unique("results/raw/fallback_dial_replication_*.npz"),
    }
    output_csv = ROOT / "results/processed/fallback_routing_axis_v1.csv"
    output_json = ROOT / "results/processed/fallback_routing_axis_audit_v1.json"
    output_figure = ROOT / "figures/fallback_routing_axis_v1.png"
    for output in (output_csv, output_json, output_figure):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")

    records = []
    utilities: dict[str, dict[tuple[int, int], np.ndarray]] = {"classification": {}, "regression": {}}
    for (n_context, n_features), path in inputs.items():
        raw = np.load(path, allow_pickle=False)
        rho = raw["rho"].astype(float)
        task = raw["task_type"].astype(str)
        mechanism = raw["mechanism"].astype(int)
        selected = raw["combined_oof_probability"].argmax(axis=1)
        routing = raw["stable_expert_loss"] - raw["combined_expert_loss"]
        matched = raw["stable_expert_loss"] - raw["oracle_expert_loss"]
        for task_type in utilities:
            mask = (task == task_type) & np.isclose(rho, 1.0)
            values = routing[mask].astype(float)
            utilities[task_type][(n_context, n_features)] = values
            rng = np.random.default_rng(8100 + n_context + n_features)
            indices = rng.integers(0, values.size, size=(10_000, values.size))
            route_draw = values[indices].mean(axis=1)
            match_draw = matched[mask][indices].mean(axis=1)
            records.append({
                "task_type": task_type,
                "n_context": n_context,
                "n_features": n_features,
                "tasks": int(mask.sum()),
                "mechanism_selector_accuracy": float(np.mean(selected[mask] == mechanism[mask])),
                "shape_routing_gain": float(values.mean()),
                "shape_routing_ci_low": float(np.quantile(route_draw, 0.025)),
                "shape_routing_ci_high": float(np.quantile(route_draw, 0.975)),
                "matched_family_gain": float(matched[mask].mean()),
                "matched_family_ci_low": float(np.quantile(match_draw, 0.025)),
                "matched_family_ci_high": float(np.quantile(match_draw, 0.975)),
                "raw_bundle": str(path.relative_to(ROOT)),
            })
    frame = pd.DataFrame(records)
    frame.to_csv(output_csv, index=False)
    contrasts = {
        task: bootstrap(sample, 10_000, 9000 + index)
        for index, (task, sample) in enumerate(utilities.items())
    }
    audit = {
        "scope": "post-hoc factorial axis diagnostic; independent seeds per cell",
        "positive_gain_means": "shape-informed routing lowers loss",
        "contrasts": contrasts,
    }
    output_json.write_text(json.dumps(audit, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for axis, task_type in zip(axes, ("classification", "regression"), strict=True):
        cell = frame[frame["task_type"] == task_type]
        matrix = np.empty((2, 2))
        for row, n_context in enumerate((64, 96)):
            for column, n_features in enumerate((8, 12)):
                matrix[row, column] = cell[
                    (cell["n_context"] == n_context) & (cell["n_features"] == n_features)
                ]["shape_routing_gain"].iloc[0]
        limit = max(abs(matrix.min()), abs(matrix.max()))
        image = axis.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit)
        for row in range(2):
            for column in range(2):
                axis.text(column, row, f"{matrix[row, column]:+.4f}", ha="center", va="center")
        axis.set_xticks((0, 1), (8, 12))
        axis.set_yticks((0, 1), (64, 96))
        axis.set_xlabel("Features")
        axis.set_ylabel("Context rows")
        axis.set_title(task_type.capitalize())
        fig.colorbar(image, ax=axis, shrink=0.75)
    fig.suptitle("rho=1 shape-routing gain: post-hoc regime-axis diagnostic")
    fig.tight_layout()
    fig.savefig(output_figure, dpi=180)
    plt.close(fig)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
