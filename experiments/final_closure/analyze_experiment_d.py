"""Analyze the finite coupling-mechanism decomposition (Experiment D)."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import closure_core as core
from analysis_utils import (
    dataset_cluster_bootstrap, full_factor_components, markdown_table,
    squared_residual, write_summary,
)
from closure_designs import mechanism_design, rows_to_ids


OUT = core.HERE / "summaries"
TABLES = core.HERE / "tables"
FIGURES = core.HERE / "figures"
DRAWS = 512


def analyze_cell(path) -> tuple[list[dict], dict]:
    manifest = json.loads((path / "manifest.json").read_text())
    predictions = np.load(path / "test_predictions.npy", mmap_mode="r")
    cards = tuple(int(value) for value in manifest["finite_cards"])
    if len(predictions) != int(np.prod(cards)):
        raise AssertionError("D tensor is not the declared complete finite product")
    reference = np.asarray(predictions, dtype=np.float64).mean(axis=0)
    rng = np.random.default_rng(
        core.stable_seed("D-analysis", manifest["dataset"], manifest["split_seed"], manifest["model"])
    )
    rows = []
    for method in core.CONFIG["experiment_d"]["methods"]:
        residuals = np.empty(DRAWS)
        for draw in range(DRAWS):
            design = mechanism_design(method, cards, rng)
            ids = rows_to_ids(design, cards)
            residuals[draw] = squared_residual(np.asarray(predictions[ids]).mean(axis=0), reference)
        rows.append({
            "dataset": manifest["dataset"], "split_seed": manifest["split_seed"],
            "model": manifest["model"], "task": manifest["task"], "method": method,
            "budget": 16, "residual_mean": float(residuals.mean()),
            "residual_median": float(np.median(residuals)),
            "residual_se": float(residuals.std(ddof=1) / np.sqrt(DRAWS)),
        })
    energies = full_factor_components(np.asarray(predictions, dtype=np.float64), cards)
    total = float(np.mean((np.asarray(predictions) - reference) ** 2))
    component = {
        "dataset": manifest["dataset"], "split_seed": manifest["split_seed"],
        "model": manifest["model"], "task": manifest["task"],
        "total_variance": total,
        "schema_only_mass": float(sum(value for subset, value in energies.items() if set(subset).issubset({0, 1, 2}))),
        "initialization_main_mass": float(energies.get((3,), 0.0)),
        "order_main_mass": float(energies.get((4,), 0.0)),
        "schema_initialization_mass": float(sum(value for subset, value in energies.items() if 3 in subset and bool(set(subset) & {0, 1, 2}) and 4 not in subset)),
        "schema_order_mass": float(sum(value for subset, value in energies.items() if 4 in subset and bool(set(subset) & {0, 1, 2}) and 3 not in subset)),
        "initialization_order_mass": float(sum(value for subset, value in energies.items() if 3 in subset and 4 in subset and not bool(set(subset) & {0, 1, 2}))),
        "joint_higher_mass": float(sum(value for subset, value in energies.items() if 3 in subset and 4 in subset and bool(set(subset) & {0, 1, 2}))),
        "fanova_reconstruction_error": total - float(sum(energies.values())),
    }
    return rows, component


def main() -> None:
    manifests = sorted((core.RAW / "experiment_d").glob("*/manifest.json"))
    expected = (
        len(core.CONFIG["experiment_d"]["datasets"])
        * len(core.CONFIG["split_seeds"])
        * len(core.CONFIG["primary_models"])
    )
    if len(manifests) != expected:
        raise AssertionError(f"Experiment D missing cells {len(manifests)}/{expected}")
    rows = []; components = []
    for manifest in manifests:
        current, component = analyze_cell(manifest.parent)
        rows.extend(current); components.append(component)
    frame = pd.DataFrame(rows); component_frame = pd.DataFrame(components)
    iid = frame[frame["method"] == "none"].set_index(
        ["dataset", "split_seed", "model"]
    )["residual_mean"]
    frame = frame.join(iid.rename("iid_residual"), on=["dataset", "split_seed", "model"])
    frame["relative_reduction_vs_none"] = 1 - frame["residual_mean"] / frame["iid_residual"]
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "experiment_d_cells.csv", index=False)
    component_frame.to_csv(OUT / "experiment_d_fanova_components.csv", index=False)
    table_rows = []
    for method, group in frame.groupby("method"):
        source = group.groupby("dataset", as_index=False)["relative_reduction_vs_none"].mean()
        low, high = dataset_cluster_bootstrap(
            source, "relative_reduction_vs_none", draws=10000,
            seed=core.stable_seed("D-bootstrap", method) % (2**32),
        )
        table_rows.append({
            "method": method, "mean_residual": float(group["residual_mean"].mean()),
            "mean_relative_reduction_vs_none": float(source["relative_reduction_vs_none"].mean()),
            "median_relative_reduction_vs_none": float(group["relative_reduction_vs_none"].median()),
            "cell_wins": int((group["residual_mean"] < group["iid_residual"]).sum()),
            "cells": len(group), "clustered_95_low": low, "clustered_95_high": high,
        })
    table = pd.DataFrame(table_rows).sort_values("mean_residual")
    TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES / "table_D_coupling_ablation.csv", index=False)
    markdown_table(table, TABLES / "table_D_coupling_ablation.md")
    summary = {
        "status": "complete", "cells": expected, "estimator_draws": DRAWS,
        "best_method_by_mean_residual": str(table.iloc[0]["method"]),
        "method_means": table.set_index("method")["mean_residual"].to_dict(),
        "mean_fanova_components": component_frame.select_dtypes(include=[np.number]).mean().to_dict(),
    }
    write_summary(OUT / "experiment_d_summary.json", summary)

    FIGURES.mkdir(parents=True, exist_ok=True)
    plot = table.sort_values("mean_residual")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(plot["method"], plot["mean_residual"])
    ax.set_yscale("log"); ax.set(ylabel="B=16 quotient residual", xlabel="balanced dimensions")
    ax.tick_params(axis="x", rotation=45); fig.tight_layout()
    fig.savefig(FIGURES / "figure_10_coupling_mechanism.png", dpi=180)
    fig.savefig(FIGURES / "figure_10_coupling_mechanism.pdf"); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
