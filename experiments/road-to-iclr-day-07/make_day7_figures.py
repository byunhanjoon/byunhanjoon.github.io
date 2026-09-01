#!/usr/bin/env python3
"""Paper-style figures for the Day-7 learned structured-PFN screen."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE / "results" / "learned_pfn"
OUT = ROOT / "figures"


def annotate_heatmap(axis, values: np.ndarray, threshold: float | None = None) -> None:
    if threshold is None:
        threshold = float(np.nanmean(values))
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            color = "white" if value > threshold else "black"
            axis.text(column, row, f"{value:.3f}", ha="center", va="center", color=color, fontsize=9)


def phase_figure(cells: pd.DataFrame) -> None:
    matched = cells[cells.true_prior == 0.5]
    phase = matched.groupby(["variant", "scale", "noise"], as_index=False).mean(numeric_only=True)
    structured = phase[phase.variant == "structured"].set_index(["scale", "noise"])
    control = phase[phase.variant == "set"].set_index(["scale", "noise"])
    metrics = [
        (control.mse_model - structured.mse_model, "Gain over geometry-free transformer", "viridis"),
        (structured.mse_model - structured.mse_bayes_mixture, "Regret to analytic Bayes rule", "magma_r"),
        (structured.trust_regime_auroc, "Implicit trust regime AUROC", "cividis"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.55), constrained_layout=True)
    for axis, (series, title, cmap) in zip(axes, metrics):
        table = series.unstack("noise").reindex(index=[0.3, 1.0, 3.0], columns=[0.1, 0.3, 1.0])
        values = table.to_numpy()
        image = axis.imshow(values, cmap=cmap, aspect="auto")
        annotate_heatmap(axis, values)
        axis.set_title(title, fontsize=11)
        axis.set_xticks(range(3), ["0.1", "0.3", "1.0"])
        axis.set_yticks(range(3), ["0.3", "1.0", "3.0"])
        axis.set_xlabel("Observation noise")
        axis.set_ylabel("Heat-kernel scale")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    fig.suptitle("A small transformer learns the optional-geometry phase boundary", fontsize=13)
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"learned_pfn_phase.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def prior_shift_figure(cells: pd.DataFrame) -> None:
    frame = cells[cells.variant == "structured"].groupby("true_prior").mean(numeric_only=True)
    methods = [
        ("mse_zero", "Ignore geometry", "#9aa0a6"),
        ("mse_always_smooth", "Always geometry", "#d97706"),
        ("mse_model", "Learned soft router", "#2563eb"),
        ("mse_bayes_mixture", "Analytic Bayes", "#059669"),
    ]
    priors = np.asarray([0.1, 0.5, 0.9])
    positions = np.arange(len(priors))
    width = 0.19
    fig, axis = plt.subplots(figsize=(7.4, 4.2), constrained_layout=True)
    for offset, (column, label, color) in enumerate(methods):
        axis.bar(
            positions + (offset - 1.5) * width,
            frame.loc[priors, column],
            width=width,
            label=label,
            color=color,
        )
    axis.set_xticks(positions, [f"{prior:.1f}" for prior in priors])
    axis.set_xlabel("True smooth-task probability at deployment")
    axis.set_ylabel("Query MSE (lower is better)")
    axis.set_title("The learned router adapts from context, but retains pretraining-prior bias")
    axis.legend(frameon=False, ncol=2)
    axis.spines[["top", "right"]].set_visible(False)
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"learned_pfn_prior_shift.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(ROOT / "cells.csv")
    phase_figure(cells)
    prior_shift_figure(cells)
    print(OUT)


if __name__ == "__main__":
    main()
