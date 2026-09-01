#!/usr/bin/env python3
"""Create versioned E1 figures from processed summaries; never mutates raw bundles."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--primary",
        type=Path,
        default=ROOT / "results/processed/e1_primary_b779842a24_t420_n64_summary.csv",
    )
    parser.add_argument(
        "--context-sweep",
        type=Path,
        default=ROOT / "results/processed/e1_context_sweep_b779842a24_t120_n32-64-128-256-512_summary.csv",
    )
    parser.add_argument("--tag", default="v1_1")
    return parser.parse_args()


def save_exclusive(fig: plt.Figure, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite figure: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = arguments()
    primary = pd.read_csv(args.primary)
    sweep = pd.read_csv(args.context_sweep)
    colors = {"classification": "#2667ff", "regression": "#ef476f"}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for task_type, group in primary.groupby("task_type"):
        group = group.sort_values("rho")
        axes[0].plot(group.rho, group.mechanism_accuracy, marker="o", label=task_type, color=colors[task_type])
        axes[0].fill_between(
            group.rho,
            group.mechanism_accuracy_ci_low,
            group.mechanism_accuracy_ci_high,
            alpha=0.18,
            color=colors[task_type],
        )
        axes[1].plot(group.rho, group.marginal_query_utility, marker="o", label=task_type, color=colors[task_type])
        axes[1].fill_between(
            group.rho,
            group.marginal_query_utility_ci_low,
            group.marginal_query_utility_ci_high,
            alpha=0.18,
            color=colors[task_type],
        )
    axes[0].axhline(1 / 6, ls="--", lw=1, color="black", label="chance")
    axes[0].set(title="Marginal shape predicts mechanism", xlabel=r"dial $\rho$", ylabel="5-fold OOF accuracy")
    axes[1].axhline(0, ls="--", lw=1, color="black")
    axes[1].set(title="Shape adds query value beyond stable channel", xlabel=r"dial $\rho$", ylabel="stable loss − stable+shape loss")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.suptitle("PriorDial v1.1 development validation (420 tasks/cell, n=64, d=8)")
    fig.tight_layout()
    save_exclusive(fig, ROOT / f"figures/e1_priordial_phase_{args.tag}.png")

    contexts = sorted(sweep.context_size.unique())
    rhos = sorted(sweep.rho.unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), constrained_layout=True)
    extrema = float(np.max(np.abs(sweep.marginal_query_utility)))
    for axis, task_type in zip(axes, ("classification", "regression"), strict=True):
        cell = sweep[sweep.task_type == task_type].pivot(
            index="context_size", columns="rho", values="marginal_query_utility"
        ).loc[contexts, rhos]
        image = axis.imshow(cell.values, aspect="auto", cmap="coolwarm", vmin=-extrema, vmax=extrema)
        axis.set_xticks(range(len(rhos)), [f"{rho:g}" for rho in rhos])
        axis.set_yticks(range(len(contexts)), contexts)
        axis.set(xlabel=r"dial $\rho$", ylabel="context size", title=task_type)
        for row in range(cell.shape[0]):
            for column in range(cell.shape[1]):
                axis.text(column, row, f"{cell.iloc[row, column]:.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=axes, label="stable loss − stable+shape loss", shrink=0.85)
    fig.suptitle("PriorDial conditional marginal utility across context sizes (120 tasks/cell)")
    save_exclusive(fig, ROOT / f"figures/e1_context_surface_{args.tag}.png")


if __name__ == "__main__":
    main()

