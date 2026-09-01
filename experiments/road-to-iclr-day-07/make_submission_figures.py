"""Regenerate compact paper-only figures from authoritative closure tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments/final_closure/summaries/experiment_a_references.csv"
OUT = Path(__file__).resolve().parent / "paper/figures"


def main() -> None:
    frame = pd.read_csv(SOURCE)
    OUT.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.55), constrained_layout=True)

    distances = [
        frame["canonical_joint_distance"].to_numpy(),
        frame["canonical_coupled_distance"].to_numpy(),
        frame["joint_coupled_distance"].to_numpy(),
    ]
    labels = ["canonical–joint", "canonical–coupled", "joint–coupled"]
    colors = ["#4C78A8", "#E45756", "#F2A541"]
    rng = np.random.default_rng(20260830)
    for index, (values, color) in enumerate(zip(distances, colors), start=1):
        jitter = rng.normal(0.0, 0.055, size=len(values))
        axes[0].scatter(
            index + jitter,
            values,
            s=7,
            alpha=0.28,
            color=color,
            linewidths=0,
            rasterized=True,
        )
    box = axes[0].boxplot(
        distances,
        positions=np.arange(1, 4),
        widths=0.42,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.2},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set(facecolor=color, alpha=0.22, edgecolor=color)
    axes[0].set_yscale("log")
    axes[0].set_xticks(np.arange(1, 4), labels, rotation=18, ha="right")
    axes[0].set_ylabel("squared prediction distance")
    axes[0].set_title("(a) Symmetrization changes the target")
    axes[0].grid(axis="y", which="both", alpha=0.18)

    threshold = frame["canonical_joint_mc_noise_95"].to_numpy()
    observed = frame["canonical_joint_distance"].to_numpy()
    distinguished = frame["canonical_joint_distinguishable_from_mc"].astype(bool).to_numpy()
    axes[1].scatter(
        threshold[~distinguished],
        observed[~distinguished],
        s=13,
        alpha=0.52,
        color="#4C78A8",
        linewidths=0,
        label="within MC threshold",
    )
    axes[1].scatter(
        threshold[distinguished],
        observed[distinguished],
        s=19,
        alpha=0.9,
        color="#D62728",
        linewidths=0.25,
        edgecolors="white",
        label="exceeds threshold",
    )
    lower = min(threshold.min(), observed.min()) * 0.72
    upper = max(threshold.max(), observed.max()) * 1.35
    axes[1].plot([lower, upper], [lower, upper], color="black", linewidth=0.8, linestyle="--")
    axes[1].set(xscale="log", yscale="log", xlim=(lower, upper), ylim=(lower, upper))
    axes[1].set_xlabel("cell-specific 95% MC threshold")
    axes[1].set_ylabel("canonical–joint distance")
    axes[1].set_title("(b) 10/144 cells exceed MC noise")
    axes[1].legend(frameon=False, loc="lower right")
    axes[1].grid(which="both", alpha=0.18)

    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"target_shift_summary.{suffix}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
