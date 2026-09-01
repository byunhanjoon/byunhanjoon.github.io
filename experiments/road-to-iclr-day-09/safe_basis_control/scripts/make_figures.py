#!/usr/bin/env python3
"""Generate the eight critical Safe Basis Control figures as PNG and PDF."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "results" / "processed"
FIGURES = ROOT / "figures"

COLORS = {
    "GramAnchor-m16": "#d95f02",
    "GramAnchor": "#d95f02",
    "Raw+GramAnchor@0.75": "#7570b3",
    "SafeGram-t01": "#1b9e77",
    "SafeRankGram-t01": "#1f78b4",
    "RankAdaptiveGram": "#e6ab02",
    "Raw": "#666666",
}


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def figure1() -> None:
    summary = pd.read_csv(PROCESSED / "prospective_aggregate.csv")
    fig, ax = plt.subplots(figsize=(10, 7))
    offsets = {
        "GramAnchor-m16": (7, 7, "left"),
        "RankAdaptiveGram": (-7, -13, "right"),
        "PCA-canonicalization": (-7, -13, "right"),
        "Raw+GramAnchor@0.5": (7, -14, "left"),
        "Raw+GramAnchor@0.75": (7, 7, "left"),
        "SafeGram-t01": (7, -14, "left"),
        "SafeRankGram-t01": (7, 7, "left"),
        "Raw": (7, 7, "left"),
    }
    for row in summary.itertuples(index=False):
        ax.scatter(row.median_C, row.median_disagreement_reduction, s=85, color=COLORS.get(row.method, "#999999"), edgecolor="white", linewidth=0.8, zorder=3)
        dx, dy, alignment = offsets.get(row.method, (7, 7, "left"))
        ax.annotate(row.method, (row.median_C, row.median_disagreement_reduction), xytext=(dx, dy), textcoords="offset points", fontsize=8, ha=alignment)
    ax.axvline(0.01, color="#444444", linestyle="--", linewidth=1, label="median C safety gate")
    ax.axhline(0.70, color="#444444", linestyle=":", linewidth=1, label="70% control target")
    ax.set(xlabel="Median normalized excess risk C (lower is safer)", ylabel="Median disagreement reduction", title="Figure 1 — Basis control versus predictive safety")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.2)
    save(fig, "figure_1_reduction_vs_normalized_excess_risk")


def figure2() -> None:
    units = pd.read_csv(PROCESSED / "prospective_units.csv")
    methods = ["GramAnchor-m16", "Raw+GramAnchor@0.75", "SafeGram-t01", "SafeRankGram-t01"]
    fig, ax = plt.subplots(figsize=(10, 7))
    for method in methods:
        # The safety question is the harmful right tail.  A few near-zero
        # denominator cells produce enormous *negative* C (large benefits),
        # so clip only that benign tail to keep the critical [0, .20] region
        # legible without dropping any unit from the ECDF.
        values = np.sort(np.maximum(units[units.method == method]["normalized_excess_risk"].to_numpy(float), -0.05))
        y = np.arange(1, len(values) + 1) / len(values)
        ax.step(values, y, where="post", linewidth=2.2, label=method, color=COLORS[method])
    ax.axvline(0.0, color="#777777", linewidth=0.8)
    ax.axvline(0.05, color="#444444", linestyle="--", linewidth=1, label="p95 target")
    ax.axvline(0.20, color="#b2182b", linestyle=":", linewidth=1, label="catastrophic threshold")
    ax.set(xlim=(-0.055, 0.205), xlabel="Normalized excess risk C (benefits below −0.05 clipped)", ylabel="Empirical CDF across dataset × model units", title="Figure 2 — Tail distribution of predictive cost")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    save(fig, "figure_2_tail_cdf")


def figure3() -> None:
    cells = pd.read_csv(PROCESSED / "prospective_cells.csv")
    cells = cells[(cells.split == "test") & cells.method.isin(["SafeGram-t01", "SafeRankGram-t01"])]
    counts = cells.groupby(["method", "alpha"]).size().unstack(fill_value=0).reindex(columns=[0, 0.25, 0.5, 0.75, 1.0], fill_value=0)
    counts = counts.div(counts.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(5)
    width = 0.36
    for index, method in enumerate(counts.index):
        ax.bar(x + (index - 0.5) * width, counts.loc[method], width, label=method, color=COLORS[method])
    ax.set_xticks(x, ["0", ".25", ".5", ".75", "1"])
    ax.set(xlabel="Validation-selected alpha", ylabel="Fraction of prospective seed cells", title="Figure 3 — Interpretable fallback and control levels")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    save(fig, "figure_3_selected_alpha_histogram")


def figure4() -> None:
    frame = pd.read_csv(PROCESSED / "failure_diagnosis.csv")
    # The automatically selected controlled-MLP failures are the diagnostic
    # Steel Plates catastrophe.  Pooling them with TabM/TabICL cells hides the
    # exact generalization shift that this figure is required to expose.
    frame = frame[(frame.dataset == "steel-plates-fault") & (frame.model == "controlled_mlp")].copy()
    methods = ["Raw", "GramAnchor", "RankAdaptiveGram", "SafeGram-t01"]
    splits = ["train", "validation", "test"]
    values = np.array([[frame[frame.method == method][f"{split}_error"].median() for split in splits] for method in methods])
    fig, ax = plt.subplots(figsize=(11, 7))
    x = np.arange(len(splits))
    width = 0.19
    for index, method in enumerate(methods):
        ax.bar(x + (index - 1.5) * width, values[index], width, label=method, color=COLORS.get(method, "#999999"))
    ax.set_xticks(x, ["Training", "Validation", "Test"])
    ax.set(ylabel="Median log loss (controlled MLP, seeds 0–1)", title="Figure 4 — Steel Plates: reconstructible features, altered generalization")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.2)
    save(fig, "figure_4_steel_plates_failure_diagnosis")


def figure5() -> None:
    cells = pd.read_csv(PROCESSED / "rank_screen_cells.csv")
    blocks = pd.read_csv(PROCESSED / "rank_block_diagnostics.csv")
    test = cells[cells.split == "test"].groupby(["config_id", "anchor_rule", "relative_threshold"], as_index=False).agg(median_C=("normalized_excess_risk", "median"), median_dimension=("total_coordinate_dimension", "median"))
    anchors = blocks.groupby("config_id", as_index=False).agg(median_anchor_count=("anchor_count", "median"))
    plot = test.merge(anchors, on="config_id")
    fig, ax = plt.subplots(figsize=(10, 7))
    markers = {"rank": "o", "rank_plus_one": "s", "double_rank_capped_16": "^", "fixed_16": "D"}
    for rule, group in plot.groupby("anchor_rule"):
        ax.scatter(group.median_anchor_count, group.median_C, s=75, marker=markers.get(rule, "o"), label=rule.replace("_", " "), alpha=0.85)
    ax.axhline(0.01, color="#444444", linestyle="--", linewidth=1)
    ax.set(xlabel="Median anchors per feature block", ylabel="Median test normalized excess risk C", title="Figure 5 — Rank-adaptive anchor count versus performance")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    save(fig, "figure_5_rank_anchor_count_vs_performance")


def figure6() -> None:
    rotations = pd.read_csv(PROCESSED / "embedding_main_rotation_cells.csv")
    plot = rotations[(rotations.split == "test") & (rotations.condition == "rotated")].groupby(["embedding", "model"], as_index=False).agg(disagreement=("disagreement", "median"), task_effect=("task_effect", "median"), best_basis_effect=("task_effect", "min"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for embedding, group in plot.groupby("embedding"):
        axes[0].plot(group.model, group.disagreement, marker="o", linewidth=2, label=embedding)
        axes[1].plot(group.model, group.task_effect, marker="o", linewidth=2, label=embedding)
    axes[0].set(ylabel="Median original–rotated prediction disagreement", title="Basis sensitivity")
    axes[1].axhline(0, color="#444444", linewidth=1)
    axes[1].set(ylabel="Median rotated minus original task loss", title="Task effect of rotation")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    fig.suptitle("Figure 6 — Basis sensitivity inside PLE/RBF numerical embeddings")
    save(fig, "figure_6_embedding_basis_sensitivity")


def figure7() -> None:
    units = pd.read_csv(PROCESSED / "embedding_dimension_units.csv")
    plot = units.groupby(["embedding", "k"], as_index=False).agg(disagreement=("disagreement", "median"), task_effect=("task_effect", "median"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for embedding, group in plot.groupby("embedding"):
        axes[0].plot(group.k, group.disagreement, marker="o", linewidth=2, label=embedding)
        axes[1].plot(group.k, group.task_effect, marker="o", linewidth=2, label=embedding)
    axes[0].set(xlabel="Embedding dimension k", ylabel="Median basis disagreement", title="Sensitivity versus dimension")
    axes[1].axhline(0, color="#444444", linewidth=1)
    axes[1].set(xlabel="Embedding dimension k", ylabel="Median rotation task effect", title="Task effect versus dimension")
    for ax in axes:
        ax.set_xticks([4, 8, 16, 32])
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    fig.suptitle("Figure 7 — Embedding dimension ablation")
    save(fig, "figure_7_embedding_dimension_vs_disagreement")


def figure8() -> None:
    gate = pd.read_csv(PROCESSED / "development_gate_summary.csv")
    rank = pd.read_csv(PROCESSED / "rank_development_summary.csv")
    prospective = pd.read_csv(PROCESSED / "prospective_aggregate.csv").set_index("method")
    development = {
        "GramAnchor-m16": float(gate[gate.method == "GramAnchor"].iloc[0].p95_C),
        "Raw+GramAnchor@0.75": float(gate[gate.method == "Raw+GramAnchor@0.75"].iloc[0].p95_C),
        "SafeGram-t01": float(gate[gate.method == "SafeGram-t01"].iloc[0].p95_C),
        "SafeRankGram-t01": float(rank[rank.method == "SafeRankGram-t01"].iloc[0].p95_C),
    }
    fig, ax = plt.subplots(figsize=(9, 7))
    limit = 0.0
    for method, dev in development.items():
        pro = float(prospective.loc[method, "p95_C"])
        limit = max(limit, dev, pro)
        ax.scatter(dev, pro, s=90, color=COLORS[method], edgecolor="white", linewidth=0.8)
        ax.annotate(method, (dev, pro), xytext=(6, 5), textcoords="offset points", fontsize=9)
    limit = max(limit * 1.12, 0.055)
    ax.plot([0, limit], [0, limit], color="#777777", linestyle="--", linewidth=1)
    ax.axhline(0.05, color="#b2182b", linestyle=":", linewidth=1)
    ax.axvline(0.05, color="#b2182b", linestyle=":", linewidth=1)
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set(xlim=(-0.005, limit), ylim=(-0.005, limit), xlabel="Development p95 C", ylabel="NEW prospective p95 C", title="Figure 8 — Development versus prospective tail safety")
    ax.grid(alpha=0.2)
    save(fig, "figure_8_development_vs_prospective_safety")


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    for function in (figure1, figure2, figure3, figure4, figure5, figure6, figure7, figure8):
        function()
    print("generated 8 PNG and 8 PDF figures")


if __name__ == "__main__":
    main()
