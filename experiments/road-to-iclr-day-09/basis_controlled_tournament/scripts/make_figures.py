#!/usr/bin/env python3
"""Generate the eight figures required by the tournament brief."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "results" / "processed"
FIGURES = ROOT / "figures"


def save(figure: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"{stem}.png", dpi=180, bbox_inches="tight", facecolor="white")
    figure.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def label_points(axis: plt.Axes, frame: pd.DataFrame, x: str, y: str) -> None:
    for row in frame.itertuples():
        axis.annotate(
            str(row.method).replace("+DataInit", "+DI").replace("HybridSpectral", "HS"),
            (getattr(row, x), getattr(row, y)),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )


def figure_1() -> None:
    frame = pd.read_csv(PROCESSED / "prospective_method_summary.csv")
    frame = frame[frame["method"] != "Raw"].copy()
    frame["task_cost_percent"] = 100 * frame["median_relative_task_change"]
    frame["reduction_percent"] = 100 * frame["median_disagreement_reduction"]
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    colors = {"optimizer": "#0072B2", "interface": "#D55E00", "hybrid_prediction_mixture": "#009E73"}
    for track, part in frame.groupby("track"):
        ax.scatter(
            part["task_cost_percent"], part["reduction_percent"], s=75,
            label=track.replace("_", " "), color=colors.get(track, "#666666"), zorder=3,
        )
    offsets = {
        "GramAnchor": (5, 9),
        "BlockAdam+DataInit[equal-HPO]": (-158, -10),
        "Raw+GramAnchor@0.75": (5, 5),
    }
    for row in frame.itertuples():
        ax.annotate(
            row.method,
            (row.task_cost_percent, row.reduction_percent),
            xytext=offsets.get(row.method, (5, 5)),
            textcoords="offset points",
            fontsize=7,
        )
    ax.axvline(1, color="0.4", linestyle="--", linewidth=1, label="1% cost gate")
    ax.axhline(70, color="0.4", linestyle=":", linewidth=1, label="70% reduction gate")
    ax.set(xlabel="Median prospective task cost (%)", ylabel="Median disagreement reduction (%)", title="Prospective robustness–performance frontier")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="best")
    save(fig, "figure_1_method_pareto_frontier")


def figure_2() -> None:
    rows = pd.read_csv(PROCESSED / "mechanism_audit.csv")
    final = rows[rows["epoch"] == rows["epoch"].max()]
    order = ["AdamW", "BlockScalarAdam", "BlockAdam", "MatrixAdam", "SGD"]
    values = final.groupby("method")["disagreement"].median().reindex(order)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.bar(values.index, values.values, color=["#999999", "#56B4E9", "#0072B2", "#CC79A7", "#E69F00"])
    ax.set_yscale("log")
    ax.set(ylabel="Median matched-pair disagreement (log scale)", title="Optimizer comparison at the final matched epoch")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.2)
    save(fig, "figure_2_optimizer_comparison")


def figure_3() -> None:
    rows = pd.read_csv(PROCESSED / "mechanism_audit.csv")
    summary = rows.groupby(["method", "epoch"], as_index=False)["disagreement"].median()
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for method, part in summary.groupby("method"):
        ax.plot(part["epoch"], part["disagreement"].clip(lower=1e-12), marker="o", label=method)
    ax.set_yscale("log")
    ax.axhline(1e-5, color="0.3", linestyle="--", linewidth=1, label="equivariance tolerance")
    ax.set(xlabel="Epoch", ylabel="Median matched-function disagreement", title="Matched-function equivalence through training")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    save(fig, "figure_3_matched_function_over_epochs")


def figure_4() -> None:
    rows = pd.read_csv(PROCESSED / "development_method_summary.csv")
    wanted = ["Raw", "PCA", "GramAnchor", "GramDistance", "NystromGram", "HybridSpectral-t0.05"]
    frame = rows[(rows["track"] == "representation") & rows["method"].isin(wanted)].copy()
    frame["task_cost_percent"] = 100 * frame["median_relative_task_change"]
    frame["reduction_percent"] = 100 * frame["median_disagreement_reduction"]
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.scatter(frame["task_cost_percent"], frame["reduction_percent"], s=80, color="#D55E00")
    offsets = {
        "Raw": (5, 5), "GramAnchor": (5, 6), "HybridSpectral-t0.05": (-20, 8),
        "NystromGram": (-18, -12), "PCA": (5, 5), "GramDistance": (-72, 6),
    }
    for row in frame.itertuples():
        ax.annotate(
            row.method.replace("HybridSpectral-t0.05", "HybridSpectral"),
            (row.task_cost_percent, row.reduction_percent),
            xytext=offsets.get(row.method, (5, 5)),
            textcoords="offset points",
            fontsize=7,
        )
    ax.axvline(1, color="0.4", linestyle="--", linewidth=1)
    ax.axhline(70, color="0.4", linestyle=":", linewidth=1)
    ax.set(xlabel="Development median task cost (%)", ylabel="Disagreement reduction (%)", title="Invariant representation interfaces")
    ax.grid(alpha=0.2)
    save(fig, "figure_4_representation_methods")


def figure_5() -> None:
    frame = pd.read_csv(PROCESSED / "natural_basis_summary.csv")
    labels = frame["model"] + "\n" + frame["pair"].str.replace("_", " ")
    values = 100 * frame["median_disagreement_reduction"]
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(frame)), 5.0))
    bars = ax.bar(np.arange(len(frame)), values, color="#009E73")
    for bar, cost in zip(bars, frame["median_relative_task_change"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"cost {100*cost:.1f}%", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(np.arange(len(frame)), labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, max(110, float(values.max()) + 18))
    ax.set(ylabel="Natural-basis disagreement reduction (%)", title="Local/spectral-hat and one-hot/Helmert validation")
    ax.grid(axis="y", alpha=0.2)
    save(fig, "figure_5_natural_basis_results")


def figure_6() -> None:
    prospective = pd.read_csv(PROCESSED / "prospective_method_summary.csv")
    development = pd.read_csv(PROCESSED / "development_all_method_summary.csv")
    config = json.loads((ROOT / "configs" / "FINALIST_CONFIGS.json").read_text())
    source = {item["method_id"]: item.get("development_method", item["method_id"]) for item in config["finalists"]}
    records = []
    for method, development_method in source.items():
        if method not in set(prospective["method"]) or development_method not in set(development["method"]):
            continue
        records.append(
            {
                "method": method,
                "development": float(development.loc[development["method"] == development_method, "median_disagreement_reduction"].iloc[0]),
                "prospective": float(prospective.loc[prospective["method"] == method, "median_disagreement_reduction"].iloc[0]),
            }
        )
    frame = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.scatter(100 * frame["development"], 100 * frame["prospective"], s=90, color="#0072B2")
    maximum = max(100.0, float(100 * frame[["development", "prospective"]].max().max()) + 5)
    ax.plot([0, maximum], [0, maximum], color="0.5", linestyle="--", linewidth=1)
    offsets = {
        "GramAnchor": (5, 12),
        "BlockAdam+DataInit[equal-HPO]": (-175, -13),
        "Raw+GramAnchor@0.75": (5, 5),
    }
    for row in frame.itertuples():
        ax.annotate(
            row.method,
            (100 * row.development, 100 * row.prospective),
            xytext=offsets.get(row.method, (5, 5)),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set(xlim=(0, maximum), ylim=(0, maximum), xlabel="Development reduction (%)", ylabel="NEW prospective reduction (%)", title="Development-to-prospective transfer")
    ax.grid(alpha=0.2)
    save(fig, "figure_6_development_vs_prospective")


def figure_7() -> None:
    units = pd.read_csv(PROCESSED / "prospective_units.csv")
    test = units[(units["split"] == "test") & (units["method"] != "Raw")].copy()
    pivot = test.groupby(["method", "model"])["predictive_rank"].median().unstack()
    fig, ax = plt.subplots(figsize=(8.0, max(3.2, 0.6 * len(pivot))))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis_r")
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", color="white" if value > np.nanmedian(pivot.to_numpy()) else "black", fontsize=8)
    fig.colorbar(image, ax=ax, label="Median predictive rank (lower is better)")
    ax.set_title("Finalist predictive ranks by model family")
    save(fig, "figure_7_per_model_family_ranks")


def figure_8() -> None:
    frame = pd.read_csv(PROCESSED / "prospective_method_summary.csv")
    frame = frame[frame["method"] != "Raw"].sort_values("median_worst_orbit_gain", ascending=False)
    degradation = -100 * frame["median_worst_orbit_gain"]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    colors = np.where(degradation <= 0, "#009E73", "#D55E00")
    ax.bar(frame["method"], degradation, color=colors)
    ax.axhline(0, color="0.2", linewidth=1)
    ax.set(ylabel="Median worst-orbit task degradation vs raw (%)", title="Worst-orbit predictive performance on NEW prospective data")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.2)
    save(fig, "figure_8_worst_orbit_task_performance")


def main() -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    for function in (figure_1, figure_2, figure_3, figure_4, figure_5, figure_6, figure_7, figure_8):
        function()
    print(f"wrote 8 PNG and 8 PDF figures to {FIGURES}")


if __name__ == "__main__":
    main()
