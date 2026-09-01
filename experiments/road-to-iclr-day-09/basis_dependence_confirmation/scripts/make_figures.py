#!/usr/bin/env python3
"""Generate the eight protocol figures from audited processed results."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "results" / "processed"
FIGURES = ROOT / "figures"


def save(figure: plt.Figure, number: int, slug: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(FIGURES / f"figure_{number:02d}_{slug}.png", dpi=220, bbox_inches="tight")
    figure.savefig(FIGURES / f"figure_{number:02d}_{slug}.pdf", bbox_inches="tight")
    plt.close(figure)


def figure1() -> None:
    frame = pd.read_csv(PROCESSED / "replication_summary.csv")
    values = frame[frame["variant"] == "orthogonal_all"].pivot(
        index="dataset", columns="model", values="mean_disagreement"
    )
    order = values.mean(axis=1).sort_values(ascending=False).index
    figure, axis = plt.subplots(figsize=(8.4, 5.8))
    sns.heatmap(values.loc[order], cmap="mako", annot=True, fmt=".3f", linewidths=.3, ax=axis)
    axis.set_title("Orthogonal all-block prediction disagreement")
    axis.set_xlabel("Model"); axis.set_ylabel("Development dataset")
    save(figure, 1, "orthogonal_heatmap")


def figure2() -> None:
    summary = pd.read_csv(PROCESSED / "replication_summary.csv")
    worst = summary[summary["variant"] == "orthogonal_all"].sort_values("mean_disagreement").iloc[-1]
    bundle = ROOT / "results" / "raw" / "development" / "replication" / worst.model / worst.dataset / "seed_0"
    metrics = pd.read_csv(bundle / "metrics.csv")
    candidates = metrics[(metrics["split"] == "test") & (metrics["variant"] == "orthogonal_all")].copy()
    metric = "probability_rmse" if worst.problem_type == "classification" else "prediction_rmse_normalized"
    disagreement = candidates[metric].to_numpy()
    representation = candidates.iloc[int(np.nanargmax(disagreement))]["representation_id"]
    predictions = pd.read_csv(bundle / "predictions.csv.gz")
    reference = predictions[(predictions["split"] == "test") & (predictions["representation_id"] == "rbf_reference")]
    rotated = predictions[(predictions["split"] == "test") & (predictions["representation_id"] == representation)]
    columns = [column for column in predictions if column.startswith("prediction")]
    column = "prediction" if "prediction" in columns else columns[min(1, len(columns)-1)]
    joined = reference[["row_id", column]].merge(
        rotated[["row_id", column]], on="row_id", suffixes=("_original", "_rotated"), validate="one_to_one"
    )
    figure, axis = plt.subplots(figsize=(5.5, 5.2))
    axis.scatter(joined[f"{column}_original"], joined[f"{column}_rotated"], s=13, alpha=.55)
    plotted = joined[[f"{column}_original", f"{column}_rotated"]]
    limits = [float(plotted.min().min()), float(plotted.max().max())]
    axis.plot(limits, limits, color="black", linestyle="--", linewidth=1)
    axis.set(xlabel="Original-basis prediction", ylabel="Rotated-basis prediction",
             title=f"{worst.dataset} · {worst.model} · seed 0")
    save(figure, 2, "prediction_scatter")


def figure3() -> None:
    frame = pd.read_csv(PROCESSED / "mechanism_summary.csv")
    shown = ["ordinary_adamw", "matched_adamw", "matched_sgd_momentum", "matched_sgd_plain"]
    plot = frame[frame["condition"].isin(shown)].groupby(
        ["condition", "epoch_order"], as_index=False
    )["disagreement"].median()
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    sns.lineplot(data=plot, x="epoch_order", y="disagreement", hue="condition", marker="o", ax=axis)
    axis.set(xlabel="Training epoch", ylabel="Median prediction disagreement",
             title="Function matching and optimizer geometry")
    save(figure, 3, "mechanism_trajectory")


def figure4() -> None:
    frame = pd.read_csv(PROCESSED / "natural_summary.csv")
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    sns.boxplot(data=frame, x="family", y="disagreement", hue="model", showfliers=False, ax=axis)
    sns.stripplot(data=frame, x="family", y="disagreement", color="black", alpha=.32, size=2.5, ax=axis)
    axis.set(xlabel="Natural equivalent-basis family", ylabel="Prediction disagreement",
             title="Natural equivalent bases")
    save(figure, 4, "natural_basis_pairs")


def figure5() -> None:
    frame = pd.read_csv(PROCESSED / "repairs_summary.csv")
    frame = frame[frame["repair"] != "ORACLE INVERSE — NOT A METHOD"]
    plot = frame.groupby("repair", as_index=False).agg(
        disagreement=("disagreement", "median"), reduction=("disagreement_reduction", "median")
    ).sort_values("disagreement")
    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    sns.barplot(data=plot, x="repair", y="disagreement", color="#4878CF", ax=axis)
    axis.tick_params(axis="x", rotation=25)
    axis.set(xlabel="Repair", ylabel="Median disagreement", title="Non-oracle repair comparison")
    axis.text(.99, .98, "AnchorCanonical excluded: rank deficient", transform=axis.transAxes,
              ha="right", va="top", fontsize=8, color="dimgray")
    save(figure, 5, "repair_comparison")


def figure6() -> None:
    frame = pd.read_csv(PROCESSED / "repairs_summary.csv")
    frame = frame[frame["repair"] != "ORACLE INVERSE — NOT A METHOD"]
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    sns.scatterplot(data=frame, x="relative_task_change", y="disagreement_reduction",
                    hue="repair", style="model", alpha=.72, ax=axis)
    axis.axhline(.70, color="grey", linestyle="--", linewidth=1)
    axis.axvline(.01, color="grey", linestyle="--", linewidth=1)
    axis.set(xlabel="Relative task-error change", ylabel="Disagreement reduction",
             title="Task performance–invariance tradeoff")
    save(figure, 6, "performance_invariance_tradeoff")


def figure7() -> None:
    development = pd.read_csv(PROCESSED / "repairs_summary.csv")
    prospective = pd.read_csv(PROCESSED / "prospective_repairs_summary.csv")
    development["panel"] = "development"; prospective["panel"] = "prospective"
    frame = pd.concat([development, prospective], ignore_index=True, sort=False)
    common = ["raw", "standardization", "whitening", "pca_canonical", "ORACLE INVERSE — NOT A METHOD"]
    frame = frame[frame["repair"].isin(common)]
    plot = frame.groupby(["panel", "repair"], as_index=False)["disagreement_reduction"].median()
    figure, axis = plt.subplots(figsize=(8.6, 4.8))
    sns.barplot(data=plot, x="repair", y="disagreement_reduction", hue="panel", order=common, ax=axis)
    axis.tick_params(axis="x", rotation=25)
    axis.set(xlabel="Repair", ylabel="Median disagreement reduction",
             title="Development versus prospective holdout")
    save(figure, 7, "development_vs_prospective")


def figure8() -> None:
    random = pd.read_csv(PROCESSED / "replication_summary.csv")
    random = random[random["variant"] == "orthogonal_all"][["dataset", "model", "mean_disagreement"]].rename(
        columns={"mean_disagreement": "disagreement"}
    )
    random["basis_type"] = "random orthogonal"
    natural = pd.read_csv(PROCESSED / "natural_summary.csv")[["dataset", "model", "disagreement"]]
    natural["basis_type"] = "natural equivalent"
    frame = pd.concat([random, natural], ignore_index=True)
    figure, axis = plt.subplots(figsize=(6.4, 4.8))
    sns.boxplot(data=frame, x="basis_type", y="disagreement", showfliers=False, ax=axis)
    sns.stripplot(data=frame, x="basis_type", y="disagreement", color="black", alpha=.28, size=2.5, ax=axis)
    axis.set(xlabel="Basis perturbation", ylabel="Prediction disagreement",
             title="Random versus natural equivalent bases")
    save(figure, 8, "random_vs_natural")


def main() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    functions = [figure1, figure2, figure3, figure4, figure5, figure6, figure7, figure8]
    missing = []
    for number, function in enumerate(functions, start=1):
        try:
            function()
        except FileNotFoundError as error:
            missing.append({"figure": number, "missing": str(error.filename)})
    if missing:
        raise RuntimeError(f"processed inputs are incomplete: {missing}")


if __name__ == "__main__":
    main()
