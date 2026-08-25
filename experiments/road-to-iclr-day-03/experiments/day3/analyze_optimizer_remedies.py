"""Analyze the optimizer-remedy screen and confirmation runs."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "day3" / "optimizer_remedies"
FIGURES = RESULTS / "figures"


def interval(values: pd.Series) -> tuple[float, float, float, float]:
    x = values.dropna().to_numpy(float)
    mean = float(x.mean()) if len(x) else math.nan
    std = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    radius = float(t.ppf(0.975, len(x) - 1) * std / math.sqrt(len(x))) if len(x) > 1 else 0.0
    return mean, std, mean - radius, mean + radius


def load_confirmation() -> pd.DataFrame:
    files = [
        "confirm_adult.csv",
        "confirm_diamond.csv",
        "confirm_california_mlp.csv",
        "confirm_california_resnet.csv",
    ]
    return pd.concat([pd.read_csv(RESULTS / name) for name in files], ignore_index=True)


def paired(frame: pd.DataFrame) -> pd.DataFrame:
    wide = frame.pivot_table(
        index=["dataset", "task", "model", "remedy", "seed"],
        columns="target_kappa",
        values="test_metric",
    ).reset_index()
    wide = wide.dropna(subset=[1.0, 3000.0])
    classification = wide.task.eq("binclass")
    wide["sensitivity"] = np.where(
        classification,
        100 * (wide[1.0] - wide[3000.0]),
        100 * (wide[3000.0] - wide[1.0]) / wide[1.0],
    )
    # Positive baseline_gain always means better than AdamW at kappa=1.
    baseline = (
        wide[wide.remedy.eq("adamw")][["dataset", "model", "seed", 1.0]]
        .rename(columns={1.0: "adamw_k1"})
    )
    wide = wide.merge(baseline, on=["dataset", "model", "seed"])
    wide["baseline_gain"] = np.where(
        classification,
        100 * (wide[1.0] - wide.adamw_k1),
        100 * (wide.adamw_k1 - wide[1.0]) / wide.adamw_k1,
    )
    return wide


def summarize(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, part in wide.groupby(["dataset", "task", "model", "remedy"]):
        sensitivity = interval(part.sensitivity)
        gain = interval(part.baseline_gain)
        rows.append({
            "dataset": keys[0],
            "task": keys[1],
            "model": keys[2],
            "remedy": keys[3],
            "seeds": len(part),
            "kappa1_mean": part[1.0].mean(),
            "kappa3000_mean": part[3000.0].mean(),
            "sensitivity_mean": sensitivity[0],
            "sensitivity_std": sensitivity[1],
            "sensitivity_ci_low": sensitivity[2],
            "sensitivity_ci_high": sensitivity[3],
            "baseline_gain_mean": gain[0],
            "baseline_gain_std": gain[1],
            "baseline_gain_ci_low": gain[2],
            "baseline_gain_ci_high": gain[3],
        })
    return pd.DataFrame(rows)


def screen_plot() -> None:
    frames = []
    for dataset in ("adult", "diamond"):
        frame = pd.read_csv(RESULTS / f"screen_{dataset}.csv")
        frames.append(paired(frame))
    screen = pd.concat(frames, ignore_index=True)
    means = screen.groupby(["dataset", "remedy"]).sensitivity.mean().unstack(0)
    order = means.mean(axis=1).sort_values().index
    means = means.loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
    for axis, dataset in zip(axes, ("adult", "diamond")):
        values = means[dataset]
        colors = ["#2a9d8f" if value <= 0.1 else "#e76f51" for value in values]
        axis.barh(np.arange(len(values)), values, color=colors)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_title(dataset)
        axis.set_xlabel("κ=1→3000 degradation\n(pp accuracy or % RMSE)")
        axis.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(np.arange(len(order)), order, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "screen_18_remedies.png", dpi=190)
    plt.close(fig)


def confirmation_plots(wide: pd.DataFrame, summary: pd.DataFrame) -> None:
    remedies = [
        "adamw",
        "whiten_sgd",
        "anchor_whiten_adamw",
        "anchor_whiten_sgd",
        "natural_hybrid_invariant_init",
        "natural_hybrid_invariant_init_lr01",
    ]
    labels = {
        "adamw": "AdamW",
        "whiten_sgd": "Whiten + SGD",
        "anchor_whiten_adamw": "Invariant canonical + AdamW",
        "anchor_whiten_sgd": "Invariant canonical + SGD",
        "natural_hybrid_invariant_init": "Natural first layer (lr=.03)",
        "natural_hybrid_invariant_init_lr01": "Natural first layer (lr=.01)",
    }
    groups = [(dataset, model) for dataset in ("adult", "california", "diamond") for model in ("mlp", "resnet")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=False)
    for axis, (dataset, model) in zip(axes.flat, groups):
        part = summary[(summary.dataset == dataset) & (summary.model == model)].set_index("remedy").reindex(remedies)
        values = part.sensitivity_mean
        low = values - part.sensitivity_ci_low
        high = part.sensitivity_ci_high - values
        colors = ["#264653" if remedy == "adamw" else "#2a9d8f" for remedy in remedies]
        axis.bar(np.arange(len(remedies)), values, yerr=np.vstack((low, high)), color=colors, capsize=2)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(f"{dataset} / {model}")
        axis.set_xticks(np.arange(len(remedies)), [labels[r] for r in remedies], rotation=35, ha="right", fontsize=7)
        axis.set_ylabel("Degradation (pp accuracy or % RMSE)")
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "confirmed_basis_sensitivity.png", dpi=190)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    markers = {"mlp": "o", "resnet": "s"}
    colors = {"adult": "#264653", "california": "#e9c46a", "diamond": "#e76f51"}
    focus = summary[summary.remedy.isin(remedies)]
    for _, row in focus.iterrows():
        ax.scatter(
            row.sensitivity_mean,
            row.baseline_gain_mean,
            marker=markers[row.model],
            color=colors[row.dataset],
            alpha=0.8,
            s=45,
        )
        if row.remedy in ("adamw", "anchor_whiten_adamw", "natural_hybrid_invariant_init"):
            ax.annotate(labels[row.remedy], (row.sensitivity_mean, row.baseline_gain_mean), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("κ sensitivity (lower is better)")
    ax.set_ylabel("κ=1 gain over AdamW (higher is better)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "remedy_sensitivity_performance_tradeoff.png", dpi=190)
    plt.close(fig)

    # Paired sensitivity-reduction fraction relative to AdamW.
    baseline = wide[wide.remedy.eq("adamw")][["dataset", "model", "seed", "sensitivity"]].rename(columns={"sensitivity": "adamw_sensitivity"})
    reductions = wide.merge(baseline, on=["dataset", "model", "seed"])
    reductions["sensitivity_reduction"] = 1 - reductions.sensitivity / reductions.adamw_sensitivity
    reductions.to_csv(RESULTS / "paired_sensitivity_reductions.csv", index=False)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    confirmation = load_confirmation()
    confirmation.to_csv(RESULTS / "confirmation_all.csv", index=False)
    wide = paired(confirmation)
    summary = summarize(wide)
    wide.to_csv(RESULTS / "confirmation_paired.csv", index=False)
    summary.to_csv(RESULTS / "confirmation_summary.csv", index=False)
    screen_plot()
    confirmation_plots(wide, summary)

    anchor = summary[summary.remedy.eq("anchor_whiten_adamw")]
    natural = summary[summary.remedy.eq("natural_hybrid_invariant_init")]
    adamw = summary[summary.remedy.eq("adamw")]
    payload = {
        "trained_confirmation_runs": len(confirmation),
        "dataset_model_pairs": len(anchor),
        "anchor_max_absolute_sensitivity": float(anchor.sensitivity_mean.abs().max()),
        "anchor_mean_baseline_gain": float(anchor.baseline_gain_mean.mean()),
        "natural_max_absolute_sensitivity": float(natural.sensitivity_mean.abs().max()),
        "natural_mean_baseline_gain": float(natural.baseline_gain_mean.mean()),
        "adamw_mean_sensitivity": float(adamw.sensitivity_mean.mean()),
        "anchor_mean_sensitivity": float(anchor.sensitivity_mean.mean()),
        "natural_mean_sensitivity": float(natural.sensitivity_mean.mean()),
        "failed_screen_runs": int(
            sum(
                frame.get("failure", pd.Series("", index=frame.index)).fillna("").ne("").sum()
                for frame in (
                    pd.read_csv(RESULTS / f"screen_{dataset}.csv")
                    for dataset in ("adult", "diamond")
                )
            )
        ),
    }
    (RESULTS / "remedy_summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
