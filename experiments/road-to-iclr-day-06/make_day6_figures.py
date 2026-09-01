"""Generate paper-oriented Day-6 trajectory, survival, and forecast figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATASET_LABELS = {
    "bank_marketing_subscription": "Bank",
    "credit_card_default": "Credit",
    "fremtpl_claim_count": "FreMTPL",
}
MODEL_LABELS = {"mlp": "MLP", "resnet": "ResNet", "ft_transformer": "FT-Transformer"}
COLORS = {"fp32": "#c53b3b", "iea64": "#2774ae"}


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def h3_trajectories(results: Path, output: Path) -> None:
    frame = pd.read_csv(results / "h3_trajectories.csv")
    datasets = list(DATASET_LABELS)
    models = list(MODEL_LABELS)
    fig, axes = plt.subplots(3, 3, figsize=(12.2, 8.8), sharex=True, sharey=True)
    for row, dataset in enumerate(datasets):
        for column, model in enumerate(models):
            axis = axes[row, column]
            current = frame[(frame.dataset == dataset) & (frame.model == model)]
            for precision in ("fp32", "iea64"):
                values = current[current.precision == precision]
                if values.empty:
                    continue
                paths = values.groupby("checkpoint").validation_prediction_mse
                mean = paths.mean().clip(lower=1e-18)
                low = paths.quantile(0.1).clip(lower=1e-18)
                high = paths.quantile(0.9).clip(lower=1e-18)
                axis.plot(mean.index, mean.values, marker="o", markersize=3,
                          color=COLORS[precision], label=precision.upper())
                axis.fill_between(mean.index, low.values, high.values,
                                  color=COLORS[precision], alpha=0.15, linewidth=0)
            axis.axhline(1e-5, color="#555555", linestyle="--", linewidth=0.8)
            axis.set_yscale("log")
            axis.set_ylim(1e-18, 1)
            axis.grid(alpha=0.18)
            if row == 0:
                axis.set_title(MODEL_LABELS[model])
            if column == 0:
                axis.set_ylabel(f"{DATASET_LABELS[dataset]}\nvalidation orbit MSE")
            if row == 2:
                axis.set_xlabel("Epoch")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.963),
                   ncol=2, frameon=False)
    fig.suptitle("Full-scale schema-orbit trajectories (line: mean; band: 10–90%)", y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    save(fig, output, "h3_orbit_trajectories")


def h7_survival(results: Path, output: Path) -> None:
    frame = pd.read_csv(results / "h7_survival_pairs.csv")
    checkpoints = np.asarray([0, 1, 2, 5, 10, 20, 50, 100, 200])
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    for precision in ("fp32", "iea64"):
        hitting = frame[f"{precision}_hitting_epoch"].to_numpy()
        survival = [float(np.mean(hitting > checkpoint)) for checkpoint in checkpoints]
        axis.step(checkpoints, survival, where="post", linewidth=2.2,
                  color=COLORS[precision], label=precision.upper())
        axis.scatter(checkpoints, survival, s=18, color=COLORS[precision])
    axis.set_ylim(-0.03, 1.03)
    axis.set_xlabel("Epoch checkpoint")
    axis.set_ylabel("Fraction not yet material")
    axis.set_title("Prospective H7 material-survival curve")
    axis.text(0.98, 0.96, f"n={len(frame)} schema paths", transform=axis.transAxes,
              ha="right", va="top", fontsize=9)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    save(fig, output, "h7_material_survival")

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.9), sharex=True, sharey=True)
    for axis, dataset in zip(axes, DATASET_LABELS):
        current = frame[frame.dataset == dataset]
        for precision in ("fp32", "iea64"):
            hitting = current[f"{precision}_hitting_epoch"].to_numpy()
            survival = [
                float(np.mean(hitting > checkpoint)) if len(hitting) else float("nan")
                for checkpoint in checkpoints
            ]
            axis.step(checkpoints, survival, where="post", linewidth=2.0,
                      color=COLORS[precision], label=precision.upper())
            axis.scatter(checkpoints, survival, s=15, color=COLORS[precision])
        axis.set_title(f"{DATASET_LABELS[dataset]} (n={len(current)})")
        axis.set_xlabel("Epoch checkpoint")
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Fraction not yet material")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=2, frameon=False)
    fig.suptitle("Prospective H7 survival by dataset", y=1.08)
    fig.tight_layout()
    save(fig, output, "h7_material_survival_by_dataset")


def h6_forecast(results: Path, output: Path) -> None:
    frame = pd.read_csv(results / "h6_prospective_bundles.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), sharey=True)
    panels = [
        ("epoch20_log_mse", "Raw epoch-20 log10 MSE"),
        ("extrapolated_epoch200_log_mse", "Epoch-20 slope screen"),
    ]
    markers = {"mlp": "o", "resnet": "s", "ft_transformer": "^"}
    dataset_colors = dict(zip(DATASET_LABELS, ["#4c78a8", "#f58518", "#54a24b"]))
    for axis, (column, title) in zip(axes, panels):
        for (dataset, model), current in frame.groupby(["dataset", "model"]):
            axis.scatter(current[column], current.final_log_mse, s=52,
                         marker=markers[model], color=dataset_colors[dataset],
                         alpha=0.85, edgecolor="white", linewidth=0.5)
        axis.axvline(-5, color="#555555", linestyle="--", linewidth=0.9)
        axis.axhline(-5, color="#555555", linestyle="--", linewidth=0.9)
        axis.set_xlabel(title)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Actual epoch-200 log10 MSE")
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color,
               markeredgecolor="white", markersize=8, label=DATASET_LABELS[dataset])
        for dataset, color in dataset_colors.items()
    ] + [
        Line2D([0], [0], marker=marker, color="#333333", linestyle="none",
               markersize=7, label=MODEL_LABELS[model])
        for model, marker in markers.items()
    ]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.035),
               ncol=6, frameon=False)
    fig.suptitle("H6 early semantic-instability screening")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, output, "h6_forecast_comparison")


def h8_screen(results: Path, output: Path) -> None:
    frame = pd.read_csv(results / "h8_prospective_bundles.csv")
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    colors = np.where(frame.material, "#c53b3b", "#2774ae")
    correct = frame.material == frame.predicted_material
    axis.scatter(
        frame.log_mse_20, frame.acceleration, c=colors, s=58,
        marker="o", alpha=0.82, edgecolor="white", linewidth=0.6,
    )
    if (~correct).any():
        wrong = frame[~correct]
        axis.scatter(
            wrong.log_mse_20, wrong.acceleration, marker="x", s=80,
            color="#111111", linewidth=1.5, label="misclassified",
        )
    axis.axvline(-5.0, color="#555555", linestyle="--", linewidth=0.9)
    axis.axhline(0.02, color="#555555", linestyle="--", linewidth=0.9)
    axis.set_xlabel("Epoch-20 log10 orbit MSE")
    axis.set_ylabel("Late-minus-early log-slope")
    axis.set_title(f"H8 level-or-acceleration screen (n={len(frame)})")
    axis.grid(alpha=0.2)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#c53b3b",
               markersize=8, label="material at epoch 200"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2774ae",
               markersize=8, label="not material"),
    ]
    if (~correct).any():
        handles.append(Line2D([0], [0], marker="x", color="#111111",
                              linestyle="none", markersize=8, label="misclassified"))
    axis.legend(handles=handles, frameon=False, fontsize=8)
    save(fig, output, "h8_level_acceleration_screen")


def h4_shadow(results: Path, output: Path) -> None:
    frame = pd.read_csv(results / "h4_config_cells.csv")
    frame = frame[frame.model == "ft_transformer"]
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.9), sharex=True, sharey=True)
    for axis, dataset in zip(axes, DATASET_LABELS):
        current = frame[frame.dataset == dataset]
        axis.scatter(
            np.log10(current.mse_epoch_2.clip(lower=1e-30)),
            np.log10(current.mse_epoch_20.clip(lower=1e-30)),
            c="#6f4aa8", s=46, alpha=0.82, edgecolor="white", linewidth=0.5,
        )
        axis.axhline(-5.0, color="#555555", linestyle="--", linewidth=0.8)
        axis.set_title(DATASET_LABELS[dataset])
        axis.set_xlabel("Epoch-2 log10 orbit MSE")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Epoch-20 log10 orbit MSE")
    fig.suptitle("H4 FT-Transformer semantic-shadow forecast")
    fig.tight_layout()
    save(fig, output, "h4_semantic_shadow_forecast")


def h5_transfer(results: Path, output: Path) -> None:
    frame = pd.read_csv(results / "h5_config_cells.csv")
    frame = frame[frame.model == "ft_transformer"]
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.9), sharex=True, sharey=True)
    for axis, dataset in zip(axes, DATASET_LABELS):
        current = frame[frame.dataset == dataset]
        axis.scatter(
            np.log10(current.shadow_epoch_2.clip(lower=1e-30)),
            np.log10(current.seed_fragility_epoch_20.clip(lower=1e-30)),
            c="#2f8f72", s=46, alpha=0.82, edgecolor="white", linewidth=0.5,
        )
        axis.set_title(DATASET_LABELS[dataset])
        axis.set_xlabel("Epoch-2 log10 schema shadow")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Epoch-20 log10 seed fragility")
    fig.suptitle("H5 cross-perturbation fragility transfer")
    fig.tight_layout()
    save(fig, output, "h5_cross_perturbation_transfer")


def h9_attenuation(results: Path, output: Path) -> None:
    frame = pd.read_csv(results / "h9_prospective_pairs.csv")
    frame = frame[frame.eligible_fp32_material]
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.9), sharex=True, sharey=True)
    for axis, dataset in zip(axes, DATASET_LABELS):
        current = frame[frame.dataset == dataset]
        x = np.log10(current.fp32_final_mse + 1e-30)
        y = np.log10(current.iea64_final_mse + 1e-30)
        axis.scatter(x, y, c="#7c5c2e", s=46, alpha=0.82,
                     edgecolor="white", linewidth=0.5)
        limits = (-30.5, 0.5)
        axis.plot(limits, limits, color="#555555", linestyle="--", linewidth=0.8)
        axis.axhline(-5.0, color="#999999", linestyle=":", linewidth=0.8)
        axis.set_xlim(limits); axis.set_ylim(limits)
        axis.set_title(f"{DATASET_LABELS[dataset]} (n={len(current)})")
        axis.set_xlabel("FP32 final log10 MSE")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("IEA64 final log10 MSE")
    fig.suptitle("H9 post-breach paired attenuation")
    fig.tight_layout()
    save(fig, output, "h9_postbreach_attenuation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--output", type=Path, default=HERE / "results" / "figures")
    args = parser.parse_args()
    h3_trajectories(args.results, args.output)
    if (args.results / "h7_survival_pairs.csv").exists() and not pd.read_csv(
        args.results / "h7_survival_pairs.csv"
    ).empty:
        h7_survival(args.results, args.output)
    if (args.results / "h6_prospective_bundles.csv").exists() and not pd.read_csv(
        args.results / "h6_prospective_bundles.csv"
    ).empty:
        h6_forecast(args.results, args.output)
    if (args.results / "h8_prospective_bundles.csv").exists() and not pd.read_csv(
        args.results / "h8_prospective_bundles.csv"
    ).empty:
        h8_screen(args.results, args.output)
    if (args.results / "h4_config_cells.csv").exists() and not pd.read_csv(
        args.results / "h4_config_cells.csv"
    ).empty:
        h4_shadow(args.results, args.output)
    if (args.results / "h5_config_cells.csv").exists() and not pd.read_csv(
        args.results / "h5_config_cells.csv"
    ).empty:
        h5_transfer(args.results, args.output)
    if (args.results / "h9_prospective_pairs.csv").exists() and not pd.read_csv(
        args.results / "h9_prospective_pairs.csv"
    ).empty:
        h9_attenuation(args.results, args.output)


if __name__ == "__main__":
    main()
