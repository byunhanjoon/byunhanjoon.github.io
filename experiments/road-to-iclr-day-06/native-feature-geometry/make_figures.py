"""Generate the declared Day-6 Native Feature Geometry dose figure."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = RESULTS / "figures"
DOMAINS = ("cycle16", "ordinal16", "tree16", "nominal16")
INTERFACES = ("learned", "native_tuned")
LABELS = {"learned": "Learned lookup", "native_tuned": "Native-initialized lookup"}
COLORS = {"learned": "#2A6FDB", "native_tuned": "#D8572A"}


def main() -> None:
    grouped = defaultdict(dict)
    with (RESULTS / "h6_dose_cells.csv").open() as handle:
        for row in csv.DictReader(handle):
            key = (row["domain"], int(row["seed"]), row["interface"])
            grouped[key][float(row["alpha"])] = float(row["chart_mean_held_mse"])
    alphas = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0])
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.4), sharey=False)
    for axis, domain in zip(axes, DOMAINS):
        for interface in INTERFACES:
            curves = []
            for seed in (7301, 7302, 7303):
                values = np.asarray([grouped[(domain, seed, interface)][alpha] for alpha in alphas])
                curves.append(values / values[-1])
            curves = np.asarray(curves)
            mean = curves.mean(axis=0)
            sem = curves.std(axis=0, ddof=1) / np.sqrt(len(curves))
            axis.plot(alphas, mean, marker="o", color=COLORS[interface], label=LABELS[interface])
            axis.fill_between(alphas, mean - sem, mean + sem, color=COLORS[interface], alpha=0.16)
        axis.axhline(1.0, color="#777777", linewidth=0.8, linestyle="--")
        axis.set_title(domain.replace("16", "").capitalize())
        axis.set_xlabel("Metric corruption α")
        axis.grid(alpha=0.2)
        if domain == "nominal16":
            axis.set_ylim(0.98, 1.02)
            axis.ticklabel_format(axis="y", style="plain", useOffset=False)
            axis.text(
                0.5, 0.985, "max relative range < 4.4e-9",
                ha="center", va="bottom", fontsize=8, color="#555555",
            )
    axes[0].set_ylabel("Held-category MSE / shuffled-metric MSE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.91))
    fig.suptitle("Within-model error rises as feature geometry is corrupted", y=0.99, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "h6_metric_corruption_dose.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIGURES / "h6_metric_corruption_dose.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
