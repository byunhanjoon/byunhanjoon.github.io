#!/usr/bin/env python3
"""Consolidate the predeclared E3 failure-branch sequence."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNS = (
    ("Initial", "v1"),
    ("More train", "more_train_v1"),
    ("Featurewise", "featurewise_v1"),
    ("Auxiliary", "auxiliary_v1"),
    ("Calibrated", "calibrated_v1"),
)


def main() -> None:
    rows = []
    for order, (label, tag) in enumerate(RUNS):
        frame = pd.read_csv(ROOT / "results/processed" / f"e3_contrasts_{tag}.csv")
        for record in frame.to_dict("records"):
            rows.append({"order": order, "iteration": label, "source_tag": tag, **record})
    result = pd.DataFrame(rows)
    output = ROOT / "results/processed/e3_failure_branch_sequence_v1.csv"
    result.to_csv(output, index=False, mode="x")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), sharey=True)
    for axis, task in zip(axes, ("classification", "regression"), strict=True):
        cell = result[result.task_type == task].sort_values("order")
        values = 100 * cell.oracle_headroom_fraction_captured.to_numpy()
        colors = ["#adb5bd", "#8ecae6", "#219ebc", "#06d6a0", "#ef476f"]
        axis.bar(np.arange(len(cell)), values, color=colors)
        axis.axhline(20, color="black", linestyle="--", lw=1, label="20% G3 guide")
        axis.set_xticks(np.arange(len(cell)), cell.iteration, rotation=35, ha="right")
        axis.set(title=task.capitalize(), ylabel="oracle headroom captured (%)")
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle("E3 sequential failure branch: adaptation remains below G3")
    fig.tight_layout()
    figure = ROOT / "figures/e3_failure_branch_sequence_v1.png"
    if figure.exists():
        raise FileExistsError(figure)
    fig.savefig(figure, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(result[["iteration", "task_type", "fixed_minus_gate_mean",
                  "fixed_minus_oracle_mean", "oracle_headroom_fraction_captured"]].to_string(index=False))


if __name__ == "__main__":
    main()
