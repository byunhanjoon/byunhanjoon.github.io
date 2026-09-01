#!/usr/bin/env python3
"""Audit and summarize an immutable E3 M0--M5 run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.methods import episode_loss
from src.stats import mean_interval


METHODS = {
    "M0 raw": "m0_raw_loss",
    "M1 robust": "m1_robust_loss",
    "M2 rank": "m2_rank_loss",
    "M3 augmentation": "m3_augmentation_loss",
    "M4 50/50": "m4_50_loss",
    "M4 tuned fixed": "m4_fixed_loss",
    "M5 learned gate": "m5_gate_loss",
    "Oracle gate": "oracle_loss",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-key", required=True)
    parser.add_argument("--tag", required=True)
    return parser.parse_args()


def exclusive_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, mode="x")


def main() -> None:
    args = arguments()
    raw_path = ROOT / "results/raw" / f"{args.run_key}.npz"
    metadata_path = ROOT / "results/raw" / f"{args.run_key}.metadata.json"
    episode_path = ROOT / "results/processed" / f"{args.run_key}_episodes.csv"
    metadata = json.loads(metadata_path.read_text())
    episodes = pd.read_csv(episode_path)
    with np.load(raw_path, allow_pickle=False) as bundle:
        raw = {key: bundle[key] for key in bundle.files}
    expected = int(metadata["episodes"])
    if len(episodes) != expected or any(value.shape[0] != expected for value in raw.values()):
        raise AssertionError("raw/CSV episode count mismatch")
    if not np.array_equal(episodes["split"].to_numpy(), raw["split"]):
        raise AssertionError("split order mismatch")
    if not np.array_equal(episodes["episode_seed"].to_numpy(), raw["episode_seed"]):
        raise AssertionError("episode seed order mismatch")
    numeric = [value for value in raw.values() if np.issubdtype(value.dtype, np.number)]
    if not all(np.all(np.isfinite(value)) for value in numeric):
        raise AssertionError("nonfinite raw array")

    # Recompute the two primary predictions from float32 immutable arrays. The small
    # tolerance covers only serialization precision relative to the float64 run CSV.
    clip = 1e-6
    for index in np.linspace(0, expected - 1, min(200, expected), dtype=int):
        task = str(raw["task_type"][index])
        y = raw["y_query"][index]
        fixed_alpha = float(episodes.iloc[index]["m4_fixed_alpha"])
        fixed = fixed_alpha * raw["prediction_raw"][index] + (1 - fixed_alpha) * raw["prediction_rank"][index]
        learned_alpha = float(raw["learned_alpha"][index])
        learned = learned_alpha * raw["prediction_raw"][index] + (1 - learned_alpha) * raw["prediction_rank"][index]
        if not np.isclose(episode_loss(y, fixed, task, clip), episodes.iloc[index]["m4_fixed_loss"], rtol=2e-5, atol=2e-7):
            raise AssertionError(f"fixed loss mismatch at {index}")
        if not np.isclose(episode_loss(y, learned, task, clip), episodes.iloc[index]["m5_gate_loss"], rtol=2e-5, atol=2e-7):
            raise AssertionError(f"gate loss mismatch at {index}")

    test = episodes[episodes.split == "test"].copy()
    summaries = []
    counter = 0
    for (task, rho), group in test.groupby(["task_type", "rho"], sort=True):
        for label, column in METHODS.items():
            estimate = mean_interval(group[column].to_numpy(), draws=10000, seed=1000 + counter)
            summaries.append({
                "task_type": task, "rho": rho, "method": label, "episodes": len(group),
                "loss_mean": estimate[0], "loss_ci_low": estimate[1], "loss_ci_high": estimate[2],
            })
            counter += 1
    summary = pd.DataFrame(summaries)

    contrasts = []
    for task_index, (task, group) in enumerate(test.groupby("task_type", sort=True)):
        fixed_minus_gate = group.m4_fixed_loss.to_numpy() - group.m5_gate_loss.to_numpy()
        fixed_minus_oracle = group.m4_fixed_loss.to_numpy() - group.oracle_loss.to_numpy()
        rank_minus_gate = group.m2_rank_loss.to_numpy() - group.m5_gate_loss.to_numpy()
        estimates = {
            "fixed_minus_gate": mean_interval(fixed_minus_gate, draws=10000, seed=3000 + task_index),
            "fixed_minus_oracle": mean_interval(fixed_minus_oracle, draws=10000, seed=4000 + task_index),
            "rank_minus_gate": mean_interval(rank_minus_gate, draws=10000, seed=5000 + task_index),
        }
        headroom = estimates["fixed_minus_oracle"][0]
        captured = estimates["fixed_minus_gate"][0] / headroom if headroom > 0 else np.nan
        contrasts.append({
            "task_type": task, "test_episodes": len(group),
            **{f"{name}_{suffix}": value[position]
               for name, value in estimates.items()
               for position, suffix in enumerate(("mean", "ci_low", "ci_high"))},
            "oracle_headroom_fraction_captured": captured,
            "fraction_gate_beats_fixed": float(np.mean(fixed_minus_gate > 0)),
            "fixed_alpha": float(group.m4_fixed_alpha.iloc[0]),
            "gate_alpha_mean": float(group.m5_gate_alpha.mean()),
            "oracle_alpha_mean": float(group.oracle_alpha.mean()),
        })
    contrast = pd.DataFrame(contrasts)
    integrity = {
        "run_key": args.run_key,
        "episodes": expected,
        "test_episodes": len(test),
        "raw_csv_alignment": True,
        "all_numeric_arrays_finite": True,
        "sampled_primary_losses_recomputed": min(200, expected),
        "expert_fits": metadata["expert_fits"],
        "wall_clock_seconds": metadata["wall_clock_seconds"],
        "selections": metadata["selections"],
    }

    output = ROOT / "results/processed"
    exclusive_csv(summary, output / f"e3_phase_summary_{args.tag}.csv")
    exclusive_csv(contrast, output / f"e3_contrasts_{args.tag}.csv")
    integrity_path = output / f"e3_integrity_{args.tag}.json"
    with integrity_path.open("x", encoding="utf-8") as handle:
        json.dump(integrity, handle, indent=2, sort_keys=True)
        handle.write("\n")

    selected = ["M0 raw", "M2 rank", "M4 tuned fixed", "M5 learned gate", "Oracle gate"]
    colors = {"M0 raw": "#ef476f", "M2 rank": "#2667ff", "M4 tuned fixed": "#6c757d",
              "M5 learned gate": "#06d6a0", "Oracle gate": "#8e44ad"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for axis, task in zip(axes, ("classification", "regression"), strict=True):
        for method in selected:
            cell = summary[(summary.task_type == task) & (summary.method == method)].sort_values("rho")
            axis.plot(cell.rho, cell.loss_mean, marker="o", color=colors[method], label=method)
            axis.fill_between(cell.rho, cell.loss_ci_low, cell.loss_ci_high, color=colors[method], alpha=0.10)
        axis.set(title=task.capitalize(), xlabel="PriorDial coupling rho",
                 ylabel="log loss" if task == "classification" else "MSE")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("E3 held-out M0--M5 kill test")
    fig.tight_layout()
    figure_path = ROOT / "figures" / f"e3_method_kill_{args.tag}.png"
    if figure_path.exists():
        raise FileExistsError(figure_path)
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(contrast.to_string(index=False))
    print(json.dumps(integrity, indent=2))


if __name__ == "__main__":
    main()
