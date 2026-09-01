#!/usr/bin/env python3
"""Calibrate PriorDial in information units and audit an independent replication."""

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

from src.priors import population_coupling_mi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--primary-summary",
        type=Path,
        default=ROOT / "results/processed/e1_primary_b779842a24_t420_n64_summary.csv",
    )
    parser.add_argument(
        "--primary-raw",
        type=Path,
        default=ROOT / "results/raw/e1_primary_b779842a24_t420_n64.npz",
    )
    parser.add_argument("--replication-summary", type=Path)
    parser.add_argument("--replication-raw", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    return parser.parse_args()


def unique_glob(pattern: str) -> Path:
    matches = sorted(ROOT.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one match for {pattern!r}, found {matches}")
    return matches[0]


def bootstrap_mean(values: np.ndarray, draws: int, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for start in range(0, draws, 500):
        stop = min(start + 500, draws)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[indices].mean(axis=1)
    return float(values.mean()), *np.quantile(means, [0.025, 0.975]).tolist()


def endpoint_contrasts(raw_path: Path, run: str, draws: int, seed: int) -> list[dict]:
    raw = np.load(raw_path, allow_pickle=False)
    rho = raw["rho"].astype(float)
    task_type = raw["task_type"].astype(str)
    utility = raw["stable_expert_loss"].astype(float) - raw["combined_expert_loss"].astype(float)
    records = []
    for task_index, task in enumerate(("classification", "regression")):
        low = utility[(task_type == task) & np.isclose(rho, 0.0)]
        high = utility[(task_type == task) & np.isclose(rho, 1.0)]
        if low.size == 0 or high.size == 0:
            raise RuntimeError(f"missing endpoint tasks for {run}/{task}")
        # Endpoint tasks are independently generated, so bootstrap both samples.
        rng = np.random.default_rng(seed + task_index)
        difference = np.empty(draws)
        for start in range(0, draws, 500):
            stop = min(start + 500, draws)
            low_idx = rng.integers(0, low.size, size=(stop - start, low.size))
            high_idx = rng.integers(0, high.size, size=(stop - start, high.size))
            difference[start:stop] = high[high_idx].mean(axis=1) - low[low_idx].mean(axis=1)
        low_stats = bootstrap_mean(low, draws, seed + 10 + task_index)
        high_stats = bootstrap_mean(high, draws, seed + 20 + task_index)
        records.append({
            "run": run,
            "task_type": task,
            "tasks_per_endpoint": int(low.size),
            "rho0_utility": low_stats[0],
            "rho0_ci_low": low_stats[1],
            "rho0_ci_high": low_stats[2],
            "rho1_utility": high_stats[0],
            "rho1_ci_low": high_stats[1],
            "rho1_ci_high": high_stats[2],
            "endpoint_utility_gain": float(high.mean() - low.mean()),
            "endpoint_gain_ci_low": float(np.quantile(difference, 0.025)),
            "endpoint_gain_ci_high": float(np.quantile(difference, 0.975)),
        })
    return records


def linear_r2(x: np.ndarray, y: np.ndarray) -> float:
    design = np.c_[np.ones(len(x)), x]
    prediction = design @ np.linalg.lstsq(design, y, rcond=None)[0]
    denominator = np.sum((y - y.mean()) ** 2)
    return float(1.0 - np.sum((y - prediction) ** 2) / denominator) if denominator > 0 else 1.0


def main() -> None:
    args = parse_args()
    replication_summary = args.replication_summary or unique_glob(
        "results/processed/fallback_dial_replication_*_summary.csv"
    )
    replication_raw = args.replication_raw or unique_glob(
        "results/raw/fallback_dial_replication_*.npz"
    )
    inputs = (
        ("development", args.primary_summary.resolve(), args.primary_raw.resolve()),
        ("independent_replication", replication_summary.resolve(), replication_raw.resolve()),
    )
    for _, summary, raw in inputs:
        if not summary.exists() or not raw.exists():
            raise FileNotFoundError(summary if not summary.exists() else raw)

    calibrated_frames = []
    contrasts = []
    fits = []
    for run_index, (run, summary_path, raw_path) in enumerate(inputs):
        frame = pd.read_csv(summary_path)
        frame.insert(0, "run", run)
        frame["population_mi_c_w_nats"] = [
            population_coupling_mi(float(rho), 6) for rho in frame["effective_rho"]
        ]
        frame["finite_schedule_mi_residual"] = (
            frame["empirical_mi_c_w_nats"] - frame["population_mi_c_w_nats"]
        )
        frame["relative_shape_risk_reduction"] = (
            frame["marginal_query_utility"] / frame["stable_expert_loss"]
        )
        calibrated_frames.append(frame)
        contrasts.extend(endpoint_contrasts(raw_path, run, args.bootstrap_draws, 4900 + run_index * 100))
        for task, cell in frame.groupby("task_type"):
            ordered = cell.sort_values("rho")
            utility = ordered["marginal_query_utility"].to_numpy()
            fits.append({
                "run": run,
                "task_type": task,
                "utility_r2_vs_rho": linear_r2(ordered["effective_rho"].to_numpy(), utility),
                "utility_r2_vs_population_mi": linear_r2(
                    ordered["population_mi_c_w_nats"].to_numpy(), utility
                ),
                "max_abs_finite_mi_residual": float(
                    ordered["finite_schedule_mi_residual"].abs().max()
                ),
            })

    calibrated = pd.concat(calibrated_frames, ignore_index=True)
    contrast_frame = pd.DataFrame(contrasts)
    fit_frame = pd.DataFrame(fits)
    calibrated_path = ROOT / "results/processed/fallback_dial_calibration_v1.csv"
    contrast_path = ROOT / "results/processed/fallback_dial_replication_contrasts_v1.csv"
    audit_path = ROOT / "results/processed/fallback_dial_calibration_audit_v1.json"
    figure_path = ROOT / "figures/fallback_dial_information_performance_v1.png"
    for output in (calibrated_path, contrast_path, audit_path, figure_path):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite analysis artifact: {output}")

    calibrated.to_csv(calibrated_path, index=False)
    contrast_frame.to_csv(contrast_path, index=False)
    audit = {
        "inputs": [
            {"run": run, "summary": str(summary.relative_to(ROOT)), "raw": str(raw.relative_to(ROOT))}
            for run, summary, raw in inputs
        ],
        "bootstrap_draws": args.bootstrap_draws,
        "population_information_endpoints": {
            "rho0": population_coupling_mi(0.0, 6),
            "rho1": population_coupling_mi(1.0, 6),
        },
        "linear_fit_diagnostics": fits,
        "endpoint_contrasts": contrasts,
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex="col")
    colors = {"development": "#3366cc", "independent_replication": "#d95f02"}
    labels = {"development": "development n=64,d=8", "independent_replication": "replication n=96,d=12"}
    for column, task in enumerate(("classification", "regression")):
        for run in colors:
            cell = calibrated[(calibrated["run"] == run) & (calibrated["task_type"] == task)].sort_values("rho")
            x = cell["population_mi_c_w_nats"]
            axes[0, column].plot(x, cell["mechanism_accuracy"], marker="o", color=colors[run], label=labels[run])
            axes[1, column].plot(x, cell["marginal_query_utility"], marker="o", color=colors[run], label=labels[run])
        axes[0, column].set_title(task.capitalize())
        axes[0, column].set_ylabel("Marginal-only mechanism accuracy")
        axes[1, column].set_ylabel("Stable minus stable+shape loss")
        axes[1, column].set_xlabel("Exact population I(C;W) [nats]")
        axes[1, column].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        axes[0, column].grid(alpha=0.2)
        axes[1, column].grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("PriorDial information calibration and independent performance replication")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
