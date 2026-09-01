#!/usr/bin/env python3
"""Trace fixed-to-competence loss paths on synthetic and real frozen predictions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.methods import competence_weights, prediction_loss, weighted_prediction


LAMBDAS = np.linspace(0.0, 1.0, 11)
N_BOOT = 20_000
REAL_SOURCES = {
    "small_panel": "real_panel_competence_55553b7ffd",
    "breadth_panel": "openml_breadth_competence_48170161d0",
    "regression_confirmation": "regression_confirmation_1e4911698d",
}


def episode_curve(
    y: np.ndarray, experts: np.ndarray, cv_loss: np.ndarray,
    task_type: str, tuning: dict[str, object],
) -> np.ndarray:
    fixed_weights = np.asarray(tuning["fixed_weights"], dtype=float)
    adaptive_weights = competence_weights(
        cv_loss, float(tuning["temperature"]), float(tuning["uniform_shrinkage"])
    )
    fixed = weighted_prediction(experts, fixed_weights)
    adaptive = weighted_prediction(experts, adaptive_weights)
    return np.asarray([
        prediction_loss(y, (1 - amount) * fixed + amount * adaptive, task_type)
        for amount in LAMBDAS
    ])


def load_synthetic() -> pd.DataFrame:
    stem = "fallback_loss_router_2e46ddf857_test"
    metadata = json.loads((ROOT / "results" / "raw" / f"{stem}.metadata.json").read_text())
    with np.load(ROOT / "results" / "raw" / f"{stem}.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    rows = []
    for index in range(len(arrays["task_type"])):
        task_type = str(arrays["task_type"][index])
        curve = episode_curve(
            arrays["query_y"][index].astype(float),
            arrays["expert_prediction"][index].astype(float),
            arrays["cv_expert_loss"][index].astype(float),
            task_type, metadata["tuning"][task_type],
        )
        unit = f"n{int(arrays['context_size'][index])}_d{int(arrays['feature_count'][index])}_r{float(arrays['rho'][index]):.2f}"
        for amount, loss in zip(LAMBDAS, curve):
            rows.append({"scope": "synthetic_test", "task_type": task_type, "unit": unit,
                         "episode": index, "lambda": amount, "loss": loss})
    return pd.DataFrame(rows)


def load_real() -> pd.DataFrame:
    rows = []
    for panel, stem in REAL_SOURCES.items():
        metadata = json.loads((ROOT / "results" / "raw" / f"{stem}.metadata.json").read_text())
        with np.load(ROOT / "results" / "raw" / f"{stem}.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        for index in range(len(arrays["task_type"])):
            task_type = str(arrays["task_type"][index])
            tuning = metadata["synthetic_tuning"]
            if task_type in tuning:
                tuning = tuning[task_type]
            curve = episode_curve(
                arrays["query_y"][index].astype(float),
                arrays["expert_prediction"][index].astype(float),
                arrays["cv_expert_loss"][index].astype(float), task_type, tuning,
            )
            for amount, loss in zip(LAMBDAS, curve):
                rows.append({
                    "scope": "real_panel", "task_type": task_type,
                    "unit": str(arrays["dataset"][index]),
                    "episode": f"{panel}_{index}", "lambda": amount, "loss": loss,
                })
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    unit_curve = frame.groupby(
        ["scope", "task_type", "unit", "lambda"], as_index=False
    )["loss"].mean()
    rows = []
    audit: dict[str, object] = {}
    for offset, ((scope, task_type), group) in enumerate(
        unit_curve.groupby(["scope", "task_type"], sort=True)
    ):
        wide = group.pivot(index="unit", columns="lambda", values="loss").sort_index()
        baseline = wide[0.0].to_numpy()
        rng = np.random.default_rng(215001 + offset)
        indices = rng.integers(0, len(wide), size=(N_BOOT, len(wide)))
        curve_records = []
        for amount in LAMBDAS:
            gain = baseline - wide[float(amount)].to_numpy()
            samples = gain[indices].mean(axis=1)
            low, high = np.quantile(samples, [0.025, 0.975])
            record = {
                "scope": scope, "task_type": task_type, "lambda": float(amount),
                "gain": float(gain.mean()), "ci_low": float(low), "ci_high": float(high),
            }
            rows.append(record); curve_records.append(record)
        aggregate_losses = wide.mean(axis=0)
        aggregate_optimum = float(aggregate_losses.index[np.argmin(aggregate_losses.to_numpy())])
        unit_optima = LAMBDAS[np.argmin(wide.to_numpy(), axis=1)]
        audit[f"{scope}:{task_type}"] = {
            "units": len(wide),
            "aggregate_grid_optimum": aggregate_optimum,
            "unit_optimum_counts": {
                f"{amount:.1f}": int(np.sum(unit_optima == amount)) for amount in LAMBDAS
            },
            "curve": curve_records,
        }
    return pd.DataFrame(rows), audit


def make_figure(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharex=True)
    colors = {"synthetic_test": "#4C78A8", "real_panel": "#E45756"}
    for axis, task_type in zip(axes, ("classification", "regression")):
        for scope, group in summary[summary["task_type"] == task_type].groupby("scope"):
            axis.plot(group["lambda"], group["gain"], marker="o", color=colors[scope],
                      label=scope.replace("_", " "))
            axis.fill_between(group["lambda"], group["ci_low"], group["ci_high"],
                              color=colors[scope], alpha=0.15)
        axis.axhline(0, color="#555555", linestyle="--", linewidth=1)
        axis.set_title(task_type.capitalize())
        axis.set_xlabel("Adaptation strength λ")
        axis.set_ylabel("Gain over fixed mixture")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle("Synthetic-to-real transfer of competence-routing strength")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "shrinkage_transfer_v1.png", dpi=200)
    plt.close(fig)


def main() -> None:
    episodes = pd.concat([load_synthetic(), load_real()], ignore_index=True)
    detail_path = ROOT / "results" / "processed" / "shrinkage_transfer_unit_curves_v1.csv"
    summary_path = ROOT / "results" / "processed" / "shrinkage_transfer_summary_v1.csv"
    audit_path = ROOT / "results" / "processed" / "shrinkage_transfer_audit_v1.json"
    unit_curves = episodes.groupby(
        ["scope", "task_type", "unit", "lambda"], as_index=False
    )["loss"].mean()
    unit_curves.to_csv(detail_path, index=False)
    summary, audit_scopes = summarize(episodes)
    summary.to_csv(summary_path, index=False)
    audit = {
        "protocol": "SHRINKAGE_TRANSFER_PROTOCOL.md",
        "lambdas": LAMBDAS.tolist(),
        "bootstrap_replicates": N_BOOT,
        "scopes": audit_scopes,
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    make_figure(summary)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
