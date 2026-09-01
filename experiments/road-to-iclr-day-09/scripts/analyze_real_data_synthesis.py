#!/usr/bin/env python3
"""Dataset-balanced synthesis of all completed real competence-routing panels."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "results" / "processed"
SOURCES = {
    "small_panel": PROCESSED / "real_panel_competence_55553b7ffd_cells.csv",
    "breadth_panel": PROCESSED / "openml_breadth_competence_48170161d0_cells.csv",
    "regression_confirmation": PROCESSED / "regression_confirmation_1e4911698d_cells.csv",
}
SEED = 175001
N_BOOT = 50_000
PANEL_COLORS = {
    "small_panel": "#4C78A8",
    "breadth_panel": "#F58518",
    "regression_confirmation": "#54A24B",
}


def load_dataset_gains() -> pd.DataFrame:
    frames = []
    for panel, path in SOURCES.items():
        cells = pd.read_csv(path)
        cells = cells[cells["method"].isin(["fixed", "competence"])].copy()
        index = ["episode_index", "dataset", "task_type", "repeat", "feature_count"]
        wide = cells.pivot(index=index, columns="method", values="loss").reset_index()
        if wide[["fixed", "competence"]].isna().any().any():
            raise ValueError(f"unpaired method rows in {path.name}")
        wide["gain"] = wide["fixed"] - wide["competence"]
        by_dataset = (
            wide.groupby(["dataset", "task_type"], as_index=False)
            .agg(mean_gain=("gain", "mean"), repeats=("gain", "size"))
        )
        by_dataset.insert(0, "panel", panel)
        frames.append(by_dataset)

    detail = pd.concat(frames, ignore_index=True)
    duplicated = detail.duplicated("dataset", keep=False)
    if duplicated.any():
        names = sorted(detail.loc[duplicated, "dataset"].unique())
        raise ValueError(f"dataset identities are not disjoint: {names}")
    return detail.sort_values(["task_type", "panel", "dataset"]).reset_index(drop=True)


def summarize(values: np.ndarray, rng: np.random.Generator) -> dict[str, object]:
    values = np.asarray(values, dtype=float)
    d = len(values)
    boot = values[rng.integers(0, d, size=(N_BOOT, d))].mean(axis=1)
    trim = int(np.floor(0.1 * d))
    ordered = np.sort(values)
    trimmed = ordered[trim : d - trim] if trim else ordered
    loo = np.array([np.delete(values, i).mean() for i in range(d)])
    ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
    return {
        "datasets": d,
        "mean_gain": float(values.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "median_gain": float(np.median(values)),
        "positive_datasets": int((values > 0).sum()),
        "trimmed_mean_gain": float(trimmed.mean()),
        "trimmed_each_tail": trim,
        "loo_mean_min": float(loo.min()),
        "loo_mean_max": float(loo.max()),
        "robustness_gate_pass": bool(
            ci_low > 0
            and np.median(values) > 0
            and (values > 0).sum() > d / 2
            and trimmed.mean() > 0
            and loo.min() > 0
        ),
    }


def make_figure(detail: pd.DataFrame, task_summary: dict[str, object]) -> None:
    figure_path = ROOT / "figures" / "real_data_synthesis_v1.png"
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 7.3), gridspec_kw={"wspace": 0.75})
    for axis, task_type in zip(axes, ["classification", "regression"]):
        subset = detail[detail["task_type"] == task_type].sort_values("mean_gain")
        y = np.arange(len(subset))
        for panel, part in subset.groupby("panel", sort=False):
            positions = [int(np.flatnonzero(subset.index == idx)[0]) for idx in part.index]
            axis.scatter(
                part["mean_gain"], positions, s=40, color=PANEL_COLORS[panel],
                edgecolor="white", linewidth=0.5, label=panel.replace("_", " "), zorder=3,
            )
        axis.axvline(0, color="#444444", linewidth=1, linestyle="--", zorder=1)
        axis.set_yticks(y, subset["dataset"].str.replace("_", " "))
        axis.grid(axis="x", color="#DDDDDD", linewidth=0.7, zorder=0)
        axis.set_title(f"{task_type.capitalize()} ({len(subset)} datasets)")
        axis.set_xlabel("Fixed loss − competence loss  (positive is better)")
        if task_type == "regression":
            axis.set_xscale("symlog", linthresh=0.02, linscale=0.8)

        summary = task_summary[task_type]
        mean_y = len(subset) + 0.75
        axis.errorbar(
            summary["mean_gain"], mean_y,
            xerr=[[summary["mean_gain"] - summary["ci_low"]],
                  [summary["ci_high"] - summary["mean_gain"]]],
            fmt="D", color="black", capsize=4, markersize=6, linewidth=1.5, zorder=4,
        )
        axis.set_yticks(list(y) + [mean_y], list(subset["dataset"].str.replace("_", " ")) + ["dataset mean (95% CI)"])
        axis.invert_yaxis()
        axis.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Synthetic-tuned competence routing transfers only to numeric regression", y=0.99)
    fig.subplots_adjust(bottom=0.12, top=0.92)
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    detail = load_dataset_gains()
    detail_path = PROCESSED / "real_data_synthesis_dataset_gains_v1.csv"
    audit_path = PROCESSED / "real_data_synthesis_audit_v1.json"
    detail.to_csv(detail_path, index=False)

    rng = np.random.default_rng(SEED)
    task_summary: dict[str, object] = {}
    for task_type, group in detail.groupby("task_type", sort=True):
        record = summarize(group["mean_gain"].to_numpy(), rng)
        record["panel_means"] = {
            panel: float(part["mean_gain"].mean())
            for panel, part in group.groupby("panel", sort=True)
        }
        task_summary[str(task_type)] = record

    audit = {
        "protocol": "REAL_DATA_SYNTHESIS_PROTOCOL.md",
        "status": "retrospective_synthesis",
        "bootstrap_seed": SEED,
        "bootstrap_replicates": N_BOOT,
        "source_files": {key: str(path.relative_to(ROOT)) for key, path in SOURCES.items()},
        "dataset_identities_disjoint": True,
        "tasks": task_summary,
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    make_figure(detail, task_summary)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
