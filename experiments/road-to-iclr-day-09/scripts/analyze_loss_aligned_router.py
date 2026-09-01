#!/usr/bin/env python3
"""Frozen analysis for the untouched loss-aligned routing test."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def unique(pattern: str) -> Path:
    paths = sorted(ROOT.glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(f"expected one {pattern}, found {paths}")
    return paths[0]


def paired_draws(
    frame: pd.DataFrame,
    method_a: str,
    method_b: str,
    draws: int,
    seed: int,
) -> tuple[float, float, float, np.ndarray]:
    """Return equal-cell mean of loss(a)-loss(b), positive when b is better."""
    cells = []
    for _, cell in frame.groupby(["context_size", "feature_count", "rho"], sort=True):
        pivot = cell.pivot(index="episode_index", columns="method", values="loss")
        cells.append((pivot[method_a] - pivot[method_b]).to_numpy())
    observed = float(np.mean([values.mean() for values in cells]))
    rng = np.random.default_rng(seed)
    sampled = np.empty(draws)
    for start in range(0, draws, 250):
        stop = min(start + 250, draws)
        chunk = np.zeros(stop - start)
        for values in cells:
            indices = rng.integers(0, values.size, size=(stop - start, values.size))
            chunk += values[indices].mean(axis=1)
        sampled[start:stop] = chunk / len(cells)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return observed, float(low), float(high), sampled


def main() -> None:
    cells_path = unique("results/processed/fallback_loss_router_*_test_cells.csv")
    metadata_path = unique("results/raw/fallback_loss_router_*_test.metadata.json")
    frame = pd.read_csv(cells_path)
    metadata = json.loads(metadata_path.read_text())
    draws = 10_000
    summary_path = ROOT / "results/processed/fallback_loss_router_summary_v1.csv"
    contrasts_path = ROOT / "results/processed/fallback_loss_router_contrasts_v1.csv"
    audit_path = ROOT / "results/processed/fallback_loss_router_audit_v1.json"
    figure_path = ROOT / "figures/fallback_loss_router_v1.png"
    for output in (summary_path, contrasts_path, audit_path, figure_path):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")

    summary = (
        frame.groupby(["task_type", "context_size", "feature_count", "rho", "method"], as_index=False)
        .agg(mean_loss=("loss", "mean"), episodes=("episode_index", "nunique"))
    )
    summary.to_csv(summary_path, index=False)

    contrast_records = []
    audit_tasks = {}
    sampled_by_task = {}
    for task_index, task_type in enumerate(("classification", "regression")):
        task = frame[frame["task_type"] == task_type]
        task_draws = {}
        for comparison_index, (label, method_a, method_b) in enumerate((
            ("competence_vs_fixed", "fixed", "competence"),
            ("competence_vs_uniform", "uniform", "competence"),
            ("competence_vs_shape_family", "shape_family", "competence"),
            ("fixed_to_best_individual_headroom", "fixed", "best_individual_oracle"),
        )):
            # Reuse the same resample indices across comparisons so headroom ratios are paired.
            value, low, high, sampled = paired_draws(
                task, method_a, method_b, draws, 14_000 + task_index * 100
            )
            task_draws[label] = sampled
            contrast_records.append({
                "scope": "all_cells",
                "task_type": task_type,
                "comparison": label,
                "gain": value,
                "ci_low": low,
                "ci_high": high,
            })
        high_dim = task[(task["feature_count"] == 12) & (task["rho"] >= 0.75)]
        high_value, high_low, high_high, _ = paired_draws(
            high_dim, "fixed", "competence", draws, 15_000 + task_index
        )
        contrast_records.append({
            "scope": "d12_rho_ge_075",
            "task_type": task_type,
            "comparison": "competence_vs_fixed",
            "gain": high_value,
            "ci_low": high_low,
            "ci_high": high_high,
        })
        gain = next(
            record["gain"] for record in contrast_records
            if record["scope"] == "all_cells" and record["task_type"] == task_type
            and record["comparison"] == "competence_vs_fixed"
        )
        headroom = next(
            record["gain"] for record in contrast_records
            if record["scope"] == "all_cells" and record["task_type"] == task_type
            and record["comparison"] == "fixed_to_best_individual_headroom"
        )
        numerator_draw = task_draws["competence_vs_fixed"]
        denominator_draw = task_draws["fixed_to_best_individual_headroom"]
        valid = denominator_draw > 1e-12
        capture_draw = numerator_draw[valid] / denominator_draw[valid]
        capture = gain / headroom if headroom > 0 else float("nan")
        route = task.drop_duplicates("episode_index")
        route_accuracy = float(np.mean(route["competence_argmin"] == route["query_best_expert"]))
        audit_tasks[task_type] = {
            "competence_vs_fixed_gain": gain,
            "fixed_to_best_individual_headroom": headroom,
            "headroom_capture": capture,
            "headroom_capture_ci": np.quantile(capture_draw, [0.025, 0.975]).tolist(),
            "context_cv_argmin_matches_query_best_rate": route_accuracy,
            "episode_win_rate_vs_fixed": float(np.mean(
                task.pivot(index="episode_index", columns="method", values="loss")["competence"]
                < task.pivot(index="episode_index", columns="method", values="loss")["fixed"]
            )),
            "high_dim_high_rho_gain": high_value,
            "high_dim_high_rho_ci": [high_low, high_high],
        }
        sampled_by_task[task_type] = task_draws

    contrasts = pd.DataFrame(contrast_records)
    contrasts.to_csv(contrasts_path, index=False)
    cls = audit_tasks["classification"]
    reg = audit_tasks["regression"]
    passing = []
    for task_type, result in audit_tasks.items():
        row = contrasts[
            (contrasts["scope"] == "all_cells")
            & (contrasts["task_type"] == task_type)
            & (contrasts["comparison"] == "competence_vs_fixed")
        ].iloc[0]
        if row["ci_low"] > 0 and result["headroom_capture"] >= 0.20:
            passing.append(task_type)
    no_material_harm = {
        "classification": not (
            contrasts.query("scope == 'all_cells' and task_type == 'classification' and comparison == 'competence_vs_fixed'")
            .iloc[0]["ci_high"] < -0.001
        ),
        "regression": not (
            contrasts.query("scope == 'all_cells' and task_type == 'regression' and comparison == 'competence_vs_fixed'")
            .iloc[0]["ci_high"] < -0.01
        ),
    }
    gate_pass = bool(passing and all(no_material_harm.values()))
    audit = {
        "protocol": "LOSS_ALIGNED_ROUTING_PROTOCOL.md",
        "test_metadata": metadata,
        "tasks": audit_tasks,
        "passing_task_types": passing,
        "no_material_harm": no_material_harm,
        "performance_opportunity_gate_pass": gate_pass,
        "high_dim_classification_claim_pass": cls["high_dim_high_rho_ci"][0] > 0,
        "interpretation": "Positive gains mean the second-named method has lower loss.",
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    methods = ("fixed", "shape_family", "competence", "best_individual_oracle")
    colors = {"fixed": "#555555", "shape_family": "#d95f02", "competence": "#1b9e77", "best_individual_oracle": "#7570b3"}
    for column, task_type in enumerate(("classification", "regression")):
        cell = summary[summary["task_type"] == task_type]
        phase = cell.groupby(["rho", "method"], as_index=False)["mean_loss"].mean()
        for method in methods:
            curve = phase[phase["method"] == method].sort_values("rho")
            axes[0, column].plot(curve["rho"], curve["mean_loss"], marker="o", color=colors[method], label=method)
        axes[0, column].set_title(task_type.capitalize())
        axes[0, column].set_xlabel("rho")
        axes[0, column].set_ylabel("Test loss")
        axes[0, column].grid(alpha=0.2)
        matrix = np.empty((2, 2))
        task_long = frame[frame["task_type"] == task_type]
        for row, n_context in enumerate((64, 96)):
            for col, n_features in enumerate((8, 12)):
                pivot = task_long[
                    (task_long["context_size"] == n_context)
                    & (task_long["feature_count"] == n_features)
                ].pivot(index="episode_index", columns="method", values="loss")
                matrix[row, col] = (pivot["fixed"] - pivot["competence"]).mean()
        limit = max(abs(matrix.min()), abs(matrix.max()), 1e-6)
        image = axes[1, column].imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit)
        for row in range(2):
            for col in range(2):
                axes[1, column].text(col, row, f"{matrix[row, col]:+.4f}", ha="center", va="center")
        axes[1, column].set_xticks((0, 1), (8, 12))
        axes[1, column].set_yticks((0, 1), (64, 96))
        axes[1, column].set_xlabel("Features")
        axes[1, column].set_ylabel("Context rows")
        axes[1, column].set_title("Competence gain vs fixed")
        fig.colorbar(image, ax=axes[1, column], shrink=0.75)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Context-only loss-aligned routing: untouched test")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
