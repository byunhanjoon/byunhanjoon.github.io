#!/usr/bin/env python3
"""Generate frozen Phase I tables and plots from immutable raw predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.io import code_digest  # noqa: E402
from src.analysis.phase1 import (  # noqa: E402
    collect_records,
    dataset_level_summary,
    records_frame,
    transform_summary,
    write_json,
)
from src.analysis.runner import load_config  # noqa: E402


def _plot_model_landscape(dataset, path: Path) -> None:
    models = sorted(dataset["model"].unique())
    figure, axes = plt.subplots(2, 2, figsize=(max(10, len(models) * 1.3), 8.5), squeeze=False)
    colors = ["#4477AA" if model not in {"xgboost", "catboost", "lightgbm"} else "#CC6677" for model in models]
    groups = [("classification", dataset["problem_type"] != "regression"), ("regression", dataset["problem_type"] == "regression")]
    for row, (label, mask) in enumerate(groups):
        subset = dataset[mask]
        loss = subset.groupby("model")["matched_normalized_loss_gap"].mean().reindex(models)
        disagreement = subset.groupby("model")["matched_excess_disagreement"].mean().reindex(models)
        axes[row][0].bar(models, loss, color=colors)
        axes[row][0].set_ylabel(f"{label}: normalized loss gap")
        axes[row][1].bar(models, disagreement, color=colors)
        metric = "JS minus identity noise" if label == "classification" else "normalized |prediction| difference minus identity noise"
        axes[row][1].set_ylabel(f"{label}: {metric}")
        for axis in axes[row]:
            axis.axhline(0, color="black", linewidth=0.8)
            axis.tick_params(axis="x", rotation=45)
            axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_matched_mismatch(frame, path: Path) -> None:
    transformed = frame[frame["transform"] != "identity"]
    models = sorted(transformed["model"].unique())
    positions = np.arange(len(models))
    figure, axes = plt.subplots(1, 2, figsize=(max(12, len(models) * 2.2), 4.8), squeeze=False)
    width = 0.36
    groups = [("classification (JS)", transformed["problem_type"] != "regression"), ("regression (normalized absolute)", transformed["problem_type"] == "regression")]
    for axis, (label, mask) in zip(axes[0], groups):
        subset = transformed[mask]
        matched = subset.groupby("model")["matched_disagreement"].median().reindex(models)
        mismatch = subset.groupby("model")["mean_mismatch_disagreement"].median().reindex(models)
        axis.bar(positions - width / 2, matched, width, label="matched", color="#4477AA")
        axis.bar(positions + width / 2, mismatch, width, label="mean mismatch", color="#EE6677")
        axis.set_xticks(positions, models, rotation=45, ha="right")
        axis.set_ylabel(label)
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_severity(summary, path: Path) -> None:
    transforms = sorted(summary["transform"].unique())
    figure, axes = plt.subplots(2, len(transforms), figsize=(5 * len(transforms), 8.5), squeeze=False)
    groups = [("classification", summary["problem_type"] != "regression"), ("regression", summary["problem_type"] == "regression")]
    for row, (problem_label, problem_mask) in enumerate(groups):
        for column, transform in enumerate(transforms):
            axis = axes[row][column]
            subset = summary[(summary["transform"] == transform) & problem_mask]
            for model, group in subset.groupby("model"):
                group = group.groupby("transform_severity", as_index=False)["mean_excess_disagreement"].mean()
                group = group.sort_values("transform_severity")
                axis.plot(group["transform_severity"], group["mean_excess_disagreement"], marker="o", label=model)
            axis.axhline(0, color="black", linewidth=0.8)
            axis.set_title(f"{problem_label}: {transform}")
            axis.set_xlabel("Declared severity")
            axis.grid(alpha=0.25)
    axes[0][0].set_ylabel("JS minus identity noise")
    axes[1][0].set_ylabel("Normalized disagreement minus identity noise")
    axes[0][-1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "audit" / "pilot.yaml")
    parser.add_argument("--code-sha256")
    parser.add_argument("--coverage-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    selected_code = args.code_sha256 or code_digest(ROOT)
    records, coverage = collect_records(ROOT, config_path, config, selected_code)
    if args.coverage_only:
        print(json.dumps({key: value for key, value in coverage.items() if key != "missing"}, sort_keys=True))
        return
    if coverage["missing_jobs"]:
        raise RuntimeError(
            f"Phase I grid is incomplete for code {selected_code}: "
            f"{coverage['complete_jobs']}/{coverage['expected_jobs']} complete"
        )
    output = ROOT / "results" / "analysis" / "phase1" / selected_code[:16]
    done = output / "DONE.json"
    if done.exists():
        print(output)
        return
    # A failed integrity check may leave an empty/incomplete directory but never
    # a DONE marker.  All generated files below are deterministic replacements,
    # so a no-DONE rerun is safe and preserves immutable raw artifacts.
    output.mkdir(parents=True, exist_ok=True)
    frame = records_frame(records)
    dataset, model_summary = dataset_level_summary(frame)
    by_transform = transform_summary(frame)
    frame.to_csv(output / "run_metrics.csv", index=False)
    dataset.to_csv(output / "dataset_effects.csv", index=False)
    model_summary.to_csv(output / "model_summary.csv", index=False)
    by_transform.to_csv(output / "transform_summary.csv", index=False)
    write_json(output / "coverage.json", coverage)
    _plot_model_landscape(dataset, output / "model_landscape.png")
    _plot_matched_mismatch(frame, output / "matched_vs_mismatch.png")
    _plot_severity(by_transform, output / "severity_curves.png")
    report_lines = [
        "# Phase I kill-test analysis draft",
        "",
        "This draft is generated only after the frozen job grid and all artifact integrity checks complete.",
        "",
        "## Question",
        "",
        "Do current tabular foundation models change predictions or performance under fully matched, invertible numerical reparameterizations beyond identity-refit noise and more than tree controls?",
        "",
        "## Exact protocol",
        "",
        f"- Config: `{config_path}` (`{coverage['config_sha256']}`)",
        f"- Code digest: `{selected_code}`",
        f"- Complete jobs: {coverage['complete_jobs']}/{coverage['expected_jobs']}",
        f"- Datasets: {len(config['datasets'])}; seeds: {config['seeds']}; context/query caps: {config.get('max_context')}/{config.get('max_query')}",
        "- Each job uses two fitted contexts: original for clean/query-only and transformed for matched/context-only.",
        "- Primary aggregate unit is dataset. Confidence intervals are 10,000-draw paired dataset bootstraps.",
        "",
        "## Result table",
        "",
        "| Model | Task type | Datasets | Normalized loss gap (95% CI) | Excess disagreement (95% CI) | W/T/L |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in model_summary.itertuples(index=False):
        report_lines.append(
            f"| {row.model} | {row.problem_type} | {row.datasets} | "
            f"{row.mean_matched_normalized_loss_gap:.5g} [{row.loss_gap_ci_low:.5g}, {row.loss_gap_ci_high:.5g}] | "
            f"{row.mean_excess_disagreement:.5g} [{row.excess_disagreement_ci_low:.5g}, {row.excess_disagreement_ci_high:.5g}] | "
            f"{row.loss_wins}/{row.loss_ties}/{row.loss_losses} |"
        )
    report_lines += [
        "",
        "## Plots",
        "",
        "- `model_landscape.png`",
        "- `matched_vs_mismatch.png`",
        "- `severity_curves.png`",
        "",
        "## Interpretation and alternative explanations",
        "",
        "Interpretation and Gate G1 are intentionally left for evidence review. Check identity-refit noise, transform severity, default-vs-single ensembles, categorical preprocessing, optimization nondeterminism, and any model-specific preprocessing before attributing an effect to the learned prior.",
        "",
        "## Raw results",
        "",
        f"- Manifest: `{ROOT / 'results' / 'MANIFEST.jsonl'}`",
        f"- Validated run paths and metrics: `{output / 'run_metrics.csv'}`",
        "",
    ]
    (output / "phase1_report_draft.md").write_text("\n".join(report_lines))
    write_json(
        done,
        {
            **coverage,
            "artifacts": [
                "run_metrics.csv",
                "dataset_effects.csv",
                "model_summary.csv",
                "transform_summary.csv",
                "coverage.json",
                "model_landscape.png",
                "matched_vs_mismatch.png",
                "severity_curves.png",
                "phase1_report_draft.md",
            ],
        },
    )
    print(output)


if __name__ == "__main__":
    main()
