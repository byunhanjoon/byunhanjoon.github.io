#!/usr/bin/env python3
"""Validate Phase II artifacts and generate the six prespecified figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.io import code_digest  # noqa: E402
from src.analysis.phase1 import collect_records, records_frame, write_json  # noqa: E402
from src.analysis.phase2 import (  # noqa: E402
    MODEL_FAMILY,
    applicable_transformed,
    dataset_transform_summary,
    flatten_descriptors,
    model_summary,
)
from src.analysis.runner import load_config  # noqa: E402


def _save(figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def identity_paired_records(records: list[dict]) -> tuple[list[dict], int]:
    """Retain only cells with the identity baseline required for excess disagreement."""
    identities = {
        (record["dataset"], record["model"], int(record["seed"]), int(record.get("split_seed", 0)))
        for record in records
        if record["transformation"]["name"] == "identity"
    }
    paired = [
        record
        for record in records
        if (record["dataset"], record["model"], int(record["seed"]), int(record.get("split_seed", 0)))
        in identities
    ]
    return paired, len(records) - len(paired)


def plot_matched_mismatch(frame: pd.DataFrame, path: Path) -> None:
    data = applicable_transformed(frame)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    for axis, (label, mask) in zip(
        axes,
        [("classification: JS", data.problem_type != "regression"), ("regression: normalized |Δprediction|", data.problem_type == "regression")],
    ):
        subset = data[mask]
        for condition, color in [("matched", "#4477AA"), ("context_only", "#EE6677"), ("query_only", "#228833")]:
            curve = subset.groupby("transform_severity")[f"{condition}_disagreement"].mean().sort_index()
            axis.plot(curve.index, curve.values, marker="o", label=condition, color=color)
        axis.set(xlabel="declared severity", ylabel=label)
        axis.grid(alpha=.25); axis.legend()
    _save(figure, path)


def plot_model_robustness(summary: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 5))
    for axis, (problem, group) in zip(axes, summary.groupby(summary.problem_type == "regression", sort=True)):
        group = group.sort_values("mean_excess_disagreement")
        x = np.arange(len(group))
        y = group.mean_excess_disagreement.to_numpy()
        errors = np.vstack([y - group.excess_disagreement_ci_low, group.excess_disagreement_ci_high - y])
        colors = [{"TFM":"#4477AA", "tree":"#CC6677", "trained neural":"#228833", "linear":"#AA3377"}.get(x, "gray") for x in group.model_family]
        axis.bar(x, y, color=colors)
        axis.errorbar(x, y, yerr=errors, fmt="none", color="black", capsize=3)
        axis.set_xticks(x, group.model, rotation=45, ha="right")
        axis.set_ylabel("excess disagreement")
        axis.set_title("regression" if problem else "classification")
        axis.axhline(0, color="black", lw=.8); axis.grid(axis="y", alpha=.25)
    _save(figure, path)


def plot_disagreement_loss(frame: pd.DataFrame, path: Path) -> None:
    data = applicable_transformed(frame)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, (label, mask) in zip(axes, [("classification", data.problem_type != "regression"), ("regression", data.problem_type == "regression")]):
        subset = data[mask]
        for family, group in subset.groupby(subset.model.map(MODEL_FAMILY).fillna("other")):
            axis.scatter(group.matched_excess_disagreement, group.matched_normalized_loss_gap, s=12, alpha=.35, label=family)
        axis.axhline(0, color="black", lw=.8); axis.axvline(0, color="black", lw=.8)
        axis.set(xlabel="excess prediction disagreement", ylabel="normalized loss gap", title=label)
        axis.grid(alpha=.2); axis.legend()
    _save(figure, path)


def plot_sensitivity_map(dataset: pd.DataFrame, path: Path) -> None:
    matrix = dataset.groupby(["dataset", "transform"])["matched_normalized_loss_gap"].mean().unstack()
    figure, axis = plt.subplots(figsize=(max(9, .8 * len(matrix.columns)), max(7, .35 * len(matrix))))
    limit = np.nanquantile(np.abs(matrix.to_numpy()), .95) if matrix.notna().any().any() else 1.0
    image = axis.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(matrix.index)), matrix.index)
    figure.colorbar(image, ax=axis, label="normalized matched loss gap")
    _save(figure, path)


def plot_ensemble_single(frame: pd.DataFrame, path: Path) -> None:
    data = applicable_transformed(frame)
    pairs = [("tabpfn_v25_single", "tabpfn_v25_default", "TabPFN v2.5"), ("tabicl_v2_single", "tabicl_v2_default", "TabICLv2")]
    figure, axes = plt.subplots(1, 2, figsize=(11, 5))
    for axis, (single, default, title) in zip(axes, pairs):
        cells = data[data.model.isin([single, default])].groupby(["dataset", "split_seed", "model"])["matched_excess_disagreement"].mean().unstack()
        axis.scatter(cells[single], cells[default], alpha=.7)
        limit = max(float(cells.max().max()), 1e-8)
        axis.plot([0, limit], [0, limit], ls="--", color="black")
        axis.set(xlabel="single estimator", ylabel="default ensemble", title=title); axis.grid(alpha=.2)
    _save(figure, path)


def plot_tree_neural(dataset: pd.DataFrame, path: Path) -> None:
    family = dataset.groupby(["dataset", "split_seed", "model_family"])["matched_excess_disagreement"].mean().unstack()
    figure, axis = plt.subplots(figsize=(6, 5))
    neural = family[[name for name in ["TFM", "trained neural"] if name in family]].mean(axis=1)
    tree = family["tree"]
    axis.scatter(tree, neural, alpha=.75)
    limit = max(float(tree.max()), float(neural.max()), 1e-8)
    axis.plot([0, limit], [0, limit], ls="--", color="black")
    axis.set(xlabel="tree excess disagreement", ylabel="neural excess disagreement", title="Paired dataset/split controls")
    axis.grid(alpha=.2)
    _save(figure, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/audit/main.yaml")
    parser.add_argument("--code-sha256")
    parser.add_argument("--coverage-only", action="store_true")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Generate explicitly exploratory outputs from the currently complete jobs.",
    )
    args = parser.parse_args()
    config_path = args.config.resolve(); config = load_config(config_path)
    selected_code = args.code_sha256 or code_digest(ROOT)
    records, coverage = collect_records(ROOT, config_path, config, selected_code)
    if args.coverage_only:
        print(json.dumps({k: v for k, v in coverage.items() if k != "missing"}, sort_keys=True)); return
    if coverage["missing_jobs"] and not args.allow_partial:
        raise RuntimeError(f"Phase II grid incomplete: {coverage['complete_jobs']}/{coverage['expected_jobs']}")
    is_partial = bool(coverage["missing_jobs"])
    if is_partial:
        output = (
            ROOT
            / "results"
            / "analysis"
            / "phase2_partial"
            / f"{selected_code[:16]}-n{coverage['complete_jobs']}"
        )
        marker = output / "PARTIAL.json"
    else:
        output = ROOT / "results" / "analysis" / "phase2" / selected_code[:16]
        marker = output / "DONE.json"
    if marker.exists(): print(output); return
    output.mkdir(parents=True, exist_ok=True)
    excluded_without_identity = 0
    analysis_records = records
    if is_partial:
        analysis_records, excluded_without_identity = identity_paired_records(records)
    frame = records_frame(analysis_records)
    # Add structural fields that are intentionally omitted from the Phase I table.
    lookup = {record["run_id"]: record for record in analysis_records}
    frame["transform_scope"] = frame.run_id.map(lambda run: lookup[run]["transform_scope"])
    frame["n_numeric"] = frame.run_id.map(lambda run: lookup[run]["features"]["numeric"])
    frame["n_categorical"] = frame.run_id.map(lambda run: lookup[run]["features"]["categorical"])
    summary = model_summary(frame)
    dataset = dataset_transform_summary(frame)
    descriptors = flatten_descriptors(analysis_records)
    frame.to_csv(output / "run_metrics.csv", index=False)
    summary.to_csv(output / "model_summary.csv", index=False)
    dataset.to_csv(output / "dataset_transform_effects.csv", index=False)
    descriptors.to_csv(output / "feature_descriptors.csv", index=False)
    write_json(output / "coverage.json", coverage)
    plot_matched_mismatch(frame, output / "matched_vs_mismatch.png")
    plot_model_robustness(summary, output / "model_family_robustness.png")
    plot_disagreement_loss(frame, output / "disagreement_vs_loss.png")
    plot_sensitivity_map(dataset, output / "dataset_sensitivity_map.png")
    plot_ensemble_single(frame, output / "ensemble_vs_single.png")
    plot_tree_neural(dataset, output / "tree_vs_neural.png")
    write_json(
        marker,
        {
            **coverage,
            "hierarchical_bootstrap_draws": 10_000,
            "analyzed_identity_paired_jobs": len(analysis_records),
            "excluded_jobs_without_identity_baseline": excluded_without_identity,
            "analysis_status": "exploratory_partial" if is_partial else "complete_confirmatory",
            "missingness_warning": (
                "Model-dependent incomplete coverage; cross-model comparisons may be biased."
                if is_partial
                else None
            ),
        },
    )
    print(output)


if __name__ == "__main__":
    main()
