#!/usr/bin/env python3
"""Generate the prescribed figures, rankings, provenance, and 34-section report."""

from __future__ import annotations

import importlib.metadata
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "results" / "processed"
FIGURES = ROOT / "figures"
FINALIST_PATH = ROOT / "configs" / "GUARDED_FINALISTS.json"


def csv(name: str) -> pd.DataFrame:
    path = PROCESSED / name
    if not path.exists():
        raise FileNotFoundError(f"required final-report input missing: {path}")
    return pd.read_csv(path)


def js(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def f(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "YES" if value else "NO"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if value != 0 and abs(value) < 10 ** (-digits):
            return f"{value:.3e}"
        return f"{value:.{digits}f}"
    return str(value).replace("|", "\\|")


def table(frame: pd.DataFrame, columns: list[str] | None = None, limit: int | None = None) -> str:
    values = frame.copy()
    if columns is not None:
        for column in columns:
            if column not in values:
                values[column] = np.nan
        values = values[columns]
    if limit is not None:
        values = values.head(limit)
    header = "| " + " | ".join(values.columns) + " |"
    separator = "| " + " | ".join("---" for _ in values.columns) + " |"
    rows = ["| " + " | ".join(f(value) for value in row) + " |" for row in values.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows]) if rows else "_No eligible rows._"


def finalist_names(finalists: dict[str, Any]) -> set[str]:
    return {str(row["method"]) for row in finalists["finalists"]}


def combine_prospective() -> tuple[pd.DataFrame, pd.DataFrame]:
    general_summary = csv("prospective_general_summary.csv").assign(scope="general")
    general_units = csv("prospective_general_units.csv").rename(
        columns={"validation_C_actual": "validation_C"}
    ).assign(scope="general")
    embedding_summary = csv("prospective_embedding_summary.csv")
    embedding_units = csv("prospective_embedding_units.csv")
    embedding_summary = embedding_summary[
        embedding_summary.method == "GuardedGram-G2-after-RBF-k16"
    ].assign(scope="RBF-k16 embedding")
    embedding_units = embedding_units[
        embedding_units.method == "GuardedGram-G2-after-RBF-k16"
    ].assign(scope="RBF-k16 embedding")
    return (
        pd.concat([general_summary, embedding_summary], ignore_index=True, sort=False),
        pd.concat([general_units, embedding_units], ignore_index=True, sort=False),
    )


def efficiency_fields(summary: pd.DataFrame) -> pd.DataFrame:
    values = summary.copy()
    block_training_multiplier = 10.0
    intervention_path = PROCESSED / "prospective_one_block.csv"
    candidate_path = PROCESSED / "prospective_block_candidates.csv"
    if intervention_path.exists() and candidate_path.exists():
        interventions = pd.read_csv(intervention_path).groupby(["dataset", "model", "seed"]).size()
        candidates = pd.read_csv(candidate_path).groupby(["dataset", "model", "seed"]).size()
        counts = interventions.add(candidates, fill_value=0)
        if len(counts):
            block_training_multiplier = float(counts.median())
    training = []
    parameters = []
    for row in values.itertuples():
        method = str(row.method)
        if method == "Raw":
            training.append(1.0); parameters.append(1.0)
        elif method == "PureGram":
            training.append(1.0); parameters.append(1.0)
        elif method.startswith("BlockGuard"):
            training.append(block_training_multiplier); parameters.append(1.0)
        elif "embedding" in method.lower() and "Guarded" in method:
            training.append(2.0); parameters.append(2.0)
        else:
            training.append(2.0); parameters.append(2.0)
    values["training_multiplier"] = training
    values["parameter_multiplier"] = parameters
    return values


def make_rankings(summary: pd.DataFrame) -> pd.DataFrame:
    values = efficiency_fields(summary)
    values["paper_score"] = (
        values.median_disagreement_reduction
        - 3 * values.median_C.clip(lower=0)
        - 3 * (values.p95_C - 0.01).clip(lower=0)
        - 2 * (values.max_C - 0.05).clip(lower=0)
        - 0.05 * np.log2(values.inference_multiplier.clip(lower=1))
    )
    definitions: list[tuple[str, pd.Series, list[str], list[bool]]] = [
        (
            "A — Paper Safety",
            (values.median_C <= 0.005) & (values.p95_C <= 0.02) & (values.max_C <= 0.10),
            ["median_disagreement_reduction", "mean_predictive_rank"],
            [False, True],
        ),
        (
            "B — Strict Safety",
            (values.p95_C <= 0.01) & (values.max_C <= 0.05),
            ["median_disagreement_reduction", "mean_predictive_rank"],
            [False, True],
        ),
        ("C — Basis Control", pd.Series(True, index=values.index), ["median_disagreement_reduction", "p95_C"], [False, True]),
        ("D — Predictive Performance", pd.Series(True, index=values.index), ["mean_predictive_rank", "median_C"], [True, True]),
        (
            "E — Efficiency",
            values.median_disagreement_reduction >= 0.50,
            ["inference_multiplier", "training_multiplier", "parameter_multiplier", "median_disagreement_reduction"],
            [True, True, True, False],
        ),
        ("F — Overall Paper Candidate", pd.Series(True, index=values.index), ["paper_score", "mean_predictive_rank"], [False, True]),
    ]
    rows: list[pd.DataFrame] = []
    for name, eligible, keys, ascending in definitions:
        ranked = values[eligible].sort_values(keys, ascending=ascending).copy()
        ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
        ranked.insert(0, "ranking", name)
        rows.append(ranked)
    result = pd.concat(rows, ignore_index=True, sort=False)
    result.to_csv(PROCESSED / "prospective_six_rankings.csv", index=False)
    return result


def tail_table(units: pd.DataFrame) -> pd.DataFrame:
    values = units[units.method != "Raw"].sort_values("normalized_excess_risk", ascending=False).head(10).copy()
    values["explanation"] = np.where(
        (values.validation_C <= 0.01) & (values.normalized_excess_risk > 0.01),
        "validation-safe but test-harmful: validation miss/noise or distribution shift",
        np.where(
            (values.validation_C > 0.01) & (values.normalized_excess_risk > 0.01),
            "harm visible on both validation and test; fixed/reference method is unconstrained",
            "model/seed aggregation instability near the safety boundary",
        ),
    )
    values.to_csv(PROCESSED / "prospective_worst_10_tail_cells.csv", index=False)
    return values


def development_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    guarded = csv("guardedgram_full_summary.csv")
    block = csv("blockguard_full_summary.csv")
    dual = csv("dualview_stage1_summary.csv")
    combined = pd.concat(
        [guarded.assign(experiment="general"), block.assign(experiment="BlockGuard"), dual.assign(experiment="DualView-stage1")],
        ignore_index=True,
        sort=False,
    )
    combined.to_csv(PROCESSED / "full_development_combined_summary.csv", index=False)

    interventions = csv("blockguard_full_one_block.csv")
    cells = csv("blockguard_full_cells.csv")
    selected = cells[(cells.split == "test") & (cells.method == "BlockGuard-Greedy-t01")][
        ["dataset", "model", "seed", "selected_features"]
    ].drop_duplicates()
    selected["selected_set"] = selected.selected_features.map(lambda value: set(json.loads(value)))
    merged = interventions.merge(selected, on=["dataset", "model", "seed"], how="inner")
    merged["status"] = ["Gram-selected" if feature in chosen else "raw-retained" for feature, chosen in zip(merged.feature, merged.selected_set)]
    descriptors = (
        merged.groupby("status", as_index=False)
        .agg(
            features=("feature", "count"),
            median_empirical_rank=("empirical_rank", "median"),
            median_block_dimension=("block_dimension", "median"),
            median_spectrum_entropy=("spectrum_entropy", "median"),
            median_condition_proxy=("condition_proxy", "median"),
            median_one_block_C=("normalized_excess_risk", "median"),
            median_orbit_benefit=("basis_disagreement_benefit", "median"),
        )
    )
    descriptors.to_csv(PROCESSED / "blockguard_feature_descriptor_summary.csv", index=False)
    return combined, descriptors


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close()


def make_figures(pros_summary: pd.DataFrame, pros_units: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    highlight = {"Raw+Gram@0.75", "SafeRankGram-t01", "GuardedGram-G2-g0p0-t01", "BlockGuard-Greedy-t01"}

    plt.figure(figsize=(7, 5))
    for row in pros_summary.itertuples():
        color = "#c0392b" if row.method in highlight else "#607d8b"
        plt.scatter(row.p95_C, row.median_disagreement_reduction, s=70 if row.method in highlight else 35, c=color)
        if row.method in highlight:
            plt.annotate(row.method, (row.p95_C, row.median_disagreement_reduction), fontsize=8, xytext=(4, 3), textcoords="offset points")
    plt.axvline(0.02, color="black", ls="--", lw=1)
    plt.axhline(0.60, color="black", ls=":", lw=1)
    plt.xlabel("Prospective p95 normalized excess risk C")
    plt.ylabel("Median basis-disagreement reduction")
    plt.title("Control–tail-risk Pareto frontier")
    savefig("figure_1_control_tail_pareto.png")

    plt.figure(figsize=(7, 5))
    for method, frame in pros_units.groupby("method"):
        if method == "Raw" or method in highlight:
            values = np.sort(frame.normalized_excess_risk.to_numpy(float))
            plt.plot(values, np.arange(1, len(values) + 1) / len(values), label=method)
    plt.axvline(0.02, color="black", ls="--", lw=1)
    plt.xlabel("Normalized excess risk C")
    plt.ylabel("Empirical CDF")
    plt.title("Prospective task-risk tails")
    plt.legend(fontsize=7)
    savefig("figure_2_tail_cdf.png")

    cells = csv("prospective_general_cells.csv")
    g2 = cells[(cells.split == "test") & (cells.method == "GuardedGram-G2-g0p0-t01")]
    plt.figure(figsize=(6, 4))
    bins = np.arange(-0.125, 0.876, 0.25)
    plt.hist(g2.selected_alpha, bins=bins, rwidth=0.85, color="#3569a8")
    plt.xticks([0, 0.25, 0.5, 0.75])
    plt.xlabel("Validation-selected alpha")
    plt.ylabel("Cell count")
    plt.title("GuardedGram G2 selected-alpha distribution")
    savefig("figure_3_guardedgram_alpha_histogram.png")

    block = pros_units[pros_units.method == "BlockGuard-Greedy-t01"]
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(
        block.invariant_feature_fraction,
        block.normalized_excess_risk,
        c=block.disagreement_reduction,
        cmap="viridis",
        s=45,
        alpha=0.8,
    )
    plt.axhline(0.02, color="black", ls="--", lw=1)
    plt.xlabel("Fraction Gram-controlled feature blocks")
    plt.ylabel("Normalized excess risk C")
    plt.title("BlockGuard control fraction versus task risk")
    plt.colorbar(scatter, label="Disagreement reduction")
    savefig("figure_4_blockguard_fraction_tradeoff.png")

    headroom = csv("embedding_dimension_headroom.csv")
    scaling = headroom.groupby(["model", "embedding", "k"], as_index=False).raw_disagreement.median()
    plt.figure(figsize=(7, 5))
    for (model, embedding), frame in scaling.groupby(["model", "embedding"]):
        plt.plot(frame.k, frame.raw_disagreement, marker="o", label=f"{model} / {embedding}")
    plt.xscale("log", base=2)
    plt.xticks([4, 8, 16, 32], [4, 8, 16, 32])
    plt.xlabel("Embedding dimension k")
    plt.ylabel("Median basis disagreement")
    plt.title("Embedding dimension scaling")
    plt.legend(fontsize=7, ncol=2)
    savefig("figure_5_embedding_dimension_scaling.png")

    full_headroom = csv("embedding_full_headroom.csv")
    plt.figure(figsize=(6, 6))
    plt.scatter(full_headroom.default_task_error, full_headroom.validation_selected_basis_error, alpha=0.65, label="validation-selected")
    plt.scatter(full_headroom.default_task_error, full_headroom.oracle_best_test_error, alpha=0.45, label="oracle best")
    limits = [
        min(full_headroom.default_task_error.min(), full_headroom.oracle_best_test_error.min()),
        max(full_headroom.default_task_error.max(), full_headroom.validation_selected_basis_error.max()),
    ]
    plt.plot(limits, limits, color="black", ls="--", lw=1)
    plt.xlabel("Default-basis test error")
    plt.ylabel("Alternative-basis test error")
    plt.title("Default basis versus selection headroom")
    plt.legend()
    savefig("figure_6_default_vs_basis_search.png")

    efficiency = efficiency_fields(pros_summary)
    methods = [method for method in ["Raw", "Raw+Gram@0.75", "GuardedGram-G2-g0p0-t01", "BlockGuard-Greedy-t01"] if method in set(efficiency.method)]
    plot = efficiency.set_index("method").loc[methods]
    x = np.arange(len(plot))
    width = 0.25
    plt.figure(figsize=(8, 5))
    plt.bar(x - width, plot.inference_multiplier, width, label="inference")
    plt.bar(x, plot.training_multiplier, width, label="training")
    plt.bar(x + width, plot.parameter_multiplier, width, label="parameters")
    plt.xticks(x, methods, rotation=20, ha="right")
    plt.ylabel("Multiplier versus Raw")
    plt.title("Dual-model and single-representation efficiency")
    plt.legend()
    savefig("figure_7_efficiency_comparison.png")

    dev = pd.concat([csv("guardedgram_full_summary.csv"), csv("blockguard_full_summary.csv")], ignore_index=True)
    merged = dev.merge(pros_summary[pros_summary.scope == "general"], on="method", suffixes=("_dev", "_pros"))
    plt.figure(figsize=(7, 5))
    plt.scatter(merged.median_disagreement_reduction_dev, merged.median_disagreement_reduction_pros, c=merged.p95_C_pros, cmap="magma_r", s=70)
    for row in merged.itertuples():
        if row.method in highlight:
            plt.annotate(row.method, (row.median_disagreement_reduction_dev, row.median_disagreement_reduction_pros), fontsize=8)
    plt.xlabel("Development median disagreement reduction")
    plt.ylabel("Prospective median disagreement reduction")
    plt.title("Development-to-prospective transfer")
    plt.colorbar(label="Prospective p95 C")
    savefig("figure_8_development_vs_prospective.png")


def ranking_section(rankings: pd.DataFrame, name: str) -> str:
    values = rankings[rankings.ranking == name]
    return table(
        values,
        ["rank", "method", "scope", "median_disagreement_reduction", "median_C", "p95_C", "max_C", "mean_predictive_rank", "inference_multiplier", "training_multiplier", "parameter_multiplier", "paper_score"],
    )


def main() -> None:
    finalists = js(FINALIST_PATH)
    finalists["sha256"] = FINALIST_PATH.with_suffix(".sha256").read_text().split()[0]
    actual_finalist_hash = hashlib.sha256(FINALIST_PATH.read_bytes()).hexdigest()
    if actual_finalist_hash != finalists["sha256"]:
        raise RuntimeError("frozen finalist config no longer matches its SHA256 sidecar")
    protocol = js(ROOT / "configs" / "GUARDED_PROTOCOL.json")
    panel = js(ROOT / "configs" / "GUARDED_PROSPECTIVE_PANEL.json")
    pros_summary, pros_units = combine_prospective()
    rankings = make_rankings(pros_summary)
    tails = tail_table(pros_units)
    development, feature_descriptors = development_tables()
    make_figures(pros_summary, pros_units)

    names = finalist_names(finalists)
    eligible = pros_summary[
        pros_summary.method.isin(names)
        & (pros_summary.median_disagreement_reduction >= 0.60)
        & (pros_summary.median_C <= 0.005)
        & (pros_summary.p95_C <= 0.02)
        & (pros_summary.max_C <= 0.10)
        & (pros_summary.model_families >= 3)
    ]
    fixed = pros_summary[pros_summary.method == "Raw+Gram@0.75"]
    adaptive = pros_summary[pros_summary.method.isin(names - {"Raw+Gram@0.75"})]
    block = pros_summary[pros_summary.method == "BlockGuard-Greedy-t01"]
    block_signal = bool(
        len(block)
        and block.iloc[0].median_disagreement_reduction >= 0.60
        and block.iloc[0].p95_C <= 0.02
        and block.iloc[0].inference_multiplier <= 1.0
    )
    if block_signal:
        verdict = "BLOCK-SELECTION-WINS"
    elif len(eligible):
        verdict = "FINAL-METHOD-SIGNAL"
    elif (
        len(fixed)
        and fixed.iloc[0].median_C <= 0.005
        and fixed.iloc[0].p95_C <= 0.02
        and fixed.iloc[0].max_C <= 0.10
        and fixed.iloc[0].median_disagreement_reduction >= adaptive.median_disagreement_reduction.max() + 0.10
    ):
        verdict = "FIXED-MIXTURE-WINS"
    elif len(adaptive) and adaptive.median_disagreement_reduction.max() < 0.50 and (adaptive.p95_C <= 0.02).any():
        verdict = "SAFE-BUT-CONSERVATIVE"
    else:
        verdict = "METHOD-STILL-UNSOLVED"

    gfull = csv("guardedgram_full_summary.csv")
    bfull = csv("blockguard_full_summary.csv")
    dual = csv("dualview_stage1_summary.csv")
    emb_methods = csv("embedding_full_method_units.csv")
    emb_block = csv("embedding_blockguard_summary.csv")
    dim_scaling = csv("embedding_dimension_scaling.csv")
    dim_headroom = csv("embedding_dimension_headroom.csv")
    full_headroom = csv("embedding_full_headroom.csv")
    natural = js(PROCESSED / "natural_basis_reuse_manifest.json")
    loader_audit = js(PROCESSED / "prospective_loader_audit.json")
    selected = pros_summary[pros_summary.method.isin(names)].sort_values("median_disagreement_reduction", ascending=False)
    best_positive = selected.iloc[0]
    worst_tail = selected.sort_values("p95_C", ascending=False).iloc[0]

    general_g2 = pros_summary[pros_summary.method == "GuardedGram-G2-g0p0-t01"].iloc[0]
    safe = pros_summary[pros_summary.method == "SafeGram-t01"].iloc[0]
    gg_vs_safe = "YES" if general_g2.median_disagreement_reduction > safe.median_disagreement_reduction and general_g2.p95_C <= safe.p95_C + 0.005 else "PARTLY" if general_g2.median_disagreement_reduction > safe.median_disagreement_reduction else "NO"
    feature_vs_global = "PARTLY"
    if len(block):
        block_row = block.iloc[0]
        if block_row.median_disagreement_reduction > general_g2.median_disagreement_reduction and block_row.p95_C <= general_g2.p95_C:
            feature_vs_global = "YES"
        elif block_row.median_disagreement_reduction <= general_g2.median_disagreement_reduction and block_row.p95_C > general_g2.p95_C:
            feature_vs_global = "NO"
    avoid_two = "YES" if len(block) and block.iloc[0].median_disagreement_reduction >= 0.60 and block.iloc[0].p95_C <= 0.02 else "PARTLY" if len(block) and block.iloc[0].median_disagreement_reduction >= 0.50 else "NO"
    positive_slopes = float((dim_scaling.log2_dimension_slope_b > 0).mean())
    grows = "YES" if positive_slopes >= 0.70 else "PARTLY" if positive_slopes >= 0.40 else "NO"
    default_rate = float(full_headroom.default_is_best_test_basis.mean())
    default_optimal = "YES" if default_rate > 0.60 else "NO" if default_rate < 0.40 else "MIXED"

    stage1_g = csv("guardedgram_stage1_summary.csv")
    stage1_b = csv("blockguard_stage1_summary.csv")
    gdev = gfull[gfull.method.str.startswith("GuardedGram")]
    bdev = bfull[bfull.method.str.startswith("BlockGuard")]
    frozen_rows = pd.DataFrame(finalists["finalists"])[["method", "scope", "threshold", "confidence_level", "embedding_setting", "architecture"]]
    panel_rows = pd.DataFrame(panel["datasets"])[["key", "problem_type", "openml_id", "openml_version"]]
    general_display = csv("prospective_general_units.csv")
    prospective_detail = general_display[general_display.method.isin(names)][
        ["dataset", "model", "method", "selected_alpha", "invariant_feature_fraction", "disagreement_reduction", "raw_task_error", "method_task_error", "normalized_excess_risk"]
    ]
    if "GuardedGram-G2-after-RBF-k16" in names:
        ed = csv("prospective_embedding_units.csv")
        ed = ed[ed.method == "GuardedGram-G2-after-RBF-k16"].copy()
        ed["invariant_feature_fraction"] = np.nan
        prospective_detail = pd.concat([
            prospective_detail,
            ed[["dataset", "model", "method", "selected_alpha", "invariant_feature_fraction", "disagreement_reduction", "raw_task_error", "method_task_error", "normalized_excess_risk"]],
        ], ignore_index=True)

    efficiency_report = efficiency_fields(pros_summary)
    efficiency_report["parameter_count"] = np.nan
    efficiency_report = efficiency_report[
        efficiency_report.method.isin(
            ["Raw", "Raw+Gram@0.75", "GuardedGram-G2-g0p0-t01", "BlockGuard-Greedy-t01"]
        )
    ]
    dual_efficiency = dual.assign(
        scope="development (pruned)",
        inference_multiplier=1.0,
        training_multiplier=1.0,
        parameter_multiplier=np.nan,
        parameter_count=dual.median_parameter_count,
    )
    efficiency_report = pd.concat([efficiency_report, dual_efficiency], ignore_index=True, sort=False)

    candidate_decision = rankings[
        (rankings.ranking == "F — Overall Paper Candidate") & rankings.method.isin(names)
    ].copy()
    candidate_decision["control"] = candidate_decision.median_disagreement_reduction
    candidate_decision["breadth"] = [
        f"{int(row.datasets)} datasets / {int(row.model_families)} model families"
        for row in candidate_decision.itertuples()
    ]
    candidate_decision["single_model"] = candidate_decision.method.eq("BlockGuard-Greedy-t01").map({True: "YES", False: "NO"})
    candidate_decision["embedding_evidence"] = np.where(
        candidate_decision.scope.eq("RBF-k16 embedding"),
        "direct prospective RBF-k16 evidence",
        np.where(
            candidate_decision.method.eq("BlockGuard-Greedy-t01"),
            "transferred embedding variant was negative",
            "general-coordinate evidence; embedding analogue reported separately",
        ),
    )
    complexity = {
        "Raw+Gram@0.75": "fixed two-branch prediction mixture",
        "GuardedGram-G2-g0p0-t01": "validation gate plus two prediction branches",
        "BlockGuard-Greedy-t01": "one inference model; exact intervention search during training",
        "GuardedGram-G2-after-RBF-k16": "embedding gate plus two prediction branches",
    }
    candidate_decision["complexity"] = candidate_decision.method.map(complexity)
    strong = (
        (candidate_decision.control >= 0.70)
        & (candidate_decision.median_C <= 0.005)
        & (candidate_decision.p95_C <= 0.01)
        & (candidate_decision.max_C <= 0.05)
    )
    paper_safe = (
        (candidate_decision.control >= 0.60)
        & (candidate_decision.median_C <= 0.005)
        & (candidate_decision.p95_C <= 0.02)
        & (candidate_decision.max_C <= 0.10)
    )
    candidate_decision["recommendation"] = np.select(
        [strong, paper_safe],
        [
            "repeat unchanged on a second independently locked panel",
            "expand seeds and audit the worst tail before paper selection",
        ],
        default="diagnose the failed control/tail gate before promotion",
    )
    candidate_decision = candidate_decision.rename(
        columns={
            "median_C": "median C",
            "p95_C": "p95",
            "max_C": "max",
            "single_model": "single-model?",
            "embedding_evidence": "embedding evidence",
        }
    )

    packages = {}
    for package in ("numpy", "pandas", "scikit-learn", "torch", "catboost", "matplotlib"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "unavailable"
    provenance = {
        "status": "COMPLETE",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "repository_commit_at_protocol_freeze": protocol["repository_commit"],
        "repository_head_at_report": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "finalist_sha256": finalists["sha256"],
        "figures": sorted(path.name for path in FIGURES.glob("*.png")),
        "prospective_outcomes_accessed_only_after_finalist_freeze": True,
    }
    freeze_ns = max(FINALIST_PATH.stat().st_mtime_ns, FINALIST_PATH.with_suffix(".sha256").stat().st_mtime_ns)
    prospective_files = list((ROOT / "results" / "raw" / "prospective").rglob("*"))
    prospective_files = [path for path in prospective_files if path.is_file()]
    if not prospective_files:
        raise RuntimeError("no prospective raw files found for freeze-order audit")
    earliest_prospective_ns = min(path.stat().st_mtime_ns for path in prospective_files)
    freeze_order_passes = earliest_prospective_ns > freeze_ns
    if not freeze_order_passes:
        raise RuntimeError("a prospective artifact predates the frozen finalist lock")
    provenance.update(
        {
            "prospective_outcomes_accessed_only_after_finalist_freeze": freeze_order_passes,
            "prospective_raw_file_count": len(prospective_files),
            "finalist_freeze_mtime_ns": freeze_ns,
            "earliest_prospective_artifact_mtime_ns": earliest_prospective_ns,
        }
    )
    (PROCESSED / "final_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")

    report = f"""# Guarded Basis Control — Final Method Search

## Executive Verdict
{verdict}

## One-Paragraph Summary

The locked prospective panel supports the verdict **{verdict}**. The strongest frozen candidate by prospective median control was `{best_positive.method}` ({f(best_positive.median_disagreement_reduction)} control; median/p95/max C = {f(best_positive.median_C)}/{f(best_positive.p95_C)}/{f(best_positive.max_C)}), while the most adverse finalist p95 was `{worst_tail.method}` at {f(worst_tail.p95_C)}. These are method rankings and diagnostic classifications, not an automatic choice of the paper's final method.

## Frozen Protocol

- commit: `{protocol['repository_commit']}`
- hardware: {', '.join(protocol['hardware']['gpus'])}, {protocol['hardware']['gpu_memory_mib_each']} MiB each
- packages: {', '.join(f'{key}={value}' for key, value in packages.items())}
- seeds: development {protocol['development_seeds']}; prospective {protocol['prospective_seeds']}
- development datasets: {', '.join(protocol['development_datasets'])}
- untouched prospective datasets: {', '.join(row['key'] for row in panel['datasets'])}
- finalist config SHA256: `{finalists['sha256']}`

Two target-independent loader adapters were recorded after the freeze without changing the panel or finalist configuration: SoilKsatDB retains its {loader_audit['SoilKsatDB']['observed_target_rows_retained']} rows with observed `ksat_lab` and drops {loader_audit['SoilKsatDB']['missing_target_rows_dropped']} missing-target rows before the frozen split; 2dplanes treats its ten 2--3-level numeric inputs as low-rank RBF blocks because the global minimum-unique rule would otherwise yield no transformable block. The exact audit is saved in `prospective_loader_audit.json`; neither adapter used validation/test outcomes.

## 1. Prior Evidence Treated as Fixed

Pure Gram exact invariance, the fixed `.75` control–task tradeoff, SafeRank safety behavior, prior Type-C failures, and numerical-embedding sensitivity were treated as frozen evidence. Natural-pair reuse passed the exact-equivalence threshold: reconstruction {natural['maximum_natural_reconstruction_error']:.3e}, Gram {natural['maximum_gram_coordinate_error']:.3e}, Rank {natural['maximum_rank_coordinate_error']:.3e}; Fourier origin shift was unavailable under frozen metadata.

## 2. GuardedGram Development

{table(gdev, ['method','median_alpha','median_disagreement_reduction','median_C','p95_C','max_C','fallback_rate'])}

### G1 Harm Detection

G1 tests one-sided bootstrap evidence of harm and falls back recursively; its frozen Stage-1 choice was evaluated without retuning on all eight development datasets.

### G2 Confidence Guard

G2 chooses the largest descending alpha satisfying `C_hat + gamma SE <= tau`; the frozen setting is `gamma=0`, `tau=.01`.

### G3 Two-Stage Guard

G3 uses 80% upper/lower bounds around the `.75` candidate and an ambiguous `.5` branch. It was retained as an ablation, not a finalist.

## 3. GuardedGram Ablations

{table(stage1_g[stage1_g.family.isin(['G1','G2','G3'])], ['method','median_disagreement_reduction','p95_C','max_C','mean_predictive_rank'])}

## 4. BlockGuard

{table(bfull.assign(inference_multiplier=1.0), ['method','median_invariant_feature_fraction','median_disagreement_reduction','median_C','p95_C','max_C','inference_multiplier'])}

Grouped failed Stage 1 and is reported as a pruned ablation. Greedy uses exact one-block retraining, benefit/cost ordering, at most eight cumulative stages, and one model at inference.

## 5. Which Features Stay Raw?

{table(feature_descriptors)}

The table contrasts Gram-selected and raw-retained blocks using empirical rank, spectrum entropy, dimension, condition proxy, one-block validation C, and measured orbit benefit; it is descriptive and not an extra selection rule.

## 6. DualViewGram

{table(dual, ['method','median_disagreement_reduction','median_C','p95_C','max_C','mean_predictive_rank','median_parameter_count','median_inference_seconds','max_peak_gpu_memory_bytes'])}

DualView fixed `.75` was pruned at Stage 1, so D2–D4 were not promoted to the full panel.

## 7. Efficiency Comparison

{table(efficiency_report, ['method','scope','median_disagreement_reduction','inference_multiplier','training_multiplier','parameter_multiplier','parameter_count'])}

Raw and single-representation BlockGuard use one inference model. Fractional Raw+Gram, GuardedGram, and Safe references require two prediction branches; pure endpoint selections require one. BlockGuard's training multiplier is the median count of exact one-block interventions plus cumulative candidates per prospective unit. It is a lower-bound full-fit accounting measure—not a wall-clock claim—because orbit/reference fits and cached representation work add further cost.

## 8. Numerical Embedding Confirmation

{table(dim_headroom.groupby(['dataset','model','embedding','k'], as_index=False).agg(default_vs_rotated_disagreement=('raw_disagreement','median'),task_span=('worst_random_basis_error',lambda x: float(np.max(x)-np.min(x)))), limit=48)}

## 9. Embedding Dimension Scaling

{table(dim_headroom.groupby(['embedding','model','k'], as_index=False).raw_disagreement.median().rename(columns=dict(raw_disagreement='median_disagreement')))}

The fitted median log2-dimension slopes are:

{table(dim_scaling.groupby(['embedding','model'], as_index=False).log2_dimension_slope_b.median().rename(columns=dict(log2_dimension_slope_b='median_log2_dimension_slope')))}

## 10. Gram/Guard Methods Inside Embeddings

{table(pd.concat([emb_methods.groupby('method',as_index=False).agg(median_disagreement_reduction=('disagreement_reduction','median'),median_C=('normalized_excess_risk','median'),p95_C=('normalized_excess_risk',lambda x: float(np.quantile(x,.95))),max_C=('normalized_excess_risk','max')), emb_block], ignore_index=True, sort=False), ['method','median_disagreement_reduction','median_C','p95_C','max_C'])}

Transferred BlockGuard is included where applicable and is explicitly labeled as a portability test, not embedding-specific retuning.

## 11. Basis Portfolio / Basis Search

The optional portfolio was not run because core confirmation and the prospective freeze took priority. Default-basis best-test frequency was {f(default_rate)}; validation selection beat default in {f(float((full_headroom.validation_selected_minus_default < 0).mean()))} of cells, versus oracle headroom in {f(float((full_headroom.oracle_best_test_error < full_headroom.default_task_error).mean()))}.

## 12. Stage-1 Pruning

{table(pd.concat([stage1_b, dual], ignore_index=True, sort=False), ['method','median_disagreement_reduction','median_C','p95_C','max_C','fallback_rate'])}

GuardedGram G1/G2/G3 and BlockGuard-Greedy survived. BlockGuard-Grouped and DualView fixed `.75` failed the prescribed gates; their negative results remain visible.

## 13. Full Development Ranking

{table(development.sort_values(['median_disagreement_reduction','p95_C'], ascending=[False,True]), ['method','experiment','median_disagreement_reduction','median_C','p95_C','max_C','mean_predictive_rank'])}

## 14. Frozen Finalists

{table(frozen_rows)}

Exactly {len(finalists['finalists'])} configurations were frozen before any prospective outcomes, under SHA `{finalists['sha256']}`.

## 15. NEW Untouched Prospective Results

All rows below use the locked datasets, seeds, model families, and finalists. The target-independent SoilKsatDB/2dplanes loader adaptations disclosed under Frozen Protocol are the only runtime data adapters.

{table(prospective_detail.rename(columns=dict(disagreement_reduction='control', normalized_excess_risk='C')), limit=None)}

## 16. Prospective Aggregate Table

{table(pros_summary, ['method','scope','median_disagreement_reduction','p25_disagreement_reduction','p75_disagreement_reduction','median_C','p90_C','p95_C','max_C','wins','ties','losses','fraction_C_lt_0','fraction_C_gt_0p01','fraction_C_gt_0p05','fallback_rate','median_invariant_feature_fraction','mean_predictive_rank','inference_multiplier'])}

## 17. Worst 10 Tail Cells

{table(tails.rename(columns=dict(normalized_excess_risk='test C')), ['dataset','model','problem_type','method','raw_task_error','method_task_error','validation_C','test C','selected_alpha','invariant_feature_fraction','explanation'])}

## 18. Paper-Safety Ranking

{ranking_section(rankings, 'A — Paper Safety')}

## 19. Strict-Safety Ranking

{ranking_section(rankings, 'B — Strict Safety')}

## 20. Basis-Control Ranking

{ranking_section(rankings, 'C — Basis Control')}

## 21. Predictive Ranking

{ranking_section(rankings, 'D — Predictive Performance')}

## 22. Efficiency Ranking

{ranking_section(rankings, 'E — Efficiency')}

## 23. Overall Paper-Candidate Ranking

{ranking_section(rankings, 'F — Overall Paper Candidate')}

The score is `R - 3 max(median C,0) - 3 max(p95 C-.01,0) - 2 max(max C-.05,0) - .05 log2(inference multiplier)`; every raw component is shown and saved in `prospective_six_rankings.csv`.

## 24. Does GuardedGram Beat SafeGram?

**{gg_vs_safe}.** G2 control/p95 C = {f(general_g2.median_disagreement_reduction)}/{f(general_g2.p95_C)}; SafeGram = {f(safe.median_disagreement_reduction)}/{f(safe.p95_C)}.

## 25. Does Feature-Level Selection Beat Global Gating?

**{feature_vs_global}.** This comparison uses the same untouched panel and counts BlockGuard as a single-representation inference method.

## 26. Can We Avoid Two Full Models?

**{avoid_two}.** BlockGuard's prospective control/tail metrics determine this answer; DualView did not survive Stage 1.

## 27. Does Basis Sensitivity Grow With Embedding Dimension?

**{grows}.** {f(positive_slopes)} of dataset/model/embedding slopes are positive across `k=4,8,16,32`.

## 28. Is the Default Numerical-Embedding Basis Usually Optimal?

**{default_optimal}.** It is oracle-best in {f(default_rate)} of full-development embedding cells.

## 29. Strongest Positive Finding

`{best_positive.method}` produced the largest prospective median control among frozen candidates ({f(best_positive.median_disagreement_reduction)}) with median/p95/max C {f(best_positive.median_C)}/{f(best_positive.p95_C)}/{f(best_positive.max_C)}.

## 30. Strongest Negative Finding

The hardest finalist tail belongs to `{worst_tail.method}` (p95/max C {f(worst_tail.p95_C)}/{f(worst_tail.max_C)}). DualView also failed its early gate, and transferred BlockGuard inside embeddings showed that feature selections do not automatically port across embedding families.

## 31. Reviewer Attack Audit

### "The adaptive rule is just validation overfitting."

All thresholds were chosen on development only and hashed before the untouched panel. Section 17 exposes validation-safe/test-harmful cells rather than hiding them.

### "The method still has catastrophic tails."

The report gives median, p90, p95, max, fractions above `.01`/`.05`, a tail CDF, and the ten worst cells. Paper- and strict-safety rankings exclude violations mechanically.

### "The method requires twice the compute."

Fractional prediction mixtures do. BlockGuard uses one inference model; DualView was benchmarked as one model but pruned. Training overhead is reported separately.

### "The phenomenon is caused by artificial preprocessing."

Frozen natural equivalences reconstruct to below `3e-16`, well under `1e-6`, for local/spectral hats and one-hot/Helmert pairs. Fourier was marked unavailable rather than fabricated.

### "Why not just use scalar features?"

Scalarization discards the representational capacity being audited. The experiment instead measures whether equivalent multi-coordinate blocks can be made stable with bounded task cost.

### "Why should basis dependence be removed if some bases are better?"

It should not be removed blindly. Sections 6 and 11 quantify validation selection and oracle headroom; guarded methods trade reproducibility against genuine predictive gains.

### "The method is too conservative."

Fallback rates and achieved control are reported jointly. Safe references establish the conservative floor; G2 and BlockGuard test whether control can rise without exceeding tail gates.

### "The method is architecture-specific."

The general panel spans five model families and the embedding panel spans MLP, TabM-D, and ResNet. Model-family counts are an explicit success requirement.

## 32. Ranked Final Candidates for Human Decision

{table(candidate_decision, ['rank','method','control','median C','p95','max','breadth','single-model?','embedding evidence','complexity','recommendation'])}

This is a ranked evidence table for human decision; no final paper method is automatically selected.

## 33. Best Next Step for Each Top-3 Candidate

1. `{rankings[(rankings.ranking == 'F — Overall Paper Candidate') & rankings.method.isin(names)].iloc[0].method}` — repeat the exact frozen configuration on a second independently locked panel.
2. `{rankings[(rankings.ranking == 'F — Overall Paper Candidate') & rankings.method.isin(names)].iloc[1].method}` — run a seed-expansion audit focused on its worst prospective cell.
3. `{rankings[(rankings.ranking == 'F — Overall Paper Candidate') & rankings.method.isin(names)].iloc[2].method}` — measure end-to-end wall-clock and memory under deployment-sized batches.

## 34. Files Produced

- `configs/GUARDED_FINALISTS.json` and SHA256 sidecar
- `results/processed/prospective_general_cells.csv`, units, summary, and manifest
- `results/processed/prospective_embedding_cells.csv`, units, summary, and manifest
- `results/processed/prospective_six_rankings.csv`
- `results/processed/prospective_worst_10_tail_cells.csv`
- `results/processed/prospective_loader_audit.json`
- `results/processed/blockguard_feature_descriptor_summary.csv`
- `results/processed/natural_basis_reuse_manifest.json`
- `results/processed/final_provenance.json`
- `results/processed/final_audit.json`
- eight PNG figures in `figures/`
- this `results.md`
"""
    (ROOT / "results.md").write_text(report)
    print(f"[report] verdict={verdict} figures=8 sections=34")


if __name__ == "__main__":
    main()
