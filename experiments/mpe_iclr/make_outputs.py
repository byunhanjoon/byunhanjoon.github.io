#!/usr/bin/env python3
"""Regenerate the ten frozen tables and Figures 1--11 from raw evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
ANALYSIS = HERE / "analysis"
FIGURES = HERE / "figures"

METHOD_LABELS = {
    "mpe": "MPE",
    "similarity_same_metric": "Similarity (normalized)",
    "similarity_unnormalized": "Similarity (raw)",
    "nystrom": "Nyström",
    "unknown_embedding": "UNK embedding",
    "support_complete_categorical": "Support-complete categorical",
    "q_ple": "Q-PLE",
    "uniform_ple": "Uniform PLE",
    "ancestor_multihot": "Ancestor multi-hot",
    "path_to_root": "Path to root",
    "wu_palmer": "Wu-Palmer",
    "lch_path": "LCH path",
    "laplacian": "Laplacian",
    "graph_laplacian": "Graph Laplacian",
    "node2vec": "node2vec",
    "raw_coordinates": "Raw coordinates",
    "raw_latlon": "Raw lat/lon",
    "coordinate_fourier": "Coordinate Fourier",
    "spatial_rbf": "Spatial RBF",
}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def pretty(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if isinstance(value, (np.bool_, bool)):
        return "yes" if bool(value) else "no"
    if isinstance(value, (np.integer, int)):
        return f"{int(value):,}"
    if isinstance(value, (np.floating, float)):
        magnitude = abs(float(value))
        if magnitude != 0 and (magnitude < 1e-4 or magnitude >= 1e5):
            return f"{float(value):.3e}"
        return f"{float(value):.4f}"
    return str(value)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No runnable cells._\n"
    columns = [str(column) for column in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    rows = []
    for values in frame.itertuples(index=False, name=None):
        escaped = [pretty(value).replace("|", "\\|").replace("\n", " ") for value in values]
        rows.append("| " + " | ".join(escaped) + " |")
    return "\n".join([header, divider, *rows]) + "\n"


def save_table(number: int, slug: str, title: str, frame: pd.DataFrame, note: str = "") -> None:
    stem = HERE / f"TABLE_{number}_{slug}"
    clean = frame.copy()
    clean.to_csv(stem.with_suffix(".csv"), index=False)
    clean.to_parquet(stem.with_suffix(".parquet"), index=False, compression="zstd")
    body = f"# Table {number} — {title}\n\n"
    if note:
        body += note.strip() + "\n\n"
    body += markdown_table(clean)
    stem.with_suffix(".md").write_text(body)


def save_figure(fig: plt.Figure, number: int, slug: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / f"FIGURE_{number}_{slug}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"FIGURE_{number}_{slug}.pdf", bbox_inches="tight")
    plt.close(fig)


def source_table(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source in ("ACS", "NYC_TLC", "CITI_BIKE", "BTS", "AMAZON_2023"):
        group = cells[cells.source_unit == source] if not cells.empty else pd.DataFrame()
        if group.empty:
            rows.append({"Source": source, "Status": "NOT RUN", "MPE": np.nan, "Best metric baseline": np.nan,
                         "Baseline name(s)": "—", "Similarity": np.nan, "PLE": np.nan, "UNK": np.nan,
                         "Mean corrupt MPE": np.nan, "Relative gain (%)": np.nan, "Winner": "—"})
            continue
        mpe = float(group.mpe.mean())
        best = float(group.best_non_mpe.mean())
        names = ", ".join(sorted(group.best_non_mpe_name.astype(str).unique()))
        rows.append(
            {
                "Source": source,
                "Status": "RUN",
                "MPE": mpe,
                "Best metric baseline": best,
                "Baseline name(s)": names,
                "Similarity": float(group.similarity.mean()),
                "PLE": float(group.ple.mean()),
                "UNK": float(group.unknown.mean()),
                "Mean corrupt MPE": float(group.mean_corrupt_mpe.mean()),
                "Relative gain (%)": 100.0 * (best - mpe) / best,
                "Winner": "MPE" if mpe < best else "baseline",
            }
        )
    return pd.DataFrame(rows)


def baseline_table() -> pd.DataFrame:
    rows = [
        ("Candidate", "MPE", "normalized Gaussian weights × learned landmark tokens", "all"),
        ("Categorical", "UNK / support-complete", "lookup or one-hot with one unseen fallback", "all"),
        ("Code", "Q-PLE / uniform PLE", "piecewise-linear features of storage codes", "all"),
        ("Same metric", "Similarity Encoding", "normalized and raw affinities to identical landmarks", "all"),
        ("Same metric", "RBF / Nyström", "classical kernel landmark coordinates", "all"),
        ("Same metric", "metric kNN", "training-only neighbor target interpolation", "all"),
        ("Causal control", "10 corrupt MPEs", "state-to-geometry permutation, capacity preserved", "primary"),
        ("Negative control", "equality MPE", "all distinct unseen nominal states collapse", "all"),
        ("Hierarchy", "ancestor / path / Wu-Palmer / LCH", "official hierarchy specialists", "ACS/Amazon"),
        ("Graph", "Laplacian / node2vec / route similarity", "target-independent topology", "graph tasks"),
        ("Geographic", "coordinates / Fourier / spatial RBF", "published latitude/longitude", "TLC/Citi/BTS"),
        ("String", "trigram / Jaro-Winkler / Levenshtein", "surface-string similarity", "secondary"),
        ("Trees", "CatBoost / LightGBM", "native categorical GBDT, plus MPE subset", "all"),
        ("Neural", "MLP / ResNet / FT-Transformer / TabM", "identical frozen trials per representation", "all"),
    ]
    return pd.DataFrame(rows, columns=["Family", "Method", "Information exposed", "Scope"])


def support_table(mechanism: pd.DataFrame, bins: pd.DataFrame) -> pd.DataFrame:
    if mechanism.empty:
        return pd.DataFrame()
    near_medium_column = (
        "spearman_near_medium_support_advantage"
        if "spearman_near_medium_support_advantage" in mechanism.columns
        else "spearman_support_advantage"
    )
    source_rho = mechanism.groupby("source_unit", as_index=False).agg(
        **{"Near–medium Spearman": (near_medium_column, "mean"),
           "Overall Spearman": ("spearman_support_advantage", "mean"), "Cells": ("task", "size")}
    )
    if bins.empty:
        return source_rho.rename(columns={"source_unit": "Source"})
    pivot = bins.groupby(["source_unit", "support_bin"])["mean_mpe_advantage"].mean().unstack()
    pivot = pivot.rename(columns={name: f"{name.title()} gain" for name in pivot.columns}).reset_index()
    return source_rho.merge(pivot, on="source_unit", how="left").rename(columns={"source_unit": "Source"})


def specialist_table(ridge: pd.DataFrame, tasks: set[str], methods: list[str]) -> pd.DataFrame:
    if ridge.empty:
        return pd.DataFrame()
    data = ridge[ridge.task.isin(tasks) & ridge.representation.isin(methods)]
    if data.empty:
        return pd.DataFrame()
    result = data.groupby(["task", "representation"], as_index=False).agg(
        **{"State-balanced MSE": ("state_balanced_standardized_mse", "mean"),
           "Row-weighted MSE": ("row_weighted_standardized_mse", "mean"), "Cells": ("split", "size")}
    )
    result["representation"] = result.representation.map(lambda value: METHOD_LABELS.get(value, value))
    return result.rename(columns={"task": "Task", "representation": "Method"})


def ablation_table(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data
    result = data.groupby(["task", "family", "representation"], as_index=False).agg(
        **{"State-balanced MSE": ("state_balanced_standardized_mse", "mean"), "Cells": ("split", "size")}
    )
    return result.rename(columns={"task": "Task", "family": "Family", "representation": "Variant"})


def theorem_table() -> pd.DataFrame:
    summary = json.loads((RAW / "theory_summary.json").read_text())
    rows = [
        ("Theorem 1", "exact chart/relabeling invariance", summary["theorem_1"]["passed"], summary["theorem_1"]["relabelings"], summary["theorem_1"]["max_representation_difference"]),
        ("Theorem 2", "partition-of-unity interpolation bound", summary["theorem_2"]["passed"], summary["theorem_2"]["smooth_cells"], summary["theorem_2"]["max_bound_violation"]),
        ("Theorem 3", "linear-head realizability", summary["theorem_3"]["passed"], summary["theorem_3"]["cases"], 0.0),
        ("Theorem 4", "equality-metric impossibility", summary["theorem_4"]["passed"], 1, summary["theorem_4"]["max_unseen_weight_difference"]),
        ("Theorem 5", "metric perturbation stability", summary["theorem_5"]["passed"], summary["theorem_5"]["corruption_cells"], summary["theorem_5"]["max_bound_ratio"]),
        ("Theorem 6", "landmark coverage / metric complexity", summary["theorem_6"]["all_finite"], summary["theorem_6"]["coverage_cells"], 0.0),
        ("Proposition 7", "interval triangular special case", summary["proposition_7"]["passed"], 1, summary["proposition_7"]["max_difference"]),
    ]
    return pd.DataFrame(rows, columns=["Result", "Statement", "Passed", "Validation cells", "Maximum violation/difference"])


def efficiency_table(ridge: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    state = read_csv(RAW / "scalability_state_results.csv")
    if not state.empty:
        chosen = state[(state.states == state.states.max()) & ((state.landmarks == 32) | state.method.isin(["full_similarity", "lookup_unknown"]))]
        for row in chosen.itertuples(index=False):
            rows.append({"Scope": f"{int(row.states):,} states", "Method": row.method, "Dimension": row.feature_dimension,
                         "Parameters": np.nan, "Representation bytes": row.representation_bytes,
                         "Precompute/fit seconds": row.precompute_seconds + row.fit_seconds, "Peak GPU bytes": 0})
    if not ridge.empty:
        for method, group in ridge[ridge.representation.isin(["mpe", "similarity_unnormalized", "nystrom", "unknown_embedding"])].groupby("representation"):
            rows.append({"Scope": "real ridge mean", "Method": method, "Dimension": float(group.feature_dimension.mean()),
                         "Parameters": float(group.feature_dimension.mean()), "Representation bytes": np.nan,
                         "Precompute/fit seconds": float(group.fit_seconds.mean()), "Peak GPU bytes": 0})
    neural_rows = []
    for path in sorted((RAW / "neural_cells").glob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") != "complete":
            continue
        if payload.get("representation", "").startswith("mpe") and payload.get("mpe_implementation_version") != 2:
            continue
        telemetry = payload.get("final_fit_telemetry", [])
        if not telemetry:
            continue
        neural_rows.append({"Method": f"{payload['backbone']} / {payload['representation']}",
                            "Dimension": payload.get("feature_dimension"), "Parameters": telemetry[0].get("parameters"),
                            "Precompute/fit seconds": np.mean([item.get("wall_seconds", np.nan) for item in telemetry]),
                            "Peak GPU bytes": max(item.get("peak_gpu_bytes", 0) for item in telemetry)})
    if neural_rows:
        frame = pd.DataFrame(neural_rows).groupby("Method", as_index=False).mean(numeric_only=True)
        for row in frame.to_dict("records"):
            rows.append({"Scope": "neural mean", "Representation bytes": np.nan, **row})
    return pd.DataFrame(rows)


def concept_figure() -> None:
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axis("off")
    boxes = [
        (0.02, "typed state\n$x\\in(X,d)$"),
        (0.22, "distances\n$d(x,\\ell_j)$"),
        (0.42, "partition weights\n$w_j\\geq0,\\;\\sum w_j=1$"),
        (0.64, "learned token\n$\\sum_j w_jv_j$"),
        (0.84, "tabular\nbackbone"),
    ]
    for x, label in boxes:
        ax.add_patch(plt.Rectangle((x, 0.48), 0.14, 0.23, facecolor="#e8f1fb", edgecolor="#285f8f", lw=1.5))
        ax.text(x + 0.07, 0.595, label, ha="center", va="center", fontsize=11)
    for left, _ in boxes[:-1]:
        ax.annotate("", xy=(left + 0.195, 0.595), xytext=(left + 0.14, 0.595), arrowprops={"arrowstyle": "->", "lw": 1.5})
    # Tiny metric-family glyphs.
    xs = np.linspace(0.07, 0.93, 4)
    labels = ["line", "cycle", "tree", "graph"]
    for index, (x, label) in enumerate(zip(xs, labels)):
        y = 0.18
        if label == "line":
            ax.plot([x - .055, x + .055], [y, y], "o-", color="#a04442", ms=5)
        elif label == "cycle":
            angle = np.linspace(0, 2 * np.pi, 9)
            ax.plot(x + .05 * np.cos(angle), y + .08 * np.sin(angle), "o-", color="#a04442", ms=3)
        elif label == "tree":
            edges = [((x, y+.08),(x-.04,y)),((x,y+.08),(x+.04,y)),((x-.04,y),(x-.065,y-.06)),((x-.04,y),(x-.015,y-.06))]
            for (x1,y1),(x2,y2) in edges: ax.plot([x1,x2],[y1,y2],"-o",color="#a04442",ms=3)
        else:
            points = np.array([[-.05,.02],[0,.08],[.05,.02],[-.025,-.06],[.04,-.055]]) + [x,y]
            for i,j in [(0,1),(1,2),(0,3),(2,4),(3,4),(1,4)]: ax.plot(points[[i,j],0],points[[i,j],1],color="#a04442")
            ax.scatter(points[:,0],points[:,1],s=18,color="#a04442")
        ax.text(x, 0.035, label, ha="center", fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.9)
    save_figure(fig, 1, "MPE_CONCEPT")


def benchmark_figure(cells: pd.DataFrame) -> None:
    table = source_table(cells)
    run = table[table.Status == "RUN"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    methods = ["MPE", "Similarity", "PLE", "UNK", "Best metric baseline"]
    x = np.arange(len(run)); width = 0.15
    for index, method in enumerate(methods):
        values = run[method].to_numpy(float)
        ax.bar(x + (index - 2) * width, values, width, label=method)
    ax.set_xticks(x, run.Source, rotation=20); ax.set_ylabel("State-balanced standardized MSE ↓")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    ax.set_title("Real unseen-state benchmark")
    save_figure(fig, 2, "REAL_BENCHMARK")


def support_figure(bins: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    order = ["near", "medium", "far"]
    for source, group in bins.groupby("source_unit"):
        values = group.groupby("support_bin")["mean_mpe_advantage"].mean().reindex(order)
        ax.plot(order, values, marker="o", label=source)
    ax.axhline(0, color="black", lw=.8); ax.set_ylabel("Baseline loss − MPE loss")
    ax.set_title("MPE advantage versus distance to training support"); ax.legend(fontsize=8)
    save_figure(fig, 3, "SUPPORT_DISTANCE")


def corruption_figure(data: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    if not data.empty:
        x = np.arange(len(data)); width = .35
        ax.bar(x-width/2, data.correct, width, label="correct metric")
        ax.bar(x+width/2, data.corrupt, width, label="mean of 10 corrupt metrics")
        ax.set_xticks(x, data.source_unit, rotation=20)
    ax.set_ylabel("State-balanced standardized MSE ↓"); ax.set_title("Correct versus corrupted metric")
    ax.legend(); save_figure(fig, 4, "CORRECT_VS_CORRUPT")


def specialist_figure(data: pd.DataFrame, number: int, slug: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2))
    if not data.empty:
        pivot = data.pivot(index="Task", columns="Method", values="State-balanced MSE")
        pivot.plot(kind="bar", ax=ax, width=.85)
    ax.set_ylabel("State-balanced standardized MSE ↓"); ax.set_title(title)
    ax.legend(fontsize=7, ncol=3); save_figure(fig, number, slug)


def similarity_figure(cells: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    if not cells.empty:
        for source, group in cells.groupby("source_unit"):
            ax.scatter(group.similarity, group.mpe, s=20, alpha=.7, label=source)
        values = np.r_[cells.similarity.to_numpy(float), cells.mpe.to_numpy(float)]
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("Same-metric Similarity Encoding MSE"); ax.set_ylabel("MPE MSE")
    ax.set_title("Identical metric and landmarks"); ax.legend(fontsize=7)
    save_figure(fig, 7, "SIMILARITY_SHOWDOWN")


def theory_figure() -> None:
    data = pd.read_parquet(RAW / "theory_interpolation.parquet")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sample = data.iloc[:: max(1, len(data)//4000)].copy()
    radius = next(column for column in (
        "weighted_metric_radius", "weighted_radius", "mean_weighted_radius_test"
    ) if column in sample.columns)
    error = next(column for column in (
        "absolute_error", "error", "mean_absolute_error_test"
    ) if column in sample.columns)
    ax.scatter(sample[radius], sample[error], s=8, alpha=.25)
    if "bound" in sample:
        ordered = sample.sort_values(radius)
        ax.plot(ordered[radius], ordered.bound, color="#a04442", alpha=.6, label="proved bound")
        ax.legend()
    elif "empirical_lipschitz" in sample:
        bound = sample["empirical_lipschitz"] * sample[radius]
        ax.scatter(sample[radius], bound, s=8, alpha=.20, color="#a04442", label="$L R_w$ bound term")
        ax.legend()
    ax.set_xlabel("Weighted metric radius"); ax.set_ylabel("Absolute interpolation error")
    ax.set_title("Synthetic interpolation error and support radius")
    save_figure(fig, 8, "THEORETICAL_BOUND")


def noise_figure(ablation: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    partial = ablation[ablation.family == "partial_corruption"].copy() if not ablation.empty else pd.DataFrame()
    if not partial.empty:
        for task, group in partial.groupby("task"):
            curve = group.groupby("fraction")["state_balanced_standardized_mse"].mean().sort_index()
            ax.plot(100 * curve.index.to_numpy(float), curve, marker="o", label=task)
    ax.set_xlabel("State-to-metric association corrupted (%)"); ax.set_ylabel("State-balanced MSE ↓")
    ax.set_title("Partial metric corruption"); ax.legend(fontsize=7)
    save_figure(fig, 9, "METRIC_NOISE")


def nominal_figure(nominal: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    methods = ["mpe_equality", "lookup_unknown", "support_complete_onehot", "uniform_ple"]
    data = nominal[nominal.representation.isin(methods)] if not nominal.empty else nominal
    if not data.empty:
        pivot = data.groupby(["task", "representation"])["state_balanced_standardized_mse"].mean().unstack()
        pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("State-balanced standardized MSE ↓"); ax.set_title("Equality metric no-free-lunch")
    ax.legend(fontsize=8); save_figure(fig, 10, "EQUALITY_CONTROL")


def landmark_figure(ablation: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    budget = ablation[ablation.family == "landmark_budget"].copy() if not ablation.empty else pd.DataFrame()
    if not budget.empty:
        for task, group in budget.groupby("task"):
            curve = group.groupby("m")["state_balanced_standardized_mse"].mean().sort_index()
            ax.plot(curve.index, curve, marker="o", label=task)
    ax.set_xscale("log", base=2); ax.set_xlabel("Landmarks m"); ax.set_ylabel("State-balanced MSE ↓")
    ax.set_title("Landmark budget"); ax.legend(fontsize=7)
    save_figure(fig, 11, "LANDMARK_BUDGET")


def main() -> None:
    cells = read_csv(ANALYSIS / "cell_comparisons.csv")
    ridge = read_csv(RAW / "ridge_results.csv")
    mechanism = read_csv(ANALYSIS / "support_mechanism.csv")
    bins = read_csv(ANALYSIS / "support_bins.csv")
    corruption = read_csv(ANALYSIS / "corruption_source_summary.csv")
    ablation = read_csv(RAW / "ablation_results.csv")
    nominal = read_csv(RAW / "nominal_results.csv")

    panel = read_csv(ANALYSIS / "dataset_panel.csv").rename(
        columns={"source": "Source", "task": "Task", "status": "Status", "rows": "Rows",
                 "states": "States", "metric": "Metric", "train_states": "Train states",
                 "validation_states": "Val states", "test_states": "Test states",
                 "median_support_gap": "Median support gap", "reason": "Reason"}
    )
    save_table(1, "DATASETS", "Final dataset panel", panel, "State counts use frozen split 0; unavailable sources remain visible.")
    save_table(2, "BASELINES", "Mandatory comparison set", baseline_table())
    main_real = source_table(cells)
    save_table(3, "MAIN_REAL_RESULTS", "Main real unseen-state result", main_real,
               "Primary loss is state-balanced standardized MSE; values average tasks, settings, splits, seeds, and available frozen backbones equally.")
    support = support_table(mechanism, bins)
    save_table(4, "SUPPORT_DISTANCE", "Support-distance mechanism", support)
    save_table(5, "CORRUPT_METRIC", "Correct versus corrupted metric", corruption.rename(
        columns={"source_unit": "Source", "correct": "Correct MPE", "corrupt": "Mean corrupt MPE", "wins": "Wins", "cells": "Cells"}))
    hierarchy = specialist_table(ridge, {"acs_occupation", "acs_industry", "amazon_leaf_category"},
        ["mpe", "ancestor_multihot", "path_to_root", "hierarchy_shortest_path_similarity", "wu_palmer", "lch_path", "laplacian", "node2vec", "nystrom"])
    save_table(6, "HIERARCHY_BASELINES", "Hierarchy showdown", hierarchy, "Amazon remains NOT RUN in Table 1; this table contains every runnable hierarchy task.")
    geography = specialist_table(ridge, {"tlc_pickup_zone", "tlc_dropoff_zone", "citibike_start_station", "airline_origin_airport", "airline_destination_airport"},
        ["mpe", "raw_coordinates", "raw_latlon", "coordinate_fourier", "spatial_rbf", "graph_laplacian", "node2vec", "nystrom"])
    save_table(7, "GEOGRAPHIC_BASELINES", "Geographic and network showdown", geography)
    save_table(8, "ABLATIONS", "MPE ablations", ablation_table(ablation))
    save_table(9, "THEOREM_VALIDATION", "Theorem validation", theorem_table())
    save_table(10, "EFFICIENCY", "Efficiency and scalability", efficiency_table(ridge))

    concept_figure()
    benchmark_figure(cells)
    support_figure(bins)
    corruption_figure(corruption)
    specialist_figure(hierarchy, 5, "HIERARCHY_SHOWDOWN", "Hierarchy-specific baselines")
    specialist_figure(geography, 6, "GEOGRAPHIC_SHOWDOWN", "Geographic and network baselines")
    similarity_figure(cells)
    theory_figure()
    noise_figure(ablation)
    nominal_figure(nominal)
    landmark_figure(ablation)
    print("regenerated Tables 1--10 and Figures 1--11", flush=True)


if __name__ == "__main__":
    main()
