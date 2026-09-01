#!/usr/bin/env python3
"""Regenerate statistical summaries and frozen success gates from raw cells."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ridge_benchmark import DEFAULT_TASKS, consolidate as consolidate_ridge


HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
ANALYSIS = HERE / "analysis"
PRIMARY_SOURCES = ["ACS", "NYC_TLC", "CITI_BIKE", "BTS", "AMAZON_2023"]
METRIC_BASELINES = {
    "similarity_same_metric", "similarity_unnormalized", "rbf_normalized",
    "rbf_unnormalized", "nystrom", "knn_metric", "hierarchy_shortest_path_similarity",
    "tree_rbf", "ancestor_multihot", "path_to_root", "wu_palmer", "lch_path",
    "laplacian", "node2vec", "raw_coordinates", "raw_latlon", "coordinate_fourier",
    "spatial_rbf", "graph_laplacian", "character_3gram_hash",
}


def json_dump(value: Any, path: Path) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def validation_score(result: dict[str, Any]) -> float:
    trials = result.get("validation_trials", [])
    values = []
    for row in trials:
        for key in ("score", "validation_score", "state_balanced_brier"):
            if key in row:
                values.append(float(row[key]))
                break
    return min(values) if values else float("nan")


def ridge_long() -> pd.DataFrame:
    rows = []
    for path in sorted((RAW / "ridge_cells").glob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") != "complete":
            continue
        for result in payload["results"]:
            rows.append(
                {
                    "task": payload["task"], "source_unit": payload["source_unit"],
                    "split": payload["split"], "setting": payload["setting"],
                    "backbone": "ridge", "representation": result["representation"],
                    "validation_score": validation_score(result),
                    **{
                        key: result.get(key)
                        for key in (
                            "state_balanced_standardized_mse", "row_weighted_standardized_mse",
                            "rmse", "mae", "worst_quartile_state_mse", "worst_decile_state_mse",
                            "feature_dimension", "fit_seconds", "alias_of",
                        )
                    },
                }
            )
    return pd.DataFrame(rows)


def choose(group: pd.DataFrame, names: Iterable[str]) -> pd.Series | None:
    subset = group[group["representation"].isin(set(names))].copy()
    if subset.empty:
        return None
    subset["selection"] = subset["validation_score"].fillna(np.inf)
    subset = subset.sort_values(["selection", "representation"], kind="stable")
    return subset.iloc[0]


def cell_comparisons(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["task", "source_unit", "split", "setting", "backbone"]
    for values, group in long.groupby(keys, sort=True):
        mpe = choose(group, ["mpe"])
        best = choose(group, METRIC_BASELINES)
        similarity = choose(group, ["similarity_same_metric", "similarity_unnormalized"])
        ple = choose(group, ["q_ple", "uniform_ple"])
        unknown = choose(group, ["unknown_embedding", "support_complete_categorical"])
        nystrom = choose(group, ["nystrom"])
        corrupt = group[group["representation"].str.startswith("mpe_corrupt_")]
        if mpe is None or best is None:
            continue
        metric = "state_balanced_standardized_mse"
        corrupt_value = float(corrupt[metric].mean()) if not corrupt.empty else float("nan")
        rows.append(
            {
                **dict(zip(keys, values)),
                "mpe": float(mpe[metric]),
                "mpe_row_weighted": float(mpe["row_weighted_standardized_mse"]),
                "best_non_mpe": float(best[metric]),
                "best_non_mpe_row_weighted": float(best["row_weighted_standardized_mse"]),
                "best_non_mpe_name": str(best["representation"]),
                "similarity": float(similarity[metric]) if similarity is not None else float("nan"),
                "similarity_name": str(similarity["representation"]) if similarity is not None else None,
                "nystrom": float(nystrom[metric]) if nystrom is not None else float("nan"),
                "ple": float(ple[metric]) if ple is not None else float("nan"),
                "ple_name": str(ple["representation"]) if ple is not None else None,
                "unknown": float(unknown[metric]) if unknown is not None else float("nan"),
                "mean_corrupt_mpe": corrupt_value,
                "relative_gain_percent": 100.0 * (float(best[metric]) - float(mpe[metric])) / float(best[metric]),
                "mpe_wins": bool(float(mpe[metric]) < float(best[metric]) - 1e-12),
                "correct_beats_mean_corrupt": bool(float(mpe[metric]) < corrupt_value) if np.isfinite(corrupt_value) else None,
            }
        )
    return pd.DataFrame(rows)


def source_summary(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source in PRIMARY_SOURCES:
        group = cells[cells["source_unit"] == source]
        if group.empty:
            rows.append(
                {
                    "source_unit": source, "status": "NOT RUN",
                    "mpe": np.nan, "best_non_mpe": np.nan, "relative_gain_percent": np.nan,
                    "cell_wins": 0, "cells": 0,
                }
            )
            continue
        mpe = float(group["mpe"].mean())
        baseline = float(group["best_non_mpe"].mean())
        rows.append(
            {
                "source_unit": source, "status": "RUN", "mpe": mpe,
                "best_non_mpe": baseline,
                "relative_gain_percent": 100.0 * (baseline - mpe) / baseline,
                "cell_wins": int(group["mpe_wins"].sum()), "cells": len(group),
                "state_balanced_difference": baseline - mpe,
                "mpe_row_weighted": float(group["mpe_row_weighted"].mean()),
                "best_non_mpe_row_weighted": float(group["best_non_mpe_row_weighted"].mean()),
            }
        )
    return pd.DataFrame(rows)


def source_bootstrap(source: pd.DataFrame, replicates: int = 10000) -> dict[str, Any]:
    run = source[source["status"] == "RUN"].copy()
    effects = run["state_balanced_difference"].to_numpy(np.float64)
    relative = run["relative_gain_percent"].to_numpy(np.float64)
    if not len(effects):
        return {"sources": 0, "replicates": replicates}
    rng = np.random.default_rng(20261201)
    indices = rng.integers(0, len(effects), size=(replicates, len(effects)))
    boot = effects[indices].mean(axis=1)
    boot_relative = relative[indices].mean(axis=1)
    return {
        "sources": len(effects), "replicates": replicates,
        "source_balanced_difference": float(effects.mean()),
        "source_balanced_relative_gain_percent": float(relative.mean()),
        "median_relative_gain_percent": float(np.median(relative)),
        "difference_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "relative_gain_ci95": np.quantile(boot_relative, [0.025, 0.975]).tolist(),
        "low_discrete_resolution": True,
    }


def corruption_summary(cells: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    usable = cells[
        np.isfinite(cells["mean_corrupt_mpe"]) & cells["source_unit"].isin(PRIMARY_SOURCES)
    ].copy()
    task_split = usable.groupby(["source_unit", "task", "split"], as_index=False).agg(
        correct=("mpe", "mean"), corrupt=("mean_corrupt_mpe", "mean")
    )
    task_split["correct_wins"] = task_split["correct"] < task_split["corrupt"]
    source = task_split.groupby("source_unit", as_index=False).agg(
        correct=("correct", "mean"), corrupt=("corrupt", "mean"),
        wins=("correct_wins", "sum"), cells=("correct_wins", "size"),
    )
    fraction = float(task_split["correct_wins"].mean()) if len(task_split) else float("nan")
    gate = {
        "task_split_win_fraction": fraction,
        "wins": int(task_split["correct_wins"].sum()), "cells": len(task_split),
        "required_fraction": 0.8, "passes_fraction": bool(fraction >= 0.8) if np.isfinite(fraction) else False,
        "source_wins": int((source["correct"] < source["corrupt"]).sum()),
        "sources": len(source),
    }
    gate["required_source_wins"] = max(1, len(source) - 1)
    gate["passes"] = bool(
        gate["passes_fraction"] and gate["source_wins"] >= gate["required_source_wins"]
    )
    return source, gate


def support_analysis(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    path = RAW / "ridge_state_results.parquet"
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame(), {"available": False}
    state = pd.read_parquet(path)
    rows, bins = [], []
    for cell in cells.itertuples(index=False):
        subset = state[
            (state.task == cell.task) & (state.split == cell.split) &
            (state.setting == cell.setting) &
            (state.representation.isin(["mpe", cell.best_non_mpe_name]))
        ]
        pivot = subset.pivot_table(
            index=["state_id", "support_distance", "weighted_landmark_radius"],
            columns="representation", values="standardized_mse", aggfunc="mean",
        ).reset_index()
        if "mpe" not in pivot or cell.best_non_mpe_name not in pivot:
            continue
        pivot["advantage"] = pivot[cell.best_non_mpe_name] - pivot["mpe"]
        ordered = pivot.sort_values(["support_distance", "state_id"], kind="stable").copy()
        ordered["support_bin"] = pd.qcut(
            np.arange(len(ordered)), q=3, labels=["near", "medium", "far"], duplicates="drop"
        )
        rho_overall = (
            float(spearmanr(ordered["support_distance"], ordered["advantage"]).statistic)
            if len(ordered) >= 3 and ordered["advantage"].nunique() > 1
            else float("nan")
        )
        near_medium = ordered[ordered["support_bin"].astype(str).isin(["near", "medium"])]
        rho_near_medium = (
            float(spearmanr(near_medium["support_distance"], near_medium["advantage"]).statistic)
            if len(near_medium) >= 3 and near_medium["advantage"].nunique() > 1
            else float("nan")
        )
        rows.append(
            {
                "task": cell.task, "source_unit": cell.source_unit, "split": cell.split,
                "setting": cell.setting, "baseline": cell.best_non_mpe_name,
                "states": len(ordered), "spearman_support_advantage": rho_overall,
                "spearman_near_medium_support_advantage": rho_near_medium,
            }
        )
        for support_bin, group in ordered.groupby("support_bin", observed=True):
            bins.append(
                {
                    "task": cell.task, "source_unit": cell.source_unit, "split": cell.split,
                    "setting": cell.setting, "support_bin": str(support_bin),
                    "states": len(group), "mean_support_distance": float(group["support_distance"].mean()),
                    "mean_mpe_advantage": float(group["advantage"].mean()),
                }
            )
    mechanism = pd.DataFrame(rows)
    primary = mechanism[mechanism["source_unit"].isin(PRIMARY_SOURCES)] if not mechanism.empty else mechanism
    source = primary.groupby("source_unit", as_index=False)["spearman_near_medium_support_advantage"].mean() if not primary.empty else pd.DataFrame()
    positive = int((source["spearman_near_medium_support_advantage"] > 0).sum()) if not source.empty else 0
    gate = {
        "available": not mechanism.empty, "positive_sources": positive,
        "sources": len(source), "required_positive_sources": 3,
        "passes": bool(positive >= 3),
        "source_values": source.to_dict("records") if not source.empty else [],
    }
    return mechanism, pd.DataFrame(bins), gate


def payload_long(folder: Path, metric: str = "state_balanced_standardized_mse") -> pd.DataFrame:
    rows = []
    for path in sorted(folder.glob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") != "complete" or "results" not in payload:
            continue
        for result in payload["results"]:
            if metric not in result:
                continue
            rows.append(
                {
                    "task": payload["task"], "source_unit": payload["source_unit"],
                    "split": payload.get("split", "natural"), "setting": payload["setting"],
                    "representation": result["representation"], "validation_score": validation_score(result),
                    metric: result[metric],
                }
            )
    return pd.DataFrame(rows)


def seen_summary(unseen_sources: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    long = payload_long(RAW / "seen_cells")
    if long.empty:
        return pd.DataFrame(), {"available": False}
    seen_cells = cell_comparisons(long.assign(
        backbone="ridge", row_weighted_standardized_mse=long["state_balanced_standardized_mse"],
        rmse=np.nan, mae=np.nan, worst_quartile_state_mse=np.nan, worst_decile_state_mse=np.nan,
        feature_dimension=np.nan, fit_seconds=np.nan, alias_of=None,
    ))
    source = source_summary(seen_cells)
    merged = unseen_sources.merge(
        source[["source_unit", "relative_gain_percent"]].rename(columns={"relative_gain_percent": "seen_relative_gain_percent"}),
        on="source_unit", how="left",
    )
    merged["unseen_minus_seen_gain_points"] = merged["relative_gain_percent"] - merged["seen_relative_gain_percent"]
    run = merged[(merged.status == "RUN") & merged.seen_relative_gain_percent.notna()]
    unseen_mean = float(run["relative_gain_percent"].mean()) if len(run) else float("nan")
    seen_mean = float(run["seen_relative_gain_percent"].mean()) if len(run) else float("nan")
    ratio = unseen_mean / seen_mean if np.isfinite(seen_mean) and abs(seen_mean) > 1e-12 else float("nan")
    gate = {
        "available": len(run) > 0, "unseen_relative_gain_percent": unseen_mean,
        "seen_relative_gain_percent": seen_mean, "unseen_minus_seen_gain_points": unseen_mean - seen_mean,
        "gain_ratio": ratio,
        "passes_materiality": bool((unseen_mean - seen_mean) >= 2.0 or ratio > 1.5) if len(run) else False,
        "majority_direction": bool((run["unseen_minus_seen_gain_points"] > 0).mean() > 0.5) if len(run) else False,
    }
    gate["passes"] = gate["passes_materiality"] and gate["majority_direction"]
    return merged, gate


def nominal_gate() -> tuple[pd.DataFrame, dict[str, Any]]:
    path = RAW / "nominal_results.parquet"
    if not path.exists():
        return pd.DataFrame(), {"available": False}
    data = pd.read_parquet(path)
    rows = []
    for (task, source), group in data.groupby(["task", "source_unit"]):
        mpe = float(group[group.representation == "mpe_equality"]["state_balanced_standardized_mse"].mean())
        controls = group[group.representation.isin(["lookup_unknown", "support_complete_onehot", "uniform_ple"])]
        best = float(controls.groupby("representation")["state_balanced_standardized_mse"].mean().min())
        gain = 100.0 * (best - mpe) / best
        rows.append({"task": task, "source_unit": source, "mpe_equality": mpe, "best_control": best, "relative_gain_percent": gain, "favors_mpe_over_2pct": gain > 2.0})
    frame = pd.DataFrame(rows)
    count = int(frame["favors_mpe_over_2pct"].sum())
    gate = {
        "available": True, "fields_favoring_mpe_over_2pct": count, "fields": len(frame),
        "maximum_allowed": 1, "passes": bool(count <= 1),
    }
    return frame, gate


def classical_gate(long: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    methods = ["similarity_same_metric", "similarity_unnormalized", "nystrom"]
    rows = []
    eligible = long[
        long.representation.isin(methods + ["mpe"]) & long.source_unit.isin(PRIMARY_SOURCES)
    ]
    for (source, method), group in eligible.groupby(["source_unit", "representation"]):
        rows.append({"source_unit": source, "representation": method, "loss": float(group.state_balanced_standardized_mse.mean())})
    pivot = pd.DataFrame(rows).pivot(index="source_unit", columns="representation", values="loss") if rows else pd.DataFrame()
    output = []
    for source, row in pivot.iterrows():
        if "mpe" not in row or not np.isfinite(row["mpe"]):
            continue
        for method in methods:
            if method in row and np.isfinite(row[method]):
                gain = 100.0 * (row[method] - row["mpe"]) / row[method]
                output.append({"source_unit": source, "baseline": method, "relative_gain_percent": gain, "over_2pct": gain > 2.0})
    frame = pd.DataFrame(output)
    source_pass = frame.groupby("source_unit")["over_2pct"].any() if not frame.empty else pd.Series(dtype=bool)
    count = int(source_pass.sum())
    return frame, {"sources_over_2pct_vs_at_least_one": count, "required": 3, "passes": bool(count >= 3)}


def smoothness_analysis(cells: pd.DataFrame) -> dict[str, Any]:
    path = RAW / "smoothness_results.parquet"
    if not path.exists():
        return {"available": False}
    smooth = pd.read_parquet(path)
    benefit = cells.groupby(["source_unit", "task", "split"], as_index=False)["relative_gain_percent"].mean()
    merged = smooth.merge(benefit, on=["source_unit", "task", "split"], how="inner")
    rho = float(spearmanr(merged["prespecified_smoothness"], merged["relative_gain_percent"]).statistic) if len(merged) >= 3 else float("nan")
    source = merged.groupby("source_unit", as_index=False).agg(
        smoothness=("prespecified_smoothness", "mean"), benefit=("relative_gain_percent", "mean")
    )
    predictions = []
    if len(source) >= 3:
        for index in range(len(source)):
            train = source.drop(index=index)
            coefficients = np.polyfit(train["smoothness"], train["benefit"], deg=1)
            predictions.append(float(np.polyval(coefficients, source.iloc[index]["smoothness"])))
        source["looso_prediction"] = predictions
        mae = float(np.mean(np.abs(source["looso_prediction"] - source["benefit"])))
        pred_rho = float(spearmanr(source["looso_prediction"], source["benefit"]).statistic)
    else:
        mae, pred_rho = float("nan"), float("nan")
    source.to_csv(ANALYSIS / "smoothness_source_prediction.csv", index=False)
    return {
        "available": True, "task_split_spearman": rho, "task_splits": len(merged),
        "sources": len(source), "looso_mae_gain_points": mae, "looso_spearman": pred_rho,
        "useful": bool(np.isfinite(pred_rho) and pred_rho > 0.5),
    }


def neural_long() -> pd.DataFrame:
    rows = []
    for path in sorted((RAW / "neural_cells").glob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") != "complete":
            continue
        uses = bool(payload.get("uses_learned_landmark_tokens", False))
        if payload.get("representation", "").startswith("mpe") and payload.get("mpe_implementation_version") != 2:
            continue
        if payload.get("representation") == "unknown_embedding" and payload.get("categorical_implementation_version") != 2:
            continue
        validation = min(float(row["validation_score"]) for row in payload["hpo_trials"])
        for result in payload["results"]:
            rows.append(
                {
                    "task": payload["task"], "source_unit": payload["source_unit"],
                    "split": payload["split"], "setting": payload["setting"],
                    "backbone": payload["backbone"], "representation": payload["representation"],
                    "seed": result["seed"], "validation_score": validation,
                    "uses_learned_landmark_tokens": uses,
                    "state_balanced_standardized_mse": result["state_balanced_standardized_mse"],
                    "row_weighted_standardized_mse": result["row_weighted_standardized_mse"],
                }
            )
    return pd.DataFrame(rows)


def aggregate_neural_seeds(long: pd.DataFrame) -> pd.DataFrame:
    """Return one row per frozen neural representation cell.

    Seeds are repeated fits, not independent representation choices.  Validation
    selection is shared within the cell, while test metrics are averaged over
    the three frozen seeds before source/task aggregation.
    """
    if long.empty:
        return long
    keys = ["task", "source_unit", "split", "setting", "backbone", "representation"]
    return long.groupby(keys, as_index=False).agg(
        validation_score=("validation_score", "first"),
        state_balanced_standardized_mse=("state_balanced_standardized_mse", "mean"),
        row_weighted_standardized_mse=("row_weighted_standardized_mse", "mean"),
        neural_seeds=("seed", "nunique"),
    ).assign(
        rmse=np.nan, mae=np.nan, worst_quartile_state_mse=np.nan,
        worst_decile_state_mse=np.nan, feature_dimension=np.nan,
        fit_seconds=np.nan, alias_of=None,
    )


def dataset_panel() -> pd.DataFrame:
    rows = []
    for folder in sorted((HERE / "processed").iterdir()):
        manifest_path = folder / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        row = {
            "source": manifest.get("source_unit"), "task": manifest.get("task", folder.name),
            "status": manifest.get("status"), "rows": manifest.get("rows"),
            "states": manifest.get("states"), "metric": manifest.get("metric"),
            "reason": manifest.get("reason"),
        }
        if manifest.get("status") == "RUN":
            split = json.loads((folder / "splits.json").read_text())["0"]
            row.update({f"{part}_states": len(split[part]) for part in ("train", "validation", "test")})
            task = load_task_local(folder)
            lookup = {state: index for index, state in enumerate(task["states"])}
            train = np.asarray([lookup[state] for state in split["train"]])
            test = np.asarray([lookup[state] for state in split["test"]])
            row["median_support_gap"] = float(np.median(np.min(task["distance"][np.ix_(test, train)], axis=1)))
        rows.append(row)
    return pd.DataFrame(rows)


def load_task_local(folder: Path) -> dict[str, Any]:
    return {
        "states": pd.read_parquet(folder / "states.parquet")["state_id"].astype(str).tolist(),
        "distance": np.asarray(np.load(folder / "distance_primary.npy"), dtype=np.float64),
    }


def completeness() -> dict[str, Any]:
    runnable = []
    for task in DEFAULT_TASKS:
        manifest = json.loads((HERE / "processed" / task / "manifest.json").read_text())
        if manifest["status"] == "RUN":
            runnable.append(task)
    ridge_complete = len(list((RAW / "ridge_cells").glob("*.json")))
    neural = neural_long()
    tree_complete = sum(
        json.loads(path.read_text()).get("status") == "complete"
        for path in (RAW / "tree_cells").glob("*.json")
    )
    return {
        "runnable_tasks": len(runnable), "runnable_task_names": runnable,
        "ridge_expected_cells": len(runnable) * 5 * 2, "ridge_complete_cells": ridge_complete,
        "neural_valid_cells": int(neural.groupby(["task", "split", "setting", "backbone", "representation"]).ngroups) if not neural.empty else 0,
        "neural_test_seed_rows": len(neural), "tree_complete_cells": tree_complete,
    }


def main() -> None:
    ANALYSIS.mkdir(exist_ok=True)
    consolidate_ridge(RAW / "ridge_cells")
    ridge = ridge_long()
    ridge.to_parquet(ANALYSIS / "ridge_long.parquet", index=False, compression="zstd")
    neural_seed_rows = neural_long()
    neural = aggregate_neural_seeds(neural_seed_rows)
    long = pd.concat([ridge, neural], ignore_index=True, sort=False) if not neural.empty else ridge.copy()
    long.to_parquet(ANALYSIS / "all_long.parquet", index=False, compression="zstd")
    cells = cell_comparisons(long)
    cells.to_parquet(ANALYSIS / "cell_comparisons.parquet", index=False, compression="zstd")
    cells.to_csv(ANALYSIS / "cell_comparisons.csv", index=False)
    sources = source_summary(cells)
    sources.to_csv(ANALYSIS / "source_comparisons.csv", index=False)
    bootstrap = source_bootstrap(sources)
    corrupt_sources, gate_b = corruption_summary(cells)
    corrupt_sources.to_csv(ANALYSIS / "corruption_source_summary.csv", index=False)
    # The per-state support records currently come from the exhaustive ridge
    # mechanism view, so use the matching ridge comparisons here rather than
    # pairing neural aggregate losses with ridge state rows.
    ridge_cells = cell_comparisons(ridge)
    mechanism, bins, gate_d = support_analysis(ridge_cells)
    mechanism.to_csv(ANALYSIS / "support_mechanism.csv", index=False)
    bins.to_csv(ANALYSIS / "support_bins.csv", index=False)
    ridge_sources = source_summary(ridge_cells)
    seen, gate_c = seen_summary(ridge_sources)
    seen.to_csv(ANALYSIS / "seen_vs_unseen.csv", index=False)
    nominal, gate_e = nominal_gate()
    nominal.to_csv(ANALYSIS / "nominal_gate.csv", index=False)
    classical, gate_f = classical_gate(long)
    classical.to_csv(ANALYSIS / "classical_same_metric_gate.csv", index=False)
    smoothness = smoothness_analysis(cells)
    panel = dataset_panel()
    panel.to_csv(ANALYSIS / "dataset_panel.csv", index=False)
    if not neural_seed_rows.empty:
        neural_seed_rows.to_parquet(ANALYSIS / "neural_long.parquet", index=False, compression="zstd")
    run_sources = sources[sources.status == "RUN"]
    source_wins = int((run_sources.mpe < run_sources.best_non_mpe).sum())
    gate_a = {
        "available_sources": len(run_sources), "required_sources": 5,
        "source_wins": source_wins, "required_wins": 4,
        "bootstrap_ci_excludes_zero_positive": bool(
            len(run_sources) == 5 and bootstrap.get("difference_ci95", [0])[0] > 0
        ),
    }
    gate_a["passes"] = bool(
        gate_a["available_sources"] == 5 and source_wins >= 4 and gate_a["bootstrap_ci_excludes_zero_positive"]
    )
    gates = {"A": gate_a, "B": gate_b, "C": gate_c, "D": gate_d, "E": gate_e, "F": gate_f}
    gates["verdict"] = (
        "SUPPORTED" if all(gates[key].get("passes", False) for key in ("A", "B", "E", "F"))
        else "NOT SUPPORTED"
    )
    json_dump(gates, ANALYSIS / "gate_summary.json")
    json_dump(bootstrap, ANALYSIS / "source_bootstrap.json")
    json_dump(smoothness, ANALYSIS / "smoothness_summary.json")
    json_dump(completeness(), ANALYSIS / "completeness.json")
    print(json.dumps({"gates": gates, "bootstrap": bootstrap, "completeness": completeness()}, indent=2, default=str))


if __name__ == "__main__":
    main()
