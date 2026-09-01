#!/usr/bin/env python3
"""Calibrate, score, and summarize all frozen evaluation baselines.

The dataset, rather than a functional query or episode, is the unit of
inference.  The script keeps diagonal ablations in the raw results but omits
them from the main-model rank table.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest, trim_mean

from common import CACHE, CONFIG, atomic_json, gaussian_scores


CLASSICAL_METHODS = ["bayesian_linear", "gp_rbf", "gp_matern32", "catboost_process"]
MAIN_METHODS = [
    "projtabicl",
    "tabiclv2_diagonal",
    "tabpfn3_diagonal",
    "tabpfn25_diagonal",
    "bayesian_linear",
    "gp_rbf",
    "gp_matern32",
    "catboost_process",
]


def metadata(data: Any) -> dict[str, Any]:
    return json.loads(str(data["metadata"].item()))


def coverage_fields(error: float, sd: float) -> dict[str, float]:
    result: dict[str, float] = {}
    for level, z in (
        (50, 0.6744897501960817),
        (80, 1.2815515655446004),
        (90, 1.6448536269514722),
        (95, 1.959963984540054),
    ):
        result[f"coverage_{level}"] = float(abs(error) <= z * sd)
        result[f"width_{level}"] = float(2.0 * z * sd)
    return result


def paired_randomization(effects: np.ndarray, repetitions: int = 100_000) -> float:
    effects = np.asarray(effects, dtype=np.float64)
    observed = abs(float(effects.mean()))
    rng = np.random.default_rng(20270311)
    exceed = 0
    for start in range(0, repetitions, 10_000):
        count = min(10_000, repetitions - start)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(count, len(effects)))
        exceed += int(np.sum(np.abs((signs * effects).mean(axis=1)) >= observed - 1e-15))
    return float((exceed + 1) / (repetitions + 1))


def bootstrap_interval(effects: np.ndarray, repetitions: int | None = None) -> tuple[float, float]:
    effects = np.asarray(effects, dtype=np.float64)
    repetitions = int(repetitions or CONFIG["bootstrap_repetitions"])
    rng = np.random.default_rng(20270312)
    indices = rng.integers(len(effects), size=(repetitions, len(effects)))
    values = effects[indices].mean(axis=1)
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def paired_summary(effects: pd.Series) -> dict[str, Any]:
    values = effects.to_numpy(dtype=np.float64)
    wins = int(np.sum(values > 0))
    losses = int(np.sum(values < 0))
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "bootstrap_95": list(bootstrap_interval(values)),
        "wins": wins,
        "losses": losses,
        "ties": int(np.sum(values == 0)),
        "win_rate": float(wins / max(wins + losses, 1)),
        "sign_p": float(binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses else 1.0,
        "paired_randomization_p": paired_randomization(values),
    }


def select_tabpfn_temperatures(
    prediction_root: str = "tabpfn_episodes",
    source_root: str = "tabicl_episodes",
    artifact_root: str = "baselines",
    artifact_prefix: str = "tabpfn",
    model_label: str = "TabPFN-2.5",
) -> dict[int, float]:
    """Select global point-NLL scales on the six development datasets."""
    tabicl_root = CACHE / source_root / "dev"
    tabpfn_root = CACHE / prediction_root / "dev"
    source_paths = sorted(tabicl_root.glob("*.npz"))
    expected = (
        len(CONFIG["development_datasets"])
        * len(CONFIG["development_splits"])
        * int(CONFIG["development_context_replicates"])
        * len(CONFIG["context_sizes"])
    )
    if len(source_paths) != expected:
        raise RuntimeError(f"development TabICL cache incomplete: {len(source_paths)} != {expected}")
    records: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    for source_path in source_paths:
        prediction_path = tabpfn_root / source_path.name
        if not prediction_path.exists():
            raise RuntimeError(f"missing TabPFN development prediction: {prediction_path}")
        with np.load(source_path, allow_pickle=False) as source, np.load(
            prediction_path, allow_pickle=False
        ) as prediction:
            meta = metadata(source)
            pred_meta = metadata(prediction)
            for key in ("dataset", "split", "replicate", "context_size", "query_index_sha256"):
                if pred_meta[key] != meta[key]:
                    raise ValueError(f"cache mismatch for {source_path.name}: {key}")
            point = source["coefficients"].astype(np.float64)[0]
            target = source["target"].astype(np.float64)
            mean = prediction["mean"].astype(np.float64)
            variance = prediction["variance"].astype(np.float64)
            examples.append(
                {
                    "dataset": meta["dataset"],
                    "context_size": int(meta["context_size"]),
                    "truth": point @ target,
                    "mean": point @ mean,
                    "variance": (point**2) @ variance,
                }
            )

    grid = np.asarray(CONFIG["marginal_temperature_grid"], dtype=np.float64)
    selected: dict[int, float] = {}
    for context_size in map(int, CONFIG["context_sizes"]):
        subset = [row for row in examples if row["context_size"] == context_size]
        datasets = sorted({str(row["dataset"]) for row in subset})
        scores = []
        for temperature in grid:
            dataset_scores = []
            for dataset in datasets:
                rows = [row for row in subset if row["dataset"] == dataset]
                truth = np.concatenate([row["truth"] for row in rows])
                mean = np.concatenate([row["mean"] for row in rows])
                variance = np.concatenate([row["variance"] for row in rows])
                dataset_scores.append(float(gaussian_scores(truth, mean, temperature * variance)["nll"].mean()))
            score = float(np.mean(dataset_scores))
            records.append(
                {
                    "context_size": context_size,
                    "temperature": float(temperature),
                    "dataset_balanced_point_nll": score,
                }
            )
            scores.append(score)
        selected[context_size] = float(grid[int(np.argmin(scores))])

    out = CACHE / "results" / artifact_root
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out / f"{artifact_prefix}_temperature_hpo.csv", index=False)
    atomic_json(
        out / f"{artifact_prefix}_temperature_selected.json",
        {
            "selected": {str(key): value for key, value in selected.items()},
            "model": model_label,
            "objective": "dataset-balanced sampled-point Gaussian NLL on six development datasets",
            "grid": list(map(float, grid)),
            "development_episode_count": len(source_paths),
        },
    )
    return selected


def functional_rows(
    meta: dict[str, Any],
    method: str,
    mean: np.ndarray,
    covariance: np.ndarray,
    target: np.ndarray,
    coefficients: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    truth = np.einsum("fgn,n->fg", coefficients, target)
    prediction = np.einsum("fgn,n->fg", coefficients, mean)
    variance = np.einsum("fgn,nm,fgm->fg", coefficients, covariance, coefficients)
    families = list(CONFIG["query_families"])
    for family_index, family in enumerate(families):
        for group in range(int(CONFIG["query_groups"])):
            value = float(truth[family_index, group])
            location = float(prediction[family_index, group])
            var = float(max(variance[family_index, group], 1e-10))
            scores = {key: float(score) for key, score in gaussian_scores(value, location, var).items()}
            rows.append(
                {
                    "dataset": meta["dataset"],
                    "source_id": meta["source_id"],
                    "split": meta["split"],
                    "replicate": int(meta["replicate"]),
                    "context_size": int(meta["context_size"]),
                    "method": method,
                    "family": family,
                    "group": group,
                    "target": value,
                    "mean": location,
                    "variance": var,
                    **scores,
                    **coverage_fields(value - location, math.sqrt(var)),
                }
            )
    return rows


def point_rows(
    meta: dict[str, Any], method: str, mean: np.ndarray, variance: np.ndarray, target: np.ndarray
) -> list[dict[str, Any]]:
    scores = gaussian_scores(target, mean, variance)
    result: list[dict[str, Any]] = []
    for row in range(len(target)):
        sd = math.sqrt(max(float(variance[row]), 1e-10))
        result.append(
            {
                "dataset": meta["dataset"],
                "split": meta["split"],
                "replicate": int(meta["replicate"]),
                "context_size": int(meta["context_size"]),
                "method": method,
                "query_row": row,
                "target": float(target[row]),
                "mean": float(mean[row]),
                "variance": float(variance[row]),
                "metric_scale": float(meta["metric_scale"]),
                **{key: float(value[row]) for key, value in scores.items()},
                **coverage_fields(float(target[row] - mean[row]), sd),
            }
        )
    return result


def mean_only_point_rows(
    meta: dict[str, Any], method: str, mean: np.ndarray, target: np.ndarray
) -> list[dict[str, Any]]:
    """Score a point-only method without fabricating predictive variances."""
    result: list[dict[str, Any]] = []
    for row in range(len(target)):
        error = float(target[row] - mean[row])
        result.append(
            {
                "dataset": meta["dataset"],
                "split": meta["split"],
                "replicate": int(meta["replicate"]),
                "context_size": int(meta["context_size"]),
                "method": method,
                "query_row": row,
                "target": float(target[row]),
                "mean": float(mean[row]),
                "variance": np.nan,
                "metric_scale": float(meta["metric_scale"]),
                "squared_error": error**2,
                "nll": np.nan,
                "crps": np.nan,
                "coverage_50": np.nan,
                "coverage_80": np.nan,
                "coverage_90": np.nan,
                "coverage_95": np.nan,
                "width_50": np.nan,
                "width_80": np.nan,
                "width_90": np.nan,
                "width_95": np.nan,
            }
        )
    return result


def score_evaluation(
    tabpfn_temperatures: dict[int, float],
    query_mode: str = "batched",
    tabpfn3_temperatures: dict[int, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    singleton = query_mode == "singleton"
    tabicl_root = CACHE / ("tabicl_singleton_episodes" if singleton else "tabicl_episodes") / "eval"
    tabpfn_root = CACHE / "tabpfn_episodes" / "eval"
    tabpfn3_root = CACHE / "tabpfn3_episodes" / "eval"
    tabdpt_root = CACHE / "tabdpt_episodes" / "eval"
    classical_root = CACHE / "classical_episodes" / "eval"
    source_paths = sorted(tabicl_root.glob("*.npz"))
    expected = (
        len(CONFIG["evaluation_tasks"])
        * len(CONFIG["evaluation_folds"])
        * int(CONFIG["context_replicates"])
        * len(CONFIG["context_sizes"])
    )
    if len(source_paths) != expected:
        raise RuntimeError(f"evaluation TabICL cache incomplete: {len(source_paths)} != {expected}")
    head_dir = "head_singleton" if singleton else "head"
    head_summary = json.loads((CACHE / head_dir / "training_summary.json").read_text())
    aggregate: list[dict[str, Any]] = []
    points: list[dict[str, Any]] = []
    integrity: list[dict[str, Any]] = []
    timing: list[dict[str, Any]] = []

    for index, source_path in enumerate(source_paths):
        tabpfn_path = tabpfn_root / source_path.name
        tabpfn3_path = tabpfn3_root / source_path.name
        tabdpt_path = tabdpt_root / source_path.name
        classical_path = classical_root / source_path.name
        if (
            not tabpfn_path.exists()
            or not classical_path.exists()
            or (singleton and not tabdpt_path.exists())
            or (singleton and not tabpfn3_path.exists())
        ):
            raise RuntimeError(f"missing baseline prediction for {source_path.name}")
        with np.load(source_path, allow_pickle=False) as source, np.load(
            tabpfn_path, allow_pickle=False
        ) as tabpfn, np.load(classical_path, allow_pickle=False) as classical:
            meta = metadata(source)
            tabpfn_meta = metadata(tabpfn)
            classical_meta = metadata(classical)
            for other, name in ((tabpfn_meta, "TabPFN"), (classical_meta, "classical")):
                for key in ("dataset", "split", "replicate", "context_size", "query_index_sha256"):
                    if other[key] != meta[key]:
                        raise ValueError(f"{name} cache mismatch for {source_path.name}: {key}")
            target = source["target"].astype(np.float64)
            coefficients = source["coefficients"].astype(np.float64)
            context_size = int(meta["context_size"])

            tabicl_mean = source["mean"].astype(np.float64)
            tabicl_variance = source["variance"].astype(np.float64) * float(
                head_summary["marginal_temperatures"][str(context_size)]
            )
            points.extend(
                point_rows(meta, "tabiclv2_projtabicl_marginal", tabicl_mean, tabicl_variance, target)
            )
            timing.append(
                {
                    "dataset": meta["dataset"],
                    "context_size": context_size,
                    "method": "tabiclv2_backbone",
                    "elapsed_seconds": float(meta["elapsed_seconds"]),
                }
            )

            if singleton:
                if tabpfn3_temperatures is None:
                    raise RuntimeError("TabPFN-3 temperatures missing in singleton scoring")
                with np.load(tabpfn3_path, allow_pickle=False) as tabpfn3:
                    tabpfn3_meta = metadata(tabpfn3)
                    for key in ("dataset", "split", "replicate", "context_size", "query_index_sha256"):
                        if tabpfn3_meta[key] != meta[key]:
                            raise ValueError(f"TabPFN-3 cache mismatch for {source_path.name}: {key}")
                    tabpfn3_mean = tabpfn3["mean"].astype(np.float64)
                    tabpfn3_variance = (
                        tabpfn3["variance"].astype(np.float64)
                        * tabpfn3_temperatures[context_size]
                    )
                    aggregate.extend(
                        functional_rows(
                            meta,
                            "tabpfn3_diagonal",
                            tabpfn3_mean,
                            np.diag(tabpfn3_variance),
                            target,
                            coefficients,
                        )
                    )
                    points.extend(
                        point_rows(meta, "tabpfn3", tabpfn3_mean, tabpfn3_variance, target)
                    )
                    timing.append(
                        {
                            "dataset": meta["dataset"],
                            "context_size": context_size,
                            "method": "tabpfn3",
                            "elapsed_seconds": float(tabpfn3_meta["elapsed_seconds"]),
                        }
                    )

            if singleton:
                with np.load(tabdpt_path, allow_pickle=False) as tabdpt:
                    tabdpt_meta = metadata(tabdpt)
                    for key in ("dataset", "split", "replicate", "context_size", "query_index_sha256"):
                        if tabdpt_meta[key] != meta[key]:
                            raise ValueError(f"TabDPT cache mismatch for {source_path.name}: {key}")
                    tabdpt_mean = tabdpt["mean"].astype(np.float64)
                    points.extend(mean_only_point_rows(meta, "tabdpt_turbo_1_2", tabdpt_mean, target))
                    timing.append(
                        {
                            "dataset": meta["dataset"],
                            "context_size": context_size,
                            "method": "tabdpt_turbo_1_2",
                            "elapsed_seconds": float(tabdpt_meta["elapsed_seconds"]),
                        }
                    )

            tabpfn_mean = tabpfn["mean"].astype(np.float64)
            tabpfn_variance = tabpfn["variance"].astype(np.float64) * tabpfn_temperatures[context_size]
            tabpfn_covariance = np.diag(tabpfn_variance)
            aggregate.extend(
                functional_rows(
                    meta,
                    "tabpfn25_diagonal",
                    tabpfn_mean,
                    tabpfn_covariance,
                    target,
                    coefficients,
                )
            )
            points.extend(point_rows(meta, "tabpfn25", tabpfn_mean, tabpfn_variance, target))
            timing.append(
                {
                    "dataset": meta["dataset"],
                    "context_size": context_size,
                    "method": "tabpfn25",
                    "elapsed_seconds": float(tabpfn_meta["elapsed_seconds"]),
                }
            )

            means = classical["means"].astype(np.float64)
            covariances = classical["covariances"].astype(np.float64)
            if list(classical_meta["methods"]) != CLASSICAL_METHODS:
                raise ValueError(f"unexpected classical method order in {classical_path}")
            for method_index, method in enumerate(CLASSICAL_METHODS):
                mean = means[method_index]
                covariance = 0.5 * (covariances[method_index] + covariances[method_index].T)
                aggregate.extend(functional_rows(meta, method, mean, covariance, target, coefficients))
                aggregate.extend(
                    functional_rows(
                        meta,
                        f"{method}_diagonal",
                        mean,
                        np.diag(np.diag(covariance)),
                        target,
                        coefficients,
                    )
                )
                points.extend(point_rows(meta, method, mean, np.diag(covariance), target))
                timing.append(
                    {
                        "dataset": meta["dataset"],
                        "context_size": context_size,
                        "method": method,
                        "elapsed_seconds": float(classical_meta["elapsed_seconds"][method_index]),
                    }
                )
                integrity.append(
                    {
                        "path": str(classical_path),
                        "dataset": meta["dataset"],
                        "context_size": context_size,
                        "method": method,
                        "symmetry_max_abs": float(np.max(np.abs(covariance - covariance.T))),
                        "minimum_eigenvalue": float(np.linalg.eigvalsh(covariance).min()),
                        "minimum_diagonal": float(np.diag(covariance).min()),
                    }
                )
        if (index + 1) % 50 == 0:
            print(f"scored baselines {index + 1}/{len(source_paths)}", flush=True)

    return pd.DataFrame(aggregate), pd.DataFrame(points), pd.DataFrame(integrity), pd.DataFrame(timing)


def macro_table(cells: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    within = cells.groupby(["dataset", *keys], as_index=False)[metrics].mean()
    return within.groupby(keys, as_index=False)[metrics].mean()


def summarize(
    combined: pd.DataFrame, points: pd.DataFrame, timing: pd.DataFrame, out: Path
) -> dict[str, Any]:
    primary = combined[combined["family"].isin(CONFIG["primary_aggregate_families"])].copy()
    metrics = ["nll", "crps", "squared_error", "coverage_90", "width_90"]
    dataset_primary = primary.groupby(["dataset", "method"], as_index=False)[metrics].mean()
    all_methods = dataset_primary.groupby("method", as_index=False)[metrics].mean()
    all_methods["calibration_error_90"] = (all_methods["coverage_90"] - 0.9).abs()
    all_methods.to_csv(out / "aggregate_by_method_all.csv", index=False)

    available_main_methods = [
        method for method in MAIN_METHODS if method in set(dataset_primary["method"])
    ]
    main_dataset = dataset_primary[dataset_primary["method"].isin(available_main_methods)].copy()
    main_dataset["nll_rank"] = main_dataset.groupby("dataset")["nll"].rank(method="average")
    main_dataset["crps_rank"] = main_dataset.groupby("dataset")["crps"].rank(method="average")
    main = main_dataset.groupby("method", as_index=False).agg(
        nll=("nll", "mean"),
        crps=("crps", "mean"),
        squared_error=("squared_error", "mean"),
        coverage_90=("coverage_90", "mean"),
        width_90=("width_90", "mean"),
        nll_rank=("nll_rank", "mean"),
        crps_rank=("crps_rank", "mean"),
    )
    main.to_csv(out / "aggregate_main_methods.csv", index=False)
    dataset_primary.to_csv(out / "aggregate_by_dataset.csv", index=False)

    by_context = macro_table(primary, ["context_size", "method"], metrics)
    by_family = macro_table(primary, ["family", "method"], metrics)
    by_context.to_csv(out / "aggregate_by_context.csv", index=False)
    by_family.to_csv(out / "aggregate_by_family.csv", index=False)

    point_dataset = points.groupby(["dataset", "method"], as_index=False).agg(
        mse=("squared_error", "mean"),
        nll=("nll", "mean"),
        crps=("crps", "mean"),
        coverage_90=("coverage_90", "mean"),
        width_90=("width_90", "mean"),
    )
    point_dataset["nrmse"] = np.sqrt(point_dataset["mse"])
    point_summary = point_dataset.groupby("method", as_index=False)[
        ["nrmse", "nll", "crps", "coverage_90", "width_90"]
    ].mean()
    point_dataset.to_csv(out / "point_by_dataset.csv", index=False)
    point_summary.to_csv(out / "point_by_method.csv", index=False)

    timing_summary = timing.groupby("method", as_index=False).agg(
        median_seconds=("elapsed_seconds", "median"),
        mean_seconds=("elapsed_seconds", "mean"),
        p90_seconds=("elapsed_seconds", lambda values: float(np.quantile(values, 0.9))),
        total_seconds=("elapsed_seconds", "sum"),
    )
    timing_summary.to_csv(out / "timing_by_method.csv", index=False)

    wide_nll = dataset_primary.pivot(index="dataset", columns="method", values="nll")
    wide_crps = dataset_primary.pivot(index="dataset", columns="method", values="crps")
    paired: dict[str, Any] = {}
    for comparator in available_main_methods:
        if comparator == "projtabicl":
            continue
        nll_effect = wide_nll[comparator] - wide_nll["projtabicl"]
        crps_effect = wide_crps[comparator] - wide_crps["projtabicl"]
        paired[comparator] = {
            "nll_comparator_minus_projtabicl": paired_summary(nll_effect),
            "crps_comparator_minus_projtabicl": paired_summary(crps_effect),
        }

    diagonal_effect = wide_nll["tabiclv2_diagonal"] - wide_nll["projtabicl"]
    abs_largest = str(diagonal_effect.abs().idxmax())
    loo = {
        str(dataset): float(diagonal_effect.drop(index=dataset).mean())
        for dataset in diagonal_effect.index
    }
    robustness = {
        "effect_definition": "TabICLv2 diagonal NLL minus ProjTabICL NLL; positive favors ProjTabICL",
        "mean": float(diagonal_effect.mean()),
        "median": float(diagonal_effect.median()),
        "trimmed_mean_10_percent": float(trim_mean(diagonal_effect.to_numpy(), 0.1)),
        "quantiles": {
            str(q): float(diagonal_effect.quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
        "largest_absolute_dataset": abs_largest,
        "mean_without_largest_absolute_dataset": float(diagonal_effect.drop(index=abs_largest).mean()),
        "leave_one_dataset_out_min": float(min(loo.values())),
        "leave_one_dataset_out_max": float(max(loo.values())),
        "leave_one_dataset_out": loo,
    }
    pd.DataFrame(
        {
            "dataset": diagonal_effect.index,
            "nll_diagonal_minus_projtabicl": diagonal_effect.values,
            "crps_diagonal_minus_projtabicl": (
                wide_crps["tabiclv2_diagonal"] - wide_crps["projtabicl"]
            ).reindex(diagonal_effect.index).values,
        }
    ).to_csv(out / "projective_effects_by_dataset.csv", index=False)

    return {
        "evaluation_episodes": int(
            combined[["dataset", "split", "replicate", "context_size"]].drop_duplicates().shape[0]
        ),
        "datasets": int(combined["dataset"].nunique()),
        "aggregate_cells": int(len(combined)),
        "point_cells": int(len(points)),
        "main_methods": available_main_methods,
        "main_table": main.to_dict(orient="records"),
        "point_table": point_summary.to_dict(orient="records"),
        "paired_against_projtabicl": paired,
        "projective_robustness": robustness,
    }


def main(args: argparse.Namespace) -> None:
    singleton = args.query_mode == "singleton"
    temperatures = select_tabpfn_temperatures()
    tabpfn3_temperatures = (
        select_tabpfn_temperatures(
            prediction_root="tabpfn3_episodes",
            source_root="tabicl_singleton_episodes",
            artifact_root="baselines_singleton",
            artifact_prefix="tabpfn3",
            model_label="TabPFN-3",
        )
        if singleton
        else None
    )
    baseline_cells, points, integrity, timing = score_evaluation(
        temperatures, args.query_mode, tabpfn3_temperatures
    )
    projective_dir = "projective_singleton" if singleton else "projective"
    projective_path = CACHE / "results" / projective_dir / "cells.parquet"
    if not projective_path.exists():
        raise RuntimeError("score_projective.py must be run first")
    projective_cells = pd.read_parquet(projective_path)
    combined = pd.concat([projective_cells, baseline_cells], ignore_index=True)
    out = CACHE / "results" / ("baselines_singleton" if singleton else "baselines")
    out.mkdir(parents=True, exist_ok=True)
    baseline_cells.to_parquet(out / "baseline_cells.parquet", index=False)
    combined.to_parquet(out / "all_cells.parquet", index=False)
    points.to_parquet(out / "point_cells.parquet", index=False)
    integrity.to_csv(out / "classical_integrity.csv", index=False)
    timing.to_csv(out / "timing_cells.csv", index=False)
    summary = summarize(combined, points, timing, out)
    summary["query_mode"] = args.query_mode
    summary["tabpfn_temperatures"] = {str(key): value for key, value in temperatures.items()}
    summary["tabpfn3_temperatures"] = (
        {str(key): value for key, value in tabpfn3_temperatures.items()}
        if tabpfn3_temperatures is not None
        else None
    )
    summary["integrity"] = {
        "minimum_classical_eigenvalue": float(integrity["minimum_eigenvalue"].min()),
        "minimum_classical_diagonal": float(integrity["minimum_diagonal"].min()),
        "maximum_classical_symmetry_error": float(integrity["symmetry_max_abs"].max()),
    }
    atomic_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-mode", choices=["batched", "singleton"], default="batched")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
