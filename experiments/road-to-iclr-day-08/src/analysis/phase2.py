"""Split-aware summaries and hierarchical inference for Phase II."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


MODEL_FAMILY = {
    "tabpfn_v25_single": "TFM",
    "tabpfn_v25_default": "TFM",
    "tabicl_v2_single": "TFM",
    "tabicl_v2_default": "TFM",
    "mitra_default": "TFM",
    "tabm_default": "trained neural",
    "realmlp_default": "trained neural",
    "xgboost": "tree",
    "catboost": "tree",
    "lightgbm": "tree",
    "random_forest": "tree",
    "linear": "linear",
}


def applicable_transformed(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove identity and categorical no-op cells from scientific summaries."""
    result = frame[frame["transform"] != "identity"].copy()
    categorical_noop = (result["transform_scope"] == "categorical") & (result["n_categorical"] == 0)
    return result[~categorical_noop]


def hierarchical_bootstrap(
    frame: pd.DataFrame,
    value: str,
    *,
    seed: int = 20260831,
    draws: int = 10_000,
) -> dict[str, float]:
    """Equal-dataset bootstrap with split resampling nested within datasets."""
    cells = (
        frame.groupby(["dataset", "split_seed"], as_index=False)[value]
        .mean()
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=[value])
    )
    datasets = sorted(cells["dataset"].unique())
    if not datasets:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "datasets": 0}
    split_values = {
        dataset: cells.loc[cells["dataset"] == dataset, value].to_numpy(dtype=np.float64)
        for dataset in datasets
    }
    observed = float(np.mean([values.mean() for values in split_values.values()]))
    if len(datasets) == 1 and len(split_values[datasets[0]]) == 1:
        return {"mean": observed, "ci_low": observed, "ci_high": observed, "datasets": 1}
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled = rng.choice(datasets, size=len(datasets), replace=True)
        dataset_means = []
        for dataset in sampled:
            values = split_values[str(dataset)]
            dataset_means.append(float(rng.choice(values, size=len(values), replace=True).mean()))
        estimates[draw] = np.mean(dataset_means)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "mean": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "datasets": len(datasets),
    }


def model_summary(frame: pd.DataFrame, draws: int = 10_000) -> pd.DataFrame:
    transformed = applicable_transformed(frame)
    rows: list[dict[str, Any]] = []
    for (model, problem), group in transformed.groupby(["model", "problem_type"], sort=True):
        loss = hierarchical_bootstrap(group, "matched_normalized_loss_gap", draws=draws)
        disagreement = hierarchical_bootstrap(group, "matched_excess_disagreement", draws=draws)
        rows.append(
            {
                "model": model,
                "model_family": MODEL_FAMILY.get(model, "other"),
                "problem_type": problem,
                "datasets": loss["datasets"],
                "mean_matched_normalized_loss_gap": loss["mean"],
                "loss_gap_ci_low": loss["ci_low"],
                "loss_gap_ci_high": loss["ci_high"],
                "mean_excess_disagreement": disagreement["mean"],
                "excess_disagreement_ci_low": disagreement["ci_low"],
                "excess_disagreement_ci_high": disagreement["ci_high"],
            }
        )
    return pd.DataFrame(rows)


def dataset_transform_summary(frame: pd.DataFrame) -> pd.DataFrame:
    transformed = applicable_transformed(frame)
    return (
        transformed.assign(model_family=transformed["model"].map(MODEL_FAMILY).fillna("other"))
        .groupby(
            ["dataset", "split_seed", "problem_type", "model", "model_family", "transform"],
            as_index=False,
        )
        .agg(
            matched_normalized_loss_gap=("matched_normalized_loss_gap", "mean"),
            matched_excess_disagreement=("matched_excess_disagreement", "mean"),
            matched_disagreement=("matched_disagreement", "mean"),
            context_only_disagreement=("context_only_disagreement", "mean"),
            query_only_disagreement=("query_only_disagreement", "mean"),
        )
    )


def flatten_descriptors(records: list[dict[str, Any]]) -> pd.DataFrame:
    """One immutable train-only descriptor row per dataset/split/feature."""
    seen: set[tuple[str, int, str]] = set()
    rows: list[dict[str, Any]] = []
    for record in records:
        dataset, split_seed = record["dataset"], int(record["split_seed"])
        for feature in record["dataset_descriptors"]["features"]:
            key = (dataset, split_seed, feature["feature"])
            if key in seen:
                continue
            seen.add(key)
            row = {"dataset": dataset, "split_seed": split_seed, **feature}
            quantiles = row.pop("quantiles", [])
            for label, value in zip(("q01", "q10", "q25", "q50", "q75", "q90", "q99"), quantiles):
                row[label] = value
            rows.append(row)
    return pd.DataFrame(rows)
