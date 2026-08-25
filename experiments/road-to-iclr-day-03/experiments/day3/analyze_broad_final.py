"""Final statistical analysis for broad, confirmation, and robustness tiers."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t, wilcoxon

from .analyze_broad_phase1 import bootstrap, controlled_pairs, load_phase1
from .broad_data import config


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/day3/broad_benchmark"
FIGURES = RESULTS / "figures"


def interval(values: pd.Series) -> tuple[float, float, float, float]:
    x = values.dropna().to_numpy(float)
    mean = float(x.mean()) if len(x) else math.nan
    std = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    radius = float(t.ppf(0.975, len(x) - 1) * std / math.sqrt(len(x))) if len(x) > 1 else 0.0
    return mean, std, mean - radius, mean + radius


def load_confirmation() -> pd.DataFrame:
    paths = sorted(
        path
        for path in RESULTS.glob("confirmation_shard*.csv")
        if not path.stem.endswith("_curves")
    )
    if not paths:
        raise FileNotFoundError("No confirmation shards")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    key = ["dataset", "target_kappa", "model", "remedy", "seed"]
    return frame.drop_duplicates(key, keep="last")


def confirmation_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, part in pairs.groupby(["dataset", "task", "model", "remedy"]):
        sensitivity = interval(part.sensitivity_normalized)
        gain = interval(part.k1_gain_normalized)
        reduction = interval(part.sensitivity_reduction)
        rows.append({
            "dataset": keys[0],
            "task": keys[1],
            "model": keys[2],
            "remedy": keys[3],
            "seeds": len(part),
            "sensitivity_mean": sensitivity[0],
            "sensitivity_std": sensitivity[1],
            "sensitivity_ci_low": sensitivity[2],
            "sensitivity_ci_high": sensitivity[3],
            "k1_gain_mean": gain[0],
            "k1_gain_std": gain[1],
            "k1_gain_ci_low": gain[2],
            "k1_gain_ci_high": gain[3],
            "sensitivity_reduction_mean": reduction[0],
            "sensitivity_reduction_ci_low": reduction[2],
            "sensitivity_reduction_ci_high": reduction[3],
        })
    return pd.DataFrame(rows)


def confirmation_aggregate(pairs: pd.DataFrame) -> pd.DataFrame:
    source_map = config()["source_groups"]
    rows = []
    for remedy, part in pairs.groupby("remedy"):
        part = part.copy()
        part["source_group"] = part.dataset.map(source_map).fillna(part.dataset)
        sensitivity = part.groupby(["source_group", "model"]).sensitivity_normalized.mean()
        k1_gain = part.groupby(["source_group", "model"]).k1_gain_normalized.mean()
        reduction = part.groupby(["source_group", "model"]).sensitivity_reduction.mean()
        sensitivity_ci = bootstrap(sensitivity.to_numpy())
        gain_ci = bootstrap(k1_gain.to_numpy())
        nonzero = sensitivity[sensitivity.abs() > 1e-12]
        rows.append({
            "remedy": remedy,
            "pairs": len(part),
            "datasets": int(part.dataset.nunique()),
            "models": int(part.model.nunique()),
            "source_model_clusters": len(sensitivity),
            "mean_sensitivity_normalized": float(part.sensitivity_normalized.mean()),
            "median_sensitivity_normalized": float(part.sensitivity_normalized.median()),
            "harmful_fraction": float((part.sensitivity_normalized < 0).mean()),
            "cluster_sensitivity_mean": float(sensitivity.mean()),
            "cluster_sensitivity_ci_low": sensitivity_ci[0],
            "cluster_sensitivity_ci_high": sensitivity_ci[1],
            "cluster_sensitivity_wilcoxon_p": (
                float(wilcoxon(nonzero).pvalue) if len(nonzero) >= 3 else math.nan
            ),
            "mean_k1_gain_normalized": float(part.k1_gain_normalized.mean()),
            "cluster_k1_gain_mean": float(k1_gain.mean()),
            "cluster_k1_gain_ci_low": gain_ci[0],
            "cluster_k1_gain_ci_high": gain_ci[1],
            "mean_sensitivity_reduction": float(reduction.mean()),
        })
    output = pd.DataFrame(rows)
    output.to_csv(RESULTS / "confirmation_aggregate.csv", index=False)
    return output


def paper_gates(
    phase_pairs: pd.DataFrame,
    confirmation: pd.DataFrame,
    confirmation_raw: pd.DataFrame,
) -> dict[str, object]:
    cfg = config()["claim_gates"]
    adam = phase_pairs[phase_pairs.remedy.eq("adamw")].copy()
    source_map = config()["source_groups"]
    adam["source_group"] = adam.dataset.map(source_map).fillna(adam.dataset)
    grouped = adam.groupby(["source_group", "model"]).sensitivity_normalized.mean()
    ci = bootstrap(grouped.to_numpy(), int(cfg["bootstrap_samples"]))
    sensitivity_gate = bool(
        grouped.median() < 0
        and ci[1] < 0
        and (adam.sensitivity_normalized < 0).mean() >= float(cfg["minimum_harmful_fraction"])
    )
    remedies = {}
    expected_pairs = (
        len(config()["architecture_confirmation_datasets"])
        * len(config()["models"])
        * len(config()["confirmation_seeds"])
    )
    expected_raw = expected_pairs * len(config()["kappas"])
    raw_keys = ["dataset", "target_kappa", "model", "remedy", "seed"]
    raw_unique = confirmation_raw.drop_duplicates(raw_keys, keep="last")
    failure_rates = {}
    for remedy in raw_unique.remedy.unique():
        raw_part = raw_unique[raw_unique.remedy.eq(remedy)]
        missing = max(expected_raw - len(raw_part), 0)
        failures = int(raw_part.failure.fillna("").ne("").sum()) + missing
        failure_rates[remedy] = failures / expected_raw
    adam_failure_rate = failure_rates.get("adamw", 1.0)
    for remedy, part in confirmation.groupby("remedy"):
        mean_reduction = float(part.sensitivity_reduction.mean())
        mean_loss = float(-part.k1_gain_normalized.mean())
        failure_rate = failure_rates.get(remedy, 1.0)
        complete = len(part) == expected_pairs
        no_divergence_excess = failure_rate <= adam_failure_rate
        remedies[remedy] = {
            "mean_sensitivity_reduction": mean_reduction,
            "mean_unperturbed_normalized_loss": mean_loss,
            "confirmation_pairs": len(part),
            "expected_confirmation_pairs": expected_pairs,
            "failure_rate_including_missing": failure_rate,
            "adamw_failure_rate_including_missing": adam_failure_rate,
            "no_divergence_excess": no_divergence_excess,
            "passes": bool(
                mean_reduction >= float(cfg["minimum_sensitivity_reduction"])
                and mean_loss <= float(cfg["maximum_unperturbed_normalized_loss"])
                and complete
                and no_divergence_excess
            ),
        }
    return {
        "adamw_median_sensitivity": float(grouped.median()),
        "adamw_bootstrap_ci": ci,
        "adamw_harmful_fraction": float((adam.sensitivity_normalized < 0).mean()),
        "basis_sensitivity_gate_passes": sensitivity_gate,
        "remedy_gates": remedies,
    }


def natural_and_preprocessing(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = frame[frame.failure.fillna("").eq("") & frame.target_kappa.eq(1.0)]
    keys = ["dataset", "task", "model", "seed"]
    natural = valid[valid.representation.isin(["cumulative_helmert", "local_adjacent"])]
    natural_wide = natural.pivot_table(index=keys, columns="representation", values="test_primary").reset_index()
    natural_wide["local_minus_cumulative"] = natural_wide.local_adjacent - natural_wide.cumulative_helmert
    natural_wide["scale"] = np.where(
        natural_wide.task.eq("regression"),
        natural_wide.cumulative_helmert.abs().clip(lower=1e-12),
        1.0,
    )
    natural_wide["local_minus_cumulative_normalized"] = (
        natural_wide.local_minus_cumulative / natural_wide.scale
    )
    natural_wide.to_csv(RESULTS / "final_natural_encoding_pairs.csv", index=False)
    preprocess = valid[valid.representation.isin([
        "cumulative_helmert", "raw_standard", "quantile_standard"
    ])]
    preprocess_wide = preprocess.pivot_table(index=keys, columns="representation", values="test_primary").reset_index()
    preprocess_wide["scale"] = np.where(
        preprocess_wide.task.eq("regression"),
        preprocess_wide.cumulative_helmert.abs().clip(lower=1e-12),
        1.0,
    )
    for family in ("raw_standard", "quantile_standard"):
        preprocess_wide[f"{family}_minus_ple"] = preprocess_wide[family] - preprocess_wide.cumulative_helmert
        preprocess_wide[f"{family}_minus_ple_normalized"] = (
            preprocess_wide[f"{family}_minus_ple"] / preprocess_wide.scale
        )
    preprocess_wide.to_csv(RESULTS / "preprocessing_baseline_pairs.csv", index=False)
    return natural_wide, preprocess_wide


def encoding_contrast_summary(
    natural: pd.DataFrame, preprocess: pd.DataFrame
) -> pd.DataFrame:
    source_map = config()["source_groups"]
    contrasts = {
        "local_minus_cumulative": natural.local_minus_cumulative_normalized,
        "raw_standard_minus_ple": preprocess.raw_standard_minus_ple_normalized,
        "quantile_standard_minus_ple": preprocess.quantile_standard_minus_ple_normalized,
    }
    frames = {
        "local_minus_cumulative": natural,
        "raw_standard_minus_ple": preprocess,
        "quantile_standard_minus_ple": preprocess,
    }
    rows = []
    for name, values in contrasts.items():
        working = frames[name][["dataset", "model"]].copy()
        working["value"] = values.to_numpy()
        working["source_group"] = working.dataset.map(source_map).fillna(working.dataset)
        clustered = working.groupby(["source_group", "model"]).value.mean()
        ci = bootstrap(clustered.to_numpy())
        nonzero = clustered[clustered.abs() > 1e-12]
        rows.append({
            "contrast": name,
            "pairs": len(working),
            "datasets": int(working.dataset.nunique()),
            "models": int(working.model.nunique()),
            "mean_normalized_difference": float(working.value.mean()),
            "median_normalized_difference": float(working.value.median()),
            "median_absolute_normalized_difference": float(working.value.abs().median()),
            "positive_fraction": float((working.value > 0).mean()),
            "source_model_cluster_mean": float(clustered.mean()),
            "source_model_ci_low": ci[0],
            "source_model_ci_high": ci[1],
            "source_model_wilcoxon_p": (
                float(wilcoxon(nonzero).pvalue) if len(nonzero) >= 3 else math.nan
            ),
        })
    output = pd.DataFrame(rows)
    output.to_csv(RESULTS / "encoding_contrast_summary.csv", index=False)
    return output


def robustness_analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = sorted(
        path
        for path in RESULTS.glob("robustness_shard*.csv")
        if not path.stem.endswith("_curves")
    )
    if not paths:
        return pd.DataFrame(), pd.DataFrame()
    raw = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    raw.to_csv(RESULTS / "robustness_all.csv", index=False)
    valid = raw[raw.failure.fillna("").eq("")]
    failure_summary = (
        raw.assign(failed=raw.failure.fillna("").ne(""))
        .groupby(["duplicate_fraction", "remedy", "ridge"])
        .agg(runs=("seed", "size"), failures=("failed", "sum"))
        .reset_index()
    )
    failure_summary["failure_rate"] = failure_summary.failures / failure_summary.runs
    failure_summary.to_csv(RESULTS / "robustness_failure_summary.csv", index=False)
    rank_summary = (
        valid.groupby(["duplicate_fraction", "remedy", "ridge"])
        .agg(
            runs=("seed", "size"),
            rank_mean=("rank", "mean"),
            effective_rank_mean=("effective_rank", "mean"),
            condition_number_median=("condition_number", "median"),
            condition_number_max=("condition_number", "max"),
            preconditioner_condition_median=("preconditioner_condition", "median"),
        )
        .reset_index()
    )
    rank_summary.to_csv(RESULTS / "robustness_rank_summary.csv", index=False)
    keys = ["dataset", "task", "duplicate_fraction", "remedy", "ridge", "seed"]
    pairs = valid.pivot_table(index=keys, columns="target_kappa", values="test_primary").reset_index()
    pairs = pairs.dropna(subset=[1.0, 1000.0])
    pairs["scale"] = np.where(pairs.task.eq("regression"), pairs[1.0].abs().clip(lower=1e-12), 1.0)
    pairs["sensitivity_normalized"] = (pairs[1000.0] - pairs[1.0]) / pairs.scale
    pairs.to_csv(RESULTS / "robustness_pairs.csv", index=False)
    sensitivity_summary = (
        pairs.groupby(["dataset", "duplicate_fraction", "remedy", "ridge"])
        .agg(
            seeds=("seed", "size"),
            sensitivity_mean=("sensitivity_normalized", "mean"),
            sensitivity_std=("sensitivity_normalized", "std"),
            harmful_fraction=("sensitivity_normalized", lambda x: float((x < 0).mean())),
        )
        .reset_index()
    )
    sensitivity_summary.to_csv(RESULTS / "robustness_sensitivity_summary.csv", index=False)
    ridge_summary = (
        pairs[pairs.remedy.eq("input_natural")]
        .groupby(["duplicate_fraction", "ridge"])
        .agg(
            pairs=("seed", "size"),
            sensitivity_mean=("sensitivity_normalized", "mean"),
            sensitivity_median=("sensitivity_normalized", "median"),
            harmful_fraction=("sensitivity_normalized", lambda x: float((x < 0).mean())),
        )
        .reset_index()
    )
    ridge_summary.to_csv(RESULTS / "robustness_ridge_summary.csv", index=False)
    efficiency = (
        valid.groupby(["duplicate_fraction", "remedy", "ridge"])
        .agg(
            runs=("seed", "size"),
            preprocessing_seconds=("preprocessing_seconds", "mean"),
            training_seconds=("train_seconds", "mean"),
            peak_cuda_bytes=("peak_cuda_bytes", "mean"),
        )
        .reset_index()
    )
    efficiency.to_csv(RESULTS / "robustness_efficiency.csv", index=False)
    return pairs, efficiency


def efficiency_analysis(phase: pd.DataFrame, confirmation: pd.DataFrame) -> None:
    """Export training cost and de-duplicated remedy-transform cost summaries."""

    def summarize(frame: pd.DataFrame, output: str) -> pd.DataFrame:
        valid = frame[frame.failure.fillna("").eq("")].copy()
        training = (
            valid.groupby(["model", "remedy"])
            .agg(
                runs=("seed", "size"),
                train_seconds_mean=("train_seconds", "mean"),
                train_seconds_median=("train_seconds", "median"),
                peak_cuda_bytes_mean=("peak_cuda_bytes", "mean"),
                peak_cuda_bytes_median=("peak_cuda_bytes", "median"),
                valid_peak_memory_runs=("peak_memory_observation_valid", "sum"),
                parameters_mean=("parameters", "mean"),
            )
            .reset_index()
        )
        transforms = valid.drop_duplicates(
            ["dataset", "representation", "target_kappa", "remedy"]
        )
        prep = (
            transforms.groupby("remedy")
            .agg(
                unique_transforms=("dataset", "size"),
                preprocessing_seconds_mean=("preprocessing_seconds", "mean"),
                preprocessing_seconds_median=("preprocessing_seconds", "median"),
                preprocessing_seconds_total=("preprocessing_seconds", "sum"),
            )
            .reset_index()
        )
        summary = training.merge(prep, on="remedy", how="left")
        baselines = summary[summary.remedy.eq("adamw")].set_index("model")
        summary["training_time_ratio_to_adamw"] = [
            row.train_seconds_mean / baselines.loc[row.model, "train_seconds_mean"]
            if row.model in baselines.index
            else math.nan
            for row in summary.itertuples()
        ]
        summary["memory_ratio_to_adamw"] = [
            row.peak_cuda_bytes_mean / baselines.loc[row.model, "peak_cuda_bytes_mean"]
            if row.model in baselines.index
            and baselines.loc[row.model, "peak_cuda_bytes_mean"] > 0
            else math.nan
            for row in summary.itertuples()
        ]
        summary.to_csv(RESULTS / output, index=False)
        return summary

    controlled_phase = phase[
        phase.representation.eq("controlled") & phase.model.eq("mlp")
    ]
    summarize(controlled_phase, "phase1_remedy_efficiency.csv")
    summarize(confirmation, "confirmation_efficiency.csv")
    preprocessing = phase[
        phase.remedy.eq("adamw")
        & phase.representation.isin(
            ["cumulative_helmert", "local_adjacent", "raw_standard", "quantile_standard"]
        )
    ]
    (
        preprocessing[preprocessing.failure.fillna("").eq("")]
        .groupby(["model", "representation"])
        .agg(
            runs=("seed", "size"),
            train_seconds_mean=("train_seconds", "mean"),
            train_seconds_median=("train_seconds", "median"),
            peak_cuda_bytes_mean=("peak_cuda_bytes", "mean"),
            valid_peak_memory_runs=("peak_memory_observation_valid", "sum"),
            parameters_mean=("parameters", "mean"),
        )
        .reset_index()
        .to_csv(RESULTS / "preprocessing_baseline_efficiency.csv", index=False)
    )


def figures(summary: pd.DataFrame, phase_pairs: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    aggregate = summary.groupby(["model", "remedy"]).sensitivity_mean.mean().unstack(0)
    order = aggregate.mean(axis=1).sort_values().index
    aggregate = aggregate.loc[order]
    fig, axis = plt.subplots(figsize=(10, 6))
    aggregate.plot.barh(ax=axis)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Mean normalized κ=1000 − κ=1 utility")
    axis.set_ylabel("")
    axis.set_title("Five-seed architecture confirmation")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "confirmation_sensitivity.png", dpi=190)
    plt.close(fig)

    adam = phase_pairs[phase_pairs.remedy.eq("adamw")]
    table = adam.groupby(["dataset", "model"]).sensitivity_normalized.mean().unstack(1)
    fig, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(100 * table.to_numpy(), aspect="auto", cmap="RdYlGn", vmin=-30, vmax=5)
    axis.set_yticks(range(len(table)), table.index, fontsize=7)
    axis.set_xticks(range(len(table.columns)), table.columns, rotation=25, ha="right")
    axis.set_title("AdamW basis sensitivity across 25 datasets")
    fig.colorbar(image, ax=axis, label="κ=1000 − κ=1 utility (%)")
    fig.tight_layout()
    fig.savefig(FIGURES / "adamw_25_dataset_heatmap.png", dpi=190)
    plt.close(fig)


def main() -> None:
    phase = load_phase1()
    phase_pairs = controlled_pairs(phase, "test_primary")
    confirmation_raw = load_confirmation()
    confirmation_pairs = controlled_pairs(confirmation_raw, "test_primary")
    confirmation_raw.to_csv(RESULTS / "confirmation_all.csv", index=False)
    confirmation_pairs.to_csv(RESULTS / "confirmation_pairs.csv", index=False)
    summary = confirmation_summary(confirmation_pairs)
    summary.to_csv(RESULTS / "confirmation_summary.csv", index=False)
    confirmation_aggregate(confirmation_pairs)
    natural, preprocess = natural_and_preprocessing(phase)
    encoding_contrast_summary(natural, preprocess)
    robustness, efficiency = robustness_analysis()
    efficiency_analysis(phase, confirmation_raw)
    gates = paper_gates(phase_pairs, confirmation_pairs, confirmation_raw)
    figures(summary, phase_pairs)

    temporal = phase_pairs[
        phase_pairs.remedy.eq("adamw") & phase_pairs.dataset.isin(config()["temporal_shift_datasets"])
    ]
    non_temporal = phase_pairs[
        phase_pairs.remedy.eq("adamw") & ~phase_pairs.dataset.isin(config()["temporal_shift_datasets"])
    ]
    payload = {
        "phase1_runs": len(phase),
        "confirmation_runs": len(confirmation_raw),
        "confirmation_failures": int(confirmation_raw.failure.fillna("").ne("").sum()),
        "confirmation_cells_with_five_seeds": int((summary.seeds == 5).sum()),
        "confirmation_cells": len(summary),
        "gates": gates,
        "natural_pair_raw_mean_difference": float(natural.local_minus_cumulative.mean()),
        "natural_pair_mean_normalized_difference": float(
            natural.local_minus_cumulative_normalized.mean()
        ),
        "natural_pair_median_absolute_normalized_difference": float(
            natural.local_minus_cumulative_normalized.abs().median()
        ),
        "raw_minus_ple_mean_normalized": float(
            preprocess.raw_standard_minus_ple_normalized.mean()
        ),
        "quantile_minus_ple_mean_normalized": float(
            preprocess.quantile_standard_minus_ple_normalized.mean()
        ),
        "temporal_adamw_mean_sensitivity": float(temporal.sensitivity_normalized.mean()),
        "non_temporal_adamw_mean_sensitivity": float(non_temporal.sensitivity_normalized.mean()),
        "robustness_pairs": len(robustness),
        "efficiency_cells": len(efficiency),
    }
    (RESULTS / "final_summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
