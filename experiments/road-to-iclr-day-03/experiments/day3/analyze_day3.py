"""Analyze raw Day 3 CSVs and generate every figure from machine-readable data."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "day3"
FIGURES = RESULTS / "figures"


def utility(frame: pd.DataFrame) -> pd.Series:
    return np.where(frame["task"].eq("regression"), -frame["test_metric"], frame["test_metric"])


def ci(values: pd.Series) -> tuple[float, float, float, float]:
    x = values.dropna().to_numpy(float)
    mean = float(np.mean(x)) if len(x) else math.nan
    std = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    radius = float(t.ppf(0.975, len(x) - 1) * std / math.sqrt(len(x))) if len(x) > 1 else 0.0
    return mean, std, mean - radius, mean + radius


def line_plot(frame: pd.DataFrame, x: str, y: str, group: str, filename: str, xlabel: str, ylabel: str, logx: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, part in frame.groupby(group, dropna=False):
        summary = part.groupby(x)[y].agg(["mean", "sem"]).reset_index().sort_values(x)
        ax.plot(summary[x], summary["mean"], marker="o", label=str(name))
        ax.fill_between(summary[x], summary["mean"] - 1.96 * summary["sem"].fillna(0), summary["mean"] + 1.96 * summary["sem"].fillna(0), alpha=0.15)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{filename}.png", dpi=180)
    plt.close(fig)


def bar_plot(frame: pd.DataFrame, x: str, y: str, filename: str, ylabel: str, hue: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    if hue is None:
        summary = frame.groupby(x)[y].agg(["mean", "sem"])
        ax.bar(np.arange(len(summary)), summary["mean"], yerr=1.96 * summary["sem"].fillna(0), capsize=3)
        ax.set_xticks(np.arange(len(summary)), summary.index, rotation=25, ha="right")
    else:
        means = frame.pivot_table(index=x, columns=hue, values=y, aggfunc="mean")
        errors = frame.pivot_table(index=x, columns=hue, values=y, aggfunc="sem").fillna(0)
        width = 0.8 / len(means.columns)
        base = np.arange(len(means))
        for i, column in enumerate(means.columns):
            ax.bar(base + (i - (len(means.columns) - 1) / 2) * width, means[column], width, yerr=1.96 * errors[column], label=str(column), capsize=2)
        ax.set_xticks(base, means.index, rotation=25, ha="right")
        ax.legend(frameon=False, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{filename}.png", dpi=180)
    plt.close(fig)


def paired_delta(frame: pd.DataFrame, baseline_filter: pd.Series, key: list[str], value: str = "utility") -> pd.DataFrame:
    base = frame[baseline_filter][key + [value]].rename(columns={value: "baseline"})
    merged = frame.merge(base, on=key, how="inner")
    merged["delta"] = merged[value] - merged["baseline"]
    return merged


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    files = [
        "numeric_kappa.csv",
        "categorical_kappa.csv",
        "ordinal_basis.csv",
        "ordinal_kappa.csv",
        "ple_identity_whitening_exact.csv",
        "invariant_regularizer.csv",
        "block_residualization.csv",
        "cyclic_geometry.csv",
        "residual_te.csv",
        "frequency_preconditioning.csv",
    ]
    frames = []
    inventory = []
    for name in files:
        path = RESULTS / name
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frames.append(frame)
        inventory.append({"file": name, "rows": len(frame), "datasets": sorted(frame.dataset.unique().tolist()), "models": sorted(frame.model.unique().tolist())})
    if not frames:
        raise SystemExit("No Day 3 result CSVs found")
    all_results = pd.concat(frames, ignore_index=True, sort=False)
    all_results["utility"] = utility(all_results)
    all_results.to_csv(RESULTS / "all_results.csv", index=False)
    summary: dict[str, object] = {"inventory": inventory, "total_runs": len(all_results)}

    numeric = all_results[all_results.experiment.eq("numeric_kappa")].copy()
    if len(numeric):
        numeric["display_delta"] = numeric.groupby(["dataset", "model", "seed"])["utility"].transform(lambda s: s - s.iloc[np.argmin(numeric.loc[s.index, "target_kappa"].to_numpy())])
        line_plot(numeric, "target_kappa", "display_delta", "dataset", "numeric_kappa_vs_metric", "Target condition number κ", "Utility change from κ=1", True)
        line_plot(numeric, "target_kappa", "best_epoch", "dataset", "numeric_kappa_vs_convergence", "Target condition number κ", "Best validation epoch", True)
        correlations = {}
        max_deltas = []
        for (dataset, model), part in numeric.groupby(["dataset", "model"]):
            means = part.groupby("target_kappa").utility.mean()
            rho, pvalue = spearmanr(np.log10(means.index), means.values)
            correlations[f"{dataset}/{model}"] = {"spearman_rho": float(rho), "pvalue": float(pvalue)}
            wide = part.pivot(index="seed", columns="target_kappa", values="utility")
            if 1.0 in wide and max(wide.columns) in wide:
                max_deltas.extend((wide[max(wide.columns)] - wide[1.0]).tolist())
        summary["numeric"] = {"correlations": correlations, "max_kappa_delta_ci": ci(pd.Series(max_deltas))}

    categorical = all_results[all_results.experiment.eq("categorical_kappa")].copy()
    if len(categorical):
        categorical["display_delta"] = categorical.groupby(["dataset", "model", "seed"])["utility"].transform(lambda s: s - s.iloc[np.argmin(categorical.loc[s.index, "target_kappa"].to_numpy())])
        line_plot(categorical, "target_kappa", "display_delta", "dataset", "categorical_kappa_vs_metric", "Target condition number κ", "Utility change from κ=1", True)
        correlations = {}
        max_deltas = []
        for (dataset, model), part in categorical.groupby(["dataset", "model"]):
            means = part.groupby("target_kappa").utility.mean()
            rho, pvalue = spearmanr(np.log10(means.index), means.values)
            correlations[f"{dataset}/{model}"] = {"spearman_rho": float(rho), "pvalue": float(pvalue)}
            wide = part.pivot(index="seed", columns="target_kappa", values="utility")
            if 1.0 in wide and max(wide.columns) in wide:
                max_deltas.extend((wide[max(wide.columns)] - wide[1.0]).tolist())
        summary["categorical"] = {"correlations": correlations, "max_kappa_delta_ci": ci(pd.Series(max_deltas))}

    audit_path = RESULTS / "structured_feature_audit.csv"
    if audit_path.exists():
        audit = pd.read_csv(audit_path)
        cat_audit = audit[audit.semantic_type.isin(["nominal", "ordinal"])].dropna(subset=["categorical_covariance_condition"])
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        scatter = ax.scatter(cat_audit.entropy, cat_audit.categorical_covariance_condition, c=np.log1p(cat_audit.cardinality), cmap="viridis", alpha=0.8)
        ax.set_yscale("log")
        ax.set_xlabel("Category entropy")
        ax.set_ylabel("Nonconstant covariance condition number")
        fig.colorbar(scatter, ax=ax, label="log(1 + cardinality)")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIGURES / "categorical_frequency_spectrum.png", dpi=180)
        plt.close(fig)

    whitening = all_results[all_results.experiment.eq("ple_identity_whitening")].copy()
    if len(whitening):
        gaps = whitening.pivot_table(index=["dataset", "model", "seed", "canonicalization"], columns="family", values="utility").reset_index()
        gaps["gap"] = gaps["identity"] - gaps["ple"]
        gaps["absolute_gap"] = gaps.gap.abs()
        bar_plot(gaps, "canonicalization", "absolute_gap", "ple_identity_gap_before_after_whitening", "Absolute identity–PLE utility gap")
        raw = gaps[gaps.canonicalization.eq("raw")].absolute_gap.mean()
        white = gaps[gaps.canonicalization.eq("whitened")].absolute_gap.mean()
        summary["whitening"] = {"raw_absolute_gap": float(raw), "whitened_absolute_gap": float(white), "gap_reduction": float(1 - white / raw) if raw else math.nan}

    regularizer = all_results[all_results.experiment.eq("invariant_regularizer")].copy()
    if len(regularizer):
        sensitivity = regularizer.groupby(["dataset", "model", "seed", "regularizer"]).utility.agg(["std", lambda s: s.max() - s.min()]).reset_index()
        sensitivity.columns = ["dataset", "model", "seed", "regularizer", "std", "spread"]
        bar_plot(sensitivity, "regularizer", "spread", "basis_sensitivity_standard_vs_invariant_regularizer", "Max–min utility spread")
        summary["regularizer"] = sensitivity.groupby("regularizer")[["std", "spread"]].mean().to_dict(orient="index")

    ordinal = all_results[all_results.experiment.eq("ordinal_basis")].copy()
    if len(ordinal):
        bar_plot(ordinal, "representation", "ordinal_block_condition_mean", "ordinal_local_vs_cumulative_spectrum", "Mean ordinal-block condition number", "dataset")
        ordinal["utility_delta"] = ordinal.groupby(["dataset", "model", "seed"])["utility"].transform(lambda s: s - s.loc[ordinal.loc[s.index, "representation"].eq("local")].iloc[0])
        bar_plot(ordinal, "representation", "utility_delta", "ordinal_basis_metric_and_convergence", "Utility change from local basis", "dataset")
        gap = ordinal[ordinal.representation.isin(["cumulative", "whitened", "local"])].copy()
        gap["utility_delta"] = gap.groupby(["dataset", "model", "seed"])["utility"].transform(lambda s: s - s.loc[gap.loc[s.index, "representation"].eq("local")].iloc[0])
        bar_plot(gap, "representation", "utility_delta", "ordinal_gap_before_after_whitening", "Utility change from local basis", "dataset")
        summary["ordinal_natural"] = {f"{dataset}/{representation}": float(value) for (dataset, representation), value in ordinal.groupby(["dataset", "representation"]).utility.mean().items()}
    ordinal_kappa = all_results[all_results.experiment.eq("ordinal_kappa")].copy()
    if len(ordinal_kappa):
        ordinal_kappa["display_delta"] = ordinal_kappa.groupby(["dataset", "model", "seed"])["utility"].transform(lambda s: s - s.iloc[np.argmin(ordinal_kappa.loc[s.index, "target_kappa"].to_numpy())])
        line_plot(ordinal_kappa, "target_kappa", "display_delta", "dataset", "ordinal_kappa_vs_metric", "Target condition number κ", "Utility change from κ=1", True)

    cyclic = all_results[all_results.experiment.eq("cyclic_geometry")].copy()
    if len(cyclic):
        phases = cyclic[cyclic.representation.str.startswith("full_fourier_phase")]
        bar_plot(phases, "phase", "test_metric", "cyclic_full_fourier_phase_control", "RMSE (lower is better)")
        ck = cyclic[cyclic.representation.str.startswith("cyclic_kappa")]
        line_plot(ck, "target_kappa", "test_metric", "model", "cyclic_kappa_vs_metric", "Target condition number κ", "RMSE (lower is better)", True)
        comparison = cyclic[cyclic.representation.isin(["centered_onehot", "full_fourier_phase_0", "truncated_first_harmonic"])]
        bar_plot(comparison, "representation", "test_metric", "full_vs_truncated_fourier", "RMSE (lower is better)")
        summary["cyclic"] = comparison.groupby("representation").utility.mean().to_dict()

    block = all_results[all_results.experiment.eq("block_residualization")].copy()
    if len(block):
        bar_plot(block, "representation", "condition_number", "joint_condition_before_after_block_residualization", "Joint condition number", "dataset")
        diamond = block[block.dataset.eq("diamond")]
        raw = diamond[diamond.representation.eq("raw_joint")][["seed", "test_metric"]].rename(columns={"test_metric": "raw_rmse"})
        diamond = diamond.merge(raw, on="seed")
        diamond["relative_rmse_improvement_pct"] = 100 * (diamond.raw_rmse - diamond.test_metric) / diamond.raw_rmse
        bar_plot(diamond, "representation", "relative_rmse_improvement_pct", "diamonds_variants", "Relative RMSE improvement vs raw joint (%)")
        block["utility_delta"] = block.groupby(["dataset", "model", "seed"])["utility"].transform(lambda s: s - s.loc[block.loc[s.index, "representation"].eq("raw_joint")].iloc[0])
        summary["block"] = {f"{dataset}/{representation}": float(value) for (dataset, representation), value in block.groupby(["dataset", "representation"]).utility_delta.mean().items()}

    residual_te = all_results[all_results.experiment.eq("residual_te")].copy()
    if len(residual_te):
        residual_te["utility_delta"] = residual_te.groupby(["dataset", "model", "seed"])["utility"].transform(lambda s: s - s.loc[residual_te.loc[s.index, "representation"].eq("plain_contrast")].iloc[0])
        bar_plot(residual_te, "representation", "utility_delta", "residual_target_encoding", "Utility change from plain contrast", "dataset")
        summary["residual_te"] = {f"{dataset}/{representation}": float(value) for (dataset, representation), value in residual_te.groupby(["dataset", "representation"]).utility_delta.mean().items()}

    frequency = all_results[all_results.experiment.eq("frequency_preconditioning")].copy()
    if len(frequency):
        frequency["utility_delta"] = frequency.groupby(["dataset", "model", "seed"])["utility"].transform(lambda s: s - s.loc[frequency.loc[s.index, "gamma"].eq(0.0)].iloc[0])
        bar_plot(frequency, "gamma", "utility_delta", "frequency_preconditioning", "Utility change from γ=0", "dataset")
        summary["frequency_preconditioning"] = {f"{dataset}/{gamma}": float(value) for (dataset, gamma), value in frequency.groupby(["dataset", "gamma"]).utility_delta.mean().items()}

    # A transparent diagnostic plot substitutes for optimizer-row statistics when
    # the preconditioning branch is not run: update opportunity equals frequency.
    frequency_stats_path = RESULTS / "frequency_update_statistics.csv"
    if frequency_stats_path.exists():
        category_updates = pd.read_csv(frequency_stats_path)
        category_updates = category_updates[category_updates.gamma.eq(0.5)]
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.scatter(category_updates.probability, category_updates.lr_or_activation_multiplier, alpha=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("Training category frequency")
        ax.set_ylabel("Frequency-only multiplier (γ=0.5)")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIGURES / "frequency_vs_embedding_update_statistics.png", dpi=180)
        plt.close(fig)

    controlled = all_results[all_results.experiment.isin(["numeric_kappa", "categorical_kappa", "ordinal_kappa", "cyclic_geometry"]) & all_results.target_kappa.notna()].copy()
    if len(controlled):
        controlled["display_delta"] = controlled.groupby(["experiment", "dataset", "model", "seed"])["utility"].transform(lambda s: s - s.iloc[np.argmin(controlled.loc[s.index, "target_kappa"].to_numpy())])
        line_plot(controlled, "target_kappa", "display_delta", "experiment", "summary_geometry_vs_performance", "Target condition number κ", "Utility change from κ=1", True)

    (RESULTS / "analysis_summary.json").write_text(json.dumps(summary, indent=2, default=lambda x: list(x) if isinstance(x, tuple) else str(x)))
    pd.DataFrame(inventory).to_csv(RESULTS / "experiment_inventory.csv", index=False)
    print(json.dumps({"total_runs": len(all_results), "figures": len(list(FIGURES.glob("*.png"))), "summary": str(RESULTS / "analysis_summary.json")}, indent=2))


if __name__ == "__main__":
    main()
