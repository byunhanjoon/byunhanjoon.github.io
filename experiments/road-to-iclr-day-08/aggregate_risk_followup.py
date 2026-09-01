"""Audit and aggregate the frozen candidate-reliability follow-up."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
REAL = RAW / "prospective_risk"
SYNTHETIC = RAW / "prospective_risk_synthetic"
FIGURES = HERE / "figures"
FIGURES.mkdir(exist_ok=True)


def flatten_real() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, fits = [], []
    paths = sorted(REAL.glob("*.json"))
    if len(paths) != 216:
        raise AssertionError(f"expected 216 real cells, found {len(paths)}")
    for path in paths:
        payload = json.loads(path.read_text())
        if payload.get("status") != "complete" or len(payload.get("methods", [])) != 3:
            raise AssertionError(f"incomplete payload: {path}")
        fixed = {
            key: payload[key]
            for key in (
                "dataset", "task", "split_seed", "model", "model_seed", "n_train",
                "n_validation", "n_test", "n_num", "n_cat_onehot", "uncertainty_mean",
                "uncertainty_sd", "uncertainty_iqr",
            )
        }
        for method in payload["methods"]:
            row = {**fixed, **method}
            row["validation_grid"] = json.dumps(row["validation_grid"], sort_keys=True)
            rows.append(row)
        fits.append({**{key: fixed[key] for key in ("dataset", "task", "split_seed", "model", "model_seed")}, **payload["fit"]})
    frame = pd.DataFrame(rows)
    fit_frame = pd.DataFrame(fits)
    keys = ["dataset", "split_seed", "model", "model_seed", "method"]
    if frame.duplicated(keys).any() or len(frame) != 648:
        raise AssertionError("real result keys are duplicated or incomplete")
    baseline = frame[frame.method == "distance"][
        ["dataset", "split_seed", "model", "model_seed", "score", "topk_proxy_risk", "risk_spearman"]
    ].rename(columns={"score": "baseline_score", "topk_proxy_risk": "baseline_proxy_risk", "risk_spearman": "baseline_risk_spearman"})
    frame = frame.merge(baseline, on=["dataset", "split_seed", "model", "model_seed"], validate="many_to_one")
    frame["score_gain"] = frame.score - frame.baseline_score
    frame["proxy_risk_change"] = frame.topk_proxy_risk - frame.baseline_proxy_risk
    frame["risk_spearman_gain"] = frame.risk_spearman - frame.baseline_risk_spearman
    return frame, fit_frame


def flatten_synthetic() -> pd.DataFrame:
    rows = []
    paths = sorted(SYNTHETIC.glob("*.json"))
    if len(paths) != 64:
        raise AssertionError(f"expected 64 synthetic cells, found {len(paths)}")
    for path in paths:
        payload = json.loads(path.read_text())
        if payload.get("status") != "complete" or len(payload.get("methods", [])) != 4:
            raise AssertionError(f"incomplete payload: {path}")
        for method in payload["methods"]:
            row = {
                "task": payload["task"],
                "model": payload["model"],
                "seed": payload["seed"],
                "estimated_exact_uncertainty_spearman": payload["estimated_exact_uncertainty_spearman"],
                **method,
            }
            row["validation_grid"] = json.dumps(row["validation_grid"], sort_keys=True)
            rows.append(row)
    frame = pd.DataFrame(rows)
    if len(frame) != 256 or frame.duplicated(["task", "model", "seed", "method"]).any():
        raise AssertionError("synthetic result keys are duplicated or incomplete")
    baseline = frame[frame.method == "distance"][["task", "model", "seed", "metric", "topk_proxy_risk"]].rename(
        columns={"metric": "baseline_metric", "topk_proxy_risk": "baseline_proxy_risk"}
    )
    frame = frame.merge(baseline, on=["task", "model", "seed"], validate="many_to_one")
    frame["rmse_change"] = frame.metric - frame.baseline_metric
    frame["proxy_risk_change"] = frame.topk_proxy_risk - frame.baseline_proxy_risk
    return frame


def real_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(["dataset", "task", "model", "method"], as_index=False).agg(
        cells=("score", "size"),
        score_mean=("score", "mean"),
        score_sd=("score", "std"),
        score_gain_mean=("score_gain", "mean"),
        score_gain_sd=("score_gain", "std"),
        proxy_risk_mean=("topk_proxy_risk", "mean"),
        proxy_risk_change_mean=("proxy_risk_change", "mean"),
        risk_spearman_mean=("risk_spearman", "mean"),
        risk_spearman_gain_mean=("risk_spearman_gain", "mean"),
        lambda_mean=("lambda", "mean"),
        lambda_nonzero_fraction=("lambda", lambda x: float(np.mean(x > 0))),
        uncertainty_iqr=("uncertainty_iqr", "mean"),
    )


def synthetic_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(["task", "model", "method"], as_index=False).agg(
        seeds=("seed", "nunique"),
        rmse_mean=("metric", "mean"),
        rmse_change_mean=("rmse_change", "mean"),
        rmse_change_sd=("rmse_change", "std"),
        prediction_wins=("rmse_change", lambda x: int(np.sum(x < 0))),
        proxy_risk_mean=("topk_proxy_risk", "mean"),
        proxy_risk_change_mean=("proxy_risk_change", "mean"),
        lambda_mean=("lambda", "mean"),
        lambda_nonzero=("lambda", lambda x: int(np.sum(x > 0))),
        uncertainty_spearman=("estimated_exact_uncertainty_spearman", "mean"),
    )


def fmt(value: float, digits: int = 5) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def model_gate(summary: pd.DataFrame, model: str, method: str) -> dict[str, float | int]:
    x = summary[(summary.model == model) & (summary.method == method)]
    return {
        "wins": int(np.sum(x.score_gain_mean > 1e-9)),
        "losses": int(np.sum(x.score_gain_mean < -1e-9)),
        "ties": int(np.sum(np.abs(x.score_gain_mean) <= 1e-9)),
        "dataset_balanced_gain": float(x.score_gain_mean.mean()),
        "risk_improvements": int(np.sum(x.proxy_risk_change_mean < 0)),
        "dataset_balanced_risk_change": float(x.proxy_risk_change_mean.mean()),
        "nonzero_lambda_fraction": float(x.lambda_nonzero_fraction.mean()),
    }


def save_figures(summary: pd.DataFrame, synthetic: pd.DataFrame) -> None:
    methods = ["oof_reliability", "permuted_reliability"]
    colors = {"oof_reliability": "#315da8", "permuted_reliability": "#d97432"}
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
    datasets = list(summary.dataset.drop_duplicates())
    x = np.arange(len(datasets))
    width = 0.19
    for model_offset, model in enumerate(("TabR", "ModernNCA")):
        for method_offset, method in enumerate(methods):
            values = summary[(summary.model == model) & (summary.method == method)].set_index("dataset").reindex(datasets)
            offset = ((model_offset * 2 + method_offset) - 1.5) * width
            label = f"{model} / {method.replace('_reliability', '')}"
            hatch = "" if model == "TabR" else "//"
            axes[0].bar(x + offset, values.score_gain_mean, width, color=colors[method], alpha=.78, hatch=hatch, label=label)
            axes[1].bar(x + offset, -values.proxy_risk_change_mean, width, color=colors[method], alpha=.78, hatch=hatch)
    axes[0].axhline(0, color="black", linewidth=.8)
    axes[1].axhline(0, color="black", linewidth=.8)
    axes[0].set_ylabel("prediction score gain")
    axes[1].set_ylabel("proxy-risk reduction")
    axes[1].set_xticks(x, datasets, rotation=35, ha="right")
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    fig.suptitle("Candidate reliability lowers proxy risk without consistent prediction gains")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_8_prospective_reliability.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIGURES / "figure_8_prospective_reliability.pdf", bbox_inches="tight")
    plt.close(fig)

    s3 = synthetic[(synthetic.task == "S3_noise") & synthetic.method.isin([
        "distance", "exact_reliability", "estimated_reliability", "permuted_exact_reliability"
    ])].copy()
    order = ["distance", "exact_reliability", "estimated_reliability", "permuted_exact_reliability"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for model, marker in (("TabR", "o"), ("ModernNCA", "s")):
        group = s3[s3.model == model].set_index("method").reindex(order)
        axes[0].plot(order, group.rmse_mean, marker=marker, label=model)
        axes[1].plot(order, group.proxy_risk_mean, marker=marker, label=model)
    axes[0].set_ylabel("test RMSE")
    axes[1].set_ylabel("exact top-16 one-neighbor risk")
    for ax in axes:
        ax.tick_params(axis="x", rotation=28)
        ax.legend(frameon=False)
    fig.suptitle("S3: large diagnostic movement, negligible predictive movement")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_9_s3_reliability_gap.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIGURES / "figure_9_s3_reliability_gap.pdf", bbox_inches="tight")
    plt.close(fig)


def write_report(
    frame: pd.DataFrame,
    fit_frame: pd.DataFrame,
    summary: pd.DataFrame,
    synth_summary: pd.DataFrame,
) -> dict[str, Any]:
    gates = {
        model: {
            method: model_gate(summary, model, method)
            for method in ("oof_reliability", "permuted_reliability")
        }
        for model in ("TabR", "ModernNCA")
    }
    associations = {}
    for model in ("TabR", "ModernNCA"):
        x = summary[(summary.model == model) & (summary.method == "oof_reliability")]
        associations[model] = {
            "risk_reduction_vs_score_gain_spearman": float(
                spearmanr(-x.proxy_risk_change_mean, x.score_gain_mean).statistic
            ),
            "uncertainty_iqr_vs_score_gain_spearman": float(
                spearmanr(x.uncertainty_iqr, x.score_gain_mean).statistic
            ),
        }
    s3 = synth_summary[synth_summary.task == "S3_noise"].set_index(["model", "method"])
    fit_stats = fit_frame.groupby("model").agg(
        cells=("epochs", "size"),
        mean_epochs=("epochs", "mean"),
        median_epochs=("epochs", "median"),
        max_epochs=("epochs", "max"),
        summed_fit_seconds=("fit_wall_seconds", "sum"),
    )

    lines = [
        "# PROSPECTIVE RESULT — COMPATIBILITY × CANDIDATE RELIABILITY",
        "",
        "## Verdict",
        "",
        "**KILL the candidate-wise reliability reranker as the next ICLR method.**  The intervention",
        "reliably changes the diagnostic in the intended direction, but that change does not",
        "reliably improve prediction and is not stronger than the permutation control.",
        "The Retrieval Risk Law remains exact; the failed step was treating the mean of",
        "one-neighbor risks as a sufficient diagnostic for a multi-neighbor aggregate.",
        "",
        "## Frozen panel integrity",
        "",
        f"- Real: {frame.dataset.nunique()} datasets, {frame[['dataset','split_seed','model','model_seed']].drop_duplicates().shape[0]} trained cells, {len(frame)} method rows.",
        f"- Synthetic: 4 tasks, 8 fresh seeds, 2 models, {len(synth_summary)} aggregate rows.",
        "- Every real cell has distance, true OOF-reliability, and permuted-reliability results.",
        "- All lambda choices used validation loss only; no test label entered retrieval scoring.",
        "",
        "## Real primary gates",
        "",
        "| Model | Method | score gain | W/L/T | proxy-risk change | risk improves | nonzero lambda |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in ("TabR", "ModernNCA"):
        for method in ("oof_reliability", "permuted_reliability"):
            gate = gates[model][method]
            lines.append(
                f"| {model} | {method} | {fmt(gate['dataset_balanced_gain'])} | "
                f"{gate['wins']}/{gate['losses']}/{gate['ties']} | "
                f"{fmt(gate['dataset_balanced_risk_change'])} | {gate['risk_improvements']}/12 | "
                f"{gate['nonzero_lambda_fraction']:.1%} |"
            )
    lines.extend([
        "",
        "True OOF reliability lowered proxy risk on 11/12 TabR and 9/12 ModernNCA",
        "datasets, yet the prediction gates were only 7/12 and 4/12.  Its dataset-balanced",
        "score gain was smaller than the permutation control for both models.  Risk reduction",
        f"versus score gain had Spearman rho {associations['TabR']['risk_reduction_vs_score_gain_spearman']:.3f} for TabR and "
        f"{associations['ModernNCA']['risk_reduction_vs_score_gain_spearman']:.3f} for ModernNCA.",
        "",
        "## Per-dataset OOF-reliability result",
        "",
        "| Dataset | TabR score gain | TabR risk change | ModernNCA score gain | ModernNCA risk change |",
        "|---|---:|---:|---:|---:|",
    ])
    for dataset in summary.dataset.drop_duplicates():
        t = summary[(summary.dataset == dataset) & (summary.model == "TabR") & (summary.method == "oof_reliability")].iloc[0]
        n = summary[(summary.dataset == dataset) & (summary.model == "ModernNCA") & (summary.method == "oof_reliability")].iloc[0]
        lines.append(
            f"| {dataset} | {fmt(t.score_gain_mean)} | {fmt(t.proxy_risk_change_mean)} | "
            f"{fmt(n.score_gain_mean)} | {fmt(n.proxy_risk_change_mean)} |"
        )
    lines.extend([
        "",
        "## Synthetic S3 gate",
        "",
        "| Model | Method | RMSE change | wins/8 | exact proxy-risk change | mean lambda |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for model in ("TabR", "ModernNCA"):
        for method in ("exact_reliability", "estimated_reliability", "permuted_exact_reliability"):
            row = s3.loc[(model, method)]
            lines.append(
                f"| {model} | {method} | {fmt(row.rmse_change_mean)} | {int(row.prediction_wins)}/8 | "
                f"{fmt(row.proxy_risk_change_mean)} | {fmt(row.lambda_mean, 3)} |"
            )
    lines.extend([
        "",
        "On S3, exact candidate variance sharply reduced the mean top-16 one-neighbor risk",
        "but changed neural test RMSE by less than 0.001 on average.  Estimated variance did",
        "not recover a clean ModernNCA gain, and TabR's permutation control was comparable.",
        "The frozen synthetic gate therefore fails.",
        "",
        "## Why the original diagnostic can fail",
        "",
        "For normalized aggregation weights `w`, signed conditional-mean discrepancies `d`,",
        "and candidate variances `sigma2`, the exact aggregate risk is",
        "",
        "```text",
        "R_aggregate = (sum_i w_i d_i)^2 + sum_i w_i^2 sigma2_i.",
        "```",
        "",
        "The weighted mean of one-neighbor risks used by the diagnostic is",
        "",
        "```text",
        "R_one_mean = sum_i w_i (d_i^2 + sigma2_i).",
        "```",
        "",
        "Their exact nonnegative gap is",
        "",
        "```text",
        "R_one_mean - R_aggregate",
        "  = Var_w(d) + sum_i w_i(1-w_i)sigma2_i >= 0.",
        "```",
        "",
        "Thus a neighborhood can look much better under average candidate risk while losing",
        "useful signed-bias cancellation or receiving little benefit because averaging already",
        "dilutes candidate noise by squared weights.  This explains the observed diagnostic–",
        "prediction decoupling and invalidates top-k mean one-neighbor risk as a standalone",
        "mechanism certificate for TabR/ModernNCA.",
        "",
        "## Epoch and compute diagnostic",
        "",
        "| Model | cells | mean epochs | median | max | summed fit seconds |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for model, row in fit_stats.iterrows():
        lines.append(
            f"| {model} | {int(row.cells)} | {row.mean_epochs:.2f} | {row.median_epochs:.1f} | "
            f"{row.max_epochs:.0f} | {row.summed_fit_seconds:.1f} |"
        )
    lines.extend([
        "",
        "Most compact models early-stopped well before 48 epochs, though some reached the cap.",
        "This does not establish large-scale convergence behavior; it does show that the failed",
        "reranking mechanism is not contingent on one extremely short training run.",
        "",
        "## Novelty/readiness decision",
        "",
        "- Arithmetic helices are occupied by 2025 work with causal interventions, and 2026",
        "  work already analyzes carry fibers, layer transitions, and convergence-dependent",
        "  sharpening. A reproduction/atlas alone is not a credible ICLR novelty claim.",
        "- Generic uncertainty-aware or reliability-weighted neighbor retrieval is crowded.",
        "- The exact aggregation gap above is useful and empirically exposed here, but it is",
        "  bias-variance algebra rather than a sufficiently new theorem by itself.",
        "- No new Day-8 embedding/retrieval method currently has both defensible novelty and",
        "  strong prospective results. Status: **INTERESTING NEGATIVE MECHANISM, NOT ICLR-READY.**",
        "",
        "## Post-hoc corrective outcome",
        "",
        "The one allowed post-hoc experiment optimized the *aggregate* plug-in risk directly",
        "over a frozen shortlist. It passed both real-data subgates, but mismatch-only weighting",
        "matched the full estimator and ModernNCA failed the frozen S3 transfer gate. The joint",
        "stop rule is therefore met; see `POSTHOC_AGGREGATION_RESULTS.md`.",
    ])
    (HERE / "PROSPECTIVE_RISK_RESULTS.md").write_text("\n".join(lines) + "\n")
    audit = {
        "status": "complete",
        "decision": "kill_candidate_wise_reliability_reranker",
        "real_cells": int(frame[["dataset", "split_seed", "model", "model_seed"]].drop_duplicates().shape[0]),
        "real_method_rows": int(len(frame)),
        "synthetic_cells": 64,
        "synthetic_method_rows": 256,
        "gates": gates,
        "associations": associations,
        "protocol_sha256": "965dd814c74f01ff66cfb38a5e92bff7e3eff69369ba93b57f3bc1433e418ca5",
    }
    (HERE / "prospective_risk_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
    return audit


def main() -> None:
    frame, fit_frame = flatten_real()
    synthetic = flatten_synthetic()
    summary = real_summary(frame)
    synth_summary = synthetic_summary(synthetic)
    frame.to_csv(HERE / "table_prospective_risk_cells.csv", index=False)
    summary.to_csv(HERE / "table_prospective_risk_summary.csv", index=False)
    synthetic.to_csv(HERE / "table_synthetic_risk_followup_cells.csv", index=False)
    synth_summary.to_csv(HERE / "table_synthetic_risk_followup.csv", index=False)
    fit_frame.to_csv(HERE / "table_prospective_fit_dynamics.csv", index=False)
    save_figures(summary, synth_summary)
    audit = write_report(frame, fit_frame, summary, synth_summary)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
