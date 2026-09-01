"""Audit and aggregate the permanently post-hoc aggregate-risk correction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest


HERE = Path(__file__).resolve().parent
REAL = HERE / "raw" / "posthoc_aggregate_risk"
SYNTHETIC = HERE / "raw" / "posthoc_aggregate_risk_synthetic"
FIGURES = HERE / "figures"
FIGURES.mkdir(exist_ok=True)
REAL_METHODS = (
    "distance_model",
    "aggregate_full",
    "aggregate_mismatch",
    "aggregate_reliability",
    "direct_proxy",
)
SYNTHETIC_METHODS = (
    "distance_model",
    "aggregate_exact",
    "aggregate_estimated",
    "direct_estimated_proxy",
)


def flatten_real() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    paths = sorted(REAL.glob("*.json"))
    if len(paths) != 216:
        raise AssertionError(f"expected 216 real cells, found {len(paths)}")
    for path in paths:
        payload = json.loads(path.read_text())
        methods = payload.get("methods", [])
        if (
            payload.get("status") != "complete"
            or payload.get("posthoc") is not True
            or tuple(item["method"] for item in methods) != REAL_METHODS
        ):
            raise AssertionError(f"invalid payload: {path}")
        fixed = {
            key: payload[key]
            for key in ("dataset", "task", "split_seed", "model", "model_seed")
        }
        rows.extend({**fixed, **item} for item in methods)
    frame = pd.DataFrame(rows)
    keys = ["dataset", "split_seed", "model", "model_seed", "method"]
    if len(frame) != 1080 or frame.duplicated(keys).any():
        raise AssertionError("real result keys are duplicated or incomplete")
    cell_keys = ["dataset", "split_seed", "model", "model_seed"]
    model_base = frame[frame.method == "distance_model"][cell_keys + ["score", "metric"]].rename(
        columns={"score": "model_score", "metric": "model_metric"}
    )
    proxy_base = frame[frame.method == "direct_proxy"][cell_keys + ["score", "metric"]].rename(
        columns={"score": "proxy_score", "metric": "proxy_metric"}
    )
    frame = frame.merge(model_base, on=cell_keys, validate="many_to_one")
    frame = frame.merge(proxy_base, on=cell_keys, validate="many_to_one")
    frame["score_gain_vs_model"] = frame.score - frame.model_score
    frame["score_gain_vs_proxy"] = frame.score - frame.proxy_score
    return frame


def flatten_synthetic() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    paths = sorted(SYNTHETIC.glob("*.json"))
    if len(paths) != 64:
        raise AssertionError(f"expected 64 synthetic cells, found {len(paths)}")
    for path in paths:
        payload = json.loads(path.read_text())
        methods = payload.get("methods", [])
        if (
            payload.get("status") != "complete"
            or payload.get("posthoc") is not True
            or tuple(item["method"] for item in methods) != SYNTHETIC_METHODS
        ):
            raise AssertionError(f"invalid payload: {path}")
        fixed = {key: payload[key] for key in ("task", "seed", "model")}
        rows.extend({**fixed, **item} for item in methods)
    frame = pd.DataFrame(rows)
    keys = ["task", "seed", "model", "method"]
    if len(frame) != 256 or frame.duplicated(keys).any():
        raise AssertionError("synthetic result keys are duplicated or incomplete")
    cell_keys = ["task", "seed", "model"]
    baseline = frame[frame.method == "distance_model"][cell_keys + ["score", "metric"]].rename(
        columns={"score": "model_score", "metric": "model_metric"}
    )
    frame = frame.merge(baseline, on=cell_keys, validate="many_to_one")
    frame["score_gain_vs_model"] = frame.score - frame.model_score
    frame["rmse_change_vs_model"] = frame.metric - frame.model_metric
    return frame


def summarize_real(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(["dataset", "task", "model", "method"], as_index=False).agg(
        cells=("score", "size"),
        score_mean=("score", "mean"),
        score_sd=("score", "std"),
        score_gain_vs_model_mean=("score_gain_vs_model", "mean"),
        score_gain_vs_proxy_mean=("score_gain_vs_proxy", "mean"),
        k_mean=("k", "mean"),
        k_16_fraction=("k", lambda x: float(np.mean(x == 16)) if x.notna().any() else np.nan),
        k_32_fraction=("k", lambda x: float(np.mean(x == 32)) if x.notna().any() else np.nan),
        k_64_fraction=("k", lambda x: float(np.mean(x == 64)) if x.notna().any() else np.nan),
    )


def summarize_synthetic(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(["task", "model", "method"], as_index=False).agg(
        seeds=("seed", "nunique"),
        rmse_mean=("metric", "mean"),
        rmse_change_mean=("rmse_change_vs_model", "mean"),
        rmse_change_sd=("rmse_change_vs_model", "std"),
        wins=("rmse_change_vs_model", lambda x: int(np.sum(x < -1e-12))),
        losses=("rmse_change_vs_model", lambda x: int(np.sum(x > 1e-12))),
        k_mean=("k", "mean"),
    )


def integrity_audit(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict[str, float | bool]:
    optimized_real = real[real.method.isin(["aggregate_full", "aggregate_mismatch"])]
    optimized_synth = synthetic[synthetic.method.isin(["aggregate_exact", "aggregate_estimated"])]
    optimized = pd.concat([optimized_real, optimized_synth], ignore_index=True)
    audit = {
        "objective_nonincrease_min": float(optimized.test_objective_nonincrease_fraction.min()),
        "max_simplex_error": float(optimized.test_max_simplex_error.max()),
        "minimum_weight": float(optimized.test_minimum_weight.min()),
    }
    independent = json.loads((HERE / "aggregation_solver_audit.json").read_text())
    audit["independent_solver_systems"] = int(independent["systems"])
    audit["independent_max_objective_gap"] = float(independent["max_objective_gap_vs_slsqp"])
    audit["independent_solver_passed"] = bool(independent["passed"])
    audit["passed"] = bool(
        audit["objective_nonincrease_min"] >= 0.999
        and audit["max_simplex_error"] <= 5e-6
        and audit["minimum_weight"] >= -1e-7
        and audit["independent_solver_passed"]
    )
    return audit


def bootstrap_mean_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(50_000, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def model_gate(summary: pd.DataFrame, model: str) -> dict[str, float | int | bool | list[float]]:
    full = summary[(summary.model == model) & (summary.method == "aggregate_full")]
    wins_model = int(np.sum(full.score_gain_vs_model_mean > 1e-9))
    losses_model = int(np.sum(full.score_gain_vs_model_mean < -1e-9))
    wins_proxy = int(np.sum(full.score_gain_vs_proxy_mean > 1e-9))
    losses_proxy = int(np.sum(full.score_gain_vs_proxy_mean < -1e-9))
    gain = float(full.score_gain_vs_model_mean.mean())
    ci = bootstrap_mean_ci(
        full.score_gain_vs_model_mean.to_numpy(),
        20261101 if model == "TabR" else 20261102,
    )
    return {
        "wins_vs_model": wins_model,
        "losses_vs_model": losses_model,
        "ties_vs_model": int(np.sum(np.abs(full.score_gain_vs_model_mean) <= 1e-9)),
        "dataset_balanced_score_gain": gain,
        "dataset_bootstrap_gain_ci95": list(ci),
        "sign_test_p_vs_model": float(binomtest(wins_model, wins_model + losses_model).pvalue),
        "wins_vs_direct_proxy": wins_proxy,
        "losses_vs_direct_proxy": losses_proxy,
        "sign_test_p_vs_direct_proxy": float(binomtest(wins_proxy, wins_proxy + losses_proxy).pvalue),
        "passes_real_gate": bool(wins_model >= 8 and gain > 0 and wins_proxy >= 8),
    }


def synthetic_s3_gate(summary: pd.DataFrame) -> dict[str, Any]:
    s3 = summary[summary.task == "S3_noise"].set_index(["model", "method"])
    rows: dict[str, Any] = {}
    passed = True
    for model in ("TabR", "ModernNCA"):
        rows[model] = {}
        for method in ("aggregate_exact", "aggregate_estimated"):
            row = s3.loc[(model, method)]
            clear = bool(row.rmse_change_mean < 0 and row.wins >= 6)
            rows[model][method] = {
                "rmse_change_mean": float(row.rmse_change_mean),
                "wins": int(row.wins),
                "clear_gain": clear,
            }
            passed = passed and clear
    return {"models": rows, "passes_clear_gain_operationalization": passed}


def save_figure(summary: pd.DataFrame) -> None:
    datasets = list(summary.dataset.drop_duplicates())
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
    colors = {"TabR": "#315da8", "ModernNCA": "#d97432"}
    x = np.arange(len(datasets))
    width = 0.36
    for offset, model in enumerate(("TabR", "ModernNCA")):
        group = summary[(summary.model == model) & (summary.method == "aggregate_full")].set_index("dataset").reindex(datasets)
        dx = (offset - 0.5) * width
        axes[0].bar(x + dx, group.score_gain_vs_model_mean, width, color=colors[model], alpha=.82, label=model)
        axes[1].bar(x + dx, group.score_gain_vs_proxy_mean, width, color=colors[model], alpha=.82, label=model)
    for ax in axes:
        ax.axhline(0, color="black", linewidth=.8)
        ax.legend(frameon=False)
    axes[0].set_ylabel("score gain vs neural model")
    axes[1].set_ylabel("score gain vs direct proxy")
    axes[1].set_xticks(x, datasets, rotation=35, ha="right")
    fig.suptitle("Post-hoc full aggregate-risk weighting")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_10_posthoc_aggregate_risk.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIGURES / "figure_10_posthoc_aggregate_risk.pdf", bbox_inches="tight")
    plt.close(fig)


def fmt(value: float, digits: int = 5) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def write_report(
    real: pd.DataFrame,
    real_summary: pd.DataFrame,
    synthetic_summary: pd.DataFrame,
    integrity: dict[str, float | bool],
    gates: dict[str, dict[str, float | int | bool]],
    s3_gate: dict[str, Any],
) -> str:
    passes = bool(integrity["passed"] and all(g["passes_real_gate"] for g in gates.values()) and s3_gate["passes_clear_gain_operationalization"])
    decision = "posthoc_promising_requires_new_prospective_replication" if passes else "stop_retrieval_risk_method_direction"
    task_breakdown = (
        real_summary[real_summary.method == "aggregate_full"]
        .groupby(["model", "task"])
        .score_gain_vs_model_mean.mean()
    )
    full_cells = real[real.method == "aggregate_full"]
    lines = [
        "# POST-HOC RESULT — AGGREGATION-AWARE RETRIEVAL RISK",
        "",
        "## Verdict",
        "",
        ("**POST-HOC PROMISING, BUT NOT A PASSED PROSPECTIVE RESULT.**" if passes else "**STOP the retrieval-risk method direction.**"),
        "This corrective experiment was designed after the frozen candidate-wise screen failed;",
        "its status cannot be upgraded retroactively. The frozen stop rule is applied below.",
        "",
        "## Integrity",
        "",
        f"- Real panel: 216 cells and {len(real)} method rows across 12 datasets.",
        "- Synthetic panel: 64 cells and 256 method rows across four mechanisms.",
        f"- Minimum objective-nonincreasing fraction: {integrity['objective_nonincrease_min']:.6f}.",
        f"- Maximum simplex error: {integrity['max_simplex_error']:.3e}; minimum weight: {integrity['minimum_weight']:.3e}.",
        f"- Independent SLSQP audit: {integrity['independent_solver_systems']} systems; maximum objective gap {integrity['independent_max_objective_gap']:.3e}.",
        "- Shortlist size was selected on validation only; no test target entered the QP or selection.",
        "",
        "## Frozen real-data gates",
        "",
        "| Model | gain vs model [dataset bootstrap 95% CI] | W/L/T vs model | W/L vs direct proxy | real gate |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in ("TabR", "ModernNCA"):
        gate = gates[model]
        ci = gate["dataset_bootstrap_gain_ci95"]
        lines.append(
            f"| {model} | {fmt(gate['dataset_balanced_score_gain'])} [{fmt(ci[0])}, {fmt(ci[1])}] | "
            f"{gate['wins_vs_model']}/{gate['losses_vs_model']}/{gate['ties_vs_model']} | "
            f"{gate['wins_vs_direct_proxy']}/{gate['losses_vs_direct_proxy']} | "
            f"{'PASS' if gate['passes_real_gate'] else 'FAIL'} |"
        )
    lines.extend([
        "",
        "The gain is positive in both task families: TabR classification/regression",
        f"`{task_breakdown.loc[('TabR', 'classification')]:+.5f}` / `{task_breakdown.loc[('TabR', 'regression')]:+.5f}`, and ModernNCA",
        f"`{task_breakdown.loc[('ModernNCA', 'classification')]:+.5f}` / `{task_breakdown.loc[('ModernNCA', 'regression')]:+.5f}`.",
        f"Validation selected the maximum tested shortlist `k=64` in {int(np.sum((full_cells.model == 'TabR') & (full_cells.k == 64)))}/108 TabR and",
        f"{int(np.sum((full_cells.model == 'ModernNCA') & (full_cells.k == 64)))}/108 ModernNCA cells. This boundary preference leaves a wider-neighborhood",
        "alternative unresolved; the frozen protocol forbids expanding the grid after outcomes.",
        "",
        "## Per-dataset full-QP result",
        "",
        "| Dataset | TabR vs model | TabR vs proxy | ModernNCA vs model | ModernNCA vs proxy |",
        "|---|---:|---:|---:|---:|",
    ])
    for dataset in real_summary.dataset.drop_duplicates():
        tabr = real_summary[(real_summary.dataset == dataset) & (real_summary.model == "TabR") & (real_summary.method == "aggregate_full")].iloc[0]
        nca = real_summary[(real_summary.dataset == dataset) & (real_summary.model == "ModernNCA") & (real_summary.method == "aggregate_full")].iloc[0]
        lines.append(
            f"| {dataset} | {fmt(tabr.score_gain_vs_model_mean)} | {fmt(tabr.score_gain_vs_proxy_mean)} | "
            f"{fmt(nca.score_gain_vs_model_mean)} | {fmt(nca.score_gain_vs_proxy_mean)} |"
        )
    lines.extend([
        "",
        "## Ablations",
        "",
        "Dataset-balanced score gain relative to the original neural model:",
        "",
        "| Model | full | mismatch-only | reliability-only |",
        "|---|---:|---:|---:|",
    ])
    for model in ("TabR", "ModernNCA"):
        values = {}
        for method in ("aggregate_full", "aggregate_mismatch", "aggregate_reliability"):
            x = real_summary[(real_summary.model == model) & (real_summary.method == method)]
            values[method] = float(x.score_gain_vs_model_mean.mean())
        lines.append(
            f"| {model} | {fmt(values['aggregate_full'])} | {fmt(values['aggregate_mismatch'])} | {fmt(values['aggregate_reliability'])} |"
        )
    lines.extend([
        "",
        "## Synthetic S3 gate",
        "",
        "A clear gain is operationalized before aggregation as negative mean RMSE change with",
        "at least 6/8 seed wins. This makes the protocol's qualitative word `clear` auditable.",
        "",
        "| Model | estimator | RMSE change | wins/8 | clear |",
        "|---|---|---:|---:|---:|",
    ])
    s3 = synthetic_summary[synthetic_summary.task == "S3_noise"].set_index(["model", "method"])
    for model in ("TabR", "ModernNCA"):
        for method in ("aggregate_exact", "aggregate_estimated"):
            row = s3.loc[(model, method)]
            clear = s3_gate["models"][model][method]["clear_gain"]
            lines.append(f"| {model} | {method} | {fmt(row.rmse_change_mean)} | {int(row.wins)}/8 | {'YES' if clear else 'NO'} |")
    lines.extend([
        "",
        "## All synthetic mechanisms",
        "",
        "| Task | Model | exact RMSE change (wins) | estimated RMSE change (wins) |",
        "|---|---|---:|---:|",
    ])
    for task in synthetic_summary.task.drop_duplicates():
        for model in ("TabR", "ModernNCA"):
            exact = synthetic_summary[(synthetic_summary.task == task) & (synthetic_summary.model == model) & (synthetic_summary.method == "aggregate_exact")].iloc[0]
            estimated = synthetic_summary[(synthetic_summary.task == task) & (synthetic_summary.model == model) & (synthetic_summary.method == "aggregate_estimated")].iloc[0]
            lines.append(
                f"| {task} | {model} | {fmt(exact.rmse_change_mean)} ({int(exact.wins)}/8) | "
                f"{fmt(estimated.rmse_change_mean)} ({int(estimated.wins)}/8) |"
            )
    lines.extend([
        "",
        "## Decision",
        "",
        f"Machine-readable decision: `{decision}`.",
        "",
        "Even a positive post-hoc result would require a newly frozen prospective replication.",
        "Per protocol, this Day-8 run launches no larger benchmark.",
    ])
    (HERE / "POSTHOC_AGGREGATION_RESULTS.md").write_text("\n".join(lines) + "\n")
    return decision


def main() -> None:
    real = flatten_real()
    synthetic = flatten_synthetic()
    real_summary = summarize_real(real)
    synthetic_summary = summarize_synthetic(synthetic)
    integrity = integrity_audit(real, synthetic)
    gates = {model: model_gate(real_summary, model) for model in ("TabR", "ModernNCA")}
    s3_gate = synthetic_s3_gate(synthetic_summary)
    save_figure(real_summary)
    decision = write_report(real, real_summary, synthetic_summary, integrity, gates, s3_gate)
    real.to_csv(HERE / "table_posthoc_aggregation_cells.csv", index=False)
    real_summary.to_csv(HERE / "table_posthoc_aggregation_summary.csv", index=False)
    synthetic.to_csv(HERE / "table_posthoc_aggregation_synthetic_cells.csv", index=False)
    synthetic_summary.to_csv(HERE / "table_posthoc_aggregation_synthetic_summary.csv", index=False)
    audit = {
        "status": "complete",
        "posthoc": True,
        "decision": decision,
        "protocol_sha256": "d40145656845b0bcc7ca03215db6e88f8ecceb682d3dc7f8c1a53ae9108fa27f",
        "real_cells": 216,
        "real_method_rows": len(real),
        "synthetic_cells": 64,
        "synthetic_method_rows": len(synthetic),
        "integrity": integrity,
        "real_gates": gates,
        "synthetic_s3_gate": s3_gate,
    }
    (HERE / "posthoc_aggregation_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
