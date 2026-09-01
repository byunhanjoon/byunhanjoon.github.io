#!/usr/bin/env python3
"""Regenerate paper tables, figures, macros, and the final result manifest."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import CACHE, CONFIG, HERE, atomic_json


PAPER = HERE / "paper" / "iclr2027"
GENERATED = PAPER / "generated"
FIGURES = PAPER / "figures"
RESULTS = CACHE / "results"

LABELS = {
    "projtabicl": r"\textsc{ProjTabICL}",
    "tabiclv2_diagonal": r"\textsc{TabICL-Diag}",
    "tabpfn3_diagonal": r"TabPFN-3 (Diag)",
    "tabpfn25_diagonal": r"TabPFN-2.5 (Diag)",
    "gp_matern32": r"GP Mat\'ern-$3/2$",
    "gp_rbf": "GP RBF",
    "catboost_process": "Bootstrap CatBoost",
    "bayesian_linear": "Bayesian linear",
    "tabiclv2_projtabicl_marginal": "TabICLv2",
    "tabpfn3": "TabPFN-3",
    "tabpfn25": "TabPFN-2.5",
    "tabdpt_turbo_1_2": "TabDPT-Turbo 1.2",
    "projtabicl_shuffled": "Row-shuffled head",
    "hidden_cosine": "Untrained hidden cosine",
    "raw_feature_rbf": "Raw-feature RBF lift",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def tex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def fmt(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "--"
    if value != 0 and (abs(value) >= 10_000 or abs(value) < 10 ** (-(digits + 1))):
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def command(name: str, value: str) -> str:
    return rf"\newcommand{{\{name}}}{{{value}}}"


def metadata(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(data["metadata"].item()))


def dataset_inventory() -> pd.DataFrame:
    root = CACHE / "tabicl_singleton_episodes" / "eval"
    rows = []
    seen = set()
    for path in sorted(root.glob("*.npz")):
        meta = metadata(path)
        name = str(meta["dataset"])
        if name in seen:
            continue
        seen.add(name)
        rows.append(
            {
                "dataset": name,
                "rows": int(meta["train_pool_size"]) + int(meta["test_pool_size"]),
                "features": int(meta["n_features"]),
                "source_id": str(meta["source_id"]),
            }
        )
    frame = pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)
    if len(frame) != len(CONFIG["evaluation_tasks"]):
        raise RuntimeError(f"dataset inventory incomplete: {len(frame)}")
    return frame


def write_dataset_table(frame: pd.DataFrame) -> None:
    split = int(math.ceil(len(frame) / 2))
    left = frame.iloc[:split].reset_index(drop=True)
    right = frame.iloc[split:].reset_index(drop=True)
    lines = [
        r"\begin{tabular}{lrr@{\qquad}lrr}",
        r"\toprule Dataset & Rows & $p$ & Dataset & Rows & $p$ \\",
        r"\midrule",
    ]
    for index in range(split):
        a = left.iloc[index]
        if index < len(right):
            b = right.iloc[index]
            right_fields = f"{tex_escape(b.dataset)} & {int(b.rows):,} & {int(b.features)}"
        else:
            right_fields = " & & "
        lines.append(
            f"{tex_escape(a.dataset)} & {int(a.rows):,} & {int(a.features)} & {right_fields} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GENERATED / "dataset_table.tex").write_text("\n".join(lines) + "\n")


def write_main_table(frame: pd.DataFrame) -> None:
    order = [
        "projtabicl",
        "tabiclv2_diagonal",
        "tabpfn3_diagonal",
        "tabpfn25_diagonal",
        "gp_matern32",
        "gp_rbf",
        "catboost_process",
        "bayesian_linear",
    ]
    frame = frame.set_index("method").reindex(order).dropna(how="all").reset_index()
    metrics = ["nll", "crps", "squared_error", "coverage_90", "nll_rank", "crps_rank"]
    best = {metric: float(frame[metric].min()) for metric in metrics if metric != "coverage_90"}
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule Method & NLL & CRPS & MSE & Cov.90 & NLL rank & CRPS rank \\",
        r"\midrule",
    ]
    for row in frame.itertuples():
        values = []
        for metric, digits in zip(metrics, (3, 4, 4, 3, 2, 2)):
            value = float(getattr(row, metric))
            rendered = fmt(value, digits)
            if metric != "coverage_90" and np.isclose(value, best[metric], rtol=0, atol=5e-12):
                rendered = rf"\textbf{{{rendered}}}"
            values.append(rendered)
        label = LABELS.get(row.method, tex_escape(row.method))
        lines.append(f"{label} & " + " & ".join(values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GENERATED / "main_table.tex").write_text("\n".join(lines) + "\n")


def write_point_table(frame: pd.DataFrame) -> None:
    order = [
        "tabiclv2_projtabicl_marginal",
        "tabpfn3",
        "tabpfn25",
        "tabdpt_turbo_1_2",
        "gp_matern32",
        "gp_rbf",
        "catboost_process",
        "bayesian_linear",
    ]
    frame = frame.set_index("method").reindex(order).dropna(how="all").reset_index()
    best_rmse = float(frame["nrmse"].min())
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule Method & nRMSE & NLL & CRPS & Cov.90 \\",
        r"\midrule",
    ]
    for row in frame.itertuples():
        rmse = fmt(float(row.nrmse), 4)
        if np.isclose(float(row.nrmse), best_rmse, atol=5e-12):
            rmse = rf"\textbf{{{rmse}}}"
        label = LABELS.get(row.method, tex_escape(row.method))
        lines.append(
            f"{label} & {rmse} & {fmt(float(row.nll), 3)} & "
            f"{fmt(float(row.crps), 4)} & {fmt(float(row.coverage_90), 3)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GENERATED / "point_table.tex").write_text("\n".join(lines) + "\n")


def write_controls_table(frame: pd.DataFrame) -> None:
    order = [
        "projtabicl",
        "tabiclv2_diagonal",
        "projtabicl_shuffled",
        "hidden_cosine",
        "raw_feature_rbf",
    ]
    frame = frame.set_index("method").reindex(order).dropna(how="all").reset_index()
    diagonal = float(frame.loc[frame["method"] == "tabiclv2_diagonal", "nll"].iloc[0])
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule Method & NLL & $\Delta$NLL vs. diag. & CRPS & Cov.90 \\",
        r"\midrule",
    ]
    for row in frame.itertuples():
        effect = diagonal - float(row.nll)
        label = LABELS.get(row.method, tex_escape(row.method))
        lines.append(
            f"{label} & {fmt(float(row.nll), 3)} & {fmt(effect, 4)} & "
            f"{fmt(float(row.crps), 4)} & {fmt(float(row.coverage_90), 3)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GENERATED / "controls_table.tex").write_text("\n".join(lines) + "\n")


def write_timing_table(frame: pd.DataFrame) -> None:
    order = [
        "tabiclv2_backbone",
        "tabpfn3",
        "tabpfn25",
        "tabdpt_turbo_1_2",
        "gp_matern32",
        "gp_rbf",
        "catboost_process",
        "bayesian_linear",
    ]
    frame = frame.set_index("method").reindex(order).dropna(how="all").reset_index()
    labels = {**LABELS, "tabiclv2_backbone": "TabICLv2 singleton"}
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule Method & Median (s) & Mean (s) & 90th pct. (s) \\",
        r"\midrule",
    ]
    for row in frame.itertuples():
        lines.append(
            f"{labels.get(row.method, tex_escape(row.method))} & "
            f"{fmt(float(row.median_seconds), 3)} & {fmt(float(row.mean_seconds), 3)} & "
            f"{fmt(float(row.p90_seconds), 3)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GENERATED / "timing_table.tex").write_text("\n".join(lines) + "\n")


def write_effects_table(effects: pd.DataFrame) -> None:
    effects = effects.sort_values("nll_diagonal_minus_projtabicl", ascending=False)
    lines = [
        r"\begin{longtable}{lrr}",
        r"\caption{All 35 fixed-marginal covariance effects.}\label{tab:all-effects}\\",
        r"\toprule Dataset & NLL effect & CRPS effect \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule Dataset & NLL effect & CRPS effect \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in effects.itertuples():
        lines.append(
            f"{tex_escape(row.dataset)} & {fmt(float(row.nll_diagonal_minus_projtabicl), 5)} & "
            f"{fmt(float(row.crps_diagonal_minus_projtabicl), 5)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{longtable}"]
    (GENERATED / "effects_table.tex").write_text("\n".join(lines) + "\n")


def write_gates_table(projective: dict[str, Any], audit: dict[str, Any]) -> None:
    nll = projective["nll_advantage"]
    crps = projective["crps_advantage"]
    maximum = max(map(float, audit["maxima"].values()))
    gate_rows = [
        ("Mean NLL advantage", fmt(float(nll["mean"]), 4), "$>0$", float(nll["mean"]) > 0),
        ("NLL dataset wins", f"{nll['wins']}/35", "$\\geq21/35$", int(nll["wins"]) >= 21),
        ("Mean CRPS advantage", fmt(float(crps["mean"]), 5), "$>0$", float(crps["mean"]) > 0),
        ("Randomization $p$", fmt(float(nll["paired_randomization_p"]), 3), "$<0.05$", float(nll["paired_randomization_p"]) < 0.05),
        ("Mean/diagonal identity", fmt(float(projective["integrity"]["max_diagonal_abs"]), 1), "$\\leq10^{-10}$", float(projective["integrity"]["max_diagonal_abs"]) <= 1e-10),
        ("Restriction/permutation", f"{maximum:.2e}", "$\\leq10^{-5}$", maximum <= 1e-5),
    ]
    lines = [
        r"\begin{tabular}{lrrc}",
        r"\toprule Criterion & Estimate & Threshold & Pass \\",
        r"\midrule",
    ]
    for name, estimate, threshold, passed in gate_rows:
        lines.append(f"{name} & {estimate} & {threshold} & {'yes' if passed else 'no'} \\\\ ")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GENERATED / "gates_table.tex").write_text("\n".join(lines) + "\n")


def write_audit_table(batched: dict[str, Any], singleton: dict[str, Any]) -> None:
    def max_for(payload: dict[str, Any], prefix: str, kind: str) -> float:
        return max(
            float(value)
            for key, value in payload["maxima"].items()
            if key.startswith(prefix) and kind in key
        )

    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule Query mode & Mean & Variance & Hidden & Covariance \\",
        r"\midrule",
    ]
    for label, payload in (("Batched", batched), ("Singleton", singleton)):
        values = [
            max(max_for(payload, "restriction", kind), max_for(payload, "permutation", kind))
            for kind in ("mean", "variance", "hidden", "covariance")
        ]
        lines.append(label + " & " + " & ".join(f"{value:.2e}" for value in values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GENERATED / "audit_table.tex").write_text("\n".join(lines) + "\n")


def write_application_table(frame: pd.DataFrame) -> None:
    primary = frame[frame["method"].isin(["projtabicl", "tabiclv2_diagonal"])]
    wide = primary.pivot(index="dataset", columns="method", values=["nll", "crps"])
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule & \multicolumn{3}{c}{NLL} & \multicolumn{3}{c}{CRPS} \\",
        r"Dataset & Proj. & Diag. & Effect & Proj. & Diag. & Effect \\",
        r"\midrule",
    ]
    for dataset in wide.index:
        pn = float(wide.loc[dataset, ("nll", "projtabicl")])
        dn = float(wide.loc[dataset, ("nll", "tabiclv2_diagonal")])
        pc = float(wide.loc[dataset, ("crps", "projtabicl")])
        dc = float(wide.loc[dataset, ("crps", "tabiclv2_diagonal")])
        lines.append(
            f"{tex_escape(dataset)} & {fmt(pn, 3)} & {fmt(dn, 3)} & {fmt(dn-pn, 4)} & "
            f"{fmt(pc, 4)} & {fmt(dc, 4)} & {fmt(dc-pc, 5)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GENERATED / "application_table.tex").write_text("\n".join(lines) + "\n")


def plot_dataset_effects(effects: pd.DataFrame) -> None:
    effects = effects.sort_values("nll_diagonal_minus_projtabicl").reset_index(drop=True)
    values = effects["nll_diagonal_minus_projtabicl"].to_numpy(float)
    names = effects["dataset"].map(lambda x: str(x).replace("_", " ")).to_list()
    colors = np.where(values >= 0, "#2b8cbe", "#d95f0e")
    fig, (ax, zoom) = plt.subplots(1, 2, figsize=(10.5, 7.2), gridspec_kw={"width_ratios": [1.25, 1]})
    y = np.arange(len(values))
    ax.barh(y, values, color=colors, height=0.72)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y, names, fontsize=6.5)
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("NLL effect (diag - projective), symlog scale")
    ax.set_title("All effects (tail-preserving scale)", fontsize=10)

    central = effects.iloc[:-1] if abs(values[-1]) == max(abs(values)) else effects.iloc[1:]
    cvalues = central["nll_diagonal_minus_projtabicl"].to_numpy(float)
    cy = np.arange(len(central))
    zoom.barh(cy, cvalues, color=np.where(cvalues >= 0, "#2b8cbe", "#d95f0e"), height=0.72)
    zoom.axvline(0, color="black", linewidth=0.8)
    zoom.set_yticks([])
    lo, hi = float(min(cvalues.min(), 0.0)), float(max(cvalues.max(), 0.0))
    pad = max((hi - lo) * 0.06, 0.002)
    zoom.set_xlim(lo - pad, hi + pad)
    zoom.set_xlabel("Effects excluding Solar Flare (linear scale)")
    zoom.set_title("Solar Flare removed", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "dataset_effects.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "dataset_effects.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_family_context(cells: pd.DataFrame) -> None:
    primary = cells[
        cells["family"].isin(CONFIG["primary_aggregate_families"])
        & cells["method"].isin(["projtabicl", "tabiclv2_diagonal"])
    ]
    within = primary.groupby(
        ["dataset", "context_size", "family", "method"], as_index=False
    )["nll"].mean()
    wide = within.pivot(
        index=["dataset", "context_size", "family"], columns="method", values="nll"
    ).reset_index()
    wide["effect"] = wide["tabiclv2_diagonal"] - wide["projtabicl"]
    # The arithmetic mean is already shown in the dataset-effect figure and is
    # dominated by Solar Flare.  Median-over-dataset cells expose breadth.
    heat = wide.groupby(["context_size", "family"])["effect"].median().unstack("family")
    heat = heat.reindex(index=list(map(int, CONFIG["context_sizes"])), columns=CONFIG["primary_aggregate_families"])
    values = heat.to_numpy(float)
    limit = max(float(np.max(np.abs(values))), 1e-5)
    fig, ax = plt.subplots(figsize=(9.2, 2.8))
    image = ax.imshow(values, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    short = ["mean", "total", "pair diff.", "contrast", "signed", "positive"]
    ax.set_xticks(np.arange(len(short)), short)
    ax.set_yticks(np.arange(len(heat.index)), [str(x) for x in heat.index])
    ax.set_ylabel("Context size")
    ax.set_xlabel("Aggregate family")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i,j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(values[i,j]) > 0.6 * limit else "black")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("NLL effect")
    fig.tight_layout()
    fig.savefig(FIGURES / "family_context_heatmap.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "family_context_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    projective = load_json(RESULTS / "projective_singleton" / "summary.json")
    baseline = load_json(RESULTS / "baselines_singleton" / "summary.json")
    applications = load_json(RESULTS / "applications_singleton" / "summary.json")
    audit_singleton = load_json(RESULTS / "projectivity_audit_singleton" / "summary.json")
    audit_batched = load_json(RESULTS / "projectivity_audit_batched" / "summary.json")
    main_table = pd.read_csv(RESULTS / "baselines_singleton" / "aggregate_main_methods.csv")
    point_table = pd.read_csv(RESULTS / "baselines_singleton" / "point_by_method.csv")
    effects = pd.read_csv(RESULTS / "baselines_singleton" / "projective_effects_by_dataset.csv")
    all_cells = pd.read_parquet(RESULTS / "baselines_singleton" / "all_cells.parquet")
    application_frame = pd.read_csv(RESULTS / "applications_singleton" / "aggregate_by_dataset.csv")

    write_dataset_table(dataset_inventory())
    write_main_table(main_table)
    write_point_table(point_table)
    write_controls_table(pd.read_csv(RESULTS / "baselines_singleton" / "aggregate_by_method_all.csv"))
    write_timing_table(pd.read_csv(RESULTS / "baselines_singleton" / "timing_by_method.csv"))
    write_effects_table(effects)
    write_gates_table(projective, audit_singleton)
    write_audit_table(audit_batched, audit_singleton)
    write_application_table(application_frame)
    plot_dataset_effects(effects)
    plot_family_context(all_cells)

    effect = effects.set_index("dataset")["nll_diagonal_minus_projtabicl"]
    largest = str(effect.abs().idxmax())
    trimmed = np.sort(effect.to_numpy(float))
    trim_count = int(math.floor(0.1 * len(trimmed)))
    trimmed_mean = float(trimmed[trim_count:-trim_count].mean())
    singleton_elapsed, batched_elapsed = [], []
    for single_path in sorted((CACHE / "tabicl_singleton_episodes" / "eval").glob("*.npz")):
        batch_path = CACHE / "tabicl_episodes" / "eval" / single_path.name
        singleton_elapsed.append(float(metadata(single_path)["elapsed_seconds"]))
        batched_elapsed.append(float(metadata(batch_path)["elapsed_seconds"]))
    slowdown = float(np.median(np.asarray(singleton_elapsed) / np.asarray(batched_elapsed)))
    audit_max = max(map(float, audit_singleton["maxima"].values()))
    nll = projective["nll_advantage"]
    crps = projective["crps_advantage"]
    macro_lines = [
        "% Automatically generated from frozen result artifacts.",
        command("NLLAdv", fmt(float(nll["mean"]), 4)),
        command("NLLCILow", fmt(float(nll["bootstrap_95"][0]), 4)),
        command("NLLCIHigh", fmt(float(nll["bootstrap_95"][1]), 4)),
        command("CRPSAdv", fmt(float(crps["mean"]), 5)),
        command("CRPSCILow", fmt(float(crps["bootstrap_95"][0]), 5)),
        command("CRPSCIHigh", fmt(float(crps["bootstrap_95"][1]), 5)),
        command("NLLWins", f"{nll['wins']}/35"),
        command("NLLPValue", fmt(float(nll["paired_randomization_p"]), 3)),
        command("SingletonAuditMax", f"{audit_max:.2e}"),
        command("BatchedAuditMax", fmt(max(map(float, audit_batched["maxima"].values())), 4)),
        command("OutlierRemovedAdv", fmt(float(effect.drop(index=largest).mean()), 4)),
        command("MedianAdv", fmt(float(effect.median()), 5)),
        command("TrimmedAdv", fmt(trimmed_mean, 5)),
        command("SingletonSlowdown", fmt(slowdown, 1)),
    ]
    (GENERATED / "results_macros.tex").write_text("\n".join(macro_lines) + "\n")

    files = [
        GENERATED / name
        for name in (
            "results_macros.tex",
            "gates_table.tex",
            "main_table.tex",
            "point_table.tex",
            "controls_table.tex",
            "timing_table.tex",
            "dataset_table.tex",
            "effects_table.tex",
            "application_table.tex",
            "audit_table.tex",
        )
    ] + [FIGURES / "dataset_effects.pdf", FIGURES / "family_context_heatmap.pdf"]
    manifest = {
        "primary_conclusion": "predeclared covariance performance claim rejected",
        "projective_summary": projective,
        "baseline_summary_path": str(RESULTS / "baselines_singleton" / "summary.json"),
        "application_summary": applications,
        "singleton_audit": audit_singleton,
        "batched_audit": audit_batched,
        "largest_absolute_effect_dataset": largest,
        "nll_mean_without_largest": float(effect.drop(index=largest).mean()),
        "singleton_median_runtime_slowdown": slowdown,
        "generated_files": [
            {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in files
        ],
    }
    atomic_json(RESULTS / "paper_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
