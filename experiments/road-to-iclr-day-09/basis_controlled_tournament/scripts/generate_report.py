#!/usr/bin/env python3
"""Generate the exact 21-section tournament report from processed artifacts."""

from __future__ import annotations

from importlib import metadata as package_metadata
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "results" / "processed"


def table(frame: pd.DataFrame, columns: list[str] | None = None, digits: int = 4) -> str:
    value = frame.copy() if columns is None else frame[columns].copy()
    for column in value.select_dtypes(include=["float"]).columns:
        value[column] = value[column].map(lambda item: f"{item:.{digits}g}" if np.isfinite(item) else "NA")
    def cell(item: object) -> str:
        if pd.isna(item):
            return "NA"
        return str(item).replace("|", "\\|").replace("\n", " ")

    headers = [cell(column) for column in value.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend(
        "| " + " | ".join(cell(item) for item in row) + " |"
        for row in value.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def pct(value: float) -> str:
    return f"{100 * float(value):.2f}%"


def package_versions() -> str:
    names = ["torch", "numpy", "scipy", "pandas", "scikit-learn", "catboost", "pytabkit", "tabicl", "tabpfn"]
    records = []
    for name in names:
        try:
            records.append(f"{name} {package_metadata.version(name)}")
        except package_metadata.PackageNotFoundError:
            records.append(f"{name} unavailable")
    return ", ".join(records)


def method_summary(frame: pd.DataFrame, track: str) -> pd.Series:
    part = frame[(frame["track"] == track) & (frame["method"] != "Raw")]
    return part.sort_values(
        ["paper_method_score", "median_disagreement_reduction"], ascending=[False, False]
    ).iloc[0]


def aggregate_optimizer(units: pd.DataFrame) -> pd.DataFrame:
    wanted = units[(units["split"] == "test") & units["track"].str.startswith("optimizer")]
    return (
        wanted.groupby(["method", "dataset"], as_index=False)
        .agg(
            disagreement=("disagreement", "median"),
            reduction=("disagreement_reduction", "median"),
            task_metric=("task_error", "median"),
            task_change=("relative_task_change", "median"),
        )
    )


def aggregate_representations(units: pd.DataFrame) -> pd.DataFrame:
    named = {
        "PCA", "GramAnchor", "GramAnchor-m8", "GramDistance", "NystromGram",
        "HybridSpectral-t0.01", "HybridSpectral-t0.05", "HybridSpectral-t0.10",
    }
    wanted = units[
        (units["split"] == "test") & (units["track"] == "representation") & units["method"].isin(named)
    ]
    return (
        wanted.groupby(["method", "model", "dataset"], as_index=False)
        .agg(
            disagreement=("disagreement", "median"),
            reduction=("disagreement_reduction", "median"),
            task_change=("relative_task_change", "median"),
        )
    )


def prospective_detail(units: pd.DataFrame) -> pd.DataFrame:
    wanted = units[(units["split"] == "test") & (units["method"] != "Raw")]
    return (
        wanted.groupby(["dataset", "model", "method"], as_index=False)
        .agg(
            raw_disagreement=("raw_disagreement", "median"),
            method_disagreement=("disagreement", "median"),
            reduction=("disagreement_reduction", "median"),
            raw_task=("raw_task_error", "median"),
            method_task=("task_error", "median"),
            relative_task_change=("relative_task_change", "median"),
        )
    )


def main() -> None:
    protocol = json.loads((ROOT / "configs" / "TOURNAMENT_PROTOCOL.json").read_text())
    panel = json.loads((ROOT / "configs" / "NEW_PROSPECTIVE_PANEL.json").read_text())
    finalists = json.loads((ROOT / "configs" / "FINALIST_CONFIGS.json").read_text())
    finalist_sha = (ROOT / "configs" / "FINALIST_CONFIGS.sha256").read_text().split()[0]
    stage1 = pd.read_csv(PROCESSED / "stage1_summary.csv")
    dev_units = pd.read_csv(PROCESSED / "development_all_units.csv")
    dev_summary = pd.read_csv(PROCESSED / "development_all_method_summary.csv")
    hpo_summary = pd.read_csv(PROCESSED / "equal_hpo_summary.csv")
    hpo_lr = pd.read_csv(PROCESSED / "equal_hpo_lr_selection.csv")
    mechanism = pd.read_csv(PROCESSED / "mechanism_audit_summary.csv")
    mechanism_verdict = pd.read_csv(PROCESSED / "mechanism_equivariance_verdict.csv")
    anchors = pd.read_csv(PROCESSED / "anchor_ablation_summary.csv")
    natural = pd.read_csv(PROCESSED / "natural_basis_summary.csv")
    prospective_units = pd.read_csv(PROCESSED / "prospective_units.csv")
    prospective_summary = pd.read_csv(PROCESSED / "prospective_method_summary.csv")
    ranking_a = pd.read_csv(PROCESSED / "prospective_ranking_A.csv")
    ranking_b = pd.read_csv(PROCESSED / "prospective_ranking_B_pareto.csv")
    ranking_c = pd.read_csv(PROCESSED / "prospective_ranking_C_predictive.csv")
    ranking_d = pd.read_csv(PROCESSED / "prospective_ranking_D_score.csv")
    model_matrix = pd.read_csv(PROCESSED / "prospective_method_model_matrix_wide.csv")
    condition = pd.read_csv(PROCESSED / "condition_exploratory_summary.csv")

    strongest_optimizer = method_summary(prospective_summary, "optimizer")
    strongest_representation = method_summary(prospective_summary, "interface")
    strongest_hybrid = method_summary(prospective_summary, "hybrid_prediction_mixture")
    keep_methods = prospective_summary[
        (prospective_summary["method"] != "Raw") & (prospective_summary["category"] == "KEEP")
    ]["method"].tolist()
    prospective_table = prospective_detail(prospective_units)

    report: list[str] = []
    add = report.append
    add("# Basis-Controlled Tabular Learning — Method Tournament")
    add("")
    add("## Executive Summary")
    add("")
    add(
        f"No final paper method is chosen here. The strongest optimizer candidate is **{strongest_optimizer.method}** "
        f"({pct(strongest_optimizer.median_disagreement_reduction)} prospective reduction, "
        f"{pct(strongest_optimizer.median_relative_task_change)} median task change). The strongest representation "
        f"candidate is **{strongest_representation.method}** ({pct(strongest_representation.median_disagreement_reduction)}, "
        f"{pct(strongest_representation.median_relative_task_change)}), and the strongest nontrivial hybrid is "
        f"**{strongest_hybrid.method}** ({pct(strongest_hybrid.median_disagreement_reduction)}, "
        f"{pct(strongest_hybrid.median_relative_task_change)}). "
        + (f"Methods satisfying KEEP: {', '.join(keep_methods)}." if keep_methods else "No method satisfies KEEP on the prospective panel.")
    )
    add("")
    add("The central result is a tradeoff, not a winner declaration: invariant Gram coordinates can remove arbitrary orthogonal-basis dependence exactly, while a fixed raw/invariant mixture can retain more of the useful raw-coordinate prior. The optimizer route is mechanistically valid under matched functions, but its predictive cost determines whether it remains a serious paper candidate.")
    add("")
    add("## Frozen Protocol")
    add("")
    add(f"- Git commit at freeze: `{protocol['repository_commit']}` (tournament files are an uncommitted experiment subtree on that base).")
    add(f"- Hardware: two NVIDIA H100 NVL GPUs, {protocol['hardware']['gpu_memory_mib_each']} MiB each; driver {protocol['hardware']['driver']}.")
    add(f"- Packages: {package_versions()}.")
    add(f"- Split seed: {protocol['split_seed']}; model seeds: {protocol['model_seeds']}; eight orbit members per reference.")
    add(f"- Development datasets: {', '.join(protocol['development_datasets'])}.")
    add(f"- NEW prospective datasets: {', '.join(item['key'] for item in panel['datasets'])}.")
    add(f"- `FINALIST_CONFIGS.json` SHA256: `{finalist_sha}`; frozen at {finalists['frozen_at_utc']} before prospective data loading.")
    add("")
    add("## Previous Findings Treated as Fixed")
    add("")
    add("The tournament treated the prior confirmation as pilot evidence rather than rerunning discovery: orthogonal and natural equivalent-basis effects were established; AdamW's coordinatewise second moment was implicated; SGD was predictively weak despite symmetry; PCA incurred task cost; generic consistency training failed; and AnchorCanonical suffered rank failure. All baselines, splits, RBF blocks, transformations, and metrics were imported from that frozen implementation.")
    add("")
    add("## 1. Stage-1 Method Screening")
    add("")
    stage1_view = stage1.rename(columns={"median_disagreement_reduction": "disagreement reduction", "median_relative_task_change": "task change", "runtime_seconds": "runtime"})
    add(table(stage1_view, ["method", "disagreement reduction", "task change", "runtime", "verdict"]))
    add("")
    add("Stage 1 killed the default-initialization optimizer variants for inadequate reduction, while retaining PCA and the three Gram-family interfaces. The specified data-equivariant initialization and SoftBlock rescues were therefore evaluated in Stage 2.")
    add("")
    add("## 2. Optimizer Methods")
    add("")
    add("### AdamW")
    add("")
    add("Raw AdamW is the zero-reduction reference and received the same three learning-rate trials as every surviving optimizer rescue.")
    add("")
    add("### BlockScalarAdam")
    add("")
    add("BlockScalarAdam is exactly orthogonally equivariant in the matched-function audit, but its Stage-1 predictive/reduction tradeoff did not pass the survival rule.")
    add("")
    add("### BlockAdam")
    add("")
    add("Per-output block second moments preserve matched equivalence numerically. Default initialization still leaves different initial functions across ordinary orbit fits, so data-equivariant initialization was tested explicitly.")
    add("")
    add("### MatrixAdam")
    add("")
    add("Full within-block matrix adaptivity also preserves matched equivalence. It improves the optimization symmetry but is more expensive and does not automatically preserve AdamW's task performance.")
    add("")
    add("### Data-equivariant initialization")
    add("")
    add("The first-layer blocks use a target-free training-design construction that transforms covariantly. For TabM-D, the diagonal per-coordinate input adapter is frozen to one because it is not closed under general within-block rotations; this makes the intervention honest but changes the effective architecture and is reported as a limitation.")
    add("")
    add("### SoftBlockAdam")
    add("")
    add("SoftBlockAdam with alpha 0.1 and 0.25 was the declared interpolation fallback. Alpha 0.1 entered equal HPO; neither setting was assumed invariant merely from its name.")
    add("")
    add(table(aggregate_optimizer(dev_units).rename(columns={"task_metric": "task metric", "task_change": "task change"})))
    add("")
    add("## 3. Optimizer Equivariance Audit")
    add("")
    checkpoints = mechanism[mechanism["epoch"].isin([0, 1, 5, mechanism["epoch"].max()])]
    audit_wide = checkpoints.pivot(index="method", columns="epoch", values="max_disagreement").reset_index()
    audit_wide.columns = ["method"] + [f"epoch{int(value)}" if int(value) != int(mechanism['epoch'].max()) else "final disagreement" for value in audit_wide.columns[1:]]
    add(table(audit_wide))
    add("")
    valid = mechanism_verdict[(mechanism_verdict["epoch"] == mechanism_verdict["epoch"].max()) & mechanism_verdict["preserves_matched_equivalence"]]["method"].tolist()
    invalid = mechanism_verdict[(mechanism_verdict["epoch"] == mechanism_verdict["epoch"].max()) & ~mechanism_verdict["preserves_matched_equivalence"]]["method"].tolist()
    add(f"Matched-function equivalence remained below 1e-5 through the final epoch for {', '.join(valid)}. It did not for {', '.join(invalid)}. Thus BlockAdam and MatrixAdam pass the implementation audit and AdamW fails it, as predicted by the mechanism hypothesis.")
    add("")
    add("## 4. Representation Methods")
    add("")
    add("### PCA")
    add("")
    add("PCA is exactly invariant in nondegenerate cases after deterministic orientation, but its development task cost remains material.")
    add("")
    add("### GramAnchor")
    add("")
    add("GramAnchor uses target-free Gram-pivot training anchors. The tolerance-aware pivot rule was necessary to remove near-rank-saturation tie instability; all final coordinate audits pass 1e-8.")
    add("")
    add("### GramDistance")
    add("")
    add("GramDistance is exactly orthogonally invariant but loses more task information than GramAnchor on the full panel.")
    add("")
    add("### NyströmGram")
    add("")
    add("NyströmGram uses deterministic canonicalization inside repeated eigenspaces. This repaired an initial numerical degeneracy without using labels or outcomes.")
    add("")
    add("### HybridSpectral")
    add("")
    add("HybridSpectral keeps separated spectral directions and Gram-maps degenerate groups. Three frozen gap thresholds were tested.")
    add("")
    add("### MahalanobisGram if run")
    add("")
    add("MahalanobisGram was run only in the separate condition<=3 exploratory screen at ridges 1e-6, 1e-4, and 1e-2. Ridge regularization prevents a blanket claim of exact general-linear invariance.")
    add("")
    add(table(aggregate_representations(dev_units)))
    add("")
    add("## 5. Anchor / Rank Ablations")
    add("")
    anchor_view = anchors.rename(columns={"anchors": "m", "selection": "selection method", "median_empirical_rank": "empirical rank", "median_disagreement_reduction": "disagreement", "median_relative_task_change": "task change"})
    add(table(anchor_view, ["m", "selection method", "normalize", "empirical rank", "min_anchor_rank", "disagreement", "task change"]))
    add("")
    add("Eight pivoted anchors were best on the controlled-MLP ablation, but their five-model full-panel task cost exceeded the 16-anchor setting. The frozen interface therefore retains m=16 rather than extrapolating the MLP-only ablation across model families.")
    add("")
    add("## 6. Natural Equivalent Basis Results")
    add("")
    add("### Local vs spectral hat")
    add("")
    local = natural[natural["pair"] == "local_vs_spectral_hat"]
    add(table(local))
    add("")
    add("### One-hot vs Helmert")
    add("")
    helmert = natural[natural["pair"] == "onehot_vs_helmert"]
    add(table(helmert) if len(helmert) else "No applicable categorical block was present in the evaluated natural-basis cells.")
    add("")
    add("Every natural pair was required to reconstruct below 1e-6 and every invariant-interface coordinate pair below 1e-8 before metrics were accepted.")
    add("")
    add("## 7. Hybrid Methods")
    add("")
    hybrid_rows = dev_summary[(dev_summary["track"] == "hybrid_prediction_mixture") & ~dev_summary["method"].str.endswith("@1")]
    add(table(hybrid_rows.sort_values("paper_method_score", ascending=False).head(16), ["method", "median_disagreement_reduction", "median_relative_task_change", "median_worst_orbit_gain", "paper_method_score", "wins", "ties", "losses"]))
    add("")
    add("H1 prediction mixtures were complete and alpha was selected from development validation only. H2 was optional ('if easy') and was not built because it would add architecture-specific training confounds after H1 already exposed the tradeoff. H3's stated trigger was not met: no block optimizer simultaneously had strong invariance and near-raw predictive performance on development, so the combined branch was not run.")
    add("")
    add("## 8. Equal-HPO Control")
    add("")
    add("Every surviving optimizer rescue and raw AdamW received exactly learning-rate multipliers 0.5, 1, and 2. A single multiplier was selected development-wide for each model/method from validation error; per-orbit oracle selections were retained only as diagnostics and were ineligible for freezing.")
    add("")
    add(table(hpo_lr[hpo_lr["selected"].astype(bool)]))
    add("")
    add(table(hpo_summary, ["method", "median_disagreement_reduction", "median_relative_task_change", "median_worst_orbit_gain", "paper_method_score", "wins", "ties", "losses"]))
    add("")
    add("## 9. Development Ranking")
    add("")
    add("### Performance-Preserving Invariance Ranking")
    add("")
    add(table(pd.read_csv(PROCESSED / "development_all_ranking_A.csv").head(15), ["method", "track", "median_disagreement_reduction", "median_relative_task_change", "median_worst_orbit_gain", "paper_method_score"]))
    add("")
    add("### Pareto Ranking")
    add("")
    add(table(pd.read_csv(PROCESSED / "development_all_ranking_B_pareto.csv"), ["method", "track", "median_disagreement_reduction", "median_relative_task_change"]))
    add("")
    add("### Predictive Performance Ranking")
    add("")
    add(table(pd.read_csv(PROCESSED / "development_all_ranking_C_predictive.csv").head(15), ["method", "track", "median_predictive_rank", "mean_predictive_rank", "median_relative_task_change"]))
    add("")
    add("### Paper-Method Score")
    add("")
    add(table(pd.read_csv(PROCESSED / "development_all_ranking_D_score.csv").head(15), ["method", "track", "median_disagreement_reduction", "median_relative_task_change", "failure_fraction", "paper_method_score"]))
    add("")
    add("## 10. Frozen Finalists")
    add("")
    frozen_rows = []
    for item in finalists["finalists"]:
        frozen_rows.append({"method": item["method_id"], "type": item["type"], "models": ", ".join(item["applicable_models"]), "config": json.dumps({key: item[key] for key in item if key in {"interface", "interface_parameters", "alpha", "per_model"}}, sort_keys=True)})
    add(table(pd.DataFrame(frozen_rows), digits=3))
    add("")
    add(f"Exactly three configurations were frozen under SHA `{finalist_sha}`. The prospective runner refuses to resolve prospective data before verifying this hash and finalist cap.")
    add("")
    add("## 11. NEW Prospective Results")
    add("")
    add("These seven datasets were untouched until `FINALIST_CONFIGS.json` and its SHA existed. No learning rate, anchor count, alpha, or model setting changed afterward.")
    add("")
    add(table(prospective_table))
    add("")
    add("## 12. Prospective Rankings")
    add("")
    add("### Ranking A — Performance-Preserving Invariance")
    add("")
    aview = ranking_a.copy()
    aview["win/tie/loss"] = aview["wins"].astype(str) + "/" + aview["ties"].astype(str) + "/" + aview["losses"].astype(str)
    add(table(aview, ["method", "median_disagreement_reduction", "median_relative_task_change", "median_worst_orbit_gain", "win/tie/loss"]))
    add("")
    add("### Ranking B — Pareto Frontier")
    add("")
    add(table(ranking_b, ["method", "track", "median_disagreement_reduction", "median_relative_task_change", "median_worst_orbit_gain"]))
    add("")
    add("### Ranking C — Predictive Performance")
    add("")
    add(table(ranking_c, ["method", "track", "median_predictive_rank", "mean_predictive_rank", "median_relative_task_change"]))
    add("")
    add("### Ranking D — Paper-Method Score")
    add("")
    add(table(ranking_d, ["method", "track", "median_disagreement_reduction", "median_relative_task_change", "failure_fraction", "paper_method_score"]))
    add("")
    add("## 13. Method-by-Model Matrix")
    add("")
    add(table(model_matrix))
    add("")
    add("Categories apply the frozen KEEP/PROMISING/NICHE/FAIL thresholds within each model family; the overall multi-family KEEP decision remains the authoritative one.")
    add("")
    add("## 14. Strongest Result")
    add("")
    strongest = ranking_a[ranking_a["method"] != "Raw"].iloc[0]
    add(f"The strongest primary-ranking result is **{strongest.method}**: {pct(strongest.median_disagreement_reduction)} median prospective disagreement reduction at {pct(strongest.median_relative_task_change)} median task change across {int(strongest.model_families)} model families, with {pct(strongest.median_worst_orbit_gain)} median worst-orbit gain. This is the most direct evidence that arbitrary-coordinate sensitivity can be reduced without paying the PCA-scale task penalty.")
    add("")
    add("## 15. Strongest Negative Result")
    add("")
    optimizer_by_model = (
        prospective_units[
            (prospective_units["split"] == "test")
            & (prospective_units["method"] == strongest_optimizer.method)
        ]
        .groupby("model")["relative_task_change"]
        .median()
    )
    worst_optimizer_model = str(optimizer_by_model.idxmax())
    worst_optimizer_cost = float(optimizer_by_model.max())
    development_optimizer_cost = float(
        dev_summary.loc[
            dev_summary["method"] == strongest_optimizer.method,
            "median_relative_task_change",
        ].iloc[0]
    )
    add(
        f"The strongest negative result is that mechanistic optimizer equivariance does not guarantee broad predictive parity. "
        f"Although {strongest_optimizer.method} has only {pct(strongest_optimizer.median_relative_task_change)} pooled prospective cost, "
        f"it cost {pct(worst_optimizer_cost)} on {worst_optimizer_model} and {pct(development_optimizer_cost)} on the development panel; "
        "only the controlled MLP clears the per-family KEEP gate. The TabM diagonal-adapter closure problem and the altered "
        "data-equivariant initialization make the optimizer route substantially less plug-and-play than the interface route."
    )
    add("")
    add("## 16. Failed Methods and Why")
    add("")
    add("- Default BlockScalarAdam, BlockAdam, and MatrixAdam failed Stage-1 reduction despite correct matched-function symmetry because independent default initializations start from different functions.")
    add("- GramDistance discarded too much predictive information; its exact invariance was not enough.")
    add("- PCA and most pure spectral interfaces crossed the 1% development task-cost gate.")
    add("- The m=8 GramAnchor rescue improved controlled-MLP cost but underperformed m=16 across all five families, so it was not frozen.")
    add("- H2 was optional and omitted; H3 was conditionally specified and its trigger failed. Neither is silently counted as a negative empirical result.")
    add("- MahalanobisGram is reported as exploratory; ridge/rank sensitivity prevents an orthogonal-result-style general-linear claim.")
    add("- The frozen TabPFN setting uses one estimator; on the 576-column satimage and 512-column gesture GramAnchor interfaces it emitted a feature-coverage warning (500-column maximum). The protocol was not changed after freezing, and this is a scaling limitation of that cell rather than a tuned-away exception.")
    add("")
    add("## 17. Mechanistic Interpretation")
    add("")
    add("- **Can blockwise adaptivity retain Adam's task performance?** It retains optimizer-state equivariance, but the prospective task comparison determines whether it retains enough predictive strength; the answer is not uniformly yes.")
    add("- **Does full matrix adaptivity help beyond scalar BlockAdam?** It can alter the reduction/cost point, but its extra matrix state did not dominate the simpler alternatives across both trainable architectures.")
    add("- **Does data-equivariant initialization matter after optimizer correction?** Yes for orbit fits: it closes the epoch-0 function gap that an equivariant update alone cannot repair. It also changes TabM's diagonal input adapter, which is a real cost.")
    add("- **Can invariant Gram interfaces preserve predictive information?** GramAnchor does so much better than PCA, GramDistance, or Nyström on the median, though per-unit losses show information/inductive-bias removal is not free.")
    add("- **Is raw-coordinate information genuinely useful?** Yes. Mixtures often improve task error relative to full invariance, demonstrating that some raw basis dependence acts as useful inductive bias.")
    add("- **Is a hybrid preferable to full invariance?** It is preferable when the primary objective values near-zero task cost over exact invariance; full GramAnchor remains preferable when exact orthogonal invariance is the requirement.")
    add("")
    add("Condition<=3 is separate from the orthogonal claim:")
    add("")
    add(table(condition, ["method", "median_disagreement_reduction", "median_relative_task_change", "failure_fraction", "paper_method_score"]))
    add("")
    add("## 18. Reviewer Attack Audit")
    add("")
    add('### "You merely replaced Adam with SGD."')
    add("")
    add("No. BlockScalar/Block/Matrix optimizers retain first-moment adaptivity and were directly compared with SGD in the matched-function audit; SGD remains a mechanistic control, not the proposed optimizer.")
    add("")
    add('### "The new optimizer loses Adam\'s performance."')
    add("")
    add("Often it does, and the report treats this as the optimizer track's main negative result. Equal HPO and prospective task costs are shown rather than hidden.")
    add("")
    add('### "The representation throws information away."')
    add("")
    add("The Gram maps can discard coordinate-specific marginal structure even when they preserve within-block geometry. Pure-interface losses and hybrid gains quantify that cost; no sufficiency claim is made.")
    add("")
    add('### "PCA already solves this."')
    add("")
    add("PCA is an invariance baseline, but its task cost and repeated-eigenvalue ambiguity are both worse than the leading Gram candidate on this tournament.")
    add("")
    add('### "The method only handles random rotations."')
    add("")
    add("Finalist interfaces were also tested on local/spectral hat and one-hot/Helmert pairs. Condition<=3 transforms are reported separately, with no false claim that ordinary Gram inner products are generally invariant.")
    add("")
    add('### "It only works for MLPs."')
    add("")
    add("The interface track spans controlled MLP, TabM-D, TabICLv2, TabPFN 2.6, and CatBoost. The optimizer track is intentionally restricted to architectures whose first layer is accessible.")
    add("")
    add('### "The method was tuned on the test datasets."')
    add("")
    add(f"The prospective panel was locked first; configurations were frozen under `{finalist_sha}` before data loading. All alpha/LR/anchor choices used development validation only, and the runner enforces the hash gate.")
    add("")
    add('### "Basis dependence may actually be beneficial."')
    add("")
    add("Agreed in part: hybrid task gains are evidence that raw-coordinate priors can help. The scientific target is harmful arbitrary dependence, not invariance maximization.")
    add("")
    add("## 19. Ranked Candidates for Human Decision")
    add("")
    natural_success = {
        method: bool((part["median_disagreement_reduction"] >= 0.99).all() and (part["max_reconstruction_error"] < 1e-6).all())
        for method, part in natural.groupby("method")
    }
    condition_lookup = condition.set_index("method")
    candidate_records = []
    serious = prospective_summary[
        prospective_summary["method"].isin([item["method_id"] for item in finalists["finalists"]])
    ].copy()
    serious = serious.sort_values(["performance_preserving_eligible", "median_disagreement_reduction", "paper_method_score"], ascending=[False, False, False])
    for rank, row in enumerate(serious.itertuples(), start=1):
        finalist = next(item for item in finalists["finalists"] if item["method_id"] == row.method)
        base_method = finalist.get("interface")
        natural_key = "GramAnchor" if base_method == "gram_anchor" else row.method
        condition_row = condition_lookup.loc[row.method] if row.method in condition_lookup.index else None
        candidate_records.append(
            {
                "rank": rank,
                "method": row.method,
                "type": finalist["type"],
                "prospective reduction": pct(row.median_disagreement_reduction),
                "task cost": pct(row.median_relative_task_change),
                "model breadth": int(row.model_families),
                "natural-basis success": "N/A (optimizer)" if finalist["type"] == "optimizer" else natural_success.get(natural_key, False),
                "condition<=3 behavior": "not invariant / exploratory" if condition_row is None else f"{pct(condition_row.median_disagreement_reduction)} reduction",
                "complexity": "high" if finalist["type"] == "optimizer" else "two fits" if finalist["type"].startswith("hybrid") else "one target-free frontend",
                "recommendation": row.category,
            }
        )
    add(table(pd.DataFrame(candidate_records)))
    add("")
    add("This is a ranking for human decision, not an automatic paper-method choice.")
    add("")
    add("## 20. Suggested Next Experiment for Each Top-3 Method")
    add("")
    for record in candidate_records[:3]:
        if record["type"] == "optimizer":
            suggestion = "Run a closure-preserving TabM input-adapter design that transforms as a full block matrix, then repeat equal-HPO without freezing the diagonal adapter."
        elif record["type"] == "interface":
            suggestion = "Scale the target-free anchor bank and train size on a larger locked benchmark to test whether task cost falls or rank saturation returns."
        else:
            suggestion = "Learn a validation-only per-dataset gate from target-free rank/spectrum descriptors and compare it with this fixed alpha on a second untouched panel."
        add(f"- **{record['method']}** — {suggestion}")
    add("")
    add("## 21. Files Produced")
    add("")
    add("- `configs/NEW_PROSPECTIVE_PANEL.json` and SHA; `configs/TOURNAMENT_PROTOCOL.json`; `configs/STAGE1_SURVIVORS.json` and SHA; `configs/FINALIST_CONFIGS.json` and SHA.")
    add("- `results/raw/`: immutable prediction bundles and metadata for Stage 1, Stage 2, equal HPO, natural bases, prospective evaluation, and condition<=3 exploration.")
    add("- `results/processed/`: cell tables, coordinate audits, four development rankings, four prospective rankings, method/model categories, ablations, mechanism trajectories, and integrity metadata.")
    add("- `figures/figure_1_...` through `figures/figure_8_...` in PNG and PDF.")
    add("- `tournament/`: shared representations, optimizers, model adapters, and protocol helpers; `scripts/`: runners, analyzers, freezer, figures, report, and audit; `tests/`: numerical and lock tests.")

    (ROOT / "results.md").write_text("\n".join(report) + "\n")
    print(f"wrote {ROOT / 'results.md'} ({len(report)} blocks)")


if __name__ == "__main__":
    main()
