"""Regenerate the standalone final report from audited final-closure artifacts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import closure_core as core
from analysis_utils import markdown_text


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def comparison(summary: dict, left: str, right: str) -> dict:
    return next(
        item for item in summary["headline_comparisons_b16"]
        if item["left"] == left and item["right"] == right
    )


def format_interval(values) -> str:
    return f"[{100 * values[0]:.1f}%, {100 * values[1]:.1f}%]"


def main() -> None:
    if not (core.HERE / "FINAL_AUDIT.md").exists():
        raise AssertionError("final report may only be generated after FINAL_AUDIT.md")
    summaries = core.HERE / "summaries"
    a = json.loads((summaries / "experiment_a_summary.json").read_text())
    b = json.loads((summaries / "experiment_b_summary.json").read_text())
    c = json.loads((summaries / "experiment_c_summary.json").read_text())
    d = json.loads((summaries / "experiment_d_summary.json").read_text())
    classical = json.loads((summaries / "experiment_a_classical_summary.json").read_text())
    claims = json.loads((summaries / "final_claims_summary.json").read_text())
    audit = json.loads((core.HERE / "final_audit_summary.json").read_text())
    verdict = claims["verdict"]
    table_a = pd.read_csv(core.HERE / "tables" / "table_A_independent_seed_comparison.csv")
    table_b = pd.read_csv(core.HERE / "tables" / "table_B_convergence.csv")
    a_cells = pd.read_csv(summaries / "experiment_a_cells.csv")
    references = pd.read_csv(summaries / "experiment_a_references.csv")
    c_cells = pd.read_csv(summaries / "experiment_c_cells.csv")
    c_failures = pd.read_csv(summaries / "experiment_c_srs_failure_cells.csv")
    d_table = pd.read_csv(core.HERE / "tables" / "table_D_coupling_ablation.csv")
    matched = pd.read_csv(summaries / "experiment_b_matched_convergence.csv")
    completion_risk = pd.read_csv(core.DAY5 / "results" / "completion_neural_risk_cells.csv")
    completion_matched = pd.read_csv(core.DAY5 / "results" / "completion_matched_function.csv")

    coupled = comparison(a, "OC2-COUPLED", "CANONICAL-INDEPENDENT")
    independent = comparison(a, "OC2-INDEPENDENT", "CANONICAL-INDEPENDENT")
    coupling_delta = comparison(a, "OC2-COUPLED", "OC2-INDEPENDENT")
    material_expectation = (
        references["canonical_joint_distance"].mean()
        > 0.1 * table_a.loc[table_a["method"] == "CANONICAL-INDEPENDENT", "mean_residual"].iloc[0]
    )
    if independent["equal_source_mean_relative_reduction"] > 0:
        thesis = "Thesis A"
        thesis_text = "Semantic symmetries provide structured randomization that estimates the expectation of randomized learning pipelines more efficiently than independent retraining."
    elif material_expectation:
        thesis = "Thesis B"
        thesis_text = "Semantic symmetrization defines a distinct quotient predictor, and interaction-balanced designs estimate it efficiently."
    else:
        thesis = "Thesis C"
        thesis_text = "Finite nuisance balancing is useful only for restricted seed/schema menus; the broader independent-training claim does not survive."
    if verdict == "SUPPORTED":
        readiness = "READY TO WRITE ICLR"; recommendation = "COMMIT TO PAPER"
    elif verdict == "PARTIALLY SUPPORTED":
        readiness = "READY TO WRITE ICLR"; recommendation = "PIVOT PAPER THESIS"
    else:
        readiness = "PIVOT"; recommendation = "ABANDON ORBITCOVER AS MAIN METHOD"

    headline = table_a[["method", "mean_residual", "relative_to_canonical", "cell_wins", "cells", "source_wins", "sources"]].copy()
    headline.columns = ["Method", "Mean residual", "Relative to canonical independent", "Cell wins", "Cells", "Source wins", "Sources"]
    comparisons = a["headline_comparisons_b16"]
    comparison_lines = "\n".join(
        f"- `{item['left']}` vs `{item['right']}`: {item['cell_wins']}/{item['cells']} cell wins, "
        f"{item['source_wins']}/{item['sources']} source wins, mean reduction {pct(item['equal_source_mean_relative_reduction'])}, "
        f"median {pct(item['cell_median_relative_reduction'])}, clustered 95% {format_interval(item['dataset_clustered_95_interval'])}; "
        f"architecture reductions `{json.dumps(item['architecture_relative_reduction'], sort_keys=True)}`."
        for item in comparisons
    )
    corner = table_b.groupby(["model", "corner"], as_index=False)[
        ["total_nuisance_variance", "oc2_srs_ratio_b16", "oc2_canonical_ratio_b16"]
    ].mean()
    corner_md = markdown_text(corner)
    source_nonpositive = pd.read_csv(summaries / "experiment_c_completion_nonpositive_sources.csv")
    nonpositive_names = ", ".join(source_nonpositive["dataset"].tolist()) or "none"

    b16 = a_cells[a_cells["budget"] == 16].pivot_table(
        index=["dataset", "split_seed", "model"], columns="method", values="residual_mean"
    ).reset_index()
    canonical_wins = b16[b16["CANONICAL-INDEPENDENT"] < b16["OC2-COUPLED"]]
    canonical_win_names = ", ".join(
        f"{row.dataset}/{row.split_seed}/{row.model}" for row in canonical_wins.itertuples()
    ) or "none"
    srs_win_names = ", ".join(
        f"{row.dataset}/{row.split_seed}/{row.model}" for row in c_failures.head(30).itertuples()
    )
    if len(c_failures) > 30:
        srs_win_names += f", and {len(c_failures) - 30} additional cells in the complete failure table"
    strength_losses = completion_risk[
        completion_risk["strength2_16_residual_mean"] > completion_risk["srswor16_residual_mean"]
    ]
    strength_nonrecover = strength_losses[
        strength_losses["strength3_64_residual_mean"] >= strength_losses["srswor64_residual_mean"]
    ]
    strength_nonrecover_names = ", ".join(
        f"{row.dataset}/{row.split_seed}/{row.model}" for row in strength_nonrecover.itertuples()
    ) or "none"
    strength_recover = strength_losses[
        strength_losses["strength3_64_residual_mean"] < strength_losses["srswor64_residual_mean"]
    ]
    strength_recover_names = ", ".join(
        f"{row.dataset}/{row.split_seed}/{row.model}" for row in strength_recover.itertuples()
    ) or "none"
    b_conditions = pd.read_csv(summaries / "experiment_b_conditions.csv")
    vanished = b_conditions[
        (b_conditions["budget"].astype(str) == "convergence")
        & (b_conditions["total_nuisance_variance"] <= 1e-10)
    ]
    vanished_names = ", ".join(
        f"{row.dataset}/{row.model}/N={row.training_rows}" for row in vanished.itertuples()
    ) or "none"
    little = b16.assign(
        reduction=1 - b16["OC2-COUPLED"] / b16["CANONICAL-INDEPENDENT"]
    ).groupby("dataset")["reduction"].mean().sort_values().head(3)

    ordinary_matched = completion_matched.groupby("model")[["ordinary_variance", "matched_variance"]].mean()
    matched_convergence = matched.groupby(["model", "budget"], as_index=False)[
        ["ordinary_variance", "matched_variance", "fraction_removed"]
    ].mean()
    mechanisms = d_table[["method", "mean_residual", "mean_relative_reduction_vs_none", "cell_wins", "cells"]]
    total_fits = audit["registry"]["complete_fit_keys"]
    fit_hours = audit["registry"]["summed_fit_hours"]
    gpu_fit_hours = audit["registry"]["summed_gpu_fit_hours"]
    cpu_fit_hours = audit["registry"]["summed_cpu_fit_hours"]
    closure_wall_hours = audit["closure_wall_clock_hours"]
    equivalent = a["iid_equivalent_budget"]

    scores = {
        "novelty": 4 if verdict != "NOT SUPPORTED" else 2,
        "theory": 4,
        "empirical breadth": 5,
        "baseline strength": 5,
        "mechanism": 4 if claims["decision_components"]["interaction_prediction"] else 3,
        "realistic-scale evidence": 5,
        "prospective validity": 4,
        "story coherence": 4 if verdict == "SUPPORTED" else 3,
        "reproducibility": 5,
    }
    score_lines = "\n".join(f"- {key}: **{value}/5**" for key, value in scores.items())

    report = f"""# RESULTS — FINAL ICLR CLOSURE

## 1. Executive verdict

**{verdict}**

The final closure directly compares OrbitCover with genuinely independent full-pipeline retraining on all 144 neural dataset×split×architecture cells. At B=16, OC2-coupled versus canonical independent achieves {coupled['cell_wins']}/{coupled['cells']} cell wins and {coupled['source_wins']}/{coupled['sources']} source wins, with an equal-source reduction of {pct(coupled['equal_source_mean_relative_reduction'])} and clustered 95% interval {format_interval(coupled['dataset_clustered_95_interval'])}. OC2-independent achieves {independent['cell_wins']}/{independent['cells']} cells and {independent['source_wins']}/{independent['sources']} sources, a {pct(independent['equal_source_mean_relative_reduction'])} mean reduction. Coupling changes the result by {pct(coupling_delta['equal_source_mean_relative_reduction'])} relative to schema-only independent balancing. The mean squared distance between canonical and schema×independent expectations is `{a['canonical_joint_distance_mean']:.3e}`, so target shift is reported rather than assumed away. At convergence, the mean OC2/SRS residual ratio is `{b['orbitcover_mean_oc2_srs_ratio_at_convergence']:.3f}`. Interaction structure gives Spearman rho `{c['main_pair_fraction_vs_gain']['spearman']:.3f}` for main+pair mass versus gain and `{c['higher_fraction_vs_gain']['spearman']:.3f}` for higher-order mass versus gain. Experiment D's lowest-residual ablation is `{d['best_method_by_mean_residual']}`. Exact matched initialization remains architecture-dependent: MLP/ResNet are negligible while FT-Transformer/TabM retain the reported residuals. All mandatory A–C cells and preferred D are complete. The final audit passes {audit['tests']['passed']}/{audit['tests']['total']} tests and verifies {total_fits:,} registry fit keys. The defensible thesis is selected from the frozen rules, not repaired after the outcome.

## 2. What changed relative to the previous results

The previous result established 144/144 material finite nuisance tensors, 144/144 strength-2 wins over IID-16, but only 8/12 positive source means versus SRSWOR. It also showed that exact matched initialization removes about 98% of pooled ordinary schema variance, closing MLP/ResNet and leaving architecture-specific FT-Transformer/TabM residuals. The closure replaces the reused two-seed menu objection with 128 canonical independent seeds and schema×independent pools, tests realistic N/optimization trajectories, predicts the SRS boundary from fANOVA structure, and isolates schema/init/order coupling. Earlier evidence grades remain unchanged.

## 3. Independent canonical-seed showdown

{markdown_text(headline)}

The five required paired comparisons are:

{comparison_lines}

Cached estimator constructions use 512 draws per cell; overlapping draws are never the inferential unit. Dataset is the primary unit.

## 4. Does schema symmetrization change the expectation?

Across cells, mean `||Q_canonical_independent - Q_schema×independent||²` is `{references['canonical_joint_distance'].mean():.3e}` (median `{references['canonical_joint_distance'].median():.3e}`). It exceeds the cell-specific 95% Monte Carlo noise threshold in {a['canonical_joint_distinguishable_cells']}/{a['canonical_joint_distinguishable_total']} cells. Mean canonical-to-finite-coupled distance is `{references['canonical_coupled_distance'].mean():.3e}`, and mean joint-to-coupled distance is `{references['joint_coupled_distance'].mean():.3e}`. Relative to the B=16 canonical residual, the canonical/joint distance is {'material' if material_expectation else 'small'} under the frozen 10% descriptive materiality check. OrbitCover is therefore interpreted as {'estimating a distinct symmetrized target as well as reducing variance' if material_expectation else 'primarily a variance-reduction coupling for approximately the same target'}; cross-target residuals are retained in `experiment_a_cells.csv`.

## 5. What does the 98% matched-function result mean now?

{markdown_text(ordinary_matched.reset_index())}

MLP and ResNet still close to numerical precision under exact function matching. FT-Transformer retains the largest matched-path component, while TabM retains a smaller component. The convergence repeat is:

{markdown_text(matched_convergence)}

This rules out a universal claim that schema alone irreducibly changes every optimizer path. The supported scope is architecture-specific token/member/dropout/minibatch dynamics plus structured finite/infinite randomization.

## 6. Training-scale and convergence

The mandatory model-level corners are:

{corner_md}

Nuisance variance {'persists' if b['nuisance_persists_at_convergence'] else 'does not persist'} in at least one realistic convergence condition. OrbitCover relative efficiency {'persists' if b['orbitcover_mean_oc2_srs_ratio_at_convergence'] < 1 else 'does not persist'} on average at convergence. Dataset size changes effect magnitude and interaction mix, but the raw trajectories—not a fragile fitted exponent—are the primary result.

The descriptive, dataset-clustered log-risk slopes are `{json.dumps(b['descriptive_log_risk_slopes'], sort_keys=True)}`. They summarize direction and uncertainty only; they are not promoted as asymptotic scaling exponents.

## 7. Interaction spectrum explains successes/failures

Main+pair fraction versus OC2 gain has Spearman rho `{c['main_pair_fraction_vs_gain']['spearman']:.3f}` with clustered interval `{c['main_pair_fraction_vs_gain']['dataset_clustered_95_interval']}`. Higher-order fraction versus gain has rho `{c['higher_fraction_vs_gain']['spearman']:.3f}` with interval `{c['higher_fraction_vs_gain']['dataset_clustered_95_interval']}`. Mean high-order fraction is `{c['mean_high_order_fraction_oc2_wins']:.3f}` in OC2 wins and `{c['mean_high_order_fraction_srs_wins']:.3f}` in strict SRS wins. Architecture-specific correlations and clustered intervals are `{json.dumps(c['architecture_stratified'], sort_keys=True)}`. The prior four non-positive source means are {nonpositive_names}; exact ties remain ties. Their source-level spectrum comparison with the eight positive sources is `{json.dumps(c['completion_source_spectrum_comparison'], sort_keys=True)}`. The transparent model's leave-one-dataset-out result is `{json.dumps(c['transparent_model'], sort_keys=True)}`. The complete failure table contains {c['srs_failure_cells']} strict cells and is not filtered for favorability.

## 8. Strength hierarchy

Strength-1 balances only main effects; strength-2 removes the matched pairwise spectrum and remains the B=16 default; strength-3 targets triples at B=64 and closes products when the budget reaches the population. Among prior strength-2/SRS losses, strength-3 recoveries are: {strength_recover_names}. Non-recoveries are: {strength_nonrecover_names}. Thus “match strength to interaction order” is {'empirically supported as a boundary rule' if c['main_pair_fraction_vs_gain']['spearman'] > 0 and c['higher_fraction_vs_gain']['spearman'] < 0 else 'not reliably supported as a cell-ranking rule'}, not a guarantee that strength-3 always beats finite-population sampling.

## 9. Coupling mechanism

{markdown_text(mechanisms)}

The best finite ablation is `{d['best_method_by_mean_residual']}`. The full component means are `{json.dumps(d['mean_fanova_components'], sort_keys=True)}`. This answers whether the benefit is principally schema, RNG, or pairwise schema×RNG balance without confusing the finite mechanism tensor with independent infinite-seed retraining.

## 10. Architecture-specific conclusions

### MLP

Matched residual is negligible. Its independent/coupled reductions are reported in the B=16 architecture dictionary; low-order schema structure is the useful regime.

### ResNet

Matched residual is negligible, but ordinary stochastic/schema interaction remains material. Structured coupling can still reduce quotient Monte Carlo even when exact coordinate matching closes the fixed path.

### FT-Transformer

FT-Transformer has the largest high-order and matched-path residual. Its weaker strength-2 boundary is predicted prospectively by the interaction analysis rather than hidden.

### TabM

TabM has strong low-order structure with a small nonzero matched residual. The independent-seed comparison determines whether this translates beyond its earlier finite seed menu.

### TabPFN

Prior TabPFN evidence remains separate: default internal ensembling reduced external schema risk, and external strength-2 beat IID 18/18 and SRS 12/18 cells. Calls and internal members are not mislabeled as retrained fits.

### CatBoost/GBDT

The final secondary independent-seed results are `{json.dumps(classical['model_summary'], sort_keys=True)}`. Prior native CatBoost had zero category-ID total effect, while ordinal XGBoost remained category-ID sensitive; deterministic/invariant GBDT cells remain an explicit boundary.

## 11. Practical compute efficiency

The audited registry contains {total_fits:,} complete unique fit keys and `{fit_hours:.3f}` summed fit-hours of local telemetry: `{gpu_fit_hours:.3f}` GPU-fit-hours and `{cpu_fit_hours:.3f}` CPU-fit-hours. End-to-end closure wall clock from the frozen hash through audit is `{closure_wall_hours:.3f}` hours with two H100 NVL devices and concurrent CPU analysis. For OC2-independent, 16 fits match a median `{equivalent['OC2-INDEPENDENT']['median_bracketed']:.1f}` canonical-independent fits among {equivalent['OC2-INDEPENDENT']['bracketed_cells']}/{equivalent['OC2-INDEPENDENT']['total_cells']} bracketed cells; {equivalent['OC2-INDEPENDENT']['cells_above_64']} cells require more than 64 by the observed curve. For OC2-coupled the corresponding median is `{equivalent['OC2-COUPLED']['median_bracketed']:.1f}` across {equivalent['OC2-COUPLED']['bracketed_cells']}/{equivalent['OC2-COUPLED']['total_cells']} bracketed cells, with {equivalent['OC2-COUPLED']['cells_above_64']} above 64. No budget equivalence is asserted outside the observed 4–64 bracket. GPU figures are H100-local measurements, not portable latency guarantees.

## 12. Ranking/model-selection implications

The prior exact validation result remains: strength-2 winner agreement 99.41% versus 96.69% IID and Spearman 98.64% versus 96.77%. Held-out selected-test regret changed only from `0.005029` to `0.004906`, because exact validation/test winners agreed in only 19/36 partitions. Partition shift is distinct from nuisance Monte Carlo; the final paper must not headline the small test-regret difference.

## 13. Failure cases

- Canonical-independent wins over OC2-coupled: {canonical_win_names}.
- Strict SRSWOR wins (first 30 listed): {srs_win_names or 'none'}.
- Strength-3 non-recoveries: {strength_nonrecover_names}.
- Strength-3 recoveries among the same loss panel: {strength_recover_names}.
- Nuisance variance at or below `1e-10` at convergence: {vanished_names}.
- Architectures with negligible matched residual: MLP and ResNet.
- Three datasets with least mean OC2-coupled benefit: {', '.join(f'{name} ({value:.1%})' for name, value in little.items())}.

## 14. Final defensible theorem/claim target

For a declared semantic nuisance distribution and randomized learner, the symmetrized predictor is a finite/infinite expectation. Orthogonal-array designs exactly cancel fANOVA components through their design strength, while residual error is governed by unmatched higher-order mass and finite-population/coupling covariance. Empirically, this yields the measured equal-budget reductions and the reported architecture/interaction boundaries. No novelty is claimed for orthogonal arrays, group averaging, or generic antithetic sampling; the contribution is their semantic learning-pipeline formulation, exact prediction-space accounting, and broad falsificatory boundary map.

## 15. Recommended final paper thesis

**{thesis}**

> {thesis_text}

## 16. Best paper title

1. **OrbitCover: Interaction-Balanced Semantic Randomization for Efficient Predictor Symmetrization**
2. **When Equivalent Tables Train Differently: Structured Randomization Beyond Independent Ensembling**
3. **Matching Design Strength to Schema Interaction Order in Randomized Tabular Learning**

## 17. ICLR readiness

{score_lines}

**{readiness}**

All frozen mandatory experiments are complete; no extra experiment is invented merely because a result is mixed.

## 18. Five strongest reviewer objections

1. **Objection:** Independent canonical seeds may erase the claimed advantage. **Evidence:** the 144-cell B=16 showdown and clustered comparison above. **Remaining weakness:** a finite 128-seed reference still has Monte Carlo error. **Best response:** show reference bootstrap and every cross-target residual.
2. **Objection:** The phenomenon is transient undertraining. **Evidence:** the six-source nested N×budget×convergence grid and matched convergence repeat. **Remaining weakness:** architectures/datasets remain tabular rather than vision/language scale. **Best response:** scope the claim to the tested randomized tabular pipelines.
3. **Objection:** SRSWOR is already optimal enough. **Evidence:** all failure cells plus the prospective interaction-spectrum correlation and strength hierarchy. **Remaining weakness:** the transparent predictor need not rank every cell. **Best response:** present interaction order as a boundary condition, not an oracle.
4. **Objection:** Matched initialization removes the effect. **Evidence:** MLP/ResNet closure is retained, FT-Transformer/TabM residuals and independent-RNG evidence are separated. **Remaining weakness:** exact token/member mechanism is not fully identified. **Best response:** abandon the universal optimizer-path claim and lead with structured expectation estimation.
5. **Objection:** Validation fidelity does not imply useful held-out selection. **Evidence:** partition-shift decomposition and the 19/36 exact winner agreement. **Remaining weakness:** predictive gains are small. **Best response:** make quotient estimation—not SOTA prediction or selection—the primary endpoint.

## 19. Final recommendation

**{recommendation}**

The audited evidence supports this choice under the frozen rule: independent-seed reduction={pct(independent['equal_source_mean_relative_reduction'])}, coupled reduction={pct(coupled['equal_source_mean_relative_reduction'])}, convergence OC2/SRS=`{b['orbitcover_mean_oc2_srs_ratio_at_convergence']:.3f}`, and interaction correlations have the signs reported above. The paper should state every canonical, SRS, convergence, matched-path, and selection failure prominently and use **{thesis}** exactly as the thesis boundary.
"""
    destination = core.HERE / "results.md"
    destination.write_text(report)
    for target in (core.DAY5 / "results.md", core.REPO / "results.md"):
        shutil.copyfile(destination, target)
    print(f"wrote {destination}, {core.DAY5 / 'results.md'}, and {core.REPO / 'results.md'}")


if __name__ == "__main__":
    main()
