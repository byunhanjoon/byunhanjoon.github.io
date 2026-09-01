"""Apply the frozen decision rules and emit table E before the final audit."""

from __future__ import annotations

import json

import pandas as pd

import closure_core as core
from analysis_utils import markdown_table, write_summary


def comparison(summary: dict, left: str, right: str) -> dict:
    for item in summary["headline_comparisons_b16"]:
        if item["left"] == left and item["right"] == right:
            return item
    raise KeyError((left, right))


def main() -> None:
    summaries = core.HERE / "summaries"
    a = json.loads((summaries / "experiment_a_summary.json").read_text())
    b = json.loads((summaries / "experiment_b_summary.json").read_text())
    c = json.loads((summaries / "experiment_c_summary.json").read_text())
    d = json.loads((summaries / "experiment_d_summary.json").read_text())
    coupled = comparison(a, "OC2-COUPLED", "CANONICAL-INDEPENDENT")
    independent = comparison(a, "OC2-INDEPENDENT", "CANONICAL-INDEPENDENT")
    a_pass = coupled["equal_source_mean_relative_reduction"] > 0
    b_pass = b["orbitcover_mean_oc2_srs_ratio_at_convergence"] < 1
    main_pair = c["main_pair_fraction_vs_gain"]
    higher = c["higher_fraction_vs_gain"]
    c_pass = main_pair["spearman"] > 0 and higher["spearman"] < 0
    architecture_pass = all(
        value > 0 for value in coupled["architecture_relative_reduction"].values()
    )
    if a_pass and b_pass and c_pass and architecture_pass:
        verdict = "SUPPORTED"
    elif (
        coupled["equal_source_mean_relative_reduction"] <= 0
        and independent["equal_source_mean_relative_reduction"] <= 0
        and d["best_method_by_mean_residual"] == "none"
    ):
        verdict = "NOT SUPPORTED"
    else:
        verdict = "PARTIALLY SUPPORTED"
    rows = [
        {
            "claim": "OrbitCover beats canonical independent retraining",
            "decision": "pass" if a_pass else "fail",
            "evidence": f"OC2-coupled equal-source B16 reduction {coupled['equal_source_mean_relative_reduction']:.3%}, interval {coupled['dataset_clustered_95_interval']}",
            "boundary": f"OC2-independent reduction {independent['equal_source_mean_relative_reduction']:.3%}",
        },
        {
            "claim": "Relative efficiency survives convergence",
            "decision": "pass" if b_pass else "fail",
            "evidence": f"mean convergence OC2/SRS ratio {b['orbitcover_mean_oc2_srs_ratio_at_convergence']:.4f}",
            "boundary": "absolute risk and per-architecture corners remain separately reported",
        },
        {
            "claim": "Interaction order explains the SRS boundary",
            "decision": "pass" if c_pass else "fail",
            "evidence": f"rho(main+pair,gain)={main_pair['spearman']:.3f}; rho(higher,gain)={higher['spearman']:.3f}",
            "boundary": "transparent leave-one-dataset-out performance is descriptive",
        },
        {
            "claim": "No unexplained systematic architecture failure",
            "decision": "pass" if architecture_pass else "fail",
            "evidence": json.dumps(coupled["architecture_relative_reduction"], sort_keys=True),
            "boundary": "matched-path residual remains architecture-dependent",
        },
        {
            "claim": "Coupling mechanism localized",
            "decision": "complete",
            "evidence": f"lowest mean-residual ablation: {d['best_method_by_mean_residual']}",
            "boundary": "finite init/order menu mechanism, not an infinite-RNG theorem",
        },
    ]
    table = pd.DataFrame(rows)
    table.to_csv(core.HERE / "tables" / "table_E_final_claims.csv", index=False)
    markdown_table(table, core.HERE / "tables" / "table_E_final_claims.md")
    payload = {
        "status": "complete", "verdict": verdict,
        "decision_components": {
            "independent_showdown": a_pass, "convergence": b_pass,
            "interaction_prediction": c_pass, "architecture_boundary": architecture_pass,
        },
    }
    write_summary(summaries / "final_claims_summary.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
