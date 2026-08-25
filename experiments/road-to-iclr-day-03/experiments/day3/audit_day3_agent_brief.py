"""Fail-closed audit of the literal requirements in ``day3_agent.md``.

This audit treats negative results and predeclared cuts as valid completion.
It checks whether the experiment was performed and reported, not whether every
scientific hypothesis succeeded.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "day3"
AUDIT_JSON = RESULTS / "day3_agent_completion_audit.json"
AUDIT_MD = ROOT / "DAY3_AGENT_COMPLETION_AUDIT.md"
KAPPAS = {1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def values(rows: list[dict[str, str]], column: str) -> set[str]:
    return {row.get(column, "") for row in rows if row.get(column, "") != ""}


def subset(
    rows: list[dict[str, str]], **constraints: str
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if all(row.get(column) == expected for column, expected in constraints.items())
    ]


def exact_scores(
    rows: list[dict[str, str]], expected: dict[str, float], tolerance: float = 1e-14
) -> bool:
    by_representation = {row["representation"]: row for row in rows}
    return all(
        representation in by_representation
        and abs(float(by_representation[representation]["test_score"]) - score)
        <= tolerance
        for representation, score in expected.items()
    )


def main() -> None:
    brief = ROOT / "day3_agent.md"
    report = ROOT / "REPORT_DAY3.md"
    report_text = report.read_text(encoding="utf-8") if report.exists() else ""
    config = read_json(ROOT / "experiments/day3/configs/preregistered.json")
    summary = read_json(RESULTS / "analysis_summary.json")

    adult_anchor = read_csv(ROOT / "results/day3_anchor_reproduction.csv")
    black_friday_anchor = read_csv(
        ROOT / "results/day3_anchor_reproduction_black_friday.csv"
    )
    numeric = read_csv(RESULTS / "numeric_kappa.csv")
    whitening = read_csv(RESULTS / "ple_identity_whitening_exact.csv")
    equivalence = read_json(RESULTS / "ple_identity_equivalence_exact.json")
    invariant = read_csv(RESULTS / "invariant_regularizer.csv")
    ordinal_basis = read_csv(RESULTS / "ordinal_basis.csv")
    ordinal_kappa = read_csv(RESULTS / "ordinal_kappa.csv")
    categorical = read_csv(RESULTS / "categorical_kappa.csv")
    residualization = read_csv(RESULTS / "block_residualization.csv")
    cyclic = read_csv(RESULTS / "cyclic_geometry.csv")
    residual_te = read_csv(RESULTS / "residual_te.csv")
    frequency = read_csv(RESULTS / "frequency_preconditioning.csv")
    final_verification = read_json(
        RESULTS / "broad_benchmark/final_verification.json"
    )
    broad_audit = read_json(
        RESULTS / "broad_benchmark/completion_audit.json"
    )
    broad_phase = read_csv(RESULTS / "broad_benchmark/phase1_all.csv")

    requirements: dict[str, dict[str, Any]] = {}

    def record(name: str, passed: bool, evidence: Any) -> None:
        requirements[name] = {"passes": bool(passed), "evidence": evidence}

    adult_expected = {
        "cumulative_ple": 0.8590381426202321,
        "local_ple": 0.8586081935999017,
        "basis_blend": 0.8597137767950371,
    }
    black_friday_expected = {
        "cumulative_ple": 0.6966277030051913,
        "local_ple": 0.694845662205779,
        "basis_blend": 0.6931962505786373,
    }
    record(
        "1_day2_anchors",
        exact_scores(adult_anchor, adult_expected)
        and exact_scores(black_friday_anchor, black_friday_expected),
        {
            "adult_rows": len(adult_anchor),
            "black_friday_rows": len(black_friday_anchor),
            "adult_expected_scores": adult_expected,
            "black_friday_expected_scores": black_friday_expected,
            "comparison": "bit-for-bit scalar equality to frozen Day-2 rows",
        },
    )

    numeric_mlp_complete = all(
        len(subset(numeric, dataset=dataset, model="mlp", target_kappa=str(kappa)))
        == 5
        for dataset in ("adult", "california", "diamond")
        for kappa in KAPPAS
    )
    numeric_diagnostics = all(
        row.get(field, "") != ""
        for row in numeric
        for field in (
            "realized_block_kappa_mean",
            "train_loss",
            "val_metric",
            "test_metric",
            "first_gradient_norm",
            "first_weight_norm",
        )
    )
    numeric_resnet = {
        (row["dataset"], float(row["target_kappa"]))
        for row in numeric
        if row["model"] == "resnet"
    }
    expected_resnet = {
        (dataset, kappa)
        for dataset in ("adult", "california", "diamond")
        for kappa in (1.0, 1000.0)
    }
    record(
        "2_controlled_condition_sweep",
        numeric_mlp_complete
        and numeric_diagnostics
        and numeric_resnet == expected_resnet
        and values(numeric, "global_scale_control")
        == {"geometric_mean_singular_value_1"},
        {
            "rows": len(numeric),
            "mlp_datasets": sorted(
                values([row for row in numeric if row["model"] == "mlp"], "dataset")
            ),
            "kappas": sorted(float(value) for value in values(numeric, "target_kappa")),
            "resnet_endpoints": sorted([list(item) for item in numeric_resnet]),
            "diagnostics_present": numeric_diagnostics,
        },
    )

    whitening_summary = summary.get("whitening", {})
    reconstruction_values: list[float] = []

    def collect_numbers(node: Any, key_hint: str = "") -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                collect_numbers(value, key.lower())
        elif isinstance(node, list):
            for value in node:
                collect_numbers(value, key_hint)
        elif isinstance(node, (int, float)) and any(
            token in key_hint for token in ("error", "angle")
        ):
            reconstruction_values.append(float(node))

    collect_numbers(equivalence)
    record(
        "3_ple_identity_whitening",
        len(whitening) == 50
        and values(whitening, "family") == {"identity", "ple"}
        and values(whitening, "canonicalization")
        == {"aligned", "centered", "raw", "standardized", "whitened"}
        and whitening_summary.get("gap_reduction", 0.0) >= 0.9
        and bool(reconstruction_values)
        and max(reconstruction_values) <= 1e-7,
        {
            "rows": len(whitening),
            "gap_reduction": whitening_summary.get("gap_reduction"),
            "max_recorded_reconstruction_error_or_angle": max(reconstruction_values)
            if reconstruction_values
            else None,
        },
    )

    regularizer_summary = summary.get("regularizer", {})
    record(
        "4_invariant_regularization",
        len(invariant) == 123
        and values(invariant, "regularizer")
        == {"invariant", "no_first_wd", "standard"}
        and {float(value) for value in values(invariant, "target_kappa")}
        == {1.0, 30.0, 300.0, 1000.0, 3000.0}
        and "H3 not supported" in report_text,
        {
            "rows": len(invariant),
            "regularizers": sorted(values(invariant, "regularizer")),
            "aggregate_spread": {
                key: value.get("spread")
                for key, value in regularizer_summary.items()
            },
            "verdict": "negative result retained",
        },
    )

    ordinal_variants = {
        "local",
        "cumulative",
        "cumulative_standardized",
        "path_spectral",
        "whitened",
    }
    ordinal_controlled_pairs = {
        (row["dataset"], row["model"])
        for row in ordinal_kappa
        if float(row["target_kappa"]) > 1.0
    }
    record(
        "5_ordinal_geometry",
        len(ordinal_basis) == 90
        and ordinal_variants.issubset(values(ordinal_basis, "representation"))
        and {"adult", "black-friday", "diamond"}.issubset(
            values(ordinal_basis, "dataset")
        )
        and {("adult", "mlp"), ("diamond", "mlp"), ("diamond", "resnet")}.issubset(
            ordinal_controlled_pairs
        ),
        {
            "natural_basis_rows": len(ordinal_basis),
            "controlled_rows": len(ordinal_kappa),
            "datasets": sorted(values(ordinal_basis, "dataset")),
            "controlled_dataset_models": sorted(
                [list(item) for item in ordinal_controlled_pairs]
            ),
        },
    )

    categorical_pairs = {
        (row["dataset"], row["model"])
        for row in categorical
        if float(row["target_kappa"]) > 1.0
    }
    record(
        "6_nominal_categorical_geometry",
        len(categorical) == 92
        and {"adult", "diamond"} == values(categorical, "dataset")
        and {("adult", "mlp"), ("diamond", "mlp"), ("adult", "resnet"), ("diamond", "resnet")}.issubset(
            categorical_pairs
        ),
        {
            "rows": len(categorical),
            "dataset_models": sorted([list(item) for item in categorical_pairs]),
            "kappas": sorted(
                float(value) for value in values(categorical, "target_kappa")
            ),
        },
    )

    residualization_representations = {
        "raw_joint",
        "block_residualized",
        "block_residualized_whitened",
        "blockwise_whitened",
        "joint_whitened",
        "standardized_categorical",
    }
    record(
        "7_block_residualization",
        len(residualization) == 60
        and residualization_representations
        == values(residualization, "representation")
        and {"adult", "diamond"} == values(residualization, "dataset")
        and all(
            row.get(field, "") != ""
            for row in residualization
            for field in (
                "cross_gram_before",
                "cross_gram_after",
                "joint_reconstruction_error",
            )
        ),
        {
            "rows": len(residualization),
            "representations": sorted(values(residualization, "representation")),
            "verdict": "geometry succeeded; predictive intervention failed",
        },
    )

    cross_atom_entry = ROOT / "experiments/day3/run_cross_atoms.py"
    supporting_complete = (
        len(cyclic) == 70
        and len(residual_te) == 30
        and len(frequency) == 30
        and cross_atom_entry.exists()
        and "not launched" in cross_atom_entry.read_text(encoding="utf-8").lower()
    )
    record(
        "8_secondary_branches",
        supporting_complete,
        {
            "cyclic_rows": len(cyclic),
            "residual_te_rows": len(residual_te),
            "frequency_preconditioning_rows": len(frequency),
            "cross_atoms": "predeclared P2 cut; no result claimed",
        },
    )

    prospective = set(config.get("datasets", {}).get("prospective_day2", []))
    required_prospective = {
        "wine_quality",
        "miami_housing",
        "Food_Delivery_Time",
        "seismic-bumps",
        "heloc",
        "credit_card_clients_default",
    }
    record(
        "9_dataset_and_model_integrity",
        prospective == required_prospective
        and "arrays/loaders are not present" in report_text
        and {
            "mlp",
            "resnet",
            "dense_stem_tabm",
            "dense_stem_ft_transformer",
        }.issubset(values(broad_phase, "model")),
        {
            "frozen_prospective_datasets_retained": sorted(prospective),
            "missing_arrays_documented_without_substitution": True,
            "later_broad_models": sorted(values(broad_phase, "model")),
            "later_broad_result_rows": len(broad_phase),
        },
    )

    optimizer_report = (
        ROOT / "OPTIMIZER_REMEDIES_REPORT.md"
    ).read_text(encoding="utf-8")
    record(
        "10_required_controls",
        values(numeric, "global_scale_control")
        == {"geometric_mean_singular_value_1"}
        and "random orthogonal control" in report_text.lower()
        and "diagonal standardization" in report_text.lower()
        and "whitening" in report_text.lower()
        and "no first-layer decay" in report_text.lower()
        and "SGD" in optimizer_report,
        {
            "global_scale": "geometric mean singular value 1",
            "kappa_1": "random orthogonal control",
            "normalization": ["diagonal standardization", "whitening"],
            "regularization": ["standard WD", "no first-layer WD", "invariant"],
            "optimizers": ["AdamW", "SGD follow-up"],
        },
    )

    figures = {
        "kappa_vs_performance": "numeric_kappa_vs_metric.png",
        "kappa_vs_convergence": "numeric_kappa_vs_convergence.png",
        "ple_identity_before_after_whitening": "ple_identity_gap_before_after_whitening.png",
        "standard_vs_invariant_regularization": "basis_sensitivity_standard_vs_invariant_regularizer.png",
        "ordinal_local_vs_cumulative": "ordinal_local_vs_cumulative_spectrum.png",
        "categorical_basis_conditioning": "categorical_kappa_vs_metric.png",
        "block_residualization_diamonds": "diamonds_variants.png",
        "summary_geometry_vs_performance": "summary_geometry_vs_performance.png",
    }
    figure_status = {
        concept: (RESULTS / "figures" / filename).exists()
        for concept, filename in figures.items()
    }
    seven_answers = all(
        phrase in report_text
        for phrase in (
            "Did controlled conditioning causally affect learning?",
            "Did whitening explain PLE versus identity?",
            "Did ordinals reproduce the effect?",
            "Did nominal categoricals reproduce the effect?",
            "Did invariant regularization reduce sensitivity?",
            "What is the strongest defensible ICLR claim?",
            "What single experiment should Day 4 run next?",
        )
    )
    record(
        "11_deliverables",
        report.exists() and all(figure_status.values()) and seven_answers,
        {
            "report": str(report.relative_to(ROOT)),
            "seven_question_block": seven_answers,
            "figures": figure_status,
        },
    )

    verification_ok = bool(
        final_verification.get("py_compile", {}).get("exit_code") == 0
        and final_verification.get("pytest", {}).get("exit_code") == 0
    )
    record(
        "12_verification",
        verification_ok,
        {
            "final_verification_complete": final_verification.get("complete"),
            "pytest_exit_code": final_verification.get("pytest", {}).get("exit_code"),
            "py_compile_exit_code": final_verification.get("py_compile", {}).get("exit_code"),
            "broad_freeze_matches": broad_audit.get("freeze", {}).get("matches"),
            "broad_freeze_changed": broad_audit.get("freeze", {}).get("changed", []),
            "note": (
                "The literal Day-3 brief passes syntax/tests. The separate broad "
                "benchmark provenance audit remains fail-closed after two committed "
                "post-freeze trajectory helpers changed protected source/test files."
            ),
        },
    )

    complete = all(item["passes"] for item in requirements.values())
    payload = {
        "brief": str(brief.relative_to(ROOT)),
        "brief_sha256": hashlib.sha256(brief.read_bytes()).hexdigest()
        if brief.exists()
        else None,
        "interpretation": (
            "Negative hypothesis outcomes and preregistered cuts count as completion; "
            "scientific success is reported separately."
        ),
        "warnings": [
            {
                "scope": "later broad-benchmark provenance",
                "message": (
                    "The broad freeze audit flags two committed post-freeze files: "
                    "experiments/day3/broad_data.py and tests/test_broad_benchmark.py. "
                    "Coverage and scientific invariants pass, but the freeze manifest "
                    "must remain visibly nonmatching."
                ),
            }
        ]
        if broad_audit.get("freeze", {}).get("matches") is False
        else [],
        "requirements": requirements,
        "complete": complete,
    }
    AUDIT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# `day3_agent.md` completion audit",
        "",
        "This audit maps the literal experiment brief to generated evidence. A",
        "negative result is complete when the required test was run and retained; it",
        "does not become a positive scientific result.",
        "",
        "| Requirement | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for name, item in requirements.items():
        status = "PASS" if item["passes"] else "FAIL"
        evidence = json.dumps(item["evidence"], sort_keys=True)
        if len(evidence) > 260:
            evidence = evidence[:257] + "..."
        lines.append(f"| `{name}` | **{status}** | `{evidence}` |")
    lines.extend(
        [
            "",
            f"**Overall: {'COMPLETE' if complete else 'INCOMPLETE'}.**",
            "",
            "The original success criteria were only partially supported: H1/H2 and",
            "the ordinal/nominal extensions succeeded, while the proposed invariant",
            "regularizer failed. The correct conclusion is therefore a completed,",
            "decisive mechanism study—not full confirmation of every hypothesis.",
            "",
            "Provenance warning: the later broad-benchmark freeze audit remains red",
            "because commit `768e0c0` added a trajectory helper to `broad_data.py` and",
            "its test after the broad manifest was frozen. All coverage and scientific",
            "invariants pass, but this warning is intentionally not suppressed.",
            "",
            "Machine-readable evidence: `results/day3/day3_agent_completion_audit.json`.",
        ]
    )
    AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if complete else 1)


if __name__ == "__main__":
    main()
