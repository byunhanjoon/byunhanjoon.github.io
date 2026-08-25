"""Fail-closed requirement-by-requirement audit for the seven-item Day 3 goal."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .broad_data import config
from .broad_extension_data import extension_config


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/day3/broad_benchmark"


def read_json(name: str) -> dict:
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else {}


def read_csv(name: str) -> pd.DataFrame:
    path = RESULTS / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def text_contains(path: Path, values: list[str]) -> bool:
    if not path.exists():
        return False
    text = path.read_text().lower()
    return all(value.lower() in text for value in values)


def main() -> None:
    broad = read_json("completion_audit.json")
    extension = read_json("extension_completion_audit.json")
    shift = read_json("distribution_shift_completion_audit.json")
    combined = read_json("combined_30_summary.json")
    verification = read_json("final_verification.json")
    phase = read_csv("phase1_all.csv")
    phase_valid = phase[phase.failure.fillna("").eq("")] if len(phase) else phase
    confirmation = read_csv("confirmation_all.csv")
    remedy_efficiency = read_csv("phase1_remedy_efficiency.csv")
    confirmation_efficiency = read_csv("confirmation_efficiency.csv")
    robustness_rank = read_csv("robustness_rank_summary.csv")
    natural = read_csv("final_natural_encoding_pairs.csv")

    required_models = {
        "mlp", "resnet", "dense_stem_ft_transformer", "dense_stem_tabm"
    }
    required_remedies = {
        "adamw",
        "diagonal_adamw",
        "whiten_adamw",
        "anchor_whiten_adamw",
        "sketch_anchor_whiten_adamw",
        "input_natural",
        "first_layer_kfac",
        "shampoo",
        "soap",
    }
    required_representations = {
        "cumulative_helmert", "local_adjacent", "raw_standard", "quantile_standard"
    }

    requirement_1 = bool(
        broad.get("phase1", {}).get("complete")
        and extension.get("complete")
        and combined.get("design", {}).get("combined_distinct_datasets") == 30
        and combined.get("design", {}).get("original_datasets") == 25
        and combined.get("design", {}).get("prospective_extension_datasets") == 5
    )
    confirmation_models = set(confirmation.model.unique()) if len(confirmation) else set()
    preprocessing_representations = (
        set(phase_valid.representation.unique()) if len(phase_valid) else set()
    )
    requirement_2 = bool(
        broad.get("confirmation", {}).get("complete")
        and required_models == set(config()["models"])
        and required_models.issubset(confirmation_models)
        and required_representations.issubset(preprocessing_representations)
    )
    phase_remedies = set(phase_valid.remedy.unique()) if len(phase_valid) else set()
    requirement_3 = bool(
        required_remedies.issubset(phase_remedies)
        and len(phase_valid[phase_valid.remedy.isin(required_remedies)]) > 0
    )
    natural_error = broad.get("scientific_invariants", {}).get(
        "natural_equivalence_max_error"
    )
    requirement_4 = bool(
        len(natural) == 150
        and natural.dataset.nunique() == 25
        and set(natural.model.unique()) == {"mlp", "resnet"}
        and natural_error is not None
        and natural_error <= 1e-8
    )
    memory_valid = 0
    for table in (remedy_efficiency, confirmation_efficiency):
        if len(table) and "valid_peak_memory_runs" in table:
            memory_valid += int(table.valid_peak_memory_runs.sum())
    requirement_5 = bool(
        broad.get("robustness", {}).get("complete")
        and shift.get("complete")
        and len(remedy_efficiency)
        and len(confirmation_efficiency)
        and len(robustness_rank)
        and memory_valid > 0
    )
    requirement_6 = bool(
        text_contains(
            ROOT / "THEORY_DAY3.md",
            ["k-fac relationship", "prior art", "anchor canonicalization theorem"],
        )
        and text_contains(
            ROOT / "RELATED_WORK_DAY3.md",
            ["vectoradam", "grinsztajn", "shampoo / soap", "canonicalization"],
        )
    )
    requirement_7 = bool(
        text_contains(
            ROOT / "BROAD_BENCHMARK_REPORT.md",
            [
                "final iclr verdict",
                "systematic empirical phenomenon",
                "not a new invariant optimizer",
                "natural encodings",
            ],
        )
        and text_contains(ROOT / "REPORT_DAY3.md", ["broad benchmark report"])
    )
    verification_ok = bool(
        verification.get("pytest", {}).get("exit_code") == 0
        and verification.get("py_compile", {}).get("exit_code") == 0
        and verification.get("audits", {}).get("broad") is True
        and verification.get("audits", {}).get("extension") is True
        and verification.get("audits", {}).get("distribution_shift") is True
    )
    requirements = {
        "1_broad_30_dataset_prospective_evidence": {
            "passes": requirement_1,
            "evidence": {
                "original_phase1_complete": broad.get("phase1", {}).get("complete"),
                "extension_complete": extension.get("complete"),
                "combined_design": combined.get("design"),
            },
        },
        "2_models_and_strong_preprocessing": {
            "passes": requirement_2,
            "evidence": {
                "required_models": sorted(required_models),
                "confirmation_models": sorted(confirmation_models),
                "preprocessing_representations": sorted(
                    required_representations.intersection(preprocessing_representations)
                ),
                "confirmation_complete": broad.get("confirmation", {}).get("complete"),
            },
        },
        "3_optimizer_and_canonicalization_comparisons": {
            "passes": requirement_3,
            "evidence": {"phase1_remedies": sorted(phase_remedies)},
        },
        "4_natural_encodings": {
            "passes": requirement_4,
            "evidence": {
                "pairs": len(natural),
                "datasets": int(natural.dataset.nunique()) if len(natural) else 0,
                "equivalence_max_error": natural_error,
            },
        },
        "5_compute_rank_and_distribution_shift": {
            "passes": requirement_5,
            "evidence": {
                "robustness_complete": broad.get("robustness", {}).get("complete"),
                "distribution_shift_complete": shift.get("complete"),
                "valid_peak_memory_runs": memory_valid,
                "rank_summary_rows": len(robustness_rank),
            },
        },
        "6_theory_and_prior_art_boundary": {
            "passes": requirement_6,
            "evidence": ["THEORY_DAY3.md", "RELATED_WORK_DAY3.md"],
        },
        "7_systematic_framing_and_iclr_verdict": {
            "passes": requirement_7,
            "evidence": ["BROAD_BENCHMARK_REPORT.md", "REPORT_DAY3.md"],
        },
        "final_verification": {
            "passes": verification_ok,
            "evidence": verification,
        },
    }
    complete = all(item["passes"] for item in requirements.values())
    payload = {"requirements": requirements, "complete": complete}
    output = RESULTS / "day3_goal_completion_audit.json"
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if complete else 1)


if __name__ == "__main__":
    main()
