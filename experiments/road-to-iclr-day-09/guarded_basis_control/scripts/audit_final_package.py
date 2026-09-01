#!/usr/bin/env python3
"""Fail-closed integrity audit for the final Guarded Basis Control package."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
PROCESSED = ROOT / "results" / "processed"
RAW_PROSPECTIVE = ROOT / "results" / "raw" / "prospective"
EXPECTED_FIGURES = {
    "figure_1_control_tail_pareto.png",
    "figure_2_tail_cdf.png",
    "figure_3_guardedgram_alpha_histogram.png",
    "figure_4_blockguard_fraction_tradeoff.png",
    "figure_5_embedding_dimension_scaling.png",
    "figure_6_default_vs_basis_search.png",
    "figure_7_efficiency_comparison.png",
    "figure_8_development_vs_prospective.png",
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: Any) -> None:
        checks.append({"name": name, "passes": bool(condition), "detail": detail})

    locked: dict[str, str] = {}
    for stem in (
        "PRIOR_DATASET_BLACKLIST",
        "GUARDED_PROSPECTIVE_PANEL",
        "GUARDED_PROTOCOL",
        "GUARDED_FINALISTS",
    ):
        path = CONFIGS / f"{stem}.json"
        expected = (CONFIGS / f"{stem}.sha256").read_text().split()[0]
        actual = digest(path)
        locked[stem] = actual
        check(f"lock_sha256:{stem}", actual == expected, {"expected": expected, "actual": actual})

    blacklist = load(CONFIGS / "PRIOR_DATASET_BLACKLIST.json")
    panel = load(CONFIGS / "GUARDED_PROSPECTIVE_PANEL.json")
    protocol = load(CONFIGS / "GUARDED_PROTOCOL.json")
    finalists = load(CONFIGS / "GUARDED_FINALISTS.json")
    panel_keys = [str(row["key"]) for row in panel["datasets"]]
    prior_keys = set(map(str, blacklist["datasets"]))
    check("prospective_panel_size", 10 <= len(panel_keys) <= 14, len(panel_keys))
    check("prospective_panel_unique", len(set(panel_keys)) == len(panel_keys), panel_keys)
    check("prospective_panel_not_prior", not (set(panel_keys) & prior_keys), sorted(set(panel_keys) & prior_keys))
    check("finalists_frozen_before_access", finalists["status"] == "FROZEN_BEFORE_GUARDED_PROSPECTIVE_DATA_ACCESS", finalists["status"])
    check("no_automatic_paper_winner", finalists["automatic_final_paper_winner"] is False, finalists["automatic_final_paper_winner"])
    check("at_most_four_finalists", finalists["finalist_count"] == len(finalists["finalists"]) <= 4, finalists["finalist_count"])

    finalist_hash = locked["GUARDED_FINALISTS"]
    models = list(map(str, protocol["general_models"]))
    seeds = list(map(int, protocol["prospective_seeds"]))
    expected_general = {(model, dataset, seed) for model in models for dataset in panel_keys for seed in seeds}
    actual_general: set[tuple[str, str, int]] = set()
    general_coordinate_max = 0.0
    general_cell_methods: set[str] = set()
    general_paths = sorted((RAW_PROSPECTIVE / "units").glob("*/*/seed_*.json"))
    for path in general_paths:
        payload = load(path)
        key = (str(payload["model"]), str(payload["dataset"]), int(payload["seed"]))
        actual_general.add(key)
        check(f"general_unit_complete:{path.relative_to(ROOT)}", payload.get("status") == "COMPLETE", payload.get("status"))
        check(f"general_unit_hash:{path.relative_to(ROOT)}", payload.get("finalist_sha256") == finalist_hash, payload.get("finalist_sha256"))
        check(
            f"general_validation_only:{path.relative_to(ROOT)}",
            payload.get("selection_split") == "validation_only" and payload.get("test_outcomes_used_for_selection") is False,
            {"selection_split": payload.get("selection_split"), "test_used": payload.get("test_outcomes_used_for_selection")},
        )
        audit = payload["block_selection"]["coordinate_audit"]
        general_coordinate_max = max(general_coordinate_max, float(audit["maximum_selected_block_relative_error"]))
        general_cell_methods.update(str(row["method"]) for row in payload["cells"])
        for row in payload["cells"]:
            for field in ("raw_disagreement", "method_disagreement", "disagreement_reduction", "raw_loss", "method_loss", "normalized_excess_risk"):
                check(f"finite:{path.relative_to(ROOT)}:{row['split']}:{row['method']}:{field}", math.isfinite(float(row[field])), row[field])
    check("general_unit_matrix_exact", actual_general == expected_general, {"expected": len(expected_general), "actual": len(actual_general), "missing": sorted(expected_general - actual_general), "extra": sorted(actual_general - expected_general)})
    check("general_coordinate_error_lt_1e-6", general_coordinate_max < 1e-6, general_coordinate_max)
    check(
        "general_required_methods",
        general_cell_methods == {"Raw", "Raw+Gram@0.5", "Raw+Gram@0.75", "PureGram", "SafeGram-t01", "SafeRankGram-t01", "GuardedGram-G2-g0p0-t01", "BlockGuard-Greedy-t01"},
        sorted(general_cell_methods),
    )

    embedding_models = list(map(str, finalists["embedding_models"]))
    expected_embedding = {(model, dataset, seed) for model in embedding_models for dataset in panel_keys for seed in seeds}
    actual_embedding: set[tuple[str, str, int]] = set()
    embedding_coordinate_max = 0.0
    embedding_paths = sorted((RAW_PROSPECTIVE / "embedding_units").glob("*/*/seed_*.json"))
    for path in embedding_paths:
        payload = load(path)
        actual_embedding.add((str(payload["model"]), str(payload["dataset"]), int(payload["seed"])))
        check(f"embedding_unit_complete:{path.relative_to(ROOT)}", payload.get("status") == "COMPLETE", payload.get("status"))
        check(f"embedding_unit_hash:{path.relative_to(ROOT)}", payload.get("finalist_sha256") == finalist_hash, payload.get("finalist_sha256"))
        check(
            f"embedding_validation_only:{path.relative_to(ROOT)}",
            payload.get("selection_split") == "validation_only" and payload.get("test_outcomes_used_for_selection") is False,
            {"selection_split": payload.get("selection_split"), "test_used": payload.get("test_outcomes_used_for_selection")},
        )
        embedding_coordinate_max = max(embedding_coordinate_max, float(payload["maximum_coordinate_error"]))
    check("embedding_unit_matrix_exact", actual_embedding == expected_embedding, {"expected": len(expected_embedding), "actual": len(actual_embedding), "missing": sorted(expected_embedding - actual_embedding), "extra": sorted(actual_embedding - expected_embedding)})
    check("embedding_coordinate_error_lt_1e-6", embedding_coordinate_max < 1e-6, embedding_coordinate_max)

    for name in (
        "guardedgram_stage1_manifest.json",
        "guardedgram_full_manifest.json",
        "blockguard_stage1_manifest.json",
        "blockguard_full_manifest.json",
        "dualview_stage1_manifest.json",
        "embedding_dimension_manifest.json",
        "embedding_full_manifest.json",
        "embedding_blockguard_manifest.json",
        "prospective_general_manifest.json",
        "prospective_embedding_manifest.json",
        "final_provenance.json",
    ):
        payload = load(PROCESSED / name)
        check(f"manifest_complete:{name}", str(payload.get("status", "")).startswith("COMPLETE"), payload.get("status"))

    general_units = pd.read_csv(PROCESSED / "prospective_general_units.csv")
    embedding_units = pd.read_csv(PROCESSED / "prospective_embedding_units.csv")
    check("general_aggregate_units", len(general_units) == len(panel_keys) * len(models) * 8, len(general_units))
    check("embedding_aggregate_units", len(embedding_units) == len(panel_keys) * len(embedding_models) * 4, len(embedding_units))
    natural = load(PROCESSED / "natural_basis_reuse_manifest.json")
    natural_max = max(float(natural["maximum_natural_reconstruction_error"]), float(natural["maximum_gram_coordinate_error"]), float(natural["maximum_rank_coordinate_error"]))
    check("natural_equivalence_lt_1e-6", natural_max < 1e-6 and natural["passes_exact_equivalence_threshold"] is True, natural_max)
    for name in ("embedding_dimension_coordinate_audits.csv", "embedding_full_coordinate_audits.csv"):
        frame = pd.read_csv(PROCESSED / name)
        check(f"coordinate_csv_lt_1e-6:{name}", float(frame.maximum_coordinate_error.max()) < 1e-6 and bool(frame.passes_1e_minus_6.all()), float(frame.maximum_coordinate_error.max()))

    freeze_ns = max((CONFIGS / "GUARDED_FINALISTS.json").stat().st_mtime_ns, (CONFIGS / "GUARDED_FINALISTS.sha256").stat().st_mtime_ns)
    raw_files = [path for path in RAW_PROSPECTIVE.rglob("*") if path.is_file()]
    earliest_ns = min(path.stat().st_mtime_ns for path in raw_files)
    check("freeze_precedes_every_prospective_artifact", earliest_ns > freeze_ns, {"freeze_ns": freeze_ns, "earliest_prospective_ns": earliest_ns})

    report = (ROOT / "results.md").read_text()
    numbered = [int(value) for value in re.findall(r"^## (\d+)\.", report, flags=re.MULTILINE)]
    check("report_exact_34_numbered_sections", numbered == list(range(1, 35)), numbered)
    check("report_disclaims_automatic_selection", "no final paper method is automatically selected" in report.lower(), None)
    figures = {path.name for path in (ROOT / "figures").glob("*.png")}
    check("exact_eight_required_figures", figures == EXPECTED_FIGURES, sorted(figures))

    failures = [row for row in checks if not row["passes"]]
    result = {
        "status": "COMPLETE" if not failures else "FAILED",
        "checks": len(checks),
        "failures": failures,
        "locks": locked,
        "general_units": len(actual_general),
        "embedding_units": len(actual_embedding),
        "maximum_general_coordinate_error": general_coordinate_max,
        "maximum_embedding_coordinate_error": embedding_coordinate_max,
        "maximum_natural_equivalence_error": natural_max,
    }
    (PROCESSED / "final_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if failures:
        raise RuntimeError(f"final package audit failed {len(failures)} checks; see final_audit.json")
    print(f"[audit] COMPLETE checks={len(checks)} general={len(actual_general)} embedding={len(actual_embedding)} figures={len(figures)}")


if __name__ == "__main__":
    main()
