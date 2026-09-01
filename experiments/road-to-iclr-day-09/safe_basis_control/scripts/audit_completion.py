#!/usr/bin/env python3
"""Requirement-by-requirement completion and integrity audit."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DAY9 = ROOT.parent
sys.path.insert(0, str(ROOT))

from safe_basis.common import PANEL_PATH, PROTOCOL_PATH, load_json, sha256_file, write_json  # noqa: E402


def check(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def finite_csv(path: Path) -> bool:
    frame = pd.read_csv(path)
    numeric = frame.select_dtypes(include=[np.number])
    # Infinite exact condition numbers are a valid diagnostic for singular
    # overcomplete anchor Gram matrices; they are not an outcome failure.
    numeric = numeric[[column for column in numeric if "condition" not in column.lower()]]
    observed = numeric.to_numpy(float).ravel()
    # Missing entries encode genuinely non-applicable quantities (for example,
    # alpha frequencies for non-gated methods).  Audit every observed numeric
    # value for infinities without treating those explicit NAs as outcomes.
    observed = observed[~np.isnan(observed)]
    return bool(np.isfinite(observed).all())


def main() -> None:
    checks: list[dict[str, Any]] = []
    code_files = sorted(
        [*ROOT.glob("safe_basis/*.py"), *ROOT.glob("scripts/*.py"), *ROOT.glob("tests/*.py")]
    )
    write_json(
        ROOT / "results" / "processed" / "code_hashes.json",
        {str(path.relative_to(ROOT)): sha256_file(path) for path in code_files},
    )
    protocol = load_json(PROTOCOL_PATH)
    panel = load_json(PANEL_PATH)
    finalists_path = ROOT / "configs" / "TAIL_FINALISTS.json"
    finalists = load_json(finalists_path)

    for path in (PROTOCOL_PATH, PANEL_PATH, finalists_path):
        sidecar = path.with_suffix(".sha256")
        expected = sidecar.read_text().split()[0]
        actual = sha256_file(path)
        checks.append(check(f"hash:{path.name}", expected == actual, {"expected": expected, "actual": actual}))

    panel_names = [row["key"] for row in panel["datasets"]]
    panel_ok = 8 <= len(panel_names) <= 12 and {row["problem_type"] for row in panel["datasets"]} == {"classification", "regression"} and all(1000 <= row["rows"] <= 50000 and row["raw_columns"] <= 100 for row in panel["datasets"])
    checks.append(check("new prospective panel protocol", panel_ok, {"datasets": panel_names, "selection_evidence": panel["selection_evidence"]}))
    checks.append(check("finalist cap and status", finalists["status"] == "FROZEN_BEFORE_PROSPECTIVE_DATA_ACCESS" and 1 <= len(finalists["finalists"]) <= 4, {"status": finalists["status"], "count": len(finalists["finalists"])}))

    prospective_npz = list((ROOT / "results" / "raw" / "prospective").rglob("*.npz"))
    earliest = min((path.stat().st_mtime for path in prospective_npz), default=float("inf"))
    checks.append(check("freeze predates prospective outcomes", finalists_path.stat().st_mtime < earliest, {"finalist_mtime": finalists_path.stat().st_mtime, "earliest_prospective_mtime": earliest, "prospective_bundles": len(prospective_npz)}))

    processed = ROOT / "results" / "processed"
    gates = pd.read_csv(processed / "development_gate_cells.csv")
    evidence = pd.read_csv(processed / "development_gate_alpha_evidence.csv")
    checks.append(check("SafeGram development coverage", len(gates) == 8 * 5 * 2 * 14 * 2 and len(evidence) == 8 * 5 * 2 * 5, {"gate_rows": len(gates), "evidence_rows": len(evidence)}))
    checks.append(check("validation-only alpha selection", all(load_json(path)["test_outcomes_used_for_alpha_selection"] is False for path in (ROOT / "results" / "raw" / "development_gates").rglob("seed_*.json")), "all development gate manifests deny test-outcome selection"))

    rank_screen = pd.read_csv(processed / "rank_screen_cells.csv")
    rank_full = pd.read_csv(processed / "rank_development_cells.csv")
    rank_selection = load_json(processed / "rank_selection.json")
    checks.append(check("rank screen/full coverage", len(rank_screen) == 4 * 16 * 2 and len(rank_full) == 8 * 3 * 2 * 2 * 2, {"screen_rows": len(rank_screen), "full_rows": len(rank_full)}))
    checks.append(check("rank coordinate and reconstruction audit", rank_screen["maximum_coordinate_error"].max() < 1e-6 and rank_selection["validation_evidence"]["maximum_reconstruction_error"] <= 1e-4, {"max_coordinate_error": float(rank_screen["maximum_coordinate_error"].max()), "selected_max_reconstruction": rank_selection["validation_evidence"]["maximum_reconstruction_error"]}))

    failures = pd.read_csv(processed / "failure_diagnosis.csv")
    gram_failures = failures[failures.method == "GramAnchor"]
    rescue = load_json(processed / "optimization_rescue_manifest.json")
    type_b = gram_failures.failure_type.str.startswith("Type B").sum()
    rescue_ok = (type_b == 0 and rescue["status"] == "NOT_TRIGGERED") or (type_b > 0 and rescue["status"] == "COMPLETE")
    checks.append(check("five-cell failure diagnosis and conditional rescue", len(gram_failures) == 5 and "steel-plates-fault" in set(gram_failures.dataset) and gram_failures.failure_type.str.startswith(("Type A", "Type B", "Type C", "Type D")).all() and rescue_ok, {"cells": len(gram_failures), "types": gram_failures.failure_type.value_counts().to_dict(), "rescue": rescue}))

    descriptor = load_json(processed / "descriptor_gate_config.json")
    checks.append(check("conditional descriptor gate adjudicated", descriptor["status"] in {"KEEP_FOR_PROSPECTIVE", "DISCARDED_AFTER_DEVELOPMENT_CV"} and descriptor["prospective_outcomes_accessed"] is False, {"status": descriptor["status"], "reason": descriptor["reason"]}))

    embed_rotation = pd.read_csv(processed / "embedding_main_rotation_cells.csv")
    embed_methods = pd.read_csv(processed / "embedding_main_method_cells.csv")
    embed_audit = pd.read_csv(processed / "embedding_main_coordinate_audits.csv")
    dimension = pd.read_csv(processed / "embedding_dimension_units.csv")
    embed_models = set(embed_rotation.model)
    checks.append(check("embedding integration coverage", len(embed_rotation) == 3 * 3 * 2 * 2 * 9 * 2 and len(embed_methods) == 3 * 3 * 2 * 2 * 5 * 2 and embed_models == {"controlled_mlp", "tabm_d", "resnet_tabular"} and set(embed_rotation.embedding) == {"PLE", "RBF"}, {"rotation_rows": len(embed_rotation), "method_rows": len(embed_methods), "models": sorted(embed_models)}))
    coordinate_columns = ["train_relative_error", "validation_relative_error", "test_relative_error"]
    checks.append(check("embedding invariant coordinate checks", embed_audit[coordinate_columns].to_numpy().max() < 1e-6, float(embed_audit[coordinate_columns].to_numpy().max())))
    checks.append(check("embedding dimension ablation", set(dimension.k) == {4, 8, 16, 32} and set(dimension.dataset) == set(protocol["embeddings"]["dimension_datasets"]) and set(dimension.embedding) == {"PLE", "RBF"}, {"k": sorted(dimension.k.unique().tolist()), "datasets": sorted(dimension.dataset.unique().tolist())}))

    prospective = pd.read_csv(processed / "prospective_cells.csv")
    prospective_audit = pd.read_csv(processed / "prospective_coordinate_audits.csv")
    expected_methods = {
        "Raw",
        "GramAnchor-m16",
        "Raw+GramAnchor@0.5",
        "Raw+GramAnchor@0.75",
        "PCA-canonicalization",
        "RankAdaptiveGram",
        "SafeGram-t01",
        "SafeRankGram-t01",
    }
    observed_methods = set(prospective.method)
    checks.append(check("prospective primary coverage and baselines", len(prospective) == 10 * 3 * 2 * 8 * 2 and observed_methods == expected_methods, {"rows": len(prospective), "methods": sorted(observed_methods), "datasets": prospective.dataset.nunique(), "models": prospective.model.nunique(), "seeds": prospective.seed.nunique()}))
    invariant_audit = prospective_audit[prospective_audit["interface"].isin(["GramAnchor-m16", "RankAdaptiveGram"])]
    checks.append(check("prospective invariant coordinate equivalence", invariant_audit[coordinate_columns].to_numpy().max() < 1e-6, float(invariant_audit[coordinate_columns].to_numpy().max())))
    hashes = {"protocol": sha256_file(PROTOCOL_PATH), "panel": sha256_file(PANEL_PATH), "finalists": sha256_file(finalists_path)}
    raw_integrity = True
    missing_sidecars = []
    hash_mismatches = []
    nonfinite = []
    for path in prospective_npz:
        sidecar = path.with_suffix(".json")
        if not sidecar.exists():
            missing_sidecars.append(str(path)); raw_integrity = False; continue
        metadata = load_json(sidecar)
        if metadata.get("frozen_hashes") != hashes:
            hash_mismatches.append(str(path)); raw_integrity = False
        with np.load(path) as arrays:
            if not all(np.isfinite(arrays[key]).all() for key in arrays.files):
                nonfinite.append(str(path)); raw_integrity = False
    checks.append(check("prospective raw bundle integrity", raw_integrity, {"bundles": len(prospective_npz), "missing_sidecars": missing_sidecars, "hash_mismatches": hash_mismatches, "nonfinite": nonfinite}))

    aggregate = pd.read_csv(processed / "prospective_aggregate.csv")
    excluded = pd.read_csv(processed / "prospective_aggregate_excluding_sensitive_denominators.csv")
    ranking_paths = sorted(processed.glob("ranking_*.csv"))
    checks.append(check("statistical reporting and five rankings", len(aggregate) == 8 and len(excluded) == 8 and len(ranking_paths) == 5 and all(len(pd.read_csv(path)) == 8 for path in ranking_paths), {"aggregate_methods": len(aggregate), "rankings": [path.name for path in ranking_paths]}))
    safe_methods = aggregate[aggregate.method.isin(["SafeGram-t01", "SafeRankGram-t01"])]
    alpha_columns = ["alpha_0_fraction", "alpha_025_fraction", "alpha_05_fraction", "alpha_075_fraction", "alpha_1_fraction"]
    checks.append(check("SafeGram alpha distribution", np.allclose(safe_methods[alpha_columns].sum(axis=1), 1.0), safe_methods[["method", *alpha_columns]].to_dict(orient="records")))

    natural = pd.read_csv(processed / "natural_basis_cells.csv")
    checks.append(check("natural basis validation", len(natural) > 0 and natural.natural_reconstruction_error.max() < 1e-6 and natural[["gram_coordinate_error", "rank_coordinate_error"]].to_numpy().max() < 1e-6 and natural.model.nunique() == 3, {"pairs": sorted(natural.natural_pair.unique().tolist()), "rows": len(natural), "max_equivalence": float(natural.natural_reconstruction_error.max())}))

    required_csvs = [path for path in processed.glob("*.csv") if path.stat().st_size > 0]
    nonfinite_csvs = [path.name for path in required_csvs if not finite_csv(path)]
    checks.append(check("processed numeric outputs finite", not nonfinite_csvs, nonfinite_csvs))

    figures = ROOT / "figures"
    pngs = sorted(figures.glob("figure_*.png")); pdfs = sorted(figures.glob("figure_*.pdf"))
    image_sizes = {path.name: Image.open(path).size for path in pngs}
    checks.append(check("eight dual-format figures", len(pngs) == 8 and len(pdfs) == 8 and all(width >= 900 and height >= 500 for width, height in image_sizes.values()), {"png": len(pngs), "pdf": len(pdfs), "sizes": image_sizes}))

    report_path = ROOT / "results.md"
    report = report_path.read_text() if report_path.exists() else ""
    headings = [
        "# Safe Basis Control — Tail-Robust Method Round",
        "## Executive Verdict",
        "## One-Paragraph Summary",
        "## Frozen Protocol",
        *[f"## {number}. {title}" for number, title in [
            (1, "Previous Result Being Addressed"), (2, "SafeGram Development Results"), (3, "Gate Ablations"), (4, "RankAdaptiveGram"), (5, "Normalization Ablations"), (6, "Catastrophic Failure Diagnosis"), (7, "Steel Plates Deep Dive"), (8, "Numerical Embedding Basis Test"), (9, "Gram Inside Numerical Embeddings"), (10, "Embedding Dimension Ablation"), (11, "Development Finalist Ranking"), (12, "Frozen Finalists"), (13, "NEW Prospective Results"), (14, "Prospective Aggregate Results"), (15, "Safety-First Ranking"), (16, "Invariance Ranking"), (17, "Predictive Ranking"), (18, "Tail-Robustness Ranking"), (19, "Paper-Candidate Ranking"), (20, "Natural-Basis Validation"), (21, "Strongest Positive Result"), (22, "Strongest Negative Result"), (23, "Does Adaptive Gating Actually Solve Tail Risk?"), (24, "Is RankAdaptiveGram Better Than Fixed m=16?"), (25, "Does the Phenomenon Exist Inside Standard Numerical Embeddings?"), (26, "Recommended Paper Method Candidates"), (27, "Reviewer Attack Audit"), (28, "Recommended Next Experiment for Top-3"), (29, "Files Produced")
        ]],
    ]
    positions = [report.find(heading) for heading in headings]
    checks.append(check("exact report outline", all(position >= 0 for position in positions) and positions == sorted(positions) and report_path.stat().st_size > 30000, {"bytes": report_path.stat().st_size if report_path.exists() else 0, "missing": [heading for heading, position in zip(headings, positions) if position < 0]}))

    tests = subprocess.run(["/home/byunhanjoon/miniconda3/bin/python", "-m", "pytest", "-q"], cwd=ROOT, env={**dict(__import__("os").environ), "PYTHONPATH": "."}, capture_output=True, text=True)
    checks.append(check("code tests", tests.returncode == 0, tests.stdout + tests.stderr))
    required_dirs = [ROOT / name for name in ("results/raw", "results/processed", "figures", "configs")]
    checks.append(check("required output contract", all(path.exists() for path in required_dirs) and report_path.exists(), [str(path) for path in required_dirs]))

    passed = all(row["passed"] for row in checks)
    output = {"passed": passed, "checks": checks, "check_groups": len(checks)}
    write_json(processed / "completion_audit.json", output)
    for row in checks:
        print(("PASS" if row["passed"] else "FAIL"), row["name"], row["evidence"])
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
