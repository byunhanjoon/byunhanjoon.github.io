#!/usr/bin/env python3
"""Fail closed unless every required tournament artifact is complete and coherent."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tournament.common import (  # noqa: E402
    bd,
    development_specs,
    load_blocks,
    load_protocol,
    prospective_specs,
    sha256_file,
    write_json,
)


REQUIRED_HEADINGS = [
    "# Basis-Controlled Tabular Learning — Method Tournament",
    "## Executive Summary",
    "## Frozen Protocol",
    "## Previous Findings Treated as Fixed",
] + [f"## {index}. {title}" for index, title in enumerate([
    "Stage-1 Method Screening", "Optimizer Methods", "Optimizer Equivariance Audit",
    "Representation Methods", "Anchor / Rank Ablations", "Natural Equivalent Basis Results",
    "Hybrid Methods", "Equal-HPO Control", "Development Ranking", "Frozen Finalists",
    "NEW Prospective Results", "Prospective Rankings", "Method-by-Model Matrix",
    "Strongest Result", "Strongest Negative Result", "Failed Methods and Why",
    "Mechanistic Interpretation", "Reviewer Attack Audit", "Ranked Candidates for Human Decision",
    "Suggested Next Experiment for Each Top-3 Method", "Files Produced",
], start=1)]


def finite_csv(path: Path) -> bool:
    frame = pd.read_csv(path)
    numeric = frame.select_dtypes(include=[np.number])
    return bool(np.isfinite(numeric.to_numpy()).all())


def main() -> None:
    checks: dict[str, dict[str, object]] = {}

    def check(name: str, passed: bool, detail: object) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}

    protocol = load_protocol()
    config_path = ROOT / "configs" / "FINALIST_CONFIGS.json"
    sha_path = ROOT / "configs" / "FINALIST_CONFIGS.sha256"
    finalists = json.loads(config_path.read_text())
    expected_sha = sha_path.read_text().split()[0]
    check("finalist_sha", sha256_file(config_path) == expected_sha, expected_sha)
    check("finalist_cap", 1 <= len(finalists["finalists"]) <= 3, len(finalists["finalists"]))
    check("finalist_status", finalists["status"] == "FROZEN_BEFORE_PROSPECTIVE_DATA_ACCESS", finalists["status"])

    processed = ROOT / "results" / "processed"
    counts = {
        "stage1_cells": len([path for path in (processed / "stage1_cells").glob("*.csv") if "coordinate_audit" not in path.name]),
        "stage2_optimizer_cells": len(list((processed / "stage2_optimizer_cells").glob("*.csv"))),
        "stage2_representation_cells": len([path for path in (processed / "stage2_representation_cells").glob("*.csv") if "coordinate_audit" not in path.name]),
        "equal_hpo_trials": len(list((processed / "equal_hpo").glob("*__trials.csv"))),
        "anchor_cells": len(list((processed / "anchor_ablation").glob("*.csv"))),
        "mechanism_cells": len(list((processed / "mechanism").glob("*.csv"))),
        "natural_cells": len(list((processed / "natural_bases").glob("*.csv"))),
        "prospective_cells": len(list((processed / "prospective_cells").glob("*.csv"))),
        "condition_cells": len(list((processed / "condition_exploratory").glob("*.csv"))),
    }
    expected = {
        "stage1_cells": 6,
        "stage2_optimizer_cells": 36,
        "stage2_representation_cells": 90,
        "equal_hpo_trials": 36,
        "anchor_cells": 18,
        "mechanism_cells": 6,
        "natural_cells": 90,
        "prospective_cells": len(prospective_specs()) * len(protocol["model_seeds"]) * 5,
        "condition_cells": len(protocol["stage1_datasets"]) * 5,
    }
    check("cell_counts", counts == expected, {"actual": counts, "expected": expected})

    coordinate_files = list((processed / "stage2_representation_cells").glob("*coordinate_audit.csv"))
    coordinate_max = 0.0
    shape_ok = True
    for path in coordinate_files:
        frame = pd.read_csv(path)
        shape_ok = shape_ok and bool(frame["shape_match"].astype(bool).all())
        coordinate_max = max(
            coordinate_max,
            float(frame[["train_relative_error", "validation_relative_error", "test_relative_error"]].max().max()),
        )
    check("stage2_coordinate_audits", len(coordinate_files) == 90 and shape_ok and coordinate_max < 1e-8, {"files": len(coordinate_files), "max": coordinate_max})

    natural = pd.read_csv(processed / "natural_basis_units.csv")
    check(
        "natural_basis_contract",
        float(natural["reconstruction_error"].max()) < 1e-6 and float(natural["coordinate_error"].max()) < 1e-8,
        {"max_reconstruction": float(natural["reconstruction_error"].max()), "max_coordinate": float(natural["coordinate_error"].max())},
    )
    mechanism = pd.read_csv(processed / "mechanism_equivariance_verdict.csv")
    claimed = mechanism[mechanism["method"].isin(["BlockAdam", "MatrixAdam"])]
    check("optimizer_equivariance", bool(claimed["preserves_matched_equivalence"].astype(bool).all()), claimed.to_dict(orient="records"))

    prospective_files = list((processed / "prospective_cells").glob("*.csv"))
    prospective_method_ok = True
    for path in prospective_files:
        model = path.name.split("__", 1)[0]
        observed = set(pd.read_csv(path, usecols=["method"])["method"])
        required = {"Raw"} | {
            item["method_id"] for item in finalists["finalists"] if model in item["applicable_models"]
        }
        prospective_method_ok = prospective_method_ok and observed == required
    raw_prospective = list((ROOT / "results" / "raw" / "prospective").rglob("*.npz"))
    earliest_prospective = min((path.stat().st_mtime for path in raw_prospective), default=float("inf"))
    check("prospective_methods", prospective_method_ok, len(prospective_files))
    check("prospective_after_freeze", config_path.stat().st_mtime <= earliest_prospective, {"config_mtime": config_path.stat().st_mtime, "earliest_raw_mtime": earliest_prospective})
    expected_raw_bundles = (
        5 * len(prospective_specs()) * len(protocol["model_seeds"]) * 9
        + 5 * len(prospective_specs()) * len(protocol["model_seeds"]) * 2
        + 2 * len(prospective_specs()) * len(protocol["model_seeds"]) * 9
    )
    raw_bundle_ok = len(raw_prospective) == expected_raw_bundles
    prospective_coordinate_max = 0.0
    hash_ok = True
    for path in raw_prospective:
        with np.load(path) as arrays:
            raw_bundle_ok = raw_bundle_ok and all(
                np.isfinite(arrays[key]).all() for key in arrays.files
            )
        metadata = json.loads(path.with_suffix(".json").read_text())
        hash_ok = hash_ok and (
            metadata["frozen_hashes"]["finalist_configs_sha256"] == expected_sha
        )
        prospective_coordinate_max = max(
            prospective_coordinate_max,
            float(metadata.get("maximum_coordinate_error", 0.0)),
        )
    check(
        "prospective_raw_bundles",
        raw_bundle_ok and hash_ok and prospective_coordinate_max < 1e-8,
        {
            "actual": len(raw_prospective),
            "expected": expected_raw_bundles,
            "hashes_match": hash_ok,
            "max_coordinate_error": prospective_coordinate_max,
        },
    )

    maximum_condition = 1.0
    maximum_condition_reconstruction = 0.0
    specs = [
        spec for spec in development_specs(protocol)
        if spec["key"] in protocol["stage1_datasets"]
    ]
    for spec in specs:
        blocks = load_blocks(spec, protocol)
        reps = bd.build_primary_representations(blocks, int(protocol["orbit_members"]))
        for rep in (item for item in reps if item.variant == "condition_le_3_all"):
            for record in rep.metadata["equivalence"].values():
                maximum_condition = max(maximum_condition, float(record["condition_number"]))
                maximum_condition_reconstruction = max(
                    maximum_condition_reconstruction, float(record["reconstruction_error"])
                )
    check(
        "condition_transform_contract",
        maximum_condition <= 3.0 + 1e-8 and maximum_condition_reconstruction < 1e-6,
        {"max_condition": maximum_condition, "max_reconstruction": maximum_condition_reconstruction},
    )

    required_processed = [
        "stage1_summary.csv", "development_all_method_summary.csv", "development_all_ranking_A.csv",
        "development_all_ranking_B_pareto.csv", "development_all_ranking_C_predictive.csv",
        "development_all_ranking_D_score.csv", "equal_hpo_summary.csv", "anchor_ablation_summary.csv",
        "mechanism_audit_summary.csv", "natural_basis_summary.csv", "prospective_method_summary.csv",
        "prospective_ranking_A.csv", "prospective_ranking_B_pareto.csv",
        "prospective_ranking_C_predictive.csv", "prospective_ranking_D_score.csv",
        "prospective_method_model_matrix_wide.csv", "condition_exploratory_summary.csv",
    ]
    missing = [name for name in required_processed if not (processed / name).exists()]
    nonfinite = [name for name in required_processed if (processed / name).exists() and not finite_csv(processed / name)]
    check("processed_outputs", not missing and not nonfinite, {"missing": missing, "nonfinite": nonfinite})

    pngs = sorted((ROOT / "figures").glob("figure_*.png"))
    pdfs = sorted((ROOT / "figures").glob("figure_*.pdf"))
    image_sizes = {}
    images_ok = len(pngs) == 8 and len(pdfs) == 8
    for path in pngs:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image_sizes[path.name] = image.size
            images_ok = images_ok and image.width >= 800 and image.height >= 500
    check("figures", images_ok, {"png": len(pngs), "pdf": len(pdfs), "sizes": image_sizes})

    report_path = ROOT / "results.md"
    report_text = report_path.read_text() if report_path.exists() else ""
    actual_headings = [line for line in report_text.splitlines() if line.startswith("#")]
    missing_headings = [heading for heading in REQUIRED_HEADINGS if heading not in actual_headings]
    ordered = all(
        report_text.find(first) < report_text.find(second)
        for first, second in zip(REQUIRED_HEADINGS, REQUIRED_HEADINGS[1:])
    )
    check("results_report", report_path.exists() and not missing_headings and ordered, {"missing_headings": missing_headings, "bytes": len(report_text.encode())})

    check("directory_contract", all((ROOT / name).is_dir() for name in ["results/raw", "results/processed", "figures", "configs"]), "required directories")
    passed = all(bool(value["passed"]) for value in checks.values())
    write_json(processed / "completion_audit.json", {"passed": passed, "checks": checks})
    for name, result in checks.items():
        print(f"{'PASS' if result['passed'] else 'FAIL'} {name}: {result['detail']}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
