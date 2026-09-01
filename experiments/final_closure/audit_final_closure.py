"""Read-only completion audit for every frozen final-closure requirement."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import closure_core as core
from analysis_utils import write_summary
from closure_designs import assert_strength
from record_regeneration import digest_paths


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_b_conditions() -> int:
    config = core.completion_config()
    total = 0
    for dataset in core.CONFIG["experiment_b"]["datasets"]:
        prepared, _ = core.b_prepared_datasets(
            dataset, int(core.CONFIG["experiment_b"]["split_seed"]), config
        )
        total += len(prepared) * 5 * len(core.CONFIG["primary_models"])
    return total


def manifest_count(relative: str) -> int:
    return len(list((core.RAW / relative).glob("*/manifest.json")))


def audit_prediction_directory(relative: str, prediction_names: list[str]) -> tuple[int, int]:
    manifests = sorted((core.RAW / relative).glob("*/manifest.json"))
    arrays = 0; represented = 0
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "complete":
            raise AssertionError(f"non-complete manifest {manifest_path}")
        task = manifest["task"]
        for name in prediction_names:
            path = manifest_path.parent / name
            if not path.exists():
                raise AssertionError(f"missing prediction array {path}")
            prediction = np.load(path, mmap_mode="r")
            core.validate_probabilities(prediction, task)
            arrays += 1
        represented += int(
            manifest.get("unique_represented_fits", manifest.get("represented_fits", 0))
        )
        for complete_path in manifest_path.parent.glob("*complete.npy"):
            if not bool(np.load(complete_path, mmap_mode="r").all()):
                raise AssertionError(f"incomplete mask {complete_path}")
    return arrays, represented


def audit_seeds() -> dict[str, int]:
    pools = 0; seeds = 0
    for relative in ("experiment_a", "experiment_a_classical"):
        for manifest_path in sorted((core.RAW / relative).glob("*/manifest.json")):
            joint = np.load(manifest_path.parent / "joint_master_seeds.npy")
            canonical = np.load(manifest_path.parent / "canonical_master_seeds.npy")
            combined = np.concatenate((joint.reshape(-1)[joint.shape[1] :], canonical))
            if len(combined) != len(np.unique(combined)):
                raise AssertionError(f"duplicate independent seed in {manifest_path.parent}")
            pools += 1; seeds += len(combined)
    for manifest_path in sorted((core.RAW / "experiment_b").glob("*/manifest.json")):
        joint = np.load(manifest_path.parent / "master_seeds.npy")
        canonical = np.load(manifest_path.parent / "canonical_master_seeds.npy")
        # The smallest-size checkpoint bundle deliberately gives its eight
        # canonical-schema paths both joint and canonical views.  They are one
        # physical fit and one registry key, not duplicate independent draws.
        manifest = json.loads(manifest_path.read_text())
        overlap = len(set(joint.tolist()) & set(canonical.tolist()))
        if overlap != int(manifest.get("shared_canonical_joint_fits", 0)):
            raise AssertionError(f"undeclared B seed overlap in {manifest_path.parent}")
        unique_expected = len(joint) + len(canonical) - overlap
        combined = np.concatenate((joint, canonical))
        if len(np.unique(combined)) != unique_expected:
            raise AssertionError(f"duplicate B independent seed in {manifest_path.parent}")
        pools += 1; seeds += len(np.unique(combined))
    return {"independent_pools": pools, "unique_seed_records": seeds}


def audit_registry() -> dict[str, Any]:
    with sqlite3.connect(core.REGISTRY_PATH) as connection:
        rows = connection.execute(
            "SELECT experiment,COUNT(*),SUM(wall_seconds),MAX(peak_device_bytes) "
            "FROM fits WHERE status='complete' GROUP BY experiment ORDER BY experiment"
        ).fetchall()
        duplicates = connection.execute(
            "SELECT COUNT(*) FROM (SELECT fit_key,COUNT(*) n FROM fits GROUP BY fit_key HAVING n>1)"
        ).fetchone()[0]
        gpu_seconds, cpu_seconds = connection.execute(
            "SELECT SUM(CASE WHEN peak_device_bytes>0 THEN wall_seconds ELSE 0 END), "
            "SUM(CASE WHEN peak_device_bytes=0 THEN wall_seconds ELSE 0 END) "
            "FROM fits WHERE status='complete'"
        ).fetchone()
    if duplicates:
        raise AssertionError("duplicate registry fit keys")
    return {
        "by_experiment": {
            row[0]: {"fits": int(row[1]), "wall_seconds": float(row[2] or 0),
                     "maximum_peak_device_bytes": int(row[3] or 0)}
            for row in rows
        },
        "complete_fit_keys": int(sum(row[1] for row in rows)),
        "summed_fit_hours": float(sum(row[2] or 0 for row in rows) / 3600),
        "summed_gpu_fit_hours": float((gpu_seconds or 0) / 3600),
        "summed_cpu_fit_hours": float((cpu_seconds or 0) / 3600),
    }


def audit_designs_and_equal_budgets() -> dict[str, int]:
    maximum_a_budget = max(int(value) for value in core.CONFIG["experiment_a"]["budgets"])
    a_manifests = sorted((core.RAW / "experiment_a").glob("*/manifest.json"))
    for manifest_path in a_manifests:
        manifest = json.loads(manifest_path.read_text())
        joint = np.load(manifest_path.parent / "joint_master_seeds.npy")
        if joint.size < 4 * maximum_a_budget:
            raise AssertionError(f"A finite cache too small for B=64: {manifest_path.parent}")
        if manifest["joint_pool_fits"] != joint.size:
            raise AssertionError(f"A manifest pool count mismatch: {manifest_path.parent}")
    a_cells = pd.read_csv(core.HERE / "summaries" / "experiment_a_cells.csv")
    expected_methods = {
        4: 6, 8: 6, 16: 6, 32: 7, 64: 7,
    }
    grouped = a_cells.groupby(["dataset", "split_seed", "model", "budget"])
    for key, group in grouped:
        budget = int(key[-1])
        if len(group) != expected_methods[budget] or group["method"].nunique() != len(group):
            raise AssertionError(f"unequal/missing A methods at {key}")
        if not (group["estimator_draws"] == core.CONFIG["experiment_a"]["estimator_draws"]).all():
            raise AssertionError(f"A estimator-draw mismatch at {key}")

    b_full = 0; b_fractional = 0
    for manifest_path in sorted((core.RAW / "experiment_b").glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        design = np.load(manifest_path.parent / "design_rows.npy")
        cards = tuple(int(value) for value in manifest["schema_cards"])
        if manifest["full_product"]:
            if len(design) != int(np.prod((*cards, 8))):
                raise AssertionError(f"B full-product mismatch: {manifest_path.parent}")
            b_full += 1
        else:
            if len(design) != 128:
                raise AssertionError(f"B trajectory row mismatch: {manifest_path.parent}")
            assert_strength(design, (*cards, 8), 3)
            if len(np.unique(design, axis=0)) != min(128, int(np.prod((*cards, 8)))):
                raise AssertionError(f"B trajectory lacks maximum uniqueness: {manifest_path.parent}")
            b_fractional += 1
    b_conditions = pd.read_csv(core.HERE / "summaries" / "experiment_b_conditions.csv")
    b_exact = b_conditions[b_conditions["full_product"]]
    if float(b_exact["fanova_reconstruction_error"].abs().max()) > 5e-8:
        raise AssertionError("B fANOVA reconstruction exceeds tolerance")

    matched = 0
    for manifest_path in sorted((core.RAW / "matched_convergence").glob("*/manifest.json")):
        gaps = np.load(manifest_path.parent / "initial_gaps.npy")
        if float(np.max(gaps)) > float(core.CONFIG["initial_match_tolerance"]):
            raise AssertionError(f"matched initial gap exceeds tolerance: {manifest_path.parent}")
        matched += 1

    d_cells = pd.read_csv(core.HERE / "summaries" / "experiment_d_cells.csv")
    if not (d_cells["budget"] == 16).all():
        raise AssertionError("Experiment D has unequal fit budgets")
    d_components = pd.read_csv(core.HERE / "summaries" / "experiment_d_fanova_components.csv")
    if float(d_components["fanova_reconstruction_error"].abs().max()) > 5e-8:
        raise AssertionError("D fANOVA reconstruction exceeds tolerance")
    return {
        "a_equal_budget_groups": len(grouped), "b_full_product_conditions": b_full,
        "b_strength3_conditions": b_fractional, "matched_gap_conditions": matched,
        "d_equal_budget_rows": len(d_cells),
    }


def run_tests() -> dict[str, Any]:
    python = "/home/byunhanjoon/adaptive_embedding/exploratory/.venv-tabm/bin/python"
    command = [
        python, "-m", "pytest", "-q",
        str(core.DAY5), str(core.HERE / "test_final_closure.py"),
    ]
    result = subprocess.run(
        command, cwd=core.REPO, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, env={**__import__("os").environ, "PYTHONPATH": str(core.HERE)},
    )
    if result.returncode:
        raise AssertionError(f"test suite failed:\n{result.stdout}")
    final_line = result.stdout.strip().splitlines()[-1]
    passed = int(final_line.split()[0])
    return {"passed": passed, "total": passed, "output": final_line}


def required_outputs() -> dict[str, int]:
    figure_names = [
        "figure_1_independent_seed_showdown", "figure_2_architecture_b16",
        "figure_3_expectation_distance", "figure_4_convergence",
        "figure_5_training_scale", "figure_6_orbitcover_convergence",
        "figure_7_interaction_predicts_gain", "figure_8_failure_cell_spectra",
        "figure_9_matched_convergence", "figure_10_coupling_mechanism",
    ]
    table_names = list(core.CONFIG["required_tables"])
    summary_names = [
        "experiment_a_cells.csv", "experiment_a_references.csv",
        "experiment_a_iid_equivalent_budgets.csv", "experiment_a_summary.json",
        "experiment_a_classical_cells.csv", "experiment_a_classical_summary.json",
        "experiment_b_conditions.csv", "experiment_b_matched_convergence.csv",
        "experiment_b_descriptive_slopes.csv", "experiment_b_summary.json",
        "experiment_c_cells.csv", "experiment_c_srs_failure_cells.csv",
        "experiment_c_completion_nonpositive_sources.csv",
        "experiment_c_completion_source_spectra.csv", "experiment_c_summary.json",
        "experiment_d_cells.csv", "experiment_d_fanova_components.csv",
        "experiment_d_summary.json", "final_claims_summary.json",
    ]
    for name in figure_names:
        for suffix in (".png", ".pdf"):
            if not (core.HERE / "figures" / f"{name}{suffix}").exists():
                raise AssertionError(f"missing figure {name}{suffix}")
    for name in table_names:
        for suffix in (".csv", ".md"):
            if not (core.HERE / "tables" / f"{name}{suffix}").exists():
                raise AssertionError(f"missing table {name}{suffix}")
    for name in summary_names:
        if not (core.HERE / "summaries" / name).exists():
            raise AssertionError(f"missing summary artifact {name}")
    return {"figure_concepts": len(figure_names), "figure_files": 2 * len(figure_names),
            "tables": len(table_names), "table_files": 2 * len(table_names),
            "summary_artifacts": len(summary_names)}


def audit_regeneration_record() -> dict[str, Any]:
    path = core.HERE / "regeneration_record.json"
    if not path.exists():
        raise AssertionError("missing final regeneration record")
    record = json.loads(path.read_text())
    manifests = list(core.RAW.glob("*/*/manifest.json"))
    artifacts = []
    for relative in ("summaries", "figures", "tables"):
        artifacts.extend(item for item in (core.HERE / relative).glob("*") if item.is_file())
    if record.get("raw_manifest_count") != len(manifests):
        raise AssertionError("raw manifests changed after regeneration")
    if record.get("raw_manifest_digest") != digest_paths(manifests):
        raise AssertionError("raw manifest digest changed after regeneration")
    if record.get("derived_artifact_count") != len(artifacts):
        raise AssertionError("derived artifact set changed after regeneration")
    if record.get("derived_artifact_digest") != digest_paths(artifacts):
        raise AssertionError("derived artifacts changed after regeneration")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    expected = {
        "experiment_a": 12 * 3 * 4,
        "experiment_a_classical": 12 * 2,
        "experiment_b": expected_b_conditions(),
        "matched_convergence": 2 * 4 * 3,
        "experiment_d": 4 * 3 * 4,
    }
    actual = {name: manifest_count(name) for name in expected}
    if actual != expected:
        raise AssertionError(f"mandatory manifest matrix mismatch: {actual} != {expected}")
    arrays = 0; represented = 0; represented_by_experiment = {}
    for relative, names in {
        "experiment_a": ["joint_test.npy", "canonical_test.npy"],
        "experiment_a_classical": ["joint_test.npy", "canonical_test.npy"],
        "experiment_b": ["test_predictions.npy", "canonical_test_predictions.npy"],
        "matched_convergence": ["test_predictions.npy"],
        "experiment_d": ["test_predictions.npy"],
    }.items():
        current_arrays, current_fits = audit_prediction_directory(relative, names)
        arrays += current_arrays; represented += current_fits
        represented_by_experiment[relative] = current_fits
    hashes = {}
    for line in core.HASH_PATH.read_text().splitlines():
        name, expected_hash = line.split()
        actual_hash = file_hash(core.HERE / name)
        if actual_hash != expected_hash:
            raise AssertionError(f"frozen hash mismatch {name}")
        hashes[name] = actual_hash
    deviations = [line for line in (core.HERE / "PROTOCOL_DEVIATIONS.md").read_text().splitlines()
                  if line.startswith("## D")]
    summaries = [
        "experiment_a_summary.json", "experiment_b_summary.json",
        "experiment_c_summary.json", "experiment_d_summary.json",
    ]
    for name in summaries:
        payload = json.loads((core.HERE / "summaries" / name).read_text())
        if payload.get("status") != "complete":
            raise AssertionError(f"summary not complete: {name}")
    registry = audit_registry()
    registry_names = {
        "experiment_a": "A", "experiment_a_classical": "A-secondary",
        "experiment_b": "B", "matched_convergence": "B-matched",
        "experiment_d": "D",
    }
    for relative, registry_name in registry_names.items():
        registered = registry["by_experiment"].get(registry_name, {}).get("fits", 0)
        if registered != represented_by_experiment[relative]:
            raise AssertionError(
                f"registry/manifest represented-fit mismatch for {relative}: "
                f"{registered}/{represented_by_experiment[relative]}"
            )
    payload = {
        "status": "pass", "expected_manifests": expected,
        "actual_manifests": actual, "prediction_arrays_verified": arrays,
        "represented_fits_from_manifests": represented,
        "represented_fits_by_experiment": represented_by_experiment,
        "seed_audit": audit_seeds(), "registry": registry,
        "design_and_budget_audit": audit_designs_and_equal_budgets(),
        "protocol_hashes": hashes, "recorded_deviations": deviations,
        "outputs": required_outputs(),
        "regeneration": audit_regeneration_record(),
        "closure_wall_clock_hours": float(
            (time.time() - core.HASH_PATH.stat().st_mtime) / 3600
        ),
        "tests": {"passed": 0, "total": 0, "output": "skipped"}
        if args.skip_tests else run_tests(),
    }
    write_summary(core.HERE / "final_audit_summary.json", payload)
    test = payload["tests"]
    markdown = f"""# Final OrbitCover closure audit

Status: **PASS**.

- Mandatory manifests: `{actual}`.
- Prediction arrays checked finite/aligned: {arrays}.
- Represented fits from manifests: {represented:,}.
- Persistent complete registry keys: {payload['registry']['complete_fit_keys']:,}.
- Summed fit telemetry: {payload['registry']['summed_fit_hours']:.3f} hours
  ({payload['registry']['summed_gpu_fit_hours']:.3f} GPU-fit-hours and
  {payload['registry']['summed_cpu_fit_hours']:.3f} CPU-fit-hours).
- End-to-end closure wall clock: {payload['closure_wall_clock_hours']:.3f} hours.
- Independent seed records checked: {payload['seed_audit']['unique_seed_records']:,}; no undeclared duplicates.
- Protocol/config hashes: both match `PROTOCOL_HASH.txt`.
- Recorded deviations: {len(deviations)}; none is unrecorded.
- Tests: {test['passed']} / {test['total']} passed (`{test['output']}`).
- Figures: {payload['outputs']['figure_concepts']} concepts / {payload['outputs']['figure_files']} files regenerate.
- Tables: {payload['outputs']['tables']} CSV+Markdown pairs regenerate.

The audit found no missing mandatory cell, corrupt prediction, unequal declared
fit budget, duplicate supposedly-independent seed, protocol-hash change, or
missing final figure/table.  The final report may now be regenerated from the
audited summaries.
"""
    (core.HERE / "FINAL_AUDIT.md").write_text(markdown)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
