"""Read-only structural and semantic integrity audit for Day-5 artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PANELS = (
    ("confirmation", "tier1_confirmation_config.json", "tier1_confirmation"),
    ("menu_repeat", "tier1_menu_repeat_config.json", "tier1_menu_repeat"),
    ("subsample_repeat", "tier1_subsample_repeat_config.json", "tier1_subsample_repeat"),
    ("openml_external", "openml_external_cover_config.json", "openml_external_cover"),
    ("openml_taskbalanced", "openml_taskbalanced_cover_config.json", "openml_taskbalanced_cover"),
    ("openml_multiclass", "openml_multiclass_cover_config.json", "openml_multiclass_cover"),
    ("openml_late_source", "openml_late_source_cover_config.json", "openml_late_source_cover"),
    ("openml_late_source_b", "openml_late_source_b_cover_config.json", "openml_late_source_b_cover"),
    ("openml_modern_model", "openml_modern_model_cover_config.json", "openml_modern_model_cover"),
    ("openml_modern_split_2901", "openml_modern_model_cover_config.json", "openml_modern_model_split_2026082901"),
    ("openml_modern_split_2911", "openml_modern_model_cover_config.json", "openml_modern_model_split_2026082911"),
    ("openml_modern_split_2921", "openml_modern_model_cover_config.json", "openml_modern_model_split_2026082921"),
    ("openml_late_source_c", "openml_late_source_c_cover_config.json", "openml_late_source_c_cover"),
    ("openml_late_source_c_split", "openml_late_source_c_split_cover_config.json", "openml_late_source_c_split_cover"),
)
SCREEN_MAP = {
    "strength2_confirmation": "strength2_confirmation_cells.csv",
    "strength2_openml_external": "strength2_openml_external_cells.csv",
    "strength2_openml_taskbalanced": "strength2_openml_taskbalanced_cells.csv",
    "strength2_openml_multiclass": "strength2_openml_multiclass_cells.csv",
}
HASH_FILES = (
    "THEORY_FOUNDATIONS.md", "PAPER_BLUEPRINT.md", "RECENT_LITERATURE_AUDIT.md",
    "tier1_confirmation_config.json", "openml_external_cover_config.json",
    "openml_taskbalanced_cover_config.json", "openml_multiclass_cover_config.json",
    "openml_late_source_cover_config.json", "openml_external_cover.py",
    "openml_late_source_b_cover_config.json",
    "MODERN_MODEL_EXTENSION_PROTOCOL.md", "openml_modern_model_cover_config.json",
    "EXPANDED_MODEL_SOURCE_PROTOCOL.md",
    "REPEATED_SPLIT_MODERN_PROTOCOL.md",
    "REPEATED_SPLIT_METRIC_SCOPE.md", "PARTITION_NUISANCE_SCALE_PROTOCOL.md",
    "ANTITHETIC_OPERATOR_BOUNDARY_PROTOCOL.md", "ANTITHETIC_CV_COMPARISON.md",
    "LATE_SOURCE_C_PROTOCOL.md", "openml_late_source_c_cover_config.json",
    "LATE_SOURCE_C_SPLIT_REPEAT_PROTOCOL.md", "openml_late_source_c_split_cover_config.json",
    "REVIEWER_ATTACK_AUDIT.md",
    "MATCHED_FUNCTION_PROTOCOL.md", "matched_function_config.json",
    "matched_function_control.py",
    "analyze_strength2_cover.py", "analyze_cross_quotient_selection.py",
    "analyze_cross_score_budget_frontier.py",
    "analyze_disjoint_pair_cross.py", "analyze_disjoint_pack64.py",
    "analyze_disjoint_pack_cross128.py", "analyze_mixed_resolvable_packing.py",
    "analyze_exhaustive128_control.py", "analyze_log_loss_support.py",
    "analyze_smoothed_log_packing.py",
    "analyze_smoothed_log_taylor.py",
    "analyze_late_source_extension.py", "analyze_combined_packing_sources.py",
    "SOURCE_SENSITIVITY_AUDIT.md", "analyze_source_sensitivity.py",
    "analyze_late_source_metric_scope.py",
    "analyze_timed_refit.py",
    "analyze_late_strength_failure.py",
    "analyze_combined_late_metric_sources.py",
    "analyze_modern_model_extension.py", "analyze_expanded_model_sources.py",
    "analyze_repeated_split_modern.py",
    "analyze_repeated_split_metric_scope.py", "analyze_partition_nuisance_scale.py",
    "analyze_antithetic_operator_boundary.py",
    "analyze_late_source_c_audit.py", "analyze_final_combined_sources.py",
    "analyze_source_c_operator_prediction.py",
    "analyze_source_c_two_split.py",
    "results/exact_panel_meta_summary.json",
    "results/cross_quotient_selection_summary.json",
    "results/cross_score_budget_frontier_summary.json",
    "results/disjoint_pair_cross_summary.json",
    "results/disjoint_pack64_summary.json",
    "results/disjoint_pack_cross128_summary.json",
    "results/mixed_resolvable_summary.json",
    "results/exhaustive128_control_summary.json",
    "results/log_loss_support_summary.json",
    "results/smoothed_log_packing_summary.json",
    "results/smoothed_log_taylor_summary.json",
    "results/late_source_extension_summary.json",
    "results/late_source_b_extension_summary.json",
    "results/combined_packing_source_summary.json",
    "results/combined_packing_source_sensitivity_summary.json",
    "results/late_source_metric_scope_summary.json",
    "results/late_source_b_metric_scope_summary.json",
    "results/timed_refit_summary.json",
    "results/late_strength_failure_summary.json",
    "results/combined_late_metric_source_summary.json",
    "results/modern_model_strength2_summary.json",
    "results/modern_model_extension_summary.json",
    "results/modern_model_metric_scope_summary.json",
    "results/modern_model_extension_audit_summary.json",
    "results/expanded_model_source_summary.json",
    "results/repeated_split_modern_summary.json",
    "results/repeated_split_metric_scope_summary.json",
    "results/partition_nuisance_scale_summary.json",
    "results/antithetic_operator_boundary_summary.json",
    "results/late_source_c_strength2_summary.json",
    "results/late_source_c_extension_summary.json",
    "results/late_source_c_audit_summary.json",
    "results/late_source_c_metric_scope_summary.json",
    "results/final_combined_source_summary.json",
    "results/source_c_operator_prediction_summary.json",
    "results/late_source_c_split_strength2_summary.json",
    "results/late_source_c_split_extension_summary.json",
    "results/late_source_c_split_audit_summary.json",
    "results/source_c_two_split_summary.json",
    "results/matched_function_summary.json",
    "FINAL_COMPLETION_PROTOCOL.md", "completion_config.json",
    "completion_environment.json",
    "TABPFN_COMPLETION_NOTES.md",
    "COMPLETION_PROTOCOL_DEVIATIONS.md", "PROGRAM_COMPLETION_MATRIX.md",
    "EXPERIMENT_LEDGER.md", "results.md", "DAY5_FINAL_REPORT.md", "README.md",
    "completion_neural_panel.py", "completion_classical_config.json",
    "completion_classical_panel.py", "completion_tabpfn.py",
    "completion_menu_size.py", "completion_row_order.py",
    "analyze_completion_panel.py", "analyze_completion_tabpfn.py",
    "make_completion_outputs.py",
    "results/completion_panel_summary.json",
    "results/completion_tabpfn_summary.json",
    "results/completion_menu_size_summary.json",
    "results/completion_row_order_summary.json",
    "results/completion_outputs_manifest.json",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    issues: list[str] = []
    panel_records = {}
    maximum_probability_sum_error = 0.0
    minimum_probability = float("inf")
    maximum_probability = float("-inf")
    tensors = fits = 0
    for panel, config_name, directory_name in PANELS:
        config = json.loads((HERE / config_name).read_text())
        directory = RESULTS / directory_name
        panel_tensors = panel_fits = 0
        for dataset in config["datasets"]:
            reference_val = reference_test = None
            reference_shape = None
            for model in config["models"]:
                stem = f"{dataset}__{model}"
                npz_path, json_path = directory / f"{stem}.npz", directory / f"{stem}.json"
                if not npz_path.exists() or not json_path.exists():
                    issues.append(f"missing pair: {panel}/{stem}")
                    continue
                manifest = json.loads(json_path.read_text())
                if manifest.get("status") != "complete":
                    issues.append(f"non-complete manifest: {panel}/{stem}")
                if manifest.get("full_product_verified") is False:
                    issues.append(f"full product not verified: {panel}/{stem}")
                with np.load(npz_path) as archive:
                    val = archive["validation_predictions"]
                    test = archive["test_predictions"]
                    val_y, test_y = archive["validation_y"], archive["test_y"]
                    shape = tuple(int(value) for value in val.shape[:4])
                    if tuple(test.shape[:4]) != shape:
                        issues.append(f"validation/test factor mismatch: {panel}/{stem}")
                    manifest_shape = manifest.get("shape")
                    if manifest_shape is None and "factor_shape" in manifest:
                        manifest_shape = [*manifest["factor_shape"], len(manifest["seeds"])]
                    if manifest_shape is not None and tuple(manifest_shape) != shape:
                        issues.append(f"manifest factor mismatch: {panel}/{stem}")
                    if reference_shape is not None and shape != reference_shape:
                        issues.append(f"candidate factor mismatch: {panel}/{dataset}")
                    reference_shape = shape
                    if reference_val is not None and not np.array_equal(val_y, reference_val):
                        issues.append(f"validation labels differ: {panel}/{dataset}")
                    if reference_test is not None and not np.array_equal(test_y, reference_test):
                        issues.append(f"test labels differ: {panel}/{dataset}")
                    reference_val, reference_test = val_y, test_y
                    for split, values in (("validation", val), ("test", test)):
                        if not np.isfinite(values).all():
                            issues.append(f"nonfinite predictions: {panel}/{stem}/{split}")
                        if values.shape[-1] > 1:
                            sums = values.astype(np.float64).sum(axis=-1)
                            maximum_probability_sum_error = max(
                                maximum_probability_sum_error,
                                float(np.max(np.abs(sums - 1))),
                            )
                            minimum_probability = min(minimum_probability, float(values.min()))
                            maximum_probability = max(maximum_probability, float(values.max()))
                    panel_fits += int(np.prod(shape))
                    panel_tensors += 1
        panel_records[panel] = {
            "configured_tensors": len(config["datasets"]) * len(config["models"]),
            "verified_tensors": panel_tensors,
            "represented_fits": panel_fits,
        }
        tensors += panel_tensors; fits += panel_fits

    matched_config = json.loads((HERE / "matched_function_config.json").read_text())
    matched_directory = RESULTS / "matched_function"
    matched_tensors = matched_fits = 0
    for dataset in matched_config["datasets"]:
        npz_path = matched_directory / f"{dataset}.npz"
        json_path = matched_directory / f"{dataset}.json"
        if not npz_path.exists() or not json_path.exists():
            issues.append(f"missing matched-function pair: {dataset}")
            continue
        manifest = json.loads(json_path.read_text())
        if manifest.get("status") != "complete":
            issues.append(f"non-complete matched-function manifest: {dataset}")
        with np.load(npz_path) as archive:
            val = archive["validation_predictions"]
            test = archive["test_predictions"]
            expected = (2, len(matched_config["seeds"]), int(matched_config["representatives"]))
            if tuple(val.shape[:3]) != expected or tuple(test.shape[:3]) != expected:
                issues.append(f"matched-function factor mismatch: {dataset}")
            for split, values in (("validation", val), ("test", test)):
                if not np.isfinite(values).all():
                    issues.append(f"nonfinite matched-function predictions: {dataset}/{split}")
                sums = values.astype(np.float64).sum(axis=-1)
                maximum_probability_sum_error = max(
                    maximum_probability_sum_error, float(np.max(np.abs(sums - 1)))
                )
                minimum_probability = min(minimum_probability, float(values.min()))
                maximum_probability = max(maximum_probability, float(values.max()))
            matched_fits += int(np.prod(expected))
            matched_tensors += 1
    panel_records["matched_function"] = {
        "configured_tensors": len(matched_config["datasets"]),
        "verified_tensors": matched_tensors,
        "represented_fits": matched_fits,
    }
    tensors += matched_tensors
    fits += matched_fits

    # Frozen Day-5 completion program: modern neural, classical, TabPFN,
    # enlarged-menu, matched-function, and semantic row-order artifacts.
    completion = json.loads((HERE / "completion_config.json").read_text())
    completion_records = {}
    neural_dir = RESULTS / "completion_neural"
    completion_tensors = completion_fits = 0
    for mode, expected in (("broad", 12 * 4 * 3), ("exact", 4 * 4), ("matched", 4 * 4)):
        mode_tensors = mode_fits = 0
        paths = sorted(neural_dir.glob(f"*__{mode}.json"))
        if len(paths) != expected:
            issues.append(f"completion neural {mode} count: {len(paths)} != {expected}")
        for json_path in paths:
            try:
                manifest = json.loads(json_path.read_text())
                npz_path = json_path.with_suffix(".npz")
                if manifest.get("status") != "complete" or not npz_path.exists():
                    issues.append(f"completion neural incomplete pair: {json_path.stem}")
                    continue
                with np.load(npz_path) as archive:
                    predictions = archive["test_predictions"]
                    if not np.isfinite(predictions).all():
                        issues.append(f"completion neural nonfinite: {json_path.stem}")
                    if predictions.shape[-1] > 1:
                        error = float(np.max(np.abs(predictions.astype(np.float64).sum(-1) - 1)))
                        maximum_probability_sum_error = max(maximum_probability_sum_error, error)
                        minimum_probability = min(minimum_probability, float(predictions.min()))
                        maximum_probability = max(maximum_probability, float(predictions.max()))
                    if mode != "matched" and len(archive["actions"]) != manifest["actions"]:
                        issues.append(f"completion neural action mismatch: {json_path.stem}")
                represented = int(manifest.get("represented_fits", 0))
                mode_fits += represented; mode_tensors += 1
            except Exception as exc:
                issues.append(f"completion neural unreadable {json_path.stem}: {exc!r}")
        completion_records[f"neural_{mode}"] = {
            "configured_tensors": expected, "verified_tensors": mode_tensors,
            "represented_fits": mode_fits,
        }
        completion_tensors += mode_tensors; completion_fits += mode_fits

    paired_exact_mismatches = 0
    for exact_path in neural_dir.glob("*__exact.npz"):
        broad_path = exact_path.with_name(exact_path.name.replace("__exact.npz", "__broad.npz"))
        if not broad_path.exists():
            paired_exact_mismatches += 1
            continue
        with np.load(exact_path) as exact, np.load(broad_path) as broad:
            if not np.array_equal(exact["actions"], broad["actions"]):
                paired_exact_mismatches += 1
            elif not np.allclose(exact["test_predictions"], broad["test_predictions"], atol=2e-6, rtol=1e-6):
                paired_exact_mismatches += 1
    if paired_exact_mismatches:
        issues.append(f"completion exact/broad reproducibility mismatches: {paired_exact_mismatches}")

    simple_panels = (
        ("classical", RESULTS / "completion_classical", 12 * 5),
        ("tabpfn", RESULTS / "completion_tabpfn", 6 * 3),
        ("menu_size", RESULTS / "completion_menu_size", 2 * 4),
        ("row_order", RESULTS / "completion_row_order", 4 * 4),
    )
    for label, directory, expected in simple_panels:
        panel_tensors = panel_fits = 0
        paths = sorted(directory.glob("*.json")) if directory.exists() else []
        if len(paths) != expected:
            issues.append(f"completion {label} count: {len(paths)} != {expected}")
        for json_path in paths:
            try:
                manifest = json.loads(json_path.read_text())
                npz_path = json_path.with_suffix(".npz")
                if manifest.get("status") != "complete" or not npz_path.exists():
                    issues.append(f"completion {label} incomplete pair: {json_path.stem}")
                    continue
                with np.load(npz_path) as archive:
                    prediction_keys = [
                        key for key in archive.files
                        if "predictions" in key or key.startswith("test__") or key.startswith("validation__")
                    ]
                    for key in prediction_keys:
                        values = archive[key]
                        if not np.isfinite(values).all():
                            issues.append(f"completion {label} nonfinite: {json_path.stem}/{key}")
                        if values.shape[-1] > 1:
                            error = float(np.max(np.abs(values.astype(np.float64).sum(-1) - 1)))
                            maximum_probability_sum_error = max(maximum_probability_sum_error, error)
                            minimum_probability = min(minimum_probability, float(values.min()))
                            maximum_probability = max(maximum_probability, float(values.max()))
                panel_fits += int(manifest.get("represented_fits", manifest.get("tabpfn_calls", 0)))
                panel_tensors += 1
            except Exception as exc:
                issues.append(f"completion {label} unreadable {json_path.stem}: {exc!r}")
        completion_records[label] = {
            "configured_tensors": expected, "verified_tensors": panel_tensors,
            "represented_fits_or_calls": panel_fits,
        }
        completion_tensors += panel_tensors; completion_fits += panel_fits

    output_manifest = RESULTS / "completion_outputs_manifest.json"
    if output_manifest.exists():
        payload = json.loads(output_manifest.read_text())
        table_files = list((RESULTS / "completion_tables").glob("table*"))
        figure_files = list((RESULTS / "completion_figures").glob("figure*"))
        if len(table_files) != payload.get("table_files") or len(figure_files) != payload.get("figure_files"):
            issues.append("completion publication-output file count mismatch")
    else:
        issues.append("missing completion output manifest")
    panel_records["day5_completion"] = {
        "verified_tensors": completion_tensors,
        "represented_fits_or_calls": completion_fits,
        "subpanels": completion_records,
        "exact_broad_reproducibility_mismatches": paired_exact_mismatches,
    }
    tensors += completion_tensors; fits += completion_fits
    if minimum_probability < -1e-7 or maximum_probability > 1 + 1e-7:
        issues.append("classification probability outside tolerance")
    if maximum_probability_sum_error > 1e-6:
        issues.append("classification probability sum outside tolerance")

    screened = pd.read_csv(RESULTS / "validation_screened_cover_cells.csv")
    screen_records = {}
    for study, exact_name in SCREEN_MAP.items():
        exact = pd.read_csv(RESULTS / exact_name)
        validation = exact[exact.split == "validation"].set_index(["dataset", "model"])
        selected = screened[
            (screened.study == study) & (screened.split == "test")
        ]
        mismatches = 0
        for row in selected.itertuples(index=False):
            current = validation.loc[(row.dataset, row.model)]
            if not bool(current.material):
                mismatches += 1
        if mismatches:
            issues.append(f"test-selected material cells: {study}={mismatches}")
        screen_records[study] = {
            "headline_test_cells": len(selected),
            "validation_material_mismatches": mismatches,
        }

    summaries = sorted(RESULTS.glob("*_summary.json"))
    noncomplete = []
    for path in summaries:
        payload = json.loads(path.read_text())
        if payload.get("status") not in {"complete", None}:
            noncomplete.append(path.name)
    if noncomplete:
        issues.append(f"non-complete summaries: {noncomplete}")
    hashes = {str(path): digest(HERE / path) for path in map(Path, HASH_FILES)}
    output = {
        "status": "complete", "audit_passed": not issues,
        "issues": issues, "panels": panel_records,
        "verified_tensors": tensors, "represented_complete_product_fits": fits,
        "maximum_class_probability_sum_error": maximum_probability_sum_error,
        "minimum_class_probability": minimum_probability,
        "maximum_class_probability": maximum_probability,
        "validation_only_screen_audit": screen_records,
        "top_level_summary_files_parsed": len(summaries),
        "noncomplete_summary_files": noncomplete,
        "sha256": hashes,
    }
    (RESULTS / "integrity_audit_summary.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
