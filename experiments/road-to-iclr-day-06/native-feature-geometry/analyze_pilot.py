"""Frozen-gate analysis for the Native Feature Geometry pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

import native_geometry as ng


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "pilot_config.json"
RESULTS = HERE / "results"
PILOT = RESULTS / "pilot"
STRUCTURED = ("cycle16", "ordinal16", "tree16")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_spearman(x: list[float], y: list[float]) -> float:
    value = spearmanr(np.asarray(x), np.asarray(y)).statistic
    return float(value) if np.isfinite(value) else float("nan")


def rel_reduction(candidate: float, baseline: float) -> float:
    return float((baseline - candidate) / baseline) if baseline > 0 else float("nan")


def load_all(allow_incomplete: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]], dict[tuple[str, str, int], dict[str, Any]]]:
    config = json.loads(CONFIG_PATH.read_text())
    rows: list[dict[str, Any]] = []
    bundles: dict[tuple[str, str, int], dict[str, Any]] = {}
    errors: list[str] = []
    for domain in config["domains"]:
        for regime in config["regimes"]:
            for seed in config["seeds"]:
                stem = f"{domain}__{regime}__seed{seed}"
                artifact = PILOT / f"{stem}.npz"
                manifest_path = PILOT / f"{stem}.json"
                if not artifact.exists() or not manifest_path.exists():
                    errors.append(f"missing {stem}")
                    continue
                manifest = json.loads(manifest_path.read_text())
                if manifest.get("status") != "complete":
                    errors.append(f"incomplete {stem}")
                    continue
                if manifest.get("config_sha256") != sha256(CONFIG_PATH):
                    errors.append(f"config hash mismatch {stem}")
                if manifest.get("artifact_sha256") != sha256(artifact):
                    errors.append(f"artifact hash mismatch {stem}")
                archive = np.load(artifact, allow_pickle=False)
                interfaces = archive["interfaces"].tolist()
                charts = archive["charts"]
                predictions = archive["predictions"]
                if predictions.shape != (6, 5, 1024):
                    errors.append(f"prediction shape {stem}: {predictions.shape}")
                if charts.shape != (5, 16) or len(interfaces) != 6:
                    errors.append(f"menu shape {stem}")
                if not np.isfinite(predictions).all():
                    errors.append(f"nonfinite predictions {stem}")
                metric_lookup = {
                    (entry["interface"], int(entry["chart"])): entry
                    for entry in manifest["metrics"]
                }
                target = archive["test_target"]
                category = archive["test_category"]
                held_values = archive["held"]
                held_mask = np.isin(category, held_values)
                for interface_index, interface in enumerate(interfaces):
                    quotient = predictions[interface_index].mean(axis=0)
                    for chart_index in range(len(charts)):
                        entry = metric_lookup[(interface, chart_index)]
                        orbit_mask = held_mask if held_mask.any() else np.ones(len(target), dtype=bool)
                        orbit_damage = float(np.mean(
                            (predictions[interface_index, chart_index, orbit_mask] - quotient[orbit_mask]) ** 2
                        ))
                        rows.append({
                            "domain": domain,
                            "regime": regime,
                            "seed": seed,
                            "interface": interface,
                            "chart": chart_index,
                            "test_mse": float(entry["test_mse"]),
                            "seen_mse": float(entry["seen_mse"]),
                            "held_mse": "" if entry["held_mse"] is None else float(entry["held_mse"]),
                            "initial_native_cka": float(entry["initial_native_cka"]),
                            "final_native_cka": float(entry["final_native_cka"]),
                            "final_corrupt_cka": float(entry["final_corrupt_cka"]),
                            "orbit_damage": orbit_damage,
                        })
                bundles[(domain, regime, seed)] = {
                    name: archive[name].copy() for name in archive.files
                }
    if errors and not allow_incomplete:
        raise RuntimeError("; ".join(errors))
    return config, rows, bundles


def subset(rows: list[dict[str, Any]], **conditions: Any) -> list[dict[str, Any]]:
    return [row for row in rows if all(row[key] == value for key, value in conditions.items())]


def analyze_h1(config: dict[str, Any], bundles: dict[tuple[str, str, int], dict[str, Any]]) -> dict[str, Any]:
    errors = []
    stored_cast_errors = []
    for (domain, regime, seed), bundle in bundles.items():
        # The frozen H1 gate is explicitly a float64 compiler audit.  Stored
        # training tables are float32 and are reported separately rather than
        # compared to the 1e-10 construction threshold.
        embedding, gram = ng.native_embedding(domain, int(config["embedding_dim"]))
        base_error = float(np.max(np.abs(embedding @ embedding.T - gram)))
        errors.append(base_error)
        stored = bundle["native_embedding"].astype(np.float64)
        stored_cast_errors.append(float(np.max(np.abs(stored @ stored.T - gram))))
        for chart in bundle["charts"]:
            code_table = np.empty_like(embedding)
            code_table[chart] = embedding
            code_gram = code_table @ code_table.T
            expected = np.empty_like(code_gram)
            expected[np.ix_(chart, chart)] = gram
            errors.append(float(np.max(np.abs(code_gram - expected))))
    maximum = max(errors) if errors else float("nan")
    return {
        "pass": bool(len(bundles) == 24 and maximum <= 1e-10),
        "bundle_count": len(bundles),
        "maximum_aligned_gram_error": maximum,
        "maximum_float32_training_copy_gram_error": max(stored_cast_errors),
        "threshold": 1e-10,
        "analysis_correction": "float64 compiler audited per frozen gate; float32 training copy reported separately",
    }


def analyze_h2(rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary = [
        row for row in rows
        if row["domain"] in STRUCTURED and row["regime"] == "interpolation" and row["interface"] == "learned"
    ]
    gains = [row["final_native_cka"] - row["initial_native_cka"] for row in primary]
    domain_medians = {
        domain: float(np.median([
            row["final_native_cka"] - row["initial_native_cka"] for row in primary if row["domain"] == domain
        ])) for domain in STRUCTURED
    }
    cell_positive = []
    for domain in STRUCTURED:
        for seed in (7301, 7302, 7303):
            values = [
                row["final_native_cka"] - row["initial_native_cka"]
                for row in primary if row["domain"] == domain and row["seed"] == seed
            ]
            cell_positive.append(float(np.median(values)) > 0)
    corrupt_wins = [row["final_native_cka"] > row["final_corrupt_cka"] for row in primary]
    pooled = float(np.median(gains))
    cell_fraction = float(np.mean(cell_positive))
    corrupt_fraction = float(np.mean(corrupt_wins))
    passed = pooled >= 0.10 and all(value > 0 for value in domain_medians.values()) and cell_fraction >= 0.80 and corrupt_fraction >= 0.80
    return {
        "pass": bool(passed),
        "path_count": len(primary),
        "pooled_median_cka_gain": pooled,
        "domain_median_cka_gain": domain_medians,
        "positive_cell_fraction": cell_fraction,
        "native_over_corrupt_fraction": corrupt_fraction,
    }


def analyze_h3(rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary = [
        row for row in rows
        if row["domain"] in STRUCTURED
        and row["regime"] == "category_holdout"
        and row["interface"] in {"learned", "native_tuned"}
    ]
    cka = [row["final_native_cka"] for row in primary]
    held = [float(row["held_mse"]) for row in primary]
    orbit = [row["orbit_damage"] for row in primary]
    pooled_held = safe_spearman(cka, held)
    pooled_orbit = safe_spearman(cka, orbit)
    by_domain = {
        domain: safe_spearman(
            [row["final_native_cka"] for row in primary if row["domain"] == domain],
            [float(row["held_mse"]) for row in primary if row["domain"] == domain],
        ) for domain in STRUCTURED
    }
    domain_passes = sum(value <= -0.40 for value in by_domain.values())
    passed = pooled_held <= -0.60 and domain_passes >= 2 and pooled_orbit <= -0.50
    return {
        "pass": bool(passed),
        "path_count": len(primary),
        "pooled_cka_vs_held_mse_spearman": pooled_held,
        "domain_cka_vs_held_mse_spearman": by_domain,
        "domain_threshold_pass_count": domain_passes,
        "pooled_cka_vs_orbit_damage_spearman": pooled_orbit,
    }


def chart_mean(rows: list[dict[str, Any]], domain: str, regime: str, seed: int, interface: str, key: str) -> float:
    values = [float(row[key]) for row in rows if row["domain"] == domain and row["regime"] == regime and row["seed"] == seed and row["interface"] == interface]
    return float(np.mean(values))


def analyze_h4(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baselines = ("label", "learned", "random_fixed")
    cell_rows = []
    all_three_wins = 0
    relative = []
    orbit_label_ratios = []
    orbit_learned_ratios = []
    for domain in STRUCTURED:
        for seed in (7301, 7302, 7303):
            native = chart_mean(rows, domain, "category_holdout", seed, "native_fixed", "held_mse")
            values = {name: chart_mean(rows, domain, "category_holdout", seed, name, "held_mse") for name in baselines}
            wins = {name: native < value for name, value in values.items()}
            all_three_wins += int(all(wins.values()))
            best = min(values.values())
            reduction = rel_reduction(native, best)
            relative.append(reduction)
            native_orbit = chart_mean(rows, domain, "category_holdout", seed, "native_fixed", "orbit_damage")
            label_orbit = chart_mean(rows, domain, "category_holdout", seed, "label", "orbit_damage")
            learned_orbit = chart_mean(rows, domain, "category_holdout", seed, "learned", "orbit_damage")
            orbit_label_ratios.append(native_orbit / label_orbit if label_orbit > 0 else float("inf"))
            orbit_learned_ratios.append(native_orbit / learned_orbit if learned_orbit > 0 else float("inf"))
            cell_rows.append({
                "domain": domain, "seed": seed, "native_held_mse": native,
                "best_baseline_held_mse": best, "relative_reduction": reduction,
                "beats_all_three": all(wins.values()),
            })
    interpolation_ratios = []
    for domain in STRUCTURED:
        for seed in (7301, 7302, 7303):
            native = chart_mean(rows, domain, "interpolation", seed, "native_fixed", "test_mse")
            best = min(
                chart_mean(rows, domain, "interpolation", seed, "learned", "test_mse"),
                chart_mean(rows, domain, "interpolation", seed, "random_fixed", "test_mse"),
            )
            interpolation_ratios.append(native / best)
    nominal_relative = []
    for seed in (7301, 7302, 7303):
        native = chart_mean(rows, "nominal16", "category_holdout", seed, "native_fixed", "held_mse")
        best = min(chart_mean(rows, "nominal16", "category_holdout", seed, name, "held_mse") for name in baselines)
        nominal_relative.append(rel_reduction(native, best))
    pooled_reduction = float(np.median(relative))
    schema_ratio_label = float(np.mean(orbit_label_ratios))
    schema_ratio_learned = float(np.mean(orbit_learned_ratios))
    interpolation_ratio = float(np.median(interpolation_ratios))
    nominal_improvement = float(np.median(nominal_relative))
    passed = (
        all_three_wins >= 7 and pooled_reduction >= 0.20
        and schema_ratio_label <= 0.10 and schema_ratio_learned <= 0.10
        and interpolation_ratio <= 1.05 and nominal_improvement < 0.05
    )
    return {
        "pass": bool(passed),
        "beats_all_three_cell_count": all_three_wins,
        "pooled_median_relative_held_mse_reduction": pooled_reduction,
        "native_to_label_orbit_ratio_mean": schema_ratio_label,
        "native_to_learned_orbit_ratio_mean": schema_ratio_learned,
        "interpolation_native_to_best_ratio_median": interpolation_ratio,
        "nominal_median_relative_improvement": nominal_improvement,
        "cells": cell_rows,
    }


def held_patch_mse(bundle: dict[str, Any], interface: str, patch: str) -> np.ndarray:
    interface_index = bundle["transport_interfaces"].tolist().index(interface)
    patch_index = bundle["patch_names"].tolist().index(patch)
    predictions = bundle["patch_predictions"][interface_index, :, patch_index]
    held_mask = np.isin(bundle["test_category"], bundle["held"])
    target = bundle["test_target"][held_mask]
    return np.mean((predictions[:, held_mask].astype(np.float64) - target[None, :]) ** 2, axis=1)


def analyze_h5(bundles: dict[tuple[str, str, int], dict[str, Any]]) -> dict[str, Any]:
    controls = ("original", "mean", "random", "shuffled_transport")
    win_count = 0
    reductions = []
    correct_vs_original = []
    shuffled_vs_original = []
    max_seen_change = 0.0
    cells = []
    for domain in STRUCTURED:
        for seed in (7301, 7302, 7303):
            bundle = bundles[(domain, "category_holdout", seed)]
            correct = float(np.mean(held_patch_mse(bundle, "native_tuned", "native_transport")))
            control_values = {name: float(np.mean(held_patch_mse(bundle, "native_tuned", name))) for name in controls}
            win = all(correct < value for value in control_values.values())
            win_count += int(win)
            best = min(control_values.values())
            reductions.append(rel_reduction(correct, best))
            original = control_values["original"]
            correct_vs_original.append(rel_reduction(correct, original))
            shuffled_vs_original.append(rel_reduction(control_values["shuffled_transport"], original))
            changes = bundle["patch_seen_changes"]
            max_seen_change = max(max_seen_change, float(np.nanmax(changes)))
            cells.append({
                "domain": domain, "seed": seed, "correct_mse": correct,
                "best_control_mse": best, "relative_reduction": reductions[-1],
                "beats_every_control": win,
            })
    nominal_wins = 0
    nominal_reductions = []
    for seed in (7301, 7302, 7303):
        bundle = bundles[("nominal16", "category_holdout", seed)]
        correct = float(np.mean(held_patch_mse(bundle, "native_tuned", "native_transport")))
        control = min(float(np.mean(held_patch_mse(bundle, "native_tuned", name))) for name in controls)
        nominal_wins += int(correct < control)
        nominal_reductions.append(rel_reduction(correct, control))
    median_reduction = float(np.median(reductions))
    correct_rescue = float(np.median(correct_vs_original))
    shuffled_rescue = float(np.median(shuffled_vs_original))
    corruption_boundary = shuffled_rescue < 0.5 * correct_rescue
    nominal_passes_primary = nominal_wins >= 3 and float(np.median(nominal_reductions)) >= 0.20
    passed = (
        win_count >= 7 and median_reduction >= 0.20 and max_seen_change == 0.0
        and corruption_boundary and not nominal_passes_primary
    )
    return {
        "pass": bool(passed),
        "beats_every_control_cell_count": win_count,
        "pooled_median_relative_reduction_vs_best_control": median_reduction,
        "maximum_seen_prediction_change": max_seen_change,
        "median_correct_reduction_vs_original": correct_rescue,
        "median_shuffled_reduction_vs_original": shuffled_rescue,
        "corruption_boundary_pass": bool(corruption_boundary),
        "nominal_win_count": nominal_wins,
        "nominal_median_relative_reduction": float(np.median(nominal_reductions)),
        "nominal_passes_primary": bool(nominal_passes_primary),
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    config, rows, bundles = load_all(args.allow_incomplete)
    if len(bundles) < 24:
        print(json.dumps({"status": "incomplete", "bundles": len(bundles)}, indent=2))
        return
    summary = {
        "status": "complete",
        "bundle_count": len(bundles),
        "path_count": len(rows),
        "hypotheses": {
            "H1_native_gram_equivariance": analyze_h1(config, bundles),
            "H2_native_geometry_emergence": analyze_h2(rows),
            "H3_geometry_schema_risk_coupling": analyze_h3(rows),
            "H4_native_geometry_mitigation": analyze_h4(rows),
            "H5_native_chart_transport": analyze_h5(bundles),
        },
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_csv(RESULTS / "pilot_paths.csv", rows)
    (RESULTS / "pilot_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
