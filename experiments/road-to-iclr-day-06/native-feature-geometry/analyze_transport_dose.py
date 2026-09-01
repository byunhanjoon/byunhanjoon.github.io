"""Frozen H6 dose-response analysis."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "pilot_config.json"
H6 = HERE / "results" / "h6"
RESULTS = HERE / "results"
STRUCTURED = ("cycle16", "ordinal16", "tree16")
INTERFACES = ("learned", "native_tuned")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    max_replay = max_seen = max_endpoint = 0.0
    cell_curves: dict[tuple[str, int, str], np.ndarray] = {}
    for domain in config["domains"]:
        for seed in config["seeds"]:
            stem = f"{domain}__seed{seed}"
            artifact = H6 / f"{stem}.npz"
            manifest_path = H6 / f"{stem}.json"
            if not artifact.exists() or not manifest_path.exists():
                errors.append(f"missing {stem}")
                continue
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("artifact_sha256") != sha256(artifact):
                errors.append(f"hash {stem}")
            bundle = np.load(artifact, allow_pickle=False)
            predictions = bundle["predictions"]
            if predictions.shape != (2, 5, 5, 1024) or not np.isfinite(predictions).all():
                errors.append(f"shape/finiteness {stem}")
            max_replay = max(max_replay, float(np.max(bundle["replay_errors"])))
            max_seen = max(max_seen, float(np.max(bundle["seen_changes"])))
            max_endpoint = max(max_endpoint, float(np.max(bundle["endpoint_errors"])))
            held_mask = np.isin(bundle["test_category"], bundle["held"])
            target = bundle["test_target"][held_mask]
            for interface_index, interface in enumerate(bundle["interfaces"].tolist()):
                curve = []
                for alpha_index, alpha in enumerate(bundle["alphas"]):
                    held_predictions = predictions[interface_index, :, alpha_index, :][:, held_mask]
                    chart_mse = np.mean(
                        (held_predictions.astype(np.float64) - target[None, :]) ** 2, axis=1
                    )
                    curve.append(float(np.mean(chart_mse)))
                    rows.append({
                        "domain": domain,
                        "seed": seed,
                        "interface": interface,
                        "alpha": float(alpha),
                        "chart_mean_held_mse": curve[-1],
                        "chart_sd_held_mse": float(np.std(chart_mse)),
                    })
                cell_curves[(domain, seed, interface)] = np.asarray(curve)
    if errors:
        raise RuntimeError("; ".join(errors))

    interface_summary = {}
    overall_pass = True
    for interface in INTERFACES:
        endpoint_wins = 0
        dose_passes = 0
        reductions = []
        correlations = {}
        for domain in STRUCTURED:
            for seed in config["seeds"]:
                curve = cell_curves[(domain, seed, interface)]
                endpoint_wins += int(curve[0] < curve[-1])
                correlation = float(spearmanr(np.arange(len(curve)), curve).statistic)
                correlations[f"{domain}__seed{seed}"] = correlation
                dose_passes += int(correlation >= 0.80)
                reductions.append((curve[-1] - curve[0]) / curve[-1])
        median_reduction = float(np.median(reductions))
        passed = endpoint_wins == 9 and dose_passes >= 7 and median_reduction >= 0.50
        overall_pass &= passed
        interface_summary[interface] = {
            "pass": bool(passed),
            "endpoint_win_count": endpoint_wins,
            "dose_spearman_pass_count": dose_passes,
            "median_endpoint_relative_reduction": median_reduction,
            "cell_spearman": correlations,
        }
    nominal_ranges = []
    for interface in INTERFACES:
        for seed in config["seeds"]:
            curve = cell_curves[("nominal16", seed, interface)]
            nominal_ranges.append(float((curve.max() - curve.min()) / max(curve.mean(), 1e-12)))
    maximum_nominal_range = max(nominal_ranges)
    integrity_pass = max_replay == 0.0 and max_seen == 0.0
    nominal_pass = maximum_nominal_range <= 1e-5
    overall_pass &= integrity_pass and nominal_pass
    summary = {
        "status": "complete",
        "pass": bool(overall_pass),
        "interfaces": interface_summary,
        "maximum_replay_prediction_error": max_replay,
        "maximum_seen_prediction_change": max_seen,
        "maximum_endpoint_reproduction_error": max_endpoint,
        "maximum_nominal_relative_mse_range": maximum_nominal_range,
        "integrity_pass": bool(integrity_pass),
        "nominal_invariance_pass": bool(nominal_pass),
        "bundle_count": len(config["domains"]) * len(config["seeds"]),
        "trained_path_count": 120,
        "intervention_count": 600,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "h6_dose_cells.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (RESULTS / "h6_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

