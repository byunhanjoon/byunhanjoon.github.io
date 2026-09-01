#!/usr/bin/env python3
"""Screen, select, and evaluate RankAdaptiveGram on the development panel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safe_basis.common import (  # noqa: E402
    PANEL_PATH,
    PROTOCOL_PATH,
    bd,
    development_specs,
    disagreement,
    load_blocks,
    load_frozen_development_predictions,
    load_json,
    load_prediction_bundle,
    load_protocol,
    mix_predictions,
    normalized_excess_risk,
    orthogonal_orbit,
    save_prediction_bundle,
    sha256_file,
    task_error,
    write_json,
)
from safe_basis.gating import alpha_evidence, select_gates, verify_alpha_zero  # noqa: E402
from safe_basis.models import fit_predictions  # noqa: E402
from safe_basis.rankgram import build_rank_adaptive_interface, orbit_coordinate_audit  # noqa: E402

from tournament.representations import build_interface  # noqa: E402


SELECTION_PATH = ROOT / "results" / "processed" / "rank_selection.json"


def cached_fit(
    path: Path,
    *,
    model: str,
    blocks: Any,
    rep: Any,
    seed: int,
    device: str,
    stage: str,
    definition: dict[str, Any],
    include_train: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    hashes = {"protocol": sha256_file(PROTOCOL_PATH), "panel": sha256_file(PANEL_PATH)}
    if path.exists() and path.with_suffix(".json").exists():
        arrays, metadata = load_prediction_bundle(path)
        if metadata.get("frozen_hashes") != hashes or metadata.get("definition") != definition:
            raise RuntimeError(f"cache drift at {path}")
        return arrays, metadata
    predictions, telemetry = fit_predictions(
        model,
        blocks.dataset.problem_type,
        rep,
        blocks.dataset.y_train,
        blocks.dataset.y_validation,
        seed,
        device,
        include_train=include_train,
    )
    metadata = {
        "status": "COMPLETE",
        "stage": stage,
        "dataset": blocks.dataset.key,
        "model": model,
        "seed": int(seed),
        "definition": definition,
        "frozen_hashes": hashes,
        "telemetry": telemetry,
    }
    save_prediction_bundle(
        path,
        predictions.get("train"),
        predictions["validation"],
        predictions["test"],
        metadata,
    )
    return predictions, metadata


def config_id(config: dict[str, Any]) -> str:
    return (
        f"eps{config['relative_threshold']:g}__{config['anchor_rule']}__"
        f"{config['normalization']}__std{int(config['standardize'])}"
    ).replace("+", "")


def coordinate_summary(mapped: list[Any]) -> tuple[float, list[dict[str, Any]], list[dict[str, Any]]]:
    audits = orbit_coordinate_audit(mapped[0], mapped[1:])
    maximum = max(
        max(row["train_relative_error"], row["validation_relative_error"], row["test_relative_error"])
        for row in audits
    )
    blocks = []
    for feature, metadata in mapped[0].metadata["block_audits"].items():
        blocks.append({"feature": feature, **metadata})
    return maximum, audits, blocks


def screen_configs(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    rank = protocol["rank_adaptive_gram"]
    configs = []
    for threshold in rank["rank_thresholds"]:
        for rule in rank["anchor_rules"]:
            configs.append(
                {
                    "relative_threshold": float(threshold),
                    "anchor_rule": rule,
                    "normalization": "N1_anchor_norm",
                    "standardize": True,
                    "screen": "rank_anchor",
                }
            )
    for normalization in rank["normalizations"]:
        configs.append(
            {
                "relative_threshold": 1e-4,
                "anchor_rule": "rank_plus_one",
                "normalization": normalization,
                "standardize": False,
                "screen": "normalization",
            }
        )
    unique = {config_id(config): config for config in configs}
    return list(unique.values())


def run_screen(protocol: dict[str, Any], device: str) -> None:
    wanted = set(protocol["rank_adaptive_gram"]["normalization_screen_datasets"])
    specs = [spec for spec in development_specs(protocol) if spec["key"] in wanted]
    rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    for spec in specs:
        blocks = load_blocks(spec, protocol)
        orbit = orthogonal_orbit(blocks, protocol)
        raw, _, _ = load_frozen_development_predictions("controlled_mlp", blocks.dataset.key, 0)
        reference_id = orbit[0].representation_id
        for config in screen_configs(protocol):
            mapped = [build_rank_adaptive_interface(rep, blocks.dataset.key, **{key: config[key] for key in ("relative_threshold", "anchor_rule", "normalization", "standardize")}) for rep in orbit]
            maximum_error, coordinate_audits, diagnostics = coordinate_summary(mapped)
            if maximum_error >= 1e-6:
                raise RuntimeError(f"RankAdaptiveGram coordinate invariance failed: {blocks.dataset.key} {config} {maximum_error}")
            identifier = config_id(config)
            path = ROOT / "results" / "raw" / "rank_screen" / blocks.dataset.key / "controlled_mlp" / "seed_0" / f"{identifier}.npz"
            predictions, metadata = cached_fit(
                path,
                model="controlled_mlp",
                blocks=blocks,
                rep=mapped[0],
                seed=0,
                device=device,
                stage="rank_screen",
                definition=config,
            )
            for split, target in (("validation", blocks.dataset.y_validation), ("test", blocks.dataset.y_test)):
                safety = normalized_excess_risk(
                    blocks.dataset.problem_type,
                    target,
                    raw[reference_id][split],
                    predictions[split],
                    blocks.dataset.y_train,
                )
                rows.append(
                    {
                        "dataset": blocks.dataset.key,
                        "problem_type": blocks.dataset.problem_type,
                        "model": "controlled_mlp",
                        "seed": 0,
                        "config_id": identifier,
                        **config,
                        "split": split,
                        "maximum_coordinate_error": maximum_error,
                        "median_coordinate_dimension": float(np.median([item["coordinate_dimension"] for item in diagnostics])),
                        "total_coordinate_dimension": int(mapped[0].X_train.shape[1]),
                        "median_reconstruction_error": float(np.median([item["reconstruction_error"] for item in diagnostics])),
                        "maximum_reconstruction_error": float(np.max([item["reconstruction_error"] for item in diagnostics])),
                        "median_empirical_rank": float(np.median([item["empirical_rank"] for item in diagnostics])),
                        "fit_seconds": float(metadata["telemetry"]["fit_seconds"]),
                        **safety,
                    }
                )
            block_rows.extend(
                {
                    "dataset": blocks.dataset.key,
                    "config_id": identifier,
                    **config,
                    **diagnostic,
                }
                for diagnostic in diagnostics
            )
            write_json(path.with_name(path.stem + "__coordinate_audit.json"), coordinate_audits)
            print(f"[rank screen] {blocks.dataset.key} {identifier}", flush=True)
    processed = ROOT / "results" / "processed"
    frame = pd.DataFrame(rows)
    block_frame = pd.DataFrame(block_rows)
    frame.to_csv(processed / "rank_screen_cells.csv", index=False)
    block_frame.to_csv(processed / "rank_block_diagnostics.csv", index=False)
    validation = frame[frame["split"] == "validation"]
    summary = (
        validation.groupby(["config_id", "relative_threshold", "anchor_rule", "normalization", "standardize", "screen"], as_index=False)
        .agg(
            median_C=("normalized_excess_risk", "median"),
            p95_C=("normalized_excess_risk", lambda x: float(np.quantile(x, 0.95))),
            maximum_C=("normalized_excess_risk", "max"),
            median_total_coordinate_dimension=("total_coordinate_dimension", "median"),
            maximum_reconstruction_error=("maximum_reconstruction_error", "max"),
            maximum_coordinate_error=("maximum_coordinate_error", "max"),
        )
    )
    summary["information_preserving"] = summary["maximum_reconstruction_error"] <= 1e-4
    summary["coordinate_invariant"] = summary["maximum_coordinate_error"] < 1e-6
    summary["development_safe"] = summary["median_C"] <= 0.02
    summary.to_csv(processed / "rank_screen_summary.csv", index=False)
    eligible = summary[summary["information_preserving"] & summary["coordinate_invariant"] & summary["development_safe"]]
    selection_pool = eligible if len(eligible) else summary[summary["coordinate_invariant"]]
    selected = selection_pool.sort_values(
        ["median_C", "p95_C", "median_total_coordinate_dimension", "maximum_reconstruction_error"]
    ).iloc[0]
    config = {
        "relative_threshold": float(selected["relative_threshold"]),
        "anchor_rule": str(selected["anchor_rule"]),
        "normalization": str(selected["normalization"]),
        "standardize": bool(selected["standardize"]),
    }
    write_json(
        SELECTION_PATH,
        {
            "status": "DEVELOPMENT_SELECTED",
            "selection_split": "development_validation_only",
            "prospective_outcomes_accessed": False,
            "config_id": str(selected["config_id"]),
            "config": config,
            "validation_evidence": {key: selected[key] for key in ("median_C", "p95_C", "maximum_C", "median_total_coordinate_dimension", "maximum_reconstruction_error", "maximum_coordinate_error")},
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "prospective_panel_sha256": sha256_file(PANEL_PATH),
        },
    )
    print(f"[rank selection] {config_id(config)}", flush=True)


def raw_orbit_for_model(
    model: str,
    blocks: Any,
    orbit: list[Any],
    seed: int,
    device: str,
) -> dict[str, dict[str, np.ndarray]]:
    if model in {"controlled_mlp", "tabm_d"}:
        raw, _, _ = load_frozen_development_predictions(model, blocks.dataset.key, seed)
        return raw
    output = {}
    for rep in orbit:
        definition = {"method": "Raw", "representation_id": rep.representation_id}
        path = ROOT / "results" / "raw" / "rank_full" / model / blocks.dataset.key / f"seed_{seed}" / "Raw" / f"{rep.representation_id}.npz"
        prediction, _ = cached_fit(
            path,
            model=model,
            blocks=blocks,
            rep=rep,
            seed=seed,
            device=device,
            stage="rank_full",
            definition=definition,
        )
        output[rep.representation_id] = prediction
    return output


def evaluate_full_unit(spec: dict[str, Any], model: str, seed: int, device: str, protocol: dict[str, Any], selected: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    blocks = load_blocks(spec, protocol)
    orbit = orthogonal_orbit(blocks, protocol)
    raw = raw_orbit_for_model(model, blocks, orbit, seed, device)
    reference_id = orbit[0].representation_id
    config = dict(selected["config"])
    mapped = [build_rank_adaptive_interface(rep, blocks.dataset.key, **config) for rep in orbit]
    maximum_error, coordinate_audits, diagnostics = coordinate_summary(mapped)
    if maximum_error >= 1e-6:
        raise RuntimeError(f"rank coordinate audit failed for {blocks.dataset.key}/{model}/{seed}: {maximum_error}")
    identifier = config_id(config)
    path = ROOT / "results" / "raw" / "rank_full" / model / blocks.dataset.key / f"seed_{seed}" / "RankAdaptiveGram" / f"{identifier}.npz"
    rank_prediction, rank_metadata = cached_fit(
        path,
        model=model,
        blocks=blocks,
        rep=mapped[0],
        seed=seed,
        device=device,
        stage="rank_full",
        definition={"method": "RankAdaptiveGram", **config},
    )
    gate_config = protocol["safe_alpha"]
    evidence = alpha_evidence(
        blocks.dataset.problem_type,
        blocks.dataset.y_validation,
        blocks.dataset.y_train,
        raw[reference_id]["validation"],
        rank_prediction["validation"],
        alphas=[float(value) for value in gate_config["alphas"]],
        bootstrap_resamples=int(gate_config["bootstrap_resamples"]),
        seed=bd.stable_seed("SafeRankGram", blocks.dataset.key, model, seed),
        epsilon=float(gate_config["epsilon"]),
    )
    verify_alpha_zero(evidence)
    gates = select_gates(
        evidence,
        taus=[float(value) for value in gate_config["taus"]],
        constrained_lambda_multipliers=[float(value) for value in gate_config["constrained_lambda_multipliers"]],
    )
    methods = {"RankAdaptiveGram": 1.0, "SafeRankGram-t01": gates["SafeGram-t01"]}
    rows = []
    for method, alpha in methods.items():
        for split, target in (("validation", blocks.dataset.y_validation), ("test", blocks.dataset.y_test)):
            raw_reference = raw[reference_id][split]
            method_reference = mix_predictions(raw_reference, rank_prediction[split], alpha)
            raw_disagreements = [disagreement(blocks.dataset.problem_type, target, raw_reference, raw[rep.representation_id][split]) for rep in orbit[1:]]
            method_disagreements = [
                disagreement(
                    blocks.dataset.problem_type,
                    target,
                    method_reference,
                    mix_predictions(raw[rep.representation_id][split], rank_prediction[split], alpha),
                )
                for rep in orbit[1:]
            ]
            raw_d = float(np.mean(raw_disagreements))
            method_d = float(np.mean(method_disagreements))
            safety = normalized_excess_risk(
                blocks.dataset.problem_type,
                target,
                raw_reference,
                method_reference,
                blocks.dataset.y_train,
            )
            rows.append(
                {
                    "panel": "development",
                    "dataset": blocks.dataset.key,
                    "problem_type": blocks.dataset.problem_type,
                    "model": model,
                    "seed": int(seed),
                    "method": method,
                    "split": split,
                    "alpha": float(alpha),
                    "raw_fallback": bool(alpha == 0),
                    "raw_disagreement": raw_d,
                    "method_disagreement": method_d,
                    "disagreement_reduction": 0.0 if raw_d <= 1e-12 else 1.0 - method_d / raw_d,
                    "maximum_coordinate_error": maximum_error,
                    "median_empirical_rank": float(np.median([row["empirical_rank"] for row in diagnostics])),
                    "median_anchor_count": float(np.median([row["anchor_count"] for row in diagnostics])),
                    "total_coordinate_dimension": int(mapped[0].X_train.shape[1]),
                    "maximum_reconstruction_error": float(np.max([row["reconstruction_error"] for row in diagnostics])),
                    "fit_seconds": float(rank_metadata["telemetry"]["fit_seconds"]),
                    **safety,
                }
            )
    diagnostic_rows = [
        {
            "dataset": blocks.dataset.key,
            "model": model,
            "seed": int(seed),
            **config,
            **row,
        }
        for row in diagnostics
    ]
    audit_path = path.with_name(path.stem + "__audit.json")
    write_json(audit_path, {"coordinate_audits": coordinate_audits, "gate_evidence": evidence, "selected_gates": gates})
    print(f"[rank full] {blocks.dataset.key} {model} seed={seed} alpha={gates['SafeGram-t01']}", flush=True)
    return rows, diagnostic_rows


def run_full(protocol: dict[str, Any], device: str, model_filter: str | None, dataset_filter: str | None, seed_filter: int | None) -> None:
    if not SELECTION_PATH.exists():
        raise RuntimeError("run --stage screen before --stage full")
    selected = load_json(SELECTION_PATH)
    specs = [spec for spec in development_specs(protocol) if dataset_filter is None or spec["key"] == dataset_filter]
    models = [model_filter] if model_filter else list(protocol["rank_and_embedding_models"])
    seeds = [seed_filter] if seed_filter is not None else [int(value) for value in protocol["model_seeds"]]
    rows = []
    diagnostics = []
    for spec in specs:
        for model in models:
            for seed in seeds:
                unit, block = evaluate_full_unit(spec, model, seed, device, protocol, selected)
                rows.extend(unit)
                diagnostics.extend(block)
    processed = ROOT / "results" / "processed"
    if model_filter or dataset_filter or seed_filter is not None:
        suffix = f"{dataset_filter or 'all'}__{model_filter or 'all'}__{seed_filter if seed_filter is not None else 'all'}"
        pd.DataFrame(rows).to_csv(processed / f"rank_development_cells__{suffix}.csv", index=False)
        pd.DataFrame(diagnostics).to_csv(processed / f"rank_development_block_diagnostics__{suffix}.csv", index=False)
    else:
        frame = pd.DataFrame(rows)
        frame.to_csv(processed / "rank_development_cells.csv", index=False)
        pd.DataFrame(diagnostics).to_csv(processed / "rank_development_block_diagnostics.csv", index=False)
        test = frame[frame["split"] == "test"]
        units = test.groupby(["dataset", "problem_type", "model", "method"], as_index=False).median(numeric_only=True)
        units.to_csv(processed / "rank_development_units.csv", index=False)
        summary = []
        for method, group in units.groupby("method"):
            costs = group["normalized_excess_risk"].to_numpy(float)
            summary.append(
                {
                    "method": method,
                    "units": len(group),
                    "model_families": group["model"].nunique(),
                    "median_disagreement_reduction": float(group["disagreement_reduction"].median()),
                    "median_C": float(np.median(costs)),
                    "p95_C": float(np.quantile(costs, 0.95)),
                    "max_C": float(np.max(costs)),
                    "raw_fallback_rate": float(group["raw_fallback"].mean()),
                }
            )
        pd.DataFrame(summary).to_csv(processed / "rank_development_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["screen", "full"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model")
    parser.add_argument("--dataset")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    protocol = load_protocol()
    if args.stage == "screen":
        run_screen(protocol, args.device)
    else:
        run_full(protocol, args.device, args.model, args.dataset, args.seed)


if __name__ == "__main__":
    main()
