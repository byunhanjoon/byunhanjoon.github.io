#!/usr/bin/env python3
"""Run the locked new tail-prospective panel after finalist freeze verification."""

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
    disagreement,
    load_blocks,
    load_prediction_bundle,
    load_protocol,
    mix_predictions,
    normalized_excess_risk,
    orthogonal_orbit,
    prospective_specs,
    save_prediction_bundle,
    sha256_file,
    task_error,
    write_json,
)
from safe_basis.gating import alpha_evidence, select_gates, verify_alpha_zero  # noqa: E402
from safe_basis.models import fit_predictions  # noqa: E402
from safe_basis.rankgram import build_rank_adaptive_interface, orbit_coordinate_audit  # noqa: E402

from tournament.representations import audit_orbit_coordinates, build_interface  # noqa: E402


def cached_fit(
    path: Path,
    *,
    model: str,
    blocks: Any,
    rep: Any,
    seed: int,
    device: str,
    definition: dict[str, Any],
    finalist_hash: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    hashes = {
        "protocol": sha256_file(PROTOCOL_PATH),
        "panel": sha256_file(PANEL_PATH),
        "finalists": finalist_hash,
    }
    if path.exists() and path.with_suffix(".json").exists():
        arrays, metadata = load_prediction_bundle(path)
        if metadata.get("frozen_hashes") != hashes or metadata.get("definition") != definition:
            raise RuntimeError(f"prospective cache drift at {path}")
        return arrays, metadata
    predictions, telemetry = fit_predictions(
        model,
        blocks.dataset.problem_type,
        rep,
        blocks.dataset.y_train,
        blocks.dataset.y_validation,
        seed,
        device,
    )
    metadata = {
        "status": "COMPLETE",
        "panel": "NEW_tail_prospective",
        "dataset": blocks.dataset.key,
        "model": model,
        "seed": int(seed),
        "definition": definition,
        "frozen_hashes": hashes,
        "telemetry": telemetry,
    }
    save_prediction_bundle(path, None, predictions["validation"], predictions["test"], metadata)
    return predictions, metadata


def gate(
    *,
    label: str,
    blocks: Any,
    raw_validation: np.ndarray,
    invariant_validation: np.ndarray,
    model: str,
    seed: int,
    protocol: dict[str, Any],
) -> tuple[float, list[dict[str, Any]]]:
    config = protocol["safe_alpha"]
    evidence = alpha_evidence(
        blocks.dataset.problem_type,
        blocks.dataset.y_validation,
        blocks.dataset.y_train,
        raw_validation,
        invariant_validation,
        alphas=[float(value) for value in config["alphas"]],
        bootstrap_resamples=int(config["bootstrap_resamples"]),
        seed=bd.stable_seed(label, blocks.dataset.key, model, seed),
        epsilon=float(config["epsilon"]),
    )
    verify_alpha_zero(evidence)
    gates = select_gates(
        evidence,
        taus=[float(value) for value in config["taus"]],
        constrained_lambda_multipliers=[float(value) for value in config["constrained_lambda_multipliers"]],
    )
    return float(gates["SafeGram-t01"]), evidence


def evaluate_prediction_orbit(
    *,
    blocks: Any,
    orbit: list[Any],
    raw: dict[str, dict[str, np.ndarray]],
    predictions: dict[str, dict[str, np.ndarray]],
    method: str,
    alpha: float,
    model: str,
    seed: int,
    fit_seconds: float,
) -> list[dict[str, Any]]:
    reference_id = orbit[0].representation_id
    rows = []
    for split, target in (("validation", blocks.dataset.y_validation), ("test", blocks.dataset.y_test)):
        raw_reference = raw[reference_id][split]
        method_reference = predictions[reference_id][split]
        raw_disagreements = [disagreement(blocks.dataset.problem_type, target, raw_reference, raw[rep.representation_id][split]) for rep in orbit[1:]]
        method_disagreements = [disagreement(blocks.dataset.problem_type, target, method_reference, predictions[rep.representation_id][split]) for rep in orbit[1:]]
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
                "panel": "NEW_tail_prospective",
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
                "maximum_method_disagreement": float(np.max(method_disagreements)),
                "disagreement_reduction": 0.0 if raw_d <= 1e-12 else 1.0 - method_d / raw_d,
                "fit_seconds": float(fit_seconds),
                **safety,
            }
        )
    return rows


def mixed_orbit(raw: dict[str, dict[str, np.ndarray]], invariant: dict[str, np.ndarray], alpha: float) -> dict[str, dict[str, np.ndarray]]:
    return {
        representation_id: {
            split: mix_predictions(prediction[split], invariant[split], alpha)
            for split in ("validation", "test")
        }
        for representation_id, prediction in raw.items()
    }


def run_unit(spec: dict[str, Any], model: str, seed: int, device: str, protocol: dict[str, Any], finalists: dict[str, Any], finalist_hash: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    blocks = load_blocks(spec, protocol)
    orbit = orthogonal_orbit(blocks, protocol)
    reference_id = orbit[0].representation_id
    root = ROOT / "results" / "raw" / "prospective" / model / blocks.dataset.key / f"seed_{seed}"
    raw: dict[str, dict[str, np.ndarray]] = {}
    raw_seconds = 0.0
    for rep in orbit:
        prediction, metadata = cached_fit(
            root / "Raw" / f"{rep.representation_id}.npz",
            model=model,
            blocks=blocks,
            rep=rep,
            seed=seed,
            device=device,
            definition={"method": "Raw", "representation_id": rep.representation_id, "equal_budget": True},
            finalist_hash=finalist_hash,
        )
        raw[rep.representation_id] = prediction
        raw_seconds += float(metadata["telemetry"]["fit_seconds"])

    fixed_mapped = [build_interface(rep, "gram_anchor", blocks.dataset.key, anchors=16, selection="gram_pivot", normalize=True) for rep in orbit]
    fixed_audits = audit_orbit_coordinates(fixed_mapped[0], fixed_mapped[1:])
    fixed_max = max(max(row["train_relative_error"], row["validation_relative_error"], row["test_relative_error"]) for row in fixed_audits)
    if fixed_max >= 1e-6:
        raise RuntimeError(f"prospective fixed Gram invariance failed at {blocks.dataset.key}: {fixed_max}")
    fixed_prediction, fixed_meta = cached_fit(
        root / "GramAnchor-m16.npz",
        model=model,
        blocks=blocks,
        rep=fixed_mapped[0],
        seed=seed,
        device=device,
        definition={"method": "GramAnchor-m16", "anchors": 16, "selection": "gram_pivot", "normalization": "anchor_norm_plus_coordinate_standardization"},
        finalist_hash=finalist_hash,
    )

    rank_finalist = next(item for item in finalists["finalists"] if item["method_id"] == "RankAdaptiveGram")
    rank_config = rank_finalist["rank_config"]
    rank_mapped = [build_rank_adaptive_interface(rep, blocks.dataset.key, **rank_config) for rep in orbit]
    rank_audits = orbit_coordinate_audit(rank_mapped[0], rank_mapped[1:])
    rank_max = max(max(row["train_relative_error"], row["validation_relative_error"], row["test_relative_error"]) for row in rank_audits)
    if rank_max >= 1e-6:
        raise RuntimeError(f"prospective rank Gram invariance failed at {blocks.dataset.key}: {rank_max}")
    rank_prediction, rank_meta = cached_fit(
        root / "RankAdaptiveGram.npz",
        model=model,
        blocks=blocks,
        rep=rank_mapped[0],
        seed=seed,
        device=device,
        definition={"method": "RankAdaptiveGram", **rank_config},
        finalist_hash=finalist_hash,
    )

    pca_mapped = [build_interface(rep, "pca", blocks.dataset.key) for rep in orbit]
    pca_audits = audit_orbit_coordinates(pca_mapped[0], pca_mapped[1:])
    pca_max = max(max(row["train_relative_error"], row["validation_relative_error"], row["test_relative_error"]) for row in pca_audits)
    pca: dict[str, dict[str, np.ndarray]] = {}
    pca_seconds = 0.0
    if pca_max < 1e-6:
        prediction, metadata = cached_fit(
            root / "PCA" / f"{pca_mapped[0].representation_id}.npz",
            model=model,
            blocks=blocks,
            rep=pca_mapped[0],
            seed=seed,
            device=device,
            definition={"method": "PCA-canonicalization", "coordinate_identity_reuse": True},
            finalist_hash=finalist_hash,
        )
        pca = {rep.representation_id: prediction for rep in orbit}
        pca_seconds = float(metadata["telemetry"]["fit_seconds"])
    else:
        for source_rep, pca_rep in zip(orbit, pca_mapped):
            prediction, metadata = cached_fit(
                root / "PCA" / f"{pca_rep.representation_id}.npz",
                model=model,
                blocks=blocks,
                rep=pca_rep,
                seed=seed,
                device=device,
                definition={"method": "PCA-canonicalization", "source_representation": source_rep.representation_id, "coordinate_identity_reuse": False},
                finalist_hash=finalist_hash,
            )
            pca[source_rep.representation_id] = prediction
            pca_seconds += float(metadata["telemetry"]["fit_seconds"])

    safe_alpha, safe_evidence = gate(
        label="SafeGram-t01",
        blocks=blocks,
        raw_validation=raw[reference_id]["validation"],
        invariant_validation=fixed_prediction["validation"],
        model=model,
        seed=seed,
        protocol=protocol,
    )
    safe_rank_alpha, safe_rank_evidence = gate(
        label="SafeRankGram-t01",
        blocks=blocks,
        raw_validation=raw[reference_id]["validation"],
        invariant_validation=rank_prediction["validation"],
        model=model,
        seed=seed,
        protocol=protocol,
    )
    methods: dict[str, tuple[dict[str, dict[str, np.ndarray]], float, float]] = {
        "Raw": (raw, 0.0, raw_seconds),
        "GramAnchor-m16": ({key: fixed_prediction for key in raw}, 1.0, float(fixed_meta["telemetry"]["fit_seconds"])),
        "Raw+GramAnchor@0.5": (mixed_orbit(raw, fixed_prediction, 0.5), 0.5, raw_seconds + float(fixed_meta["telemetry"]["fit_seconds"])),
        "Raw+GramAnchor@0.75": (mixed_orbit(raw, fixed_prediction, 0.75), 0.75, raw_seconds + float(fixed_meta["telemetry"]["fit_seconds"])),
        "PCA-canonicalization": (pca, 1.0, pca_seconds),
        "RankAdaptiveGram": ({key: rank_prediction for key in raw}, 1.0, float(rank_meta["telemetry"]["fit_seconds"])),
        "SafeGram-t01": (mixed_orbit(raw, fixed_prediction, safe_alpha), safe_alpha, raw_seconds + float(fixed_meta["telemetry"]["fit_seconds"])),
        "SafeRankGram-t01": (mixed_orbit(raw, rank_prediction, safe_rank_alpha), safe_rank_alpha, raw_seconds + float(rank_meta["telemetry"]["fit_seconds"])),
    }
    rows = []
    for method, (prediction, alpha, seconds) in methods.items():
        rows.extend(
            evaluate_prediction_orbit(
                blocks=blocks,
                orbit=orbit,
                raw=raw,
                predictions=prediction,
                method=method,
                alpha=alpha,
                model=model,
                seed=seed,
                fit_seconds=seconds,
            )
        )
    audit_rows = [
        {"dataset": blocks.dataset.key, "model": model, "seed": seed, "interface": "GramAnchor-m16", **row}
        for row in fixed_audits
    ] + [
        {"dataset": blocks.dataset.key, "model": model, "seed": seed, "interface": "RankAdaptiveGram", **row}
        for row in rank_audits
    ] + [
        {"dataset": blocks.dataset.key, "model": model, "seed": seed, "interface": "PCA-canonicalization", **row}
        for row in pca_audits
    ]
    write_json(
        root / "unit_audit.json",
        {
            "status": "COMPLETE",
            "finalist_hash": finalist_hash,
            "fixed_coordinate_max": fixed_max,
            "rank_coordinate_max": rank_max,
            "pca_coordinate_max": pca_max,
            "SafeGram_alpha": safe_alpha,
            "SafeGram_evidence": safe_evidence,
            "SafeRankGram_alpha": safe_rank_alpha,
            "SafeRankGram_evidence": safe_rank_evidence,
        },
    )
    print(f"[prospective] {blocks.dataset.key} {model} seed={seed} safe={safe_alpha} safe-rank={safe_rank_alpha}", flush=True)
    return rows, audit_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model")
    parser.add_argument("--dataset")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    specs, finalists, finalist_hash = prospective_specs()
    protocol = load_protocol()
    specs = [spec for spec in specs if args.dataset is None or spec["key"] == args.dataset]
    models = [args.model] if args.model else list(protocol["prospective_models"])
    seeds = [args.seed] if args.seed is not None else [int(value) for value in protocol["model_seeds"]]
    rows = []
    audits = []
    for spec in specs:
        for model in models:
            for seed in seeds:
                unit_rows, unit_audits = run_unit(spec, model, seed, args.device, protocol, finalists, finalist_hash)
                rows.extend(unit_rows); audits.extend(unit_audits)
    processed = ROOT / "results" / "processed"
    filtered = args.model is not None or args.dataset is not None or args.seed is not None
    suffix = f"__{args.dataset or 'all'}__{args.model or 'all'}__{args.seed if args.seed is not None else 'all'}" if filtered else ""
    pd.DataFrame(rows).to_csv(processed / f"prospective_cells{suffix}.csv", index=False)
    pd.DataFrame(audits).to_csv(processed / f"prospective_coordinate_audits{suffix}.csv", index=False)
    if not filtered:
        write_json(
            processed / "prospective_run_manifest.json",
            {
                "status": "COMPLETE",
                "finalist_hash": finalist_hash,
                "cells": len(rows),
                "coordinate_audits": len(audits),
                "datasets": len(specs),
                "models": len(models),
                "seeds": len(seeds),
                "methods": 8,
                "prospective_access_after_freeze": True,
            },
        )


if __name__ == "__main__":
    main()
