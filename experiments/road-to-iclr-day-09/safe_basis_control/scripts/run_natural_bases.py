#!/usr/bin/env python3
"""Validate frozen invariant methods on natural local/spectral, Helmert, and Fourier pairs."""

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
    prospective_specs,
    save_prediction_bundle,
    sha256_file,
    write_json,
)
from safe_basis.gating import alpha_evidence, select_gates  # noqa: E402
from safe_basis.models import fit_predictions  # noqa: E402
from safe_basis.rankgram import build_rank_adaptive_interface  # noqa: E402

from tournament.representations import build_interface  # noqa: E402


def cached_fit(path: Path, *, model: str, blocks: Any, rep: Any, seed: int, device: str, definition: dict[str, Any], finalist_hash: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    hashes = {"protocol": sha256_file(PROTOCOL_PATH), "panel": sha256_file(PANEL_PATH), "finalists": finalist_hash}
    if path.exists() and path.with_suffix(".json").exists():
        arrays, metadata = load_prediction_bundle(path)
        if metadata.get("frozen_hashes") != hashes or metadata.get("definition") != definition:
            raise RuntimeError(f"natural-basis cache drift at {path}")
        return arrays, metadata
    prediction, telemetry = fit_predictions(model, blocks.dataset.problem_type, rep, blocks.dataset.y_train, blocks.dataset.y_validation, seed, device)
    metadata = {"status": "COMPLETE", "definition": definition, "frozen_hashes": hashes, "telemetry": telemetry}
    save_prediction_bundle(path, None, prediction["validation"], prediction["test"], metadata)
    return prediction, metadata


def coordinate_error(left: Any, right: Any) -> float:
    return max(
        float(np.linalg.norm(getattr(left, f"X_{split}") - getattr(right, f"X_{split}")) / max(np.linalg.norm(getattr(left, f"X_{split}")), 1e-12))
        for split in ("train", "validation", "test")
    )


def natural_sets(blocks: Any) -> tuple[list[tuple[str, list[Any]]], list[dict[str, str]]]:
    sets: list[tuple[str, list[Any]]] = []
    unavailable = []
    try:
        sets.append(("local_hat_to_spectral_hat", list(bd.local_spectral_pair(blocks))))
    except ValueError as error:
        unavailable.append({"pair": "local_hat_to_spectral_hat", "reason": str(error)})
    helmert = bd.helmert_pair(blocks)
    if helmert is None:
        unavailable.append({"pair": "one_hot_to_helmert", "reason": "no categorical block with at least three levels"})
    else:
        sets.append(("one_hot_to_helmert", list(helmert)))
    fourier = bd.fourier_origin_pairs(blocks, members=8)
    if not fourier:
        unavailable.append({"pair": "fourier_origin_shift", "reason": "no frozen cyclic metadata with an eight-coordinate block"})
    else:
        sets.append(("fourier_origin_shift", fourier))
    return sets, unavailable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    specs, finalists, finalist_hash = prospective_specs()
    protocol = load_protocol()
    selected_keys = {
        "ozone-level-8hr",
        "wall-robot-navigation",
        "quake",
        "visualizing_soil",
        "CPMP-2015-regression",
    }
    specs = [spec for spec in specs if spec["key"] in selected_keys]
    rank_config = next(item for item in finalists["finalists"] if item["method_id"] == "RankAdaptiveGram")["rank_config"]
    rows = []
    unavailable_rows = []
    for spec in specs:
        blocks = load_blocks(spec, protocol)
        sets, unavailable = natural_sets(blocks)
        unavailable_rows.extend({"dataset": blocks.dataset.key, **row} for row in unavailable)
        for pair_name, representations in sets:
            reconstruction = max(float(rep.metadata.get("reconstruction_error", 0.0)) for rep in representations[1:])
            if reconstruction >= 1e-6:
                raise RuntimeError(f"natural equivalence failed for {blocks.dataset.key}/{pair_name}: {reconstruction}")
            for model in protocol["prospective_models"]:
                seed = 0
                root = ROOT / "results" / "raw" / "natural_bases" / model / blocks.dataset.key / pair_name
                raw = {}
                raw_seconds = 0.0
                for rep in representations:
                    prediction, metadata = cached_fit(
                        root / "Raw" / f"{rep.representation_id}.npz",
                        model=model,
                        blocks=blocks,
                        rep=rep,
                        seed=seed,
                        device=args.device,
                        definition={"method": "Raw", "natural_pair": pair_name, "representation": rep.representation_id},
                        finalist_hash=finalist_hash,
                    )
                    raw[rep.representation_id] = prediction
                    raw_seconds += float(metadata["telemetry"]["fit_seconds"])
                gram_reps = [build_interface(rep, "gram_anchor", blocks.dataset.key, anchors=16, selection="gram_pivot", normalize=True) for rep in representations]
                rank_reps = [build_rank_adaptive_interface(rep, blocks.dataset.key, **rank_config) for rep in representations]
                gram_error = max(coordinate_error(gram_reps[0], rep) for rep in gram_reps[1:])
                rank_error = max(coordinate_error(rank_reps[0], rep) for rep in rank_reps[1:])
                if max(gram_error, rank_error) >= 1e-6:
                    raise RuntimeError(f"natural invariant interface failed for {blocks.dataset.key}/{pair_name}")
                gram_prediction, gram_meta = cached_fit(
                    root / "GramAnchor-m16.npz",
                    model=model,
                    blocks=blocks,
                    rep=gram_reps[0],
                    seed=seed,
                    device=args.device,
                    definition={"method": "GramAnchor-m16", "natural_pair": pair_name},
                    finalist_hash=finalist_hash,
                )
                rank_prediction, rank_meta = cached_fit(
                    root / "RankAdaptiveGram.npz",
                    model=model,
                    blocks=blocks,
                    rep=rank_reps[0],
                    seed=seed,
                    device=args.device,
                    definition={"method": "RankAdaptiveGram", "natural_pair": pair_name, **rank_config},
                    finalist_hash=finalist_hash,
                )
                reference_id = representations[0].representation_id
                gate_config = protocol["safe_alpha"]
                evidence = alpha_evidence(
                    blocks.dataset.problem_type,
                    blocks.dataset.y_validation,
                    blocks.dataset.y_train,
                    raw[reference_id]["validation"],
                    rank_prediction["validation"],
                    alphas=[float(value) for value in gate_config["alphas"]],
                    bootstrap_resamples=int(gate_config["bootstrap_resamples"]),
                    seed=bd.stable_seed("natural-SafeRank", blocks.dataset.key, model, pair_name),
                    epsilon=float(gate_config["epsilon"]),
                )
                gates = select_gates(evidence, taus=[float(value) for value in gate_config["taus"]], constrained_lambda_multipliers=[float(value) for value in gate_config["constrained_lambda_multipliers"]])
                safe_alpha = gates["SafeGram-t01"]
                fixed_evidence = alpha_evidence(
                    blocks.dataset.problem_type,
                    blocks.dataset.y_validation,
                    blocks.dataset.y_train,
                    raw[reference_id]["validation"],
                    gram_prediction["validation"],
                    alphas=[float(value) for value in gate_config["alphas"]],
                    bootstrap_resamples=int(gate_config["bootstrap_resamples"]),
                    seed=bd.stable_seed("natural-SafeGram", blocks.dataset.key, model, pair_name),
                    epsilon=float(gate_config["epsilon"]),
                )
                fixed_gates = select_gates(fixed_evidence, taus=[float(value) for value in gate_config["taus"]], constrained_lambda_multipliers=[float(value) for value in gate_config["constrained_lambda_multipliers"]])
                fixed_safe_alpha = fixed_gates["SafeGram-t01"]
                methods = {
                    "Raw": ({key: value for key, value in raw.items()}, 0.0, raw_seconds),
                    "GramAnchor-m16": ({rep.representation_id: gram_prediction for rep in representations}, 1.0, gram_meta["telemetry"]["fit_seconds"]),
                    "Raw+GramAnchor@0.75": ({rep.representation_id: {split: mix_predictions(raw[rep.representation_id][split], gram_prediction[split], 0.75) for split in ("validation", "test")} for rep in representations}, 0.75, raw_seconds + gram_meta["telemetry"]["fit_seconds"]),
                    "RankAdaptiveGram": ({rep.representation_id: rank_prediction for rep in representations}, 1.0, rank_meta["telemetry"]["fit_seconds"]),
                    "SafeGram-t01": ({rep.representation_id: {split: mix_predictions(raw[rep.representation_id][split], gram_prediction[split], fixed_safe_alpha) for split in ("validation", "test")} for rep in representations}, fixed_safe_alpha, raw_seconds + gram_meta["telemetry"]["fit_seconds"]),
                    "SafeRankGram-t01": ({rep.representation_id: {split: mix_predictions(raw[rep.representation_id][split], rank_prediction[split], safe_alpha) for split in ("validation", "test")} for rep in representations}, safe_alpha, raw_seconds + rank_meta["telemetry"]["fit_seconds"]),
                }
                for method, (prediction, alpha, seconds) in methods.items():
                    for split, target in (("validation", blocks.dataset.y_validation), ("test", blocks.dataset.y_test)):
                        raw_reference = raw[reference_id][split]
                        method_reference = prediction[reference_id][split]
                        raw_d = float(np.mean([disagreement(blocks.dataset.problem_type, target, raw_reference, raw[rep.representation_id][split]) for rep in representations[1:]]))
                        method_d = float(np.mean([disagreement(blocks.dataset.problem_type, target, method_reference, prediction[rep.representation_id][split]) for rep in representations[1:]]))
                        safety = normalized_excess_risk(blocks.dataset.problem_type, target, raw_reference, method_reference, blocks.dataset.y_train)
                        rows.append(
                            {
                                "dataset": blocks.dataset.key,
                                "problem_type": blocks.dataset.problem_type,
                                "model": model,
                                "seed": seed,
                                "natural_pair": pair_name,
                                "method": method,
                                "split": split,
                                "alpha": alpha,
                                "natural_reconstruction_error": reconstruction,
                                "gram_coordinate_error": gram_error,
                                "rank_coordinate_error": rank_error,
                                "raw_disagreement": raw_d,
                                "method_disagreement": method_d,
                                "disagreement_reduction": 0.0 if raw_d <= 1e-12 else 1.0 - method_d / raw_d,
                                "fit_seconds": seconds,
                                **safety,
                            }
                        )
                print(f"[natural] {blocks.dataset.key} {model} {pair_name} safe={fixed_safe_alpha} safe-rank={safe_alpha}", flush=True)
    processed = ROOT / "results" / "processed"
    frame = pd.DataFrame(rows)
    frame.to_csv(processed / "natural_basis_cells.csv", index=False)
    pd.DataFrame(unavailable_rows).to_csv(processed / "natural_basis_unavailable.csv", index=False)
    test = frame[frame["split"] == "test"]
    summary = test.groupby(["natural_pair", "method"], as_index=False).agg(
        median_disagreement_reduction=("disagreement_reduction", "median"),
        median_C=("normalized_excess_risk", "median"),
        p95_C=("normalized_excess_risk", lambda x: float(np.quantile(x, 0.95))),
        max_C=("normalized_excess_risk", "max"),
        max_equivalence_error=("natural_reconstruction_error", "max"),
        max_coordinate_error=("rank_coordinate_error", "max"),
        model_families=("model", "nunique"),
    )
    summary.to_csv(processed / "natural_basis_summary.csv", index=False)
    write_json(processed / "natural_basis_manifest.json", {"status": "COMPLETE", "available_pairs": sorted(frame["natural_pair"].unique().tolist()), "unavailable_records": unavailable_rows, "exact_equivalence_threshold": 1e-6, "cells": len(frame)})


if __name__ == "__main__":
    main()
