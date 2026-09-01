#!/usr/bin/env python3
"""Run PLE/RBF embedding-basis tests with interfaces between embedding and backbone."""

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
    load_json,
    load_prediction_bundle,
    load_protocol,
    mix_predictions,
    normalized_excess_risk,
    save_prediction_bundle,
    sha256_file,
    task_error,
    write_json,
)
from safe_basis.embeddings import embedding_orbit  # noqa: E402
from safe_basis.gating import alpha_evidence, select_gates, verify_alpha_zero  # noqa: E402
from safe_basis.models import fit_predictions  # noqa: E402
from safe_basis.rankgram import build_rank_adaptive_interface, orbit_coordinate_audit  # noqa: E402

from tournament.representations import audit_orbit_coordinates, build_interface  # noqa: E402


RANK_SELECTION = ROOT / "results" / "processed" / "rank_selection.json"


def cached_fit(
    path: Path,
    *,
    model: str,
    dataset: Any,
    rep: Any,
    seed: int,
    device: str,
    definition: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    hashes = {"protocol": sha256_file(PROTOCOL_PATH), "panel": sha256_file(PANEL_PATH)}
    if path.exists() and path.with_suffix(".json").exists():
        arrays, metadata = load_prediction_bundle(path)
        if metadata.get("frozen_hashes") != hashes or metadata.get("definition") != definition:
            raise RuntimeError(f"embedding cache drift at {path}")
        return arrays, metadata
    predictions, telemetry = fit_predictions(
        model,
        dataset.problem_type,
        rep,
        dataset.y_train,
        dataset.y_validation,
        seed,
        device,
    )
    metadata = {
        "status": "COMPLETE",
        "interface_location": "between_embedding_and_backbone",
        "definition": definition,
        "frozen_hashes": hashes,
        "telemetry": telemetry,
    }
    save_prediction_bundle(path, None, predictions["validation"], predictions["test"], metadata)
    return predictions, metadata


def gates_for(
    *,
    dataset: Any,
    raw: np.ndarray,
    invariant: np.ndarray,
    protocol: dict[str, Any],
    seed_parts: tuple[Any, ...],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    config = protocol["safe_alpha"]
    evidence = alpha_evidence(
        dataset.problem_type,
        dataset.y_validation,
        dataset.y_train,
        raw,
        invariant,
        alphas=[float(value) for value in config["alphas"]],
        bootstrap_resamples=int(config["bootstrap_resamples"]),
        seed=bd.stable_seed(*seed_parts),
        epsilon=float(config["epsilon"]),
    )
    verify_alpha_zero(evidence)
    selections = select_gates(
        evidence,
        taus=[float(value) for value in config["taus"]],
        constrained_lambda_multipliers=[float(value) for value in config["constrained_lambda_multipliers"]],
    )
    return selections, evidence


def run_unit(spec: dict[str, Any], model: str, seed: int, embedding: str, dimension: int, device: str, protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = bd.load_dataset(spec, {
        "split_seed": protocol["split_seed"],
        "max_train_rows": protocol["max_train_rows"],
        "max_validation_rows": protocol["max_validation_rows"],
        "max_test_rows": protocol["max_test_rows"],
    })
    orbit = embedding_orbit(dataset, embedding, dimension, protocol["embeddings"]["rotation_members"])
    unit = ROOT / "results" / "raw" / "embeddings" / model / dataset.key / f"seed_{seed}" / embedding / f"k{dimension}"
    raw: dict[str, dict[str, np.ndarray]] = {}
    raw_meta: dict[str, dict[str, Any]] = {}
    for rep in orbit:
        definition = {
            "condition": "Raw embedding" if rep.is_reference else "Rotated embedding",
            "embedding": embedding,
            "dimension": int(dimension),
            "rotation_member": int(rep.member),
            "optimizer": "AdamW" if model != "catboost" else "native",
        }
        prediction, metadata = cached_fit(
            unit / "Raw" / f"{rep.representation_id}.npz",
            model=model,
            dataset=dataset,
            rep=rep,
            seed=seed,
            device=device,
            definition=definition,
        )
        raw[rep.representation_id] = prediction
        raw_meta[rep.representation_id] = metadata
    reference_id = orbit[0].representation_id
    fixed_mapped = [build_interface(rep, "gram_anchor", dataset.key, anchors=16, selection="gram_pivot", normalize=True) for rep in orbit]
    fixed_audits = audit_orbit_coordinates(fixed_mapped[0], fixed_mapped[1:])
    fixed_max = max(max(row["train_relative_error"], row["validation_relative_error"], row["test_relative_error"]) for row in fixed_audits)
    if fixed_max >= 1e-6:
        raise RuntimeError(f"fixed Gram embedding audit failed: {dataset.key}/{embedding}/k{dimension}/{fixed_max}")
    rank_config = load_json(RANK_SELECTION)["config"]
    rank_mapped = [build_rank_adaptive_interface(rep, dataset.key, **rank_config) for rep in orbit]
    rank_audits = orbit_coordinate_audit(rank_mapped[0], rank_mapped[1:])
    rank_max = max(max(row["train_relative_error"], row["validation_relative_error"], row["test_relative_error"]) for row in rank_audits)
    if rank_max >= 1e-6:
        raise RuntimeError(f"rank Gram embedding audit failed: {dataset.key}/{embedding}/k{dimension}/{rank_max}")
    fixed_prediction, fixed_meta = cached_fit(
        unit / "GramAfterEmbedding.npz",
        model=model,
        dataset=dataset,
        rep=fixed_mapped[0],
        seed=seed,
        device=device,
        definition={"condition": "Gram-after-embedding", "embedding": embedding, "dimension": int(dimension), "anchors": 16},
    )
    rank_prediction, rank_meta = cached_fit(
        unit / "RankAdaptiveGramAfterEmbedding.npz",
        model=model,
        dataset=dataset,
        rep=rank_mapped[0],
        seed=seed,
        device=device,
        definition={"condition": "RankAdaptiveGram-after-embedding", "embedding": embedding, "dimension": int(dimension), **rank_config},
    )
    fixed_gates, fixed_evidence = gates_for(
        dataset=dataset,
        raw=raw[reference_id]["validation"],
        invariant=fixed_prediction["validation"],
        protocol=protocol,
        seed_parts=("embedding-SafeGram", dataset.key, model, seed, embedding, dimension),
    )
    rank_gates, rank_evidence = gates_for(
        dataset=dataset,
        raw=raw[reference_id]["validation"],
        invariant=rank_prediction["validation"],
        protocol=protocol,
        seed_parts=("embedding-SafeRankGram", dataset.key, model, seed, embedding, dimension),
    )
    rotation_rows = []
    for split, target in (("validation", dataset.y_validation), ("test", dataset.y_test)):
        reference = raw[reference_id][split]
        reference_loss = task_error(dataset.problem_type, target, reference)
        for rep in orbit:
            prediction = raw[rep.representation_id][split]
            rotation_rows.append(
                {
                    "dataset": dataset.key,
                    "problem_type": dataset.problem_type,
                    "model": model,
                    "seed": int(seed),
                    "embedding": embedding,
                    "k": int(dimension),
                    "split": split,
                    "condition": "original" if rep.is_reference else "rotated",
                    "rotation_member": int(rep.member),
                    "original_task": reference_loss,
                    "rotated_task": task_error(dataset.problem_type, target, prediction),
                    "task_effect": task_error(dataset.problem_type, target, prediction) - reference_loss,
                    "disagreement": disagreement(dataset.problem_type, target, reference, prediction),
                    "fit_seconds": float(raw_meta[rep.representation_id]["telemetry"]["fit_seconds"]),
                    "interface_location": "embedding_to_backbone_boundary",
                }
            )
    method_definitions = {
        "Raw embedding": (None, 0.0),
        "Gram-after-embedding": (fixed_prediction, 1.0),
        "RankAdaptiveGram-after-embedding": (rank_prediction, 1.0),
        "SafeGram-after-embedding": (fixed_prediction, fixed_gates["SafeGram-t01"]),
        "SafeRankGram prediction hybrid": (rank_prediction, rank_gates["SafeGram-t01"]),
    }
    method_rows = []
    for method, (invariant, alpha) in method_definitions.items():
        for split, target in (("validation", dataset.y_validation), ("test", dataset.y_test)):
            raw_reference = raw[reference_id][split]
            method_reference = raw_reference if invariant is None else mix_predictions(raw_reference, invariant[split], alpha)
            raw_disagreements = [disagreement(dataset.problem_type, target, raw_reference, raw[rep.representation_id][split]) for rep in orbit[1:]]
            method_disagreements = [
                disagreement(
                    dataset.problem_type,
                    target,
                    method_reference,
                    raw[rep.representation_id][split] if invariant is None else mix_predictions(raw[rep.representation_id][split], invariant[split], alpha),
                )
                for rep in orbit[1:]
            ]
            raw_d = float(np.mean(raw_disagreements))
            method_d = float(np.mean(method_disagreements))
            safety = normalized_excess_risk(dataset.problem_type, target, raw_reference, method_reference, dataset.y_train)
            method_rows.append(
                {
                    "dataset": dataset.key,
                    "problem_type": dataset.problem_type,
                    "model": model,
                    "seed": int(seed),
                    "embedding": embedding,
                    "k": int(dimension),
                    "method": method,
                    "split": split,
                    "alpha": float(alpha),
                    "raw_fallback": bool(alpha == 0),
                    "raw_disagreement": raw_d,
                    "method_disagreement": method_d,
                    "disagreement_reduction": 0.0 if raw_d <= 1e-12 else 1.0 - method_d / raw_d,
                    "fixed_coordinate_error": fixed_max,
                    "rank_coordinate_error": rank_max,
                    "fit_seconds": 0.0 if invariant is None else float((fixed_meta if invariant is fixed_prediction else rank_meta)["telemetry"]["fit_seconds"]),
                    **safety,
                }
            )
    audit_rows = [
        {"dataset": dataset.key, "model": model, "seed": seed, "embedding": embedding, "k": dimension, "interface": "Gram-after-embedding", **row}
        for row in fixed_audits
    ] + [
        {"dataset": dataset.key, "model": model, "seed": seed, "embedding": embedding, "k": dimension, "interface": "RankAdaptiveGram-after-embedding", **row}
        for row in rank_audits
    ]
    write_json(
        unit / "unit_audit.json",
        {
            "status": "COMPLETE",
            "interface_location": "between_embedding_and_backbone",
            "fixed_coordinate_max": fixed_max,
            "rank_coordinate_max": rank_max,
            "SafeGram_evidence": fixed_evidence,
            "SafeGram_alpha": fixed_gates["SafeGram-t01"],
            "SafeRankGram_evidence": rank_evidence,
            "SafeRankGram_alpha": rank_gates["SafeGram-t01"],
        },
    )
    print(f"[embedding] {dataset.key} {model} seed={seed} {embedding} k={dimension} rawD={method_rows[-10]['raw_disagreement']:.4g} safe-rank-alpha={rank_gates['SafeGram-t01']}", flush=True)
    return rotation_rows, method_rows, audit_rows


def write_frames(prefix: str, rotations: list[dict[str, Any]], methods: list[dict[str, Any]], audits: list[dict[str, Any]], suffix: str | None = None) -> None:
    processed = ROOT / "results" / "processed"
    ending = f"__{suffix}" if suffix else ""
    pd.DataFrame(rotations).to_csv(processed / f"{prefix}_rotation_cells{ending}.csv", index=False)
    pd.DataFrame(methods).to_csv(processed / f"{prefix}_method_cells{ending}.csv", index=False)
    pd.DataFrame(audits).to_csv(processed / f"{prefix}_coordinate_audits{ending}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["main", "dimension"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model")
    parser.add_argument("--dataset")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if not RANK_SELECTION.exists():
        raise RuntimeError("rank selection required before embedding integration")
    protocol = load_protocol()
    wanted = set(protocol["embeddings"]["dimension_datasets"])
    specs = [spec for spec in development_specs(protocol) if spec["key"] in wanted and (args.dataset is None or spec["key"] == args.dataset)]
    if args.stage == "main":
        models = [args.model] if args.model else list(protocol["rank_and_embedding_models"])
        seeds = [args.seed] if args.seed is not None else [int(value) for value in protocol["model_seeds"]]
        dimensions = [int(protocol["embeddings"]["primary_dimension"])]
    else:
        models = [args.model or "controlled_mlp"]
        seeds = [args.seed if args.seed is not None else 0]
        dimensions = [int(value) for value in protocol["embeddings"]["dimension_ablation"]]
    rotations: list[dict[str, Any]] = []
    methods: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for spec in specs:
        for model in models:
            for seed in seeds:
                for embedding in protocol["embeddings"]["types"]:
                    for dimension in dimensions:
                        unit = run_unit(spec, model, seed, embedding, dimension, args.device, protocol)
                        rotations.extend(unit[0]); methods.extend(unit[1]); audits.extend(unit[2])
    filtered = args.model is not None or args.dataset is not None or args.seed is not None
    suffix = f"{args.dataset or 'all'}__{args.model or 'all'}__{args.seed if args.seed is not None else 'all'}" if filtered else None
    prefix = "embedding_main" if args.stage == "main" else "embedding_dimension"
    write_frames(prefix, rotations, methods, audits, suffix)
    if not filtered:
        rotation_frame = pd.DataFrame(rotations)
        test_rotated = rotation_frame[(rotation_frame["split"] == "test") & (rotation_frame["condition"] == "rotated")]
        dimension_units = (
            test_rotated.groupby(["dataset", "model", "seed", "embedding", "k"], as_index=False)
            .agg(
                disagreement=("disagreement", "mean"),
                task_effect=("task_effect", "mean"),
                best_basis_task_effect=("task_effect", "min"),
                worst_basis_task_effect=("task_effect", "max"),
            )
        )
        dimension_units.to_csv(ROOT / "results" / "processed" / f"{prefix}_units.csv", index=False)
        method_frame = pd.DataFrame(methods)
        test_methods = method_frame[method_frame["split"] == "test"]
        method_units = test_methods.groupby(["dataset", "model", "embedding", "k", "method"], as_index=False).median(numeric_only=True)
        method_units.to_csv(ROOT / "results" / "processed" / f"{prefix}_method_units.csv", index=False)
        write_json(ROOT / "results" / "processed" / f"{prefix}_manifest.json", {"status": "COMPLETE", "rotation_cells": len(rotation_frame), "method_cells": len(method_frame), "coordinate_audits": len(audits), "interface_location": "between_embedding_and_backbone"})


if __name__ == "__main__":
    main()
