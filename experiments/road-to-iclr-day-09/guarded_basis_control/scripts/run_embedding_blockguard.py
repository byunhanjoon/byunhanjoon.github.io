#!/usr/bin/env python3
"""Confirm the frozen development BlockGuard rule inside PLE/RBF embeddings.

The feature choices are transferred from the general RBF-8 BlockGuard run.  They
are not re-selected on an embedding cell, which keeps this a confirmation rather
than a second hyperparameter search.  ResNet uses the controlled-MLP choices
because ResNet is only part of the embedding interface.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from guarded_basis.blockguard import (  # noqa: E402
    coordinate_audit,
    gram_interface,
    invariant_fractions,
    mixed_representation,
    selection_key,
)
from guarded_basis.common import (  # noqa: E402
    cached_representation_predictions,
    development_specs,
    disagreement,
    load_blocks,
    load_protocol,
    normalized_excess_risk,
    task_error,
    write_json,
)
from run_embedding_confirmation import raw_prediction  # noqa: E402
from safe_basis.embeddings import embedding_orbit  # noqa: E402


BACKBONES = ("controlled_mlp", "tabm_d", "resnet_tabular")
METHOD = "BlockGuard-Greedy-t01-transferred"


def transferred_selection(dataset: str, model: str, seed: int) -> dict[str, Any]:
    source_model = "controlled_mlp" if model == "resnet_tabular" else model
    path = ROOT / "results" / "raw" / "blockguard" / "full" / source_model / dataset / f"seed_{seed}.json"
    if not path.exists():
        raise FileNotFoundError(f"full-development BlockGuard source missing: {path}")
    payload = json.loads(path.read_text())
    matches = [
        row for row in payload["selections"]
        if row["method"] == "BlockGuard-Greedy-t01"
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one frozen Greedy selection in {path}")
    return {
        **matches[0],
        "source_model": source_model,
        "source_path": str(path),
        "selection_key": selection_key(matches[0]["features"]),
    }


def run_unit(
    spec: dict[str, Any],
    model: str,
    seed: int,
    embedding: str,
    dimension: int,
    device: str,
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    blocks = load_blocks(spec, protocol)
    dataset = blocks.dataset
    selection = transferred_selection(dataset.key, model, seed)
    selected = list(selection["features"])
    orbit = embedding_orbit(dataset, embedding, dimension, int(protocol["orbit_members"]))
    gram_orbit = [gram_interface(rep, dataset.key) for rep in orbit]
    mixed_orbit = [
        mixed_representation(rep, dataset.key, selected, gram_rep=gram)
        for rep, gram in zip(orbit, gram_orbit)
    ]
    audit = coordinate_audit(mixed_orbit, selected)
    if not audit["passes_1e_minus_6"]:
        raise RuntimeError(
            f"embedding BlockGuard coordinate audit failed: {dataset.key}/{embedding}/k{dimension}"
        )

    raw: list[dict[str, np.ndarray]] = []
    method: list[dict[str, np.ndarray]] = []
    telemetry: list[dict[str, Any]] = []
    for raw_rep, mixed_rep in zip(orbit, mixed_orbit):
        raw_values, _ = raw_prediction(
            blocks=blocks,
            rep=raw_rep,
            model=model,
            seed=seed,
            embedding=embedding,
            dimension=dimension,
            device=device,
        )
        raw.append(raw_values)
        if not selected:
            method.append(raw_values)
            telemetry.append({"fit_seconds": 0.0, "predict_seconds": 0.0})
            continue
        path = (
            ROOT / "results" / "raw" / "embedding_blockguard" / "predictions" / model
            / dataset.key / f"seed_{seed}" / embedding / f"k{dimension}"
            / selection["selection_key"] / f"{raw_rep.representation_id}.npz"
        )
        values, metadata = cached_representation_predictions(
            path,
            model=model,
            blocks=blocks,
            rep=mixed_rep,
            seed=seed,
            device=device,
            definition={
                "method": METHOD,
                "tau": 0.01,
                "selection_rule": "general-RBF8-BlockGuard-Greedy-transfer",
                "selection_source_model": selection["source_model"],
                "selected_features": sorted(selected),
                "selection_key": selection["selection_key"],
                "embedding": embedding,
                "dimension": int(dimension),
                "rotation_member": int(raw_rep.member),
                "interface_location": "between_numerical_embedding_and_backbone",
            },
        )
        method.append(values)
        telemetry.append(metadata.get("telemetry", {}))

    fraction_blocks, fraction_dimensions = invariant_fractions(orbit[0], selected)
    rows: list[dict[str, Any]] = []
    for split, target in (("validation", dataset.y_validation), ("test", dataset.y_test)):
        raw_reference = raw[0][split]
        method_reference = method[0][split]
        raw_d = float(np.mean([
            disagreement(dataset.problem_type, target, raw_reference, values[split])
            for values in raw[1:]
        ]))
        method_ds = [
            disagreement(dataset.problem_type, target, method_reference, values[split])
            for values in method[1:]
        ]
        method_d = float(np.mean(method_ds))
        rows.append(
            {
                "dataset": dataset.key,
                "problem_type": dataset.problem_type,
                "model": model,
                "seed": int(seed),
                "embedding": embedding,
                "k": int(dimension),
                "split": split,
                "method": METHOD,
                "tau": 0.01,
                "selected_features": json.dumps(sorted(selected), separators=(",", ":")),
                "selected_blocks": len(selected),
                "selection_key": selection["selection_key"],
                "selection_source_model": selection["source_model"],
                "general_validation_C": float(selection["validation_C"]),
                "invariant_feature_fraction": fraction_blocks,
                "invariant_dimension_fraction": fraction_dimensions,
                "raw_disagreement": raw_d,
                "method_disagreement": method_d,
                "maximum_method_disagreement": float(np.max(method_ds)),
                "disagreement_reduction": 0.0 if raw_d <= 1e-12 else 1.0 - method_d / raw_d,
                "coordinate_error": float(audit["maximum_selected_block_relative_error"]),
                "raw_task_error": task_error(dataset.problem_type, target, raw_reference),
                "method_task_error": task_error(dataset.problem_type, target, method_reference),
                "inference_passes": 1,
                "input_dimension": int(mixed_orbit[0].X_train.shape[1]),
                "raw_input_dimension": int(orbit[0].X_train.shape[1]),
                "input_dimension_ratio": float(mixed_orbit[0].X_train.shape[1] / orbit[0].X_train.shape[1]),
                "fit_seconds": float(telemetry[0].get("fit_seconds", 0.0)),
                "predict_seconds": float(telemetry[0].get("predict_seconds", 0.0)),
                **normalized_excess_risk(
                    dataset.problem_type,
                    target,
                    raw_reference,
                    method_reference,
                    dataset.y_train,
                    epsilon=1e-8,
                ),
            }
        )

    path = (
        ROOT / "results" / "raw" / "embedding_blockguard" / "full" / model / dataset.key
        / f"seed_{seed}" / embedding / f"k{dimension}.json"
    )
    write_json(
        path,
        {
            "status": "COMPLETE",
            "dataset": dataset.key,
            "model": model,
            "seed": int(seed),
            "embedding": embedding,
            "k": int(dimension),
            "selection_split": "transferred_development_validation_only",
            "selection_source": selection,
            "test_outcomes_used_for_selection": False,
            "prospective_outcomes_accessed": False,
            "coordinate_audit": audit,
            "rows": rows,
        },
    )
    print(
        f"[embedding BlockGuard] {dataset.key} {model} seed={seed} {embedding} k={dimension} "
        f"blocks={len(selected)} D={rows[-1]['disagreement_reduction']:.4f} "
        f"C={rows[-1]['normalized_excess_risk']:.4g}",
        flush=True,
    )
    return rows


def collect(
    specs: list[dict[str, Any]],
    models: list[str],
    seeds: list[int],
    embeddings: list[str],
    dimensions: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        for model in models:
            for seed in seeds:
                for embedding in embeddings:
                    for dimension in dimensions:
                        path = (
                            ROOT / "results" / "raw" / "embedding_blockguard" / "full" / model
                            / spec["key"] / f"seed_{seed}" / embedding / f"k{dimension}.json"
                        )
                        if not path.exists():
                            raise FileNotFoundError(f"missing embedding BlockGuard unit: {path}")
                        payload = json.loads(path.read_text())
                        if payload.get("status") != "COMPLETE":
                            raise RuntimeError(f"incomplete embedding BlockGuard unit: {path}")
                        rows.extend(payload["rows"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset")
    parser.add_argument("--model", choices=BACKBONES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--embedding", choices=["PLE", "RBF"])
    parser.add_argument("--dimension", type=int, choices=[8, 16])
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol()
    specs = development_specs(protocol)
    specs = [spec for spec in specs if args.dataset is None or spec["key"] == args.dataset]
    models = [args.model] if args.model else list(BACKBONES)
    seeds = [args.seed] if args.seed is not None else [int(value) for value in protocol["development_seeds"]]
    embeddings = [args.embedding] if args.embedding else list(protocol["embedding_confirmation"]["types"])
    dimensions = [args.dimension] if args.dimension is not None else [int(value) for value in protocol["embedding_confirmation"]["full_dimensions"]]
    if args.aggregate_only:
        rows = collect(specs, models, seeds, embeddings, dimensions)
    else:
        rows = []
        for spec in specs:
            for model in models:
                for seed in seeds:
                    for embedding in embeddings:
                        for dimension in dimensions:
                            rows.extend(run_unit(spec, model, seed, embedding, dimension, args.device, protocol))

    processed = ROOT / "results" / "processed"
    filtered = any(value is not None for value in (args.dataset, args.model, args.seed, args.embedding, args.dimension))
    suffix = (
        f"__{args.dataset or 'all'}__{args.model or 'all'}__{args.seed if args.seed is not None else 'all'}__"
        f"{args.embedding or 'all'}__{args.dimension if args.dimension is not None else 'all'}"
        if filtered else ""
    )
    frame = pd.DataFrame(rows)
    frame.to_csv(processed / f"embedding_blockguard_cells{suffix}.csv", index=False)
    if not filtered:
        units = (
            frame[frame.split == "test"]
            .groupby(["dataset", "problem_type", "model", "embedding", "k", "method"], as_index=False)
            .median(numeric_only=True)
        )
        units.to_csv(processed / "embedding_blockguard_units.csv", index=False)
        costs = units.normalized_excess_risk.to_numpy(float)
        summary = pd.DataFrame([
            {
                "method": METHOD,
                "units": len(units),
                "median_disagreement_reduction": float(units.disagreement_reduction.median()),
                "median_C": float(np.median(costs)),
                "p90_C": float(np.quantile(costs, 0.90)),
                "p95_C": float(np.quantile(costs, 0.95)),
                "max_C": float(np.max(costs)),
                "median_invariant_feature_fraction": float(units.invariant_feature_fraction.median()),
                "fallback_rate": float((units.selected_blocks == 0).mean()),
                "inference_multiplier": 1.0,
            }
        ])
        summary.to_csv(processed / "embedding_blockguard_summary.csv", index=False)
        write_json(
            processed / "embedding_blockguard_manifest.json",
            {
                "status": "COMPLETE",
                "method": METHOD,
                "transfer_rule": "general-RBF8 BlockGuard-Greedy tau=0.01; ResNet uses controlled-MLP selection",
                "cells": len(frame),
                "units": len(units),
                "datasets": len(specs),
                "models": len(models),
                "seeds": len(seeds),
                "embedding_types": embeddings,
                "dimensions": dimensions,
                "prospective_outcomes_accessed": False,
            },
        )


if __name__ == "__main__":
    main()
