#!/usr/bin/env python3
"""Run the frozen RBF-16 embedding finalist on the untouched panel."""

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

from guarded_basis.blockguard import gram_interface  # noqa: E402
from guarded_basis.common import (  # noqa: E402
    bd,
    cached_representation_predictions,
    disagreement,
    load_blocks,
    load_protocol,
    mix_predictions,
    normalized_excess_risk,
    prospective_specs,
    task_error,
    write_json,
)
from guarded_basis.gating import guarded_evidence, select_g2, strip_samples  # noqa: E402
from safe_basis.embeddings import embedding_orbit  # noqa: E402


BACKBONES = ("controlled_mlp", "tabm_d", "resnet_tabular")
FINALIST = "GuardedGram-G2-after-RBF-k16"


def prediction(
    *,
    blocks: Any,
    rep: Any,
    model: str,
    seed: int,
    device: str,
    finalist_hash: str,
    condition: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = (
        ROOT / "results" / "raw" / "prospective" / "embedding_predictions" / model
        / blocks.dataset.key / f"seed_{seed}" / "RBF" / "k16" / condition
        / f"{rep.representation_id}.npz"
    )
    return cached_representation_predictions(
        path,
        model=model,
        blocks=blocks,
        rep=rep,
        seed=seed,
        device=device,
        finalist_hash=finalist_hash,
        definition={
            "condition": condition,
            "embedding": "RBF",
            "dimension": 16,
            "rotation_member": int(rep.member),
            "interface_location": "between_numerical_embedding_and_backbone",
            "anchors": 16 if condition == "GramAfterEmbedding" else None,
            "anchor_selection": "gram_pivot" if condition == "GramAfterEmbedding" else None,
        },
    )


def run_unit(
    spec: dict[str, Any],
    model: str,
    seed: int,
    device: str,
    protocol: dict[str, Any],
    finalists: dict[str, Any],
    finalist_hash: str,
) -> list[dict[str, Any]]:
    names = {str(row["method"]) for row in finalists["finalists"]}
    if FINALIST not in names:
        raise RuntimeError(f"embedding finalist {FINALIST} is not present in frozen config")
    # The lock has been validated before any selected dataset is loaded.
    blocks = load_blocks(spec, protocol)
    dataset = blocks.dataset
    orbit = embedding_orbit(dataset, "RBF", 16, int(protocol["orbit_members"]))
    raw: list[dict[str, np.ndarray]] = []
    for rep in orbit:
        values, _ = prediction(
            blocks=blocks,
            rep=rep,
            model=model,
            seed=seed,
            device=device,
            finalist_hash=finalist_hash,
            condition="Raw",
        )
        raw.append(values)
    gram_orbit = [gram_interface(rep, dataset.key) for rep in orbit]
    coordinate_errors = []
    for rep in gram_orbit[1:]:
        for split in ("X_train", "X_validation", "X_test"):
            reference = np.asarray(getattr(gram_orbit[0], split))
            values = np.asarray(getattr(rep, split))
            coordinate_errors.append(float(np.linalg.norm(values - reference) / max(np.linalg.norm(reference), 1e-12)))
    maximum_coordinate_error = max(coordinate_errors)
    if maximum_coordinate_error >= 1e-6:
        raise RuntimeError(f"prospective embedding coordinate audit failed: {dataset.key}")
    gram, gram_meta = prediction(
        blocks=blocks,
        rep=gram_orbit[0],
        model=model,
        seed=seed,
        device=device,
        finalist_hash=finalist_hash,
        condition="GramAfterEmbedding",
    )
    evidence = guarded_evidence(
        dataset.problem_type,
        dataset.y_validation,
        dataset.y_train,
        raw[0]["validation"],
        gram["validation"],
        alphas=[0.75, 0.5, 0.25, 0.0],
        resamples=1000,
        seed=bd.stable_seed("embedding-GuardedG2", dataset.key, model, seed, "RBF", 16),
    )
    selected_alpha = select_g2(evidence, tau=0.01, gamma=0.0)
    methods = (
        ("Raw RBF-k16 embedding", 0.0),
        ("Raw+Gram@0.75 RBF-k16 embedding", 0.75),
        ("Gram-after-RBF-k16 embedding", 1.0),
        (FINALIST, selected_alpha),
    )
    rows: list[dict[str, Any]] = []
    for split, target in (("validation", dataset.y_validation), ("test", dataset.y_test)):
        raw_reference = raw[0][split]
        raw_ds = [
            disagreement(dataset.problem_type, target, raw_reference, values[split])
            for values in raw[1:]
        ]
        raw_d = float(np.mean(raw_ds))
        for method, alpha in methods:
            method_reference = mix_predictions(raw_reference, gram[split], alpha)
            method_ds = [
                disagreement(
                    dataset.problem_type,
                    target,
                    method_reference,
                    mix_predictions(values[split], gram[split], alpha),
                )
                for values in raw[1:]
            ]
            method_d = float(np.mean(method_ds))
            rows.append(
                {
                    "panel": "guarded_new_untouched_prospective_embedding",
                    "dataset": dataset.key,
                    "problem_type": dataset.problem_type,
                    "model": model,
                    "seed": int(seed),
                    "embedding": "RBF",
                    "k": 16,
                    "split": split,
                    "method": method,
                    "family": "embedding_G2" if method == FINALIST else "embedding_baseline",
                    "selected_alpha": float(alpha),
                    "raw_fallback": bool(alpha == 0.0),
                    "raw_disagreement": raw_d,
                    "method_disagreement": method_d,
                    "maximum_method_disagreement": float(np.max(method_ds)),
                    "disagreement_reduction": 0.0 if raw_d <= 1e-12 else 1.0 - method_d / raw_d,
                    "coordinate_error": maximum_coordinate_error,
                    "inference_passes": 1 if alpha in (0.0, 1.0) else 2,
                    "inference_multiplier": 1.0 if alpha in (0.0, 1.0) else 2.0,
                    "fit_seconds": 0.0 if alpha == 0.0 else float(gram_meta["telemetry"].get("fit_seconds", 0.0)),
                    "raw_task_error": task_error(dataset.problem_type, target, raw_reference),
                    "method_task_error": task_error(dataset.problem_type, target, method_reference),
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
        ROOT / "results" / "raw" / "prospective" / "embedding_units" / model
        / dataset.key / f"seed_{seed}.json"
    )
    write_json(
        path,
        {
            "status": "COMPLETE",
            "dataset": dataset.key,
            "model": model,
            "seed": int(seed),
            "embedding": "RBF",
            "k": 16,
            "finalist_sha256": finalist_hash,
            "selection_split": "validation_only",
            "test_outcomes_used_for_selection": False,
            "selected_alpha": float(selected_alpha),
            "guarded_g2_evidence": strip_samples(evidence),
            "maximum_coordinate_error": maximum_coordinate_error,
            "cells": rows,
        },
    )
    print(f"[prospective embedding] {dataset.key} {model} seed={seed} alpha={selected_alpha}", flush=True)
    return rows


def collect(specs: list[dict[str, Any]], models: list[str], seeds: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        for model in models:
            for seed in seeds:
                path = ROOT / "results" / "raw" / "prospective" / "embedding_units" / model / spec["key"] / f"seed_{seed}.json"
                if not path.exists():
                    raise FileNotFoundError(f"missing prospective embedding unit: {path}")
                payload = json.loads(path.read_text())
                if payload.get("status") != "COMPLETE":
                    raise RuntimeError(f"incomplete prospective embedding unit: {path}")
                rows.extend(payload["cells"])
    return rows


def aggregate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = frame[frame.split == "test"]
    units = (
        test.groupby(["dataset", "problem_type", "model", "embedding", "k", "method", "family"], as_index=False)
        .agg(
            selected_alpha=("selected_alpha", "median"),
            disagreement_reduction=("disagreement_reduction", "median"),
            normalized_excess_risk=("normalized_excess_risk", "median"),
            raw_task_error=("raw_loss", "median"),
            method_task_error=("method_loss", "median"),
            raw_fallback=("raw_fallback", "max"),
            inference_multiplier=("inference_multiplier", "median"),
        )
    )
    validation = (
        frame[frame.split == "validation"]
        .groupby(["dataset", "model", "method"], as_index=False)
        .normalized_excess_risk.median()
        .rename(columns={"normalized_excess_risk": "validation_C"})
    )
    units = units.merge(validation, on=["dataset", "model", "method"], how="left")
    units["predictive_rank"] = units.groupby(["dataset", "model"])["method_task_error"].rank(method="average")
    delta = units.method_task_error - units.raw_task_error
    units["task_outcome"] = np.where(delta < -1e-12, "W", np.where(delta > 1e-12, "L", "T"))
    rows: list[dict[str, Any]] = []
    for (method, family), values in units.groupby(["method", "family"], sort=False):
        costs = values.normalized_excess_risk.to_numpy(float)
        outcomes = values.task_outcome.value_counts()
        controls = values.disagreement_reduction.to_numpy(float)
        rows.append(
            {
                "method": method,
                "family": family,
                "units": len(values),
                "datasets": int(values.dataset.nunique()),
                "model_families": int(values.model.nunique()),
                "median_alpha": float(values.selected_alpha.median()),
                "p25_disagreement_reduction": float(np.quantile(controls, 0.25)),
                "median_disagreement_reduction": float(np.median(controls)),
                "p75_disagreement_reduction": float(np.quantile(controls, 0.75)),
                "median_C": float(np.median(costs)),
                "p90_C": float(np.quantile(costs, 0.90)),
                "p95_C": float(np.quantile(costs, 0.95)),
                "max_C": float(np.max(costs)),
                "wins": int(outcomes.get("W", 0)),
                "ties": int(outcomes.get("T", 0)),
                "losses": int(outcomes.get("L", 0)),
                "fraction_C_lt_0": float(np.mean(costs < 0)),
                "fraction_C_gt_0p01": float(np.mean(costs > 0.01)),
                "fraction_C_gt_0p05": float(np.mean(costs > 0.05)),
                "fallback_rate": float(values.raw_fallback.mean()),
                "mean_predictive_rank": float(values.predictive_rank.mean()),
                "inference_multiplier": float(values.inference_multiplier.median()),
            }
        )
    return units, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset")
    parser.add_argument("--model", choices=BACKBONES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    specs, finalists, finalist_hash = prospective_specs()
    protocol = load_protocol()
    specs = [spec for spec in specs if args.dataset is None or spec["key"] == args.dataset]
    models = [args.model] if args.model else list(BACKBONES)
    seeds = [args.seed] if args.seed is not None else [int(value) for value in protocol["prospective_seeds"]]
    if args.aggregate_only:
        rows = collect(specs, models, seeds)
    else:
        rows = []
        for spec in specs:
            for model in models:
                for seed in seeds:
                    rows.extend(run_unit(spec, model, seed, args.device, protocol, finalists, finalist_hash))
    processed = ROOT / "results" / "processed"
    filtered = args.dataset is not None or args.model is not None or args.seed is not None
    suffix = f"__{args.dataset or 'all'}__{args.model or 'all'}__{args.seed if args.seed is not None else 'all'}" if filtered else ""
    frame = pd.DataFrame(rows)
    frame.to_csv(processed / f"prospective_embedding_cells{suffix}.csv", index=False)
    if not filtered:
        units, summary = aggregate(frame)
        units.to_csv(processed / "prospective_embedding_units.csv", index=False)
        summary.to_csv(processed / "prospective_embedding_summary.csv", index=False)
        write_json(
            processed / "prospective_embedding_manifest.json",
            {
                "status": "COMPLETE",
                "finalist_sha256": finalist_hash,
                "cells": len(frame),
                "units": len(units),
                "datasets": len(specs),
                "models": len(models),
                "seeds": len(seeds),
                "embedding": "RBF",
                "k": 16,
                "prospective_outcomes_accessed": True,
            },
        )


if __name__ == "__main__":
    main()
