#!/usr/bin/env python3
"""Run single-model DualViewGram development variants."""

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
    PANEL_PATH,
    PROTOCOL_PATH,
    development_base_predictions,
    development_specs,
    disagreement,
    frozen_hashes,
    load_blocks,
    load_prediction_bundle,
    load_protocol,
    normalized_excess_risk,
    orthogonal_orbit,
    save_prediction_bundle,
    sha256_file,
    write_json,
)
from guarded_basis.dualview import fit_dualview_predictions  # noqa: E402


def fixed_variant(alpha: float) -> dict[str, Any]:
    label = str(alpha).replace("0.", "")
    return {
        "method": f"DualViewGram-D1-a{label}",
        "family": "DualView-D1",
        "gate_kind": "fixed",
        "global_alpha": float(alpha),
        "gate_penalty": 0.0,
    }


def variants(stage: str) -> list[dict[str, Any]]:
    if stage == "stage1":
        return [fixed_variant(0.75)]
    selection_path = ROOT / "results" / "processed" / "dualview_stage1_selection.json"
    if not selection_path.exists():
        raise RuntimeError("run DualView stage1 before full development")
    selection = json.loads(selection_path.read_text())
    if not selection["survived"]:
        return [fixed_variant(0.75)]
    # D1=.5, D1=.75 are the simple fixed controls. D2--D4 are added by the
    # full-development expansion only after their upstream development rules exist.
    return [fixed_variant(0.5), fixed_variant(0.75)]


def cached_dualview(
    *,
    path: Path,
    blocks: Any,
    raw_rep: Any,
    gram_rep: Any,
    model: str,
    seed: int,
    device: str,
    variant: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    fixed_gates = {
        feature: float(variant["global_alpha"])
        for feature in raw_rep.feature_blocks
    }
    definition = {
        **variant,
        "representation_id": raw_rep.representation_id,
        "hidden_width_per_feature": int(protocol["dual_view"]["hidden_width_per_feature"]),
        "anchors": 16,
        "anchor_selection": "gram_pivot",
        "normalize": True,
        "architecture": protocol["model_hpo"][model],
    }
    hashes = frozen_hashes()
    if path.exists() and path.with_suffix(".json").exists():
        prediction, metadata = load_prediction_bundle(path)
        if metadata.get("frozen_hashes") != hashes or metadata.get("definition") != definition:
            raise RuntimeError(f"DualView cache drift at {path}")
        return prediction, metadata
    prediction, telemetry = fit_dualview_predictions(
        problem_type=blocks.dataset.problem_type,
        raw_rep=raw_rep,
        gram_rep=gram_rep,
        y_train=blocks.dataset.y_train,
        y_validation=blocks.dataset.y_validation,
        seed=seed,
        device=device,
        backbone=model,
        gate_kind=str(variant["gate_kind"]),
        fixed_gates=fixed_gates,
        gate_penalty=float(variant["gate_penalty"]),
        inference_warmups=int(protocol["efficiency"]["warmup_repeats"]),
        inference_repeats=int(protocol["efficiency"]["inference_repeats"]),
    )
    metadata = {
        "status": "COMPLETE",
        "dataset": blocks.dataset.key,
        "model": model,
        "seed": int(seed),
        "definition": definition,
        "frozen_hashes": hashes,
        "telemetry": telemetry,
    }
    save_prediction_bundle(path, None, prediction["validation"], prediction["test"], metadata)
    return prediction, metadata


def evaluate(
    *,
    blocks: Any,
    orbit: list[Any],
    raw: dict[str, dict[str, np.ndarray]],
    predictions: list[dict[str, np.ndarray]],
    metadata: dict[str, Any],
    variant: dict[str, Any],
    model: str,
    seed: int,
) -> list[dict[str, Any]]:
    telemetry = metadata["telemetry"]
    rows: list[dict[str, Any]] = []
    for split, target in (("validation", blocks.dataset.y_validation), ("test", blocks.dataset.y_test)):
        raw_reference = raw[orbit[0].representation_id][split]
        method_reference = predictions[0][split]
        raw_ds = [
            disagreement(
                blocks.dataset.problem_type,
                target,
                raw_reference,
                raw[rep.representation_id][split],
            )
            for rep in orbit[1:]
        ]
        method_ds = [
            disagreement(
                blocks.dataset.problem_type,
                target,
                method_reference,
                predictions[index][split],
            )
            for index in range(1, len(predictions))
        ]
        raw_d, method_d = float(np.mean(raw_ds)), float(np.mean(method_ds))
        rows.append(
            {
                "panel": "guarded_development",
                "dataset": blocks.dataset.key,
                "problem_type": blocks.dataset.problem_type,
                "model": model,
                "seed": int(seed),
                "method": variant["method"],
                "family": variant["family"],
                "split": split,
                "global_alpha": float(variant["global_alpha"]),
                "gate_kind": variant["gate_kind"],
                "gate_penalty": float(variant["gate_penalty"]),
                "raw_disagreement": raw_d,
                "method_disagreement": method_d,
                "maximum_method_disagreement": float(np.max(method_ds)),
                "disagreement_reduction": 0.0 if raw_d <= 1e-12 else 1.0 - method_d / raw_d,
                "inference_passes": 1,
                "single_model": True,
                "fit_seconds": float(telemetry["fit_seconds"]),
                "mean_inference_seconds": float(telemetry["mean_inference_seconds"]),
                "parameter_count": int(telemetry["parameter_count"]),
                "peak_gpu_memory_bytes": int(telemetry["peak_gpu_memory_bytes"]),
                "mean_gate": float(telemetry["mean_gate"]),
                **normalized_excess_risk(
                    blocks.dataset.problem_type,
                    target,
                    raw_reference,
                    method_reference,
                    blocks.dataset.y_train,
                    epsilon=1e-8,
                ),
            }
        )
    return rows


def run_unit(
    spec: dict[str, Any],
    model: str,
    seed: int,
    device: str,
    protocol: dict[str, Any],
    stage: str,
) -> list[dict[str, Any]]:
    blocks = load_blocks(spec, protocol)
    orbit = orthogonal_orbit(blocks, protocol)
    gram_orbit = [gram_interface(rep, blocks.dataset.key) for rep in orbit]
    raw, _, source = development_base_predictions(model, blocks, orbit, seed, device)
    cells: list[dict[str, Any]] = []
    coordinate_errors = [
        float(
            np.linalg.norm(gram_orbit[index].X_train - gram_orbit[0].X_train)
            / max(np.linalg.norm(gram_orbit[0].X_train), np.finfo(float).tiny)
        )
        for index in range(1, len(gram_orbit))
    ]
    if max(coordinate_errors) >= 1e-6:
        raise RuntimeError(f"DualView Gram-coordinate audit failed for {blocks.dataset.key}")
    variant_audits: dict[str, Any] = {}
    for variant in variants(stage):
        predictions: list[dict[str, np.ndarray]] = []
        reference_metadata: dict[str, Any] | None = None
        for raw_rep, gram_rep in zip(orbit, gram_orbit):
            path = (
                ROOT
                / "results"
                / "raw"
                / "dualview"
                / "predictions"
                / model
                / blocks.dataset.key
                / f"seed_{seed}"
                / str(variant["method"])
                / f"{raw_rep.representation_id}.npz"
            )
            prediction, metadata = cached_dualview(
                path=path,
                blocks=blocks,
                raw_rep=raw_rep,
                gram_rep=gram_rep,
                model=model,
                seed=seed,
                device=device,
                variant=variant,
                protocol=protocol,
            )
            predictions.append(prediction)
            if raw_rep.is_reference:
                reference_metadata = metadata
        assert reference_metadata is not None
        cells.extend(
            evaluate(
                blocks=blocks,
                orbit=orbit,
                raw=raw,
                predictions=predictions,
                metadata=reference_metadata,
                variant=variant,
                model=model,
                seed=seed,
            )
        )
        variant_audits[str(variant["method"])] = {
            "reference_telemetry": reference_metadata["telemetry"],
            "maximum_gram_coordinate_relative_error": max(coordinate_errors),
        }
    path = ROOT / "results" / "raw" / "dualview" / stage / model / blocks.dataset.key / f"seed_{seed}.json"
    write_json(
        path,
        {
            "status": "COMPLETE",
            "stage": stage,
            "dataset": blocks.dataset.key,
            "model": model,
            "seed": int(seed),
            "selection_split": "validation_only",
            "test_outcomes_used_for_selection": False,
            "prospective_outcomes_accessed": False,
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "panel_sha256": sha256_file(PANEL_PATH),
            "source": source,
            "variant_audits": variant_audits,
            "cells": cells,
        },
    )
    print(f"[dualview] {blocks.dataset.key} {model} seed={seed} stage={stage}", flush=True)
    return cells


def aggregate(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = cells[cells.split == "test"]
    units = (
        test.groupby(["dataset", "problem_type", "model", "method", "family"], as_index=False)
        .agg(
            global_alpha=("global_alpha", "median"),
            disagreement_reduction=("disagreement_reduction", "median"),
            normalized_excess_risk=("normalized_excess_risk", "median"),
            absolute_task_difference=("absolute_task_difference", "median"),
            method_loss=("method_loss", "median"),
            fit_seconds=("fit_seconds", "median"),
            mean_inference_seconds=("mean_inference_seconds", "median"),
            parameter_count=("parameter_count", "median"),
            peak_gpu_memory_bytes=("peak_gpu_memory_bytes", "max"),
            mean_gate=("mean_gate", "median"),
        )
    )
    units["predictive_rank"] = units.groupby(["dataset", "model"])["method_loss"].rank(method="average")
    rows: list[dict[str, Any]] = []
    for (method, family), frame in units.groupby(["method", "family"], sort=False):
        costs = frame.normalized_excess_risk.to_numpy(float)
        rows.append(
            {
                "method": method,
                "family": family,
                "units": len(frame),
                "datasets": frame.dataset.nunique(),
                "model_families": frame.model.nunique(),
                "global_alpha": float(frame.global_alpha.median()),
                "median_disagreement_reduction": float(frame.disagreement_reduction.median()),
                "median_C": float(np.median(costs)),
                "p90_C": float(np.quantile(costs, 0.90)),
                "p95_C": float(np.quantile(costs, 0.95)),
                "max_C": float(np.max(costs)),
                "median_fit_seconds": float(frame.fit_seconds.median()),
                "median_inference_seconds": float(frame.mean_inference_seconds.median()),
                "median_parameter_count": float(frame.parameter_count.median()),
                "max_peak_gpu_memory_bytes": float(frame.peak_gpu_memory_bytes.max()),
                "mean_predictive_rank": float(frame.predictive_rank.mean()),
            }
        )
    return units, pd.DataFrame(rows)


def collect(stage: str, specs: list[dict[str, Any]], models: list[str], seeds: list[int]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for spec in specs:
        for model in models:
            for seed in seeds:
                path = ROOT / "results" / "raw" / "dualview" / stage / model / spec["key"] / f"seed_{seed}.json"
                if not path.exists():
                    raise FileNotFoundError(f"missing DualView unit: {path}")
                payload = json.loads(path.read_text())
                cells.extend(payload["cells"])
    return cells


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["stage1", "full"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset")
    parser.add_argument("--model", choices=["controlled_mlp", "tabm_d", "resnet_tabular"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol()
    specs = development_specs(protocol)
    if args.stage == "stage1":
        names = set(protocol["stage1"]["datasets"])
        specs = [spec for spec in specs if spec["key"] in names]
    specs = [spec for spec in specs if args.dataset is None or spec["key"] == args.dataset]
    models = [args.model] if args.model else list(protocol["dual_view"]["backbones"])
    seeds = [args.seed] if args.seed is not None else [int(seed) for seed in protocol["development_seeds"]]
    if args.aggregate_only:
        cells = collect(args.stage, specs, models, seeds)
    else:
        cells = []
        for spec in specs:
            for model in models:
                for seed in seeds:
                    cells.extend(run_unit(spec, model, seed, args.device, protocol, args.stage))
    processed = ROOT / "results" / "processed"
    filtered = args.dataset is not None or args.model is not None or args.seed is not None
    suffix = (
        f"__{args.dataset or 'all'}__{args.model or 'all'}__{args.seed if args.seed is not None else 'all'}"
        if filtered else ""
    )
    prefix = f"dualview_{args.stage}"
    frame = pd.DataFrame(cells)
    frame.to_csv(processed / f"{prefix}_cells{suffix}.csv", index=False)
    if not filtered:
        units, summary = aggregate(frame)
        units.to_csv(processed / f"{prefix}_units.csv", index=False)
        summary.to_csv(processed / f"{prefix}_summary.csv", index=False)
        if args.stage == "stage1":
            row = summary.iloc[0]
            survived = bool(
                row.median_disagreement_reduction >= 0.50
                and row.p95_C <= 0.03
                and row.max_C <= 0.10
            )
            write_json(
                processed / "dualview_stage1_selection.json",
                {
                    "status": "DEVELOPMENT_SELECTED",
                    "selection_scope": "four frozen stage-1 development datasets",
                    "selection_split": "development_only",
                    "prospective_outcomes_accessed": False,
                    "selected": str(row.method),
                    "survived": survived,
                    "metrics": row.to_dict(),
                },
            )
            print(f"[dualview stage1] survived={survived}")
        write_json(
            processed / f"{prefix}_manifest.json",
            {
                "status": "COMPLETE",
                "stage": args.stage,
                "cells": len(frame),
                "units": len(units),
                "datasets": len(specs),
                "models": len(models),
                "seeds": len(seeds),
                "prospective_outcomes_accessed": False,
            },
        )


if __name__ == "__main__":
    main()
