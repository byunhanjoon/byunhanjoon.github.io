#!/usr/bin/env python3
"""Evaluate SafeGram and all preregistered gate ablations on development data."""

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
    load_protocol,
    mix_predictions,
    normalized_excess_risk,
    orthogonal_orbit,
    sha256_file,
    task_error,
    write_json,
)
from safe_basis.gating import alpha_evidence, select_gates, verify_alpha_zero  # noqa: E402


FIXED = {
    "Raw": 0.0,
    "Raw+GramAnchor@0.5": 0.5,
    "Raw+GramAnchor@0.75": 0.75,
    "GramAnchor": 1.0,
}


def evaluate_method(
    *,
    blocks: Any,
    orbit: list[Any],
    raw: dict[str, dict[str, np.ndarray]],
    gram: dict[str, np.ndarray],
    alpha: float,
    method: str,
    model: str,
    seed: int,
    gate_family: str,
) -> list[dict[str, Any]]:
    reference_id = orbit[0].representation_id
    if set(raw) != {rep.representation_id for rep in orbit}:
        raise RuntimeError(f"raw orbit ID drift for {blocks.dataset.key}/{model}/seed={seed}")
    records = []
    for split, target in (("validation", blocks.dataset.y_validation), ("test", blocks.dataset.y_test)):
        raw_reference = raw[reference_id][split]
        method_reference = mix_predictions(raw_reference, gram[split], alpha)
        raw_disagreements = [
            disagreement(blocks.dataset.problem_type, target, raw_reference, raw[rep.representation_id][split])
            for rep in orbit[1:]
        ]
        method_disagreements = [
            disagreement(
                blocks.dataset.problem_type,
                target,
                method_reference,
                mix_predictions(raw[rep.representation_id][split], gram[split], alpha),
            )
            for rep in orbit[1:]
        ]
        raw_disagreement = float(np.mean(raw_disagreements))
        method_disagreement = float(np.mean(method_disagreements))
        reduction = 0.0 if raw_disagreement <= 1e-12 else 1.0 - method_disagreement / raw_disagreement
        safety = normalized_excess_risk(
            blocks.dataset.problem_type,
            target,
            raw_reference,
            method_reference,
            blocks.dataset.y_train,
            epsilon=1e-8,
        )
        orbit_costs = []
        orbit_task = []
        for rep in orbit:
            raw_prediction = raw[rep.representation_id][split]
            method_prediction = mix_predictions(raw_prediction, gram[split], alpha)
            orbit_costs.append(
                normalized_excess_risk(
                    blocks.dataset.problem_type,
                    target,
                    raw_prediction,
                    method_prediction,
                    blocks.dataset.y_train,
                    epsilon=1e-8,
                )["normalized_excess_risk"]
            )
            orbit_task.append(task_error(blocks.dataset.problem_type, target, method_prediction))
        records.append(
            {
                "panel": "development",
                "dataset": blocks.dataset.key,
                "problem_type": blocks.dataset.problem_type,
                "model": model,
                "seed": int(seed),
                "method": method,
                "gate_family": gate_family,
                "split": split,
                "alpha": float(alpha),
                "raw_fallback": bool(alpha == 0.0),
                "raw_disagreement": raw_disagreement,
                "method_disagreement": method_disagreement,
                "maximum_method_disagreement": float(np.max(method_disagreements)),
                "disagreement_reduction": float(reduction),
                "orbit_mean_task_loss": float(np.mean(orbit_task)),
                "orbit_max_C": float(np.max(orbit_costs)),
                **safety,
            }
        )
    return records


def run_unit(spec: dict[str, Any], model: str, seed: int, protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    blocks = load_blocks(spec, protocol)
    orbit = orthogonal_orbit(blocks, protocol)
    raw, gram, source = load_frozen_development_predictions(model, blocks.dataset.key, seed)
    reference_id = orbit[0].representation_id
    gate_config = protocol["safe_alpha"]
    evidence = alpha_evidence(
        blocks.dataset.problem_type,
        blocks.dataset.y_validation,
        blocks.dataset.y_train,
        raw[reference_id]["validation"],
        gram["validation"],
        alphas=[float(value) for value in gate_config["alphas"]],
        bootstrap_resamples=int(gate_config["bootstrap_resamples"]),
        seed=bd.stable_seed("SafeGram", blocks.dataset.key, model, seed),
        epsilon=float(gate_config["epsilon"]),
    )
    verify_alpha_zero(evidence)
    selections = select_gates(
        evidence,
        taus=[float(value) for value in gate_config["taus"]],
        constrained_lambda_multipliers=[float(value) for value in gate_config["constrained_lambda_multipliers"]],
    )
    methods = {**FIXED, **selections}
    rows = []
    for method, alpha in methods.items():
        family = "fixed" if method in FIXED else method.split("-", 1)[0]
        rows.extend(
            evaluate_method(
                blocks=blocks,
                orbit=orbit,
                raw=raw,
                gram=gram,
                alpha=alpha,
                method=method,
                model=model,
                seed=seed,
                gate_family=family,
            )
        )
    evidence_rows = [
        {
            "dataset": blocks.dataset.key,
            "problem_type": blocks.dataset.problem_type,
            "model": model,
            "seed": int(seed),
            **record,
        }
        for record in evidence
    ]
    destination = ROOT / "results" / "raw" / "development_gates" / model / blocks.dataset.key / f"seed_{seed}.json"
    write_json(
        destination,
        {
            "status": "COMPLETE",
            "dataset": blocks.dataset.key,
            "model": model,
            "seed": int(seed),
            "selection_split": "validation_only",
            "test_outcomes_used_for_alpha_selection": False,
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "prospective_panel_sha256": sha256_file(PANEL_PATH),
            "source_artifacts": source,
            "alpha_evidence": evidence,
            "selected_alphas": selections,
        },
    )
    print(f"[gates] {blocks.dataset.key} {model} seed={seed} SafeGram-t01={selections['SafeGram-t01']}", flush=True)
    return rows, evidence_rows


def aggregate(cells: pd.DataFrame) -> pd.DataFrame:
    test = cells[cells["split"] == "test"]
    units = (
        test.groupby(["dataset", "problem_type", "model", "method", "gate_family"], as_index=False)
        .agg(
            alpha=("alpha", "median"),
            disagreement_reduction=("disagreement_reduction", "median"),
            normalized_excess_risk=("normalized_excess_risk", "median"),
            absolute_task_difference=("absolute_task_difference", "median"),
            relative_task_difference=("relative_task_difference", "median"),
            denominator_sensitive=("denominator_sensitive", "max"),
            raw_fallback=("raw_fallback", "max"),
        )
    )
    summary = []
    for (method, family), frame in units.groupby(["method", "gate_family"], sort=False):
        costs = frame["normalized_excess_risk"].to_numpy(float)
        summary.append(
            {
                "method": method,
                "gate_family": family,
                "units": len(frame),
                "datasets": frame["dataset"].nunique(),
                "model_families": frame["model"].nunique(),
                "median_alpha": float(frame["alpha"].median()),
                "median_disagreement_reduction": float(frame["disagreement_reduction"].median()),
                "median_C": float(np.median(costs)),
                "p90_C": float(np.quantile(costs, 0.90)),
                "p95_C": float(np.quantile(costs, 0.95)),
                "max_C": float(np.max(costs)),
                "raw_fallback_rate": float(frame["raw_fallback"].mean()),
                "task_improvement_fraction": float((frame["absolute_task_difference"] < 0).mean()),
            }
        )
    return units, pd.DataFrame(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--model")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    protocol = load_protocol()
    specs = [row for row in development_specs(protocol) if args.dataset is None or row["key"] == args.dataset]
    models = [args.model] if args.model else list(protocol["development_models"])
    seeds = [args.seed] if args.seed is not None else [int(value) for value in protocol["model_seeds"]]
    all_cells: list[dict[str, Any]] = []
    all_evidence: list[dict[str, Any]] = []
    for spec in specs:
        for model in models:
            for seed in seeds:
                cells, evidence = run_unit(spec, model, seed, protocol)
                all_cells.extend(cells)
                all_evidence.extend(evidence)
    processed = ROOT / "results" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    cells_frame = pd.DataFrame(all_cells)
    evidence_frame = pd.DataFrame(all_evidence)
    if args.dataset or args.model or args.seed is not None:
        suffix = f"{args.dataset or 'all'}__{args.model or 'all'}__{args.seed if args.seed is not None else 'all'}"
        cells_frame.to_csv(processed / f"development_gate_cells__{suffix}.csv", index=False)
        evidence_frame.to_csv(processed / f"development_gate_alpha_evidence__{suffix}.csv", index=False)
    else:
        cells_frame.to_csv(processed / "development_gate_cells.csv", index=False)
        evidence_frame.to_csv(processed / "development_gate_alpha_evidence.csv", index=False)
        units, summary = aggregate(cells_frame)
        units.to_csv(processed / "development_gate_units.csv", index=False)
        summary.to_csv(processed / "development_gate_summary.csv", index=False)
        write_json(
            processed / "development_gate_manifest.json",
            {
                "status": "COMPLETE",
                "cells": len(cells_frame),
                "evidence_rows": len(evidence_frame),
                "units": len(units),
                "selection_split": "validation_only",
                "prospective_data_accessed": False,
            },
        )


if __name__ == "__main__":
    main()
