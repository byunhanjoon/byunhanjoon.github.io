#!/usr/bin/env python3
"""Run the hash-locked general-interface prospective evaluation."""

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
    gram_interface,
    greedy_candidates,
    select_greedy,
    target_free_descriptors,
)
from guarded_basis.common import (  # noqa: E402
    bd,
    disagreement,
    load_blocks,
    load_protocol,
    mix_predictions,
    normalized_excess_risk,
    orthogonal_orbit,
    prospective_base_predictions,
    prospective_rank_prediction,
    prospective_specs,
    task_error,
    write_json,
)
from guarded_basis.gating import guarded_evidence, select_g2, strip_samples  # noqa: E402
from run_blockguard import (  # noqa: E402
    evaluate_selection,
    fit_candidate_rows,
    one_block_interventions,
)
from safe_basis.gating import alpha_evidence as safe_evidence  # noqa: E402
from safe_basis.gating import select_gates as select_safe_gates  # noqa: E402


FIXED = {
    "Raw": 0.0,
    "Raw+Gram@0.5": 0.5,
    "Raw+Gram@0.75": 0.75,
    "PureGram": 1.0,
}
G2_METHOD = "GuardedGram-G2-g0p0-t01"
BLOCK_METHOD = "BlockGuard-Greedy-t01"


def evaluate_mixture(
    *,
    blocks: Any,
    orbit: list[Any],
    raw: dict[str, dict[str, np.ndarray]],
    invariant: dict[str, np.ndarray],
    alpha: float,
    method: str,
    family: str,
    model: str,
    seed: int,
    inference_passes: int,
) -> list[dict[str, Any]]:
    reference_id = orbit[0].representation_id
    rows: list[dict[str, Any]] = []
    for split, target in (("validation", blocks.dataset.y_validation), ("test", blocks.dataset.y_test)):
        raw_reference = raw[reference_id][split]
        method_reference = mix_predictions(raw_reference, invariant[split], alpha)
        raw_ds = [
            disagreement(blocks.dataset.problem_type, target, raw_reference, raw[rep.representation_id][split])
            for rep in orbit[1:]
        ]
        method_ds = [
            disagreement(
                blocks.dataset.problem_type,
                target,
                method_reference,
                mix_predictions(raw[rep.representation_id][split], invariant[split], alpha),
            )
            for rep in orbit[1:]
        ]
        raw_d = float(np.mean(raw_ds))
        method_d = float(np.mean(method_ds))
        rows.append(
            {
                "panel": "guarded_new_untouched_prospective",
                "dataset": blocks.dataset.key,
                "problem_type": blocks.dataset.problem_type,
                "model": model,
                "seed": int(seed),
                "method": method,
                "family": family,
                "split": split,
                "alpha": float(alpha),
                "selected_alpha": float(alpha),
                "invariant_feature_fraction": np.nan,
                "raw_fallback": bool(alpha == 0.0),
                "raw_disagreement": raw_d,
                "method_disagreement": method_d,
                "maximum_method_disagreement": float(np.max(method_ds)),
                "disagreement_reduction": 0.0 if raw_d <= 1e-12 else 1.0 - method_d / raw_d,
                "inference_passes": int(inference_passes),
                "inference_multiplier": float(inference_passes),
                "raw_task_error": task_error(blocks.dataset.problem_type, target, raw_reference),
                "method_task_error": task_error(blocks.dataset.problem_type, target, method_reference),
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


def general_selections(
    *,
    blocks: Any,
    raw_validation: np.ndarray,
    gram_validation: np.ndarray,
    rank_validation: np.ndarray,
    model: str,
    seed: int,
    finalist_names: set[str],
) -> tuple[dict[str, tuple[float, str, str]], dict[str, Any]]:
    evidence = guarded_evidence(
        blocks.dataset.problem_type,
        blocks.dataset.y_validation,
        blocks.dataset.y_train,
        raw_validation,
        gram_validation,
        alphas=[0.75, 0.5, 0.25, 0.0],
        resamples=1000,
        seed=bd.stable_seed("GuardedGram", blocks.dataset.key, model, seed),
    )
    methods: dict[str, tuple[float, str, str]] = {
        method: (alpha, "fixed", "Gram") for method, alpha in FIXED.items()
    }
    if G2_METHOD in finalist_names:
        methods[G2_METHOD] = (select_g2(evidence, tau=0.01, gamma=0.0), "G2", "Gram")

    safe_records = safe_evidence(
        blocks.dataset.problem_type,
        blocks.dataset.y_validation,
        blocks.dataset.y_train,
        raw_validation,
        gram_validation,
        alphas=[0.0, 0.25, 0.5, 0.75, 1.0],
        bootstrap_resamples=500,
        seed=bd.stable_seed("FrozenSafeGram", blocks.dataset.key, model, seed),
        epsilon=1e-8,
    )
    safe_alpha = select_safe_gates(
        safe_records, taus=[0.01], constrained_lambda_multipliers=[]
    )["SafeGram-t01"]
    rank_records = safe_evidence(
        blocks.dataset.problem_type,
        blocks.dataset.y_validation,
        blocks.dataset.y_train,
        raw_validation,
        rank_validation,
        alphas=[0.0, 0.25, 0.5, 0.75, 1.0],
        bootstrap_resamples=500,
        seed=bd.stable_seed("FrozenSafeRankGram", blocks.dataset.key, model, seed),
        epsilon=1e-8,
    )
    safe_rank_alpha = select_safe_gates(
        rank_records, taus=[0.01], constrained_lambda_multipliers=[]
    )["SafeGram-t01"]
    methods["SafeGram-t01"] = (safe_alpha, "safety_reference", "Gram")
    methods["SafeRankGram-t01"] = (safe_rank_alpha, "safety_reference", "Rank")
    return methods, {
        "guarded_g2_evidence": strip_samples(evidence),
        "safe_gram_evidence": safe_records,
        "safe_rank_evidence": rank_records,
        "selected_alphas": {key: float(value[0]) for key, value in methods.items()},
    }


def run_unit(
    spec: dict[str, Any],
    model: str,
    seed: int,
    device: str,
    protocol: dict[str, Any],
    finalists: dict[str, Any],
    finalist_hash: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    # prospective_specs() has already validated every lock before this load.
    blocks = load_blocks(spec, protocol)
    orbit = orthogonal_orbit(blocks, protocol)
    gram_orbit = [gram_interface(rep, blocks.dataset.key) for rep in orbit]
    raw, gram, source = prospective_base_predictions(model, blocks, orbit, seed, device, finalist_hash)
    rank, _ = prospective_rank_prediction(model, blocks, orbit[0], seed, device, finalist_hash)
    reference_id = orbit[0].representation_id
    finalist_names = {str(row["method"]) for row in finalists["finalists"]}
    selections, evidence = general_selections(
        blocks=blocks,
        raw_validation=raw[reference_id]["validation"],
        gram_validation=gram["validation"],
        rank_validation=rank["validation"],
        model=model,
        seed=seed,
        finalist_names=finalist_names,
    )
    cells: list[dict[str, Any]] = []
    for method, (alpha, family, branch) in selections.items():
        cells.extend(
            evaluate_mixture(
                blocks=blocks,
                orbit=orbit,
                raw=raw,
                invariant=rank if branch == "Rank" else gram,
                alpha=alpha,
                method=method,
                family=family,
                model=model,
                seed=seed,
                inference_passes=1 if alpha in (0.0, 1.0) else 2,
            )
        )

    interventions: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    block_selection: dict[str, Any] | None = None
    if BLOCK_METHOD in finalist_names:
        cache_root = ROOT / "results" / "raw" / "prospective" / "blockguard"
        descriptors = target_free_descriptors(orbit[0], orbit)
        interventions = one_block_interventions(
            blocks=blocks,
            orbit=orbit,
            gram_orbit=gram_orbit,
            raw=raw,
            gram=gram,
            model=model,
            seed=seed,
            device=device,
            descriptors=descriptors,
            cache_root=cache_root,
            finalist_hash=finalist_hash,
        )
        candidates = greedy_candidates(
            interventions,
            maximum_stages=int(protocol["block_guard"]["maximum_greedy_stages"]),
        )
        candidate_rows = fit_candidate_rows(
            candidates=candidates,
            family="BlockGuard-Greedy",
            blocks=blocks,
            orbit=orbit,
            gram_orbit=gram_orbit,
            raw=raw,
            gram=gram,
            model=model,
            seed=seed,
            device=device,
            cache_root=cache_root,
            finalist_hash=finalist_hash,
        )
        selected = select_greedy(candidate_rows, tau=0.01)
        block_cells, coordinate = evaluate_selection(
            method=BLOCK_METHOD,
            family="BlockGuard-Greedy",
            tau=0.01,
            candidate=selected,
            blocks=blocks,
            orbit=orbit,
            gram_orbit=gram_orbit,
            raw=raw,
            gram=gram,
            model=model,
            seed=seed,
            device=device,
            cache_root=cache_root,
            finalist_hash=finalist_hash,
            panel="guarded_new_untouched_prospective",
        )
        for row in block_cells:
            row["selected_alpha"] = np.nan
            row["inference_multiplier"] = 1.0
            row["raw_task_error"] = row["raw_loss"]
            row["method_task_error"] = row["method_loss"]
        cells.extend(block_cells)
        block_selection = {
            "candidate": selected["candidate"],
            "features": list(selected["features"]),
            "validation_C": float(selected["validation_C"]),
            "coordinate_audit": coordinate,
        }

    unit_path = (
        ROOT / "results" / "raw" / "prospective" / "units" / model
        / blocks.dataset.key / f"seed_{seed}.json"
    )
    write_json(
        unit_path,
        {
            "status": "COMPLETE",
            "dataset": blocks.dataset.key,
            "model": model,
            "seed": int(seed),
            "finalist_sha256": finalist_hash,
            "selection_split": "validation_only",
            "test_outcomes_used_for_selection": False,
            "source": source,
            "evidence": evidence,
            "block_selection": block_selection,
            "cells": cells,
            "one_block_rows": interventions,
            "candidate_rows": candidate_rows,
        },
    )
    print(
        f"[prospective] {blocks.dataset.key} {model} seed={seed} "
        f"G2={evidence['selected_alphas'].get(G2_METHOD)} "
        f"Block={None if block_selection is None else block_selection['candidate']}",
        flush=True,
    )
    return cells, interventions, candidate_rows


def collect(
    specs: list[dict[str, Any]], models: list[str], seeds: list[int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cells: list[dict[str, Any]] = []
    interventions: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for spec in specs:
        for model in models:
            for seed in seeds:
                path = ROOT / "results" / "raw" / "prospective" / "units" / model / spec["key"] / f"seed_{seed}.json"
                if not path.exists():
                    raise FileNotFoundError(f"missing prospective unit: {path}")
                payload = json.loads(path.read_text())
                if payload.get("status") != "COMPLETE":
                    raise RuntimeError(f"incomplete prospective unit: {path}")
                cells.extend(payload["cells"])
                interventions.extend(payload["one_block_rows"])
                candidates.extend(
                    {
                        **row,
                        "dataset": payload["dataset"],
                        "model": payload["model"],
                        "seed": int(payload["seed"]),
                    }
                    for row in payload["candidate_rows"]
                )
    return cells, interventions, candidates


def aggregate(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = cells[cells.split == "test"].copy()
    units = (
        test.groupby(["dataset", "problem_type", "model", "method", "family"], as_index=False, dropna=False)
        .agg(
            selected_alpha=("selected_alpha", "median"),
            invariant_feature_fraction=("invariant_feature_fraction", "median"),
            disagreement_reduction=("disagreement_reduction", "median"),
            normalized_excess_risk=("normalized_excess_risk", "median"),
            validation_C=("normalized_excess_risk", "median"),
            raw_task_error=("raw_loss", "median"),
            method_task_error=("method_loss", "median"),
            raw_fallback=("raw_fallback", "max"),
            inference_multiplier=("inference_multiplier", "median"),
        )
    )
    validation = (
        cells[cells.split == "validation"]
        .groupby(["dataset", "model", "method"], as_index=False)
        .normalized_excess_risk.median()
        .rename(columns={"normalized_excess_risk": "validation_C_actual"})
    )
    units = units.drop(columns=["validation_C"]).merge(validation, on=["dataset", "model", "method"], how="left")
    units["predictive_rank"] = units.groupby(["dataset", "model"])["method_task_error"].rank(method="average")
    delta = units.method_task_error - units.raw_task_error
    units["task_outcome"] = np.where(delta < -1e-12, "W", np.where(delta > 1e-12, "L", "T"))

    rows: list[dict[str, Any]] = []
    for (method, family), frame in units.groupby(["method", "family"], sort=False):
        costs = frame.normalized_excess_risk.to_numpy(float)
        controls = frame.disagreement_reduction.to_numpy(float)
        outcomes = frame.task_outcome.value_counts()
        rows.append(
            {
                "method": method,
                "family": family,
                "units": len(frame),
                "datasets": int(frame.dataset.nunique()),
                "model_families": int(frame.model.nunique()),
                "median_alpha": float(frame.selected_alpha.median()) if frame.selected_alpha.notna().any() else np.nan,
                "median_invariant_feature_fraction": float(frame.invariant_feature_fraction.median()) if frame.invariant_feature_fraction.notna().any() else np.nan,
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
                "fallback_rate": float(frame.raw_fallback.mean()),
                "mean_predictive_rank": float(frame.predictive_rank.mean()),
                "inference_multiplier": float(frame.inference_multiplier.median()),
            }
        )
    return units, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset")
    parser.add_argument("--model")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    if not args.aggregate_only and args.dataset is not None and args.model is not None and args.seed is not None:
        existing = (
            ROOT / "results" / "raw" / "prospective" / "units" / args.model
            / args.dataset / f"seed_{args.seed}.json"
        )
        if existing.exists() and json.loads(existing.read_text()).get("status") == "COMPLETE":
            print(f"[prospective skip] durable unit already complete: {existing}", flush=True)
            return
    specs, finalists, finalist_hash = prospective_specs()
    protocol = load_protocol()
    specs = [spec for spec in specs if args.dataset is None or spec["key"] == args.dataset]
    models = [args.model] if args.model else list(protocol["general_models"])
    seeds = [args.seed] if args.seed is not None else [int(value) for value in protocol["prospective_seeds"]]

    if args.aggregate_only:
        cells, interventions, candidates = collect(specs, models, seeds)
    else:
        cells, interventions, candidates = [], [], []
        for spec in specs:
            for model in models:
                for seed in seeds:
                    unit = run_unit(spec, model, seed, args.device, protocol, finalists, finalist_hash)
                    cells.extend(unit[0])
                    interventions.extend(unit[1])
                    candidates.extend(unit[2])

    processed = ROOT / "results" / "processed"
    filtered = args.dataset is not None or args.model is not None or args.seed is not None
    suffix = f"__{args.dataset or 'all'}__{args.model or 'all'}__{args.seed if args.seed is not None else 'all'}" if filtered else ""
    cell_frame = pd.DataFrame(cells)
    cell_frame.to_csv(processed / f"prospective_general_cells{suffix}.csv", index=False)
    pd.DataFrame(interventions).to_csv(processed / f"prospective_one_block{suffix}.csv", index=False)
    pd.DataFrame(candidates).to_csv(processed / f"prospective_block_candidates{suffix}.csv", index=False)
    if not filtered:
        units, summary = aggregate(cell_frame)
        units.to_csv(processed / "prospective_general_units.csv", index=False)
        summary.to_csv(processed / "prospective_general_summary.csv", index=False)
        write_json(
            processed / "prospective_general_manifest.json",
            {
                "status": "COMPLETE",
                "finalist_sha256": finalist_hash,
                "cells": len(cell_frame),
                "one_block_rows": len(interventions),
                "candidate_rows": len(candidates),
                "units": len(units),
                "datasets": len(specs),
                "models": len(models),
                "seeds": len(seeds),
                "prospective_outcomes_accessed": True,
            },
        )


if __name__ == "__main__":
    main()
