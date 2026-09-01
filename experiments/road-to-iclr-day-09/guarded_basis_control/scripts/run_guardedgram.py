#!/usr/bin/env python3
"""Run GuardedGram G1/G2/G3 and frozen gating baselines on development."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from guarded_basis.common import (  # noqa: E402
    PANEL_PATH,
    PROTOCOL_PATH,
    bd,
    development_base_predictions,
    development_specs,
    disagreement,
    load_blocks,
    load_protocol,
    mix_predictions,
    normalized_excess_risk,
    orthogonal_orbit,
    rank_prediction,
    sha256_file,
    task_error,
    write_json,
)
from guarded_basis.gating import (  # noqa: E402
    guarded_evidence,
    select_g1,
    select_g2,
    select_g3,
    strip_samples,
)
from safe_basis.gating import alpha_evidence as safe_evidence  # noqa: E402
from safe_basis.gating import select_gates as select_safe_gates  # noqa: E402


FIXED = {
    "Raw": 0.0,
    "Raw+Gram@0.5": 0.5,
    "Raw+Gram@0.75": 0.75,
    "PureGram": 1.0,
}


def evaluate(
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
    fit_seconds: float,
) -> list[dict[str, Any]]:
    reference_id = orbit[0].representation_id
    rows = []
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
        safety = normalized_excess_risk(
            blocks.dataset.problem_type,
            target,
            raw_reference,
            method_reference,
            blocks.dataset.y_train,
            epsilon=1e-8,
        )
        rows.append(
            {
                "panel": "guarded_development",
                "dataset": blocks.dataset.key,
                "problem_type": blocks.dataset.problem_type,
                "model": model,
                "seed": int(seed),
                "method": method,
                "family": family,
                "split": split,
                "alpha": float(alpha),
                "raw_fallback": bool(alpha == 0.0),
                "raw_disagreement": raw_d,
                "method_disagreement": method_d,
                "maximum_method_disagreement": float(np.max(method_ds)),
                "disagreement_reduction": 0.0 if raw_d <= 1e-12 else 1.0 - method_d / raw_d,
                "fit_seconds": float(fit_seconds),
                "method_task_loss": task_error(blocks.dataset.problem_type, target, method_reference),
                **safety,
            }
        )
    return rows


def selections(
    blocks: Any,
    raw_validation: np.ndarray,
    gram_validation: np.ndarray,
    rank_validation: np.ndarray,
    model: str,
    seed: int,
    protocol: dict[str, Any],
    *,
    selected_only: dict[str, str] | None,
) -> tuple[dict[str, tuple[float, str, str]], dict[str, Any]]:
    config = protocol["guarded_gram"]
    evidence = guarded_evidence(
        blocks.dataset.problem_type,
        blocks.dataset.y_validation,
        blocks.dataset.y_train,
        raw_validation,
        gram_validation,
        alphas=[float(value) for value in config["candidate_alphas_descending"]],
        resamples=int(config["g1"]["bootstrap_resamples"]),
        seed=bd.stable_seed("GuardedGram", blocks.dataset.key, model, seed),
    )
    methods: dict[str, tuple[float, str, str]] = {
        method: (alpha, "fixed", "fixed") for method, alpha in FIXED.items()
    }
    g1_decisions: dict[str, Any] = {}
    for tau in config["g1"]["taus"]:
        name = f"GuardedGram-G1-t{str(tau).replace('0.', '').replace('.', '') or '0'}"
        alpha, decisions = select_g1(evidence, tau=float(tau))
        methods[name] = (alpha, "G1", "Gram")
        g1_decisions[name] = decisions
    for gamma in config["g2"]["gammas"]:
        for tau in config["g2"]["taus"]:
            gamma_label = str(gamma).replace(".", "p")
            tau_label = str(tau).replace("0.", "").replace(".", "")
            name = f"GuardedGram-G2-g{gamma_label}-t{tau_label}"
            methods[name] = (select_g2(evidence, tau=float(tau), gamma=float(gamma)), "G2", "Gram")
    g3_alpha, g3_state = select_g3(evidence)
    methods["GuardedGram-G3"] = (g3_alpha, "G3", "Gram")

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
    safe = select_safe_gates(safe_records, taus=[0.01], constrained_lambda_multipliers=[])["SafeGram-t01"]
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
    safe_rank = select_safe_gates(rank_records, taus=[0.01], constrained_lambda_multipliers=[])["SafeGram-t01"]
    methods["SafeGram-t01"] = (safe, "safety_reference", "Gram")
    methods["SafeRankGram-t01"] = (safe_rank, "safety_reference", "Rank")

    if selected_only is not None:
        keep = set(FIXED) | {"SafeGram-t01", "SafeRankGram-t01"} | set(selected_only.values())
        methods = {key: value for key, value in methods.items() if key in keep}
    audit = {
        "guarded_evidence": strip_samples(evidence),
        "g1_decisions": g1_decisions,
        "g3_state": g3_state,
        "safe_evidence": safe_records,
        "safe_rank_evidence": rank_records,
        "selected_alphas": {key: value[0] for key, value in methods.items()},
    }
    return methods, audit


def run_unit(
    spec: dict[str, Any],
    model: str,
    seed: int,
    device: str,
    protocol: dict[str, Any],
    selected_only: dict[str, str] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    blocks = load_blocks(spec, protocol)
    orbit = orthogonal_orbit(blocks, protocol)
    raw, gram, source = development_base_predictions(model, blocks, orbit, seed, device)
    rank, rank_meta = rank_prediction(model, blocks, orbit[0], seed, device)
    reference_id = orbit[0].representation_id
    methods, audit = selections(
        blocks,
        raw[reference_id]["validation"],
        gram["validation"],
        rank["validation"],
        model,
        seed,
        protocol,
        selected_only=selected_only,
    )
    rows: list[dict[str, Any]] = []
    for method, (alpha, family, branch) in methods.items():
        invariant = rank if branch == "Rank" else gram
        rows.extend(
            evaluate(
                blocks=blocks,
                orbit=orbit,
                raw=raw,
                invariant=invariant,
                alpha=alpha,
                method=method,
                family=family,
                model=model,
                seed=seed,
                fit_seconds=float(rank_meta["telemetry"].get("fit_seconds", 0.0)) if branch == "Rank" else 0.0,
            )
        )
    evidence_rows = [
        {
            "dataset": blocks.dataset.key,
            "problem_type": blocks.dataset.problem_type,
            "model": model,
            "seed": int(seed),
            **row,
        }
        for row in audit["guarded_evidence"]
    ]
    destination = ROOT / "results" / "raw" / "guardedgram" / model / blocks.dataset.key / f"seed_{seed}.json"
    write_json(
        destination,
        {
            "status": "COMPLETE",
            "dataset": blocks.dataset.key,
            "model": model,
            "seed": int(seed),
            "selection_split": "validation_only",
            "test_outcomes_used_for_selection": False,
            "prospective_outcomes_accessed": False,
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "panel_sha256": sha256_file(PANEL_PATH),
            "source": source,
            **audit,
        },
    )
    selected_message = {key: value[0] for key, value in methods.items() if key.startswith("GuardedGram")}
    print(f"[guarded] {blocks.dataset.key} {model} seed={seed} {selected_message}", flush=True)
    return rows, evidence_rows


def aggregate(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = cells[cells.split == "test"]
    units = (
        test.groupby(["dataset", "problem_type", "model", "method", "family"], as_index=False)
        .agg(
            alpha=("alpha", "median"),
            disagreement_reduction=("disagreement_reduction", "median"),
            normalized_excess_risk=("normalized_excess_risk", "median"),
            absolute_task_difference=("absolute_task_difference", "median"),
            method_loss=("method_loss", "median"),
            raw_fallback=("raw_fallback", "max"),
        )
    )
    units["predictive_rank"] = units.groupby(["dataset", "model"])["method_loss"].rank(method="average")
    rows = []
    for (method, family), frame in units.groupby(["method", "family"], sort=False):
        costs = frame.normalized_excess_risk.to_numpy(float)
        rows.append(
            {
                "method": method,
                "family": family,
                "units": len(frame),
                "datasets": frame.dataset.nunique(),
                "model_families": frame.model.nunique(),
                "median_alpha": float(frame.alpha.median()),
                "median_disagreement_reduction": float(frame.disagreement_reduction.median()),
                "median_C": float(np.median(costs)),
                "p90_C": float(np.quantile(costs, 0.90)),
                "p95_C": float(np.quantile(costs, 0.95)),
                "max_C": float(np.max(costs)),
                "fallback_rate": float(frame.raw_fallback.mean()),
                "mean_predictive_rank": float(frame.predictive_rank.mean()),
            }
        )
    return units, pd.DataFrame(rows)


def choose_stage1(summary: pd.DataFrame) -> dict[str, str]:
    choices: dict[str, str] = {}
    for family in ("G1", "G2", "G3"):
        frame = summary[summary.family == family].copy()
        standard = frame[
            (frame.median_disagreement_reduction >= 0.50)
            & (frame.p95_C <= 0.03)
            & (frame.max_C <= 0.10)
        ]
        pool = standard if len(standard) else frame
        pool = pool.assign(
            score=pool.median_disagreement_reduction
            - 3 * pool.median_C.clip(lower=0)
            - 3 * (pool.p95_C - 0.02).clip(lower=0)
            - 2 * (pool.max_C - 0.10).clip(lower=0)
        )
        choices[family] = str(pool.sort_values(["score", "mean_predictive_rank"], ascending=[False, True]).iloc[0].method)
    return choices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["stage1", "full"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset")
    parser.add_argument("--model")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    protocol = load_protocol()
    stage1_names = set(protocol["stage1"]["datasets"])
    specs = development_specs(protocol)
    if args.stage == "stage1":
        specs = [spec for spec in specs if spec["key"] in stage1_names]
        selected_only = None
    else:
        selected_path = ROOT / "results" / "processed" / "guardedgram_stage1_selection.json"
        if not selected_path.exists():
            raise RuntimeError("run GuardedGram stage1 before full development")
        import json

        selected_only = json.loads(selected_path.read_text())["selected"]
    specs = [spec for spec in specs if args.dataset is None or spec["key"] == args.dataset]
    models = [args.model] if args.model else list(protocol["general_models"])
    seeds = [args.seed] if args.seed is not None else [int(seed) for seed in protocol["development_seeds"]]
    cells: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for spec in specs:
        for model in models:
            for seed in seeds:
                unit, unit_evidence = run_unit(spec, model, seed, args.device, protocol, selected_only)
                cells.extend(unit)
                evidence.extend(unit_evidence)
    processed = ROOT / "results" / "processed"
    filtered = args.dataset is not None or args.model is not None or args.seed is not None
    suffix = f"__{args.dataset or 'all'}__{args.model or 'all'}__{args.seed if args.seed is not None else 'all'}" if filtered else ""
    prefix = f"guardedgram_{args.stage}"
    frame = pd.DataFrame(cells)
    evidence_frame = pd.DataFrame(evidence)
    frame.to_csv(processed / f"{prefix}_cells{suffix}.csv", index=False)
    evidence_frame.to_csv(processed / f"{prefix}_evidence{suffix}.csv", index=False)
    if not filtered:
        units, summary = aggregate(frame)
        units.to_csv(processed / f"{prefix}_units.csv", index=False)
        summary.to_csv(processed / f"{prefix}_summary.csv", index=False)
        if args.stage == "stage1":
            selected = choose_stage1(summary)
            write_json(
                processed / "guardedgram_stage1_selection.json",
                {
                    "status": "DEVELOPMENT_SELECTED",
                    "selection_scope": "four frozen stage-1 development datasets",
                    "selection_split": "development_only",
                    "prospective_outcomes_accessed": False,
                    "selected": selected,
                },
            )
            print(f"[guarded stage1 selected] {selected}")
        write_json(
            processed / f"{prefix}_manifest.json",
            {
                "status": "COMPLETE",
                "stage": args.stage,
                "cells": len(frame),
                "evidence_rows": len(evidence_frame),
                "units": len(units),
                "datasets": len(specs),
                "models": len(models),
                "seeds": len(seeds),
                "prospective_outcomes_accessed": False,
            },
        )


if __name__ == "__main__":
    main()
