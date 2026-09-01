#!/usr/bin/env python3
"""Run exact one-block interventions and grouped/greedy BlockGuard development."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from guarded_basis.blockguard import (  # noqa: E402
    coordinate_audit,
    gram_interface,
    greedy_candidates,
    grouped_candidates,
    invariant_fractions,
    mixed_representation,
    one_block_raw_representation,
    select_greedy,
    select_grouped,
    selection_key,
    target_free_descriptors,
)
from guarded_basis.common import (  # noqa: E402
    PANEL_PATH,
    PROTOCOL_PATH,
    cached_representation_predictions,
    development_base_predictions,
    development_specs,
    disagreement,
    load_blocks,
    load_protocol,
    normalized_excess_risk,
    orthogonal_orbit,
    sha256_file,
    task_error,
    write_json,
)


def tau_label(tau: float) -> str:
    return str(tau).replace("0.", "").replace(".", "") or "0"


def representation_prediction(
    *,
    blocks: Any,
    orbit: list[Any],
    gram_orbit: list[Any],
    raw: dict[str, dict[str, np.ndarray]],
    gram: dict[str, np.ndarray],
    selected: list[str],
    orbit_index: int,
    model: str,
    seed: int,
    device: str,
    cache_root: Path | None = None,
    finalist_hash: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any], Any]:
    rep = orbit[orbit_index]
    all_features = set(rep.feature_blocks)
    selected_set = set(selected)
    if not selected_set:
        return raw[rep.representation_id], {"telemetry": {}, "source": "shared_raw"}, rep
    if selected_set == all_features:
        return gram, {"telemetry": {}, "source": "shared_invariant_gram"}, gram_orbit[orbit_index]
    mixed = mixed_representation(
        rep,
        blocks.dataset.key,
        selected,
        gram_rep=gram_orbit[orbit_index],
    )
    key = selection_key(selected)
    base = (
        ROOT / "results" / "raw" / "blockguard"
        if cache_root is None else Path(cache_root)
    )
    destination = base / "representations" / model / blocks.dataset.key / f"seed_{seed}" / key / f"{rep.representation_id}.npz"
    prediction, metadata = cached_representation_predictions(
        destination,
        model=model,
        blocks=blocks,
        rep=mixed,
        seed=seed,
        device=device,
        finalist_hash=finalist_hash,
        definition={
            "method": "BlockGuard",
            "selected_features": sorted(selected_set),
            "selection_key": key,
            "representation_id": rep.representation_id,
            "anchors": 16,
            "anchor_selection": "gram_pivot",
            "normalize": True,
        },
    )
    return prediction, metadata, mixed


def one_block_interventions(
    *,
    blocks: Any,
    orbit: list[Any],
    gram_orbit: list[Any],
    raw: dict[str, dict[str, np.ndarray]],
    gram: dict[str, np.ndarray],
    model: str,
    seed: int,
    device: str,
    descriptors: list[dict[str, Any]],
    cache_root: Path | None = None,
    finalist_hash: str | None = None,
) -> list[dict[str, Any]]:
    reference = orbit[0]
    reference_prediction = raw[reference.representation_id]["validation"]
    def evaluate_descriptor(descriptor: dict[str, Any]) -> dict[str, Any]:
        feature = str(descriptor["feature"])
        prediction, metadata, _ = representation_prediction(
            blocks=blocks,
            orbit=orbit,
            gram_orbit=gram_orbit,
            raw=raw,
            gram=gram,
            selected=[feature],
            orbit_index=0,
            model=model,
            seed=seed,
            device=device,
            cache_root=cache_root,
            finalist_hash=finalist_hash,
        )
        safety = normalized_excess_risk(
            blocks.dataset.problem_type,
            blocks.dataset.y_validation,
            reference_prediction,
            prediction["validation"],
            blocks.dataset.y_train,
            epsilon=1e-8,
        )

        raw_one_disagreements: list[float] = []
        raw_one_fit_seconds = 0.0
        for member in range(8):
            one_rep = one_block_raw_representation(reference, blocks.dataset.key, feature, member)
            base = (
                ROOT / "results" / "raw" / "blockguard"
                if cache_root is None else Path(cache_root)
            )
            destination = base / "raw_one_block" / model / blocks.dataset.key / f"seed_{seed}" / selection_key([feature]) / f"member_{member}.npz"
            one_prediction, one_metadata = cached_representation_predictions(
                destination,
                model=model,
                blocks=blocks,
                rep=one_rep,
                seed=seed,
                device=device,
                finalist_hash=finalist_hash,
                definition={
                    "method": "RawOneBlockOrbit",
                    "feature": feature,
                    "member": int(member),
                    "representation_id": one_rep.representation_id,
                },
            )
            raw_one_disagreements.append(
                disagreement(
                    blocks.dataset.problem_type,
                    blocks.dataset.y_validation,
                    reference_prediction,
                    one_prediction["validation"],
                )
            )
            raw_one_fit_seconds += float(one_metadata["telemetry"].get("fit_seconds", 0.0))
        benefit = float(np.mean(raw_one_disagreements))
        return {
            "dataset": blocks.dataset.key,
            "problem_type": blocks.dataset.problem_type,
            "model": model,
            "seed": int(seed),
            **descriptor,
            **safety,
            "raw_one_block_orbit_disagreement": benefit,
            "basis_disagreement_benefit": benefit,
            "gram_one_block_validation_loss": task_error(
                blocks.dataset.problem_type,
                blocks.dataset.y_validation,
                prediction["validation"],
            ),
            "gram_one_block_fit_seconds": float(metadata["telemetry"].get("fit_seconds", 0.0)),
            "raw_one_block_fit_seconds": raw_one_fit_seconds,
            "selection_split": "validation_only",
        }

    feature_workers = max(1, int(os.environ.get("BLOCKGUARD_FEATURE_WORKERS", "1")))
    if feature_workers == 1:
        rows = [evaluate_descriptor(descriptor) for descriptor in descriptors]
    else:
        # Each descriptor owns a disjoint selection-key cache. Executor.map
        # preserves frozen feature order while only changing cache scheduling.
        with concurrent.futures.ThreadPoolExecutor(max_workers=feature_workers) as executor:
            rows = list(executor.map(evaluate_descriptor, descriptors))
    for row in rows:
        print(
            f"[one-block] {blocks.dataset.key} {model} seed={seed} "
            f"feature={row['feature']} C={row['normalized_excess_risk']:.6g} "
            f"benefit={row['basis_disagreement_benefit']:.6g}",
            flush=True,
        )
    return rows


def fit_candidate_rows(
    *,
    candidates: list[dict[str, Any]],
    family: str,
    blocks: Any,
    orbit: list[Any],
    gram_orbit: list[Any],
    raw: dict[str, dict[str, np.ndarray]],
    gram: dict[str, np.ndarray],
    model: str,
    seed: int,
    device: str,
    cache_root: Path | None = None,
    finalist_hash: str | None = None,
) -> list[dict[str, Any]]:
    reference_prediction = raw[orbit[0].representation_id]["validation"]
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        selected = list(candidate["features"])
        prediction, metadata, rep = representation_prediction(
            blocks=blocks,
            orbit=orbit,
            gram_orbit=gram_orbit,
            raw=raw,
            gram=gram,
            selected=selected,
            orbit_index=0,
            model=model,
            seed=seed,
            device=device,
            cache_root=cache_root,
            finalist_hash=finalist_hash,
        )
        safety = normalized_excess_risk(
            blocks.dataset.problem_type,
            blocks.dataset.y_validation,
            reference_prediction,
            prediction["validation"],
            blocks.dataset.y_train,
            epsilon=1e-8,
        )
        fraction_blocks, fraction_dimensions = invariant_fractions(orbit[0], selected)
        rows.append(
            {
                **candidate,
                "family": family,
                "features": selected,
                "selection_key": selection_key(selected),
                "selected_blocks": len(selected),
                "selected_dimensions": sum(len(orbit[0].feature_blocks[name]) for name in selected),
                "invariant_feature_fraction": fraction_blocks,
                "invariant_dimension_fraction": fraction_dimensions,
                "validation_C": float(safety["normalized_excess_risk"]),
                "validation_raw_loss": float(safety["raw_loss"]),
                "validation_method_loss": float(safety["method_loss"]),
                "input_dimension": int(rep.X_train.shape[1]),
                "raw_input_dimension": int(orbit[0].X_train.shape[1]),
                "fit_seconds": float(metadata["telemetry"].get("fit_seconds", 0.0)),
                "predict_seconds": float(metadata["telemetry"].get("predict_seconds", 0.0)),
            }
        )
    return rows


def evaluate_selection(
    *,
    method: str,
    family: str,
    tau: float,
    candidate: dict[str, Any],
    blocks: Any,
    orbit: list[Any],
    gram_orbit: list[Any],
    raw: dict[str, dict[str, np.ndarray]],
    gram: dict[str, np.ndarray],
    model: str,
    seed: int,
    device: str,
    cache_root: Path | None = None,
    finalist_hash: str | None = None,
    panel: str = "guarded_development",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = list(candidate["features"])
    predictions: list[dict[str, np.ndarray]] = []
    mixed_orbit: list[Any] = []
    reference_metadata: dict[str, Any] = {"telemetry": {}}
    for orbit_index in range(len(orbit)):
        prediction, metadata, rep = representation_prediction(
            blocks=blocks,
            orbit=orbit,
            gram_orbit=gram_orbit,
            raw=raw,
            gram=gram,
            selected=selected,
            orbit_index=orbit_index,
            model=model,
            seed=seed,
            device=device,
            cache_root=cache_root,
            finalist_hash=finalist_hash,
        )
        predictions.append(prediction)
        mixed_orbit.append(rep)
        if orbit_index == 0:
            reference_metadata = metadata
    audit = coordinate_audit(mixed_orbit, selected)
    if not audit["passes_1e_minus_6"]:
        raise RuntimeError(f"BlockGuard coordinate audit failed: {blocks.dataset.key}/{selection_key(selected)}")
    fraction_blocks, fraction_dimensions = invariant_fractions(orbit[0], selected)
    rows: list[dict[str, Any]] = []
    for split, target in (("validation", blocks.dataset.y_validation), ("test", blocks.dataset.y_test)):
        raw_reference = raw[orbit[0].representation_id][split]
        method_reference = predictions[0][split]
        raw_ds = [
            disagreement(blocks.dataset.problem_type, target, raw_reference, raw[index][split])
            for index in [rep.representation_id for rep in orbit[1:]]
        ]
        method_ds = [
            disagreement(blocks.dataset.problem_type, target, method_reference, predictions[index][split])
            for index in range(1, len(predictions))
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
                "panel": panel,
                "dataset": blocks.dataset.key,
                "problem_type": blocks.dataset.problem_type,
                "model": model,
                "seed": int(seed),
                "method": method,
                "family": family,
                "split": split,
                "tau": float(tau),
                "selected_candidate": candidate["candidate"],
                "selection_key": selection_key(selected),
                "selected_features": json.dumps(sorted(selected), separators=(",", ":")),
                "selected_blocks": len(selected),
                "invariant_feature_fraction": fraction_blocks,
                "invariant_dimension_fraction": fraction_dimensions,
                "raw_fallback": bool(not selected),
                "raw_disagreement": raw_d,
                "method_disagreement": method_d,
                "maximum_method_disagreement": float(np.max(method_ds)),
                "disagreement_reduction": 0.0 if raw_d <= 1e-12 else 1.0 - method_d / raw_d,
                "inference_passes": 1,
                "input_dimension": int(mixed_orbit[0].X_train.shape[1]),
                "raw_input_dimension": int(orbit[0].X_train.shape[1]),
                "input_dimension_ratio": float(mixed_orbit[0].X_train.shape[1] / orbit[0].X_train.shape[1]),
                "fit_seconds": float(reference_metadata["telemetry"].get("fit_seconds", 0.0)),
                "predict_seconds": float(reference_metadata["telemetry"].get("predict_seconds", 0.0)),
                **safety,
            }
        )
    return rows, audit


def selected_taus(stage: str, protocol: dict[str, Any]) -> dict[str, list[float]]:
    all_taus = [float(value) for value in protocol["block_guard"]["taus"]]
    if stage == "stage1":
        return {"BlockGuard-Grouped": all_taus, "BlockGuard-Greedy": all_taus}
    selection_path = ROOT / "results" / "processed" / "blockguard_stage1_selection.json"
    if not selection_path.exists():
        raise RuntimeError("run BlockGuard stage1 before full development")
    selected = json.loads(selection_path.read_text())["selected"]
    return {
        family: [float(value["tau"])]
        for family, value in selected.items()
    }


def run_unit(
    spec: dict[str, Any],
    model: str,
    seed: int,
    device: str,
    protocol: dict[str, Any],
    stage: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    blocks = load_blocks(spec, protocol)
    orbit = orthogonal_orbit(blocks, protocol)
    gram_orbit = [gram_interface(rep, blocks.dataset.key) for rep in orbit]
    raw, gram, source = development_base_predictions(model, blocks, orbit, seed, device)
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
    )
    grouped, assignment = grouped_candidates(interventions)
    greedy = greedy_candidates(
        interventions, maximum_stages=int(protocol["block_guard"]["maximum_greedy_stages"])
    )
    grouped_rows = fit_candidate_rows(
        candidates=grouped,
        family="BlockGuard-Grouped",
        blocks=blocks,
        orbit=orbit,
        gram_orbit=gram_orbit,
        raw=raw,
        gram=gram,
        model=model,
        seed=seed,
        device=device,
    )
    greedy_rows = fit_candidate_rows(
        candidates=greedy,
        family="BlockGuard-Greedy",
        blocks=blocks,
        orbit=orbit,
        gram_orbit=gram_orbit,
        raw=raw,
        gram=gram,
        model=model,
        seed=seed,
        device=device,
    )
    taus = selected_taus(stage, protocol)
    cells: list[dict[str, Any]] = []
    audits: dict[str, Any] = {}
    selections: list[dict[str, Any]] = []
    for family, candidates, selector in (
        ("BlockGuard-Grouped", grouped_rows, select_grouped),
        ("BlockGuard-Greedy", greedy_rows, select_greedy),
    ):
        for tau in taus[family]:
            selected = selector(candidates, tau)
            method = f"{family}-t{tau_label(tau)}"
            evaluated, audit = evaluate_selection(
                method=method,
                family=family,
                tau=tau,
                candidate=selected,
                blocks=blocks,
                orbit=orbit,
                gram_orbit=gram_orbit,
                raw=raw,
                gram=gram,
                model=model,
                seed=seed,
                device=device,
            )
            cells.extend(evaluated)
            audits[method] = audit
            selections.append(
                {
                    "method": method,
                    "family": family,
                    "tau": tau,
                    "candidate": selected["candidate"],
                    "features": selected["features"],
                    "validation_C": selected["validation_C"],
                }
            )
    candidate_rows = [
        {
            "dataset": blocks.dataset.key,
            "problem_type": blocks.dataset.problem_type,
            "model": model,
            "seed": int(seed),
            **{key: json.dumps(value, separators=(",", ":")) if key == "features" else value for key, value in row.items()},
        }
        for row in grouped_rows + greedy_rows
    ]
    destination = ROOT / "results" / "raw" / "blockguard" / stage / model / blocks.dataset.key / f"seed_{seed}.json"
    payload = {
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
        "danger_group_assignment": assignment,
        "selections": selections,
        "coordinate_audits": audits,
        "one_block_rows": interventions,
        "candidate_rows": candidate_rows,
        "cells": cells,
    }
    write_json(destination, payload)
    print(
        f"[blockguard] {blocks.dataset.key} {model} seed={seed} "
        + " ".join(f"{row['method']}={row['candidate']}" for row in selections),
        flush=True,
    )
    return cells, interventions, candidate_rows


def aggregate(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = cells[cells.split == "test"]
    units = (
        test.groupby(["dataset", "problem_type", "model", "method", "family"], as_index=False)
        .agg(
            tau=("tau", "median"),
            disagreement_reduction=("disagreement_reduction", "median"),
            normalized_excess_risk=("normalized_excess_risk", "median"),
            absolute_task_difference=("absolute_task_difference", "median"),
            method_loss=("method_loss", "median"),
            raw_fallback=("raw_fallback", "max"),
            invariant_feature_fraction=("invariant_feature_fraction", "median"),
            invariant_dimension_fraction=("invariant_dimension_fraction", "median"),
            input_dimension_ratio=("input_dimension_ratio", "median"),
            fit_seconds=("fit_seconds", "median"),
            predict_seconds=("predict_seconds", "median"),
        )
    )
    units["predictive_rank"] = units.groupby(["dataset", "model"])["method_loss"].rank(method="average")
    summaries: list[dict[str, Any]] = []
    for (method, family), frame in units.groupby(["method", "family"], sort=False):
        costs = frame.normalized_excess_risk.to_numpy(float)
        summaries.append(
            {
                "method": method,
                "family": family,
                "units": len(frame),
                "datasets": frame.dataset.nunique(),
                "model_families": frame.model.nunique(),
                "tau": float(frame.tau.median()),
                "median_disagreement_reduction": float(frame.disagreement_reduction.median()),
                "median_C": float(np.median(costs)),
                "p90_C": float(np.quantile(costs, 0.90)),
                "p95_C": float(np.quantile(costs, 0.95)),
                "max_C": float(np.max(costs)),
                "fallback_rate": float(frame.raw_fallback.mean()),
                "median_invariant_feature_fraction": float(frame.invariant_feature_fraction.median()),
                "median_invariant_dimension_fraction": float(frame.invariant_dimension_fraction.median()),
                "median_input_dimension_ratio": float(frame.input_dimension_ratio.median()),
                "median_fit_seconds": float(frame.fit_seconds.median()),
                "median_predict_seconds": float(frame.predict_seconds.median()),
                "mean_predictive_rank": float(frame.predictive_rank.mean()),
            }
        )
    return units, pd.DataFrame(summaries)


def choose_stage1(summary: pd.DataFrame) -> dict[str, dict[str, Any]]:
    choices: dict[str, dict[str, Any]] = {}
    for family in ("BlockGuard-Grouped", "BlockGuard-Greedy"):
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
        row = pool.sort_values(
            ["score", "mean_predictive_rank", "tau"], ascending=[False, True, True]
        ).iloc[0]
        choices[family] = {"method": str(row.method), "tau": float(row.tau)}
    return choices


def collect_units(
    *, stage: str, specs: list[dict[str, Any]], models: list[str], seeds: list[int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cells: list[dict[str, Any]] = []
    interventions: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for spec in specs:
        for model in models:
            for seed in seeds:
                path = ROOT / "results" / "raw" / "blockguard" / stage / model / spec["key"] / f"seed_{seed}.json"
                if not path.exists():
                    raise FileNotFoundError(f"missing BlockGuard unit for aggregation: {path}")
                payload = json.loads(path.read_text())
                if payload["status"] != "COMPLETE":
                    raise RuntimeError(f"incomplete BlockGuard unit: {path}")
                cells.extend(payload["cells"])
                interventions.extend(payload["one_block_rows"])
                candidates.extend(payload["candidate_rows"])
    return cells, interventions, candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["stage1", "full"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset")
    parser.add_argument("--model")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol()
    specs = development_specs(protocol)
    if args.stage == "stage1":
        stage1_names = set(protocol["stage1"]["datasets"])
        specs = [spec for spec in specs if spec["key"] in stage1_names]
    specs = [spec for spec in specs if args.dataset is None or spec["key"] == args.dataset]
    models = [args.model] if args.model else list(protocol["general_models"])
    seeds = [args.seed] if args.seed is not None else [int(seed) for seed in protocol["development_seeds"]]
    if args.aggregate_only:
        cells, interventions, candidates = collect_units(
            stage=args.stage, specs=specs, models=models, seeds=seeds
        )
    else:
        cells, interventions, candidates = [], [], []
        for spec in specs:
            for model in models:
                for seed in seeds:
                    unit_cells, unit_interventions, unit_candidates = run_unit(
                        spec, model, seed, args.device, protocol, args.stage
                    )
                    cells.extend(unit_cells)
                    interventions.extend(unit_interventions)
                    candidates.extend(unit_candidates)
    processed = ROOT / "results" / "processed"
    filtered = args.dataset is not None or args.model is not None or args.seed is not None
    suffix = (
        f"__{args.dataset or 'all'}__{args.model or 'all'}__{args.seed if args.seed is not None else 'all'}"
        if filtered
        else ""
    )
    prefix = f"blockguard_{args.stage}"
    cell_frame = pd.DataFrame(cells)
    intervention_frame = pd.DataFrame(interventions)
    candidate_frame = pd.DataFrame(candidates)
    cell_frame.to_csv(processed / f"{prefix}_cells{suffix}.csv", index=False)
    intervention_frame.to_csv(processed / f"{prefix}_one_block{suffix}.csv", index=False)
    candidate_frame.to_csv(processed / f"{prefix}_candidates{suffix}.csv", index=False)
    if not filtered:
        units, summary = aggregate(cell_frame)
        units.to_csv(processed / f"{prefix}_units.csv", index=False)
        summary.to_csv(processed / f"{prefix}_summary.csv", index=False)
        if args.stage == "stage1":
            selected = choose_stage1(summary)
            write_json(
                processed / "blockguard_stage1_selection.json",
                {
                    "status": "DEVELOPMENT_SELECTED",
                    "selection_scope": "four frozen stage-1 development datasets",
                    "selection_split": "development_only",
                    "prospective_outcomes_accessed": False,
                    "selected": selected,
                },
            )
            print(f"[blockguard stage1 selected] {selected}")
        write_json(
            processed / f"{prefix}_manifest.json",
            {
                "status": "COMPLETE",
                "stage": args.stage,
                "cells": len(cell_frame),
                "one_block_rows": len(intervention_frame),
                "candidate_rows": len(candidate_frame),
                "units": len(units),
                "datasets": len(specs),
                "models": len(models),
                "seeds": len(seeds),
                "prospective_outcomes_accessed": False,
            },
        )


if __name__ == "__main__":
    main()
