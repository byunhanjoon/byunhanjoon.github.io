#!/usr/bin/env python3
"""Diagnose the five worst prior pure-Gram cells and conditionally run Type-B rescue."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DAY9 = ROOT.parent
TOURNAMENT = DAY9 / "basis_controlled_tournament"
sys.path.insert(0, str(ROOT))

from safe_basis.common import (  # noqa: E402
    PANEL_PATH,
    PROTOCOL_PATH,
    bd,
    calibration_metrics,
    development_specs,
    load_blocks,
    load_json,
    load_prediction_bundle,
    load_protocol,
    mix_predictions,
    normalized_excess_risk,
    orthogonal_orbit,
    save_prediction_bundle,
    sha256_file,
    task_error,
    write_json,
)
from safe_basis.models import fit_predictions  # noqa: E402
from safe_basis.rankgram import build_rank_adaptive_interface  # noqa: E402

from tournament.representations import build_interface  # noqa: E402


RANK_SELECTION = ROOT / "results" / "processed" / "rank_selection.json"


def select_worst_cells(allowed_seeds: set[int]) -> pd.DataFrame:
    units = pd.read_csv(TOURNAMENT / "results" / "processed" / "prospective_units.csv")
    candidates = units[
        (units["method"] == "GramAnchor")
        & (units["split"] == "test")
        & (units["seed"].isin(sorted(allowed_seeds)))
    ].copy()
    candidates = candidates.sort_values("relative_task_change", ascending=False)
    selected = candidates.head(4).copy()
    if "steel-plates-fault" not in set(selected["dataset"]):
        steel = candidates[candidates["dataset"] == "steel-plates-fault"].head(1)
        selected = pd.concat([steel, selected]).head(5)
    else:
        selected = candidates.head(5)
    if len(selected) != 5 or "steel-plates-fault" not in set(selected["dataset"]):
        raise RuntimeError("could not select five prior worst cells including Steel Plates")
    return selected[["dataset", "problem_type", "model", "seed", "relative_task_change", "task_error", "raw_task_error"]]


def cached_diagnostic_fit(
    path: Path,
    *,
    model: str,
    blocks: Any,
    rep: Any,
    seed: int,
    device: str,
    definition: dict[str, Any],
    learning_rate_multiplier: float = 1.0,
    weight_decay: float | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    hashes = {"protocol": sha256_file(PROTOCOL_PATH), "panel": sha256_file(PANEL_PATH)}
    if path.exists() and path.with_suffix(".json").exists():
        arrays, metadata = load_prediction_bundle(path)
        if metadata.get("frozen_hashes") != hashes or metadata.get("definition") != definition:
            raise RuntimeError(f"diagnostic cache drift at {path}")
        return arrays, metadata
    predictions, telemetry = fit_predictions(
        model,
        blocks.dataset.problem_type,
        rep,
        blocks.dataset.y_train,
        blocks.dataset.y_validation,
        seed,
        device,
        include_train=True,
        learning_rate_multiplier=learning_rate_multiplier,
        weight_decay=weight_decay,
    )
    metadata = {
        "status": "COMPLETE",
        "definition": definition,
        "frozen_hashes": hashes,
        "telemetry": telemetry,
    }
    save_prediction_bundle(path, predictions["train"], predictions["validation"], predictions["test"], metadata)
    return predictions, metadata


def method_row(
    *,
    blocks: Any,
    cell: Any,
    method: str,
    prediction: dict[str, np.ndarray],
    raw_prediction: dict[str, np.ndarray],
    telemetry: dict[str, Any],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": blocks.dataset.key,
        "problem_type": blocks.dataset.problem_type,
        "model": cell.model,
        "seed": int(cell.seed),
        "method": method,
        "prior_relative_task_change": float(cell.relative_task_change),
        "empirical_rank": diagnostics.get("median_empirical_rank", np.nan),
        "reconstruction_error": diagnostics.get("maximum_reconstruction_error", 0.0 if method == "Raw" else np.nan),
        "feature_dimension": diagnostics.get("feature_dimension", blocks.X_train.shape[1]),
        "anchor_condition": diagnostics.get("maximum_anchor_condition", np.nan),
        "fit_seconds": float(telemetry.get("fit_seconds", 0.0)),
        "best_epoch": telemetry.get("best_epoch", np.nan),
        "optimization_convergence": "early-stopped checkpoint" if "best_epoch" in telemetry else "native converged fit",
    }
    for split, target in (("train", blocks.dataset.y_train), ("validation", blocks.dataset.y_validation), ("test", blocks.dataset.y_test)):
        safety = normalized_excess_risk(
            blocks.dataset.problem_type,
            target,
            raw_prediction[split],
            prediction[split],
            blocks.dataset.y_train,
        )
        row[f"{split}_error"] = safety["method_loss"]
        row[f"{split}_absolute_difference"] = safety["absolute_task_difference"]
        row[f"{split}_relative_difference"] = safety["relative_task_difference"]
        row[f"{split}_C"] = safety["normalized_excess_risk"]
        row[f"{split}_denominator_sensitive"] = safety["denominator_sensitive"]
    row.update({f"test_{key}": value for key, value in calibration_metrics(blocks.dataset.problem_type, blocks.dataset.y_test, prediction["test"]).items()})
    return row


def classify_failure(raw: dict[str, Any], gram: dict[str, Any]) -> str:
    if bool(gram["test_denominator_sensitive"]):
        return "Type D — metric denominator artifact"
    train_gap = float(gram["train_C"])
    reconstruction = float(gram["reconstruction_error"])
    if train_gap > 0.05 and reconstruction > 1e-3:
        return "Type A — information/interface loss"
    if train_gap > 0.05 and reconstruction <= 1e-3:
        return "Type B — optimization difficulty"
    return "Type C — altered generalization/inductive bias"


def aggregate_diagnostics(rep: Any) -> dict[str, Any]:
    audits = list(rep.metadata["block_audits"].values())
    return {
        "median_empirical_rank": float(np.median([row["empirical_rank"] for row in audits])),
        "maximum_reconstruction_error": float(np.max([row["reconstruction_error"] for row in audits])),
        "maximum_anchor_condition": float(np.max([row["anchor_gram_effective_condition_number"] for row in audits])),
        "feature_dimension": int(rep.X_train.shape[1]),
    }


def run_diagnosis(device: str) -> pd.DataFrame:
    protocol = load_protocol()
    selected_rank = load_json(RANK_SELECTION)["config"]
    gates = pd.read_csv(ROOT / "results" / "processed" / "development_gate_cells.csv")
    specs = {spec["key"]: spec for spec in development_specs(protocol)}
    selected_cells = select_worst_cells({int(value) for value in protocol["model_seeds"]})
    selected_cells.to_csv(ROOT / "results" / "processed" / "failure_panel.csv", index=False)
    all_rows = []
    for cell in selected_cells.itertuples(index=False):
        blocks = load_blocks(specs[cell.dataset], protocol)
        reference = orthogonal_orbit(blocks, protocol)[0]
        fixed_gram = build_interface(reference, "gram_anchor", blocks.dataset.key, anchors=16, selection="gram_pivot", normalize=True)
        fixed_diagnostic = build_rank_adaptive_interface(
            reference,
            blocks.dataset.key,
            relative_threshold=1e-6,
            anchor_rule="fixed_16",
            normalization="N1_anchor_norm",
            standardize=True,
        )
        rank_rep = build_rank_adaptive_interface(reference, blocks.dataset.key, **selected_rank)
        unit = ROOT / "results" / "raw" / "failure_diagnosis" / cell.model / cell.dataset / f"seed_{cell.seed}"
        raw_prediction, raw_meta = cached_diagnostic_fit(
            unit / "Raw.npz", model=cell.model, blocks=blocks, rep=reference, seed=int(cell.seed), device=device, definition={"method": "Raw"}
        )
        gram_prediction, gram_meta = cached_diagnostic_fit(
            unit / "GramAnchor.npz", model=cell.model, blocks=blocks, rep=fixed_gram, seed=int(cell.seed), device=device, definition={"method": "GramAnchor", "anchors": 16}
        )
        rank_prediction, rank_meta = cached_diagnostic_fit(
            unit / "RankAdaptiveGram.npz", model=cell.model, blocks=blocks, rep=rank_rep, seed=int(cell.seed), device=device, definition={"method": "RankAdaptiveGram", **selected_rank}
        )
        alpha_rows = gates[
            (gates["dataset"] == cell.dataset)
            & (gates["model"] == cell.model)
            & (gates["seed"] == cell.seed)
            & (gates["method"] == "SafeGram-t01")
            & (gates["split"] == "validation")
        ]
        if len(alpha_rows) != 1:
            raise RuntimeError(f"missing SafeGram alpha for {cell.dataset}/{cell.model}/{cell.seed}")
        alpha = float(alpha_rows.iloc[0]["alpha"])
        safe_prediction = {split: mix_predictions(raw_prediction[split], gram_prediction[split], alpha) for split in ("train", "validation", "test")}
        raw_row = method_row(blocks=blocks, cell=cell, method="Raw", prediction=raw_prediction, raw_prediction=raw_prediction, telemetry=raw_meta["telemetry"], diagnostics={})
        gram_row = method_row(blocks=blocks, cell=cell, method="GramAnchor", prediction=gram_prediction, raw_prediction=raw_prediction, telemetry=gram_meta["telemetry"], diagnostics=aggregate_diagnostics(fixed_diagnostic))
        rank_row = method_row(blocks=blocks, cell=cell, method="RankAdaptiveGram", prediction=rank_prediction, raw_prediction=raw_prediction, telemetry=rank_meta["telemetry"], diagnostics=aggregate_diagnostics(rank_rep))
        safe_row = method_row(blocks=blocks, cell=cell, method="SafeGram-t01", prediction=safe_prediction, raw_prediction=raw_prediction, telemetry={"fit_seconds": raw_meta["telemetry"].get("fit_seconds", 0) + gram_meta["telemetry"].get("fit_seconds", 0)}, diagnostics=aggregate_diagnostics(fixed_diagnostic))
        safe_row["alpha"] = alpha
        failure = classify_failure(raw_row, gram_row)
        for row in (raw_row, gram_row, rank_row, safe_row):
            row["failure_type"] = failure
            row.setdefault("alpha", 0.0 if row["method"] == "Raw" else 1.0)
            all_rows.append(row)
        print(f"[failure] {cell.dataset} {cell.model} seed={cell.seed} {failure} safe-alpha={alpha}", flush=True)
    frame = pd.DataFrame(all_rows)
    gate_cells = pd.read_csv(ROOT / "results" / "processed" / "development_gate_cells.csv")
    disagreement_values = []
    maximum_disagreement_values = []
    for row in frame.itertuples(index=False):
        if row.method == "RankAdaptiveGram":
            disagreement_values.append(0.0)
            maximum_disagreement_values.append(0.0)
            continue
        lookup = gate_cells[
            (gate_cells["dataset"] == row.dataset)
            & (gate_cells["model"] == row.model)
            & (gate_cells["seed"] == row.seed)
            & (gate_cells["method"] == row.method)
            & (gate_cells["split"] == "test")
        ]
        if len(lookup) != 1:
            raise RuntimeError(f"missing diagnostic disagreement for {row.dataset}/{row.model}/{row.seed}/{row.method}")
        disagreement_values.append(float(lookup.iloc[0]["method_disagreement"]))
        maximum_disagreement_values.append(float(lookup.iloc[0]["maximum_method_disagreement"]))
    frame["disagreement"] = disagreement_values
    frame["maximum_disagreement"] = maximum_disagreement_values
    frame.to_csv(ROOT / "results" / "processed" / "failure_diagnosis.csv", index=False)
    write_json(
        ROOT / "results" / "processed" / "failure_diagnosis_manifest.json",
        {
            "status": "COMPLETE",
            "cells": len(selected_cells),
            "methods": ["Raw", "GramAnchor", "RankAdaptiveGram", "SafeGram-t01"],
            "failure_type_counts": frame[frame["method"] == "GramAnchor"]["failure_type"].value_counts().to_dict(),
        },
    )
    return frame


def run_rescue(frame: pd.DataFrame, device: str) -> None:
    type_b = frame[(frame["method"] == "GramAnchor") & frame["failure_type"].str.startswith("Type B")]
    output = ROOT / "results" / "processed" / "optimization_rescue_trials.csv"
    if not len(type_b):
        pd.DataFrame(columns=["status", "reason"]).to_csv(output, index=False)
        write_json(ROOT / "results" / "processed" / "optimization_rescue_manifest.json", {"status": "NOT_TRIGGERED", "reason": "No Type B failure was diagnosed."})
        return
    protocol = load_protocol()
    specs = {spec["key"]: spec for spec in development_specs(protocol)}
    trials = []
    for cell in type_b.itertuples(index=False):
        blocks = load_blocks(specs[cell.dataset], protocol)
        reference = orthogonal_orbit(blocks, protocol)[0]
        for method in ("Raw", "GramAnchor"):
            for standardize in (False, True):
                if method == "Raw":
                    rep = bd.standardize_representation(reference) if standardize else reference
                else:
                    rep = build_rank_adaptive_interface(
                        reference,
                        blocks.dataset.key,
                        relative_threshold=1e-6,
                        anchor_rule="fixed_16",
                        normalization="N1_anchor_norm",
                        standardize=standardize,
                    )
                for lr_multiplier in (0.5, 1.0, 2.0):
                    for weight_decay in (0.0001, 0.0):
                        definition = {
                            "method": method,
                            "standardize": standardize,
                            "learning_rate_multiplier": lr_multiplier,
                            "weight_decay": weight_decay,
                        }
                        name = f"{method}__std{int(standardize)}__lr{lr_multiplier:g}__wd{weight_decay:g}"
                        path = ROOT / "results" / "raw" / "optimization_rescue" / cell.model / cell.dataset / f"seed_{cell.seed}" / f"{name}.npz"
                        prediction, metadata = cached_diagnostic_fit(
                            path,
                            model=cell.model,
                            blocks=blocks,
                            rep=rep,
                            seed=int(cell.seed),
                            device=device,
                            definition=definition,
                            learning_rate_multiplier=lr_multiplier,
                            weight_decay=weight_decay,
                        )
                        trials.append(
                            {
                                "dataset": cell.dataset,
                                "model": cell.model,
                                "seed": int(cell.seed),
                                **definition,
                                "train_error": task_error(blocks.dataset.problem_type, blocks.dataset.y_train, prediction["train"]),
                                "validation_error": task_error(blocks.dataset.problem_type, blocks.dataset.y_validation, prediction["validation"]),
                                "test_error": task_error(blocks.dataset.problem_type, blocks.dataset.y_test, prediction["test"]),
                                "fit_seconds": metadata["telemetry"].get("fit_seconds", 0.0),
                            }
                        )
                        print(f"[rescue] {cell.dataset} {cell.model} {name}", flush=True)
    trials_frame = pd.DataFrame(trials)
    trials_frame.to_csv(output, index=False)
    selected = trials_frame.sort_values("validation_error").groupby(["dataset", "model", "seed", "method"], as_index=False).first()
    selected.to_csv(ROOT / "results" / "processed" / "optimization_rescue_selected.csv", index=False)
    write_json(ROOT / "results" / "processed" / "optimization_rescue_manifest.json", {"status": "COMPLETE", "type_b_cells": len(type_b), "trials": len(trials_frame), "equal_trials_per_method_cell": 12})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not RANK_SELECTION.exists():
        raise RuntimeError("rank selection is required before failure diagnosis")
    frame = run_diagnosis(args.device)
    run_rescue(frame, args.device)


if __name__ == "__main__":
    main()
