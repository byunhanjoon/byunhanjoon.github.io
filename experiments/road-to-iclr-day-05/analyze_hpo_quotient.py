"""Analyze the frozen prospective quotient-HPO panel."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent


def brier(y: np.ndarray, predictions: np.ndarray) -> float:
    targets = np.eye(predictions.shape[-1])[y.astype(int)]
    return float(np.mean(np.sum((predictions - targets) ** 2, axis=-1)))


def orbit_metrics(predictions: np.ndarray, y: np.ndarray) -> dict[str, float]:
    flat = predictions.reshape((-1,) + predictions.shape[-2:]).astype(np.float64)
    centroid = flat.mean(axis=0)
    risk = float(np.mean(np.sum((flat - centroid) ** 2, axis=-1)))
    member = float(np.mean([brier(y, item) for item in flat]))
    quotient = brier(y, centroid)
    hard = np.argmax(flat, axis=-1)
    return {
        "schema_risk": risk,
        "mean_member_brier": member,
        "quotient_centroid_brier": quotient,
        "ambiguity_identity_error": abs(member - quotient - risk),
        "hard_flip_fraction": float(np.mean(np.any(hard != hard[0:1], axis=0))),
    }


def validation_losses(predictions: np.ndarray, y: np.ndarray) -> np.ndarray:
    # predictions: [candidate, feature, category, class, row, output]
    targets = np.eye(predictions.shape[-1])[y.astype(int)]
    return np.mean(np.sum((predictions - targets) ** 2, axis=-1), axis=-1)


def gather_per_schema(predictions: np.ndarray, choices: np.ndarray) -> np.ndarray:
    if predictions.shape[1:-2] != choices.shape:
        raise ValueError("choice grid does not match factor shape")
    index = choices[None, ..., None, None]
    return np.take_along_axis(predictions, index, axis=0)[0]


def entropy_bits(choices: np.ndarray) -> float:
    counts = np.asarray(list(Counter(int(x) for x in choices.flat).values()), dtype=float)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def switch_decomposition(baseline: np.ndarray, selected: np.ndarray) -> dict[str, float]:
    axes = tuple(range(baseline.ndim - 2))
    base_centered = baseline - baseline.mean(axis=axes, keepdims=True)
    switch = selected - baseline
    switch_centered = switch - switch.mean(axis=axes, keepdims=True)
    base_risk = float(np.mean(np.sum(base_centered**2, axis=-1)))
    switch_dispersion = float(np.mean(np.sum(switch_centered**2, axis=-1)))
    twice_covariance = float(2 * np.mean(np.sum(base_centered * switch_centered, axis=-1)))
    selected_centered = selected - selected.mean(axis=axes, keepdims=True)
    selected_risk = float(np.mean(np.sum(selected_centered**2, axis=-1)))
    return {
        "identity_frozen_risk": base_risk,
        "per_schema_risk": selected_risk,
        "risk_change": selected_risk - base_risk,
        "switch_dispersion": switch_dispersion,
        "twice_baseline_switch_covariance": twice_covariance,
        "reconstruction_error": abs(selected_risk - base_risk - switch_dispersion - twice_covariance),
    }


def exact_two_sided_sign_p(positive: int, total: int) -> float:
    if not total:
        return float("nan")
    tail = min(positive, total - positive)
    probability = sum(math.comb(total, k) for k in range(tail + 1)) / 2**total
    return float(min(1.0, 2 * probability))


def analyze_cell(path: Path, manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    archive = np.load(path)
    validation = archive["validation_predictions"].astype(np.float64)
    test = archive["test_predictions"].astype(np.float64)
    validation_y = archive["validation_y"]
    test_y = archive["test_y"]
    metric_rows: list[dict[str, Any]] = []
    mechanism_rows: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(manifest["seeds"]):
        val = validation[..., seed_index, :, :]
        tst = test[..., seed_index, :, :]
        losses = validation_losses(val, validation_y)
        per_choices = np.argmin(losses, axis=0)
        identity = int(np.argmin(losses[:, 0, 0, 0]))
        full = int(np.argmin(losses.mean(axis=(1, 2, 3))))
        development = losses[:, :2, :2, :]
        development_quotient = int(np.argmin(development.mean(axis=(1, 2, 3))))
        development_minimax = int(np.argmin(development.max(axis=(1, 2, 3))))
        policies = {
            "per_schema": gather_per_schema(tst, per_choices),
            "identity_frozen": tst[identity],
            "full_quotient": tst[full],
            "development_quotient": tst[development_quotient],
            "development_minimax": tst[development_minimax],
        }
        choices = {
            "per_schema": None,
            "identity_frozen": identity,
            "full_quotient": full,
            "development_quotient": development_quotient,
            "development_minimax": development_minimax,
        }
        scopes = {
            "full": (slice(None), slice(None), slice(None)),
            "heldout_nuisance": (slice(2, 4), slice(2, 4), slice(None)),
        }
        for policy, predictions in policies.items():
            for scope, index in scopes.items():
                metrics = orbit_metrics(predictions[index], test_y)
                metric_rows.append({
                    "dataset": manifest["dataset"],
                    "family": manifest["family"],
                    "seed": int(seed),
                    "policy": policy,
                    "scope": scope,
                    "selected_candidate": choices[policy],
                    **metrics,
                })
        switch = switch_decomposition(policies["identity_frozen"], policies["per_schema"])
        mechanism_rows.append({
            "dataset": manifest["dataset"],
            "family": manifest["family"],
            "seed": int(seed),
            "identity_candidate": identity,
            "full_quotient_candidate": full,
            "development_quotient_candidate": development_quotient,
            "development_minimax_candidate": development_minimax,
            "per_schema_choice_entropy_bits": entropy_bits(per_choices),
            "per_schema_fraction_different_from_identity": float(np.mean(per_choices != identity)),
            "validation_quotient_identity_error_max": float(max(
                abs(
                    float(losses[h].mean())
                    - brier(validation_y, val[h].reshape((-1,) + val.shape[-2:]).mean(axis=0))
                    - orbit_metrics(val[h], validation_y)["schema_risk"]
                ) for h in range(len(losses))
            )),
            **switch,
        })
    return metric_rows, mechanism_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "hpo_quotient_config.json")
    parser.add_argument("--input-dir", type=Path, default=HERE / "results" / "hpo_quotient")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    metric_rows: list[dict[str, Any]] = []
    mechanism_rows: list[dict[str, Any]] = []
    missing = []
    for dataset, family in itertools.product(config["datasets"], config["families"]):
        stem = f"{dataset}__{family}"
        npz_path = args.input_dir / f"{stem}.npz"
        json_path = args.input_dir / f"{stem}.json"
        if not npz_path.exists() or not json_path.exists():
            missing.append(stem)
            continue
        rows, mechanisms = analyze_cell(npz_path, json.loads(json_path.read_text()))
        metric_rows.extend(rows)
        mechanism_rows.extend(mechanisms)
    if missing:
        raise RuntimeError(f"missing cells: {missing}")
    metrics = pd.DataFrame(metric_rows)
    mechanisms = pd.DataFrame(mechanism_rows)
    heldout = metrics[metrics.scope == "heldout_nuisance"]
    wide = heldout.pivot_table(
        index=["dataset", "family", "seed"], columns="policy",
        values=["schema_risk", "quotient_centroid_brier"], aggfunc="first",
    )
    cell_rows = []
    for (dataset, family), group in wide.groupby(level=[0, 1]):
        risk_q = group[("schema_risk", "development_quotient")].mean()
        risk_p = group[("schema_risk", "per_schema")].mean()
        risk_i = group[("schema_risk", "identity_frozen")].mean()
        brier_q = group[("quotient_centroid_brier", "development_quotient")].mean()
        brier_p = group[("quotient_centroid_brier", "per_schema")].mean()
        cell_rows.append({
            "dataset": dataset,
            "family": family,
            "development_quotient_schema_risk": risk_q,
            "per_schema_schema_risk": risk_p,
            "identity_frozen_schema_risk": risk_i,
            "quotient_vs_per_schema_risk_reduction": 1 - risk_q / risk_p if risk_p else 0.0,
            "quotient_vs_identity_risk_reduction": 1 - risk_q / risk_i if risk_i else 0.0,
            "development_quotient_brier": brier_q,
            "per_schema_brier": brier_p,
            "relative_brier_change_vs_per_schema": (brier_q - brier_p) / brier_p,
        })
    cells = pd.DataFrame(cell_rows)
    wins = int((cells.development_quotient_schema_risk < cells.per_schema_schema_risk).sum())
    risk_change = float(
        cells.development_quotient_schema_risk.mean() / cells.per_schema_schema_risk.mean() - 1
    )
    brier_change = float(
        cells.development_quotient_brier.mean() / cells.per_schema_brier.mean() - 1
    )
    rope = float(config["proper_loss_rope_relative"])
    gate = bool(wins > len(cells) / 2 and risk_change < 0 and brier_change <= rope)
    summary = {
        "status": "complete",
        "cells": len(cells),
        "seeds_per_cell": len(config["seeds"]),
        "cells_where_development_quotient_has_lower_schema_risk_than_per_schema": wins,
        "exact_two_sided_cell_sign_p": exact_two_sided_sign_p(wins, len(cells)),
        "panel_relative_schema_risk_change": risk_change,
        "panel_relative_quotient_brier_change": brier_change,
        "prospective_gate_passed": gate,
        "maximum_proper_loss_ambiguity_identity_error": float(metrics.ambiguity_identity_error.max()),
        "maximum_validation_quotient_identity_error": float(mechanisms.validation_quotient_identity_error_max.max()),
        "maximum_switch_decomposition_error": float(mechanisms.reconstruction_error.max()),
        "mean_per_schema_choice_entropy_bits": float(mechanisms.per_schema_choice_entropy_bits.mean()),
        "mean_per_schema_fraction_different_from_identity": float(mechanisms.per_schema_fraction_different_from_identity.mean()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "hpo_quotient_policy_metrics.csv", index=False)
    mechanisms.to_csv(args.output_dir / "hpo_quotient_mechanisms.csv", index=False)
    cells.to_csv(args.output_dir / "hpo_quotient_cell_comparison.csv", index=False)
    (args.output_dir / "hpo_quotient_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

