"""Real-tensor calibration of the exact independent cross-score variance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_cross_quotient_selection as CQS
import analyze_robust_model_selection as RMS


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DRAWS = 1_024
BATCH = 8
METHODS = ("strength2_mean16", "iid_mean16")
PANELS = CQS.PANELS


def target(y: np.ndarray, width: int) -> np.ndarray:
    if width == 1:
        return y.astype(np.float64)[:, None]
    return np.eye(width, dtype=np.float64)[y.astype(int)]


def estimator_ids(action: dict[str, np.ndarray], method: str) -> np.ndarray:
    return action["strength2"] if method == "strength2_mean16" else action["iid16"]


def audit_candidate(
    y: np.ndarray, flat: np.ndarray, streams: list[dict[str, np.ndarray]], method: str
) -> dict[str, float | bool]:
    quotient = flat.mean(axis=0)
    quotient_residual = target(y, flat.shape[-1]) - quotient
    formula_terms = np.empty(DRAWS, dtype=np.float64)
    residual_terms = np.empty(DRAWS, dtype=np.float64)
    self_interaction_terms = np.empty(DRAWS, dtype=np.float64)
    observed_scores = np.empty(DRAWS, dtype=np.float64)
    ids = [estimator_ids(stream, method) for stream in streams]
    for start in range(0, DRAWS, BATCH):
        stop = min(start + BATCH, DRAWS)
        predictions = [flat[current[start:stop]].mean(axis=1) for current in ids]
        error_a = predictions[0] - quotient[None]
        error_b = predictions[1] - quotient[None]
        inner_ra = np.mean(np.sum(quotient_residual[None] * error_a, axis=-1), axis=1)
        inner_rb = np.mean(np.sum(quotient_residual[None] * error_b, axis=-1), axis=1)
        inner_ab = np.mean(np.sum(error_a * error_b, axis=-1), axis=1)
        residual_terms[start:stop] = inner_ra ** 2 + inner_rb ** 2
        self_interaction_terms[start:stop] = inner_ab ** 2
        formula_terms[start:stop] = residual_terms[start:stop] + self_interaction_terms[start:stop]
        residual_c = target(y, flat.shape[-1])[None] - predictions[2]
        residual_d = target(y, flat.shape[-1])[None] - predictions[3]
        observed_scores[start:stop] = np.mean(
            np.sum(residual_c * residual_d, axis=-1), axis=1
        )
    predicted = float(formula_terms.mean())
    residual_aligned = float(residual_terms.mean())
    covariance_self_interaction = float(self_interaction_terms.mean())
    observed = float(observed_scores.var(ddof=1))
    predicted_se = float(formula_terms.std(ddof=1) / np.sqrt(DRAWS))
    variance_influence = (observed_scores - observed_scores.mean()) ** 2
    observed_se = float(variance_influence.std(ddof=1) / np.sqrt(DRAWS))
    combined_se = float(np.hypot(predicted_se, observed_se))
    nondegenerate = bool(max(predicted, observed) >= 1e-18)
    return {
        "predicted_variance": predicted,
        "residual_aligned_variance": residual_aligned,
        "covariance_self_interaction_variance": covariance_self_interaction,
        "observed_variance": observed,
        "predicted_standard_error": predicted_se,
        "observed_standard_error": observed_se,
        "predicted_over_observed": predicted / observed if observed > 0 else np.nan,
        "standardized_discrepancy": (
            (predicted - observed) / combined_se if combined_se > 0 else np.nan
        ),
        "nondegenerate": nondegenerate,
    }


def main() -> None:
    rows = []
    for panel, config_name, directory_name in PANELS:
        config = json.loads((HERE / config_name).read_text())
        directory = RESULTS / directory_name
        for dataset in config["datasets"]:
            first_archive = np.load(directory / f"{dataset}__{config['models'][0]}.npz")
            shape = tuple(int(x) for x in first_archive["validation_predictions"].shape[:4])
            streams = [
                RMS.action_ids(shape, RMS.stable_seed("variance-identity", panel, dataset, str(index)))
                for index in range(4)
            ]
            for model in config["models"]:
                archive = np.load(directory / f"{dataset}__{model}.npz")
                manifest = json.loads((directory / f"{dataset}__{model}.json").read_text())
                flat = archive["validation_predictions"].reshape(
                    (-1,) + archive["validation_predictions"].shape[-2:]
                ).astype(np.float64)
                for method in METHODS:
                    rows.append({
                        "panel": panel, "dataset": dataset,
                        "task": manifest["task"], "model": model, "method": method,
                        **audit_candidate(archive["validation_y"], flat, streams, method),
                    })
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "cross_variance_identity_cells.csv", index=False)
    summary: dict[str, object] = {"status": "complete", "methods": {}, "panels": {}}
    method_passes = []
    for method, current in frame.groupby("method"):
        current = current[current.nondegenerate & np.isfinite(current.standardized_discrepancy)]
        fraction = float((current.standardized_discrepancy.abs() <= 2.58).mean())
        passed = bool(fraction >= .90)
        method_passes.append(passed)
        summary["methods"][method] = {
            "nondegenerate_cells": int(len(current)),
            "fraction_absolute_standardized_discrepancy_at_most_2_58": fraction,
            "median_absolute_standardized_discrepancy": float(current.standardized_discrepancy.abs().median()),
            "gate_clause": passed,
        }
    panel_passes = []
    for (panel, method), current in frame.groupby(["panel", "method"]):
        current = current[current.nondegenerate & (current.predicted_over_observed > 0)]
        geometric_ratio = float(np.exp(np.log(current.predicted_over_observed).mean()))
        passed = bool(.8 <= geometric_ratio <= 1.25)
        panel_passes.append(passed)
        summary["panels"].setdefault(panel, {})[method] = {
            "geometric_mean_predicted_over_observed": geometric_ratio,
            "cells": int(len(current)), "gate_clause": passed,
        }
    summary["frozen_gate_passed"] = bool(all(method_passes) and all(panel_passes))
    (RESULTS / "cross_variance_identity_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
