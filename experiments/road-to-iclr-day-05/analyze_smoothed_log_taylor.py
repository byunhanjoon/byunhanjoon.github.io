"""Calibrate Taylor and global bounds for packed smoothed log scores."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_disjoint_log_loss import sample_pair_actions
from analyze_disjoint_pack64 import sample_pack_and_pairs
from analyze_log_quotient_jackknife import PANELS
from analyze_smoothed_log_packing import ALPHAS, BATCH


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def diagnostics(y: np.ndarray, flat: np.ndarray, blocks: np.ndarray, alpha: float) -> dict[str, float | bool]:
    classes = flat.shape[-1]
    raw_q = flat.mean(axis=0)[np.arange(len(y)), y.astype(int)]
    q = (1.0 - alpha) * raw_q + alpha / classes
    actual_parts, first_parts, second_parts, delta_mse_parts = [], [], [], []
    for start in range(0, len(blocks), BATCH):
        stop = min(start + BATCH, len(blocks))
        prediction = np.mean(np.stack([
            flat[blocks[start:stop, block]].mean(axis=1)
            for block in range(blocks.shape[1])
        ], axis=1), axis=1)
        raw_qhat = prediction[:, np.arange(len(y)), y.astype(int)]
        qhat = (1.0 - alpha) * raw_qhat + alpha / classes
        delta = qhat - q[None]
        scaled = delta / q[None]
        first = -scaled.mean(axis=1)
        second = first + 0.5 * (scaled ** 2).mean(axis=1)
        actual_parts.append(np.mean(-np.log(qhat) + np.log(q[None]), axis=1))
        first_parts.append(first); second_parts.append(second)
        delta_mse_parts.append((delta ** 2).mean(axis=1))
    actual = np.concatenate(actual_parts)
    first = np.concatenate(first_parts)
    second = np.concatenate(second_parts)
    delta_mse = np.concatenate(delta_mse_parts)
    actual_rmse = float(np.sqrt(np.mean(actual ** 2)))
    denom = max(actual_rmse, np.finfo(float).tiny)
    nondegenerate = bool(np.std(actual) > 1e-15 and np.std(first) > 1e-15)
    correlation = float(np.corrcoef(actual, first)[0, 1]) if nondegenerate else float("nan")
    bound = float(np.mean(delta_mse) / (alpha / classes) ** 2)
    actual_mse = float(np.mean(actual ** 2))
    quadratic_bias = float(np.mean(second - first))
    return {
        "actual_score_rmse": actual_rmse,
        "first_order_correlation": correlation,
        "first_relative_approximation_rmse": float(np.sqrt(np.mean((actual - first) ** 2)) / denom),
        "second_relative_approximation_rmse": float(np.sqrt(np.mean((actual - second) ** 2)) / denom),
        "observed_bias": float(actual.mean()),
        "quadratic_bias_term": quadratic_bias,
        "quadratic_to_observed_bias_ratio": float(quadratic_bias / actual.mean()) if abs(actual.mean()) > 1e-15 else float("nan"),
        "global_mse_bound": bound,
        "actual_score_mse": actual_mse,
        "actual_to_global_bound_ratio": float(actual_mse / bound) if bound > 0 else 0.0,
        "bound_violated": bool(actual_mse > bound * (1.0 + 1e-10) + 1e-30),
        "nondegenerate": nondegenerate,
    }


def main() -> None:
    rows = []
    for panel, config_name, directory_name in PANELS:
        config = json.loads((HERE / config_name).read_text())
        directory = RESULTS / directory_name
        for dataset in config["datasets"]:
            for model in config["models"]:
                archive = np.load(directory / f"{dataset}__{model}.npz")
                manifest = json.loads((directory / f"{dataset}__{model}.json").read_text())
                if manifest["task"] not in {"binclass", "multiclass"}:
                    continue
                predictions = archive["validation_predictions"]
                shape = tuple(int(value) for value in predictions.shape[:4])
                if int(np.prod(shape)) != 128:
                    continue
                flat = predictions.reshape((-1,) + predictions.shape[-2:]).astype(np.float64)
                y = archive["validation_y"]
                disjoint, independent = sample_pair_actions(shape, panel, dataset)
                pack, pairs, _ = sample_pack_and_pairs(shape, panel + "-log", dataset)
                actions = {
                    "disjoint_pair32": disjoint,
                    "independent_pair32": independent,
                    "mutually_disjoint_pack64": pack,
                    "two_disjoint_pairs64": pairs,
                }
                for method, blocks in actions.items():
                    for alpha in ALPHAS:
                        rows.append({
                            "panel": panel, "dataset": dataset, "model": model,
                            "method": method, "alpha": alpha,
                            **diagnostics(y, flat, blocks, alpha),
                        })
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "smoothed_log_taylor_calibration.csv", index=False)
    summaries = {}
    for alpha, current in frame.groupby("alpha"):
        nondeg = current[current.nondegenerate]
        strong = (nondeg.first_order_correlation > .99) & (nondeg.second_relative_approximation_rmse < .10)
        summaries[f"{alpha:g}"] = {
            "candidate_method_cells": int(len(current)),
            "nondegenerate_cells": int(len(nondeg)),
            "cells_correlation_above_0_99": int((nondeg.first_order_correlation > .99).sum()),
            "cells_second_relative_rmse_below_0_10": int((nondeg.second_relative_approximation_rmse < .10).sum()),
            "cells_passing_both_local_clauses": int(strong.sum()),
            "fraction_passing_both_local_clauses": float(strong.mean()),
            "median_first_relative_rmse": float(nondeg.first_relative_approximation_rmse.median()),
            "median_second_relative_rmse": float(nondeg.second_relative_approximation_rmse.median()),
            "median_actual_to_global_bound_ratio": float(current.actual_to_global_bound_ratio.median()),
            "maximum_actual_to_global_bound_ratio": float(current.actual_to_global_bound_ratio.max()),
            "global_bound_violations": int(current.bound_violated.sum()),
        }
    local_pass = summaries["0.01"]["fraction_passing_both_local_clauses"] >= .8
    bound_pass = not bool(frame.bound_violated.any())
    summary = {
        "status": "complete", "classification_candidates": int(frame[["panel", "dataset", "model"]].drop_duplicates().shape[0]),
        "methods": int(frame.method.nunique()), "alphas": ALPHAS, "by_alpha": summaries,
        "local_taylor_gate_passed": bool(local_pass), "global_bound_audit_passed": bool(bound_pass),
        "interpretation": "quantitative_taylor_pass" if local_pass and bound_pass else "formal_bound_only",
    }
    (RESULTS / "smoothed_log_taylor_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
