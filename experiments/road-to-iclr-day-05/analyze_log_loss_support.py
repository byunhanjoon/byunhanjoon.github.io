"""Audit support assumptions behind packed-estimator log-loss claims."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_disjoint_log_loss import sample_pair_actions
from analyze_disjoint_pack64 import sample_pack_and_pairs
from analyze_log_quotient_jackknife import EPS, PANELS


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
BATCH = 8
QUANTILES = (0.001, 0.01, 0.05)


def true_class_probabilities(y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Return true-class probabilities, retaining an optional draw axis."""
    if prediction.ndim == 2:
        return prediction[np.arange(len(y)), y.astype(int)]
    return prediction[:, np.arange(len(y)), y.astype(int)]


def summarize(values: np.ndarray) -> dict[str, float | int]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "probability_count": int(flat.size),
        "minimum_probability": float(flat.min()),
        "at_or_below_clip_count": int(np.count_nonzero(flat <= EPS)),
        "at_or_below_clip_fraction": float(np.mean(flat <= EPS)),
        "exact_zero_count": int(np.count_nonzero(flat == 0.0)),
        "exact_zero_fraction": float(np.mean(flat == 0.0)),
        **{f"quantile_{q:g}": float(np.quantile(flat, q)) for q in QUANTILES},
    }


def action_support(y: np.ndarray, flat: np.ndarray, blocks: np.ndarray) -> dict[str, float | int]:
    chunks = []
    for start in range(0, len(blocks), BATCH):
        stop = min(start + BATCH, len(blocks))
        prediction = np.mean(np.stack([
            flat[blocks[start:stop, block]].mean(axis=1)
            for block in range(blocks.shape[1])
        ], axis=1), axis=1)
        chunks.append(true_class_probabilities(y, prediction))
    return summarize(np.concatenate(chunks, axis=0))


def main() -> None:
    rows = []
    for panel, config_name, directory_name in PANELS:
        config = json.loads((HERE / config_name).read_text())
        for dataset in config["datasets"]:
            for model in config["models"]:
                archive = np.load(RESULTS / directory_name / f"{dataset}__{model}.npz")
                manifest = json.loads((RESULTS / directory_name / f"{dataset}__{model}.json").read_text())
                if manifest["task"] not in {"binclass", "multiclass"}:
                    continue
                predictions = archive["validation_predictions"]
                shape = tuple(int(value) for value in predictions.shape[:4])
                flat = predictions.reshape((-1,) + predictions.shape[-2:]).astype(np.float64)
                y = archive["validation_y"]
                disjoint, independent = sample_pair_actions(shape, panel, dataset)
                pack, pairs, _ = sample_pack_and_pairs(shape, panel + "-log", dataset)
                actions = {
                    "exact_quotient": None,
                    "disjoint_pair32": disjoint,
                    "independent_pair32": independent,
                    "mutually_disjoint_pack64": pack,
                    "two_disjoint_pairs64": pairs,
                }
                for method, blocks in actions.items():
                    stats = (
                        summarize(true_class_probabilities(y, flat.mean(axis=0)))
                        if blocks is None else action_support(y, flat, blocks)
                    )
                    rows.append({
                        "panel": panel,
                        "dataset": dataset,
                        "model": model,
                        "task": manifest["task"],
                        "method": method,
                        "product_cells": int(np.prod(shape)),
                        **stats,
                    })

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "log_loss_support_audit.csv", index=False)
    by_method = {}
    for method, current in frame.groupby("method"):
        by_method[method] = {
            "candidates": int(len(current)),
            "candidates_touching_clip": int((current.at_or_below_clip_count > 0).sum()),
            "candidates_with_exact_zero": int((current.exact_zero_count > 0).sum()),
            "minimum_probability": float(current.minimum_probability.min()),
            "total_at_or_below_clip": int(current.at_or_below_clip_count.sum()),
            "total_probabilities": int(current.probability_count.sum()),
            "pooled_clip_fraction": float(
                current.at_or_below_clip_count.sum() / current.probability_count.sum()
            ),
        }
    exact_frame = frame[frame.method == "exact_quotient"]
    panel = exact_frame.groupby("panel", as_index=False).agg(
        candidates=("dataset", "size"),
        candidates_touching_clip=("at_or_below_clip_count", lambda x: int((x > 0).sum())),
        minimum_probability=("minimum_probability", "min"),
    )
    clip_active = bool((frame.at_or_below_clip_count > 0).any())
    summary = {
        "status": "complete",
        "clip_epsilon": EPS,
        "classification_candidates": int(len(exact_frame)),
        "methods": by_method,
        "panels": panel.to_dict(orient="records"),
        "interpretation": "clip_active_boundary" if clip_active else "empirical_interior_support",
        "unclipped_smooth_log_assumption_holds_on_audited_population": not clip_active,
        "clipped_log_global_lipschitz_constant": 1.0 / EPS,
        "distribution_wide_support_claimed": False,
    }
    (RESULTS / "log_loss_support_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
