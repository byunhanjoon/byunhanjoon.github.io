"""Brier/log/AUC/accuracy scope for disjoint pair and four-pack predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import analyze_cross_quotient_selection as CQS
from analyze_disjoint_pair32 import paired_ids
from analyze_disjoint_pack64 import sample_pack_and_pairs
from analyze_log_quotient_jackknife import EPS


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DRAWS = 1_024
BATCH = 8
METRICS = ("brier", "log_loss", "roc_auc", "accuracy")
COMPARISONS = {
    "pair32": ("disjoint_pair32", "independent_pair32"),
    "pack64": ("mutually_disjoint_pack64", "two_disjoint_pairs64"),
}


def metric_values(y: np.ndarray, prediction: np.ndarray) -> dict[str, np.ndarray]:
    target = np.eye(2)[y.astype(int)]
    if prediction.ndim == 2:
        prediction = prediction[None]
    brier = np.mean(np.sum((target[None] - prediction) ** 2, axis=-1), axis=1)
    log = -np.mean(np.log(np.clip(prediction[:, np.arange(len(y)), y.astype(int)], EPS, 1)), axis=1)
    accuracy = np.mean(np.argmax(prediction, axis=-1) == y[None], axis=1)
    positive = y.astype(int) == 1
    positives, negatives = int(positive.sum()), int((~positive).sum())
    ranks = rankdata(prediction[:, :, 1], method="average", axis=1)
    auc = (
        ranks[:, positive].sum(axis=1) - positives * (positives + 1) / 2
    ) / (positives * negatives)
    return {"brier": brier, "log_loss": log, "roc_auc": auc, "accuracy": accuracy}


def action_metrics(y: np.ndarray, flat: np.ndarray, blocks: np.ndarray) -> dict[str, np.ndarray]:
    output = {metric: np.empty(DRAWS) for metric in METRICS}
    for start in range(0, DRAWS, BATCH):
        stop = min(start + BATCH, DRAWS)
        prediction = np.mean(np.stack([
            flat[blocks[start:stop, block]].mean(axis=1)
            for block in range(blocks.shape[1])
        ], axis=1), axis=1)
        values = metric_values(y, prediction)
        for metric in METRICS:
            output[metric][start:stop] = values[metric]
    return output


def main() -> None:
    rows = []
    for panel, config_name, directory_name in CQS.PANELS:
        config = json.loads((HERE / config_name).read_text())
        for dataset in config["datasets"]:
            for model in config["models"]:
                archive = np.load(RESULTS / directory_name / f"{dataset}__{model}.npz")
                manifest = json.loads((RESULTS / directory_name / f"{dataset}__{model}.json").read_text())
                if manifest["task"] != "binclass":
                    continue
                shape = tuple(int(value) for value in archive["validation_predictions"].shape[:4])
                flat = archive["validation_predictions"].reshape(
                    (-1,) + archive["validation_predictions"].shape[-2:]
                ).astype(np.float64)
                y = archive["validation_y"]
                disjoint, independent = paired_ids(shape, panel + "-metric", dataset)
                pack, pairs, _ = sample_pack_and_pairs(shape, panel + "-metric", dataset)
                actions = {
                    "disjoint_pair32": np.stack(disjoint, axis=1),
                    "independent_pair32": np.stack(independent, axis=1),
                    "mutually_disjoint_pack64": pack,
                    "two_disjoint_pairs64": pairs,
                }
                exact = {key: float(value[0]) for key, value in
                         metric_values(y, flat.mean(axis=0)).items()}
                for method, blocks in actions.items():
                    values = action_metrics(y, flat, blocks)
                    for metric in METRICS:
                        error = values[metric] - exact[metric]
                        rows.append({
                            "panel": panel, "dataset": dataset, "model": model,
                            "method": method, "metric": metric,
                            "product_cells": int(np.prod(shape)),
                            "exact_quotient_metric": exact[metric],
                            "bias": float(error.mean()),
                            "rmse": float(np.sqrt(np.mean(error ** 2))),
                            "max_absolute_error": float(np.max(np.abs(error))),
                        })
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "packing_metric_scope_calibration.csv", index=False)
    summary = {"status": "complete", "draws_per_candidate": DRAWS, "comparisons": {}}
    all_pass = True
    accuracy_universal = True
    for name, (action, control) in COMPARISONS.items():
        pivot = frame[frame.method.isin((action, control))].pivot(
            index=["panel", "dataset", "model", "product_cells", "metric"],
            columns="method", values="rmse"
        ).reset_index()
        pivot["strict_win"] = pivot[action] < pivot[control] - 1e-15
        panel_passes = {}
        for metric in METRICS:
            current = pivot[pivot.metric == metric]
            panel_means = current.groupby("panel")[[action, control]].mean()
            panel_passes[metric] = int((panel_means[action] <= panel_means[control] + 1e-15).sum())
        nonexact = pivot[pivot.product_cells == 128]
        strict_by_metric = {
            metric: {"wins": int(current.strict_win.sum()), "candidates": int(len(current)),
                     "fraction": float(current.strict_win.mean())}
            for metric in METRICS
            for current in [nonexact[nonexact.metric == metric]]
        }
        represented = int(pivot.panel.nunique())
        gate = bool(
            panel_passes["brier"] == represented
            and panel_passes["log_loss"] == represented
            and panel_passes["roc_auc"] >= min(4, represented)
            and all(strict_by_metric[metric]["fraction"] >= .7
                    for metric in ("brier", "log_loss", "roc_auc"))
        )
        universal = strict_by_metric["accuracy"]["fraction"] >= .7
        all_pass &= gate; accuracy_universal &= universal
        summary["comparisons"][name] = {
            "action": action, "control": control, "represented_panels": represented,
            "panels_nolower_by_metric": panel_passes,
            "nonexhaustive_strict_wins": strict_by_metric,
            "probabilistic_ranking_scope_passed": gate,
            "accuracy_scope_passed": universal,
        }
    summary["frozen_scope_interpretation"] = (
        "universal_metric_pass" if all_pass and accuracy_universal
        else ("probabilistic_ranking_pass_accuracy_boundary" if all_pass else "scope_failure")
    )
    summary["frozen_gate_passed"] = bool(all_pass)
    (RESULTS / "packing_metric_scope_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
