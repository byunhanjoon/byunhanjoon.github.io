#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from src.core import PROCESSED, RAW, ROOT, atomic_json, load_dataset, parse_indices


config = json.loads((ROOT / "config.json").read_text())
datasets = config["datasets"]
budgets = config["budgets"]
seeds = config["seeds"]
checks: dict[str, object] = {}

assert len(datasets) == 8
assert set(seeds) == {0, 1, 2}
assert set(budgets) == {16, 32, 64}
checks["dataset_panel"] = {"count": len(datasets), "datasets": datasets}

context_files = sorted((RAW / "context_evaluations").glob("*.csv"))
prediction_files = sorted((RAW / "predictions").glob("*.npz"))
assert len(context_files) == len(datasets) * len(budgets) * len(seeds) == 72
assert len(prediction_files) == 72
context_rows = 0
for dataset in datasets:
    bundle = load_dataset(dataset)
    split = json.loads((RAW / "splits" / f"{dataset}.json").read_text())
    assert len(split["candidate_indices"]) == 256
    assert len(split["selector_indices"]) == 128
    assert len(split["test_indices"]) == 256
    assert len(set(split["candidate_indices"]) | set(split["selector_indices"]) | set(split["test_indices"])) == 640
    for k in budgets:
        for seed in seeds:
            path = RAW / "context_evaluations" / f"{dataset}_k{k}_seed{seed}.csv"
            frame = pd.read_csv(path)
            assert len(frame) == 512
            assert set(frame.seed) == {seed}
            assert set(frame.K) == {k}
            for value in frame.indices:
                indices = parse_indices(value)
                assert len(indices) == len(np.unique(indices)) == k
                assert indices.min() >= 0 and indices.max() < 256
                if bundle.task == "classification":
                    assert set(bundle.y_candidate[indices]) == set(np.unique(bundle.y_candidate))
            predictions = np.load(RAW / "predictions" / f"{dataset}_k{k}_seed{seed}.npz")["predictions"]
            assert predictions.shape[0] == 512 and predictions.shape[1] == 128
            context_rows += len(frame)
assert context_rows == 36_864
checks["random_context_surface"] = {"files": len(context_files), "rows": context_rows, "predictions_cached": True}

utility = pd.read_csv(PROCESSED / "utility_prediction.csv")
required_models = {
    "constant",
    "additive_ridge",
    "id_fm",
    "residual_fm",
    "feature_fm",
    "signed_bilinear",
    "deepsets",
    "cosine_diversity",
    "rbf_diversity",
    "euclidean_neighbor_diversity",
    "label_target_complementarity",
    "geometry_plus_complementarity",
}
for (dataset, k), cell in utility.groupby(["dataset", "K"]):
    assert required_models <= set(cell.model)
    assert set(cell[cell.model == "id_fm"]["rank"].astype(int)) == {2, 4, 8, 16}
    assert set(cell[cell.model == "residual_fm"]["rank"].astype(int)) == {2, 4, 8, 16}
    assert set(cell[cell.model == "feature_fm"]["rank"].astype(int)) == {4, 8, 16}
    assert set(cell[cell.model == "signed_bilinear"]["rank"].astype(int)) == {4, 8, 16}
assert len(utility.groupby(["dataset", "K"])) == 24
checks["surrogates"] = {"rows": len(utility), "cells": 24, "all_required_models_and_ranks": True}

selectors = pd.read_csv(PROCESSED / "selector_results.csv")
required_selectors = {
    "random_stratified",
    "additive",
    "k_center",
    "k_medoids",
    "nearest_query_cluster",
    "CRUMB-like",
    "LUCoS-like",
    "DPP",
    "pairwise_FM_greedy",
    "pairwise_FM_swap",
    "feature_FM_greedy",
    "oracle_best_of_random",
}
for (dataset, k), cell in selectors.groupby(["dataset", "K"]):
    assert required_selectors <= set(cell.method)
    assert len(cell[cell.method == "random_stratified"]) == 20
    for value in cell.indices:
        indices = parse_indices(value)
        assert len(indices) == len(np.unique(indices)) == k
assert len(selectors) == 24 * 32
test_predictions = pd.read_csv(RAW / "test_predictions.csv")
assert len(test_predictions) == len(selectors) * 256
checks["selectors"] = {"rows": len(selectors), "test_prediction_rows": len(test_predictions), "all_required_methods": True}

direct = pd.read_csv(PROCESSED / "direct_interactions.csv")
runtime_cells = []
for path in context_files:
    frame = pd.read_csv(path, usecols=["dataset", "runtime_seconds"])
    runtime_cells.append(frame)
runtime = pd.concat(runtime_cells).groupby("dataset").runtime_seconds.mean().sort_values()
fastest = list(runtime.index[:2])
assert set(fastest) <= set(direct.dataset)
for _, cell in direct.groupby("dataset"):
    assert len(cell) == 300
    assert cell.pair_id.nunique() == 100
    assert cell.base_id.nunique() == 3
assert len(list((PROCESSED / "direct_search").glob("*.csv"))) == 2
checks["direct_diagnostics"] = {"fastest_datasets": fastest, "datasets_reported": sorted(direct.dataset.unique()), "rows": len(direct)}

fallback = pd.read_csv(PROCESSED / "failure_fallbacks.csv")
assert set(fallback.dataset) == {"credit-g", "diamonds"}
for dataset, cell in fallback.groupby("dataset"):
    assert set(cell[cell.model == "id_fm"]["rank"].astype(int)) == {2, 4, 8, 16}
    assert {0.01, 0.05, 0.10} <= set(cell[cell.model == "id_fm"].weight_decay.round(2))
    assert "pairwise_plus_deepsets_correction" in set(cell.model)
    assert len(cell[cell.model.str.startswith("query_cluster_")]) == 8
    raw = pd.read_csv(RAW / "fallback_128" / f"{dataset}.csv")
    assert len(raw) == 1024
checks["failure_fallbacks"] = {"datasets": ["credit-g", "diamonds"], "candidate_pool": 128, "contexts_per_dataset": 1024}

for path in (PROCESSED / "analysis_audits").glob("*.json"):
    assert not json.loads(path.read_text())["final_test_labels_used_for_selection"]
for path in (PROCESSED / "failure_fallback_audits").glob("*.json"):
    assert not json.loads(path.read_text())["final_test_labels_used_for_fitting_or_selection"]
checks["leakage_audit"] = {"final_test_labels_used_for_selection_or_fitting": False}

required_plots = {
    "surrogate_r2.png",
    "performance_vs_budget.png",
    "interaction_magnitude_histogram.png",
    "selector_win_loss_heatmap.png",
    "predicted_vs_actual_utility.png",
    "direct_search_oracle.png",
    "interaction_heatmap.png",
}
assert required_plots <= {path.name for path in (ROOT / "plots").glob("*.png")}
required_sections = [
    "Executive Verdict",
    "One-Paragraph Summary",
    "Experimental Setup",
    "Main Result Table",
    "Utility Prediction",
    "Direct Interaction Diagnostic",
    "b_ij Ablations",
    "Selector Results",
    "Cross-Model Check",
    "Failures and Negative Results",
    "Strongest Evidence FOR the Hypothesis",
    "Strongest Evidence AGAINST the Hypothesis",
    "Recommended Next Research Direction",
    "Files Produced",
]
report = (ROOT / "results.md").read_text()
for section in required_sections:
    assert report.count(f"## {section}") == 1
checks["deliverables"] = {"required_plots": sorted(required_plots), "required_results_sections": required_sections}

checks["status"] = "complete"
atomic_json(PROCESSED / "completion_audit.json", checks)
print(json.dumps(checks, indent=2))
