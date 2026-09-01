#!/usr/bin/env python3
"""Analyze regression error tails for the frozen real competence router."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.methods import competence_weights, weighted_prediction


SOURCES = {
    "small_panel": "real_panel_competence_55553b7ffd",
    "breadth_panel": "openml_breadth_competence_48170161d0",
    "regression_confirmation": "regression_confirmation_1e4911698d",
}
N_BOOT = 20_000


def squared_error_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = (prediction - y) ** 2
    order = np.argsort(error)
    split = int(np.floor(0.9 * len(error)))
    return {
        "mse": float(error.mean()),
        "se_top_decile": float(error[order[split:]].mean()),
        "se_bottom_90pct": float(error[order[:split]].mean()),
        "se_over_4_rate": float(np.mean(error > 4.0)),
    }


def hierarchical(values: dict[str, np.ndarray], seed: int) -> dict[str, object]:
    names = sorted(values)
    observed = float(np.mean([values[name].mean() for name in names]))
    rng = np.random.default_rng(seed)
    samples = np.empty(N_BOOT)
    for draw in range(N_BOOT):
        chosen = rng.choice(names, size=len(names), replace=True)
        samples[draw] = np.mean([
            rng.choice(values[str(name)], size=len(values[str(name)]), replace=True).mean()
            for name in chosen
        ])
    low, high = np.quantile(samples, [0.025, 0.975])
    dataset_means = {name: float(values[name].mean()) for name in names}
    return {
        "gain": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "positive_datasets": int(sum(value > 0 for value in dataset_means.values())),
        "per_dataset_gain": dataset_means,
    }


def main() -> None:
    records: list[dict[str, object]] = []
    parent_errors: list[float] = []
    for panel, stem in SOURCES.items():
        raw_path = ROOT / "results" / "raw" / f"{stem}.npz"
        metadata_path = ROOT / "results" / "raw" / f"{stem}.metadata.json"
        parent_path = ROOT / "results" / "processed" / f"{stem}_cells.csv"
        metadata = json.loads(metadata_path.read_text())
        tuning = metadata["synthetic_tuning"]
        if "regression" in tuning:
            tuning = tuning["regression"]
        fixed_weights = np.asarray(tuning["fixed_weights"], dtype=float)
        with np.load(raw_path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        for index in range(len(arrays["dataset"])):
            if str(arrays["task_type"][index]) != "regression":
                continue
            expert_prediction = arrays["expert_prediction"][index].astype(float)
            y = arrays["query_y"][index].astype(float)
            competence = competence_weights(
                arrays["cv_expert_loss"][index].astype(float),
                float(tuning["temperature"]), float(tuning["uniform_shrinkage"]),
            )
            for method, weights in (("fixed", fixed_weights), ("competence", competence)):
                prediction = weighted_prediction(expert_prediction, weights)
                records.append({
                    "panel": panel,
                    "source_episode_index": index,
                    "dataset": str(arrays["dataset"][index]),
                    "repeat": int(arrays["repeat"][index]),
                    "method": method,
                    **squared_error_metrics(y, prediction),
                })

        reconstructed = pd.DataFrame(records)
        reconstructed = reconstructed[reconstructed["panel"] == panel]
        parent = pd.read_csv(parent_path)
        parent = parent[
            (parent["task_type"] == "regression")
            & parent["method"].isin(["fixed", "competence"])
        ]
        check = reconstructed.merge(
            parent[["episode_index", "method", "loss"]],
            left_on=["source_episode_index", "method"], right_on=["episode_index", "method"],
            validate="one_to_one",
        )
        parent_errors.append(float(np.max(np.abs(check["mse"] - check["loss"]))))

    if max(parent_errors) > 1e-5:
        raise AssertionError(f"parent MSE mismatch: {max(parent_errors)}")
    detail = pd.DataFrame(records)
    detail_path = ROOT / "results" / "processed" / "tail_risk_contrast_detail_v1.csv"
    audit_path = ROOT / "results" / "processed" / "tail_risk_contrast_audit_v1.json"
    detail.to_csv(detail_path, index=False)

    pivot = detail.pivot(
        index=["panel", "dataset", "source_episode_index", "repeat"],
        columns="method",
    )
    comparisons: dict[str, object] = {}
    for offset, metric in enumerate(("mse", "se_top_decile", "se_bottom_90pct", "se_over_4_rate")):
        delta = pivot[(metric, "fixed")] - pivot[(metric, "competence")]
        by_dataset = {
            str(dataset): group.to_numpy()
            for dataset, group in delta.groupby(level="dataset", sort=True)
        }
        comparisons[metric] = hierarchical(by_dataset, 195001 + offset)

    audit = {
        "protocol": "TAIL_RISK_CONTRAST_PROTOCOL.md",
        "dataset_count": int(detail["dataset"].nunique()),
        "episodes": int(detail.drop_duplicates(["panel", "source_episode_index"]).shape[0]),
        "bootstrap_replicates": N_BOOT,
        "parent_max_abs_mse_error": max(parent_errors),
        "comparisons": comparisons,
        "classification_frozen_comparator": {
            "datasets": 6,
            "nll_top_decile_gain": -0.02512233373023111,
            "nll_top_decile_ci": [-0.045252449515255595, -0.00850240871971438],
            "nll_over_2_rate_gain": -0.0014973958333333332,
            "nll_over_2_rate_ci": [-0.0029296875, -0.0002278645833333333],
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
