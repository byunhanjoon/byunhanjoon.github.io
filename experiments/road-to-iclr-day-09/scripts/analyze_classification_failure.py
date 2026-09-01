#!/usr/bin/env python3
"""Diagnose ranking versus calibration in real binary competence transfer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.methods import competence_weights, prediction_loss, weighted_prediction


SOURCES = {
    "small_panel": "real_panel_competence_55553b7ffd",
    "breadth_panel": "openml_breadth_competence_48170161d0",
}
N_BOOT = 20_000


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    index = np.minimum((p * bins).astype(int), bins - 1)
    value = 0.0
    for cell in range(bins):
        mask = index == cell
        if mask.any():
            value += float(mask.mean()) * abs(float(y[mask].mean() - p[mask].mean()))
    return value


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    nll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    order = np.argsort(nll)
    split = int(np.floor(0.9 * len(nll)))
    return {
        "log_loss": float(nll.mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "auc": float(roc_auc_score(y, p)),
        "calibration_bias_abs": abs(float(p.mean() - y.mean())),
        "ece10": ece(y, p),
        "sharpness": float(np.mean(np.abs(p - 0.5))),
        "nll_top_decile": float(nll[order[split:]].mean()),
        "nll_bottom_90pct": float(nll[order[:split]].mean()),
        "nll_over_2_rate": float(np.mean(nll > 2.0)),
    }


def hierarchical(values: dict[str, np.ndarray], seed: int) -> dict[str, float]:
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
    return {"gain": observed, "ci_low": float(low), "ci_high": float(high)}


def main() -> None:
    records: list[dict[str, object]] = []
    parent_errors: list[float] = []
    for panel, stem in SOURCES.items():
        raw_path = ROOT / "results" / "raw" / f"{stem}.npz"
        metadata_path = ROOT / "results" / "raw" / f"{stem}.metadata.json"
        parent_path = ROOT / "results" / "processed" / f"{stem}_cells.csv"
        metadata = json.loads(metadata_path.read_text())
        tuning = metadata["synthetic_tuning"]["classification"]
        fixed_weights = np.asarray(tuning["fixed_weights"], dtype=float)
        with np.load(raw_path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        for index in range(len(arrays["dataset"])):
            if str(arrays["task_type"][index]) != "classification":
                continue
            predictions = arrays["expert_prediction"][index].astype(float)
            y = arrays["query_y"][index].astype(float)
            soft_weights = competence_weights(
                arrays["cv_expert_loss"][index].astype(float),
                float(tuning["temperature"]), float(tuning["uniform_shrinkage"]),
            )
            for method, weights in (("fixed", fixed_weights), ("competence", soft_weights)):
                p = np.clip(weighted_prediction(predictions, weights), 1e-7, 1 - 1e-7)
                records.append({
                    "panel": panel,
                    "source_episode_index": index,
                    "dataset": str(arrays["dataset"][index]),
                    "repeat": int(arrays["repeat"][index]),
                    "method": method,
                    **metrics(y, p),
                })

        reconstructed = pd.DataFrame(records)
        reconstructed = reconstructed[reconstructed["panel"] == panel]
        parent = pd.read_csv(parent_path)
        parent = parent[
            (parent["task_type"] == "classification")
            & parent["method"].isin(["fixed", "competence"])
        ]
        check = reconstructed.merge(
            parent[["episode_index", "method", "loss"]],
            left_on=["source_episode_index", "method"], right_on=["episode_index", "method"],
            validate="one_to_one",
        )
        parent_errors.append(float(np.max(np.abs(check["log_loss"] - check["loss"]))))

    if max(parent_errors) > 1e-5:
        raise AssertionError(f"parent log-loss mismatch: {max(parent_errors)}")

    detail = pd.DataFrame(records)
    detail_path = ROOT / "results" / "processed" / "classification_failure_detail_v1.csv"
    audit_path = ROOT / "results" / "processed" / "classification_failure_audit_v1.json"
    detail.to_csv(detail_path, index=False)

    pivot = detail.pivot(
        index=["panel", "dataset", "source_episode_index", "repeat"],
        columns="method",
    )
    comparisons: dict[str, object] = {}
    directions = {
        "log_loss": ("fixed", "competence"),
        "brier": ("fixed", "competence"),
        "auc": ("competence", "fixed"),
        "calibration_bias_abs": ("fixed", "competence"),
        "ece10": ("fixed", "competence"),
        "sharpness": ("competence", "fixed"),
        "nll_top_decile": ("fixed", "competence"),
        "nll_bottom_90pct": ("fixed", "competence"),
        "nll_over_2_rate": ("fixed", "competence"),
    }
    for offset, (metric, (left, right)) in enumerate(directions.items()):
        delta = pivot[(metric, left)] - pivot[(metric, right)]
        by_dataset = {
            str(dataset): group.to_numpy()
            for dataset, group in delta.groupby(level="dataset", sort=True)
        }
        record = hierarchical(by_dataset, 185001 + offset)
        record["per_dataset_gain"] = {
            name: float(values.mean()) for name, values in by_dataset.items()
        }
        comparisons[metric] = record

    breadth_comparisons: dict[str, object] = {}
    breadth = pivot.loc[pivot.index.get_level_values("panel") == "breadth_panel"]
    for offset, metric in enumerate(("log_loss", "nll_top_decile", "nll_bottom_90pct", "nll_over_2_rate")):
        left, right = directions[metric]
        delta = breadth[(metric, left)] - breadth[(metric, right)]
        by_dataset = {
            str(dataset): group.to_numpy()
            for dataset, group in delta.groupby(level="dataset", sort=True)
        }
        breadth_comparisons[metric] = hierarchical(by_dataset, 186001 + offset)

    audit = {
        "protocol": "CLASSIFICATION_FAILURE_PROTOCOL.md",
        "datasets": sorted(detail["dataset"].unique().tolist()),
        "dataset_count": int(detail["dataset"].nunique()),
        "episodes": int(detail.drop_duplicates(["panel", "source_episode_index"]).shape[0]),
        "bootstrap_replicates": N_BOOT,
        "parent_max_abs_log_loss_error": max(parent_errors),
        "gain_directions": {
            "loss_error_metrics": "fixed_minus_competence",
            "auc_and_sharpness": "competence_minus_fixed",
        },
        "comparisons": comparisons,
        "unseen_breadth_tail_comparisons": breadth_comparisons,
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
