#!/usr/bin/env python3
"""Frozen soft-versus-hard competence diagnostic on immutable test predictions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.methods import EXPERTS, competence_weights, prediction_loss, weighted_prediction


def unique(pattern: str) -> Path:
    paths = sorted(ROOT.glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(f"expected one {pattern}, found {paths}")
    return paths[0]


def paired_bootstrap(frame: pd.DataFrame, left: str, right: str, seed: int) -> dict:
    cells = []
    for _, cell in frame.groupby(["context_size", "feature_count", "rho"], sort=True):
        pivot = cell.pivot(index="episode_index", columns="method", values="loss")
        cells.append((pivot[left] - pivot[right]).to_numpy())
    observed = float(np.mean([values.mean() for values in cells]))
    rng = np.random.default_rng(seed)
    samples = np.zeros(10_000)
    for start in range(0, 10_000, 250):
        stop = min(start + 250, 10_000)
        chunk = np.zeros(stop - start)
        for values in cells:
            indices = rng.integers(0, len(values), size=(stop - start, len(values)))
            chunk += values[indices].mean(axis=1)
        samples[start:stop] = chunk / len(cells)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"gain": observed, "ci_low": float(low), "ci_high": float(high)}


def main() -> None:
    raw_path = unique("results/raw/fallback_loss_router_*_test.npz")
    metadata_path = unique("results/raw/fallback_loss_router_*_test.metadata.json")
    parent_path = unique("results/processed/fallback_loss_router_*_test_cells.csv")
    detail_path = ROOT / "results/processed/competence_soft_hard_detail_v1.csv"
    summary_path = ROOT / "results/processed/competence_soft_hard_audit_v1.json"
    for output in (detail_path, summary_path):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")

    # Materialize each compressed member once; repeated NpzFile indexing would
    # decompress the complete member on every episode without changing results.
    with np.load(raw_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    metadata = json.loads(metadata_path.read_text())
    records = []
    diagnostics = []
    for index in range(len(arrays["rho"])):
        task_type = str(arrays["task_type"][index])
        cv_loss = arrays["cv_expert_loss"][index].astype(float)
        query_loss = arrays["query_expert_loss"][index].astype(float)
        predictions = arrays["expert_prediction"][index].astype(float)
        y = arrays["query_y"][index].astype(float)
        tuning = metadata["tuning"][task_type]
        soft_weights = competence_weights(
            cv_loss, float(tuning["temperature"]), float(tuning["uniform_shrinkage"])
        )
        hard_weights = np.eye(len(EXPERTS))[int(np.argmin(cv_loss))]
        fixed_weights = np.asarray(tuning["fixed_weights"], dtype=float)
        method_weights = {
            "fixed": fixed_weights,
            "hard_cv": hard_weights,
            "soft_cv": soft_weights,
        }
        base = {
            "episode_index": index,
            "task_type": task_type,
            "context_size": int(arrays["context_size"][index]),
            "feature_count": int(arrays["feature_count"][index]),
            "rho": float(arrays["rho"][index]),
        }
        for method, weights in method_weights.items():
            records.append({
                **base,
                "method": method,
                "loss": prediction_loss(y, weighted_prediction(predictions, weights), task_type),
            })
        records.append({**base, "method": "best_individual_oracle", "loss": float(query_loss.min())})
        ordered = np.sort(cv_loss)
        entropy = float(-np.sum(soft_weights * np.log(np.clip(soft_weights, 1e-15, None))))
        correlation = float(spearmanr(cv_loss, query_loss).statistic)
        diagnostics.append({
            **base,
            "cv_margin": float(ordered[1] - ordered[0]),
            "soft_entropy": entropy,
            "effective_experts": float(np.exp(entropy)),
            "max_soft_weight": float(soft_weights.max()),
            "cv_argmin_matches_query_best": int(np.argmin(cv_loss) == np.argmin(query_loss)),
            "cv_query_spearman": correlation,
        })

    frame = pd.DataFrame(records)
    diagnostic = pd.DataFrame(diagnostics)
    parent = pd.read_csv(parent_path)
    parent = parent[parent["method"].isin(["fixed", "competence"])].copy()
    parent["method"] = parent["method"].replace({"competence": "soft_cv"})
    check = frame[frame["method"].isin(["fixed", "soft_cv"])].merge(
        parent[["episode_index", "method", "loss"]],
        on=["episode_index", "method"], suffixes=("_new", "_parent"), validate="one_to_one"
    )
    max_error = float(np.max(np.abs(check["loss_new"] - check["loss_parent"])))
    if max_error > 1e-6:
        raise AssertionError(f"parent loss mismatch: {max_error}")

    detail = frame.merge(diagnostic, on=[
        "episode_index", "task_type", "context_size", "feature_count", "rho"
    ], validate="many_to_one")
    detail.to_csv(detail_path, index=False)

    audit = {
        "protocol": "COMPETENCE_SOFT_HARD_PROTOCOL.md",
        "parent_run": metadata["run_key"],
        "parent_max_abs_loss_error": max_error,
        "tasks": {},
    }
    for task_index, task_type in enumerate(("classification", "regression")):
        task = detail[detail["task_type"] == task_type]
        episode = task.drop_duplicates("episode_index").copy()
        pivot = task.pivot(index="episode_index", columns="method", values="loss")
        episode = episode.set_index("episode_index")
        episode["soft_vs_hard_gain"] = pivot["hard_cv"] - pivot["soft_cv"]
        episode["margin_quintile"] = pd.qcut(
            episode["cv_margin"], 5, labels=False, duplicates="drop"
        )
        quintiles = (
            episode.groupby("margin_quintile", observed=True)["soft_vs_hard_gain"]
            .agg(["mean", "count"]).reset_index().to_dict(orient="records")
        )
        audit["tasks"][task_type] = {
            "soft_vs_hard": paired_bootstrap(task, "hard_cv", "soft_cv", 16_000 + task_index),
            "hard_vs_fixed": paired_bootstrap(task, "fixed", "hard_cv", 16_100 + task_index),
            "mean_losses": task.groupby("method")["loss"].mean().to_dict(),
            "hard_cv_matches_query_best_rate": float(episode["cv_argmin_matches_query_best"].mean()),
            "mean_cv_query_spearman": float(episode["cv_query_spearman"].mean()),
            "mean_effective_experts": float(episode["effective_experts"].mean()),
            "mean_max_soft_weight": float(episode["max_soft_weight"].mean()),
            "soft_vs_hard_by_cv_margin_quintile": quintiles,
            "soft_vs_hard_when_hard_correct": float(
                episode.loc[episode["cv_argmin_matches_query_best"] == 1, "soft_vs_hard_gain"].mean()
            ),
            "soft_vs_hard_when_hard_wrong": float(
                episode.loc[episode["cv_argmin_matches_query_best"] == 0, "soft_vs_hard_gain"].mean()
            ),
        }
    summary_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
