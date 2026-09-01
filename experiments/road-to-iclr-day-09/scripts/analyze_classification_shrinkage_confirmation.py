#!/usr/bin/env python3
"""Frozen analysis for independent lambda=0.1 binary shrinkage confirmation."""

from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.methods import competence_weights, prediction_loss, weighted_prediction


def hierarchical(values: dict[str, np.ndarray], draws: int, seed: int) -> dict[str, float]:
    names = sorted(values)
    observed = float(np.mean([values[name].mean() for name in names]))
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for draw in range(draws):
        chosen = rng.choice(names, size=len(names), replace=True)
        samples[draw] = np.mean([
            rng.choice(values[str(name)], size=len(values[str(name)]), replace=True).mean()
            for name in chosen
        ])
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"gain": observed, "ci_low": float(low), "ci_high": float(high)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    raw_paths = sorted((ROOT / "results" / "raw").glob("classification_shrinkage_confirmation_*.npz"))
    if len(raw_paths) != 1:
        raise RuntimeError(f"expected one raw bundle, found {raw_paths}")
    raw_path = raw_paths[0]
    stem = raw_path.stem
    metadata = json.loads((raw_path.parent / f"{stem}.metadata.json").read_text())
    config = __import__("yaml").safe_load((ROOT / metadata["config"]).read_text())
    candidate_lambda = float(config["candidate_lambda"])
    tuning = metadata["synthetic_tuning"]["classification"]
    fixed_weights = np.asarray(tuning["fixed_weights"], dtype=float)
    with np.load(raw_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}

    rows = []
    for index in range(len(arrays["dataset"])):
        experts = arrays["expert_prediction"][index].astype(float)
        y = arrays["query_y"][index].astype(float)
        adaptive_weights = competence_weights(
            arrays["cv_expert_loss"][index].astype(float),
            float(tuning["temperature"]), float(tuning["uniform_shrinkage"]),
        )
        fixed = np.clip(weighted_prediction(experts, fixed_weights), 1e-7, 1 - 1e-7)
        full = np.clip(weighted_prediction(experts, adaptive_weights), 1e-7, 1 - 1e-7)
        candidate = (1 - candidate_lambda) * fixed + candidate_lambda * full
        for method, prediction in (("fixed", fixed), ("shrink_0.1", candidate), ("full", full)):
            nll = -(y * np.log(prediction) + (1 - y) * np.log(1 - prediction))
            rows.append({
                "episode_index": index,
                "dataset": str(arrays["dataset"][index]),
                "repeat": int(arrays["repeat"][index]),
                "method": method,
                "log_loss": float(nll.mean()),
                "brier": float(np.mean((prediction - y) ** 2)),
                "auc": float(roc_auc_score(y, prediction)),
                "nll_over_2_rate": float(np.mean(nll > 2.0)),
            })

    detail = pd.DataFrame(rows)
    detail_path = ROOT / "results" / "processed" / "classification_shrinkage_confirmation_detail_v1.csv"
    audit_path = ROOT / "results" / "processed" / "classification_shrinkage_confirmation_audit_v1.json"
    detail.to_csv(detail_path, index=False)
    parent_path = ROOT / metadata["processed_summary"]
    parent = pd.read_csv(parent_path)
    parent = parent[parent["method"].isin(["fixed", "competence"])].copy()
    parent["method"] = parent["method"].replace({"competence": "full"})
    parity = detail[detail["method"].isin(["fixed", "full"])].merge(
        parent[["episode_index", "method", "loss"]],
        on=["episode_index", "method"], validate="one_to_one",
    )
    parent_max_error = float(np.max(np.abs(parity["log_loss"] - parity["loss"])))
    if parent_max_error > 1e-5:
        raise AssertionError(f"parent log-loss mismatch: {parent_max_error}")
    pivot = detail.pivot(index=["dataset", "episode_index"], columns="method")
    comparisons = {}
    definitions = {
        "candidate_log_loss": ("log_loss", "fixed", "shrink_0.1"),
        "full_log_loss": ("log_loss", "fixed", "full"),
        "candidate_brier": ("brier", "fixed", "shrink_0.1"),
        "candidate_auc": ("auc", "shrink_0.1", "fixed"),
        "candidate_nll_over_2_rate": ("nll_over_2_rate", "fixed", "shrink_0.1"),
    }
    for offset, (label, (metric, left, right)) in enumerate(definitions.items()):
        delta = pivot[(metric, left)] - pivot[(metric, right)]
        values = {
            str(dataset): group.to_numpy()
            for dataset, group in delta.groupby(level="dataset", sort=True)
        }
        record = hierarchical(values, int(config["bootstrap_draws"]), 225501 + offset)
        record["per_dataset_gain"] = {
            name: float(value.mean()) for name, value in values.items()
        }
        comparisons[label] = record
    primary = comparisons["candidate_log_loss"]
    positive = sum(value > 0 for value in primary["per_dataset_gain"].values())
    dataset_values = np.asarray(list(primary["per_dataset_gain"].values()), dtype=float)
    sensitivity_rng = np.random.default_rng(226001)
    dataset_only_samples = dataset_values[
        sensitivity_rng.integers(0, len(dataset_values), size=(100_000, len(dataset_values)))
    ].mean(axis=1)
    dataset_only_ci = np.quantile(dataset_only_samples, [0.025, 0.975])
    leave_one_out = {
        name: float(np.delete(dataset_values, index).mean())
        for index, name in enumerate(primary["per_dataset_gain"])
    }
    audit = {
        "protocol": "CLASSIFICATION_SHRINKAGE_CONFIRMATION_PROTOCOL.md",
        "parent_run": metadata["run_key"],
        "parent_max_abs_log_loss_error": parent_max_error,
        "raw_bundle_sha256": sha256(raw_path),
        "parent_cells_sha256": sha256(parent_path),
        "candidate_lambda": candidate_lambda,
        "datasets": sorted(detail["dataset"].unique().tolist()),
        "episodes": int(detail["episode_index"].nunique()),
        "comparisons": comparisons,
        "positive_datasets": positive,
        "confirmation_pass": bool(primary["ci_low"] > 0 and positive >= 3),
        "post_result_dependence_sensitivity": {
            "dataset_only_bootstrap_draws": 100_000,
            "dataset_only_bootstrap_seed": 226001,
            "dataset_only_ci": dataset_only_ci.tolist(),
            "leave_one_dataset_out_gain": leave_one_out,
            "minimum_leave_one_dataset_out_gain": min(leave_one_out.values()),
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    names = list(primary["per_dataset_gain"])
    gains = np.asarray(list(primary["per_dataset_gain"].values()))
    figure_path = ROOT / "figures" / "classification_shrinkage_confirmation_v1.png"
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    positions = np.arange(len(names))
    colors = np.where(gains >= 0, "#54A24B", "#E45756")
    axis.scatter(gains, positions, c=colors, s=55, zorder=3)
    axis.axvline(0, color="#555555", linestyle="--", linewidth=1)
    overall_y = len(names) + 0.5
    axis.errorbar(
        primary["gain"], overall_y,
        xerr=[[primary["gain"] - primary["ci_low"]],
              [primary["ci_high"] - primary["gain"]]],
        fmt="D", color="black", capsize=4, markersize=6,
    )
    axis.set_yticks(list(positions) + [overall_y], names + ["dataset mean (95% CI)"])
    axis.invert_yaxis()
    axis.set_xlabel("Fixed − 10%-competence log loss  (positive is better)")
    axis.set_title("Independent numeric-binary shrinkage confirmation")
    axis.grid(axis="x", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
