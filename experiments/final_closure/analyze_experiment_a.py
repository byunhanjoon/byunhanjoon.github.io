"""Analyze the frozen independent canonical-seed showdown."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import closure_core as core
from analysis_utils import dataset_cluster_bootstrap, markdown_table, squared_residual, write_summary
from closure_designs import rows_to_ids, sample_schema_design

sys.path.insert(0, str(core.DAY5))
import analyze_completion_panel as previous  # noqa: E402


OUT = core.HERE / "summaries"
FIGURES = core.HERE / "figures"
TABLES = core.HERE / "tables"
DRAWS = int(core.CONFIG["experiment_a"]["estimator_draws"])
BUDGETS = [int(value) for value in core.CONFIG["experiment_a"]["budgets"]]
INDEPENDENT_METHODS = [
    "IID-JOINT", "SRS-JOINT", "OC1-INDEPENDENT", "OC2-INDEPENDENT"
]


def choose_pool_predictions(
    pool: np.ndarray, design_rows: np.ndarray, cards: tuple[int, int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    ids = rows_to_ids(design_rows, cards)
    selected = []
    for action_id in np.unique(ids):
        locations = np.flatnonzero(ids == action_id)
        if len(locations) > pool.shape[1]:
            raise ValueError("design repeats one schema more than cached independent seeds")
        seed_indices = rng.choice(pool.shape[1], size=len(locations), replace=False)
        for location, seed_index in zip(locations, seed_indices):
            selected.append((int(location), pool[int(action_id), int(seed_index)]))
    selected.sort(key=lambda item: item[0])
    return np.stack([item[1] for item in selected])


def coupled_draws(
    predictions: np.ndarray, actions: np.ndarray, cards: tuple[int, ...],
    budget: int, rng: np.random.Generator,
) -> np.ndarray:
    lookup = {tuple(int(value) for value in row): index for index, row in enumerate(actions)}
    base = previous.strength2_base(cards)
    blocks = []
    while sum(block.shape[1] for block in blocks) < budget:
        block = previous.randomize(base, cards, DRAWS, int(rng.integers(0, 2**31)))
        blocks.append(block)
    designs = np.concatenate(blocks, axis=1)[:, :budget]
    output = np.empty((DRAWS, *predictions.shape[1:]), dtype=np.float64)
    for draw in range(DRAWS):
        indices = [lookup[tuple(int(value) for value in row)] for row in designs[draw]]
        output[draw] = predictions[indices].mean(axis=0)
    return output


def analyze_cell(path: Path) -> tuple[list[dict], dict]:
    manifest = json.loads((path / "manifest.json").read_text())
    dataset = manifest["dataset"]; split = int(manifest["split_seed"]); model = manifest["model"]
    task = manifest["task"]
    canonical = np.load(path / "canonical_test.npy", mmap_mode="r")
    joint = np.load(path / "joint_test.npy", mmap_mode="r")
    schema_actions = np.load(path / "schema_actions.npy")
    cards = tuple(int(value) for value in manifest["schema_cards"])
    qcanonical = np.asarray(canonical, dtype=np.float64).mean(axis=0)
    qjoint = np.asarray(joint, dtype=np.float64).mean(axis=(0, 1))
    target = np.load(path / "test_y.npy")
    coupled_path = (
        core.DAY5 / "results" / "completion_neural"
        / f"{dataset}__{model}__split{split}__broad.npz"
    )
    if not coupled_path.exists():
        raise FileNotFoundError(f"missing coupled tensor {coupled_path}")
    with np.load(coupled_path) as artifact:
        coupled_predictions = np.asarray(artifact["test_predictions"], dtype=np.float64)
        coupled_actions = np.asarray(artifact["actions"], dtype=np.int16)
    qcoupled = coupled_predictions.mean(axis=0)
    coupled_cards = tuple(int(coupled_actions[:, index].max()) + 1 for index in range(coupled_actions.shape[1]))
    reference_row = {
        "dataset": dataset, "split_seed": split, "model": model, "task": task,
        "canonical_joint_distance": squared_residual(qcanonical, qjoint),
        "canonical_coupled_distance": squared_residual(qcanonical, qcoupled),
        "joint_coupled_distance": squared_residual(qjoint, qcoupled),
    }
    # Reference uncertainty uses only seed/block resampling and is descriptive.
    ref_rng = np.random.default_rng(core.stable_seed("A-analysis-reference", dataset, split, model))
    boot_distances = np.empty(DRAWS)
    boot_noise = np.empty(DRAWS)
    for draw in range(DRAWS):
        # Preserve the covariance of the physically shared canonical-schema
        # seed prefix rather than bootstrapping the two references as if their
        # overlapping fits were independent.
        while True:
            canonical_indices = ref_rng.integers(0, len(canonical), len(canonical))
            canonical_weights = np.bincount(
                canonical_indices, minlength=len(canonical)
            ).astype(np.float64)
            if canonical_weights[: joint.shape[1]].sum() > 0:
                break
        qc = np.tensordot(
            canonical_weights / canonical_weights.sum(),
            np.asarray(canonical, dtype=np.float64), axes=(0, 0),
        )
        action_means = [np.average(
            np.asarray(joint[0], dtype=np.float64), axis=0,
            weights=canonical_weights[: joint.shape[1]],
        )]
        for action in range(1, joint.shape[0]):
            indices = ref_rng.integers(0, joint.shape[1], joint.shape[1])
            action_means.append(np.asarray(joint[action])[indices].mean(axis=0))
        qj = np.stack(action_means).mean(axis=0)
        boot_distances[draw] = squared_residual(qc, qj)
        boot_noise[draw] = squared_residual((qc - qcanonical) - (qj - qjoint), np.zeros_like(qjoint))
    reference_row.update({
        "canonical_joint_distance_bootstrap_low": float(np.quantile(boot_distances, 0.025)),
        "canonical_joint_distance_bootstrap_high": float(np.quantile(boot_distances, 0.975)),
        "canonical_joint_mc_noise_95": float(np.quantile(boot_noise, 0.95)),
        "canonical_joint_distinguishable_from_mc": bool(
            reference_row["canonical_joint_distance"] > np.quantile(boot_noise, 0.95)
        ),
    })
    rows: list[dict] = []
    for budget in BUDGETS:
        rng = np.random.default_rng(core.stable_seed("A-analysis", dataset, split, model, budget))
        canonical_estimators = np.stack([
            canonical[rng.choice(len(canonical), size=budget, replace=False)].mean(axis=0)
            for _ in range(DRAWS)
        ])
        method_estimators: dict[str, np.ndarray] = {"CANONICAL-INDEPENDENT": canonical_estimators}
        for method in INDEPENDENT_METHODS:
            estimates = []
            for _ in range(DRAWS):
                for _attempt in range(100):
                    design_rows = sample_schema_design(method, cards, budget, rng)
                    ids, counts = np.unique(rows_to_ids(design_rows, cards), return_counts=True)
                    if counts.max() <= joint.shape[1]:
                        break
                estimates.append(choose_pool_predictions(joint, design_rows, cards, rng).mean(axis=0))
            method_estimators[method] = np.stack(estimates)
        method_estimators["OC2-COUPLED"] = coupled_draws(
            coupled_predictions, coupled_actions, coupled_cards, budget, rng
        )
        if budget in (32, 64):
            packed = previous.packed_families(coupled_cards, int(rng.integers(0, 2**31)))
            ids = packed["disjoint_pair32" if budget == 32 else "disjoint_pack64"]
            method_estimators["OC2-PACKED"] = coupled_predictions[ids].mean(axis=1)
        for method, estimators in method_estimators.items():
            reference_name, reference = (
                ("Q_CANONICAL", qcanonical) if method == "CANONICAL-INDEPENDENT"
                else (("Q_COUPLED", qcoupled) if method in {"OC2-COUPLED", "OC2-PACKED"}
                      else ("Q_JOINT", qjoint))
            )
            residuals = np.asarray([squared_residual(value, reference) for value in estimators])
            estimator_mean = np.asarray(estimators, dtype=np.float64).mean(axis=0)
            rows.append({
                "dataset": dataset, "split_seed": split, "model": model, "task": task,
                "method": method, "budget": budget, "reference": reference_name,
                "residual_mean": float(residuals.mean()),
                "residual_median": float(np.median(residuals)),
                "residual_variance_across_constructions": float(residuals.var(ddof=1)),
                "residual_se": float(residuals.std(ddof=1) / math.sqrt(len(residuals))),
                "estimator_prediction_variance": float(np.mean(
                    (np.asarray(estimators, dtype=np.float64) - estimator_mean) ** 2
                )),
                "residual_to_qcanonical": float(np.mean([squared_residual(value, qcanonical) for value in estimators])),
                "residual_to_qjoint": float(np.mean([squared_residual(value, qjoint) for value in estimators])),
                "residual_to_qcoupled": float(np.mean([squared_residual(value, qcoupled) for value in estimators])),
                "predictive_loss": float(np.mean([core.target_loss(value, target, task) for value in estimators])),
                "estimator_draws": DRAWS,
            })
    return rows, reference_row


def paired_comparison(frame: pd.DataFrame, left: str, right: str) -> dict:
    subset = frame[(frame["budget"] == 16) & frame["method"].isin([left, right])]
    pivot = subset.pivot_table(
        index=["dataset", "split_seed", "model", "task"], columns="method",
        values="residual_mean",
    ).reset_index()
    pivot["fractional_reduction"] = 1.0 - pivot[left] / pivot[right]
    source = pivot.groupby("dataset", as_index=False)[[left, right]].mean()
    source["fractional_reduction"] = 1.0 - source[left] / source[right]
    interval = dataset_cluster_bootstrap(
        source, "fractional_reduction", draws=10000,
        seed=core.stable_seed("A-bootstrap", left, right) % (2**32),
    )
    architecture = {
        model: float(1.0 - group[left].mean() / group[right].mean())
        for model, group in pivot.groupby("model")
    }
    return {
        "left": left, "right": right, "cell_wins": int((pivot[left] < pivot[right]).sum()),
        "cells": len(pivot), "source_wins": int((source[left] < source[right]).sum()),
        "sources": len(source),
        "equal_source_mean_relative_reduction": float(source["fractional_reduction"].mean()),
        "cell_median_relative_reduction": float(pivot["fractional_reduction"].median()),
        "dataset_clustered_95_interval": list(interval),
        "architecture_relative_reduction": architecture,
    }


def make_figures(frame: pd.DataFrame, references: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    grouped_rows = []
    for (method, budget), group in frame.groupby(["method", "budget"]):
        source = group.groupby("dataset", as_index=False)["residual_mean"].mean()
        low, high = dataset_cluster_bootstrap(
            source, "residual_mean", draws=10000,
            seed=core.stable_seed("A-figure-1", method, budget) % (2**32),
        )
        grouped_rows.append({
            "method": method, "budget": budget,
            "residual_mean": float(source["residual_mean"].mean()),
            "low": low, "high": high,
        })
    grouped = pd.DataFrame(grouped_rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, group in grouped.groupby("method"):
        group = group.sort_values("budget")
        ax.plot(group["budget"], group["residual_mean"], marker="o", label=method)
        ax.fill_between(group["budget"], group["low"], group["high"], alpha=0.12)
    ax.set(xscale="log", yscale="log", xlabel="number of fits", ylabel="quotient residual")
    ax.legend(fontsize=7, ncol=2); fig.tight_layout()
    fig.savefig(FIGURES / "figure_1_independent_seed_showdown.png", dpi=180)
    fig.savefig(FIGURES / "figure_1_independent_seed_showdown.pdf"); plt.close(fig)

    b16 = frame[frame["budget"] == 16].pivot_table(
        index=["dataset", "split_seed", "model"], columns="method", values="residual_mean"
    ).reset_index()
    methods = ["IID-JOINT", "SRS-JOINT", "OC1-INDEPENDENT", "OC2-INDEPENDENT", "OC2-COUPLED"]
    architecture_rows = []
    for model, group in b16.groupby("model"):
        for method in methods:
            architecture_rows.append((model, method, 1 - group[method].mean() / group["CANONICAL-INDEPENDENT"].mean()))
    architecture = pd.DataFrame(architecture_rows, columns=["model", "method", "reduction"])
    fig, ax = plt.subplots(figsize=(9, 5))
    for index, method in enumerate(methods):
        current = architecture[architecture["method"] == method]
        x = np.arange(len(current)) + (index - 2) * 0.15
        ax.bar(x, current["reduction"], width=0.15, label=method)
    ax.set_xticks(np.arange(4), sorted(architecture["model"].unique()))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(ylabel="relative residual reduction vs canonical independent")
    ax.legend(fontsize=7, ncol=2); fig.tight_layout()
    fig.savefig(FIGURES / "figure_2_architecture_b16.png", dpi=180)
    fig.savefig(FIGURES / "figure_2_architecture_b16.pdf"); plt.close(fig)

    expectation_plot = references.groupby(["dataset", "model"], as_index=False)[[
        "canonical_joint_distance", "canonical_joint_distance_bootstrap_low",
        "canonical_joint_distance_bootstrap_high", "canonical_joint_mc_noise_95",
    ]].mean()
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(expectation_plot))
    point = expectation_plot["canonical_joint_distance"].to_numpy()
    low = expectation_plot["canonical_joint_distance_bootstrap_low"].to_numpy()
    high = expectation_plot["canonical_joint_distance_bootstrap_high"].to_numpy()
    ax.errorbar(
        x, point,
        yerr=np.vstack((np.maximum(point - low, 0), np.maximum(high - point, 0))),
        fmt="o", markersize=2.5, linewidth=0.5, alpha=0.7,
        label="canonical/joint distance with seed bootstrap interval",
    )
    ax.scatter(
        x, expectation_plot["canonical_joint_mc_noise_95"], s=7, alpha=0.5,
        label="95% Monte Carlo-noise threshold",
    )
    labels = (expectation_plot["dataset"] + "/" + expectation_plot["model"]).tolist()
    ax.set_xticks(x, labels, rotation=70, ha="right", fontsize=6)
    ax.set_yscale("log"); ax.set(ylabel="squared distance", xlabel="dataset/split/model cell")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_3_expectation_distance.png", dpi=180)
    fig.savefig(FIGURES / "figure_3_expectation_distance.pdf"); plt.close(fig)


def iid_equivalent_budgets(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["dataset", "split_seed", "model", "task"]
    for key, group in frame.groupby(keys):
        canonical = group[group["method"] == "CANONICAL-INDEPENDENT"].sort_values("budget")
        budgets = canonical["budget"].to_numpy(dtype=float)
        residuals = np.minimum.accumulate(canonical["residual_mean"].to_numpy(dtype=float))
        for method in ("OC2-INDEPENDENT", "OC2-COUPLED"):
            target = float(group[(group["method"] == method) & (group["budget"] == 16)]["residual_mean"].iloc[0])
            if target < residuals[-1]:
                equivalent = np.nan; bracket = ">64"
            elif target >= residuals[0]:
                equivalent = float(budgets[0]); bracket = "<=4"
            else:
                equivalent = float(np.exp(np.interp(
                    np.log(target), np.log(residuals[::-1]), np.log(budgets[::-1])
                )))
                bracket = "interpolated"
            rows.append({
                **dict(zip(keys, key)), "method": method,
                "orbitcover_budget": 16, "canonical_iid_equivalent_budget": equivalent,
                "bracket": bracket,
            })
    return pd.DataFrame(rows)


def main() -> None:
    manifests = sorted((core.RAW / "experiment_a").glob("*/manifest.json"))
    expected = len(core.CONFIG["all_datasets"]) * len(core.CONFIG["split_seeds"]) * len(core.CONFIG["primary_models"])
    if len(manifests) != expected:
        raise AssertionError(f"Experiment A missing cells: {len(manifests)}/{expected}")
    rows = []; references = []
    for manifest in manifests:
        current_rows, reference = analyze_cell(manifest.parent)
        rows.extend(current_rows); references.append(reference)
        print(f"analyzed A {manifest.parent.name}", flush=True)
    frame = pd.DataFrame(rows)
    reference_frame = pd.DataFrame(references)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "experiment_a_cells.csv", index=False)
    reference_frame.to_csv(OUT / "experiment_a_references.csv", index=False)
    equivalents = iid_equivalent_budgets(frame)
    equivalents.to_csv(OUT / "experiment_a_iid_equivalent_budgets.csv", index=False)
    comparisons = [
        paired_comparison(frame, "OC2-COUPLED", "CANONICAL-INDEPENDENT"),
        paired_comparison(frame, "OC2-INDEPENDENT", "CANONICAL-INDEPENDENT"),
        paired_comparison(frame, "OC2-COUPLED", "OC2-INDEPENDENT"),
        paired_comparison(frame, "OC2-COUPLED", "SRS-JOINT"),
        paired_comparison(frame, "OC2-COUPLED", "IID-JOINT"),
    ]
    b16 = frame[frame["budget"] == 16]
    table_rows = []
    canonical = b16[b16["method"] == "CANONICAL-INDEPENDENT"].set_index(
        ["dataset", "split_seed", "model"]
    )["residual_mean"]
    for method, group in b16.groupby("method"):
        indexed = group.set_index(["dataset", "split_seed", "model"])["residual_mean"]
        aligned = pd.concat([indexed.rename("method"), canonical.rename("canonical")], axis=1).dropna()
        source = aligned.groupby(level=0).mean()
        reduction = 1 - source["method"] / source["canonical"]
        low, high = dataset_cluster_bootstrap(
            reduction.rename("reduction").reset_index(), "reduction", draws=10000,
            seed=core.stable_seed("A-table", method) % (2**32),
        )
        table_rows.append({
            "method": method, "mean_residual": float(indexed.mean()),
            "relative_to_canonical": float(indexed.mean() / canonical.mean()),
            "cell_wins": int((aligned["method"] < aligned["canonical"]).sum()),
            "cells": len(aligned), "source_wins": int((source["method"] < source["canonical"]).sum()),
            "sources": len(source), "mean_relative_reduction": float(reduction.mean()),
            "median_relative_reduction": float((1 - aligned["method"] / aligned["canonical"]).median()),
            "clustered_95_low": low, "clustered_95_high": high,
        })
    table = pd.DataFrame(table_rows).sort_values("mean_residual")
    TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES / "table_A_independent_seed_comparison.csv", index=False)
    markdown_table(table, TABLES / "table_A_independent_seed_comparison.md")
    summary = {
        "status": "complete", "cells": expected, "estimator_draws": DRAWS,
        "headline_comparisons_b16": comparisons,
        "canonical_joint_distance_mean": float(reference_frame["canonical_joint_distance"].mean()),
        "canonical_joint_distance_median": float(reference_frame["canonical_joint_distance"].median()),
        "canonical_joint_distinguishable_cells": int(
            reference_frame["canonical_joint_distinguishable_from_mc"].sum()
        ),
        "canonical_joint_distinguishable_total": len(reference_frame),
        "iid_equivalent_budget": {
            method: {
                "median_bracketed": float(group["canonical_iid_equivalent_budget"].median()),
                "bracketed_cells": int(group["canonical_iid_equivalent_budget"].notna().sum()),
                "cells_above_64": int((group["bracket"] == ">64").sum()),
                "total_cells": len(group),
            }
            for method, group in equivalents.groupby("method")
        },
    }
    write_summary(OUT / "experiment_a_summary.json", summary)
    make_figures(frame, reference_frame)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
