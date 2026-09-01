"""Analyze realistic-scale, convergence, and matched-path Experiment B."""

from __future__ import annotations

import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import closure_core as core
from analysis_utils import (
    dataset_cluster_bootstrap, full_factor_components, markdown_table, oa_factor_components,
    squared_residual, summarize_orders, write_summary,
)
from closure_designs import rows_to_ids, sample_schema_design


OUT = core.HERE / "summaries"
TABLES = core.HERE / "tables"
FIGURES = core.HERE / "figures"
DRAWS = 512


def choose_schema_estimator(
    predictions: np.ndarray, rows: np.ndarray, cards: tuple[int, int, int],
    method: str, budget: int, rng: np.random.Generator,
) -> np.ndarray:
    design = sample_schema_design(method, cards, budget, rng)
    lookup: dict[tuple[int, int, int], list[int]] = {}
    for index, row in enumerate(rows):
        lookup.setdefault(tuple(int(value) for value in row[:3]), []).append(index)
    chosen = []
    for schema in design:
        key = tuple(int(value) for value in schema)
        candidates = lookup.get(key, [])
        used = sum(1 for old in chosen if tuple(int(value) for value in rows[old, :3]) == key)
        if used >= len(candidates):
            raise ValueError(f"insufficient cached seed replicates for {key}")
        available = [value for value in candidates if value not in chosen]
        chosen.append(int(rng.choice(available)))
    return np.asarray(predictions[chosen]).mean(axis=0)


def curve_summary(path) -> dict:
    curves = sorted((path / "curves").glob("*.json"))
    if not curves:
        return {
            "best_epoch_mean": np.nan, "stopped_epoch_mean": np.nan,
            "final_training_loss_mean": np.nan, "final_validation_loss_mean": np.nan,
            "final_gradient_norm_mean": np.nan, "final_parameter_update_norm_mean": np.nan,
        }
    payloads = [json.loads(curve.read_text()) for curve in curves]
    finals = [payload["trajectory"][-1] for payload in payloads]
    return {
        "best_epoch_mean": float(np.mean([payload["best_epoch"] for payload in payloads])),
        "stopped_epoch_mean": float(np.mean([payload["stopped_epoch"] for payload in payloads])),
        "final_training_loss_mean": float(np.mean([value["training_loss"] for value in finals])),
        "final_validation_loss_mean": float(np.mean([value["validation_loss"] for value in finals])),
        "final_gradient_norm_mean": float(np.mean([value["gradient_norm"] for value in finals])),
        "final_parameter_update_norm_mean": float(np.mean([value["parameter_update_norm"] for value in finals])),
    }


def analyze_condition(path) -> dict:
    manifest = json.loads((path / "manifest.json").read_text())
    predictions = np.load(path / "test_predictions.npy", mmap_mode="r")
    canonical = np.load(path / "canonical_test_predictions.npy", mmap_mode="r")
    design = np.load(path / "design_rows.npy")
    cards = tuple(int(value) for value in manifest["schema_cards"])
    full_cards = (*cards, 8)
    qjoint = np.asarray(predictions, dtype=np.float64).mean(axis=0)
    qcanonical = np.asarray(canonical, dtype=np.float64).mean(axis=0)
    total = float(np.mean((np.asarray(predictions, dtype=np.float64) - qjoint) ** 2))
    if bool(manifest["full_product"]):
        energies = full_factor_components(np.asarray(predictions, dtype=np.float64), full_cards)
    else:
        energies = oa_factor_components(
            np.asarray(predictions, dtype=np.float64), design, full_cards, maximum_order=3
        )
    spectrum = summarize_orders(energies, total)
    schema_only = float(sum(
        value for subset, value in energies.items()
        if 3 not in subset and bool(subset)
    ))
    stochastic_only = float(energies.get((3,), 0.0))
    schema_stochastic = max(total - schema_only - stochastic_only, 0.0)
    rng = np.random.default_rng(
        core.stable_seed("B-analysis", manifest["dataset"], manifest["model"],
                         manifest["training_size_label"], manifest["budget"])
    )
    method_means = {}
    for budget in (16, 32):
        residuals = {
            "canonical": [], "iid": [], "srs": [], "oc2": [],
        }
        for _ in range(DRAWS):
            canonical_estimate = canonical[rng.choice(len(canonical), size=budget, replace=False)].mean(axis=0)
            iid_estimate = predictions[rng.choice(len(predictions), size=budget, replace=True)].mean(axis=0)
            srs_estimate = predictions[rng.choice(len(predictions), size=budget, replace=False)].mean(axis=0)
            for _attempt in range(100):
                try:
                    oc2_estimate = choose_schema_estimator(
                        predictions, design, cards, "OC2-INDEPENDENT", budget, rng
                    )
                    break
                except ValueError:
                    continue
            else:
                raise AssertionError("could not realize cached B OC2 estimator")
            residuals["canonical"].append(squared_residual(canonical_estimate, qcanonical))
            residuals["iid"].append(squared_residual(iid_estimate, qjoint))
            residuals["srs"].append(squared_residual(srs_estimate, qjoint))
            residuals["oc2"].append(squared_residual(oc2_estimate, qjoint))
        for method, values in residuals.items():
            method_means[f"{method}{budget}_residual"] = float(np.mean(values))
    row = {
        "dataset": manifest["dataset"], "model": manifest["model"], "task": manifest["task"],
        "split_seed": manifest["split_seed"],
        "training_size_label": manifest["training_size_label"],
        "training_rows": manifest["training_rows"], "budget": manifest["budget"],
        "budget_order": 500 if manifest["budget"] == "convergence" else int(manifest["budget"]),
        "full_product": manifest["full_product"],
        "population": len(predictions), "factor_count": len(full_cards),
        "schema_only_variance": schema_only,
        "stochastic_only_variance": stochastic_only,
        "schema_stochastic_interaction": schema_stochastic,
        "canonical_joint_expectation_distance": squared_residual(qcanonical, qjoint),
        **spectrum, **method_means, **curve_summary(path),
    }
    row["oc2_srs_ratio_b16"] = row["oc216_residual"] / row["srs16_residual"] if row["srs16_residual"] else np.nan
    row["oc2_canonical_ratio_b16"] = row["oc216_residual"] / row["canonical16_residual"] if row["canonical16_residual"] else np.nan
    row["oc2_srs_ratio_b32"] = row["oc232_residual"] / row["srs32_residual"] if row["srs32_residual"] else np.nan
    return row


def matched_rows() -> pd.DataFrame:
    rows = []
    for manifest_path in sorted((core.RAW / "matched_convergence").glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        predictions = np.load(manifest_path.parent / "test_predictions.npy", mmap_mode="r")
        means = np.asarray(predictions, dtype=np.float64).mean(axis=1, keepdims=True)
        variances = np.mean((np.asarray(predictions, dtype=np.float64) - means) ** 2, axis=(1, 2, 3))
        ordinary, matched = [float(value) for value in variances]
        rows.append({
            "dataset": manifest["dataset"], "model": manifest["model"],
            "task": manifest["task"], "budget": manifest["budget"],
            "budget_order": 500 if manifest["budget"] == "convergence" else int(manifest["budget"]),
            "training_rows": manifest["training_rows"],
            "ordinary_variance": ordinary, "matched_variance": matched,
            "fraction_removed": 1 - matched / ordinary if ordinary else np.nan,
            "maximum_initial_gap": manifest["maximum_initial_gap"],
        })
    return pd.DataFrame(rows)


def descriptive_slopes(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Dataset-clustered descriptive log-risk scaling summaries."""
    rows = []
    positive = frame[frame["total_nuisance_variance"] > 0].copy()
    for (dataset, model, budget), group in positive.groupby(["dataset", "model", "budget"]):
        unique = group.groupby("training_rows", as_index=False)["total_nuisance_variance"].mean()
        if len(unique) >= 2:
            rows.append({
                "axis": "training_size", "dataset": dataset, "model": model,
                "condition": str(budget), "slope": float(np.polyfit(
                    np.log(unique["training_rows"]), np.log(unique["total_nuisance_variance"]), 1
                )[0]),
            })
    fixed = positive[positive["budget"].astype(str).isin(["20", "50", "100", "200"])]
    for (dataset, model, training_rows), group in fixed.groupby(
        ["dataset", "model", "training_rows"]
    ):
        unique = group.groupby("budget_order", as_index=False)["total_nuisance_variance"].mean()
        if len(unique) >= 2:
            rows.append({
                "axis": "optimization_budget", "dataset": dataset, "model": model,
                "condition": str(training_rows), "slope": float(np.polyfit(
                    np.log(unique["budget_order"]), np.log(unique["total_nuisance_variance"]), 1
                )[0]),
            })
    slopes = pd.DataFrame(rows)
    summary = {}
    for (axis, model), group in slopes.groupby(["axis", "model"]):
        source = group.groupby("dataset", as_index=False)["slope"].mean()
        low, high = dataset_cluster_bootstrap(
            source, "slope", draws=10000,
            seed=core.stable_seed("B-slope", axis, model) % (2**32),
        )
        summary[f"{axis}/{model}"] = {
            "equal_dataset_mean_slope": float(source["slope"].mean()),
            "dataset_clustered_95_interval": [low, high],
            "datasets": len(source), "trajectory_slopes": len(group),
        }
    return slopes, summary


def make_figures(frame: pd.DataFrame, matched: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    trajectory = frame.groupby(
        ["dataset", "model", "budget_order", "budget"], as_index=False
    )["total_nuisance_variance"].mean().groupby(
        ["model", "budget_order", "budget"], as_index=False
    )["total_nuisance_variance"].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, group in trajectory.groupby("model"):
        group = group.sort_values("budget_order")
        ax.plot(group["budget_order"], group["total_nuisance_variance"], marker="o", label=model)
    ax.set(yscale="log", xlabel="epochs (convergence shown at 500)", ylabel="nuisance variance")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIGURES / "figure_4_convergence.png", dpi=180)
    fig.savefig(FIGURES / "figure_4_convergence.pdf"); plt.close(fig)

    sizes = frame[frame["budget"].astype(str).isin(["20", "convergence"])].groupby(
        ["model", "training_rows", "budget"], as_index=False
    )["total_nuisance_variance"].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    for (model, budget), group in sizes.groupby(["model", "budget"]):
        ax.plot(group["training_rows"], group["total_nuisance_variance"], marker="o", label=f"{model}/{budget}")
    ax.set(xscale="log", yscale="log", xlabel="training rows", ylabel="nuisance variance")
    ax.legend(fontsize=7, ncol=2); fig.tight_layout()
    fig.savefig(FIGURES / "figure_5_training_scale.png", dpi=180)
    fig.savefig(FIGURES / "figure_5_training_scale.pdf"); plt.close(fig)

    ratio = frame.groupby(
        ["dataset", "model", "budget_order"], as_index=False
    )["oc2_srs_ratio_b16"].mean().groupby(
        ["model", "budget_order"], as_index=False
    )["oc2_srs_ratio_b16"].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, group in ratio.groupby("model"):
        ax.plot(group["budget_order"], group["oc2_srs_ratio_b16"], marker="o", label=model)
    ax.axhline(1, color="black", linewidth=0.8)
    ax.set(xlabel="epochs (convergence shown at 500)", ylabel="OC2 / SRS residual")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIGURES / "figure_6_orbitcover_convergence.png", dpi=180)
    fig.savefig(FIGURES / "figure_6_orbitcover_convergence.pdf"); plt.close(fig)

    if len(matched):
        grouped = matched.groupby(["model", "budget_order"], as_index=False)[["ordinary_variance", "matched_variance"]].mean()
        fig, axes = plt.subplots(1, 4, figsize=(13, 4), sharey=True)
        for ax, (model, group) in zip(axes, grouped.groupby("model")):
            group = group.sort_values("budget_order")
            ax.plot(group["budget_order"], group["ordinary_variance"], marker="o", label="ordinary")
            ax.plot(group["budget_order"], group["matched_variance"], marker="o", label="matched")
            ax.set_title(model); ax.set_yscale("log")
        axes[0].set_ylabel("schema variance"); axes[0].legend(); fig.tight_layout()
        fig.savefig(FIGURES / "figure_9_matched_convergence.png", dpi=180)
        fig.savefig(FIGURES / "figure_9_matched_convergence.pdf"); plt.close(fig)


def main() -> None:
    manifests = sorted((core.RAW / "experiment_b").glob("*/manifest.json"))
    expected = 0
    config = core.completion_config()
    for dataset in core.CONFIG["experiment_b"]["datasets"]:
        prepared, _ = core.b_prepared_datasets(dataset, int(core.CONFIG["experiment_b"]["split_seed"]), config)
        expected += len(prepared) * 5 * len(core.CONFIG["primary_models"])
    if len(manifests) != expected:
        raise AssertionError(f"Experiment B missing conditions {len(manifests)}/{expected}")
    frame = pd.DataFrame([analyze_condition(path.parent) for path in manifests])
    matched = matched_rows()
    matched_expected = 2 * 4 * 3
    if len(matched) != matched_expected:
        raise AssertionError(f"matched convergence missing {len(matched)}/{matched_expected}")
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "experiment_b_conditions.csv", index=False)
    matched.to_csv(OUT / "experiment_b_matched_convergence.csv", index=False)
    slopes, slope_summary = descriptive_slopes(frame)
    slopes.to_csv(OUT / "experiment_b_descriptive_slopes.csv", index=False)

    corners = []
    for (dataset, model), group in frame.groupby(["dataset", "model"]):
        smallest = int(group["training_rows"].min()); largest = int(group["training_rows"].max())
        for size_name, size in (("small", smallest), ("largest", largest)):
            for budget in (20, "convergence"):
                row = group[(group["training_rows"] == size) & (group["budget"].astype(str) == str(budget))]
                if len(row) != 1:
                    raise AssertionError(f"missing B corner {dataset}/{model}/{size_name}/{budget}")
                current = row.iloc[0].to_dict(); current["corner"] = f"{size_name}/{budget}"
                corners.append(current)
    table = pd.DataFrame(corners)
    TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES / "table_B_convergence.csv", index=False)
    markdown_table(table, TABLES / "table_B_convergence.md")
    model_corners = table.groupby(["model", "corner"], as_index=False)[
        ["total_nuisance_variance", "oc2_srs_ratio_b16", "oc2_canonical_ratio_b16"]
    ].mean()
    convergence = frame[frame["budget"].astype(str) == "convergence"]
    convergence_sources = convergence.groupby("dataset", as_index=False)["oc2_srs_ratio_b16"].mean()
    summary = {
        "status": "complete", "conditions": len(frame), "matched_conditions": len(matched),
        "model_corner_means": model_corners.to_dict(orient="records"),
        "nuisance_persists_at_convergence": bool((convergence["total_nuisance_variance"] > 1e-10).any()),
        "orbitcover_mean_oc2_srs_ratio_at_convergence": float(
            convergence_sources["oc2_srs_ratio_b16"].mean()
        ),
        "descriptive_log_risk_slopes": slope_summary,
        "matched_model_fraction_removed": matched.groupby("model")["fraction_removed"].mean().to_dict(),
        "maximum_matched_initial_gap": float(matched["maximum_initial_gap"].max()),
    }
    write_summary(OUT / "experiment_b_summary.json", summary)
    make_figures(frame, matched)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
