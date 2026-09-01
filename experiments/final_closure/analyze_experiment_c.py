"""Interaction-spectrum prediction of OrbitCover versus SRSWOR (Experiment C)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

import closure_core as core
from analysis_utils import (
    dataset_cluster_bootstrap, full_factor_components, markdown_table,
    summarize_orders, write_summary,
)


DAY5_RESULTS = core.DAY5 / "results"
OUT = core.HERE / "summaries"
TABLES = core.HERE / "tables"
FIGURES = core.HERE / "figures"


def completion_cells() -> pd.DataFrame:
    source = pd.read_csv(DAY5_RESULTS / "completion_neural_risk_cells.csv")
    output = pd.DataFrame({
        "evidence_source": "completion_neural",
        "dataset": source["dataset"], "split_seed": source["split_seed"],
        "model": source["model"], "task": source["task"],
        "population": source["population"], "factor_count": 5,
        "main_fraction": source["main_fraction"],
        "pair_fraction": source["main_pair_fraction"] - source["main_fraction"],
        "main_pair_fraction": source["main_pair_fraction"],
        "triple_fraction": source["triple_fraction"],
        "higher_fraction": source["higher_fraction"],
        "total_nuisance_variance": source["total_nuisance_variance"],
        "oc2_residual": source["strength2_16_residual_mean"],
        "srs_residual": source["srswor16_residual_mean"],
    })
    output["effective_interaction_order"] = (
        output["main_fraction"] + 2 * output["pair_fraction"]
        + 3 * output["triple_fraction"] + 4 * output["higher_fraction"]
    )
    return output


def earlier_cells() -> pd.DataFrame:
    baseline = pd.read_csv(DAY5_RESULTS / "without_replacement_baseline_cells.csv")
    interaction = pd.read_csv(DAY5_RESULTS / "interaction_phase_empirical_cells.csv")
    rows = []
    for panel, panel_rows in baseline.groupby("panel"):
        strength_path = DAY5_RESULTS / f"{panel}_cells.csv"
        strength = pd.read_csv(strength_path)
        strength = strength[strength["split"] == "validation"]
        for row in panel_rows.itertuples(index=False):
            mechanism = interaction[
                (interaction["panel"] == panel)
                & (interaction["dataset"] == row.dataset)
                & (interaction["model"] == row.model)
            ]
            detail = strength[(strength["dataset"] == row.dataset) & (strength["model"] == row.model)]
            if len(mechanism) != 1 or len(detail) != 1:
                raise AssertionError(f"non-unique C join for {panel}/{row.dataset}/{row.model}")
            mechanism = mechanism.iloc[0]; detail = detail.iloc[0]
            rows.append({
                "evidence_source": panel, "dataset": row.dataset,
                "split_seed": "original", "model": row.model, "task": detail["task"],
                "population": row.population, "factor_count": len(str(mechanism["shape"]).split("x")),
                "main_fraction": detail["main_fraction"],
                "pair_fraction": mechanism["low_order_1_2_fraction"] - detail["main_fraction"],
                "main_pair_fraction": mechanism["low_order_1_2_fraction"],
                "triple_fraction": mechanism["order3_fraction"],
                "higher_fraction": mechanism["order4_fraction"],
                "effective_interaction_order": (
                    detail["main_fraction"]
                    + 2 * (mechanism["low_order_1_2_fraction"] - detail["main_fraction"])
                    + 3 * mechanism["order3_fraction"] + 4 * mechanism["order4_fraction"]
                ),
                "total_nuisance_variance": row.joint_risk,
                "oc2_residual": row.strength2_b16_residual,
                "srs_residual": row.srswor_b16_residual,
            })
    output = pd.DataFrame(rows)
    if len(output) != len(baseline):
        raise AssertionError(f"earlier C scope mismatch {len(output)}/{len(baseline)}")
    return output


def experiment_a_schema_cells() -> pd.DataFrame:
    a_summary = pd.read_csv(OUT / "experiment_a_cells.csv")
    b16 = a_summary[
        (a_summary["budget"] == 16)
        & a_summary["method"].isin(["OC2-INDEPENDENT", "SRS-JOINT"])
    ].pivot_table(
        index=["dataset", "split_seed", "model", "task"],
        columns="method", values="residual_mean",
    ).reset_index()
    rows = []
    for manifest_path in sorted((core.RAW / "experiment_a").glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        path = manifest_path.parent
        joint = np.load(path / "joint_test.npy", mmap_mode="r")
        schema_means = np.asarray(joint, dtype=np.float64).mean(axis=1)
        cards = tuple(int(value) for value in manifest["schema_cards"])
        grand = schema_means.mean(axis=0)
        total = float(np.mean((schema_means - grand) ** 2))
        spectrum = summarize_orders(full_factor_components(schema_means, cards), total)
        match = b16[
            (b16["dataset"] == manifest["dataset"])
            & (b16["split_seed"] == manifest["split_seed"])
            & (b16["model"] == manifest["model"])
        ]
        if len(match) != 1:
            raise AssertionError("missing A interaction/residual join")
        match = match.iloc[0]
        rows.append({
            "evidence_source": "final_A_schema", "dataset": manifest["dataset"],
            "split_seed": manifest["split_seed"], "model": manifest["model"],
            "task": manifest["task"], "population": int(np.prod(cards)),
            "factor_count": len(cards), **{key: spectrum[key] for key in (
                "main_fraction", "pair_fraction", "main_pair_fraction", "triple_fraction",
                "higher_fraction", "effective_interaction_order", "total_nuisance_variance"
            )},
            "oc2_residual": match["OC2-INDEPENDENT"],
            "srs_residual": match["SRS-JOINT"],
        })
    return pd.DataFrame(rows)


def experiment_b_cells() -> pd.DataFrame:
    """Return the fully enumerated B corners with exact fANOVA spectra.

    The trajectory OA conditions are intentionally excluded: their order energies
    are projections rather than a complete-product decomposition.  Each retained
    corner is a distinct training-size/optimization-budget nuisance population.
    """
    source = pd.read_csv(OUT / "experiment_b_conditions.csv")
    source = source[source["full_product"]].copy()
    if source.empty:
        raise AssertionError("Experiment B has no full-product cells for C")
    required = [
        "dataset", "split_seed", "model", "task", "population", "factor_count",
        "training_size_label", "budget", "main_fraction", "pair_fraction",
        "main_pair_fraction", "triple_fraction", "higher_fraction",
        "effective_interaction_order", "total_nuisance_variance",
        "oc216_residual", "srs16_residual",
    ]
    missing = set(required) - set(source.columns)
    if missing:
        raise AssertionError(f"Experiment B summary missing C columns: {sorted(missing)}")
    output = source[required].rename(columns={
        "oc216_residual": "oc2_residual", "srs16_residual": "srs_residual",
    })
    output["evidence_source"] = (
        "final_B_full_product_n" + output["training_size_label"].astype(str)
        + "_e" + output["budget"].astype(str)
    )
    return output.drop(columns=["training_size_label", "budget"])


def clustered_spearman(frame: pd.DataFrame, x: str, y: str, seed: int) -> dict:
    valid = frame[["dataset", x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    estimate = float(spearmanr(valid[x], valid[y]).statistic)
    datasets = sorted(valid["dataset"].unique())
    grouped = {dataset: valid[valid["dataset"] == dataset] for dataset in datasets}
    rng = np.random.default_rng(seed)
    draws = np.empty(10000)
    for index in range(len(draws)):
        chosen = rng.choice(datasets, size=len(datasets), replace=True)
        sample = pd.concat([grouped[dataset] for dataset in chosen], ignore_index=True)
        draws[index] = spearmanr(sample[x], sample[y]).statistic
    return {
        "spearman": estimate,
        "dataset_clustered_95_interval": [float(value) for value in np.nanquantile(draws, [0.025, 0.975])],
        "cells": len(valid), "datasets": len(datasets),
    }


def leave_one_dataset_out(frame: pd.DataFrame) -> dict:
    valid = frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["gain", "main_pair_fraction", "sampling_fraction", "model"]
    ).copy()
    predictions = np.empty(len(valid))
    for dataset in valid["dataset"].unique():
        train = valid[valid["dataset"] != dataset]
        test = valid[valid["dataset"] == dataset]
        transformer = ColumnTransformer(
            [("architecture", OneHotEncoder(handle_unknown="ignore"), ["model"])],
            remainder="passthrough",
        )
        model = make_pipeline(transformer, LinearRegression())
        columns = ["model", "main_pair_fraction", "sampling_fraction"]
        model.fit(train[columns], train["gain"])
        predictions[valid.index.get_indexer(test.index)] = model.predict(test[columns])
    return {
        "cells": len(valid), "leave_one_dataset_out_r2": float(r2_score(valid["gain"], predictions)),
        "spearman_prediction": float(spearmanr(valid["gain"], predictions).statistic),
    }


def make_figures(frame: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, group in frame.groupby("model"):
        ax.scatter(group["main_pair_fraction"], group["gain"], s=13, alpha=0.55, label=model)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xlabel="main + pair fANOVA fraction", ylabel="OC2 gain over SRSWOR")
    ax.legend(fontsize=7, ncol=2); fig.tight_layout()
    fig.savefig(FIGURES / "figure_7_interaction_predicts_gain.png", dpi=180)
    fig.savefig(FIGURES / "figure_7_interaction_predicts_gain.pdf"); plt.close(fig)

    extremes = pd.concat((frame.nsmallest(8, "gain"), frame.nlargest(8, "gain")))
    components = extremes[["main_fraction", "pair_fraction", "triple_fraction", "higher_fraction"]]
    labels = (
        extremes["dataset"] + "/" + extremes["model"] + "/"
        + extremes["evidence_source"].astype(str)
    ).tolist()
    fig, ax = plt.subplots(figsize=(11, 6))
    bottom = np.zeros(len(extremes))
    for column in components:
        ax.bar(np.arange(len(extremes)), components[column], bottom=bottom, label=column)
        bottom += components[column].to_numpy()
    ax.set_xticks(np.arange(len(extremes)), labels, rotation=70, ha="right", fontsize=7)
    ax.set(ylabel="fANOVA fraction"); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(FIGURES / "figure_8_failure_cell_spectra.png", dpi=180)
    fig.savefig(FIGURES / "figure_8_failure_cell_spectra.pdf"); plt.close(fig)


def main() -> None:
    frames = [
        completion_cells(), earlier_cells(), experiment_a_schema_cells(),
        experiment_b_cells(),
    ]
    frame = pd.concat(frames, ignore_index=True)
    frame["sampling_fraction"] = np.minimum(16 / frame["population"], 1.0)
    frame["gain"] = np.where(
        frame["srs_residual"] > 0,
        1.0 - frame["oc2_residual"] / frame["srs_residual"], np.nan,
    )
    frame["relative_loss"] = np.where(
        frame["srs_residual"] > 0,
        frame["oc2_residual"] / frame["srs_residual"] - 1.0, np.nan,
    )
    frame["oc2_wins"] = frame["oc2_residual"] < frame["srs_residual"]
    frame["srs_wins"] = frame["srs_residual"] < frame["oc2_residual"]
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "experiment_c_cells.csv", index=False)
    failures = frame[frame["srs_wins"]].sort_values("gain")
    failures.to_csv(OUT / "experiment_c_srs_failure_cells.csv", index=False)
    markdown_table(failures, OUT / "experiment_c_srs_failure_cells.md")

    primary = clustered_spearman(
        frame, "main_pair_fraction", "gain", core.stable_seed("C", "mainpair") % (2**32)
    )
    secondary = clustered_spearman(
        frame, "higher_fraction", "gain", core.stable_seed("C", "higher") % (2**32)
    )
    architecture = {}
    for model, group in frame.groupby("model"):
        valid = group.dropna(subset=["gain"])
        architecture[model] = {
            "main_pair": clustered_spearman(
                valid, "main_pair_fraction", "gain",
                core.stable_seed("C", "architecture", model, "mainpair") % (2**32),
            ) if len(valid) > 2 else None,
            "higher": clustered_spearman(
                valid, "higher_fraction", "gain",
                core.stable_seed("C", "architecture", model, "higher") % (2**32),
            ) if len(valid) > 2 else None,
            "cells": len(valid),
        }
    wins = frame[frame["oc2_wins"]]
    losses = frame[frame["srs_wins"]]
    completion = frame[frame["evidence_source"] == "completion_neural"]
    completion_sources = completion.groupby("dataset", as_index=False)[["oc2_residual", "srs_residual"]].mean()
    completion_sources["gain"] = 1 - completion_sources["oc2_residual"] / completion_sources["srs_residual"]
    nonpositive_sources = completion_sources[
        ~(completion_sources["oc2_residual"] < completion_sources["srs_residual"])
    ].replace([np.inf, -np.inf], np.nan)
    nonpositive_sources.to_csv(OUT / "experiment_c_completion_nonpositive_sources.csv", index=False)
    completion_spectra = completion.groupby("dataset", as_index=False)[
        ["main_fraction", "pair_fraction", "main_pair_fraction", "triple_fraction", "higher_fraction"]
    ].mean()
    completion_spectra["source_group"] = np.where(
        completion_spectra["dataset"].isin(nonpositive_sources["dataset"]),
        "nonpositive", "positive",
    )
    source_spectrum_comparison = completion_spectra.groupby("source_group")[
        ["main_fraction", "pair_fraction", "main_pair_fraction", "triple_fraction", "higher_fraction"]
    ].mean().to_dict(orient="index")
    completion_spectra.to_csv(OUT / "experiment_c_completion_source_spectra.csv", index=False)

    summary = {
        "status": "complete", "cells": len(frame),
        "scope_counts": frame["evidence_source"].value_counts().to_dict(),
        "main_pair_fraction_vs_gain": primary,
        "higher_fraction_vs_gain": secondary,
        "architecture_stratified": architecture,
        "mean_high_order_fraction_oc2_wins": float(wins["higher_fraction"].mean()),
        "mean_high_order_fraction_srs_wins": float(losses["higher_fraction"].mean()),
        "srs_failure_cells": len(failures),
        "completion_nonpositive_sources": nonpositive_sources["dataset"].tolist(),
        "completion_source_spectrum_comparison": source_spectrum_comparison,
        "transparent_model": leave_one_dataset_out(frame),
    }
    write_summary(OUT / "experiment_c_summary.json", summary)
    table = frame.copy()
    table.to_csv(TABLES / "table_C_interaction_prediction.csv", index=False)
    markdown_table(table, TABLES / "table_C_interaction_prediction.md")
    make_figures(frame)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
