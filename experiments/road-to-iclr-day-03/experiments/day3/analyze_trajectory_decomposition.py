"""Analyze the preregistered Day 3 function-space trajectory extension."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

from .trajectory_decomposition import CONFIG_PATH, RESULTS


def _load(pattern: str) -> pd.DataFrame:
    paths = sorted(RESULTS.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No trajectory files match {pattern!r}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def _bootstrap_mean(values: np.ndarray, groups: np.ndarray, samples: int, seed: int = 2026):
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        selected = rng.choice(unique, len(unique), replace=True)
        pieces = [values[groups == group] for group in selected]
        draws.append(np.concatenate(pieces).mean())
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def _features(frame: pd.DataFrame, kind: str) -> np.ndarray:
    model = (frame["model"] == "resnet").astype(float).to_numpy()[:, None]
    if kind == "condition":
        main = np.log10(frame["realized_basis_condition"].to_numpy(float)).reshape(-1, 1)
    elif kind == "step1_drift":
        main = np.log10(frame["val_drift_step1"].to_numpy(float) + 1e-12).reshape(-1, 1)
    elif kind == "early_auc":
        main = np.log10(frame["val_early_drift_auc"].to_numpy(float) + 1e-12).reshape(-1, 1)
    else:
        raise KeyError(kind)
    return np.column_stack((main, model))


def _leave_one_dataset_out(frame: pd.DataFrame, kind: str) -> dict[str, object]:
    predictions = np.full(len(frame), np.nan)
    for dataset in sorted(frame["dataset"].unique()):
        train = frame["dataset"] != dataset
        test = ~train
        model = LinearRegression().fit(_features(frame.loc[train], kind), frame.loc[train, "final_normalized_harm"])
        predictions[test] = model.predict(_features(frame.loc[test], kind))
    actual = frame["final_normalized_harm"].to_numpy(float)
    return {
        "feature": kind,
        "r2": float(r2_score(actual, predictions)),
        "mae": float(mean_absolute_error(actual, predictions)),
        "sign_agreement": float(np.mean(np.signbit(actual) == np.signbit(predictions))),
        "predictions": predictions,
    }


def _trajectory_features(runs: pd.DataFrame, trajectories: pd.DataFrame) -> pd.DataFrame:
    identifiers = ["dataset", "pair_label", "model", "arm", "seed"]
    validation = trajectories[trajectories["probe_split"] == "val"].copy()
    wide = validation.pivot_table(
        index=identifiers, columns="step", values="prediction_drift", aggfunc="first"
    ).reset_index()
    wide.columns = [
        f"val_drift_step{int(column)}" if isinstance(column, (int, float, np.number)) else column
        for column in wide.columns
    ]
    required = [0, 1, 5, 20, 100, 200]
    for step in required:
        name = f"val_drift_step{step}"
        if name not in wide:
            wide[name] = np.nan
    x = np.log1p(np.asarray([1.0, 5.0, 20.0]))
    values = wide[["val_drift_step1", "val_drift_step5", "val_drift_step20"]].to_numpy(float)
    wide["val_early_drift_auc"] = np.trapz(values, x=x, axis=1) / (x[-1] - x[0])
    return runs.merge(wide, on=identifiers, how="left", validate="one_to_one")


def _natural_transfer(frame: pd.DataFrame) -> dict[str, object]:
    controlled = frame[
        (frame["pair_family"] == "controlled") & (frame["arm"] == "matched_adamw")
    ]
    natural = frame[(frame["pair_family"] == "natural") & (frame["arm"] == "matched_adamw")].copy()
    prediction = np.full(len(natural), np.nan)
    natural = natural.reset_index(drop=True)
    for dataset in sorted(natural["dataset"].unique()):
        train = controlled[controlled["dataset"] != dataset]
        test = natural["dataset"] == dataset
        model = LinearRegression().fit(
            _features(train, "step1_drift"), train["final_normalized_harm"]
        )
        prediction[test] = model.predict(_features(natural.loc[test], "step1_drift"))
    actual = natural["final_normalized_harm"].to_numpy(float)
    natural["controlled_trained_prediction"] = prediction
    natural.to_csv(RESULTS / "natural_transfer_predictions.csv", index=False)
    return {
        "cells": int(len(natural)),
        "sign_agreement": float(np.mean(np.signbit(actual) == np.signbit(prediction))),
        "mae": float(mean_absolute_error(actual, prediction)),
        "actual_mean": float(actual.mean()),
        "predicted_mean": float(prediction.mean()),
    }


def _plots(frame: pd.DataFrame) -> None:
    figure_dir = RESULTS / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "ordinary_adamw": "#cc4b37",
        "matched_adamw": "#7047a3",
        "covariance_adamw": "#d59624",
        "matched_input_natural": "#218c74",
        "ordinary_input_natural": "#3384a8",
    }
    controlled = frame[frame["pair_family"] == "controlled"]
    summary = (
        controlled.groupby(["arm", "target_kappa"], as_index=False)["val_drift_step200"].mean()
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for arm, group in summary.groupby("arm"):
        ax.plot(
            group["target_kappa"],
            group["val_drift_step200"],
            marker="o",
            label=arm.replace("_", " "),
            color=colors[arm],
        )
    ax.set_xscale("log")
    ax.set_xlabel("controlled basis condition number")
    ax.set_ylabel("mean prediction drift at step 200")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "trajectory_drift_by_arm.png", dpi=180)
    plt.close(fig)

    matched = controlled[controlled["arm"] == "matched_adamw"]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for dataset, group in matched.groupby("dataset"):
        ax.scatter(
            group["val_drift_step1"],
            group["final_normalized_harm"],
            s=28,
            alpha=0.75,
            label=dataset,
        )
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("matched AdamW prediction drift after one update")
    ax.set_ylabel("final normalized performance harm")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "early_drift_vs_final_harm.png", dpi=180)
    plt.close(fig)


def main() -> None:
    cfg = json.loads(CONFIG_PATH.read_text())
    runs = _load("runs_gpu[0-9].csv")
    trajectories = _load("runs_gpu*_trajectories.csv")
    identifiers = ["dataset", "pair_label", "model", "arm", "seed"]
    if runs.duplicated(identifiers).any():
        raise AssertionError("Duplicate completed trajectory cells")
    failures = runs["failure"].fillna("").astype(str).str.strip() != ""
    successful = runs[~failures].copy()
    frame = _trajectory_features(successful, trajectories)
    expected = (
        len(cfg["datasets"])
        * len(cfg["models"])
        * len(cfg["seeds"])
        * len(cfg["arms"])
        * (len(cfg["representation_pairs"]["controlled"]["kappas"]) + 1)
    )
    if len(runs) != expected:
        raise AssertionError(f"Expected {expected} cells, found {len(runs)}")
    if frame[["val_drift_step0", "val_drift_step1", "val_drift_step200"]].isna().any().any():
        raise AssertionError("Missing primary trajectory checkpoint")

    matched = frame[(frame["pair_family"] == "controlled") & (frame["arm"] == "matched_adamw")]
    step1_increase = matched["val_drift_step1"] - matched["val_drift_step0"]
    condition_result = _leave_one_dataset_out(matched, "condition")
    step1_result = _leave_one_dataset_out(matched, "step1_drift")
    auc_result = _leave_one_dataset_out(matched, "early_auc")

    endpoint = frame[(frame["pair_family"] == "controlled") & (frame["target_kappa"] == 3000.0)]
    paired = endpoint.pivot_table(
        index=["dataset", "model", "seed"], columns="arm", values="final_normalized_harm", aggfunc="first"
    ).reset_index()
    ordinary_mean = float(paired["ordinary_adamw"].mean())
    matched_mean = float(paired["matched_adamw"].mean())
    reduction = (ordinary_mean - matched_mean) / max(abs(ordinary_mean), 1e-12)

    closure = frame[frame["arm"] == "matched_input_natural"]
    controlled_closure = closure[closure["pair_family"] == "controlled"]
    natural_closure = closure[closure["pair_family"] == "natural"]
    ordinary_natural = frame[frame["arm"] == "ordinary_input_natural"]
    transfer = _natural_transfer(frame)
    samples = int(cfg["analysis_gates"]["bootstrap_samples"])
    result = {
        "coverage": {
            "expected_cells": expected,
            "observed_cells": int(len(runs)),
            "successful_cells": int(len(successful)),
            "failures": int(failures.sum()),
            "trajectory_rows": int(len(trajectories)),
            "basis_relation_max_error": float(successful["basis_relation_max_error"].max()),
        },
        "H1_matched_adamw_one_step_drift": {
            "mean_step0": float(matched["val_drift_step0"].mean()),
            "mean_step1": float(matched["val_drift_step1"].mean()),
            "mean_increase": float(step1_increase.mean()),
            "increase_cluster_bootstrap_95ci": _bootstrap_mean(
                step1_increase.to_numpy(), matched["dataset"].to_numpy(), samples
            ),
            "positive_fraction": float((step1_increase > 0).mean()),
        },
        "H2_leave_one_dataset_out": {
            "condition_number": {key: value for key, value in condition_result.items() if key != "predictions"},
            "step1_orbit_drift": {key: value for key, value in step1_result.items() if key != "predictions"},
            "early_drift_auc": {key: value for key, value in auc_result.items() if key != "predictions"},
            "step1_r2_improvement": float(step1_result["r2"] - condition_result["r2"]),
            "early_auc_r2_improvement": float(auc_result["r2"] - condition_result["r2"]),
            "preregistered_improvement_gate": float(
                cfg["analysis_gates"]["minimum_leave_one_dataset_out_r2_improvement_over_log_kappa"]
            ),
        },
        "H3_initialization_contribution_at_kappa3000": {
            "ordinary_mean_harm": ordinary_mean,
            "matched_mean_harm": matched_mean,
            "function_matching_reduction_fraction": float(reduction),
            "paired_mean_difference": float((paired["ordinary_adamw"] - paired["matched_adamw"]).mean()),
            "paired_cluster_bootstrap_95ci": _bootstrap_mean(
                (paired["ordinary_adamw"] - paired["matched_adamw"]).to_numpy(),
                paired["dataset"].to_numpy(),
                samples,
            ),
            "preregistered_reduction_gate": float(
                cfg["analysis_gates"]["minimum_function_matching_reduction_fraction"]
            ),
        },
        "H4_matched_input_natural_closure": {
            "mean_step0_drift": float(closure["val_drift_step0"].mean()),
            "mean_step200_drift": float(closure["val_drift_step200"].mean()),
            "maximum_step200_drift": float(closure["val_drift_step200"].max()),
            "mean_final_harm": float(closure["final_normalized_harm"].mean()),
            "controlled": {
                "mean_step200_drift": float(controlled_closure["val_drift_step200"].mean()),
                "maximum_step200_drift": float(controlled_closure["val_drift_step200"].max()),
                "mean_final_harm": float(controlled_closure["final_normalized_harm"].mean()),
            },
            "natural": {
                "mean_step200_drift": float(natural_closure["val_drift_step200"].mean()),
                "maximum_step200_drift": float(natural_closure["val_drift_step200"].max()),
                "mean_final_harm": float(natural_closure["final_normalized_harm"].mean()),
                "by_dataset": {
                    dataset: {
                        "mean_step200_drift": float(group["val_drift_step200"].mean()),
                        "mean_final_harm": float(group["final_normalized_harm"].mean()),
                        "reference_rank": int(group["reference_rank"].iloc[0]),
                    }
                    for dataset, group in natural_closure.groupby("dataset")
                },
            },
            "ordinary_init_natural_mean_step200_drift": float(ordinary_natural["val_drift_step200"].mean()),
            "ordinary_init_natural_mean_final_harm": float(ordinary_natural["final_normalized_harm"].mean()),
        },
        "H5_natural_transfer": {
            **transfer,
            "preregistered_direction_gate": float(
                cfg["analysis_gates"]["minimum_natural_direction_agreement_fraction"]
            ),
        },
    }
    (RESULTS / "analysis_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    frame.to_csv(RESULTS / "trajectory_cells_with_features.csv", index=False)
    predictions = matched[identifiers + ["final_normalized_harm"]].copy()
    predictions["condition_prediction"] = condition_result["predictions"]
    predictions["step1_drift_prediction"] = step1_result["predictions"]
    predictions["early_auc_prediction"] = auc_result["predictions"]
    predictions.to_csv(RESULTS / "leave_one_dataset_out_predictions.csv", index=False)
    _plots(frame)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
