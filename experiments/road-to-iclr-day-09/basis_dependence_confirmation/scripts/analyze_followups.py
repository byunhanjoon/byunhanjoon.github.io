#!/usr/bin/env python3
"""Aggregate natural bases, mechanism, HPO, repairs, and prospective results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
PROCESSED = ROOT / "results" / "processed"


def read_metric_files(pattern: str, filename: str = "metrics.csv") -> pd.DataFrame:
    paths = sorted(ROOT.glob(pattern))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(path) for path in paths if path.name == filename], ignore_index=True)


def add_common(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ("probability_rmse", "prediction_rmse_normalized", "log_loss", "rmse"):
        if column not in result:
            result[column] = np.nan
    result["prediction_disagreement"] = np.where(
        result["problem_type"].eq("classification"), result["probability_rmse"],
        result["prediction_rmse_normalized"],
    )
    result["task_error"] = np.where(
        result["problem_type"].eq("classification"), result["log_loss"], result["rmse"]
    )
    return result


def summarize_natural(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    test = add_common(frame)
    test = test[(test["split"] == "test") & (~test["is_reference"])]
    return test.groupby(
        ["dataset", "problem_type", "model", "family", "basis_pair"], as_index=False
    ).agg(
        disagreement=("prediction_disagreement", "mean"),
        disagreement_seed_sd=("prediction_disagreement", "std"),
        performance=("task_error", "mean"),
        reconstruction_error=("reconstruction_error", "max"),
        condition_number=("condition_number", "max"), seeds=("model_seed", "nunique"),
        members=("member", "nunique"),
    )


def repair_by_seed(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    test = add_common(frame)
    test = test[test["split"] == "test"]
    orbit = test[~test["is_reference"]].groupby(
        ["dataset", "problem_type", "model", "model_seed", "repair"], as_index=False
    ).agg(disagreement=("prediction_disagreement", "mean"))
    reference = test[test["is_reference"]].groupby(
        ["dataset", "problem_type", "model", "model_seed", "repair"], as_index=False
    ).agg(task_error=("task_error", "first"))
    result = orbit.merge(
        reference, on=["dataset", "problem_type", "model", "model_seed", "repair"], validate="one_to_one"
    )
    keys = ["dataset", "model", "model_seed"]
    raw = result[result["repair"] == "raw"][keys + ["disagreement", "task_error"]].rename(
        columns={"disagreement": "raw_disagreement", "task_error": "raw_task_error"}
    )
    result = result.merge(raw, on=keys, how="left", validate="many_to_one")
    result["disagreement_reduction"] = 1 - result["disagreement"] / result["raw_disagreement"].clip(lower=1e-12)
    result["relative_task_change"] = (result["task_error"] - result["raw_task_error"]) / result["raw_task_error"].abs().clip(lower=1e-12)
    return result


def summarize_repairs(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_seed = repair_by_seed(frame)
    if by_seed.empty:
        return by_seed, by_seed
    summary = by_seed.groupby(["dataset", "problem_type", "model", "repair"], as_index=False).agg(
        disagreement=("disagreement", "mean"), disagreement_seed_sd=("disagreement", "std"),
        raw_disagreement=("raw_disagreement", "mean"),
        disagreement_reduction=("disagreement_reduction", "mean"),
        task_error=("task_error", "mean"), raw_task_error=("raw_task_error", "mean"),
        relative_task_change=("relative_task_change", "mean"), seeds=("model_seed", "nunique"),
    )
    summary["useful_threshold"] = (
        summary["disagreement_reduction"].ge(0.70) & summary["relative_task_change"].le(0.01)
    )
    summary["average_rank"] = summary.groupby(["dataset", "model"])["disagreement"].rank(method="average")
    return by_seed, summary


def summarize_mechanism(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    values = add_common(frame)
    values = values[values["split"] == "test"].copy()
    values["epoch_order"] = values["epoch"].replace({"final": "100"}).astype(int)
    for column in ("reference_log_loss", "reference_rmse"):
        if column not in values:
            values[column] = np.nan
    values["reference_task_error"] = np.where(
        values["problem_type"].eq("classification"), values["reference_log_loss"], values["reference_rmse"]
    )
    return values.groupby(
        ["dataset", "problem_type", "condition", "function_matched", "optimizer", "momentum",
         "weight_decay", "epoch", "epoch_order"], as_index=False
    ).agg(
        disagreement=("prediction_disagreement", "mean"),
        disagreement_seed_sd=("prediction_disagreement", "std"),
        initial_max_logit_difference=("initial_max_logit_difference", "max"),
        reference_task=("reference_task_error", "mean"),
        seeds=("model_seed", "nunique"), members=("orbit_member", "nunique"),
    )


def summarize_hpo(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    values = add_common(frame)
    for column in ("reference_log_loss", "reference_rmse", "transformed_log_loss", "transformed_rmse"):
        if column not in values:
            values[column] = np.nan
    values["reference_task_error"] = np.where(
        values["problem_type"].eq("classification"), values["reference_log_loss"], values["reference_rmse"]
    )
    values["transformed_task_error"] = np.where(
        values["problem_type"].eq("classification"), values["transformed_log_loss"], values["transformed_rmse"]
    )
    return values[values["split"] == "test"].groupby(
        ["dataset", "problem_type", "model"], as_index=False
    ).agg(
        disagreement=("prediction_disagreement", "mean"),
        disagreement_seed_sd=("prediction_disagreement", "std"),
        reference_task=("reference_task_error", "mean"),
        transformed_task=("transformed_task_error", "mean"),
        seeds=("model_seed", "nunique"),
    )


def bootstrap_dataset_ci(
    frame: pd.DataFrame, group_columns: list[str], value_column: str, draws: int, seed: int = 20260901,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    rng = np.random.default_rng(seed)
    rows = []
    for keys, group in frame.groupby(group_columns):
        datasets = sorted(group["dataset"].unique())
        arrays = {dataset: group.loc[group["dataset"] == dataset, value_column].to_numpy() for dataset in datasets}
        estimates = []
        for _ in range(draws):
            sampled = rng.choice(datasets, len(datasets), replace=True)
            chunks = [arrays[dataset] for dataset in sampled]
            estimates.append(float(np.median(np.concatenate(chunks))))
        rows.append({
            **dict(zip(group_columns, keys if isinstance(keys, tuple) else (keys,))),
            "datasets": len(datasets), "median": float(group[value_column].median()),
            "ci_low": float(np.quantile(estimates, 0.025)), "ci_high": float(np.quantile(estimates, 0.975)),
            "bootstrap_draws": draws,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--skip-bootstrap", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "configs" / "development_protocol.yaml").read_text())
    draws = int(config["bootstrap_draws"])
    PROCESSED.mkdir(parents=True, exist_ok=True)

    natural_raw = read_metric_files("results/raw/development/natural/*/*/seed_*/metrics.csv")
    repairs_raw = read_metric_files("results/raw/development/repairs/*/*/seed_*/metrics.csv")
    consistency_raw = read_metric_files("results/raw/development/consistency/*/*/seed_*/metrics.csv")
    mechanism_raw = read_metric_files("results/raw/development/mechanism/*/seed_*/member_*/*/metrics.csv")
    hpo_raw = read_metric_files(
        "results/raw/development/equal_hpo/*/*/seed_*/selected_comparison.csv", "selected_comparison.csv"
    )
    prospective_raw = read_metric_files("results/raw/prospective/evaluation/*/*/seed_*/metrics.csv")

    natural = summarize_natural(natural_raw)
    combined_repairs = repairs_raw
    if not consistency_raw.empty:
        combined_repairs = pd.concat([combined_repairs, consistency_raw], ignore_index=True, sort=False)
    repairs_seed, repairs = summarize_repairs(combined_repairs)
    mechanism = summarize_mechanism(mechanism_raw)
    hpo = summarize_hpo(hpo_raw)
    prospective_seed, prospective = summarize_repairs(prospective_raw)

    outputs = {
        "natural_all_metrics.csv": natural_raw, "natural_summary.csv": natural,
        "repairs_by_seed.csv": repairs_seed, "repairs_summary.csv": repairs,
        "mechanism_all_metrics.csv": mechanism_raw, "mechanism_summary.csv": mechanism,
        "equal_hpo_selected_all.csv": hpo_raw, "equal_hpo_summary.csv": hpo,
        "prospective_repairs_by_seed.csv": prospective_seed,
        "prospective_repairs_summary.csv": prospective,
    }
    for filename, frame in outputs.items():
        if not frame.empty:
            frame.to_csv(PROCESSED / filename, index=False)

    ci_parts = []
    replication_path = PROCESSED / "replication_summary.csv"
    if replication_path.exists() and not args.skip_bootstrap:
        replication = pd.read_csv(replication_path)
        ci_parts.append(bootstrap_dataset_ci(replication, ["variant", "model"], "mean_disagreement", draws))
    if not repairs.empty and not args.skip_bootstrap:
        ci_parts.append(bootstrap_dataset_ci(repairs, ["repair"], "disagreement_reduction", draws))
    if not prospective.empty and not args.skip_bootstrap:
        prospective_ci = bootstrap_dataset_ci(prospective, ["repair"], "disagreement_reduction", draws)
        prospective_ci["variant"] = "prospective_repair_reduction"
        ci_parts.append(prospective_ci)
    if ci_parts:
        pd.concat(ci_parts, ignore_index=True, sort=False).to_csv(PROCESSED / "bootstrap_intervals.csv", index=False)

    manifest = {
        "allow_partial": args.allow_partial,
        "bundle_counts": {
            "natural": len(list((RAW / "development" / "natural").glob("*/*/seed_*/metadata.json"))),
            "repairs": len(list((RAW / "development" / "repairs").glob("*/*/seed_*/metadata.json"))),
            "consistency": len(list((RAW / "development" / "consistency").glob("*/*/seed_*/metadata.json"))),
            "mechanism": len(list((RAW / "development" / "mechanism").glob("*/seed_*/member_*/*/metadata.json"))),
            "equal_hpo": len(list((RAW / "development" / "equal_hpo").glob("*/*/seed_*/metadata.json"))),
            "prospective": len(list((RAW / "prospective" / "evaluation").glob("*/*/seed_*/metadata.json"))),
        },
    }
    expected = {
        "natural": 165, "repairs": 165, "consistency": 33,
        "mechanism": 240, "equal_hpo": 18, "prospective": 84,
    }
    manifest["expected_bundle_counts"] = expected
    manifest["coverage_complete"] = {
        stage: manifest["bundle_counts"][stage] == count for stage, count in expected.items()
    }
    (ROOT / "results" / "followup_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2))
    if not args.allow_partial and not all(manifest["coverage_complete"].values()):
        raise RuntimeError(f"follow-up coverage incomplete: {manifest['coverage_complete']}")


if __name__ == "__main__":
    main()
