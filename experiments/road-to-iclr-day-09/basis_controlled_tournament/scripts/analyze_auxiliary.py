#!/usr/bin/env python3
"""Aggregate equal-HPO, mechanism, natural-basis, and anchor ablations."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_stage2 import aggregate, summarize  # noqa: E402
from tournament.common import (  # noqa: E402
    development_specs,
    disagreement,
    load_blocks,
    load_protocol,
    read_prediction_bundle,
    write_json,
)


def analyze_hpo(processed: Path) -> dict:
    directory = processed / "equal_hpo"
    selected_files = sorted(directory.glob("*__selected.csv"))
    trial_files = sorted(directory.glob("*__trials.csv"))
    if not selected_files:
        return {"selected_files": 0, "trial_files": 0}
    trials = pd.concat([pd.read_csv(path) for path in trial_files], ignore_index=True)
    reference = trials[trials["is_reference"]].copy()
    # A multiplier is a development-wide method setting.  Normalize within
    # each dataset/model/seed/method so heterogeneous task scales are not
    # pooled, then choose by median and mean validation excess error.
    keys = ["dataset", "model", "seed", "method"]
    reference["unit_best"] = reference.groupby(keys)["validation_task_error"].transform("min")
    reference["relative_excess"] = (
        reference["validation_task_error"] - reference["unit_best"]
    ) / reference["unit_best"].abs().clip(lower=1e-12)
    reference["unit_rank"] = reference.groupby(keys)["validation_task_error"].rank(
        method="average", ascending=True
    )
    lr_selection = (
        reference.groupby(["model", "method", "multiplier"], as_index=False)
        .agg(
            median_validation_excess=("relative_excess", "median"),
            mean_validation_excess=("relative_excess", "mean"),
            median_validation_rank=("unit_rank", "median"),
            mean_validation_rank=("unit_rank", "mean"),
            units=("unit_rank", "size"),
        )
        .sort_values(
            ["model", "method", "median_validation_excess", "mean_validation_excess", "multiplier"]
        )
    )
    lr_selection["selected"] = ~lr_selection.duplicated(["model", "method"])
    lr_selection.to_csv(processed / "equal_hpo_lr_selection.csv", index=False)
    chosen_multiplier = {
        (str(row.model), str(row.method)): float(row.multiplier)
        for row in lr_selection[lr_selection["selected"]].itertuples()
    }

    # Materialize metrics for the single development-wide multiplier selected
    # above.  This is the actual deployable HPO control; the runner's
    # per-representation oracle selection remains available only as a raw
    # diagnostic and is never eligible for finalist selection.
    protocol = load_protocol()
    spec_by_key = {spec["key"]: spec for spec in development_specs(protocol)}
    targets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    global_rows = []
    for (dataset, problem_type, model, seed, method), frame in trials.groupby(
        ["dataset", "problem_type", "model", "seed", "method"], sort=True
    ):
        multiplier = chosen_multiplier[(str(model), str(method))]
        chosen = frame[np.isclose(frame["multiplier"], multiplier)].copy()
        if len(chosen) != 9 or int(chosen["is_reference"].sum()) != 1:
            raise RuntimeError(f"incomplete global HPO selection for {dataset}/{model}/{seed}/{method}")
        if dataset not in targets:
            blocks = load_blocks(spec_by_key[str(dataset)], protocol)
            targets[str(dataset)] = (blocks.dataset.y_validation, blocks.dataset.y_test)
        reference_row = chosen[chosen["is_reference"]].iloc[0]
        reference_path = (
            ROOT
            / "results"
            / "raw"
            / "equal_hpo"
            / str(model)
            / str(dataset)
            / f"seed_{int(seed)}"
            / str(method)
            / f"lr_multiplier_{multiplier:g}"
            / f"{reference_row['representation_id']}.npz"
        )
        reference_predictions = read_prediction_bundle(reference_path)
        for row in chosen.itertuples():
            path = (
                ROOT
                / "results"
                / "raw"
                / "equal_hpo"
                / str(model)
                / str(dataset)
                / f"seed_{int(seed)}"
                / str(method)
                / f"lr_multiplier_{multiplier:g}"
                / f"{row.representation_id}.npz"
            )
            predictions = read_prediction_bundle(path)
            for split_index, split in enumerate(("validation", "test")):
                target = targets[str(dataset)][split_index]
                global_rows.append(
                    {
                        "dataset": dataset,
                        "problem_type": problem_type,
                        "model": model,
                        "seed": int(seed),
                        "method": f"{method}[equal-HPO]",
                        "track": "optimizer_equal_hpo",
                        "representation_id": row.representation_id,
                        "member": int(row.member),
                        "is_reference": bool(row.is_reference),
                        "split": split,
                        "selected_lr_multiplier": multiplier,
                        "task_error": float(getattr(row, f"{split}_task_error")),
                        "disagreement": disagreement(
                            str(problem_type), target, reference_predictions[split_index], predictions[split_index]
                        ),
                        "fit_seconds": float(row.fit_seconds),
                    }
                )
    global_frame = pd.DataFrame(global_rows)
    global_path = processed / "equal_hpo_global_selected_rows.csv"
    global_frame.to_csv(global_path, index=False)
    units = aggregate([global_path], "AdamW[equal-HPO]")
    units["predictive_rank"] = units.groupby(
        ["dataset", "model", "seed", "split"], sort=False
    )["task_error"].rank(method="average", ascending=True)
    summary = summarize(units)
    units.to_csv(processed / "equal_hpo_units.csv", index=False)
    summary.to_csv(processed / "equal_hpo_summary.csv", index=False)
    return {
        "selected_files": len(selected_files),
        "trial_files": len(trial_files),
        "selected_learning_rates": lr_selection[lr_selection["selected"]].to_dict(orient="records"),
    }


def analyze_mechanism(processed: Path) -> dict:
    files = sorted((processed / "mechanism").glob("*.csv"))
    if not files:
        return {"files": 0}
    rows = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    rows.to_csv(processed / "mechanism_audit.csv", index=False)
    summary = (
        rows.groupby(["method", "epoch"], as_index=False)
        .agg(
            median_disagreement=("disagreement", "median"),
            max_disagreement=("disagreement", "max"),
            median_max_prediction_difference=("max_prediction_absolute_difference", "median"),
            units=("disagreement", "size"),
        )
    )
    final_epoch = int(rows["epoch"].max())
    verdict = summary[summary["epoch"].isin([0, final_epoch])].copy()
    verdict["preserves_matched_equivalence"] = verdict["max_disagreement"] < 1e-5
    summary.to_csv(processed / "mechanism_audit_summary.csv", index=False)
    verdict.to_csv(processed / "mechanism_equivariance_verdict.csv", index=False)
    return {"files": len(files), "final_epoch": final_epoch, "verdict": verdict.to_dict(orient="records")}


def analyze_natural(processed: Path) -> dict:
    files = sorted((processed / "natural_bases").glob("*.csv"))
    if not files:
        return {"files": 0}
    rows = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    rows.to_csv(processed / "natural_basis_units.csv", index=False)
    method_rows = rows[(rows["method"] != "Raw") & (rows["split"] == "test")]
    summary = (
        method_rows.groupby(["method", "model", "pair"], as_index=False)
        .agg(
            median_disagreement_reduction=("disagreement_reduction", "median"),
            median_relative_task_change=("relative_task_change", "median"),
            max_coordinate_error=("coordinate_error", "max"),
            max_reconstruction_error=("reconstruction_error", "max"),
            datasets=("dataset", "nunique"),
            units=("dataset", "size"),
        )
    )
    summary.to_csv(processed / "natural_basis_summary.csv", index=False)
    return {"files": len(files), "summary": summary.to_dict(orient="records")}


def analyze_anchor(processed: Path) -> dict:
    files = sorted((processed / "anchor_ablation").glob("*.csv"))
    if not files:
        return {"files": 0}
    rows = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    development_path = processed / "development_units.csv"
    if not development_path.exists():
        return {"files": len(files), "waiting_for_development_units": True}
    development = pd.read_csv(development_path)
    raw = development[
        (development["model"] == "controlled_mlp")
        & (development["method"] == "Raw")
        & (development["split"] == "test")
    ][["dataset", "seed", "task_error", "disagreement"]].rename(
        columns={"task_error": "raw_task_error", "disagreement": "raw_disagreement"}
    )
    rows = rows.merge(raw, on=["dataset", "seed"], how="left", validate="many_to_one")
    rows["disagreement"] = 0.0
    rows["disagreement_reduction"] = np.where(rows["raw_disagreement"] > 1e-12, 1.0, 0.0)
    rows["relative_task_change"] = (
        rows["test_task_error"] - rows["raw_task_error"]
    ) / rows["raw_task_error"].abs().clip(lower=1e-12)
    rows.to_csv(processed / "anchor_ablation_units.csv", index=False)
    summary = (
        rows.groupby(["variant", "anchors", "selection", "normalize"], as_index=False)
        .agg(
            median_empirical_rank=("median_empirical_rank", "median"),
            min_empirical_rank=("min_empirical_rank", "min"),
            min_anchor_rank=("min_anchor_rank", "min"),
            median_disagreement_reduction=("disagreement_reduction", "median"),
            median_relative_task_change=("relative_task_change", "median"),
            max_coordinate_error=("max_coordinate_error", "max"),
            units=("dataset", "size"),
        )
        .sort_values(["median_relative_task_change", "anchors"])
    )
    summary.to_csv(processed / "anchor_ablation_summary.csv", index=False)
    return {"files": len(files), "summary": summary.to_dict(orient="records")}


def combine_development_rankings(processed: Path) -> dict:
    base_path = processed / "development_units.csv"
    hpo_path = processed / "equal_hpo_units.csv"
    if not base_path.exists() or not hpo_path.exists():
        return {"complete": False}
    base = pd.read_csv(base_path)
    hpo = pd.read_csv(hpo_path)
    units = pd.concat([base, hpo], ignore_index=True)
    units["predictive_rank"] = units.groupby(
        ["dataset", "model", "seed", "split"], sort=False
    )["task_error"].rank(method="average", ascending=True)
    units.to_csv(processed / "development_all_units.csv", index=False)
    summary = summarize(units)
    summary.to_csv(processed / "development_all_method_summary.csv", index=False)
    ranking_a = summary[summary["performance_preserving_eligible"]].sort_values(
        ["median_disagreement_reduction", "median_worst_orbit_gain"], ascending=[False, False]
    )
    ranking_b = summary[summary["pareto_frontier"]].sort_values("median_relative_task_change")
    ranking_c = summary.sort_values(
        ["median_predictive_rank", "mean_predictive_rank", "median_relative_task_change"]
    )
    ranking_d = summary.sort_values("paper_method_score", ascending=False)
    ranking_a.to_csv(processed / "development_all_ranking_A.csv", index=False)
    ranking_b.to_csv(processed / "development_all_ranking_B_pareto.csv", index=False)
    ranking_c.to_csv(processed / "development_all_ranking_C_predictive.csv", index=False)
    ranking_d.to_csv(processed / "development_all_ranking_D_score.csv", index=False)
    return {
        "complete": True,
        "ranking_A": ranking_a.to_dict(orient="records"),
        "ranking_B": ranking_b.to_dict(orient="records"),
        "ranking_C": ranking_c.to_dict(orient="records"),
        "ranking_D": ranking_d.to_dict(orient="records"),
    }


def main() -> None:
    processed = ROOT / "results" / "processed"
    equal_hpo = analyze_hpo(processed)
    result = {
        "equal_hpo": equal_hpo,
        "mechanism": analyze_mechanism(processed),
        "natural_basis": analyze_natural(processed),
        "anchor_ablation": analyze_anchor(processed),
        "combined_development": combine_development_rankings(processed),
    }
    write_json(processed / "auxiliary_analysis.json", result)
    print(result)


if __name__ == "__main__":
    main()
