"""Analyze the frozen four-source late OpenML extension."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_disjoint_pair32 as DP32
import analyze_disjoint_pack64 as DP64
import analyze_disjoint_pair_cross as DPC
from analyze_strength2_cover import proper_loss


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CONFIG = HERE / "openml_late_source_cover_config.json"
TENSORS = RESULTS / "openml_late_source_cover"
NUMERICAL_STRICT_TOLERANCE = 0.0


def summarize_comparison(
    calibration: pd.DataFrame,
    cells: pd.DataFrame,
    action: str,
    control: str,
) -> dict[str, object]:
    pivot = calibration[calibration.method.isin((action, control))].pivot(
        index=["dataset", "model"], columns="method", values="score_rmse"
    )
    source_rmse = calibration[calibration.method.isin((action, control))].groupby(
        ["dataset", "method"]
    ).score_rmse.mean().unstack()
    selection = cells[cells.method.isin((action, control))].groupby("method").mean(numeric_only=True)
    return {
        "action": action,
        "control": control,
        "candidate_rmse_wins": int((pivot[action] < pivot[control] - NUMERICAL_STRICT_TOLERANCE).sum()),
        "candidates": int(len(pivot)),
        "source_mean_rmse_wins": int((source_rmse[action] < source_rmse[control] - NUMERICAL_STRICT_TOLERANCE).sum()),
        "sources": int(len(source_rmse)),
        "mean_score_rmse": {
            action: float(calibration[calibration.method == action].score_rmse.mean()),
            control: float(calibration[calibration.method == control].score_rmse.mean()),
        },
        "mean_selection_agreement": {
            action: float(selection.loc[action, "selection_agreement"]),
            control: float(selection.loc[control, "selection_agreement"]),
        },
        "mean_validation_regret": {
            action: float(selection.loc[action, "validation_quotient_regret"]),
            control: float(selection.loc[control, "validation_quotient_regret"]),
        },
        "mean_selected_quotient_test_loss": {
            action: float(selection.loc[action, "selected_quotient_test_loss"]),
            control: float(selection.loc[control, "selected_quotient_test_loss"]),
        },
    }


def exact_transfer(config: dict, tensors: Path) -> list[dict[str, object]]:
    output = []
    for dataset in config["datasets"]:
        val, test = [], []
        for model in config["models"]:
            archive = np.load(tensors / f"{dataset}__{model}.npz")
            val.append(proper_loss(archive["validation_y"], archive["validation_predictions"].reshape(
                (-1,) + archive["validation_predictions"].shape[-2:]
            ).mean(axis=0)))
            test.append(proper_loss(archive["test_y"], archive["test_predictions"].reshape(
                (-1,) + archive["test_predictions"].shape[-2:]
            ).mean(axis=0)))
        val, test = np.asarray(val), np.asarray(test)
        val_winner, test_winner = int(np.argmin(val)), int(np.argmin(test))
        output.append({
            "dataset": dataset,
            "validation_winner": config["models"][val_winner],
            "test_winner": config["models"][test_winner],
            "winner_agreement": bool(val_winner == test_winner),
            "test_regret_of_validation_winner": float(test[val_winner] - test[test_winner]),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--tensors", type=Path, default=TENSORS)
    parser.add_argument("--strength-prefix", default="late_source_strength2")
    parser.add_argument("--output-prefix", default="late_source")
    parser.add_argument("--minimum-cell-wins", type=int, default=10)
    parser.add_argument("--required-source-wins", type=int, default=4)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    strength = pd.read_csv(RESULTS / f"{args.strength_prefix}_cells.csv")
    test = strength[strength.split == "test"].copy()
    test["beats_iid_and_strength1"] = (
        (test.iid16_residual - test.strength2_residual > NUMERICAL_STRICT_TOLERANCE)
        & (test.four_strength1_residual - test.strength2_residual > NUMERICAL_STRICT_TOLERANCE)
    )
    source = test.groupby("dataset")[[
        "strength2_residual", "iid16_residual", "four_strength1_residual", "four_seed_blocks_residual"
    ]].mean()
    source_wins = (
        (source.strength2_residual < source.iid16_residual)
        & (source.strength2_residual < source.four_strength1_residual)
    )

    all_calibration, all_rows = [], []
    analyzers = {
        "pair32": DP32.analyze_dataset,
        "pack64": DP64.analyze_dataset,
        "pair_cross64": DPC.analyze_dataset,
    }
    for family, analyzer in analyzers.items():
        rows, calibration = [], []
        for dataset in config["datasets"]:
            current = analyzer(args.output_prefix, dataset, config["models"], args.tensors)
            current_rows, current_calibration = current[:2]
            rows.extend(current_rows); calibration.extend(current_calibration)
        row_frame, cal_frame = pd.DataFrame(rows), pd.DataFrame(calibration)
        row_frame["family"] = family; cal_frame["family"] = family
        all_rows.append(row_frame); all_calibration.append(cal_frame)
    rows = pd.concat(all_rows, ignore_index=True)
    calibration = pd.concat(all_calibration, ignore_index=True)
    cells = rows.groupby(["family", "dataset", "method"], as_index=False).mean(numeric_only=True)
    rows.to_csv(RESULTS / f"{args.output_prefix}_packing_draws.csv", index=False)
    cells.to_csv(RESULTS / f"{args.output_prefix}_packing_cells.csv", index=False)
    calibration.to_csv(RESULTS / f"{args.output_prefix}_packing_calibration.csv", index=False)

    comparisons = {
        "pair32": ("disjoint_pair_mean32", "independent_pair_mean32"),
        "pack64": ("mutually_disjoint_pack64", "two_disjoint_pairs64"),
        "unbiased_pair_cross64": ("disjoint_pair_cross64", "independent_block_u64"),
    }
    comparison_summary = {}
    for name, (action, control) in comparisons.items():
        family = "pair_cross64" if name == "unbiased_pair_cross64" else name
        comparison_summary[name] = summarize_comparison(
            calibration[calibration.family == family], cells[cells.family == family], action, control
        )

    transfer = exact_transfer(config, args.tensors)
    material = test[test.material]
    material_all = (
        (material.strength2_residual < material.iid16_residual)
        & (material.strength2_residual < material.four_strength1_residual)
        & (material.strength2_residual < material.four_seed_blocks_residual)
    )
    all_cell_wins = int(test.beats_iid_and_strength1.sum())
    all_source_wins = int(source_wins.sum())
    primary_pass = (
        all_cell_wins >= args.minimum_cell_wins
        and all_source_wins >= args.required_source_wins
    )
    summary = {
        "status": "complete",
        "configured_sources": len(config["datasets"]),
        "complete_tensors": int(len(list(args.tensors.glob("*.npz")))),
        "represented_complete_product_fits": int(len(config["datasets"]) * len(config["models"]) * 128),
        "post_failure_schema_adapter_sources": len(config.get("post_failure_schema_adapter", [])),
        "primary_strength2_gate": {
            "numerical_strict_tolerance": NUMERICAL_STRICT_TOLERANCE,
            "all_cell_wins_vs_iid_and_strength1": all_cell_wins,
            "required_all_cell_wins": args.minimum_cell_wins,
            "all_cells": int(len(test)),
            "material_cell_wins_vs_all_controls": int(material_all.sum()),
            "material_cells": int(len(material)),
            "source_mean_wins_vs_iid_and_strength1": all_source_wins,
            "required_source_mean_wins": args.required_source_wins,
            "sources": int(len(source)),
            "passed": bool(primary_pass),
        },
        "packing_and_selection": comparison_summary,
        "exact_validation_to_test_transfer": {
            "sources": transfer,
            "winner_agreements": int(sum(row["winner_agreement"] for row in transfer)),
            "mean_test_regret_of_validation_winner": float(np.mean([
                row["test_regret_of_validation_winner"] for row in transfer
            ])),
        },
        "interpretation": "late_source_primary_pass" if primary_pass else "late_source_primary_fail",
    }
    (RESULTS / f"{args.output_prefix}_extension_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
