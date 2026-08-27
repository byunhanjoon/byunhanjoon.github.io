#!/usr/bin/env python3
"""Consolidate regression and classification Frozen-Anchor TriChart results."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
REGRESSION_FILES = (
    "trichart_frozen_anchor.csv",
    "trichart_frozen_anchor_maps.csv",
    "trichart_frozen_anchor_20260828.csv",
    "trichart_frozen_anchor_maps_20260828.csv",
    "trichart_frozen_anchor_20260829.csv",
    "trichart_frozen_anchor_maps_20260829.csv",
)
CLASSIFICATION_FILES = (
    "trichart_frozen_anchor_classification_parity.csv",
    "trichart_frozen_anchor_classification_parity_20260828.csv",
    "trichart_frozen_anchor_classification_parity_20260829.csv",
)


def load(names: tuple[str, ...]) -> pd.DataFrame:
    return pd.concat(
        [pd.read_csv(RESULTS / name) for name in names], ignore_index=True
    )


def regression_panel() -> pd.DataFrame:
    frame = load(REGRESSION_FILES)
    for split in ("val", "test"):
        frame[f"{split}_gain_pct"] = 100.0 * (
            frame[f"{split}_anchor_rmse"] - frame[f"{split}_rmse"]
        ) / frame[f"{split}_anchor_rmse"]
    frame["validation_safe"] = frame.val_gain_pct >= -1e-10
    frame["strict_validation_win"] = frame.val_gain_pct > 1e-10
    frame["epoch_zero_fallback"] = frame.residual_best_epoch == 0
    return frame.sort_values(["dataset", "seed", "model"]).reset_index(drop=True)


def classification_panel() -> pd.DataFrame:
    frame = load(CLASSIFICATION_FILES)
    for split in ("val", "test"):
        frame[f"{split}_gain_log_loss"] = (
            frame[f"{split}_anchor_log_loss"] - frame[f"{split}_log_loss"]
        )
        frame[f"{split}_gain_pct"] = 100.0 * frame[
            f"{split}_gain_log_loss"
        ] / frame[f"{split}_anchor_log_loss"]
    frame["validation_safe"] = frame.val_gain_log_loss >= -1e-9
    frame["substantive_validation_win"] = frame.val_gain_log_loss > 1e-7
    return frame.sort_values(["dataset", "seed", "model"]).reset_index(drop=True)


def summarize(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    win_column = (
        "strict_validation_win"
        if task == "regression"
        else "substantive_validation_win"
    )
    return frame.groupby("dataset").agg(
        cells=("seed", "size"),
        validation_safe=("validation_safe", "sum"),
        validation_wins=(win_column, "sum"),
        mean_validation_gain_pct=("val_gain_pct", "mean"),
        test_wins=("test_gain_pct", lambda values: int((values > 0).sum())),
        mean_test_gain_pct=("test_gain_pct", "mean"),
    ).reset_index()


def adult_exact_comparison(classification: pd.DataFrame) -> pd.DataFrame:
    identity = pd.read_csv(RESULTS / "adult_identity_mechanism.csv")
    atom = identity.query(
        "method in ['tple_exact_supervised_additive', "
        "'tple_exact_supervised_separate']"
    ).copy()
    best_index = atom.groupby("model").val_log_loss.idxmin()
    best_atom = atom.loc[
        best_index,
        ["model", "method", "val_log_loss", "test_log_loss", "val_auc", "test_auc"],
    ].rename(
        columns={
            "method": "selected_exact_support_method",
            "val_log_loss": "exact_support_val_log_loss",
            "test_log_loss": "exact_support_test_log_loss",
            "val_auc": "exact_support_val_auc",
            "test_auc": "exact_support_test_auc",
        }
    )
    candidate = classification.query("dataset == 'adult' and seed == 20260827")[[
        "model", "val_log_loss", "test_log_loss", "val_auc", "test_auc"
    ]].rename(
        columns={
            "val_log_loss": "trichart_val_log_loss",
            "test_log_loss": "trichart_test_log_loss",
            "val_auc": "trichart_val_auc",
            "test_auc": "trichart_test_auc",
        }
    )
    comparison = candidate.merge(best_atom, on="model", validate="one_to_one")
    comparison["trichart_minus_exact_val_log_loss"] = (
        comparison.trichart_val_log_loss - comparison.exact_support_val_log_loss
    )
    comparison["trichart_minus_exact_test_log_loss"] = (
        comparison.trichart_test_log_loss - comparison.exact_support_test_log_loss
    )
    return comparison.sort_values("model").reset_index(drop=True)


def main() -> None:
    regression = regression_panel()
    classification = classification_panel()
    regression_summary = summarize(regression, "regression")
    classification_summary = summarize(classification, "classification")
    adult = adult_exact_comparison(classification)

    regression_gate = bool(
        regression.validation_safe.all()
        and regression.strict_validation_win.sum() >= 20
        and regression.val_gain_pct.mean() > 0
        and (regression_summary.mean_test_gain_pct > 0).all()
    )
    classification_gate = bool(
        classification.validation_safe.all()
        and classification.substantive_validation_win.sum() >= 18
        and classification.val_gain_pct.mean() > 0
        and (classification_summary.mean_test_gain_pct > 0).all()
    )
    decision = {
        "method": "frozen T-PLE anchor plus zero-start Q/midrank chart residual",
        "regression": {
            "gate_passed": regression_gate,
            "cells": int(len(regression)),
            "validation_safe_cells": int(regression.validation_safe.sum()),
            "strict_validation_wins": int(regression.strict_validation_win.sum()),
            "epoch_zero_fallbacks": int(regression.epoch_zero_fallback.sum()),
            "mean_validation_gain_pct": float(regression.val_gain_pct.mean()),
            "descriptive_test_wins": int((regression.test_gain_pct > 0).sum()),
            "mean_descriptive_test_gain_pct": float(regression.test_gain_pct.mean()),
            "datasets_with_positive_mean_test_gain": int(
                (regression_summary.mean_test_gain_pct > 0).sum()
            ),
        },
        "classification": {
            "gate_passed": classification_gate,
            "cells": int(len(classification)),
            "validation_safe_cells": int(classification.validation_safe.sum()),
            "substantive_validation_wins": int(
                classification.substantive_validation_win.sum()
            ),
            "mean_validation_log_loss_gain": float(
                classification.val_gain_log_loss.mean()
            ),
            "mean_validation_gain_pct": float(classification.val_gain_pct.mean()),
            "test_wins": int((classification.test_gain_pct > 0).sum()),
            "mean_test_log_loss_gain": float(
                classification.test_gain_log_loss.mean()
            ),
            "mean_test_gain_pct": float(classification.test_gain_pct.mean()),
            "datasets_with_positive_mean_test_gain": int(
                (classification_summary.mean_test_gain_pct > 0).sum()
            ),
        },
        "adult_exact_support": {
            "trichart_beats_selected_exact_support_on_validation": int(
                (adult.trichart_minus_exact_val_log_loss < 0).sum()
            ),
            "architectures": int(len(adult)),
            "mean_trichart_minus_exact_validation_log_loss": float(
                adult.trichart_minus_exact_val_log_loss.mean()
            ),
            "conclusion": "The chart residual complements T-PLE on Adult but does not replace exact atom identity.",
        },
        "claim": (
            "The frozen-anchor chart residual is a validation-safe, architecture-agnostic complement to T-PLE across the tested regression and binary-classification panels. Adult exact support remains a stronger specialized atom mechanism."
        ),
    }

    regression.to_csv(
        RESULTS / "trichart_frozen_anchor_confirmation.csv", index=False
    )
    regression_summary.to_csv(
        RESULTS / "trichart_frozen_anchor_summary_by_dataset.csv", index=False
    )
    classification.to_csv(
        RESULTS / "trichart_frozen_anchor_classification_confirmation.csv",
        index=False,
    )
    classification_summary.to_csv(
        RESULTS / "trichart_frozen_anchor_classification_summary_by_dataset.csv",
        index=False,
    )
    adult.to_csv(
        RESULTS / "trichart_frozen_anchor_adult_exact_comparison.csv", index=False
    )
    (RESULTS / "trichart_frozen_anchor_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    print(regression_summary.to_string(index=False))
    print(classification_summary.to_string(index=False))
    print(adult.to_string(index=False))
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
