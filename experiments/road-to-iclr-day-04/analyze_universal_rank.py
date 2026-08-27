#!/usr/bin/env python3
"""Consolidate the frozen universal-rank, cycle, and atom-interval pilots."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
MODELS = ["mlp", "resnet", "ft_transformer"]


def load(*names: str) -> pd.DataFrame:
    return pd.concat([pd.read_csv(RESULTS / name) for name in names], ignore_index=True)


def gain(candidate: pd.Series, baseline: pd.Series) -> pd.Series:
    return 100.0 * (baseline - candidate) / baseline


def build_rank_panel() -> pd.DataFrame:
    rank = load(
        "universal_mass_identity_weather.csv",
        "universal_mass_identity_cooking.csv",
        "universal_rank_delivery.csv",
        "universal_rank_maps.csv",
    ).query("method == 'rank_only'")
    typed = load(
        "signal_gated_support_weather.csv",
        "signal_gated_support_cooking.csv",
        "universal_rank_delivery_typed.csv",
        "universal_rank_maps_typed.csv",
    ).query("method in ['qple', 'tple']")
    typed = typed.pivot(index=["dataset", "model", "seed"], columns="method", values=["val_rmse", "test_rmse"])
    typed.columns = [f"{metric}_{method}" for metric, method in typed.columns]
    typed = typed.reset_index()
    columns = ["dataset", "model", "seed", "val_rmse", "test_rmse", "parameters"]
    panel = rank[columns].rename(
        columns={"val_rmse": "val_rmse_rank", "test_rmse": "test_rmse_rank", "parameters": "rank_parameters"}
    ).merge(typed, on=["dataset", "model", "seed"], validate="one_to_one")
    for split in ("val", "test"):
        for baseline in ("qple", "tple"):
            panel[f"{split}_gain_vs_{baseline}_pct"] = gain(
                panel[f"{split}_rmse_rank"], panel[f"{split}_rmse_{baseline}"]
            )
    panel["val_beats_both"] = (
        (panel.val_gain_vs_qple_pct > 0) & (panel.val_gain_vs_tple_pct > 0)
    )
    panel["model"] = pd.Categorical(panel.model, MODELS, ordered=True)
    return panel.sort_values(["dataset", "model"]).reset_index(drop=True)


def build_delivery_multiseed() -> pd.DataFrame:
    rank = load(
        "universal_rank_delivery.csv",
        "universal_rank_delivery_seed28.csv",
        "universal_rank_delivery_seed29.csv",
    )
    typed = load(
        "universal_rank_delivery_typed.csv",
        "universal_rank_delivery_typed_seed28.csv",
        "universal_rank_delivery_typed_seed29.csv",
    )
    typed = typed.pivot(index=["dataset", "model", "seed"], columns="method", values=["val_rmse", "test_rmse"])
    typed.columns = [f"{metric}_{method}" for metric, method in typed.columns]
    panel = rank[["dataset", "model", "seed", "val_rmse", "test_rmse"]].rename(
        columns={"val_rmse": "val_rmse_rank", "test_rmse": "test_rmse_rank"}
    ).merge(typed.reset_index(), on=["dataset", "model", "seed"], validate="one_to_one")
    for split in ("val", "test"):
        for baseline in ("qple", "tple"):
            panel[f"{split}_gain_vs_{baseline}_pct"] = gain(
                panel[f"{split}_rmse_rank"], panel[f"{split}_rmse_{baseline}"]
            )
    panel["model"] = pd.Categorical(panel.model, MODELS, ordered=True)
    return panel.sort_values(["seed", "model"]).reset_index(drop=True)


def build_confirmation() -> pd.DataFrame:
    rank = load(
        "universal_mass_identity_weather.csv",
        "universal_mass_identity_cooking.csv",
        "universal_rank_development_20260828.csv",
        "universal_rank_development_20260829.csv",
        "universal_rank_delivery.csv",
        "universal_rank_delivery_seed28.csv",
        "universal_rank_delivery_seed29.csv",
        "universal_rank_maps.csv",
        "universal_rank_maps_20260828.csv",
        "universal_rank_maps_20260829.csv",
    ).query("method == 'rank_only'")
    typed = load(
        "signal_gated_support_weather.csv",
        "signal_gated_support_cooking.csv",
        "universal_rank_development_typed_20260828.csv",
        "universal_rank_development_typed_20260829.csv",
        "universal_rank_delivery_typed.csv",
        "universal_rank_delivery_typed_seed28.csv",
        "universal_rank_delivery_typed_seed29.csv",
        "universal_rank_maps_typed.csv",
        "universal_rank_maps_typed_20260828.csv",
        "universal_rank_maps_typed_20260829.csv",
    ).query("method in ['qple', 'tple']")
    typed = typed.pivot(
        index=["dataset", "model", "seed"],
        columns="method",
        values=["val_rmse", "test_rmse"],
    )
    typed.columns = [f"{metric}_{method}" for metric, method in typed.columns]
    panel = rank[["dataset", "model", "seed", "val_rmse", "test_rmse"]].rename(
        columns={"val_rmse": "val_rmse_rank", "test_rmse": "test_rmse_rank"}
    ).merge(typed.reset_index(), on=["dataset", "model", "seed"], validate="one_to_one")
    for split in ("val", "test"):
        for baseline in ("qple", "tple"):
            panel[f"{split}_gain_vs_{baseline}_pct"] = gain(
                panel[f"{split}_rmse_rank"], panel[f"{split}_rmse_{baseline}"]
            )
    panel["model"] = pd.Categorical(panel.model, MODELS, ordered=True)
    return panel.sort_values(["dataset", "seed", "model"]).reset_index(drop=True)


def build_interval_panel(rank_panel: pd.DataFrame) -> pd.DataFrame:
    interval = load(
        "universal_interval.csv",
        "universal_interval_delivery.csv",
        "universal_interval_maps.csv",
    )
    columns = ["dataset", "model", "seed", "val_rmse", "test_rmse"]
    panel = interval[columns].rename(
        columns={"val_rmse": "val_rmse_interval", "test_rmse": "test_rmse_interval"}
    ).merge(
        rank_panel.drop(columns=["model"])[
            ["dataset", "seed", "val_rmse_rank", "test_rmse_rank"]
        ].assign(model=rank_panel.model.astype(str)),
        on=["dataset", "model", "seed"],
        validate="one_to_one",
    )
    panel["val_gain_vs_midrank_pct"] = gain(panel.val_rmse_interval, panel.val_rmse_rank)
    panel["test_gain_vs_midrank_pct"] = gain(panel.test_rmse_interval, panel.test_rmse_rank)
    panel["development"] = panel.dataset.isin(["weather", "cooking-time"])
    panel["model"] = pd.Categorical(panel.model, MODELS, ordered=True)
    return panel.sort_values(["development", "dataset", "model"], ascending=[False, True, True]).reset_index(drop=True)


def build_cycle_panel(rank_panel: pd.DataFrame) -> pd.DataFrame:
    cycle = pd.read_csv(RESULTS / "universal_cycle.csv")
    wide = cycle.pivot(index=["dataset", "model", "seed"], columns="method", values="val_rmse").reset_index()
    rank = rank_panel.assign(model=rank_panel.model.astype(str))[
        ["dataset", "model", "seed", "val_rmse_rank"]
    ]
    panel = wide.merge(rank, on=["dataset", "model", "seed"], validate="one_to_one")
    panel["gain_vs_midrank_pct"] = gain(panel.rank_cycle, panel.val_rmse_rank)
    panel["gain_vs_control_pct"] = gain(panel.rank_cycle, panel.rank_cycle_control)
    panel["cell_gate"] = (panel.gain_vs_midrank_pct > 0) & (panel.gain_vs_control_pct > 0)
    panel["model"] = pd.Categorical(panel.model, MODELS, ordered=True)
    return panel.sort_values(["dataset", "model"]).reset_index(drop=True)


def main() -> None:
    rank = build_rank_panel()
    delivery = build_delivery_multiseed()
    confirmation = build_confirmation()
    interval = build_interval_panel(rank)
    cycle = build_cycle_panel(rank)
    rank.to_csv(RESULTS / "universal_rank_panel.csv", index=False)
    delivery.to_csv(RESULTS / "universal_rank_delivery_multiseed.csv", index=False)
    confirmation.to_csv(RESULTS / "universal_rank_confirmation.csv", index=False)
    interval.to_csv(RESULTS / "universal_interval_panel.csv", index=False)
    cycle.to_csv(RESULTS / "universal_cycle_cells.csv", index=False)

    development_interval = interval[interval.development]
    transfer_interval = interval[~interval.development]
    decision = {
        "universal_midrank": {
            "cells": int(len(rank)),
            "validation_wins_vs_qple": int((rank.val_gain_vs_qple_pct > 0).sum()),
            "validation_wins_vs_tple": int((rank.val_gain_vs_tple_pct > 0).sum()),
            "validation_wins_vs_both": int(rank.val_beats_both.sum()),
            "mean_validation_gain_vs_qple_pct": float(rank.val_gain_vs_qple_pct.mean()),
            "mean_validation_gain_vs_tple_pct": float(rank.val_gain_vs_tple_pct.mean()),
            "claim": "competitive type-agnostic default; not a universal replacement for supervised T-PLE",
        },
        "delivery_three_seed": {
            "cells": int(len(delivery)),
            "validation_wins_vs_qple": int((delivery.val_gain_vs_qple_pct > 0).sum()),
            "validation_wins_vs_tple": int((delivery.val_gain_vs_tple_pct > 0).sum()),
            "mean_validation_gain_vs_qple_pct": float(delivery.val_gain_vs_qple_pct.mean()),
            "mean_validation_gain_vs_tple_pct": float(delivery.val_gain_vs_tple_pct.mean()),
        },
        "universal_midrank_confirmation": {
            "cells": int(len(confirmation)),
            "validation_wins_vs_qple": int((confirmation.val_gain_vs_qple_pct > 0).sum()),
            "validation_wins_vs_tple": int((confirmation.val_gain_vs_tple_pct > 0).sum()),
            "mean_validation_gain_vs_qple_pct": float(confirmation.val_gain_vs_qple_pct.mean()),
            "mean_validation_gain_vs_tple_pct": float(confirmation.val_gain_vs_tple_pct.mean()),
            "datasets_positive_vs_both": int(
                sum(
                    (group.val_gain_vs_qple_pct.mean() > 0)
                    and (group.val_gain_vs_tple_pct.mean() > 0)
                    for _, group in confirmation.groupby("dataset", observed=True)
                )
            ),
            "confirmation_gate_passed": False,
            "claim": "useful complementary chart, not a general standalone replacement",
        },
        "interval_rank": {
            "development_gate_passed": bool(
                (development_interval.val_gain_vs_midrank_pct > 0).sum() >= 4
                and development_interval.val_gain_vs_midrank_pct.mean() > 0
            ),
            "development_wins": int((development_interval.val_gain_vs_midrank_pct > 0).sum()),
            "development_cells": int(len(development_interval)),
            "mean_development_gain_pct": float(development_interval.val_gain_vs_midrank_pct.mean()),
            "transfer_wins": int((transfer_interval.val_gain_vs_midrank_pct > 0).sum()),
            "transfer_cells": int(len(transfer_interval)),
            "mean_transfer_gain_pct": float(transfer_interval.val_gain_vs_midrank_pct.mean()),
            "transfer_passed": False,
        },
        "rank_cycle": {
            "gate_passed": bool(cycle.cell_gate.sum() >= 4 and cycle.gain_vs_midrank_pct.mean() > 0 and cycle.gain_vs_control_pct.mean() > 0),
            "passing_cells": int(cycle.cell_gate.sum()),
            "cells": int(len(cycle)),
            "mean_gain_vs_midrank_pct": float(cycle.gain_vs_midrank_pct.mean()),
            "mean_gain_vs_control_pct": float(cycle.gain_vs_control_pct.mean()),
        },
    }
    (RESULTS / "universal_rank_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
