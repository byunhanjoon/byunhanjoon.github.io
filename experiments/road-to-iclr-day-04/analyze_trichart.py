#!/usr/bin/env python3
"""Analyze the frozen TriChart development, transfer, and seed confirmation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
sys.path.insert(0, str(HERE))
from semantic_multiview_pilot import load_tabred  # noqa: E402


MODELS = ["mlp", "resnet", "ft_transformer"]
DATASET_MODELS = {
    "weather": MODELS,
    "cooking-time": MODELS,
    "delivery-eta": MODELS,
    "maps-routing": MODELS[:2],
}
SEEDS = [20260827, 20260828, 20260829]


def load(*names: str) -> pd.DataFrame:
    return pd.concat([pd.read_csv(RESULTS / name) for name in names], ignore_index=True)


def gain(candidate: pd.Series, baseline: pd.Series) -> pd.Series:
    return 100.0 * (baseline - candidate) / baseline


def typed_baselines() -> pd.DataFrame:
    frame = load(
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
    wide = frame.pivot(
        index=["dataset", "model", "seed"],
        columns="method",
        values=["val_rmse", "test_rmse"],
    )
    wide.columns = [
        f"{split.removesuffix('_rmse')}_{method}"
        for split, method in wide.columns
    ]
    return wide.reset_index()


def rank_baseline() -> pd.DataFrame:
    return load(
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
    ).query("method == 'rank_only'")[["dataset", "model", "seed", "val_rmse", "test_rmse"]].rename(
        columns={"val_rmse": "val_rank", "test_rmse": "test_rank"}
    )


def trichart_confirmation() -> pd.DataFrame:
    tri = load(
        "trichart_shared.csv",
        "trichart_shared_delivery.csv",
        "trichart_shared_maps.csv",
        "trichart_shared_20260828.csv",
        "trichart_shared_20260829.csv",
        "trichart_shared_maps_20260828.csv",
        "trichart_shared_maps_20260829.csv",
    ).query("method == 'shared_consistency'")
    panel = tri[["dataset", "model", "seed", "val_rmse", "test_rmse", "parameters", "val_disagreement", "test_disagreement"]].merge(
        typed_baselines(), on=["dataset", "model", "seed"], validate="one_to_one"
    ).merge(rank_baseline(), on=["dataset", "model", "seed"], validate="one_to_one")
    for split in ("val", "test"):
        for baseline in ("qple", "tple", "rank"):
            panel[f"{split}_gain_vs_{baseline}_pct"] = gain(
                panel[f"{split}_rmse"], panel[f"{split}_{baseline}"]
            )
        panel[f"{split}_best_typed"] = panel[[f"{split}_qple", f"{split}_tple"]].min(axis=1)
        panel[f"{split}_gain_vs_best_typed_pct"] = gain(
            panel[f"{split}_rmse"], panel[f"{split}_best_typed"]
        )
    panel["val_beats_both_typed"] = (
        (panel.val_gain_vs_qple_pct > 0) & (panel.val_gain_vs_tple_pct > 0)
    )
    panel["model"] = pd.Categorical(panel.model, MODELS, ordered=True)
    return panel.sort_values(["dataset", "seed", "model"]).reset_index(drop=True)


def development_ablation() -> pd.DataFrame:
    frame = pd.read_csv(RESULTS / "trichart_shared.csv")
    typed = typed_baselines().query("seed == 20260827")
    panel = frame.merge(typed, on=["dataset", "model", "seed"], validate="many_to_one")
    panel["val_gain_vs_qple_pct"] = gain(panel.val_rmse, panel.val_qple)
    panel["val_gain_vs_tple_pct"] = gain(panel.val_rmse, panel.val_tple)
    panel["val_beats_both_typed"] = (
        (panel.val_gain_vs_qple_pct > 0) & (panel.val_gain_vs_tple_pct > 0)
    )
    return panel


def prediction_paths(dataset: str, model: str, seed: int) -> tuple[Path, Path, Path]:
    if dataset in {"weather", "cooking-time"}:
        short = "weather" if dataset == "weather" else "cooking"
        rank_stem = f"universal_mass_identity_{short}" if seed == 20260827 else f"universal_rank_development_{seed}"
        typed_stem = f"signal_gated_support_{short}" if seed == 20260827 else f"universal_rank_development_typed_{seed}"
    elif dataset == "delivery-eta":
        suffix = "" if seed == 20260827 else ("_seed28" if seed == 20260828 else "_seed29")
        rank_stem = f"universal_rank_delivery{suffix}"
        typed_stem = f"universal_rank_delivery_typed{suffix}"
    else:
        suffix = "" if seed == 20260827 else f"_{seed}"
        rank_stem = f"universal_rank_maps{suffix}"
        typed_stem = f"universal_rank_maps_typed{suffix}"
    return (
        RESULTS / f"{rank_stem}_predictions" / f"{dataset}__{model}__rank_only.npz",
        RESULTS / f"{typed_stem}_predictions" / f"{dataset}__{model}__{seed}__qple.npz",
        RESULTS / f"{typed_stem}_predictions" / f"{dataset}__{model}__{seed}__tple.npz",
    )


def independent_ensemble() -> pd.DataFrame:
    rows = []
    for dataset, models in DATASET_MODELS.items():
        data = load_tabred(
            dataset,
            max_train_rows=50000,
            max_eval_rows=15000,
            sample_seed=20260827,
        )
        for model in models:
            for seed in SEEDS:
                archives = [np.load(path) for path in prediction_paths(dataset, model, seed)]
                for split, part in (("val", "validation"), ("test", "test")):
                    target = data.y["val" if split == "val" else "test"]
                    views = [archive[part] for archive in archives]
                    candidates = {
                        "rank": views[0],
                        "qple": views[1],
                        "tple": views[2],
                        "rank_q": 0.5 * (views[0] + views[1]),
                        "rank_t": 0.5 * (views[0] + views[2]),
                        "q_t": 0.5 * (views[1] + views[2]),
                        "all": (views[0] + views[1] + views[2]) / 3.0,
                    }
                    values = {
                        method: float(np.sqrt(np.mean((prediction - target) ** 2)) * data.y_scale)
                        for method, prediction in candidates.items()
                    }
                    rows.append({"dataset": dataset, "model": model, "seed": seed, "split": split, **values})
    return pd.DataFrame(rows)


def main() -> None:
    confirmation = trichart_confirmation()
    ablation = development_ablation()
    ensemble = independent_ensemble()
    summary = confirmation.groupby("dataset", observed=True).agg(
        cells=("seed", "size"),
        wins_vs_qple=("val_gain_vs_qple_pct", lambda values: int((values > 0).sum())),
        wins_vs_tple=("val_gain_vs_tple_pct", lambda values: int((values > 0).sum())),
        wins_vs_both=("val_beats_both_typed", "sum"),
        mean_gain_vs_qple_pct=("val_gain_vs_qple_pct", "mean"),
        mean_gain_vs_tple_pct=("val_gain_vs_tple_pct", "mean"),
        mean_gain_vs_rank_pct=("val_gain_vs_rank_pct", "mean"),
        descriptive_test_gain_vs_best_typed_pct=("test_gain_vs_best_typed_pct", "mean"),
    ).reset_index()
    overall_wins_q = int((confirmation.val_gain_vs_qple_pct > 0).sum())
    overall_wins_t = int((confirmation.val_gain_vs_tple_pct > 0).sum())
    gate = bool(
        overall_wins_q >= 25
        and overall_wins_t >= 25
        and confirmation.val_gain_vs_qple_pct.mean() > 0
        and confirmation.val_gain_vs_tple_pct.mean() > 0
        and (summary.mean_gain_vs_qple_pct > 0).all()
        and (summary.mean_gain_vs_tple_pct > 0).all()
    )
    decision = {
        "confirmation_gate_passed": gate,
        "cells": int(len(confirmation)),
        "validation_wins_vs_qple": overall_wins_q,
        "validation_wins_vs_tple": overall_wins_t,
        "validation_wins_vs_both": int(confirmation.val_beats_both_typed.sum()),
        "mean_validation_gain_vs_qple_pct": float(confirmation.val_gain_vs_qple_pct.mean()),
        "mean_validation_gain_vs_tple_pct": float(confirmation.val_gain_vs_tple_pct.mean()),
        "mean_validation_gain_vs_rank_pct": float(confirmation.val_gain_vs_rank_pct.mean()),
        "datasets_positive_vs_qple": int((summary.mean_gain_vs_qple_pct > 0).sum()),
        "datasets_positive_vs_tple": int((summary.mean_gain_vs_tple_pct > 0).sum()),
        "descriptive_test_wins_vs_best_typed": int((confirmation.test_gain_vs_best_typed_pct > 0).sum()),
        "descriptive_mean_test_gain_vs_best_typed_pct": float(confirmation.test_gain_vs_best_typed_pct.mean()),
        "development_alignment_ablation": {
            method: {
                "wins_vs_both": int(group.val_beats_both_typed.sum()),
                "mean_gain_vs_qple_pct": float(group.val_gain_vs_qple_pct.mean()),
                "mean_gain_vs_tple_pct": float(group.val_gain_vs_tple_pct.mean()),
            }
            for method, group in ablation.groupby("method")
        },
        "claim": "shared multi-chart consistency is a generalizable performance method on this four-dataset developmental panel" if gate else "promising multi-chart method, but the frozen broad confirmation gate did not pass",
    }
    confirmation.to_csv(RESULTS / "trichart_confirmation.csv", index=False)
    summary.to_csv(RESULTS / "trichart_summary_by_dataset.csv", index=False)
    ablation.to_csv(RESULTS / "trichart_development_ablation.csv", index=False)
    ensemble.to_csv(RESULTS / "trichart_independent_ensemble.csv", index=False)
    (RESULTS / "trichart_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    print(summary.to_string(index=False))
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
