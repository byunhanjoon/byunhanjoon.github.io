"""Analyze internal and external TabPFN nuisance averaging at equal compute."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_strength2_cover import strength1_family, strength2_family
from analyze_completion_panel import decompose_array


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
INPUT = RESULTS / "completion_tabpfn"
CONFIG = json.loads((HERE / "completion_config.json").read_text())
DRAWS = 2048


def proper_loss(y: np.ndarray, prediction: np.ndarray) -> float:
    target = np.eye(2)[y.astype(int)]
    return float(np.mean(np.sum((prediction - target) ** 2, axis=-1)))


def randomized_family(base_family: np.ndarray, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chosen = rng.integers(0, len(base_family), size=count)
    return base_family[chosen]


def ids(designs: np.ndarray, cards: tuple[int, ...]) -> np.ndarray:
    return np.ravel_multi_index(np.moveaxis(designs, -1, 0), cards)


def analyze_setting(predictions: np.ndarray, cards: tuple[int, ...], seed: int) -> dict[str, tuple[float, float]]:
    flat = predictions.reshape((-1,) + predictions.shape[-2:]).astype(np.float64)
    quotient = flat.mean(0)
    population = len(flat)
    rng = np.random.default_rng(seed)
    output = {}
    for method in ("iid16", "srswor16"):
        if method == "iid16":
            chosen = rng.integers(0, population, size=(DRAWS, 16))
            calls = 16
        elif population >= 16:
            chosen = np.stack([rng.choice(population, 16, replace=False) for _ in range(DRAWS)])
            calls = 16
        else:
            chosen = np.tile(np.arange(population), (DRAWS, 1))
            calls = population
        estimates = flat[chosen].mean(1)
        residual = np.mean(np.sum((estimates - quotient[None]) ** 2, axis=-1), axis=-1)
        output[method] = (float(residual.mean()), calls)
    family1 = strength1_family(cards[1], cards[2], 1)[..., :3]
    blocks = [randomized_family(family1, DRAWS, seed + 10 + block) for block in range(4)]
    design1 = np.concatenate(blocks, axis=1)
    family2 = strength2_family(cards[1], cards[2], 1)[..., :3]
    design2 = randomized_family(family2, DRAWS, seed + 20)
    for method, design in (("strength1_16", design1), ("strength2_16", design2)):
        estimates = flat[ids(design, cards)].mean(1)
        residual = np.mean(np.sum((estimates - quotient[None]) ** 2, axis=-1), axis=-1)
        output[method] = (float(residual.mean()), len(design[0]))
    return output


def main() -> None:
    rows = []; compute_rows = []
    for dataset in CONFIG["datasets"]:
        if CONFIG["dataset_tasks"][dataset] != "classification":
            continue
        for split_seed in CONFIG["split_seeds"]:
            stem = f"{dataset}__split{split_seed}"
            archive = np.load(INPUT / f"{stem}.npz")
            manifest = json.loads((INPUT / f"{stem}.json").read_text())
            actions = archive["actions"].astype(int)
            cards = tuple(int(actions[:, index].max() + 1) for index in range(3))
            order = np.argsort(np.ravel_multi_index(actions.T, cards))
            y = archive["test_y"]
            for estimators, policy in ((1, "none"), (1, "default"), (8, "default")):
                flat = archive[f"test__{estimators}__{policy}"][order].astype(np.float64)
                predictions = flat.reshape(cards + flat.shape[-2:])
                quotient = flat.mean(0)
                canonical = flat[0]
                components = decompose_array(predictions)
                total = sum(components.values())
                factor_totals = {
                    label: sum(value for subset, value in components.items() if factor in subset)
                    for factor, label in enumerate(("feature", "category", "class"))
                }
                methods = analyze_setting(predictions, cards, 2026082861 + split_seed + estimators)
                for method, (residual, calls) in methods.items():
                    rows.append({
                        "dataset": dataset, "split_seed": split_seed,
                        "internal_estimators": estimators, "internal_policy": policy,
                        "external_method": method, "schema_risk": total,
                        **{f"{label}_total_variance": value for label, value in factor_totals.items()},
                        "external_residual": residual, "external_calls": calls,
                        "forward_ensemble_members": calls * estimators,
                        "canonical_brier": proper_loss(y, canonical),
                        "quotient_brier": proper_loss(y, quotient),
                    })
            compute_rows.append({
                "dataset": dataset, "split_seed": split_seed,
                "tabpfn_calls": manifest["tabpfn_calls"],
                "forward_ensemble_members": manifest["forward_ensemble_members"],
                "wall_seconds": manifest["wall_seconds"],
            })
    frame = pd.DataFrame(rows); compute = pd.DataFrame(compute_rows)
    frame.to_csv(RESULTS / "completion_tabpfn_external_cells.csv", index=False)
    compute.to_csv(RESULTS / "completion_tabpfn_compute.csv", index=False)
    comparisons = {}
    for setting, current in frame.groupby(["internal_estimators", "internal_policy"]):
        means = current.groupby("external_method").external_residual.mean()
        comparisons[f"{setting[0]}:{setting[1]}"] = means.to_dict()
    base = frame[(frame.internal_estimators == 1) & (frame.internal_policy == "none")]
    built = frame[(frame.internal_estimators == 8) & (frame.internal_policy == "default")]
    base_risk = base.drop_duplicates(["dataset", "split_seed"]).schema_risk.mean()
    built_risk = built.drop_duplicates(["dataset", "split_seed"]).schema_risk.mean()
    summary = {
        "status": "complete", "datasets": frame.dataset.nunique(), "splits": frame.split_seed.nunique(),
        "cells": frame[["dataset", "split_seed"]].drop_duplicates().shape[0],
        "mean_internal_default8_schema_risk_reduction": float(1 - built_risk / base_risk),
        "settings": comparisons,
        "strength2_vs_iid_wins_default8": int(sum(
            group.set_index("external_method").loc["strength2_16", "external_residual"]
            < group.set_index("external_method").loc["iid16", "external_residual"]
            for _, group in built.groupby(["dataset", "split_seed"])
        )),
        "total_tabpfn_calls": int(compute.tabpfn_calls.sum()),
        "total_forward_ensemble_members": int(compute.forward_ensemble_members.sum()),
        "wall_seconds": float(compute.wall_seconds.sum()),
        "compute_note": "calls and internal forward members are reported separately; neither is called a fitted model",
    }
    (RESULTS / "completion_tabpfn_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
