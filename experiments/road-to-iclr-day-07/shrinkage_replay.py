#!/usr/bin/env python3
"""Exploratory fixed-base replay for state-CV residual geometry trust."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
GEOMETRY = HERE.parent / "geometry_transfer"
MPE = HERE.parent / "mpe_iclr"
for path in (GEOMETRY, MPE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from geometry_transfer import empirical_gain, operator_family, stable_seed, state_means  # noqa: E402
from representations import load_task  # noqa: E402
from run_retrospective import TASKS, base_residuals  # noqa: E402


LAMBDAS = np.round(np.linspace(0.0, 1.0, 11), 10)
OUT = HERE / "results"


def state_balanced_mse(residual: np.ndarray, states: np.ndarray, ordered: np.ndarray, prediction: np.ndarray) -> float:
    values = []
    for index, state in enumerate(ordered):
        group = residual[states == state]
        values.append(float(np.mean((group - prediction[index]) ** 2)))
    return float(np.mean(values))


def folds_for(task_name: str, split: int, states: np.ndarray) -> list[np.ndarray]:
    order = states.copy()
    np.random.default_rng(stable_seed("day7-trust-folds", task_name, split)).shuffle(order)
    return [np.sort(x) for x in np.array_split(order, 5) if len(x)]


def inner_scores(task, split: int, cache: dict[str, np.ndarray]) -> pd.DataFrame:
    states = cache["t_states"]
    residual = cache["residual_t"]
    row_state = cache["row_state_t"]
    rows = []
    for fold_index, held in enumerate(folds_for(task.name, split, states)):
        train = np.setdiff1d(states, held)
        mu_train = state_means(residual, row_state, train)
        for operator, matrix in operator_family(task.distance, train, held).items():
            raw_prediction = matrix @ mu_train
            for lam in LAMBDAS:
                mse = state_balanced_mse(residual, row_state, held, lam * raw_prediction)
                rows.append({"fold": fold_index, "operator": operator, "lambda": float(lam), "mse": mse})
    frame = pd.DataFrame(rows)
    return frame.groupby(["operator", "lambda"], as_index=False).mse.mean()


def choose(scores: pd.DataFrame, allowed_lambdas: set[float]) -> tuple[str, float, float]:
    candidates = scores[scores["lambda"].isin(allowed_lambdas)].copy()
    candidates = candidates.sort_values(["mse", "lambda", "operator"], ascending=[True, True, True])
    row = candidates.iloc[0]
    return str(row.operator), float(row["lambda"]), float(row.mse)


def outer_gain(task, cache: dict[str, np.ndarray], operator: str, lam: float) -> float:
    train, test = cache["t_states"], cache["u_states"]
    mu_train = state_means(cache["residual_t"], cache["row_state_t"], train)
    prediction = operator_family(task.distance, train, test)[operator] @ mu_train
    local = {state: index for index, state in enumerate(test.tolist())}
    state_index = np.asarray([local[state] for state in cache["row_state_u"]])
    return empirical_gain(cache["residual_u"], state_index, lam * prediction)


def oracle(task, cache: dict[str, np.ndarray], continuous: bool) -> tuple[str, float, float]:
    train, test = cache["t_states"], cache["u_states"]
    mu_t = state_means(cache["residual_t"], cache["row_state_t"], train)
    mu_u = state_means(cache["residual_u"], cache["row_state_u"], test)
    best = ("", 0.0, -np.inf)
    for name, matrix in operator_family(task.distance, train, test).items():
        g = matrix @ mu_t
        cross = float(np.mean(mu_u * g))
        square = float(np.mean(g * g))
        lam = float(np.clip(cross / square, 0.0, 1.0)) if continuous and square > 0 else 1.0
        gain = float(np.mean(2 * lam * mu_u * g - (lam * g) ** 2))
        candidate = (name, lam, gain)
        if gain > best[2] + 1e-15 or (abs(gain - best[2]) <= 1e-15 and (lam, name) < (best[1], best[0])):
            best = candidate
    return best


def run() -> pd.DataFrame:
    rows = []
    for task_name in TASKS:
        task = load_task(task_name)
        for split in range(5):
            cache = base_residuals(task, split)
            scores = inner_scores(task, split, cache)
            selections = {
                "full": choose(scores, {1.0}),
                "binary": choose(scores, {0.0, 1.0}),
                "shrink": choose(scores, set(LAMBDAS.tolist())),
            }
            for rule, (operator, lam, inner_mse) in selections.items():
                rows.append({
                    "source": task.manifest["source_unit"], "task": task_name, "split": split,
                    "rule": rule, "operator": operator, "lambda": lam, "inner_mse": inner_mse,
                    "outer_gain": outer_gain(task, cache, operator, lam), "deployable": True,
                })
            for rule, continuous in (("oracle_full", False), ("oracle_shrink", True)):
                operator, lam, gain = oracle(task, cache, continuous)
                rows.append({
                    "source": task.manifest["source_unit"], "task": task_name, "split": split,
                    "rule": rule, "operator": operator, "lambda": lam, "inner_mse": np.nan,
                    "outer_gain": gain, "deployable": False,
                })
            print(f"{task_name} split={split}", flush=True)
    return pd.DataFrame(rows)


def summarize(cells: pd.DataFrame) -> dict:
    deployable = cells[cells.deployable]
    by_rule = {}
    for rule, frame in cells.groupby("rule"):
        task_split = frame.groupby(["source", "task", "split"], as_index=False).outer_gain.mean()
        source_means = task_split.groupby("source").outer_gain.mean()
        by_rule[rule] = {
            "cells": int(len(task_split)),
            "mean_outer_gain": float(task_split.outer_gain.mean()),
            "source_balanced_outer_gain": float(source_means.mean()),
            "harmful_cells": int((task_split.outer_gain < 0).sum()),
            "positive_sources": int((source_means > 0).sum()),
            "sources": int(len(source_means)),
            "median_lambda": float(frame["lambda"].median()),
        }
    pivot = deployable.pivot_table(index=["source", "task", "split"], columns="rule", values="outer_gain")
    by_rule["shrink"]["mean_gain_over_full"] = float((pivot["shrink"] - pivot["full"]).mean())
    by_rule["shrink"]["mean_gain_over_binary"] = float((pivot["shrink"] - pivot["binary"]).mean())
    gates = {
        "positive_source_balanced_gain": by_rule["shrink"]["source_balanced_outer_gain"] > 0,
        "improves_over_full": by_rule["shrink"]["mean_gain_over_full"] > 0,
        "no_more_harm_than_binary": by_rule["shrink"]["harmful_cells"] <= by_rule["binary"]["harmful_cells"],
        "nontrivial_median_trust": 0 < by_rule["shrink"]["median_lambda"] < 1,
    }
    return {"status": "complete", "rules": by_rule, "gates": gates, "recommend_confirmation": all(gates.values())}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cells = run()
    cells.to_csv(OUT / "cells.csv", index=False)
    summary = summarize(cells)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

