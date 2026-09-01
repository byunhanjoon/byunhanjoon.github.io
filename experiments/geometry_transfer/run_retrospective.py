#!/usr/bin/env python3
"""State-disjoint retrospective Geometry Transfer Law experiment.

The base model never receives ``field_state``. Training-state residual means
are formed from genuine row-level out-of-fold predictions. Test-state outcomes
are read only after the base and target-independent operators are fixed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

HERE = Path(__file__).resolve().parent
MPE = HERE.parent / "mpe_iclr"
if str(MPE) not in sys.path:
    sys.path.insert(0, str(MPE))

from representations import load_task, split_state_indices  # noqa: E402
from geometry_transfer import (  # noqa: E402
    decompose,
    empirical_gain,
    median_bandwidth,
    operator_family,
    stable_seed,
    state_mean_variance,
    state_means,
)


TASKS = [
    "acs_occupation", "acs_industry", "tlc_pickup_zone", "tlc_dropoff_zone",
    "citibike_start_station", "airline_origin_airport", "airline_destination_airport",
    "employee_salaries", "medical_charges",
]
RAW = HERE / "raw" / "retrospective"
CACHE = RAW / "base_cache"


def atomic_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def prepare_features(task, rows: np.ndarray) -> tuple[pd.DataFrame, list[int]]:
    frame = task.rows.iloc[rows][task.manifest["ordinary_covariates"]].copy()
    categorical = []
    for index, column in enumerate(frame.columns):
        if not pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = frame[column].astype("string").fillna("__MISSING__").astype(str)
            categorical.append(index)
        else:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(float(pd.to_numeric(frame[column], errors="coerce").median()))
    return frame, categorical


def balanced_subsample(indices: np.ndarray, states: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    if len(indices) <= limit:
        return indices
    groups = {}
    for state in np.unique(states[indices]):
        groups[state] = indices[states[indices] == state]
    chosen = [rng.choice(values, 1)[0] for values in groups.values()]
    remaining = limit - len(chosen)
    pool = np.setdiff1d(indices, np.asarray(chosen), assume_unique=False)
    if remaining > 0:
        chosen.extend(rng.choice(pool, min(remaining, len(pool)), replace=False).tolist())
    return np.sort(np.asarray(chosen, dtype=int))


def fit_predict(task, train_rows: np.ndarray, predict_rows: np.ndarray, y: np.ndarray, seed: int, iterations: int = 140) -> np.ndarray:
    rng = np.random.default_rng(seed)
    row_state = task.row_state_indices()
    fit_rows = balanced_subsample(train_rows, row_state, 75000, rng)
    combined = np.concatenate([fit_rows, predict_rows])
    x, categorical = prepare_features(task, combined)
    model = CatBoostRegressor(
        iterations=iterations, depth=7, learning_rate=0.08, loss_function="RMSE",
        l2_leaf_reg=5.0, random_seed=seed, verbose=False, allow_writing_files=False,
        thread_count=8, random_strength=0.25,
    )
    model.fit(x.iloc[:len(fit_rows)], y[fit_rows], cat_features=categorical)
    return np.asarray(model.predict(x.iloc[len(fit_rows):]), dtype=float)


def base_residuals(task, split: int, force: bool = False) -> dict[str, np.ndarray]:
    path = CACHE / f"{task.name}__split{split}.npz"
    if path.exists() and not force:
        with np.load(path) as data:
            return {key: data[key] for key in data.files}
    parts = split_state_indices(task, split)
    t_states = np.concatenate([parts["train"], parts["validation"]])
    u_states = parts["test"]
    row_state = task.row_state_indices()
    t_rows = np.flatnonzero(np.isin(row_state, t_states))
    u_rows = np.flatnonzero(np.isin(row_state, u_states))
    raw_y = pd.to_numeric(task.rows["target"], errors="raise").to_numpy(float)
    center = float(raw_y[t_rows].mean()); scale = float(raw_y[t_rows].std()) or 1.0
    y = (raw_y - center) / scale

    # Stable random row folds. Each residual is predicted by a model that did
    # not see that row; no held-out-state row or target enters these fits.
    rng = np.random.default_rng(stable_seed("base-fold", task.name, split))
    shuffled = t_rows.copy(); rng.shuffle(shuffled)
    folds = np.array_split(shuffled, 3)
    oof = np.empty(len(t_rows), dtype=float)
    t_position = {row: pos for pos, row in enumerate(t_rows.tolist())}
    for fold_index, held in enumerate(folds):
        fit = np.setdiff1d(t_rows, held, assume_unique=False)
        pred = fit_predict(task, fit, held, y, stable_seed("catboost-oof", task.name, split, fold_index))
        oof[[t_position[row] for row in held]] = pred
    pred_u = fit_predict(task, t_rows, u_rows, y, stable_seed("catboost-full", task.name, split))
    payload = {
        "t_rows": t_rows, "u_rows": u_rows, "t_states": t_states, "u_states": u_states,
        "row_state_t": row_state[t_rows], "row_state_u": row_state[u_rows],
        "residual_t": y[t_rows] - oof, "residual_u": y[u_rows] - pred_u,
        "target_t": y[t_rows], "target_u": y[u_rows], "base_u": pred_u,
        "target_center": np.asarray([center]), "target_scale": np.asarray([scale]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return payload


def smoothness(signal: np.ndarray, distance: np.ndarray) -> tuple[float, float, float]:
    h = median_bandwidth(distance, np.arange(len(signal)))
    w = np.exp(-0.5 * (distance / max(h, 1e-12)) ** 2)
    np.fill_diagonal(w, 0)
    centered = signal - np.mean(signal)
    variance = float(np.mean(centered**2)) + 1e-12
    energy = float(np.sum(w * (signal[:, None] - signal[None, :]) ** 2) / max(np.sum(w), 1e-12))
    moran = float(len(signal) / max(np.sum(w), 1e-12) * np.sum(w * centered[:, None] * centered[None, :]) / max(np.sum(centered**2), 1e-12))
    return energy / variance, energy, moran


def geometry_heuristics(task, t_states: np.ndarray, u_states: np.ndarray, raw_mean_t: np.ndarray, residual_mean_t: np.ndarray) -> dict[str, float]:
    block = task.distance[np.ix_(u_states, t_states)]
    nearest = np.min(block, axis=1)
    dtt = task.distance[np.ix_(t_states, t_states)]
    raw_smooth, raw_energy, _ = smoothness(raw_mean_t, dtt)
    residual_smooth, residual_energy, moran = smoothness(residual_mean_t, dtt)
    return {
        "nearest_support_distance": float(np.mean(nearest)),
        "mean_support_distance": float(np.mean(block)),
        "cover_radius": float(np.max(nearest)),
        "metric_diameter": float(np.max(task.distance)),
        "state_cardinality": float(len(t_states)),
        "raw_smoothness": raw_smooth,
        "conditional_smoothness": residual_smooth,
        "dirichlet_energy": residual_energy,
        "raw_dirichlet_energy": raw_energy,
        "moran_autocorrelation": moran,
    }


def run_cell(task_name: str, split: int, force: bool = False) -> tuple[list[dict], list[dict]]:
    task = load_task(task_name)
    cache = base_residuals(task, split, force=force)
    t_states, u_states = cache["t_states"], cache["u_states"]
    rt, ru = cache["residual_t"], cache["residual_u"]
    st, su = cache["row_state_t"], cache["row_state_u"]
    mu_t = state_means(rt, st, t_states)
    sigma = state_mean_variance(rt, st, t_states)
    mu_u = state_means(ru, su, u_states)
    raw_mean_t = state_means(cache["target_t"], st, t_states)
    counts = np.asarray([np.sum(st == state) for state in t_states])
    heur = geometry_heuristics(task, t_states, u_states, raw_mean_t, mu_t)
    heur["mean_train_state_frequency"] = float(np.mean(counts))
    operators = operator_family(task.distance, t_states, u_states)
    rows, state_rows = [], []
    q = np.full(len(u_states), 1 / len(u_states))
    for name, a in operators.items():
        dec = decompose(mu_u, mu_t, a, sigma, q)
        pred_state = a @ mu_t
        # Remap global held-state indices to local 0..|U|-1 indices.
        local = {state: index for index, state in enumerate(u_states.tolist())}
        su_local = np.asarray([local[state] for state in su])
        actual = empirical_gain(ru, su_local, pred_state)
        realized_oracle = float(np.mean(2 * mu_u * pred_state - pred_state**2))
        row = {
            "source": task.manifest["source_unit"], "task": task_name,
            "field": task.manifest.get("field", "field_state"),
            "metric": task.manifest.get("metric", task.manifest.get("metric_type", "external")),
            "split": split, "operator": name, "base_model": "catboost",
            "train_states": len(t_states), "test_states": len(u_states),
            "train_rows": len(rt), "test_rows": len(ru),
            "possible_signal": dec.possible_signal, "approximation_error": dec.approximation_error,
            "transferable_signal": dec.transferable_signal, "noise_cost": dec.noise_cost,
            "delta_theory": dec.delta, "delta_actual": actual,
            "delta_realized_oracle": realized_oracle, "gtr": dec.gtr,
            "oracle_identity_error": actual - realized_oracle,
            **heur,
        }
        rows.append(row)
        for index, state in enumerate(u_states):
            noise_u = float(np.dot(a[index] ** 2, sigma))
            signal_u = float(mu_u[index] ** 2 - (mu_u[index] - pred_state[index]) ** 2)
            mask = su == state
            actual_u = float(np.mean(ru[mask]**2 - (ru[mask] - pred_state[index])**2))
            support = float(np.min(task.distance[state, t_states]))
            state_rows.append({
                "source": task.manifest["source_unit"], "task": task_name, "split": split,
                "operator": name, "state_id": task.state_ids[state], "rows": int(mask.sum()),
                "support_distance": support, "mu_u": mu_u[index], "prediction": pred_state[index],
                "local_transferable_signal": signal_u, "local_noise_cost": noise_u,
                "delta_theory_state": signal_u-noise_u, "delta_actual_state": actual_u,
            })
    print(f"{task_name} split={split} rows={len(rows)}", flush=True)
    return rows, state_rows


def corruption_and_sample_size(tasks: list[str], all_splits: list[int]) -> None:
    corruption_rows, size_rows = [], []
    for task_name in tasks:
        task = load_task(task_name); split = all_splits[0]
        cache = base_residuals(task, split)
        t, u = cache["t_states"], cache["u_states"]
        rt, ru, st, su = cache["residual_t"], cache["residual_u"], cache["row_state_t"], cache["row_state_u"]
        mu_u = state_means(ru, su, u); local = {state:i for i,state in enumerate(u.tolist())}; su_local=np.asarray([local[x] for x in su])
        rng = np.random.default_rng(stable_seed("diagnostics", task_name))
        for fraction in (0., .1, .25, .5, 1.):
            perm = np.arange(len(task.states)); count = int(round(fraction * len(perm)))
            selected = rng.choice(len(perm), count, replace=False) if count else np.array([], dtype=int)
            perm[selected] = rng.permutation(perm[selected])
            corrupted = task.distance[np.ix_(perm, perm)]
            mt = state_means(rt, st, t); sig = state_mean_variance(rt, st, t)
            a = operator_family(corrupted, t, u)["rbf"]
            dec = decompose(mu_u, mt, a, sig)
            corruption_rows.append({"task":task_name,"source":task.manifest["source_unit"],"corruption":fraction,
                                    "transferable_signal":dec.transferable_signal,"noise_cost":dec.noise_cost,
                                    "delta_theory":dec.delta,"delta_actual":empirical_gain(ru,su_local,a@mt)})
        for n in (5, 10, 20, 50, 100, 500):
            sampled=[]
            for state in t:
                idx=np.flatnonzero(st==state)
                sampled.extend(rng.choice(idx, min(n,len(idx)), replace=False).tolist())
            sampled=np.asarray(sampled)
            mt=state_means(rt[sampled],st[sampled],t); sig=state_mean_variance(rt[sampled],st[sampled],t)
            a=operator_family(task.distance,t,u)["rbf"]; dec=decompose(mu_u,mt,a,sig)
            size_rows.append({"task":task_name,"source":task.manifest["source_unit"],"rows_per_state_cap":n,
                              "transferable_signal":dec.transferable_signal,"noise_cost":dec.noise_cost,"gtr":dec.gtr,
                              "delta_theory":dec.delta,"delta_actual":empirical_gain(ru,su_local,a@mt)})
    pd.DataFrame(corruption_rows).to_csv(RAW/"metric_perturbation.csv",index=False)
    pd.DataFrame(size_rows).to_csv(RAW/"real_sample_size.csv",index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all")
    parser.add_argument("--split", default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    tasks = TASKS if args.task == "all" else [args.task]
    splits = list(range(5)) if args.split == "all" else [int(args.split)]
    RAW.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    rows, states = [], []
    started = time.time()
    for task_name in tasks:
        for split in splits:
            r, s = run_cell(task_name, split, args.force); rows.extend(r); states.extend(s)
    pd.DataFrame(rows).to_csv(RAW / "cells.csv", index=False)
    pd.DataFrame(states).to_csv(RAW / "state_cells.csv", index=False)
    if args.task == "all" and args.split == "all":
        corruption_and_sample_size(["acs_occupation", "tlc_pickup_zone"], splits)
    atomic_json({"status":"complete","tasks":tasks,"splits":splits,"cells":len(rows),
                 "state_cells":len(states),"wall_seconds":time.time()-started,
                 "cross_fitting":"3-fold row OOF within observed states",
                 "test_outcomes_used_for_operator":False}, RAW/"run_summary.json")


if __name__ == "__main__":
    main()
