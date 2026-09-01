#!/usr/bin/env python3
"""Tune a context-only shrinkage rule for the exploratory adaptive process.

This is a post-primary development experiment.  It never reads evaluation
labels.  Cross-fitted context predictions propose an episode-level correlation
strength; a global reliability shrinkage and cap are tuned on the disjoint
development panel, with leave-one-development-dataset-out diagnostics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from common import CACHE, CONFIG, atomic_json, gaussian_scores
from score_projective import learned_covariances, load_models


TAUS = [0.0, 12.0, 24.0, 48.0, 96.0, 192.0, 384.0]
CAPS = [0.25, 0.50, 0.75, 0.99]


def average_kernel(hidden: np.ndarray, models: list[Any], device: torch.device) -> np.ndarray:
    h = torch.from_numpy(hidden.astype(np.float32)).to(device)
    kernels = []
    with torch.no_grad():
        for model in models:
            unit = model.features(h).cpu().numpy().astype(np.float64)
            kernels.append(np.einsum("snr,smr->nm", unit, unit) / len(unit))
    kernel = np.mean(kernels, axis=0)
    np.fill_diagonal(kernel, 1.0)
    return kernel


def scores_for_covariance(
    mean: np.ndarray,
    target: np.ndarray,
    coefficients: np.ndarray,
    covariance: np.ndarray,
) -> tuple[float, float]:
    truth = np.einsum("fgn,n->fg", coefficients, target)
    prediction = np.einsum("fgn,n->fg", coefficients, mean)
    variance = np.einsum("fgn,nm,fgm->fg", coefficients, covariance, coefficients)
    scores = gaussian_scores(truth, prediction, variance)
    return float(scores["nll"].mean()), float(scores["crps"].mean())


def load_context_choices() -> pd.DataFrame:
    paths = sorted((CACHE / "adaptive_rho").glob("dev_curves_shard*.parquet"))
    if len(paths) != 2:
        raise RuntimeError(f"expected two adaptive development curve shards, found {len(paths)}")
    curves = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    expected = (
        len(CONFIG["development_datasets"])
        * len(CONFIG["development_splits"])
        * int(CONFIG["development_context_replicates"])
        * len(CONFIG["context_sizes"])
    )
    if curves["episode"].nunique() != expected:
        raise RuntimeError(f"adaptive curve cache incomplete: {curves['episode'].nunique()} != {expected}")
    curves = curves.sort_values(["episode", "rho"])
    choices = curves.loc[curves.groupby("episode")["calibration_nll"].idxmin()].copy()
    return choices[
        ["dataset", "episode", "context_size", "rho", "calibration_nll", "calibration_crps", "calibration_functionals"]
    ].rename(columns={"rho": "raw_rho"})


def build_candidate_scores(
    choices: pd.DataFrame, models: list[Any], summary: dict[str, Any], device: torch.device
) -> pd.DataFrame:
    by_episode = choices.set_index("episode").to_dict(orient="index")
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(sorted((CACHE / "tabicl_episodes" / "dev").glob("*.npz"))):
        choice = by_episode[path.name]
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"].item()))
            hidden = data["hidden"].astype(np.float32)
            mean = data["mean"].astype(np.float64)
            target = data["target"].astype(np.float64)
            families = list(meta["families"])
            family_indices = [families.index(name) for name in CONFIG["primary_aggregate_families"]]
            coefficients = data["coefficients"].astype(np.float64)[family_indices]
            base_variance = data["variance"].astype(np.float64)
        context_size = int(meta["context_size"])
        context_index = list(map(int, CONFIG["context_sizes"])).index(context_size)
        variance = np.maximum(
            base_variance * float(summary["marginal_temperatures"][str(context_size)]), 1e-10
        )
        sd = np.sqrt(variance)
        diagonal_covariance = np.diag(variance)
        kernel = average_kernel(hidden, models, device)
        kernel_covariance = sd[:, None] * kernel * sd[None, :]
        fixed_covariance, _ = learned_covariances(hidden, variance, context_index, models, device)

        for name, covariance in (
            ("diagonal", diagonal_covariance),
            ("fixed_head", fixed_covariance),
        ):
            nll, crps = scores_for_covariance(mean, target, coefficients, covariance)
            rows.append(
                {
                    "dataset": meta["dataset"],
                    "episode": path.name,
                    "context_size": context_size,
                    "rule": name,
                    "tau": np.nan,
                    "cap": np.nan,
                    "raw_rho": float(choice["raw_rho"]),
                    "selected_rho": np.nan,
                    "nll": nll,
                    "crps": crps,
                }
            )

        raw_rho = float(choice["raw_rho"])
        count = float(choice["calibration_functionals"])
        for tau in TAUS:
            for cap in CAPS:
                selected_rho = min(cap, raw_rho * count / (count + tau))
                covariance = (1.0 - selected_rho) * diagonal_covariance + selected_rho * kernel_covariance
                nll, crps = scores_for_covariance(mean, target, coefficients, covariance)
                rows.append(
                    {
                        "dataset": meta["dataset"],
                        "episode": path.name,
                        "context_size": context_size,
                        "rule": "adaptive",
                        "tau": tau,
                        "cap": cap,
                        "raw_rho": raw_rho,
                        "selected_rho": selected_rho,
                        "nll": nll,
                        "crps": crps,
                    }
                )
        if (index + 1) % 27 == 0:
            print(f"scored adaptive development {index + 1}", flush=True)
    return pd.DataFrame(rows)


def dataset_table(cells: pd.DataFrame) -> pd.DataFrame:
    return cells.groupby(["dataset", "rule", "tau", "cap"], dropna=False, as_index=False)[
        ["nll", "crps"]
    ].mean()


def select_candidate(table: pd.DataFrame, datasets: list[str]) -> tuple[float, float]:
    candidates = table[(table["rule"] == "adaptive") & table["dataset"].isin(datasets)]
    scores = candidates.groupby(["tau", "cap"], as_index=False)["nll"].mean()
    # Deterministic conservative tie breaking: more shrinkage, then lower cap.
    best = scores.sort_values(["nll", "tau", "cap"], ascending=[True, False, True]).iloc[0]
    return float(best["tau"]), float(best["cap"])


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, summary = load_models(device)
    choices = load_context_choices()
    cells = build_candidate_scores(choices, models, summary, device)
    table = dataset_table(cells)
    datasets = sorted(table["dataset"].unique())
    tau, cap = select_candidate(table, datasets)

    cv_rows = []
    for heldout in datasets:
        train = [dataset for dataset in datasets if dataset != heldout]
        fold_tau, fold_cap = select_candidate(table, train)
        row = table[
            (table["dataset"] == heldout)
            & (table["rule"] == "adaptive")
            & (table["tau"] == fold_tau)
            & (table["cap"] == fold_cap)
        ].iloc[0]
        diagonal = table[(table["dataset"] == heldout) & (table["rule"] == "diagonal")].iloc[0]
        fixed = table[(table["dataset"] == heldout) & (table["rule"] == "fixed_head")].iloc[0]
        cv_rows.append(
            {
                "heldout_dataset": heldout,
                "selected_tau": fold_tau,
                "selected_cap": fold_cap,
                "adaptive_nll": float(row["nll"]),
                "adaptive_crps": float(row["crps"]),
                "diagonal_nll": float(diagonal["nll"]),
                "diagonal_crps": float(diagonal["crps"]),
                "fixed_nll": float(fixed["nll"]),
                "fixed_crps": float(fixed["crps"]),
            }
        )
    cv = pd.DataFrame(cv_rows)
    selected = table[(table["rule"] == "adaptive") & (table["tau"] == tau) & (table["cap"] == cap)]
    diagonal = table[table["rule"] == "diagonal"]
    fixed = table[table["rule"] == "fixed_head"]
    out = CACHE / "adaptive_rho"
    cells.to_parquet(out / "dev_candidate_cells.parquet", index=False)
    table.to_csv(out / "dev_candidate_by_dataset.csv", index=False)
    cv.to_csv(out / "dev_leave_one_dataset_out.csv", index=False)
    choices.to_csv(out / "dev_context_choices.csv", index=False)
    payload = {
        "status": "post-primary exploratory development only",
        "selection_objective": "dataset-balanced primary aggregate NLL on disjoint development datasets",
        "rule": "rho=min(cap, raw_context_argmin_rho * m/(m+tau))",
        "tau": tau,
        "cap": cap,
        "rho_grid": sorted(map(float, choices["raw_rho"].unique())),
        "development": {
            "adaptive_nll": float(selected["nll"].mean()),
            "adaptive_crps": float(selected["crps"].mean()),
            "diagonal_nll": float(diagonal["nll"].mean()),
            "diagonal_crps": float(diagonal["crps"].mean()),
            "fixed_nll": float(fixed["nll"].mean()),
            "fixed_crps": float(fixed["crps"].mean()),
        },
        "leave_one_development_dataset_out": {
            "adaptive_nll": float(cv["adaptive_nll"].mean()),
            "adaptive_crps": float(cv["adaptive_crps"].mean()),
            "diagonal_nll": float(cv["diagonal_nll"].mean()),
            "diagonal_crps": float(cv["diagonal_crps"].mean()),
            "fixed_nll": float(cv["fixed_nll"].mean()),
            "fixed_crps": float(cv["fixed_crps"].mean()),
            "nll_dataset_wins_vs_diagonal": int((cv["adaptive_nll"] < cv["diagonal_nll"]).sum()),
            "crps_dataset_wins_vs_diagonal": int((cv["adaptive_crps"] < cv["diagonal_crps"]).sum()),
        },
    }
    atomic_json(out / "selected_rule.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
