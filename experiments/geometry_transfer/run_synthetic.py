#!/usr/bin/env python3
"""Reproducible synthetic validation and paper figures 1--5 and 11."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr, spearmanr

from geometry_transfer import decompose, empirical_gain, operator_family


HERE = Path(__file__).resolve().parent
RAW = HERE / "raw" / "synthetic"
FIG = HERE / "figures"


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", dpi=180)
    plt.savefig(FIG / f"{name}.pdf")
    plt.close()


def circle_distance(n: int) -> np.ndarray:
    x = np.arange(n)
    raw = np.abs(x[:, None] - x[None, :])
    return np.minimum(raw, n - raw).astype(float)


def exact_identity(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for n in (12, 24, 48):
        distance = circle_distance(n)
        train = np.arange(0, n, 2)
        unseen = np.arange(1, n, 2)
        for operator_name, a in operator_family(distance, train, unseen).items():
            for signal in (0.0, 0.25, 0.75, 1.5):
                for noise in (0.1, 0.5, 1.5):
                    for rows_per_state in (5, 25, 100):
                        phase = rng.uniform(0, 2 * np.pi)
                        mu = signal * (np.sin(2 * np.pi * np.arange(n) / n + phase) + 0.25 * np.cos(4 * np.pi * np.arange(n) / n))
                        sigma = np.full(len(train), noise**2 / rows_per_state)
                        truth = decompose(mu[unseen], mu[train], a, sigma)
                        gains = []
                        # Monte Carlo over noisy state estimates and fresh test outcomes.
                        for _ in range(600):
                            mu_hat = mu[train] + rng.normal(0, np.sqrt(sigma))
                            test = mu[unseen, None] + rng.normal(0, 1.0, (len(unseen), 30))
                            gains.append(np.mean(test**2 - (test - (a @ mu_hat)[:, None]) ** 2))
                        rows.append({
                            "states": n, "operator": operator_name, "signal": signal,
                            "training_noise": noise, "rows_per_state": rows_per_state,
                            "transferable_signal": truth.transferable_signal,
                            "noise_cost": truth.noise_cost, "gtr": truth.gtr,
                            "delta_theory": truth.delta, "delta_empirical": float(np.mean(gains)),
                            "mc_se": float(np.std(gains, ddof=1) / np.sqrt(len(gains))),
                        })
    return pd.DataFrame(rows)


def phase_transition(rng: np.random.Generator) -> pd.DataFrame:
    n = 32
    distance = circle_distance(n)
    train, unseen = np.arange(0, n, 2), np.arange(1, n, 2)
    a = operator_family(distance, train, unseen)["rbf"]
    rows = []
    for signal in np.linspace(0.05, 2.0, 16):
        for noise in np.linspace(0.05, 2.0, 16):
            mu = signal * np.sin(2 * np.pi * np.arange(n) / n)
            sigma = np.full(len(train), noise**2 / 20)
            truth = decompose(mu[unseen], mu[train], a, sigma)
            gain = []
            for _ in range(500):
                mu_hat = mu[train] + rng.normal(0, np.sqrt(sigma))
                r = mu[unseen, None] + rng.normal(size=(len(unseen), 20))
                gain.append(np.mean(r**2 - (r - (a @ mu_hat)[:, None]) ** 2))
            rows.append({"signal": signal, "noise": noise, "gtr": truth.gtr,
                         "delta_theory": truth.delta, "delta_empirical": np.mean(gain)})
    return pd.DataFrame(rows)


def spectral(rng: np.random.Generator) -> pd.DataFrame:
    n = 40
    distance = circle_distance(n)
    w = np.exp(-0.5 * (distance / 2.0) ** 2)
    w /= w.sum(axis=1, keepdims=True)
    h, v = np.linalg.eigh((w + w.T) / 2)
    order = np.argsort(h)[::-1]
    h, v = np.clip(h[order], 0, 1), v[:, order]
    rows = []
    modes = {"low": 1, "middle": n // 4, "high": n - 2, "mixed": None}
    for label, mode in modes.items():
        for snr in np.geomspace(0.02, 20, 24):
            coeff = np.zeros(n)
            if mode is None:
                coeff[[1, n // 4, n - 2]] = np.sqrt(snr / 3)
            else:
                coeff[mode] = np.sqrt(snr)
            mu = v @ coeff
            sigma2 = 1.0
            predicted = float(np.sum((2 * h - h**2) * coeff**2 - sigma2 * h**2)) / n
            measured = []
            for _ in range(500):
                mu_hat = mu + rng.normal(0, np.sqrt(sigma2), n)
                measured.append(np.mean(mu**2 - (mu - w @ mu_hat) ** 2))
            threshold = np.nan if mode is None or h[mode] <= 0 else h[mode] / (2 - h[mode])
            rows.append({"mode_group": label, "mode": -1 if mode is None else mode,
                         "snr": snr, "threshold": threshold,
                         "delta_theory": predicted, "delta_empirical": np.mean(measured)})
    return pd.DataFrame(rows)


def no_free_lunch(rng: np.random.Generator) -> pd.DataFrame:
    n = 30
    distance = circle_distance(n)
    train, unseen = np.arange(0, n, 2), np.arange(1, n, 2)
    a = operator_family(distance, train, unseen)["rbf"]
    mu_t = rng.normal(size=len(train))
    transferred = a @ mu_t
    sigma = np.full(len(train), 0.005)
    rows = []
    for target, mu_u in (("aligned", transferred), ("anti_aligned", -transferred)):
        d = decompose(mu_u, mu_t, a, sigma)
        rows.append({
            "target": target, "delta_theory": d.delta,
            "transferable_signal": d.transferable_signal, "noise_cost": d.noise_cost,
            "nearest_support_distance": float(np.mean(np.min(distance[np.ix_(unseen, train)], axis=1))),
            "median_support_distance": float(np.median(np.min(distance[np.ix_(unseen, train)], axis=1))),
            "cover_radius": float(np.max(np.min(distance[np.ix_(unseen, train)], axis=1))),
            "degree_mean": 2.0, "metric_diameter": float(distance.max()),
            "landmark_coverage": float(len(train) / n), "metric_dimension": 1.0,
            "train_states": len(train), "unseen_states": len(unseen), "noise_mean": float(sigma.mean()),
        })
    return pd.DataFrame(rows)


def sample_size(rng: np.random.Generator) -> pd.DataFrame:
    n = 32
    d = circle_distance(n)
    train, unseen = np.arange(0, n, 2), np.arange(1, n, 2)
    a = operator_family(d, train, unseen)["rbf"]
    # Illustrative scale chosen to place the analytic threshold inside the
    # declared grid; the independent 2-D phase grid is the validation test.
    mu = 0.15 * np.sin(2 * np.pi * np.arange(n) / n)
    rows = []
    for nrow in (5, 10, 20, 50, 100, 500):
        sigma = np.full(len(train), 1.0 / nrow)
        truth = decompose(mu[unseen], mu[train], a, sigma)
        gain = []
        for _ in range(1000):
            mh = mu[train] + rng.normal(0, np.sqrt(sigma))
            gain.append(np.mean(mu[unseen] ** 2 - (mu[unseen] - a @ mh) ** 2))
        rows.append({"rows_per_state": nrow, "gtr": truth.gtr,
                     "delta_theory": truth.delta, "delta_empirical": np.mean(gain)})
    return pd.DataFrame(rows)


def number_of_states(rng: np.random.Generator) -> pd.DataFrame:
    n=64;d=circle_distance(n);mu=.3*np.sin(2*np.pi*np.arange(n)/n);rows=[]
    for nt in (4,8,16,24,32,48):
        train=np.unique(np.linspace(0,n-1,nt,endpoint=False).astype(int));query=np.setdiff1d(np.arange(n),train)
        per_state=max(5,640//len(train));sigma=np.full(len(train),1/per_state)
        for name,a in operator_family(d,train,query).items():
            dec=decompose(mu[query],mu[train],a,sigma);coverage=float(np.mean(np.min(d[np.ix_(query,train)],axis=1)))
            rows.append({"train_states":len(train),"rows_per_state":per_state,"operator":name,"mean_support_distance":coverage,"transferable_signal":dec.transferable_signal,"noise_cost":dec.noise_cost,"delta_theory":dec.delta})
    return pd.DataFrame(rows)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260829)
    identity = exact_identity(rng)
    phase = phase_transition(rng)
    spec = spectral(rng)
    nfl = no_free_lunch(rng)
    size = sample_size(rng)
    state_count = number_of_states(rng)
    for name, frame in (("identity", identity), ("phase_transition", phase),
                        ("spectral", spec), ("no_free_lunch", nfl), ("sample_size", size),
                        ("number_of_states",state_count)):
        frame.to_csv(RAW / f"{name}.csv", index=False)

    plt.figure(figsize=(6, 5))
    plt.scatter(identity.delta_theory, identity.delta_empirical, s=8, alpha=.45)
    lo, hi = identity[["delta_theory", "delta_empirical"]].min().min(), identity[["delta_theory", "delta_empirical"]].max().max()
    plt.plot([lo, hi], [lo, hi], "k--", lw=1); plt.xlabel("Exact theoretical Δ"); plt.ylabel("Monte Carlo Δ")
    savefig("figure_2_exact_identity")

    finite = phase[np.isfinite(phase.gtr)].copy()
    plt.figure(figsize=(6, 5)); plt.scatter(finite.gtr, finite.delta_empirical, c=finite.delta_theory, cmap="coolwarm", s=14)
    plt.axvline(1, color="black", ls="--"); plt.axhline(0, color="black", lw=.7); plt.xscale("log")
    plt.xlabel("GTR"); plt.ylabel("Monte Carlo risk improvement")
    savefig("figure_3_phase_transition")

    pivot = spec.pivot(index="mode_group", columns="snr", values="delta_empirical")
    plt.figure(figsize=(8, 3.2)); plt.imshow(pivot.values, aspect="auto", cmap="coolwarm", vmin=-np.max(np.abs(pivot.values)), vmax=np.max(np.abs(pivot.values)))
    plt.yticks(range(len(pivot)), pivot.index); ticks = np.linspace(0, len(pivot.columns)-1, 6).astype(int)
    plt.xticks(ticks, [f"{pivot.columns[i]:.2g}" for i in ticks]); plt.xlabel("Mode SNR"); plt.colorbar(label="Measured Δ")
    savefig("figure_4_spectral_phase")

    plt.figure(figsize=(5, 4)); plt.bar(nfl.target, nfl.delta_theory, color=["#3a7", "#c55"]); plt.axhline(0, color="black", lw=.8); plt.ylabel("Theoretical Δ")
    savefig("figure_5_same_metric_opposite_target")

    fig, ax1 = plt.subplots(figsize=(6, 4)); ax1.plot(size.rows_per_state, size.delta_empirical, "o-", label="actual Δ"); ax1.plot(size.rows_per_state, size.delta_theory, "s--", label="theory Δ"); ax1.axhline(0, color="black", lw=.7); ax1.set_xscale("log"); ax1.set_xlabel("Rows per training state"); ax1.set_ylabel("Risk improvement"); ax1.legend()
    savefig("figure_11_sample_size")

    r = pearsonr(identity.delta_theory, identity.delta_empirical).statistic
    rho = spearmanr(identity.delta_theory, identity.delta_empirical).statistic
    fit = linregress(identity.delta_theory, identity.delta_empirical)
    summary = {
        "identity_cells": len(identity), "pearson": float(r), "spearman": float(rho),
        "calibration_slope": float(fit.slope),
        "maximum_absolute_discrepancy": float(np.max(np.abs(identity.delta_theory - identity.delta_empirical))),
        "maximum_standardized_mc_discrepancy": float(np.max(np.abs(identity.delta_theory - identity.delta_empirical) / identity.mc_se.clip(lower=1e-15))),
        "sign_accuracy": float(np.mean(np.sign(identity.delta_theory) == np.sign(identity.delta_empirical))),
        "phase_sign_accuracy": float(np.mean(np.sign(phase.delta_theory) == np.sign(phase.delta_empirical))),
        "spectral_sign_accuracy": float(np.mean(np.sign(spec.delta_theory) == np.sign(spec.delta_empirical))),
        "no_free_lunch_opposite_signs": bool(nfl.delta_theory.iloc[0] > 0 > nfl.delta_theory.iloc[1]),
    }
    (RAW / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
