#!/usr/bin/env python3
"""Extended Direction 1: TabPFN and raw-set audits on equivalent SCMs."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNClassifier


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "direction_1"
FIG = ROOT / "figures"
NOISE_CELLS = [(0.00, 0.05), (0.05, 0.05), (0.10, 0.10), (0.15, 0.05), (0.20, 0.15), (0.25, 0.25)]
SAMPLE_SIZES = [128, 512, 2048]


def jsonify(v):
    if isinstance(v, dict):
        return {str(k): jsonify(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonify(x) for x in v]
    if isinstance(v, np.ndarray):
        return jsonify(v.tolist())
    if isinstance(v, (np.floating, float)):
        x = float(v)
        return x if math.isfinite(x) else None
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    return v


def generate_world(rng, n, q, p, world, nuisance=4):
    r = q + p - 2 * q * p
    if world == "randomized_causal":
        t = rng.binomial(1, 0.5, n)
        y = np.logical_xor(t, rng.binomial(1, r, n)).astype(int)
        ate = 1 - 2 * r
    elif world == "hidden_confounding":
        u = rng.binomial(1, 0.5, n)
        t = np.logical_xor(u, rng.binomial(1, q, n)).astype(int)
        y = np.logical_xor(u, rng.binomial(1, p, n)).astype(int)
        ate = 0.0
    else:
        raise ValueError(world)
    z = rng.normal(size=(n, nuisance))
    # Independent coding jitter avoids exact duplicate rows in pretrained
    # predictors while preserving equality of the two population laws.
    observed_t = t + rng.normal(0.0, 0.03, n)
    x = np.c_[observed_t, z]
    return x.astype(np.float32), y, float(ate), float(r)


def observational_summary(x, y):
    t = (x[:, 0] > 0.5).astype(int)
    corr = np.corrcoef(t, y)[0, 1] if np.std(t) and np.std(y) else 0.0
    return np.array([t.mean(), y.mean(), corr, np.mean(t == y), *x[:, 1:].mean(0), *x[:, 1:].std(0)])


def evaluate_predictions(y, prob):
    prob = np.clip(np.asarray(prob), 1e-7, 1 - 1e-7)
    return {
        "auroc": roc_auc_score(y, prob),
        "accuracy": accuracy_score(y, prob >= 0.5),
        "log_loss": log_loss(y, prob),
        "brier": brier_score_loss(y, prob),
        "mean_confidence": np.mean(np.maximum(prob, 1 - prob)),
    }


def fit_predict_model(name, x, y, x_all, seed, n_estimators, device):
    if name == "logistic":
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500, random_state=seed))
    elif name == "random_forest":
        model = RandomForestClassifier(n_estimators=240, min_samples_leaf=3, max_features="sqrt", n_jobs=-1, random_state=seed)
    elif name == "tabpfn":
        model = TabPFNClassifier(
            n_estimators=n_estimators,
            device=device,
            random_state=seed,
            fit_mode="fit_preprocessors",
            ignore_pretraining_limits=False,
        )
    else:
        raise ValueError(name)
    model.fit(x, y)
    prob = model.predict_proba(x_all)[:, 1]
    del model
    return prob


def run_tabpfn_grid(data_seeds, n_estimators, device):
    records = []
    checkpoint = OUT / "tabpfn_grid_checkpoint.json"
    total = len(NOISE_CELLS) * len(SAMPLE_SIZES) * len(data_seeds)
    done = 0
    begun = time.time()
    for cell_id, (q, p) in enumerate(NOISE_CELLS):
        r = q + p - 2 * q * p
        for n in SAMPLE_SIZES:
            for data_seed in data_seeds:
                world_data = {}
                for wi, world in enumerate(["randomized_causal", "hidden_confounding"]):
                    train_rng = np.random.default_rng(1_000_000 + 10_000 * cell_id + 10 * data_seed + wi)
                    test_rng = np.random.default_rng(2_000_000 + 10_000 * cell_id + 10 * data_seed + wi)
                    x, y, true_ate, _ = generate_world(train_rng, n, q, p, world)
                    xt, yt, _, _ = generate_world(test_rng, 2048, q, p, world)
                    cf_rng = np.random.default_rng(3_000_000 + 10_000 * cell_id + 10 * data_seed + wi)
                    zcf = cf_rng.normal(size=(512, 4)).astype(np.float32)
                    x0 = np.c_[np.zeros(512), zcf].astype(np.float32)
                    x1 = np.c_[np.ones(512), zcf].astype(np.float32)
                    xall = np.vstack([xt, x0, x1])
                    world_data[world] = (x, y, observational_summary(x, y))
                    for model_name in ["logistic", "random_forest", "tabpfn"]:
                        t0 = time.time()
                        prob = fit_predict_model(model_name, x, y, xall, data_seed, n_estimators, device)
                        test_prob = prob[: len(xt)]
                        plugin = prob[len(xt) + 512 :].mean() - prob[len(xt) : len(xt) + 512].mean()
                        records.append(
                            {
                                "cell_id": cell_id,
                                "q": q,
                                "p": p,
                                "observational_flip_rate_r": r,
                                "population_observational_association": 1 - 2 * r,
                                "n_train": n,
                                "data_seed": data_seed,
                                "world": world,
                                "model": model_name,
                                "true_ate": true_ate,
                                "plugin_ate": plugin,
                                "causal_absolute_error": abs(plugin - true_ate),
                                "runtime_seconds": time.time() - t0,
                                **evaluate_predictions(yt, test_prob),
                            }
                        )
                    # Prespecified shuffled-label TabPFN check: first three seeds.
                    if data_seed in data_seeds[:3]:
                        shuffled = y[train_rng.permutation(len(y))]
                        t0 = time.time()
                        prob = fit_predict_model("tabpfn", x, shuffled, xt, data_seed + 5000, max(2, n_estimators // 2), device)
                        records.append(
                            {
                                "cell_id": cell_id, "q": q, "p": p, "observational_flip_rate_r": r,
                                "population_observational_association": 1 - 2 * r, "n_train": n, "data_seed": data_seed,
                                "world": world, "model": "tabpfn_shuffled_labels", "true_ate": true_ate,
                                "plugin_ate": None, "causal_absolute_error": None, "runtime_seconds": time.time() - t0,
                                **evaluate_predictions(yt, prob),
                            }
                        )
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                distance = np.linalg.norm(world_data["randomized_causal"][2] - world_data["hidden_confounding"][2])
                for rec in records[-8:]:
                    if rec["cell_id"] == cell_id and rec["n_train"] == n and rec["data_seed"] == data_seed:
                        rec["paired_observational_summary_distance"] = distance
                done += 1
                if done % 3 == 0:
                    checkpoint.write_text(json.dumps(jsonify(records), indent=2) + "\n")
                    elapsed = time.time() - begun
                    eta = elapsed / done * (total - done)
                    print(f"grid {done}/{total}; elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m", flush=True)
    checkpoint.write_text(json.dumps(jsonify(records), indent=2) + "\n")
    return records


class RawSetATE(torch.nn.Module):
    def __init__(self, with_assumptions, width=64):
        super().__init__()
        self.with_assumptions = with_assumptions
        self.row = torch.nn.Sequential(torch.nn.Linear(6, width), torch.nn.GELU(), torch.nn.Linear(width, width), torch.nn.GELU())
        self.head = torch.nn.Sequential(torch.nn.Linear(width + (2 if with_assumptions else 0), width), torch.nn.GELU(), torch.nn.Linear(width, 1))

    def forward(self, rows, assumptions=None):
        pooled = self.row(rows).mean(1)
        if self.with_assumptions:
            pooled = torch.cat([pooled, assumptions], dim=1)
        return self.head(pooled).squeeze(1)


def task_batch(rng, batch, n, q_range, p_range):
    q = rng.uniform(*q_range, size=batch)
    p = rng.uniform(*p_range, size=batch)
    world = rng.binomial(1, 0.5, size=batch)  # 1 randomized, 0 confounded
    rows = np.empty((batch, n, 6), dtype=np.float32)
    ate = np.empty(batch, dtype=np.float32)
    for i in range(batch):
        name = "randomized_causal" if world[i] else "hidden_confounding"
        x, y, ate[i], _ = generate_world(rng, n, q[i], p[i], name, nuisance=4)
        rows[i] = np.c_[x, y]
    meta = np.c_[world, 1 - world].astype(np.float32)
    return rows, meta, ate, q, p, world


def run_raw_set(device, quick=False):
    train_steps = 600 if quick else 5000
    model_seeds = [13, 31] if quick else [13, 31, 53, 71, 97]
    train_log, predictions = [], []
    models = {False: [], True: []}
    for seed in model_seeds:
        for with_meta in [False, True]:
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            model = RawSetATE(with_meta).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
            losses = []
            t0 = time.time()
            for step in range(train_steps):
                rows, meta, ate, *_ = task_batch(rng, 96, 128, (0.01, 0.24), (0.01, 0.24))
                tr = torch.from_numpy(rows).to(device)
                tm = torch.from_numpy(meta).to(device)
                target = torch.from_numpy(ate).to(device)
                pred = model(tr, tm if with_meta else None)
                loss = torch.nn.functional.mse_loss(pred, target)
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(float(loss.detach().cpu()))
            model.eval(); models[with_meta].append(model)
            train_log.append({"model_seed": seed, "with_assumptions": with_meta, "steps": train_steps, "final_loss": np.mean(losses[-100:]), "runtime_seconds": time.time() - t0, "parameters": sum(p.numel() for p in model.parameters())})
            print(f"raw-set seed={seed} meta={with_meta} loss={train_log[-1]['final_loss']:.4f}", flush=True)

    held_cells = [(0.00, 0.30), (0.08, 0.27), (0.18, 0.22), (0.28, 0.08), (0.30, 0.30)]
    for cell, (q, p) in enumerate(held_cells):
        for eval_seed in [101, 202, 303, 404, 505]:
            rng = np.random.default_rng(100_000 + cell * 1000 + eval_seed)
            rows, meta, ate, qq, pp, world = task_batch(rng, 400, 128, (q, q + 1e-9), (p, p + 1e-9))
            tr, tm = torch.from_numpy(rows).to(device), torch.from_numpy(meta).to(device)
            for with_meta in [False, True]:
                with torch.no_grad():
                    pred = np.stack([m(tr, tm if with_meta else None).cpu().numpy() for m in models[with_meta]])
                for i in range(len(ate)):
                    predictions.append(
                        {
                            "cell": cell, "q": q, "p": p, "eval_seed": eval_seed, "task_id": i,
                            "world": "randomized_causal" if world[i] else "hidden_confounding",
                            "with_assumptions": with_meta, "true_ate": ate[i], "prediction": pred[:, i].mean(),
                            "ensemble_variance": pred[:, i].var(), "absolute_error": abs(pred[:, i].mean() - ate[i]),
                        }
                    )
    for group in models.values():
        for model in group:
            del model
    torch.cuda.empty_cache()
    return train_log, predictions


def make_figures(grid, raw):
    main = grid[grid.model == "tabpfn"]
    agg = main.groupby(["world", "n_train"]).agg(auroc=("auroc", "mean"), causal_error=("causal_absolute_error", "mean"), confidence=("mean_confidence", "mean")).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9))
    for world, color in [("randomized_causal", "#2878b5"), ("hidden_confounding", "#c44e52")]:
        q = agg[agg.world == world]
        axes[0].plot(q.n_train, q.auroc, marker="o", label=world, color=color)
        axes[1].plot(q.n_train, q.causal_error, marker="o", label=world, color=color)
        axes[2].plot(q.n_train, q.confidence, marker="o", label=world, color=color)
    for ax in axes: ax.set_xscale("log"); ax.set_xlabel("training rows")
    axes[0].set_ylabel("observational AUROC"); axes[1].set_ylabel("plug-in ATE absolute error"); axes[2].set_ylabel("predictive confidence")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "direction1_tabpfn_predictive_causal.png", dpi=180); plt.close(fig)

    rr = raw.groupby(["with_assumptions", "cell"]).absolute_error.agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for with_meta, label, color in [(False, "observations only", "#7a8793"), (True, "+ causal assumptions", "#2a9d8f")]:
        q = rr[rr.with_assumptions == with_meta]
        ax.errorbar(q.cell, q["mean"], yerr=q["std"], marker="o", capsize=3, label=label, color=color)
    ax.set(xlabel="held-out noise cell", ylabel="raw-set ATE MAE"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "direction1_rawset_assumptions.png", dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-seeds", type=int, default=15)
    parser.add_argument("--tabpfn-estimators", type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    start = time.time(); errors = []
    try:
        seeds = list(range(args.data_seeds if not args.quick else min(2, args.data_seeds)))
        grid_records = run_tabpfn_grid(seeds, 2 if args.quick else args.tabpfn_estimators, args.device)
        grid = pd.DataFrame(grid_records)
        grid.to_csv(OUT / "pretrained_model_grid.csv", index=False)
        train_log, raw_records = run_raw_set(args.device, args.quick)
        raw = pd.DataFrame(raw_records)
        raw.to_csv(OUT / "raw_set_task_metrics.csv", index=False)
        pd.DataFrame(train_log).to_csv(OUT / "raw_set_training_log.csv", index=False)
        make_figures(grid, raw)
    except Exception as exc:
        errors.append({"exception": repr(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        result = {
            "study": "Extended Direction 1 pretrained predictive-versus-causal audit",
            "parameters": vars(args),
            "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__, "torch": torch.__version__, "tabpfn": __import__("tabpfn").__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
            "noise_cells": NOISE_CELLS, "sample_sizes": SAMPLE_SIZES,
            "raw_grid_records": grid_records if "grid_records" in locals() else [],
            "raw_set_training": train_log if "train_log" in locals() else [],
            "raw_set_records": raw_records if "raw_records" in locals() else [],
            "errors": errors, "runtime_seconds": time.time() - start,
        }
        (OUT / "results.json").write_text(json.dumps(jsonify(result), indent=2, allow_nan=False) + "\n")
        print(f"direction1 complete in {(time.time()-start)/60:.1f} minutes; errors={len(errors)}", flush=True)


if __name__ == "__main__":
    main()
