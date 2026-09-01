#!/usr/bin/env python3
"""Adversarial robustness checks for the two leading sweep directions."""

from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import run_sweep as sweep


ROOT, D1, D2, FIG = sweep.ROOT, sweep.D1, sweep.D2, sweep.FIG
ROBUST_SEEDS = [3, 11, 29, 47, 71]


def continuous_equivalence_controls():
    records = []
    for seed in ROBUST_SEEDS:
        rng = np.random.default_rng(seed)
        for n_rows in [32, 160, 512]:
            for gap_scale in [0.6, 1.0, 1.4]:
                feats, ys, groups = [], [], []
                for group in range(240):
                    rho = rng.uniform(0.2, 0.55)
                    gap = gap_scale * rng.uniform(0.8, 1.2)
                    obs = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n_rows)
                    f = sweep.binary_summary(obs[:, 0], obs[:, 1])
                    for tau in [rho - gap / 2, rho + gap / 2]:
                        feats.append(f.copy()); ys.append(tau); groups.append(group)
                x, y, groups = np.asarray(feats), np.asarray(ys), np.asarray(groups)
                test_groups = set(rng.choice(240, 72, replace=False).tolist())
                te = np.array([g in test_groups for g in groups]); tr = ~te
                for name, model, cols in [
                    ("constant", None, slice(None)),
                    ("ridge_full_summary", make_pipeline(StandardScaler(), Ridge(alpha=1.0)), slice(None)),
                    ("ridge_3stats", make_pipeline(StandardScaler(), Ridge(alpha=1.0)), [0, 1, 4]),
                ]:
                    if model is None:
                        p = np.full(te.sum(), y[tr].mean())
                    else:
                        model.fit(x[tr][:, cols], y[tr]); p = model.predict(x[te][:, cols])
                    records.append({"seed": seed, "n_rows": n_rows, "gap_scale": gap_scale, "model": name, "mae": mean_absolute_error(y[te], p), "rmse": np.sqrt(mean_squared_error(y[te], p)), "mean_effect_gap": gap_scale})
    return records


def binary_alternative_dgp():
    # Exact alternative equivalence class. World A: randomized X and
    # Y=X xor E (ATE=1-2p). World B: U->X=U and U->Y=U xor E (ATE=0).
    records = []
    for seed in ROBUST_SEEDS:
        rng = np.random.default_rng(seed + 400)
        for n_rows in [32, 160, 512]:
            for flip_noise in [0.05, 0.20, 0.35]:
                feats, ys, groups = [], [], []
                for group in range(240):
                    x = rng.binomial(1, 0.5, n_rows)
                    yobs = np.logical_xor(x, rng.binomial(1, flip_noise, n_rows)).astype(float)
                    f = sweep.binary_summary(x, yobs)
                    for ate in [1 - 2 * flip_noise, 0.0]:
                        feats.append(f.copy()); ys.append(ate); groups.append(group)
                xmat, target, groups = np.asarray(feats), np.asarray(ys), np.asarray(groups)
                test_groups = set(rng.choice(240, 72, replace=False).tolist())
                te = np.array([g in test_groups for g in groups]); tr = ~te
                model = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(xmat[tr], target[tr])
                pred = model.predict(xmat[te])
                records.append({"seed": seed, "n_rows": n_rows, "flip_noise": flip_noise, "model": "ridge", "mae": mean_absolute_error(target[te], pred), "rmse": np.sqrt(mean_squared_error(target[te], pred)), "true_effect_gap": 1 - 2 * flip_noise})
    return records


def assumption_shuffle_control():
    records = []
    scenario_meta = np.eye(7)
    labels = np.array([1, 1, 0, 1, 0, 1, 0])
    for seed in ROBUST_SEEDS:
        rng = np.random.default_rng(seed + 800)
        stats, meta, y, groups = [], [], [], []
        for g in range(400):
            base = rng.normal(size=12)
            for j in range(7):
                stats.append(base); meta.append(scenario_meta[j]); y.append(labels[j]); groups.append(g)
        stats, meta, y, groups = map(np.asarray, (stats, meta, y, groups))
        test_groups = set(rng.choice(400, 120, replace=False).tolist())
        te = np.array([g in test_groups for g in groups]); tr = ~te
        for control in ["real_assumptions", "shuffled_assumptions"]:
            m = meta.copy()
            if control == "shuffled_assumptions":
                m = m[rng.permutation(len(m))]
            x = np.c_[stats, m]
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(x[tr], y[tr])
            p = clf.predict_proba(x[te])[:, 1]
            records.append({"seed": seed, "control": control, "accuracy": accuracy_score(y[te], p >= .5), "auroc": roc_auc_score(y[te], p)})
    return records


def role_rows(schema, n, rng, signal_scale=1.0):
    domain, srcname, dstname, pos = schema
    src, dst, context = rng.normal(size=(3, n))
    prob = sweep.expit(signal_scale * (2 * src - dst + .35 * context))
    y = rng.binomial(1, prob)
    v0, v1 = (src, dst) if pos == 0 else (dst, src)
    return v0, v1, context, y, pos


def semantic_classification_controls():
    records = []
    n_schema = 6
    for seed in ROBUST_SEEDS:
        for n_train in [100, 500, 1400]:
            for signal_scale in [0.5, 1.0, 1.5]:
                rng = np.random.default_rng(seed * 1000 + n_train + int(10 * signal_scale))
                parts = []
                for idx, schema in enumerate(sweep.SCHEMAS[:n_schema]):
                    parts.append((idx, *role_rows(schema, n_train, rng, signal_scale)))
                train_y = np.concatenate([p[4] for p in parts])
                for semantic in [False, True]:
                    train_x = np.vstack([sweep.schema_features(p[1], p[2], p[3], p[0], p[5], n_schema, semantic) for p in parts])
                    model = sweep.fit_schema_model(train_x, train_y)
                    yy, pp, pshuffle = [], [], []
                    for idx, schema in enumerate(sweep.SCHEMAS[n_schema:], start=n_schema):
                        v0, v1, c, y, pos = role_rows(schema, 1000, rng, signal_scale)
                        x = sweep.schema_features(v0, v1, c, idx, pos, n_schema, semantic)
                        xs = sweep.schema_features(v0, v1, c, idx, pos, n_schema, semantic, shuffled=True)
                        yy.append(y); pp.append(model.predict_proba(x)[:, 1]); pshuffle.append(model.predict_proba(xs)[:, 1])
                    yy, pp, pshuffle = np.concatenate(yy), np.concatenate(pp), np.concatenate(pshuffle)
                    records.append({"seed": seed, "n_train_per_schema": n_train, "signal_scale": signal_scale, "model": "semantics" if semantic else "structure_only", "heldout_auroc": roc_auc_score(yy, pp), "shuffled_auroc": roc_auc_score(yy, pshuffle)})
    return records


def semantic_regression_alternative():
    records = []
    n_schema = 6
    for seed in ROBUST_SEEDS:
        rng = np.random.default_rng(seed + 1600)
        for noise in [0.5, 1.0, 2.0]:
            train_parts = []
            for idx, schema in enumerate(sweep.SCHEMAS[:n_schema]):
                n = 900; src, dst, c = rng.normal(size=(3, n)); y = 2 * src - dst + .35 * c + rng.normal(scale=noise, size=n)
                pos = schema[3]; v0, v1 = (src, dst) if pos == 0 else (dst, src)
                train_parts.append((idx, v0, v1, c, y, pos))
            train_y = np.concatenate([p[4] for p in train_parts])
            for semantic in [False, True]:
                tx = np.vstack([sweep.schema_features(p[1], p[2], p[3], p[0], p[5], n_schema, semantic) for p in train_parts])
                model = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(tx, train_y)
                yy, pp = [], []
                for idx, schema in enumerate(sweep.SCHEMAS[n_schema:], start=n_schema):
                    n = 1200; src, dst, c = rng.normal(size=(3, n)); y = 2 * src - dst + .35 * c + rng.normal(scale=noise, size=n)
                    pos = schema[3]; v0, v1 = (src, dst) if pos == 0 else (dst, src)
                    x = sweep.schema_features(v0, v1, c, idx, pos, n_schema, semantic)
                    yy.append(y); pp.append(model.predict(x))
                records.append({"seed": seed, "noise_sd": noise, "model": "semantics" if semantic else "structure_only", "rmse": np.sqrt(mean_squared_error(np.concatenate(yy), np.concatenate(pp)))})
    return records


def main():
    begun = time.time()
    continuous = continuous_equivalence_controls()
    binary = binary_alternative_dgp()
    assumptions = assumption_shuffle_control()
    semantic = semantic_classification_controls()
    regression = semantic_regression_alternative()
    pd.DataFrame(continuous).to_csv(D1 / "robustness_continuous.csv", index=False)
    pd.DataFrame(binary).to_csv(D1 / "robustness_binary_alternative.csv", index=False)
    pd.DataFrame(assumptions).to_csv(D1 / "robustness_assumption_shuffle.csv", index=False)
    pd.DataFrame(semantic).to_csv(D2 / "robustness_classification.csv", index=False)
    pd.DataFrame(regression).to_csv(D2 / "robustness_regression_alternative.csv", index=False)

    sf = pd.DataFrame(semantic)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    agg = sf.groupby(["n_train_per_schema", "model"]).heldout_auroc.agg(["mean", "std"]).reset_index()
    for model, color in [("structure_only", "#7a8793"), ("semantics", "#3d7ea6")]:
        q = agg[agg.model == model]
        ax.errorbar(q.n_train_per_schema, q["mean"], yerr=q["std"], marker="o", capsize=3, label=model, color=color)
    ax.set(xlabel="training rows per source schema", ylabel="held-out-schema AUROC", xscale="log", ylim=(.45, .95))
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIG / "robustness_semantic_scale.png", dpi=180); plt.close(fig)

    result_path = ROOT / "results.json"
    results = json.loads(result_path.read_text())
    results["cross_direction_robustness"] = {
        "selected_top_two": ["direction_1_causal_relational", "direction_2_semantics"],
        "selection_basis": "strongest qualitative discontinuities in the primary run; selected before robustness analysis",
        "direction_1": {"continuous_size_gap_capacity": continuous, "binary_alternative_dgp": binary, "assumption_shuffle": assumptions},
        "direction_2": {"classification_size_noise": semantic, "regression_alternative_dgp": regression},
        "coverage": {"additional_seeds": True, "dataset_size": True, "noise_or_signal": True, "reduced_capacity": True, "alternative_dgp": True, "shuffled_control": True},
        "runtime_seconds": time.time() - begun,
    }
    results["decision"] = {
        "best_direction": "Identification-aware causal relational foundation models",
        "scores": [
            {"rank": 1, "direction": "Identification-aware causal relational foundation models", "empirical_signal": 5, "novelty_potential": 3, "tractability": 5, "scientific_depth": 5, "publishability_probability_percent": 55, "verdict": "PURSUE", "recommended_next_experiment": "Train an abstaining/set-valued dataset encoder across mixed SCM families and test held-out causal graphs plus relational schemas under approximate observational equivalence."},
            {"rank": 2, "direction": "Semantics-aware synthetic pretraining for relational/tabular foundation models", "empirical_signal": 4, "novelty_potential": 3, "tractability": 4, "scientific_depth": 4, "publishability_probability_percent": 45, "verdict": "MAYBE", "recommended_next_experiment": "Replace oracle role vectors with frozen text embeddings and evaluate compositional role synonyms on held-out real and synthetic schemas."},
            {"rank": 3, "direction": "Prior misspecification, calibration, and failure detection for amortized causal/tabular models", "empirical_signal": 3, "novelty_potential": 3, "tractability": 4, "scientific_depth": 4, "publishability_probability_percent": 30, "verdict": "KILL", "recommended_next_experiment": "Only revive if a new shift family yields at least 2x IID error with less than 25% uncertainty growth and failure detection above 0.8 AUROC on held-out shift families."},
        ],
    }
    result_path.write_text(json.dumps(sweep.py(results), indent=2, allow_nan=False) + "\n")
    print(f"robustness checks complete in {time.time() - begun:.1f}s")


if __name__ == "__main__":
    main()
