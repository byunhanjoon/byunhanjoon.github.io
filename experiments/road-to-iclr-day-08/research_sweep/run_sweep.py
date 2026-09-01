#!/usr/bin/env python3
"""CPU-feasible, reproducible sweep for three next-generation tabular-AI ideas.

The script intentionally favors controlled synthetic counterexamples over benchmark
tuning.  It writes raw records, compact summaries, and figures below this folder.
"""

from __future__ import annotations

import argparse
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
import scipy
import sklearn
import torch
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
D1 = ROOT / "direction_1_causal_relational"
D2 = ROOT / "direction_2_semantics"
D3 = ROOT / "direction_3_prior_shift"
SEEDS = [11, 29, 47]


def py(v):
    """Convert numpy values recursively for strict JSON output."""
    if isinstance(v, dict):
        return {str(k): py(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [py(x) for x in v]
    if isinstance(v, np.ndarray):
        return py(v.tolist())
    if isinstance(v, (np.floating, float)):
        x = float(v)
        return x if math.isfinite(x) else None
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    return v


def summarize(values):
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "n": int(len(a))}


def binary_summary(x, y):
    eps = 1e-12
    vx, vy = np.var(x), np.var(y)
    cov = np.mean((x - x.mean()) * (y - y.mean()))
    corr = cov / math.sqrt(max(vx * vy, eps))
    slope = cov / max(vx, eps)
    return np.array(
        [
            x.mean(), y.mean(), vx, vy, cov, corr, slope,
            *np.quantile(x, [0.1, 0.25, 0.5, 0.75, 0.9]),
            *np.quantile(y, [0.1, 0.25, 0.5, 0.75, 0.9]),
        ],
        dtype=float,
    )


def direction1(quick=False):
    started = time.time()
    n_pairs = 180 if quick else 700
    n_rows = 96 if quick else 160
    rng = np.random.default_rng(20260831)
    features, targets, groups, pair_rows = [], [], [], []

    # Each pair reuses the exact same observed sample under two valid latent SCM
    # parameterizations.  Hence observational distance is exactly zero by design.
    for pair_id in range(n_pairs):
        rho = rng.uniform(0.25, 0.55)
        gap = rng.uniform(0.8, 1.4)
        tau_lo, tau_hi = rho - gap / 2, rho + gap / 2
        obs = rng.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]], size=n_rows)
        feat = binary_summary(obs[:, 0], obs[:, 1])
        for world, tau in [("low_effect", tau_lo), ("high_effect", tau_hi)]:
            # Explicit latent linear-Gaussian realization:
            # U~N(0,1), X=.8U+sqrt(1-.8^2)eX,
            # Y=tau*X+gamma*U+sigma_y*eY.
            # gamma and sigma_y below reproduce Var(X)=Var(Y)=1,
            # Cov(X,Y)=rho while the intervention effect remains tau.
            loading = 0.8
            gamma = (rho - tau) / loading
            residual_var_y = 1.0 - tau * tau - gamma * gamma - 2 * tau * gamma * loading
            features.append(feat.copy())
            targets.append(tau)
            groups.append(pair_id)
            pair_rows.append(
                {
                    "generator_seed": 20260831,
                    "pair_id": pair_id,
                    "world": world,
                    "rho": rho,
                    "true_ate": tau,
                    "effect_gap": gap,
                    "observational_distance": 0.0,
                    "latent_loading": loading,
                    "confounder_to_outcome": gamma,
                    "outcome_residual_variance": residual_var_y,
                    "n_rows": n_rows,
                }
            )
    X, y, groups = np.asarray(features), np.asarray(targets), np.asarray(groups)
    order = rng.permutation(n_pairs)
    train_groups = set(order[: int(0.7 * n_pairs)].tolist())
    train = np.array([g in train_groups for g in groups])
    test = ~train
    models = {
        "mean_stupid": None,
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "random_forest": RandomForestRegressor(n_estimators=180 if not quick else 60, min_samples_leaf=8, n_jobs=-1, random_state=11),
        "small_mlp": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-3, max_iter=350, early_stopping=True, random_state=11)),
    }
    model_records = []
    predictions = {}
    for name, model in models.items():
        t0 = time.time()
        if model is None:
            pred = np.full(test.sum(), y[train].mean())
        else:
            model.fit(X[train], y[train])
            pred = model.predict(X[test])
        predictions[name] = pred
        model_records.append(
            {
                "model": name,
                "mae": mean_absolute_error(y[test], pred),
                "rmse": math.sqrt(mean_squared_error(y[test], pred)),
                "r2": r2_score(y[test], pred),
                "runtime_seconds": time.time() - t0,
            }
        )

    # Across-seed MLP agreement is a cheap epistemic-confidence proxy.
    ensemble_preds = []
    for seed in SEEDS:
        m = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-3, max_iter=350, early_stopping=True, random_state=seed))
        m.fit(X[train], y[train])
        ensemble_preds.append(m.predict(X[test]))
    ensemble_preds = np.asarray(ensemble_preds)
    d1_conf = {
        "ensemble_prediction_variance_mean": float(np.var(ensemble_preds, axis=0).mean()),
        "ensemble_mae": float(mean_absolute_error(y[test], ensemble_preds.mean(axis=0))),
        "irreducible_pairwise_mae_lower_bound": float(np.mean([r["effect_gap"] for r in pair_rows]) / 2),
    }

    # 1B: exact observational equivalence blocks. Assumption scenarios—not hidden
    # correlations—determine identification. Group splitting prevents pair leakage.
    scenarios = [
        ("randomized", [1, 0, 0, 0, 0, 0, 0], 1),
        ("observed_confounding", [0, 1, 0, 0, 0, 0, 0], 1),
        ("hidden_confounding", [0, 0, 0, 0, 0, 0, 0], 0),
        ("valid_iv", [0, 0, 1, 0, 0, 0, 0], 1),
        ("invalid_iv", [0, 0, 0, 1, 0, 0, 0], 0),
        ("mediator_observed", [1, 0, 0, 0, 1, 0, 0], 1),
        ("collider_conditioned", [0, 0, 0, 0, 0, 1, 1], 0),
    ]
    n_blocks = 180 if quick else 850
    stats, meta, ident_y, ident_groups, ident_scenario = [], [], [], [], []
    for block in range(n_blocks):
        z = rng.normal(size=(n_rows, 3))
        t = (z[:, 0] + 0.35 * z[:, 1] + rng.normal(size=n_rows) > 0).astype(float)
        out = 0.45 * t + 0.5 * z[:, 0] + rng.normal(size=n_rows)
        s = np.r_[binary_summary(z[:, 0], t), binary_summary(t, out)]
        for scenario, m, label in scenarios:
            stats.append(s.copy())
            meta.append(m)
            ident_y.append(label)
            ident_groups.append(block)
            ident_scenario.append(scenario)
    stats, meta, ident_y, ident_groups = map(np.asarray, (stats, meta, ident_y, ident_groups))
    order = rng.permutation(n_blocks)
    tr_groups = set(order[: int(0.7 * n_blocks)].tolist())
    tr = np.array([g in tr_groups for g in ident_groups])
    te = ~tr
    ident_records = []
    for feature_set, xmat in [("observational_only", stats), ("observational_plus_assumptions", np.c_[stats, meta])]:
        classifiers = {
            "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)),
            "random_forest": RandomForestClassifier(n_estimators=160 if not quick else 60, min_samples_leaf=4, n_jobs=-1, random_state=11),
            "small_mlp": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, early_stopping=True, random_state=11)),
        }
        for name, clf in classifiers.items():
            clf.fit(xmat[tr], ident_y[tr])
            prob = clf.predict_proba(xmat[te])[:, 1]
            ident_records.append(
                {
                    "feature_set": feature_set,
                    "model": name,
                    "accuracy": accuracy_score(ident_y[te], prob >= 0.5),
                    "auroc": roc_auc_score(ident_y[te], prob),
                    "log_loss": log_loss(ident_y[te], prob),
                }
            )
    majority = max(ident_y[tr].mean(), 1 - ident_y[tr].mean())
    ident_records.append({"feature_set": "stupid_majority", "model": "constant", "accuracy": majority, "auroc": 0.5, "log_loss": log_loss(ident_y[te], np.full(te.sum(), ident_y[tr].mean()))})

    pd.DataFrame(pair_rows).to_csv(D1 / "observational_equivalence_pairs.csv", index=False)
    pd.DataFrame(model_records).to_csv(D1 / "ate_prediction_models.csv", index=False)
    pd.DataFrame(ident_records).to_csv(D1 / "identifiability_models.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    gaps = np.asarray([r["effect_gap"] for r in pair_rows[::2]])
    ax.scatter(np.zeros_like(gaps), gaps, s=12, alpha=0.35, color="#b23a48")
    ax.set_xlim(-0.01, 0.2)
    ax.set_xlabel("distance between observational summaries")
    ax.set_ylabel("difference in true ATE")
    ax.set_title("Exact observational equivalence, incompatible causal effects")
    fig.tight_layout()
    fig.savefig(FIG / "direction1_observational_equivalence.png", dpi=180)
    plt.close(fig)

    idf = pd.DataFrame(ident_records)
    plot = idf[idf.model == "logistic"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar(["observations only", "+ assumptions"], plot.accuracy, color=["#7a8793", "#207561"])
    ax.axhline(majority, color="black", ls="--", lw=1, label="majority")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("identifiability accuracy")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "direction1_assumptions.png", dpi=180)
    plt.close(fig)

    return {
        "runtime_seconds": time.time() - started,
        "parameters": {"generator_seed": 20260831, "n_equivalence_pairs": n_pairs, "rows_per_dataset": n_rows, "n_identifiability_blocks": n_blocks, "model_seeds": SEEDS},
        "raw_equivalence_pairs": pair_rows,
        "ate_prediction": {"models": model_records, "confidence": d1_conf},
        "identifiability": {"models": ident_records, "scenarios": [x[0] for x in scenarios], "group_split": True},
        "leakage_checks": {"latent_confounder_exposed": False, "equivalence_pair_kept_within_split": True, "identical_summary_blocks_kept_within_split": True},
    }


SCHEMAS = [
    ("commerce", "buyer", "seller", 0),
    ("messaging", "sender", "receiver", 1),
    ("payments", "payer", "payee", 0),
    ("citations", "author", "reader", 1),
    ("shipping", "shipper", "recipient", 0),
    ("social", "follower", "followed", 1),
    ("medicine", "doctor", "patient", 1),
    ("credit", "lender", "borrower", 0),
    ("education", "teacher", "student", 1),
    ("legal", "plaintiff", "defendant", 0),
]


def relational_rows(schema, n, rng, force_reverse=False):
    domain, source_name, dest_name, source_pos = schema
    if force_reverse:
        source_pos = 1 - source_pos
    source = rng.normal(size=n)
    dest = rng.normal(size=n)
    context = rng.normal(size=n)
    logit = 2.0 * source - 1.0 * dest + 0.35 * context
    prob = expit(logit)
    y = rng.binomial(1, prob)
    v0, v1 = (source, dest) if source_pos == 0 else (dest, source)
    return v0, v1, context, y, source_pos


def schema_features(v0, v1, context, schema_index, source_pos, n_train_schemas, semantic, shuffled=False):
    n = len(v0)
    ids = np.zeros((n, n_train_schemas))
    if schema_index < n_train_schemas:
        ids[:, schema_index] = 1.0
    if semantic:
        pos = 1 - source_pos if shuffled else source_pos
        source_value = v0 if pos == 0 else v1
        dest_value = v1 if pos == 0 else v0
        role0 = np.full(n, 1.0 if pos == 0 else -1.0)
    else:
        source_value = np.zeros(n)
        dest_value = np.zeros(n)
        role0 = np.zeros(n)
    return np.c_[v0, v1, context, ids, ids * v0[:, None], ids * v1[:, None], source_value, dest_value, role0]


def fit_schema_model(x, y):
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=600)).fit(x, y)


def cls_metrics(y, p):
    return {"auroc": roc_auc_score(y, p), "accuracy": accuracy_score(y, p >= 0.5), "log_loss": log_loss(y, p)}


def direction2(quick=False):
    started = time.time()
    n_train_schema = 6
    n_train = 450 if quick else 1400
    n_eval = 500 if quick else 1800
    all_records, few_records, paired_control = [], [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        train_parts = []
        for idx, schema in enumerate(SCHEMAS[:n_train_schema]):
            v0, v1, c, y, pos = relational_rows(schema, n_train, rng)
            train_parts.append((idx, v0, v1, c, y, pos))
        train_y = np.concatenate([p[4] for p in train_parts])
        models = {}
        for semantic in [False, True]:
            x = np.vstack([schema_features(p[1], p[2], p[3], p[0], p[5], n_train_schema, semantic) for p in train_parts])
            models[semantic] = fit_schema_model(x, train_y)

        # Same-schema holdout uses fresh rows, ensuring both methods can exploit
        # memorized arbitrary schema IDs without row leakage.
        for semantic in [False, True]:
            ys, ps = [], []
            for idx, schema in enumerate(SCHEMAS[:n_train_schema]):
                v0, v1, c, y, pos = relational_rows(schema, n_eval // 3, rng)
                x = schema_features(v0, v1, c, idx, pos, n_train_schema, semantic)
                ys.append(y); ps.append(models[semantic].predict_proba(x)[:, 1])
            all_records.append({"seed": seed, "split": "same_schema", "model": "semantics" if semantic else "structure_only", **cls_metrics(np.concatenate(ys), np.concatenate(ps))})

        held_cache = []
        for idx, schema in enumerate(SCHEMAS[n_train_schema:], start=n_train_schema):
            v0, v1, c, y, pos = relational_rows(schema, n_eval, rng)
            held_cache.append((idx, schema, v0, v1, c, y, pos))
        for semantic in [False, True]:
            ys, normal, shuffled = [], [], []
            for idx, schema, v0, v1, c, y, pos in held_cache:
                x = schema_features(v0, v1, c, idx, pos, n_train_schema, semantic)
                xs = schema_features(v0, v1, c, idx, pos, n_train_schema, semantic, shuffled=True)
                ys.append(y)
                normal.append(models[semantic].predict_proba(x)[:, 1])
                shuffled.append(models[semantic].predict_proba(xs)[:, 1])
            yy, pp, psh = np.concatenate(ys), np.concatenate(normal), np.concatenate(shuffled)
            label = "semantics" if semantic else "structure_only"
            all_records.append({"seed": seed, "split": "heldout_schema", "model": label, **cls_metrics(yy, pp)})
            all_records.append({"seed": seed, "split": "shuffled_semantics", "model": label, **cls_metrics(yy, psh)})

        # Paired role reversal: use identical latent source/destination draws and
        # physically reverse the two fields. Semantic canonicalization should be
        # prediction invariant; a positional representation should not be.
        for semantic in [False, True]:
            ys, pnormal, prev, deltas = [], [], [], []
            for idx, schema, v0, v1, c, y, pos in held_cache:
                xn = schema_features(v0, v1, c, idx, pos, n_train_schema, semantic)
                xr = schema_features(v1, v0, c, idx, 1 - pos, n_train_schema, semantic)
                pn = models[semantic].predict_proba(xn)[:, 1]
                pr = models[semantic].predict_proba(xr)[:, 1]
                ys.append(y); pnormal.append(pn); prev.append(pr); deltas.append(np.abs(pn - pr))
            yy, pr = np.concatenate(ys), np.concatenate(prev)
            label = "semantics" if semantic else "structure_only"
            all_records.append({"seed": seed, "split": "role_reversal", "model": label, **cls_metrics(yy, pr)})
            paired_control.append({"seed": seed, "model": label, "mean_absolute_prediction_change": np.concatenate(deltas).mean()})

        # Target-only few-shot adaptation on medicine. Repeated subsamples reveal
        # whether 10–100 labels erase the zero-shot semantic advantage.
        schema = SCHEMAS[n_train_schema]
        v0p, v1p, cp, yp, posp = relational_rows(schema, 500 if quick else 2200, rng)
        v0e, v1e, ce, ye, pose = relational_rows(schema, n_eval, rng)
        for shots in [0, 10, 50, 100]:
            for semantic in [False, True]:
                label = "semantics" if semantic else "structure_only"
                xe = schema_features(v0e, v1e, ce, n_train_schema, pose, n_train_schema, semantic)
                if shots == 0:
                    p = models[semantic].predict_proba(xe)[:, 1]
                else:
                    take = rng.choice(len(yp), shots, replace=False)
                    xf = schema_features(v0p[take], v1p[take], cp[take], n_train_schema, posp, n_train_schema, semantic)
                    # Avoid degenerate tiny samples while recording failures.
                    if len(np.unique(yp[take])) < 2:
                        p = np.full(len(ye), yp[take].mean())
                    else:
                        p = fit_schema_model(xf, yp[take]).predict_proba(xe)[:, 1]
                few_records.append({"seed": seed, "shots": shots, "model": label, **cls_metrics(ye, p)})

    perf = pd.DataFrame(all_records)
    few = pd.DataFrame(few_records)
    controls = pd.DataFrame(paired_control)
    perf.to_csv(D2 / "schema_transfer_metrics.csv", index=False)
    few.to_csv(D2 / "few_shot_metrics.csv", index=False)
    controls.to_csv(D2 / "role_reversal_invariance.csv", index=False)

    means = perf.groupby(["split", "model"]).auroc.mean().unstack()
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    order = ["same_schema", "heldout_schema", "shuffled_semantics", "role_reversal"]
    xx = np.arange(len(order)); width = 0.36
    ax.bar(xx - width / 2, [means.loc[o, "structure_only"] for o in order], width, label="structure only", color="#7a8793")
    ax.bar(xx + width / 2, [means.loc[o, "semantics"] for o in order], width, label="semantic roles", color="#3d7ea6")
    ax.set_xticks(xx, [x.replace("_", "\n") for x in order])
    ax.set_ylim(0.2, 1.0); ax.set_ylabel("AUROC"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "direction2_schema_transfer.png", dpi=180); plt.close(fig)

    fm = few.groupby(["shots", "model"]).auroc.agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for label, color in [("structure_only", "#7a8793"), ("semantics", "#3d7ea6")]:
        q = fm[fm.model == label]
        ax.errorbar(q.shots, q["mean"], yerr=q["std"], marker="o", capsize=3, label=label, color=color)
    ax.set_xlabel("labeled rows in unseen schema"); ax.set_ylabel("AUROC"); ax.set_ylim(0.4, 1.0); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "direction2_few_shot.png", dpi=180); plt.close(fig)

    return {
        "runtime_seconds": time.time() - started,
        "parameters": {"train_schemas": [s[0] for s in SCHEMAS[:n_train_schema]], "heldout_schemas": [s[0] for s in SCHEMAS[n_train_schema:]], "rows_per_train_schema": n_train, "rows_per_eval_schema": n_eval, "seeds": SEEDS, "feature_dimension_each_model": 24, "downstream_logistic_parameters_each_model": 25, "semantic_embedding": "manual source/destination role vector (specified fallback)"},
        "raw_metrics": all_records,
        "few_shot": few_records,
        "role_reversal_invariance": paired_control,
        "leakage_checks": {"schema_names_disjoint": True, "heldout_schema_ids_unseen": True, "fresh_same_schema_rows": True, "shuffled_semantics_control": True},
    }


class DeepSetATE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.row = torch.nn.Sequential(torch.nn.Linear(3, 32), torch.nn.ReLU(), torch.nn.Linear(32, 32), torch.nn.ReLU())
        self.head = torch.nn.Sequential(torch.nn.Linear(32, 24), torch.nn.ReLU(), torch.nn.Linear(24, 1))

    def forward(self, x):
        return self.head(self.row(x).mean(dim=1)).squeeze(-1)


def generate_tasks(rng, batch, n, shift="iid", severity=0.0):
    x = rng.normal(size=(batch, n))
    u = rng.normal(size=(batch, n))
    tau = rng.uniform(-2, 2, size=batch)
    beta = rng.uniform(-2, 2, size=batch)
    a = rng.uniform(-1.5, 1.5, size=batch)
    sev = float(severity)
    if shift in {"covariate", "combined"}:
        heavy = rng.standard_t(df=3, size=(batch, n)) / math.sqrt(3)
        x = (x + sev * heavy) / math.sqrt(1 + sev * sev)
    logits = a[:, None] * x
    if shift in {"treatment", "combined"}:
        logits += 1.25 * sev * (x * x - 1.0)
    if shift in {"hidden_confounding", "combined"}:
        logits += 1.25 * sev * u
    t = rng.binomial(1, expit(np.clip(logits, -12, 12))).astype(float)
    base = beta[:, None] * x
    if shift in {"outcome_nonlinearity", "combined"}:
        base = beta[:, None] * ((1 - min(sev, 1.0)) * x + min(sev, 1.0) * np.sin(2 * x))
    effect = tau[:, None]
    if shift in {"heterogeneous_effect", "combined"}:
        effect = effect + sev * x
    y = effect * t + base + rng.normal(scale=0.7, size=(batch, n))
    if shift in {"hidden_confounding", "combined"}:
        y += 1.25 * sev * u
    rows = np.stack([x, t, y], axis=-1).astype(np.float32)
    return rows, tau.astype(np.float32)


def dataset_summaries(rows):
    x, t, y = rows[:, :, 0], rows[:, :, 1], rows[:, :, 2]
    out = []
    for i in range(len(rows)):
        xi, ti, yi = x[i], t[i], y[i]
        def moments(z):
            sd = np.std(z) + 1e-8
            zz = (z - np.mean(z)) / sd
            return [np.mean(z), np.var(z), np.mean(zz ** 3), np.mean(zz ** 4), *np.quantile(z, [0.1, 0.5, 0.9])]
        corr = np.corrcoef(np.c_[xi, ti, yi], rowvar=False)[np.triu_indices(3, 1)]
        design = np.c_[np.ones(len(xi)), ti, xi]
        coef = np.linalg.lstsq(design, yi, rcond=None)[0]
        treated = ti > 0.5
        diff = yi[treated].mean() - yi[~treated].mean() if treated.any() and (~treated).any() else 0.0
        out.append([*moments(xi), *moments(yi), ti.mean(), *corr, *coef, diff])
    return np.asarray(out)


def predict_models(models, rows):
    z = torch.from_numpy(rows)
    with torch.no_grad():
        return np.stack([m(z).numpy() for m in models])


def direction3(quick=False):
    started = time.time()
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    steps = 90 if quick else 360
    batch, n_train = (32, 64) if quick else (64, 96)
    models, training = [], []
    for seed in SEEDS:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = DeepSetATE()
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
        losses = []
        t0 = time.time()
        for step in range(steps):
            rows, tau = generate_tasks(rng, batch, n_train)
            pred = model(torch.from_numpy(rows))
            loss = torch.nn.functional.mse_loss(pred, torch.from_numpy(tau))
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.detach()))
        model.eval(); models.append(model)
        training.append({"seed": seed, "steps": steps, "final_loss": np.mean(losses[-20:]), "runtime_seconds": time.time() - t0, "parameters": sum(p.numel() for p in model.parameters())})

    detector_rng = np.random.default_rng(9001)
    det_rows, _ = generate_tasks(detector_rng, 500 if quick else 2200, n_train)
    det_stats = dataset_summaries(det_rows)
    scaler = StandardScaler().fit(det_stats)
    detector = IsolationForest(n_estimators=160 if not quick else 60, contamination="auto", random_state=11, n_jobs=-1).fit(scaler.transform(det_stats))

    # Separate IID calibration tasks: interval scaling and bad-error threshold are
    # frozen before shifted evaluation.
    cal_rows, cal_tau = generate_tasks(np.random.default_rng(777), 180 if quick else 600, n_train)
    cal_pred = predict_models(models, cal_rows)
    cal_mean, cal_sd = cal_pred.mean(0), cal_pred.std(0) + 1e-6
    interval_scale = float(np.quantile(np.abs(cal_mean - cal_tau) / cal_sd, 0.90))
    bad_threshold = float(np.quantile(np.abs(cal_mean - cal_tau), 0.90))
    iid_ood = -detector.score_samples(scaler.transform(dataset_summaries(cal_rows)))
    iid_unc = np.var(cal_pred, axis=0)
    ood_center, ood_scale = iid_ood.mean(), iid_ood.std() + 1e-8
    unc_center, unc_scale = iid_unc.mean(), iid_unc.std() + 1e-8

    conditions = [("iid", 0.0)]
    for shift in ["covariate", "treatment", "outcome_nonlinearity", "heterogeneous_effect", "hidden_confounding", "combined"]:
        for severity in [0.5, 1.0, 1.5]:
            conditions.append((shift, severity))
    tasks_per = 70 if quick else 220
    records, seed_metrics = [], []
    for ci, (shift, severity) in enumerate(conditions):
        rows, tau = generate_tasks(np.random.default_rng(12000 + ci), tasks_per, n_train, shift, severity)
        preds = predict_models(models, rows)
        pm, pv = preds.mean(0), preds.var(0)
        ood = -detector.score_samples(scaler.transform(dataset_summaries(rows)))
        # Stupid causal baseline: unadjusted treated-minus-control difference.
        naive = []
        for rr in rows:
            tt, yy = rr[:, 1] > 0.5, rr[:, 2]
            naive.append(yy[tt].mean() - yy[~tt].mean() if tt.any() and (~tt).any() else 0.0)
        naive = np.asarray(naive)
        for i in range(tasks_per):
            records.append(
                {
                    "generation_seed": 12000 + ci, "shift": shift, "severity": severity, "task_id": i, "true_ate": tau[i],
                    "ensemble_prediction": pm[i], "absolute_error": abs(pm[i] - tau[i]), "squared_error": (pm[i] - tau[i]) ** 2,
                    "ensemble_variance": pv[i], "ood_score": ood[i], "naive_prediction": naive[i], "naive_absolute_error": abs(naive[i] - tau[i]),
                    "covered_90": abs(pm[i] - tau[i]) <= interval_scale * math.sqrt(pv[i] + 1e-12),
                }
            )
        for mi, seed in enumerate(SEEDS):
            seed_metrics.append({"seed": seed, "shift": shift, "severity": severity, "mae": mean_absolute_error(tau, preds[mi]), "rmse": math.sqrt(mean_squared_error(tau, preds[mi])), "prediction_variance": np.var(preds[mi])})

    rdf = pd.DataFrame(records)
    rdf["bad_error"] = rdf.absolute_error > bad_threshold
    rdf["combined_score"] = (rdf.ood_score - ood_center) / ood_scale + (rdf.ensemble_variance - unc_center) / unc_scale
    detection = {}
    for score in ["ensemble_variance", "ood_score", "combined_score"]:
        detection[score] = roc_auc_score(rdf.bad_error, rdf[score])
    shifted = rdf[rdf["shift"] != "iid"]
    rho, pvalue = spearmanr(shifted.ood_score, shifted.absolute_error)
    urho, up = spearmanr(shifted.ensemble_variance, shifted.absolute_error)
    grouped = rdf.groupby(["shift", "severity"]).agg(
        mae=("absolute_error", "mean"), rmse=("squared_error", lambda x: np.sqrt(np.mean(x))),
        ensemble_variance=("ensemble_variance", "mean"), coverage_90=("covered_90", "mean"),
        naive_mae=("naive_absolute_error", "mean"), ood_score=("ood_score", "mean"), n=("task_id", "count")
    ).reset_index()
    iid = grouped[grouped["shift"] == "iid"].iloc[0]
    grouped["error_ratio_vs_iid"] = grouped.mae / iid.mae
    grouped["uncertainty_ratio_vs_iid"] = grouped.ensemble_variance / iid.ensemble_variance
    rdf.to_csv(D3 / "task_metrics.csv", index=False)
    grouped.to_csv(D3 / "shift_summary.csv", index=False)
    pd.DataFrame(seed_metrics).to_csv(D3 / "model_seed_metrics.csv", index=False)
    pd.DataFrame(training).to_csv(D3 / "training_log.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.2))
    for shift in ["covariate", "treatment", "outcome_nonlinearity", "heterogeneous_effect", "hidden_confounding", "combined"]:
        q = grouped[grouped["shift"] == shift]
        ax1.plot(q.severity, q.error_ratio_vs_iid, marker="o", label=shift)
        ax2.plot(q.severity, q.uncertainty_ratio_vs_iid, marker="o", label=shift)
    ax1.axhline(1, color="black", lw=1, ls="--"); ax2.axhline(1, color="black", lw=1, ls="--")
    ax1.set(xlabel="shift severity", ylabel="MAE / IID MAE", title="Error inflation")
    ax2.set(xlabel="shift severity", ylabel="variance / IID variance", title="Ensemble uncertainty response")
    ax2.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(FIG / "direction3_shift_response.png", dpi=180); plt.close(fig)

    sample = shifted.sample(min(1600, len(shifted)), random_state=11)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.scatter(sample.ood_score, sample.absolute_error, s=10, alpha=0.28, color="#8357a4")
    ax.set(xlabel="prior-distance score", ylabel="absolute ATE error", title=f"Mismatch/error Spearman rho={rho:.2f}")
    fig.tight_layout(); fig.savefig(FIG / "direction3_mismatch_detection.png", dpi=180); plt.close(fig)

    return {
        "runtime_seconds": time.time() - started,
        "parameters": {"training_steps": steps, "batch_tasks": batch, "rows_per_task": n_train, "evaluation_tasks_per_cell": tasks_per, "model_seeds": SEEDS, "detector_seed": 11, "calibration_seed": 777, "evaluation_seeds": [12000 + i for i in range(len(conditions))], "shifts": conditions},
        "training": training,
        "calibration": {"separate_calibration_tasks": len(cal_rows), "bad_error_threshold": bad_threshold, "interval_scale_90": interval_scale},
        "shift_summary": grouped.to_dict(orient="records"),
        "task_metrics": records,
        "seed_metrics": seed_metrics,
        "failure_detection": {"spearman_ood_error": rho, "spearman_ood_error_p": pvalue, "spearman_uncertainty_error": urho, "spearman_uncertainty_error_p": up, "auroc": detection},
        "leakage_checks": {"detector_fit_only_iid_unlabeled_summaries": True, "threshold_and_interval_frozen_on_separate_iid_calibration": True, "shift_labels_not_detector_inputs": True},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small smoke run")
    args = parser.parse_args()
    for d in [FIG, D1, D2, D3]:
        d.mkdir(parents=True, exist_ok=True)
    begun = time.time()
    results = {
        "study": "Rapid Feasibility Sweep for Next-Gen Tabular AI Research",
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "quick": args.quick,
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__, "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
        },
        "errors": [],
    }
    for key, fn in [("direction_1_causal_relational", direction1), ("direction_2_semantics", direction2), ("direction_3_prior_shift", direction3)]:
        print(f"running {key}", flush=True)
        try:
            results[key] = fn(args.quick)
        except Exception as exc:
            results["errors"].append({"direction": key, "exception": repr(exc), "traceback": traceback.format_exc()})
            print(traceback.format_exc(), flush=True)
    results["total_runtime_seconds"] = time.time() - begun
    (ROOT / "results.json").write_text(json.dumps(py(results), indent=2, allow_nan=False) + "\n")
    print(f"wrote {ROOT / 'results.json'} in {results['total_runtime_seconds']:.1f}s", flush=True)
    if results["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
