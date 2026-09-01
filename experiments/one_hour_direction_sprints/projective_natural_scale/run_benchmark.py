#!/usr/bin/env python3
"""Prospective natural-data escalation for static projective prediction."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tabicl import TabICLRegressor
from tabpfn import TabPFNRegressor
from tabpfn.utils import (
    fix_dtypes,
    process_text_na_dataframe,
    translate_probs_across_borders,
    validate_X_predict,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROJECTIVE_DIR = ROOT / "experiments" / "one_hour_direction_sprints" / "projective"
sys.path.insert(0, str(PROJECTIVE_DIR))
import run_projective as neural  # noqa: E402


CONFIG = json.loads((HERE / "config.json").read_text())
FINAL_CONFIG = json.loads(
    (ROOT / "experiments" / "final_closure" / "final_closure_config.json").read_text()
)
OUT = HERE / "results"
SHARDS = OUT / "shards_float64"
QUERY_FAMILIES = tuple(CONFIG["query_families"])
AGGREGATE_FAMILIES = ("subset", "difference", "dense", "scaled_dense")
MODELS = tuple(CONFIG["models"])
RHO_GRID = tuple(float(value) for value in CONFIG["rho_grid"])
GROUPS = int(CONFIG["evaluation_groups"])
Q = int(CONFIG["query_size"])
CONTEXT = int(CONFIG["context_size"])
REPLICATES = int(CONFIG["context_replicates"])
Z90 = float(norm.ppf(0.95))
Z84 = float(norm.ppf(0.84))
EPS = 1e-8


@dataclass
class PreparedData:
    train_full: np.ndarray
    validation_full: np.ndarray
    test_full: np.ndarray
    train_pca: np.ndarray
    validation_pca: np.ndarray
    test_pca: np.ndarray
    train_y: np.ndarray
    validation_y: np.ndarray
    test_y: np.ndarray
    metric_y_mean: float
    metric_y_scale: float
    audit: dict[str, Any]


@dataclass
class QueryBundle:
    full: np.ndarray
    pca: np.ndarray
    target: np.ndarray
    coefficients: dict[str, np.ndarray]
    indices: np.ndarray


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(digest[:15], 16) % (2**31 - 1)


def protocol_hash() -> str:
    return hashlib.sha256((HERE / "PROTOCOL.md").read_bytes()).hexdigest()


def dataset_spec(name: str) -> dict[str, Any]:
    matches = [spec for spec in CONFIG["datasets"] if spec["name"] == name]
    if len(matches) != 1:
        raise KeyError(name)
    return matches[0]


def load_raw(spec: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray]:
    if spec["source"] == "local":
        root = Path(FINAL_CONFIG["data_root"]) / spec["name"]
        x = np.concatenate(
            [np.asarray(np.load(root / f"N_{part}.npy")) for part in ("train", "val", "test")]
        )
        y = np.concatenate(
            [np.asarray(np.load(root / f"y_{part}.npy")) for part in ("train", "val", "test")]
        )
        frame = pd.DataFrame(x, columns=[f"x{j}" for j in range(x.shape[1])])
    else:
        from sklearn.datasets import fetch_openml

        bunch = fetch_openml(data_id=int(spec["openml_id"]), as_frame=True, parser="auto")
        frame = bunch.data.copy()
        y = pd.to_numeric(pd.Series(np.asarray(bunch.target)), errors="coerce").to_numpy()
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    keep = np.isfinite(y)
    frame = frame.loc[keep].reset_index(drop=True)
    y = y[keep]
    if len(frame) != len(y) or len(y) < 1_000:
        raise ValueError(f"invalid dataset {spec['name']}: X={len(frame)}, y={len(y)}")
    return frame, y


def cap_indices(indices: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if len(indices) <= maximum:
        return np.sort(indices)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=maximum, replace=False))


def make_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric = [column for column in frame if pd.api.types.is_numeric_dtype(frame[column].dtype)]
    categorical = [column for column in frame if column not in numeric]
    transforms: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transforms.append(("numeric", SimpleImputer(strategy="median"), numeric))
    if categorical:
        transforms.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            )
        )
    if not transforms:
        raise ValueError("dataset has no usable columns")
    return ColumnTransformer(transforms, sparse_threshold=0.0)


def prepare_data(spec: dict[str, Any], split_seed: int) -> PreparedData:
    frame, y = load_raw(spec)
    all_rows = np.arange(len(y))
    train_validation, test = train_test_split(
        all_rows, test_size=0.2, random_state=split_seed, shuffle=True
    )
    train, validation = train_test_split(
        train_validation, test_size=0.25, random_state=split_seed + 1, shuffle=True
    )
    sizes = CONFIG["split_sizes"]
    train = cap_indices(train, int(sizes["train"]), split_seed + 11)
    validation = cap_indices(validation, int(sizes["validation"]), split_seed + 13)
    test = cap_indices(test, int(sizes["test"]), split_seed + 17)
    if set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test):
        raise AssertionError("split overlap")
    if len(validation) < GROUPS * Q or len(test) < GROUPS * Q:
        raise ValueError("validation/test caps are too small for disjoint query groups")

    preprocessor = make_preprocessor(frame.iloc[train])
    train_unscaled = np.asarray(preprocessor.fit_transform(frame.iloc[train]), dtype=np.float64)
    validation_unscaled = np.asarray(preprocessor.transform(frame.iloc[validation]), dtype=np.float64)
    test_unscaled = np.asarray(preprocessor.transform(frame.iloc[test]), dtype=np.float64)
    scaler = StandardScaler().fit(train_unscaled)
    train_full = scaler.transform(train_unscaled)
    validation_full = scaler.transform(validation_unscaled)
    test_full = scaler.transform(test_unscaled)
    train_full = np.nan_to_num(train_full, nan=0.0, posinf=8.0, neginf=-8.0)
    validation_full = np.nan_to_num(validation_full, nan=0.0, posinf=8.0, neginf=-8.0)
    test_full = np.nan_to_num(test_full, nan=0.0, posinf=8.0, neginf=-8.0)
    train_full = np.clip(train_full, -8.0, 8.0)
    validation_full = np.clip(validation_full, -8.0, 8.0)
    test_full = np.clip(test_full, -8.0, 8.0)

    if train_full.shape[1] > neural.DIM:
        pca = PCA(n_components=neural.DIM, svd_solver="full").fit(train_full)
        train_pca = pca.transform(train_full)
        validation_pca = pca.transform(validation_full)
        test_pca = pca.transform(test_full)
    else:
        pad = neural.DIM - train_full.shape[1]
        train_pca = np.pad(train_full, ((0, 0), (0, pad)))
        validation_pca = np.pad(validation_full, ((0, 0), (0, pad)))
        test_pca = np.pad(test_full, ((0, 0), (0, pad)))
    pca_scale = np.where(train_pca.std(axis=0) > 0, train_pca.std(axis=0), 1.0)
    train_pca = np.clip(train_pca / pca_scale, -5.0, 5.0)
    validation_pca = np.clip(validation_pca / pca_scale, -5.0, 5.0)
    test_pca = np.clip(test_pca / pca_scale, -5.0, 5.0)

    metric_y_mean = float(y[train].mean())
    metric_y_scale = float(y[train].std())
    if not np.isfinite(metric_y_scale) or metric_y_scale < EPS:
        metric_y_scale = 1.0
    audit = {
        "dataset": spec["name"],
        "split_seed": split_seed,
        "raw_rows": int(len(y)),
        "raw_features": int(frame.shape[1]),
        "encoded_features": int(train_full.shape[1]),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "split_disjoint": True,
        "train_index_sha256": hashlib.sha256(train.tobytes()).hexdigest(),
        "validation_index_sha256": hashlib.sha256(validation.tobytes()).hexdigest(),
        "test_index_sha256": hashlib.sha256(test.tobytes()).hexdigest(),
    }
    return PreparedData(
        *(np.asarray(value, dtype=np.float32) for value in (
            train_full,
            validation_full,
            test_full,
            train_pca,
            validation_pca,
            test_pca,
            y[train],
            y[validation],
            y[test],
        )),
        metric_y_mean=metric_y_mean,
        metric_y_scale=metric_y_scale,
        audit=audit,
    )


def make_coefficients(rng: np.random.Generator) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    rows = np.arange(GROUPS)
    point = np.zeros((GROUPS, Q), dtype=np.float64)
    point[rows, rng.integers(Q, size=GROUPS)] = 1.0
    result["point"] = point

    subset = rng.random((GROUPS, Q)) < 0.35
    empty = ~subset.any(axis=1)
    subset[np.flatnonzero(empty), rng.integers(Q, size=int(empty.sum()))] = True
    result["subset"] = subset / subset.sum(axis=1, keepdims=True)

    difference = np.zeros((GROUPS, Q), dtype=np.float64)
    first = rng.integers(Q, size=GROUPS)
    second = (first + rng.integers(1, Q, size=GROUPS)) % Q
    difference[rows, first] = 1.0
    difference[rows, second] = -1.0
    result["difference"] = difference

    dense = rng.normal(size=(GROUPS, Q))
    dense /= np.linalg.norm(dense, axis=1, keepdims=True)
    result["dense"] = dense
    result["scaled_dense"] = dense * rng.uniform(1.75, 3.0, size=(GROUPS, 1))
    return {key: value.astype(np.float64) for key, value in result.items()}


def make_query_bundle(
    full: np.ndarray,
    pca: np.ndarray,
    y: np.ndarray,
    metric_y_mean: float,
    metric_y_scale: float,
    seed: int,
) -> QueryBundle:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(y), size=GROUPS * Q, replace=False).reshape(GROUPS, Q)
    return QueryBundle(
        full=full[indices],
        pca=pca[indices],
        target=(y[indices].astype(np.float64) - metric_y_mean) / metric_y_scale,
        coefficients=make_coefficients(rng),
        indices=indices,
    )


def load_neural_models(device: torch.device) -> tuple[list[Any], list[Any]]:
    projective, direct = [], []
    for seed in neural.SEEDS:
        pmodel = neural.ProjectiveModel().to(device)
        dmodel = neural.DirectModel().to(device)
        pmodel.load_state_dict(
            torch.load(
                PROJECTIVE_DIR / "results" / f"projective_seed{seed}.pt",
                map_location=device,
                weights_only=True,
            )
        )
        dmodel.load_state_dict(
            torch.load(
                PROJECTIVE_DIR / "results" / f"direct_seed{seed}.pt",
                map_location=device,
                weights_only=True,
            )
        )
        pmodel.eval()
        dmodel.eval()
        projective.append(pmodel)
        direct.append(dmodel)
    return projective, direct


def convert_joint_scale(
    mean: np.ndarray,
    covariance: np.ndarray,
    context_mean: float,
    context_scale: float,
    metric_mean: float,
    metric_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    ratio = context_scale / metric_scale
    offset = (context_mean - metric_mean) / metric_scale
    mean64 = np.asarray(mean, dtype=np.float64)
    covariance64 = np.asarray(covariance, dtype=np.float64)
    return offset + ratio * mean64, ratio**2 * covariance64


@torch.no_grad()
def predict_neural_projective(
    models: list[Any],
    context_x: np.ndarray,
    context_y: np.ndarray,
    query_x: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    xc = torch.from_numpy(context_x).to(device)[None].expand(GROUPS, -1, -1)
    yc = torch.from_numpy(context_y.astype(np.float32)).to(device)[None].expand(GROUPS, -1)
    xq = torch.from_numpy(query_x).to(device)
    means: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    forward_error = 0.0
    for model in models:
        mean, factor, diagonal = model.joint(xc, yc, xq)
        covariance = factor @ factor.transpose(1, 2) + torch.diag_embed(diagonal.square())
        mean_numpy = mean.cpu().numpy().astype(np.float64)
        factor_numpy = factor.cpu().numpy().astype(np.float64)
        diagonal_numpy = diagonal.cpu().numpy().astype(np.float64)
        covariance_numpy = np.einsum(
            "gqi,gri->gqr", factor_numpy, factor_numpy
        )
        covariance_numpy[:, np.arange(Q), np.arange(Q)] += diagonal_numpy**2
        means.append(mean_numpy)
        covariances.append(covariance_numpy)
        probe = torch.from_numpy(make_coefficients(np.random.default_rng(7))["dense"].astype(np.float32)).to(device)
        projected_mean, projected_variance = model(xc, yc, xq, probe)
        joint_mean = (probe * mean).sum(1)
        joint_variance = torch.einsum("gq,gqr,gr->g", probe, covariance, probe)
        forward_error = max(
            forward_error,
            float((projected_mean - joint_mean).abs().max()),
            float((projected_variance - joint_variance).abs().max()),
        )
    stacked_mean = np.stack(means)
    mean_numpy = stacked_mean.mean(0)
    centered = stacked_mean - mean_numpy[None]
    between = np.einsum("sgq,sgr->gqr", centered, centered) / len(models)
    covariance_numpy = np.stack(covariances).mean(0) + between
    return mean_numpy, covariance_numpy, forward_error


@torch.no_grad()
def predict_neural_direct(
    models: list[Any],
    context_x: np.ndarray,
    context_y: np.ndarray,
    query_x: np.ndarray,
    coefficients: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    xc = torch.from_numpy(context_x).to(device)[None].expand(GROUPS, -1, -1)
    yc = torch.from_numpy(context_y.astype(np.float32)).to(device)[None].expand(GROUPS, -1)
    xq = torch.from_numpy(query_x).to(device)
    result = {}
    for family, weights in coefficients.items():
        a = torch.from_numpy(weights.astype(np.float32)).to(device)
        member_means, member_variances = [], []
        for model in models:
            mean, variance = model(xc, yc, xq, a)
            member_means.append(mean)
            member_variances.append(variance)
        means = torch.stack(member_means)
        mean = means.mean(0)
        variance = torch.stack(member_variances).mean(0) + (means - mean[None]).square().mean(0)
        result[family] = (mean.cpu().numpy(), variance.cpu().numpy())
    return result


def predict_bayes_linear(
    context_x: np.ndarray, context_y: np.ndarray, query_x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    train = np.concatenate([context_x.astype(np.float64), np.ones((CONTEXT, 1))], axis=1)
    test = np.concatenate([query_x.reshape(-1, query_x.shape[-1]).astype(np.float64), np.ones((GROUPS * Q, 1))], axis=1)
    model = BayesianRidge(fit_intercept=False, tol=1e-5, max_iter=500)
    model.fit(train, context_y.astype(np.float64))
    mean = (test @ model.coef_).reshape(GROUPS, Q)
    latent = test @ model.sigma_ @ test.T
    covariance = latent + np.eye(len(test)) / max(float(model.alpha_), EPS)
    blocks = np.stack(
        [covariance[start : start + Q, start : start + Q] for start in range(0, len(test), Q)]
    )
    return mean, blocks


def predict_gp(
    context_x: np.ndarray,
    context_y: np.ndarray,
    query_x: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(
        0.1, (1e-4, 3.0)
    )
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-8,
        normalize_y=False,
        n_restarts_optimizer=0,
        random_state=seed,
    )
    fallback = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        try:
            model.fit(context_x.astype(np.float64), context_y.astype(np.float64))
        except Exception:
            fallback = True
            model = GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.1),
                alpha=1e-6,
                normalize_y=False,
                optimizer=None,
            ).fit(context_x.astype(np.float64), context_y.astype(np.float64))
    flat = query_x.reshape(-1, query_x.shape[-1]).astype(np.float64)
    mean, covariance = model.predict(flat, return_cov=True)
    blocks = np.stack(
        [covariance[start : start + Q, start : start + Q] for start in range(0, len(flat), Q)]
    )
    return mean.reshape(GROUPS, Q), blocks, fallback


def tabpfn_marginals_and_members(
    model: TabPFNRegressor, query_x: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = validate_X_predict(query_x, model)
    x = fix_dtypes(x, cat_indices=model.inferred_categorical_indices_)
    x = process_text_na_dataframe(x, ord_encoder=model.preprocessor_)
    _, outputs, borders = model.forward(x, use_inference_mode=True)
    probabilities = [
        translate_probs_across_borders(
            logits,
            frm=torch.as_tensor(border, device=logits.device),
            to=model.znorm_space_bardist_.borders.to(logits.device),
        )
        for logits, border in zip(outputs, borders)
    ]
    stacked = torch.stack(probabilities)
    if model.average_before_softmax:
        averaged = stacked.clamp_min(1e-30).log().mean(0).softmax(-1)
    else:
        averaged = stacked.mean(0)
    logits = averaged.clamp_min(1e-30).log().float()
    criterion = model.raw_space_bardist_
    mean = criterion.mean(logits)
    variance = criterion.variance(logits)
    member_means = torch.stack(
        [criterion.mean(probability.clamp_min(1e-30).log().float()) for probability in probabilities]
    )
    return (
        mean.detach().cpu().numpy(),
        variance.detach().cpu().numpy(),
        member_means.detach().cpu().numpy(),
    )


def covariance_from_ensemble_views(
    marginal_variance: np.ndarray,
    member_means: np.ndarray,
    rho: float,
) -> np.ndarray:
    marginal_variance = np.asarray(marginal_variance, dtype=np.float64)
    member_means = np.asarray(member_means, dtype=np.float64)
    members = member_means.reshape(member_means.shape[0], GROUPS, Q)
    centered = members - members.mean(axis=0, keepdims=True)
    epistemic = np.einsum("kgq,kgr->gqr", centered, centered) / max(len(members) - 1, 1)
    diagonal = np.maximum(np.diagonal(epistemic, axis1=1, axis2=2), EPS)
    denom = np.sqrt(diagonal[:, :, None] * diagonal[:, None, :])
    correlation = epistemic / denom
    correlation = np.clip(correlation, -1.0, 1.0)
    eye = np.broadcast_to(np.eye(Q), correlation.shape)
    correlation = 0.5 * (correlation + correlation.transpose(0, 2, 1))
    correlation[:, np.arange(Q), np.arange(Q)] = 1.0
    shrunk = (1.0 - rho) * eye + rho * correlation
    variance = np.maximum(marginal_variance.reshape(GROUPS, Q), EPS)
    scale = np.sqrt(variance)
    covariance = scale[:, :, None] * shrunk * scale[:, None, :]
    covariance = 0.5 * (covariance + covariance.transpose(0, 2, 1))
    # Avoid a sqrt(v)**2 round-off on large-scale targets: the defining
    # marginal variance is assigned exactly after constructing off-diagonals.
    covariance[:, np.arange(Q), np.arange(Q)] = variance
    return covariance


def predict_tabpfn(
    context_x: np.ndarray,
    context_y: np.ndarray,
    validation_x: np.ndarray,
    test_x: np.ndarray,
    seed: int,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model = TabPFNRegressor(
        n_estimators=int(CONFIG["tabpfn_estimators"]),
        device=device,
        random_state=seed,
    )
    model.fit(context_x, context_y)
    joined = np.concatenate([validation_x.reshape(-1, validation_x.shape[-1]), test_x.reshape(-1, test_x.shape[-1])])
    if getattr(model, "is_constant_target_", False):
        # TabPFN's documented constant-target branch is a point mass and does not
        # instantiate ensemble views.  Retain that marginal (up to EPS for proper
        # scores) and therefore introduce no cross-row correlation.
        mean = np.full(len(joined), float(model.constant_value_))
        variance = np.full(len(joined), EPS)
        members = np.broadcast_to(
            mean[None], (int(CONFIG["tabpfn_estimators"]), len(mean))
        ).copy()
    else:
        mean, variance, members = tabpfn_marginals_and_members(model, joined)
    count = GROUPS * Q
    outputs = []
    for selection in (slice(0, count), slice(count, 2 * count)):
        selected_mean = mean[selection].reshape(GROUPS, Q)
        selected_variance = np.maximum(variance[selection], EPS)
        selected_members = members[:, selection]
        covariance = {
            rho: covariance_from_ensemble_views(selected_variance, selected_members, rho)
            for rho in RHO_GRID
        }
        outputs.append(
            {
                "mean": selected_mean,
                "marginal_variance": selected_variance.reshape(GROUPS, Q),
                "covariance": covariance,
            }
        )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return outputs[0], outputs[1]


def predict_tabicl(
    context_x: np.ndarray,
    context_y: np.ndarray,
    validation_x: np.ndarray,
    test_x: np.ndarray,
    seed: int,
    device: str,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    model = TabICLRegressor(
        n_estimators=int(CONFIG["tabicl_estimators"]),
        device=device,
        random_state=seed,
    )
    model.fit(context_x, context_y)
    joined = np.concatenate([validation_x.reshape(-1, validation_x.shape[-1]), test_x.reshape(-1, test_x.shape[-1])])
    output = model.predict(joined, output_type=["mean", "quantiles"], alphas=[0.16, 0.84])
    mean = np.asarray(output["mean"], dtype=np.float64)
    quantiles = np.asarray(output["quantiles"], dtype=np.float64)
    std = np.maximum((quantiles[:, 1] - quantiles[:, 0]) / (2.0 * Z84), math.sqrt(EPS))
    variance = std**2
    count = GROUPS * Q
    outputs = []
    for selection in (slice(0, count), slice(count, 2 * count)):
        selected_mean = mean[selection].reshape(GROUPS, Q)
        selected_variance = variance[selection].reshape(GROUPS, Q)
        outputs.append((selected_mean, np.stack([np.diag(row) for row in selected_variance])))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return outputs[0], outputs[1]


def project_joint(
    mean: np.ndarray,
    covariance: np.ndarray,
    coefficients: dict[str, np.ndarray],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result = {}
    for family, weights in coefficients.items():
        projected_mean = np.einsum("gq,gq->g", weights, mean)
        projected_variance = np.einsum("gq,gqr,gr->g", weights, covariance, weights)
        result[family] = (projected_mean, np.maximum(projected_variance, EPS))
    return result


def convert_direct(
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    coefficients: dict[str, np.ndarray],
    context_mean: float,
    context_scale: float,
    metric_mean: float,
    metric_scale: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    ratio = context_scale / metric_scale
    base_offset = (context_mean - metric_mean) / metric_scale
    return {
        family: (
            base_offset * coefficients[family].sum(axis=1)
            + ratio * np.asarray(mean, dtype=np.float64),
            ratio**2 * np.asarray(variance, dtype=np.float64),
        )
        for family, (mean, variance) in predictions.items()
    }


def targets(bundle: QueryBundle) -> dict[str, np.ndarray]:
    return {
        family: np.einsum("gq,gq->g", weights, bundle.target)
        for family, weights in bundle.coefficients.items()
    }


def add_prediction(
    store: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]],
    model: str,
    prediction: dict[str, tuple[np.ndarray, np.ndarray]],
    truth: dict[str, np.ndarray],
) -> None:
    for family in QUERY_FAMILIES:
        mean, variance = prediction[family]
        if not np.isfinite(mean).all() or not np.isfinite(variance).all():
            raise FloatingPointError(f"nonfinite prediction: {model}/{family}")
        store.setdefault(model, {}).setdefault(family, []).append(
            (np.asarray(mean), np.maximum(np.asarray(variance), EPS), truth[family])
        )


def temperature_for(
    records: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]
) -> tuple[float, float]:
    standardized = []
    for family in QUERY_FAMILIES:
        for mean, variance, truth in records[family]:
            standardized.append((truth - mean) ** 2 / np.maximum(variance, EPS))
    temperature = float(np.clip(np.concatenate(standardized).mean(), 1e-3, 1e3))
    nlls = []
    for family in QUERY_FAMILIES:
        for mean, variance, truth in records[family]:
            calibrated = temperature * variance
            nlls.append(0.5 * (np.log(2 * np.pi * calibrated) + (truth - mean) ** 2 / calibrated))
    return temperature, float(np.concatenate(nlls).mean())


def metrics(mean: np.ndarray, variance: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    variance = np.maximum(variance, EPS)
    error = truth - mean
    std = np.sqrt(variance)
    z = error / std
    nll = 0.5 * (np.log(2 * np.pi * variance) + error**2 / variance)
    crps = std * (z * (2 * ndtr(z) - 1) + 2 * np.exp(-0.5 * z**2) / math.sqrt(2 * np.pi) - 1 / math.sqrt(np.pi))
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "nll": float(np.mean(nll)),
        "crps": float(np.mean(crps)),
        "coverage90": float(np.mean(np.abs(error) <= Z90 * std)),
    }


def covariance_audit(covariance: np.ndarray) -> tuple[float, float]:
    symmetry = float(np.max(np.abs(covariance - covariance.transpose(0, 2, 1))))
    minimum_eigenvalue = float(min(np.linalg.eigvalsh(matrix).min() for matrix in covariance))
    return symmetry, minimum_eigenvalue


def run_cell(
    dataset: str,
    split_seed: int,
    projective_models: list[Any],
    direct_models: list[Any],
    neural_device: torch.device,
    tabpfn_device: str,
    tabicl_device: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    started = time.perf_counter()
    data = prepare_data(dataset_spec(dataset), split_seed)
    validation_store: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]] = {}
    test_store: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]] = {}
    episode_audits = []

    for replicate in range(REPLICATES):
        rng = np.random.default_rng(stable_seed("context", dataset, split_seed, replicate))
        context_indices = rng.choice(len(data.train_y), size=CONTEXT, replace=False)
        context_y_raw = data.train_y[context_indices].astype(np.float64)
        context_mean = float(context_y_raw.mean())
        context_scale = float(context_y_raw.std())
        if not np.isfinite(context_scale) or context_scale < EPS:
            context_scale = data.metric_y_scale
        context_y = ((context_y_raw - context_mean) / context_scale).astype(np.float32)
        validation = make_query_bundle(
            data.validation_full,
            data.validation_pca,
            data.validation_y,
            data.metric_y_mean,
            data.metric_y_scale,
            stable_seed("validation", dataset, split_seed, replicate),
        )
        test = make_query_bundle(
            data.test_full,
            data.test_pca,
            data.test_y,
            data.metric_y_mean,
            data.metric_y_scale,
            stable_seed("test", dataset, split_seed, replicate),
        )
        partition_predictions: dict[str, dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]] = {
            "validation": {},
            "test": {},
        }
        audit = {
            "replicate": replicate,
            "context_index_sha256": hashlib.sha256(context_indices.tobytes()).hexdigest(),
            "validation_query_unique": int(len(np.unique(validation.indices))) == GROUPS * Q,
            "test_query_unique": int(len(np.unique(test.indices))) == GROUPS * Q,
            "gp_fallbacks": 0,
            "neural_forward_max_abs": 0.0,
            "covariance_symmetry_max_abs": 0.0,
            "covariance_min_eigenvalue": float("inf"),
            "tabpfn_marginal_max_abs": 0.0,
        }

        for partition_name, bundle in (("validation", validation), ("test", test)):
            p_mean, p_covariance, forward_error = predict_neural_projective(
                projective_models,
                data.train_pca[context_indices],
                context_y,
                bundle.pca,
                neural_device,
            )
            p_mean, p_covariance = convert_joint_scale(
                p_mean,
                p_covariance,
                context_mean,
                context_scale,
                data.metric_y_mean,
                data.metric_y_scale,
            )
            partition_predictions[partition_name]["neural_projective"] = project_joint(
                p_mean, p_covariance, bundle.coefficients
            )
            symmetry, minimum_eigenvalue = covariance_audit(p_covariance)
            audit["neural_forward_max_abs"] = max(audit["neural_forward_max_abs"], forward_error)
            audit["covariance_symmetry_max_abs"] = max(audit["covariance_symmetry_max_abs"], symmetry)
            audit["covariance_min_eigenvalue"] = min(audit["covariance_min_eigenvalue"], minimum_eigenvalue)

            direct = predict_neural_direct(
                direct_models,
                data.train_pca[context_indices],
                context_y,
                bundle.pca,
                bundle.coefficients,
                neural_device,
            )
            partition_predictions[partition_name]["neural_direct"] = convert_direct(
                direct,
                bundle.coefficients,
                context_mean,
                context_scale,
                data.metric_y_mean,
                data.metric_y_scale,
            )

            b_mean, b_covariance = predict_bayes_linear(
                data.train_full[context_indices], context_y, bundle.full
            )
            b_mean, b_covariance = convert_joint_scale(
                b_mean,
                b_covariance,
                context_mean,
                context_scale,
                data.metric_y_mean,
                data.metric_y_scale,
            )
            partition_predictions[partition_name]["bayes_linear"] = project_joint(
                b_mean, b_covariance, bundle.coefficients
            )
            symmetry, minimum_eigenvalue = covariance_audit(b_covariance)
            audit["covariance_symmetry_max_abs"] = max(audit["covariance_symmetry_max_abs"], symmetry)
            audit["covariance_min_eigenvalue"] = min(audit["covariance_min_eigenvalue"], minimum_eigenvalue)

            g_mean, g_covariance, fallback = predict_gp(
                data.train_full[context_indices],
                context_y,
                bundle.full,
                stable_seed("gp", dataset, split_seed, replicate),
            )
            audit["gp_fallbacks"] += int(fallback)
            g_mean, g_covariance = convert_joint_scale(
                g_mean,
                g_covariance,
                context_mean,
                context_scale,
                data.metric_y_mean,
                data.metric_y_scale,
            )
            partition_predictions[partition_name]["gp_rbf"] = project_joint(
                g_mean, g_covariance, bundle.coefficients
            )
            symmetry, minimum_eigenvalue = covariance_audit(g_covariance)
            audit["covariance_symmetry_max_abs"] = max(audit["covariance_symmetry_max_abs"], symmetry)
            audit["covariance_min_eigenvalue"] = min(audit["covariance_min_eigenvalue"], minimum_eigenvalue)

        tabpfn_validation, tabpfn_test = predict_tabpfn(
            data.train_full[context_indices],
            context_y,
            validation.full,
            test.full,
            stable_seed("tabpfn", dataset, split_seed, replicate),
            tabpfn_device,
        )
        for partition_name, bundle, output in (
            ("validation", validation, tabpfn_validation),
            ("test", test, tabpfn_test),
        ):
            mean = output["mean"]
            covariance_zero = output["covariance"][0.0]
            mean, covariance_zero = convert_joint_scale(
                mean,
                covariance_zero,
                context_mean,
                context_scale,
                data.metric_y_mean,
                data.metric_y_scale,
            )
            partition_predictions[partition_name]["tabpfn_independent"] = project_joint(
                mean, covariance_zero, bundle.coefficients
            )
            for rho in RHO_GRID:
                rho_mean, rho_covariance = convert_joint_scale(
                    output["mean"],
                    output["covariance"][rho],
                    context_mean,
                    context_scale,
                    data.metric_y_mean,
                    data.metric_y_scale,
                )
                key = f"tabpfn_projective_rho{rho:.2f}"
                partition_predictions[partition_name][key] = project_joint(
                    rho_mean, rho_covariance, bundle.coefficients
                )
                expected_diagonal = (
                    (context_scale / data.metric_y_scale) ** 2
                    * np.asarray(output["marginal_variance"], dtype=np.float64)
                )
                audit["tabpfn_marginal_max_abs"] = max(
                    audit["tabpfn_marginal_max_abs"],
                    float(
                        np.max(
                            np.abs(
                                np.diagonal(rho_covariance, axis1=1, axis2=2)
                                - expected_diagonal
                            )
                        )
                    ),
                )
                symmetry, minimum_eigenvalue = covariance_audit(rho_covariance)
                audit["covariance_symmetry_max_abs"] = max(audit["covariance_symmetry_max_abs"], symmetry)
                audit["covariance_min_eigenvalue"] = min(audit["covariance_min_eigenvalue"], minimum_eigenvalue)

        tabicl_validation, tabicl_test = predict_tabicl(
            data.train_full[context_indices],
            context_y,
            validation.full,
            test.full,
            stable_seed("tabicl", dataset, split_seed, replicate),
            tabicl_device,
        )
        for partition_name, bundle, (mean, covariance) in (
            ("validation", validation, tabicl_validation),
            ("test", test, tabicl_test),
        ):
            mean, covariance = convert_joint_scale(
                mean,
                covariance,
                context_mean,
                context_scale,
                data.metric_y_mean,
                data.metric_y_scale,
            )
            partition_predictions[partition_name]["tabicl_independent"] = project_joint(
                mean, covariance, bundle.coefficients
            )

        validation_truth = targets(validation)
        test_truth = targets(test)
        for model, prediction in partition_predictions["validation"].items():
            add_prediction(validation_store, model, prediction, validation_truth)
        for model, prediction in partition_predictions["test"].items():
            add_prediction(test_store, model, prediction, test_truth)
        episode_audits.append(audit)
        print(
            json.dumps(
                {
                    "dataset": dataset,
                    "split": split_seed,
                    "replicate": replicate + 1,
                    "of": REPLICATES,
                    "elapsed_seconds": round(time.perf_counter() - started, 1),
                }
            ),
            flush=True,
        )

    calibration_rows = []
    chosen: dict[str, tuple[str, float, float]] = {}
    for model in MODELS:
        if model == "tabpfn_projective":
            candidates = []
            for rho in RHO_GRID:
                key = f"tabpfn_projective_rho{rho:.2f}"
                temperature, validation_nll = temperature_for(validation_store[key])
                candidates.append((validation_nll, rho, temperature, key))
            validation_nll, rho, temperature, key = min(candidates)
        else:
            key = model
            temperature, validation_nll = temperature_for(validation_store[key])
            rho = 0.0 if model == "tabpfn_independent" else np.nan
        chosen[model] = (key, temperature, rho)
        calibration_rows.append(
            {
                "dataset": dataset,
                "split_seed": split_seed,
                "model": model,
                "variance_temperature": temperature,
                "rho": rho,
                "validation_nll": validation_nll,
            }
        )

    rows = []
    for model in MODELS:
        key, temperature, rho = chosen[model]
        for family in QUERY_FAMILIES:
            for replicate, (mean, variance, truth) in enumerate(test_store[key][family]):
                score = metrics(mean, temperature * variance, truth)
                rows.append(
                    {
                        "dataset": dataset,
                        "split_seed": split_seed,
                        "context_replicate": replicate,
                        "query_family": family,
                        "model": model,
                        "n_queries": len(truth),
                        "variance_temperature": temperature,
                        "rho": rho,
                        **score,
                    }
                )

    audit = {
        **data.audit,
        "protocol_sha256": protocol_hash(),
        "wall_seconds": time.perf_counter() - started,
        "episode_audits": episode_audits,
    }
    return pd.DataFrame(rows), pd.DataFrame(calibration_rows), audit


def analyze() -> dict[str, Any]:
    expected = [
        (spec["name"], int(split))
        for spec in CONFIG["datasets"]
        for split in CONFIG["split_seeds"]
    ]
    missing = []
    cell_frames, calibration_frames, audits = [], [], []
    for dataset, split in expected:
        stem = f"{dataset}_seed{split}"
        paths = (
            SHARDS / f"{stem}_cells.csv",
            SHARDS / f"{stem}_calibration.csv",
            SHARDS / f"{stem}_audit.json",
        )
        if not all(path.exists() for path in paths):
            missing.append(stem)
            continue
        cell_frames.append(pd.read_csv(paths[0]))
        calibration_frames.append(pd.read_csv(paths[1]))
        audits.append(json.loads(paths[2].read_text()))
    if missing:
        raise FileNotFoundError(f"missing shards: {missing}")
    cells = pd.concat(cell_frames, ignore_index=True)
    calibration = pd.concat(calibration_frames, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    cells.to_csv(OUT / "cells.csv", index=False)
    calibration.to_csv(OUT / "calibration.csv", index=False)

    aggregate = cells[cells.query_family.isin(AGGREGATE_FAMILIES)]
    neural = aggregate[aggregate.model.isin(["neural_projective", "neural_direct"])].pivot(
        index=["dataset", "split_seed", "context_replicate", "query_family"],
        columns="model",
        values="nll",
    )
    neural_advantage = neural.neural_direct - neural.neural_projective
    dense_neural = neural.loc[
        neural.index.get_level_values("query_family").isin(["dense", "scaled_dense"])
    ]
    dense_neural_advantage = dense_neural.neural_direct - dense_neural.neural_projective

    tab = aggregate[aggregate.model.isin(["tabpfn_projective", "tabpfn_independent"])]
    dataset_tab = tab.groupby(["dataset", "model"])[["nll", "crps", "coverage90"]].mean().unstack("model")
    dataset_tab.columns = [f"{metric}_{model}" for metric, model in dataset_tab.columns]
    dataset_tab["nll_advantage"] = (
        dataset_tab.nll_tabpfn_independent - dataset_tab.nll_tabpfn_projective
    )
    dataset_tab["crps_advantage"] = (
        dataset_tab.crps_tabpfn_independent - dataset_tab.crps_tabpfn_projective
    )
    dataset_tab["coverage_error_change"] = (
        np.abs(dataset_tab.coverage90_tabpfn_projective - 0.9)
        - np.abs(dataset_tab.coverage90_tabpfn_independent - 0.9)
    )
    dataset_tab.reset_index().to_csv(OUT / "tabpfn_projectivity_by_dataset.csv", index=False)

    point = cells[cells.query_family == "point"].groupby(["dataset", "model"])["rmse"].mean().unstack()
    point.to_csv(OUT / "point_rmse_by_dataset.csv")
    neural_ratio = point.neural_projective / point.tabpfn_independent

    overall = cells.groupby(["query_family", "model"])[["rmse", "nll", "crps", "coverage90"]].mean()
    overall.to_csv(OUT / "overall_metrics.csv")

    competitive_rows = []
    for comparator in ("bayes_linear", "gp_rbf", "tabicl_independent"):
        paired = aggregate[aggregate.model.isin(["tabpfn_projective", comparator])].pivot(
            index=["dataset", "split_seed", "context_replicate", "query_family"],
            columns="model",
            values="nll",
        )
        advantage = paired[comparator] - paired.tabpfn_projective
        by_dataset = advantage.groupby("dataset").mean()
        competitive_rows.append(
            {
                "comparator": comparator,
                "mean_aggregate_nll_advantage": float(advantage.mean()),
                "cell_win_rate": float((advantage > 0).mean()),
                "dataset_wins": int((by_dataset > 0).sum()),
            }
        )
    pd.DataFrame(competitive_rows).to_csv(OUT / "competitive_summary.csv", index=False)

    expected_rows = len(expected) * REPLICATES * len(QUERY_FAMILIES) * len(MODELS)
    max_symmetry = max(
        episode["covariance_symmetry_max_abs"]
        for audit in audits
        for episode in audit["episode_audits"]
    )
    min_eigenvalue = min(
        episode["covariance_min_eigenvalue"]
        for audit in audits
        for episode in audit["episode_audits"]
    )
    max_marginal_error = max(
        episode["tabpfn_marginal_max_abs"]
        for audit in audits
        for episode in audit["episode_audits"]
    )
    max_forward_error = max(
        episode["neural_forward_max_abs"]
        for audit in audits
        for episode in audit["episode_audits"]
    )
    integrity = bool(
        len(cells) == expected_rows
        and np.isfinite(cells[["rmse", "nll", "crps", "coverage90"]]).all().all()
        and all(audit["protocol_sha256"] == protocol_hash() for audit in audits)
        and all(audit["split_disjoint"] for audit in audits)
        and max_symmetry <= 1e-5
        and min_eigenvalue >= -1e-5
        and max_marginal_error <= 1e-5
        and max_forward_error <= 1e-5
    )

    tab_nll_advantage = float(
        aggregate[aggregate.model == "tabpfn_independent"].nll.mean()
        - aggregate[aggregate.model == "tabpfn_projective"].nll.mean()
    )
    tab_crps_advantage = float(
        aggregate[aggregate.model == "tabpfn_independent"].crps.mean()
        - aggregate[aggregate.model == "tabpfn_projective"].crps.mean()
    )
    coverage_not_worse = float(dataset_tab.coverage_error_change.mean()) <= 0
    original_gates = {
        "dense_scaled_mean_nll_advantage_positive": float(dense_neural_advantage.mean()) > 0,
        "dense_scaled_cell_win_rate_at_least_60pct": float((dense_neural_advantage > 0).mean()) >= 0.60,
        "point_within_25pct_tabpfn_on_at_least_6_datasets": int((neural_ratio <= 1.25).sum()) >= 6,
    }
    strong_mean_gates = {
        "uncalibrated_marginals_preserved": max_marginal_error <= 1e-5,
        "aggregate_mean_nll_improves": tab_nll_advantage > 0,
        "aggregate_mean_crps_improves": tab_crps_advantage > 0,
        "aggregate_dataset_nll_wins_at_least_7": int((dataset_tab.nll_advantage > 0).sum()) >= 7,
        "coverage_error_not_worse": coverage_not_worse,
    }
    result = {
        "status": "complete_natural_scale_projectivity",
        "protocol_sha256": protocol_hash(),
        "integrity": {
            "pass": integrity,
            "expected_rows": expected_rows,
            "observed_rows": int(len(cells)),
            "maximum_covariance_symmetry_error": max_symmetry,
            "minimum_covariance_eigenvalue": min_eigenvalue,
            "maximum_tabpfn_marginal_error": max_marginal_error,
            "maximum_neural_forward_error": max_forward_error,
        },
        "original_network": {
            "dense_scaled_mean_nll_advantage": float(dense_neural_advantage.mean()),
            "dense_scaled_cell_win_rate": float((dense_neural_advantage > 0).mean()),
            "point_datasets_within_25pct_tabpfn": int((neural_ratio <= 1.25).sum()),
            "gates": original_gates,
            "broadly_viable": integrity and all(original_gates.values()),
        },
        "strong_mean_projectivity": {
            "aggregate_mean_nll_advantage": tab_nll_advantage,
            "aggregate_mean_crps_advantage": tab_crps_advantage,
            "aggregate_dataset_nll_wins": int((dataset_tab.nll_advantage > 0).sum()),
            "mean_coverage_error_change": float(dataset_tab.coverage_error_change.mean()),
            "gates": strong_mean_gates,
            "positive": integrity and all(strong_mean_gates.values()),
        },
        "competitive_summary": competitive_rows,
        "total_wall_seconds": float(sum(audit["wall_seconds"] for audit in audits)),
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=[spec["name"] for spec in CONFIG["datasets"]])
    parser.add_argument("--split", type=int, choices=CONFIG["split_seeds"])
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--neural-device", default="cuda:0")
    parser.add_argument("--tabpfn-device", default="cuda:0")
    parser.add_argument("--tabicl-device", default="cuda:1")
    args = parser.parse_args()
    if args.analyze:
        analyze()
        return
    if args.dataset is None or args.split is None:
        parser.error("--dataset and --split are required unless --analyze is used")
    SHARDS.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.neural_device)
    projective_models, direct_models = load_neural_models(device)
    cells, calibration, audit = run_cell(
        args.dataset,
        args.split,
        projective_models,
        direct_models,
        device,
        args.tabpfn_device,
        args.tabicl_device,
    )
    stem = f"{args.dataset}_seed{args.split}"
    cells.to_csv(SHARDS / f"{stem}_cells.csv", index=False)
    calibration.to_csv(SHARDS / f"{stem}_calibration.csv", index=False)
    (SHARDS / f"{stem}_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"completed": stem, "rows": len(cells), "wall_seconds": audit["wall_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
