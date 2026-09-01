#!/usr/bin/env python3
"""Frozen Day 6 screen for metric partition embeddings.

The outcome-bearing defaults are declared in PROTOCOL_FREEZE.md.  Feature maps
never inspect labels; validation labels choose only the ridge penalty.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
METHODS = (
    "linear",
    "ple",
    "periodic",
    "code_rbf",
    "mpe_native",
    "mmpe_native",
    "mpe_corrupt",
)
DOMAINS = ("interval", "cycle", "tree", "nominal")
ALPHAS = np.asarray([1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0])


@dataclass(frozen=True)
class DomainData:
    name: str
    semantic: dict[str, np.ndarray]
    y: dict[str, np.ndarray]
    distance: np.ndarray | None
    support: np.ndarray | None
    held_out: np.ndarray | None


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def tree_distance(n: int = 31) -> np.ndarray:
    ancestors: list[list[int]] = []
    for node in range(n):
        path = []
        current = node
        while True:
            path.append(current)
            if current == 0:
                break
            current = (current - 1) // 2
        ancestors.append(path)
    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        pi = {v: k for k, v in enumerate(ancestors[i])}
        for j in range(n):
            out[i, j] = min(pi[v] + k for k, v in enumerate(ancestors[j]) if v in pi)
    return out


def make_domain(name: str, seed: int) -> DomainData:
    rng = np.random.default_rng(stable_seed("data", name, seed))
    sizes = {"train": 1200, "val": 400, "test": 600}
    if name == "interval":
        x = {part: rng.uniform(0.0, 1.0, n) for part, n in sizes.items()}
        f = lambda v: 0.8 * np.sin(2 * np.pi * v) + 0.45 * np.sin(6 * np.pi * v) + 0.65 * (v > 0.62)
        y = {part: f(v) + rng.normal(0, 0.10, len(v)) for part, v in x.items()}
        return DomainData(name, x, y, None, None, None)

    if name == "cycle":
        n_states = 32
        distance = np.fromfunction(
            lambda i, j: np.minimum(np.abs(i - j), n_states - np.abs(i - j)),
            (n_states, n_states),
        ).astype(np.float64)
        held = np.arange(0, n_states, 4, dtype=np.int64)
        seen = np.setdiff1d(np.arange(n_states), held)
        x = {
            "train": rng.choice(seen, sizes["train"]),
            "val": rng.choice(seen, sizes["val"]),
            "test": rng.choice(held, sizes["test"]),
        }
        theta = 2 * np.pi * np.arange(n_states) / n_states
        state_y = np.sin(theta) + 0.4 * np.cos(2 * theta) + 0.25 * np.sin(3 * theta)
        y = {part: state_y[v] + rng.normal(0, 0.10, len(v)) for part, v in x.items()}
        return DomainData(name, x, y, distance, np.arange(n_states), held)

    if name == "tree":
        n_states = 31
        distance = tree_distance(n_states)
        leaves = np.arange(15, 31)
        held = leaves[::2]
        seen = np.setdiff1d(np.arange(n_states), held)
        x = {
            "train": rng.choice(seen, sizes["train"]),
            "val": rng.choice(seen, sizes["val"]),
            "test": rng.choice(held, sizes["test"]),
        }
        state_y = (
            1.1 * np.exp(-distance[:, 18] / 2.0)
            - 0.9 * np.exp(-distance[:, 27] / 1.6)
            + 0.45 * np.exp(-distance[:, 6] / 2.5)
        )
        y = {part: state_y[v] + rng.normal(0, 0.10, len(v)) for part, v in x.items()}
        return DomainData(name, x, y, distance, np.arange(n_states), held)

    if name == "nominal":
        n_states = 16
        distance = 1.0 - np.eye(n_states)
        x = {part: rng.integers(0, n_states, n) for part, n in sizes.items()}
        state_y = rng.normal(0, 0.8, n_states)
        y = {part: state_y[v] + rng.normal(0, 0.10, len(v)) for part, v in x.items()}
        return DomainData(name, x, y, distance, np.arange(n_states), None)
    raise KeyError(name)


def interval_chart(x: np.ndarray, schema: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    powers = (1.0, 2.0, 3.0, 0.5)
    if schema < 4:
        return x ** powers[schema]
    if schema == 4:
        return np.expm1(2.0 * x) / np.expm1(2.0)
    if schema == 5:
        return np.log1p(9.0 * x) / np.log(10.0)
    if schema == 6:
        raw = 1.0 / (1.0 + np.exp(-6.0 * (x - 0.5)))
        lo = 1.0 / (1.0 + np.exp(3.0))
        hi = 1.0 / (1.0 + np.exp(-3.0))
        return (raw - lo) / (hi - lo)
    return np.sin(0.5 * np.pi * x)


def codebook(domain: DomainData, schema: int, seed: int) -> np.ndarray | None:
    if domain.support is None:
        return None
    rng = np.random.default_rng(stable_seed("schema", domain.name, seed, schema))
    values = np.arange(len(domain.support), dtype=np.float64)
    rng.shuffle(values)
    return values / max(1, len(values) - 1)


def stored_values(domain: DomainData, schema: int, seed: int) -> dict[str, np.ndarray]:
    if domain.name == "interval":
        return {part: interval_chart(x, schema) for part, x in domain.semantic.items()}
    codes = codebook(domain, schema, seed)
    assert codes is not None
    return {part: codes[np.asarray(x, dtype=np.int64)] for part, x in domain.semantic.items()}


def ple_fit_transform(train: np.ndarray, values: dict[str, np.ndarray], dim: int = 16) -> dict[str, np.ndarray]:
    knots = np.quantile(train, np.linspace(0, 1, dim + 1))
    out: dict[str, np.ndarray] = {}
    for part, x in values.items():
        z = np.empty((len(x), dim), dtype=np.float64)
        for j in range(dim):
            lo, hi = knots[j], knots[j + 1]
            if hi <= lo + 1e-14:
                z[:, j] = (x >= hi).astype(np.float64)
            else:
                z[:, j] = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
        out[part] = z
    return out


def periodic_transform(train: np.ndarray, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    lo, hi = float(np.min(train)), float(np.max(train))
    scale = max(hi - lo, 1e-12)
    out = {}
    freq = np.arange(1, 9, dtype=np.float64)
    for part, x in values.items():
        phase = 2 * np.pi * ((x - lo) / scale)[:, None] * freq[None, :]
        out[part] = np.concatenate([np.sin(phase), np.cos(phase)], axis=1)
    return out


def farthest_landmarks(distance: np.ndarray, candidates: np.ndarray, m: int) -> np.ndarray:
    candidates = np.asarray(np.unique(candidates), dtype=np.int64)
    sums = distance[np.ix_(candidates, candidates)].sum(axis=1)
    chosen = [int(candidates[int(np.argmin(sums))])]
    while len(chosen) < min(m, len(candidates)):
        nearest = distance[np.ix_(candidates, np.asarray(chosen))].min(axis=1)
        nearest[np.isin(candidates, chosen)] = -1
        chosen.append(int(candidates[int(np.argmax(nearest))]))
    while len(chosen) < m:
        chosen.append(chosen[-1])
    return np.asarray(chosen, dtype=np.int64)


def normalized_kernel(distances: np.ndarray, tau: float) -> np.ndarray:
    tau = max(float(tau), 1e-6)
    logits = -0.5 * (distances / tau) ** 2
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights


def native_features(domain: DomainData, seed: int, multiscale: bool, corrupt: bool) -> dict[str, np.ndarray]:
    if domain.name == "interval":
        anchors = np.quantile(domain.semantic["train"], np.linspace(0, 1, 16))
        if corrupt:
            # A valid but semantically folded pseudometric control.
            query = {p: np.sin(3 * np.pi * x) for p, x in domain.semantic.items()}
            anchor_position = np.sin(3 * np.pi * anchors)
        else:
            query = domain.semantic
            anchor_position = anchors
        distances = {p: np.abs(x[:, None] - anchor_position[None, :]) for p, x in query.items()}
        tau = max(float(np.max(np.min(distances["train"], axis=1))), 1.0 / 15.0)
    else:
        assert domain.distance is not None
        candidates = np.unique(domain.semantic["train"])
        anchors = farthest_landmarks(domain.distance, candidates, 16)
        metric = domain.distance
        if corrupt and domain.name != "nominal":
            rng = np.random.default_rng(stable_seed("corrupt", domain.name, seed))
            perm = rng.permutation(metric.shape[0])
            metric = metric[np.ix_(perm, perm)]
        distances = {
            p: metric[np.ix_(np.asarray(x, dtype=np.int64), anchors)]
            for p, x in domain.semantic.items()
        }
        tau = max(float(np.max(np.min(distances["train"], axis=1))), 1.0)
    scales = (0.5, 1.0, 2.0) if multiscale else (1.0,)
    return {
        p: np.mean([normalized_kernel(d, tau * s) for s in scales], axis=0)
        for p, d in distances.items()
    }


def code_rbf_features(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    anchors = np.quantile(values["train"], np.linspace(0, 1, 16))
    distances = {p: np.abs(x[:, None] - anchors[None, :]) for p, x in values.items()}
    tau = max(float(np.max(np.min(distances["train"], axis=1))), 1.0 / 15.0)
    return {p: normalized_kernel(d, tau) for p, d in distances.items()}


def feature_map(domain: DomainData, values: dict[str, np.ndarray], method: str, seed: int) -> dict[str, np.ndarray]:
    if method == "linear":
        lo, hi = np.min(values["train"]), np.max(values["train"])
        out = {}
        for p, x in values.items():
            z = np.zeros((len(x), 16), dtype=np.float64)
            z[:, 0] = (x - lo) / max(float(hi - lo), 1e-12)
            out[p] = z
        return out
    if method == "ple":
        return ple_fit_transform(values["train"], values)
    if method == "periodic":
        return periodic_transform(values["train"], values)
    if method == "code_rbf":
        return code_rbf_features(values)
    if method == "mpe_native":
        return native_features(domain, seed, False, False)
    if method == "mmpe_native":
        return native_features(domain, seed, True, False)
    if method == "mpe_corrupt":
        return native_features(domain, seed, False, True)
    raise KeyError(method)


def standardize(train: np.ndarray, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-10] = 1.0
    return {p: (x - mean) / std for p, x in arrays.items()}


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    xm, ym = x.mean(axis=0), float(y.mean())
    xc, yc = x - xm, y - ym
    gram = xc.T @ xc + alpha * np.eye(x.shape[1])
    coef = np.linalg.solve(gram, xc.T @ yc)
    intercept = ym - float(xm @ coef)
    return coef, intercept


def ridge_select(features: dict[str, np.ndarray], y: dict[str, np.ndarray]) -> dict[str, object]:
    x = standardize(features["train"], features)
    best = None
    for alpha in ALPHAS:
        coef, intercept = fit_ridge(x["train"], y["train"], float(alpha))
        pred = x["val"] @ coef + intercept
        loss = float(np.mean((pred - y["val"]) ** 2))
        if best is None or (loss, alpha) < (best[0], best[1]):
            best = (loss, float(alpha), coef, intercept)
    assert best is not None
    _, alpha, coef, intercept = best
    predictions = {p: x[p] @ coef + intercept for p in x}
    return {
        "alpha": alpha,
        "train_mse": float(np.mean((predictions["train"] - y["train"]) ** 2)),
        "val_mse": float(np.mean((predictions["val"] - y["val"]) ** 2)),
        "test_mse": float(np.mean((predictions["test"] - y["test"]) ** 2)),
        "predictions": predictions,
    }


def standardize_target(domain: DomainData) -> dict[str, np.ndarray]:
    mean = float(np.mean(domain.y["train"]))
    std = max(float(np.std(domain.y["train"])), 1e-12)
    return {p: (v - mean) / std for p, v in domain.y.items()}


def metric_distortion(domain: DomainData, values: dict[str, np.ndarray]) -> float:
    if domain.distance is None:
        latent = domain.semantic["train"][:300]
        native = np.abs(latent[:, None] - latent[None, :])
    else:
        ids = np.asarray(domain.semantic["train"][:300], dtype=np.int64)
        native = domain.distance[np.ix_(ids, ids)]
    code = np.abs(values["train"][:300, None] - values["train"][None, :300])
    iu = np.triu_indices_from(native, 1)
    if np.std(native[iu]) < 1e-12 or np.std(code[iu]) < 1e-12:
        return 0.0
    return float(1.0 - np.corrcoef(native[iu], code[iu])[0, 1])


def run(output: Path, seeds: list[int], schemas: int) -> None:
    rows: list[dict[str, object]] = []
    predictions: dict[tuple[str, int, int, str], np.ndarray] = {}
    started = time.time()
    for domain_name in DOMAINS:
        for seed in seeds:
            domain = make_domain(domain_name, seed)
            y = standardize_target(domain)
            for schema in range(schemas):
                values = stored_values(domain, schema, seed)
                distortion = metric_distortion(domain, values)
                for method in METHODS:
                    features = feature_map(domain, values, method, seed)
                    assert all(x.shape[1] == 16 for x in features.values())
                    assert all(np.isfinite(x).all() for x in features.values())
                    if method.startswith("mpe") or method.startswith("mmpe") or method == "code_rbf":
                        assert np.allclose(features["train"].sum(axis=1), 1.0, atol=1e-10)
                    result = ridge_select(features, y)
                    predictions[(domain_name, seed, schema, method)] = result.pop("predictions") ["test"]
                    rows.append({
                        "domain": domain_name,
                        "seed": seed,
                        "schema": schema,
                        "method": method,
                        "dimension": 16,
                        "metric_distortion": distortion,
                        "test_scope": "unseen_states" if domain.held_out is not None else "all_states",
                        **result,
                    })
            native = [predictions[(domain_name, seed, s, "mmpe_native")] for s in range(schemas)]
            reference = native[0]
            if max(float(np.max(np.abs(p - reference))) for p in native[1:]) > 1e-10:
                raise AssertionError("native MMPE is not chart invariant")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "protocol": "PROTOCOL_FREEZE.md",
        "methods": METHODS,
        "domains": DOMAINS,
        "seeds": seeds,
        "schemas": schemas,
        "expected_rows": len(DOMAINS) * len(seeds) * schemas * len(METHODS),
        "actual_rows": len(rows),
        "elapsed_seconds": time.time() - started,
        "target_access": "feature maps and landmarks use training inputs only; validation labels select ridge alpha",
        "max_native_mmpe_schema_prediction_discrepancy": max(
            float(np.max(np.abs(predictions[(d, s, k, "mmpe_native")] - predictions[(d, s, 0, "mmpe_native")])))
            for d in DOMAINS for s in seeds for k in range(schemas)
        ),
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RESULTS / "ridge_screen.csv")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20260880, 20260892)))
    parser.add_argument("--schemas", type=int, default=8)
    args = parser.parse_args()
    run(args.output, args.seeds, args.schemas)


if __name__ == "__main__":
    main()
