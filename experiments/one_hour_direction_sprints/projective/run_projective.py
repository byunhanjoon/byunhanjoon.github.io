#!/usr/bin/env python3
"""One-hour static-tabular projective-law falsification sprint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits, load_wine
from sklearn.decomposition import PCA
from torch import Tensor, nn


HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
DIM = 8
CONTEXT = 16
QUERIES = 12
RANK = 8
WIDTH = 128
TRAIN_STEPS = 5_000
BATCH = 192
EVAL_EPISODES = 1_536
SEEDS = (2027083101, 2027083102, 2027083103)
TRAIN_FAMILIES = ("linear", "additive", "interaction", "stump")
QUERY_FAMILIES = ("point", "subset", "difference", "dense", "scaled_dense")
OOD_QUERIES = ("dense", "scaled_dense")
MAX_SECONDS = 60 * 60


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def protocol_hash() -> str:
    return hashlib.sha256((HERE / "PROTOCOL.md").read_bytes()).hexdigest()


def empirical_domains(device: torch.device) -> dict[str, Tensor]:
    loaders = {
        "diabetes": load_diabetes,
        "wine": load_wine,
        "breast_cancer": load_breast_cancer,
        "digits": load_digits,
    }
    result: dict[str, Tensor] = {}
    for name, loader in loaders.items():
        x = np.asarray(loader().data, dtype=np.float64)
        x = np.nan_to_num(x, nan=np.nanmedian(x, axis=0))
        x = (x - x.mean(axis=0)) / np.where(x.std(axis=0) > 0, x.std(axis=0), 1.0)
        if x.shape[1] > DIM:
            x = PCA(n_components=DIM, svd_solver="full").fit_transform(x)
        elif x.shape[1] < DIM:
            x = np.pad(x, ((0, 0), (0, DIM - x.shape[1])))
        x = x / np.where(x.std(axis=0) > 0, x.std(axis=0), 1.0)
        x = np.clip(x, -5.0, 5.0).astype(np.float32)
        result[name] = torch.from_numpy(x).to(device)
    return result


def gaussian_x(batch: int, rows: int, generator: torch.Generator, device: torch.device) -> Tensor:
    independent = torch.randn(batch, rows, DIM, generator=generator, device=device)
    common = torch.randn(batch, rows, 1, generator=generator, device=device)
    rho = 0.35 * torch.rand(batch, 1, 1, generator=generator, device=device)
    return torch.sqrt(1.0 - rho) * independent + torch.sqrt(rho) * common


def pooled_x(pool: Tensor, batch: int, rows: int, generator: torch.Generator) -> Tensor:
    indices = torch.randint(len(pool), (batch, rows), generator=generator, device=pool.device)
    x = pool[indices]
    # Feature signs and scales prevent memorizing one fixed empirical coordinate convention.
    signs = 2 * torch.randint(2, (batch, 1, DIM), generator=generator, device=pool.device) - 1
    scales = torch.exp(0.15 * torch.randn(batch, 1, DIM, generator=generator, device=pool.device))
    return x * signs * scales


def latent_function(x: Tensor, family: str, generator: torch.Generator) -> Tensor:
    batch = len(x)
    w = torch.randn(batch, DIM, generator=generator, device=x.device)
    intercept = 0.45 * torch.randn(batch, 1, generator=generator, device=x.device)
    if family == "linear":
        value = torch.einsum("bnd,bd->bn", x, w) / math.sqrt(DIM)
    elif family == "additive":
        frequency = 0.7 + 1.8 * torch.rand(batch, DIM, generator=generator, device=x.device)
        value = torch.einsum("bnd,bd->bn", torch.sin(x * frequency[:, None]), w) / math.sqrt(DIM)
    elif family == "interaction":
        u = torch.randn(batch, DIM, generator=generator, device=x.device)
        v = torch.randn(batch, DIM, generator=generator, device=x.device)
        linear = torch.einsum("bnd,bd->bn", x, w) / math.sqrt(DIM)
        left = torch.einsum("bnd,bd->bn", x, u) / math.sqrt(DIM)
        right = torch.einsum("bnd,bd->bn", x, v) / math.sqrt(DIM)
        value = 0.55 * linear + 0.55 * left * right
    elif family == "stump":
        threshold = 0.8 * torch.randn(batch, DIM, generator=generator, device=x.device)
        value = torch.einsum(
            "bnd,bd->bn", torch.tanh(5.0 * (x - threshold[:, None])), w
        ) / math.sqrt(DIM)
    else:
        raise KeyError(family)
    return value + intercept


def episode(
    batch: int,
    family: str,
    generator: torch.Generator,
    device: torch.device,
    pool: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    rows = CONTEXT + QUERIES
    x = gaussian_x(batch, rows, generator, device) if pool is None else pooled_x(pool, batch, rows, generator)
    mean = latent_function(x, family, generator)
    noise_scale = 0.18 + 0.16 * torch.rand(batch, 1, generator=generator, device=device)
    y = mean + noise_scale * torch.randn(batch, rows, generator=generator, device=device)
    return x[:, :CONTEXT], y[:, :CONTEXT], x[:, CONTEXT:], y[:, CONTEXT:]


def coefficients(
    batch: int, family: str, generator: torch.Generator, device: torch.device
) -> Tensor:
    a = torch.zeros(batch, QUERIES, device=device)
    rows = torch.arange(batch, device=device)
    if family == "point":
        first = torch.randint(QUERIES, (batch,), generator=generator, device=device)
        a[rows, first] = 1.0
    elif family == "subset":
        mask = torch.rand(batch, QUERIES, generator=generator, device=device) < 0.35
        empty = ~mask.any(dim=1)
        if empty.any():
            fallback = torch.randint(QUERIES, (int(empty.sum()),), generator=generator, device=device)
            mask[empty, fallback] = True
        a = mask.float() / mask.sum(dim=1, keepdim=True)
    elif family == "difference":
        first = torch.randint(QUERIES, (batch,), generator=generator, device=device)
        offset = torch.randint(1, QUERIES, (batch,), generator=generator, device=device)
        second = (first + offset) % QUERIES
        a[rows, first] = 1.0
        a[rows, second] = -1.0
    elif family in {"dense", "scaled_dense"}:
        a = torch.randn(batch, QUERIES, generator=generator, device=device)
        a = a / a.norm(dim=1, keepdim=True).clamp_min(1e-8)
        if family == "scaled_dense":
            scale = 1.75 + 1.25 * torch.rand(batch, 1, generator=generator, device=device)
            a = scale * a
    else:
        raise KeyError(family)
    return a


class ContextEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DIM + 1, WIDTH), nn.GELU(),
            nn.Linear(WIDTH, WIDTH), nn.GELU(),
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        tokens = self.net(torch.cat((x, y[:, :, None]), dim=-1))
        return tokens.mean(dim=1)


class ProjectiveModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.context = ContextEncoder()
        self.row = nn.Sequential(
            nn.Linear(WIDTH + DIM, WIDTH), nn.GELU(),
            nn.Linear(WIDTH, WIDTH), nn.GELU(),
            nn.Linear(WIDTH, 2 + RANK),
        )

    def joint(self, xc: Tensor, yc: Tensor, xq: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        context = self.context(xc, yc)
        h = self.row(torch.cat((xq, context[:, None].expand(-1, QUERIES, -1)), dim=-1))
        mean = h[:, :, 0]
        factor = h[:, :, 1 : 1 + RANK] / math.sqrt(RANK)
        diagonal = nn.functional.softplus(h[:, :, -1]) + 1e-3
        return mean, factor, diagonal

    def forward(self, xc: Tensor, yc: Tensor, xq: Tensor, a: Tensor) -> tuple[Tensor, Tensor]:
        mean, factor, diagonal = self.joint(xc, yc, xq)
        projected_mean = torch.sum(a * mean, dim=1)
        projected_factor = torch.einsum("bq,bqr->br", a, factor)
        variance = projected_factor.square().sum(dim=1) + (a.square() * diagonal.square()).sum(dim=1)
        return projected_mean, variance.clamp_min(1e-6)


class DirectModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.context = ContextEncoder()
        self.query = nn.Sequential(
            nn.Linear(DIM + 1, 160), nn.GELU(),
            nn.Linear(160, 160), nn.GELU(),
        )
        self.output = nn.Sequential(
            nn.Linear(WIDTH + 160 + 4, 192), nn.GELU(),
            nn.Linear(192, 192), nn.GELU(),
            nn.Linear(192, 2),
        )

    def forward(self, xc: Tensor, yc: Tensor, xq: Tensor, a: Tensor) -> tuple[Tensor, Tensor]:
        context = self.context(xc, yc)
        query = self.query(torch.cat((xq, a[:, :, None]), dim=-1)).mean(dim=1)
        stats = torch.stack((a.sum(1), a.norm(dim=1), a.max(1).values, a.min(1).values), dim=1)
        output = self.output(torch.cat((context, query, stats), dim=1))
        return output[:, 0], nn.functional.softplus(output[:, 1]) + 1e-6


def gaussian_nll(mean: Tensor, variance: Tensor, target: Tensor) -> Tensor:
    return 0.5 * (math.log(2.0 * math.pi) + torch.log(variance) + (target - mean).square() / variance)


def train(seed: int, device: torch.device) -> tuple[ProjectiveModel, DirectModel, dict]:
    seed_everything(seed)
    generator = torch.Generator(device=device).manual_seed(seed + 101)
    projective = ProjectiveModel().to(device)
    direct = DirectModel().to(device)
    p_count = sum(parameter.numel() for parameter in projective.parameters())
    d_count = sum(parameter.numel() for parameter in direct.parameters())
    if d_count < p_count:
        raise AssertionError("direct comparator must be capacity-matched or larger")
    optimizers = {
        "projective": torch.optim.AdamW(projective.parameters(), lr=5e-4, weight_decay=1e-5),
        "direct": torch.optim.AdamW(direct.parameters(), lr=5e-4, weight_decay=1e-5),
    }
    started = time.perf_counter()
    losses = {"projective": [], "direct": []}
    models = {"projective": projective, "direct": direct}
    for step in range(TRAIN_STEPS):
        family = TRAIN_FAMILIES[step % len(TRAIN_FAMILIES)]
        query_family = QUERY_FAMILIES[step % 3]
        xc, yc, xq, yq = episode(BATCH, family, generator, device)
        a = coefficients(BATCH, query_family, generator, device)
        target = torch.sum(a * yq, dim=1)
        for name, model in models.items():
            optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            mean, variance = model(xc, yc, xq, a)
            loss = gaussian_nll(mean, variance, target).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step >= TRAIN_STEPS - 100:
                losses[name].append(float(loss.detach()))
        if time.perf_counter() - started > MAX_SECONDS * 0.82:
            raise TimeoutError("training exceeded the reserved one-hour budget")
    return projective, direct, {
        "train_seconds": time.perf_counter() - started,
        "projective_parameters": p_count,
        "direct_parameters": d_count,
        "projective_final_nll": float(np.mean(losses["projective"])),
        "direct_final_nll": float(np.mean(losses["direct"])),
    }


@torch.no_grad()
def evaluate_cell(
    model: nn.Module,
    model_name: str,
    seed: int,
    task_family: str,
    query_family: str,
    domain: str,
    device: torch.device,
    pool: Tensor | None,
) -> dict:
    generator = torch.Generator(device=device).manual_seed(
        int(hashlib.sha256(f"eval|{seed}|{task_family}|{query_family}|{domain}".encode()).hexdigest()[:15], 16)
    )
    model.eval()
    sums = {"nll": 0.0, "squared": 0.0, "covered": 0.0, "count": 0}
    for start in range(0, EVAL_EPISODES, 256):
        size = min(256, EVAL_EPISODES - start)
        xc, yc, xq, yq = episode(size, task_family, generator, device, pool)
        a = coefficients(size, query_family, generator, device)
        target = torch.sum(a * yq, dim=1)
        mean, variance = model(xc, yc, xq, a)
        sums["nll"] += float(gaussian_nll(mean, variance, target).sum())
        sums["squared"] += float((mean - target).square().sum())
        radius = 1.6448536269514722 * torch.sqrt(variance)
        sums["covered"] += float(((target >= mean - radius) & (target <= mean + radius)).sum())
        sums["count"] += size
    return {
        "seed": seed,
        "model": model_name,
        "task_family": task_family,
        "query_family": query_family,
        "domain": domain,
        "nll": sums["nll"] / sums["count"],
        "rmse": math.sqrt(sums["squared"] / sums["count"]),
        "coverage90": sums["covered"] / sums["count"],
    }


@torch.no_grad()
def identities(model: nn.Module, seed: int, device: torch.device) -> dict[str, float]:
    generator = torch.Generator(device=device).manual_seed(seed + 909)
    xc, yc, xq, _ = episode(2_048, "interaction", generator, device)
    a = coefficients(len(xc), "dense", generator, device)
    b = coefficients(len(xc), "dense", generator, device)
    scale = 0.3 + 2.4 * torch.rand(len(xc), generator=generator, device=device)
    ma, va = model(xc, yc, xq, a)
    mb, vb = model(xc, yc, xq, b)
    mapb, vapb = model(xc, yc, xq, a + b)
    mamb, vamb = model(xc, yc, xq, a - b)
    msa, vsa = model(xc, yc, xq, scale[:, None] * a)

    def relative(error: Tensor, reference: Tensor) -> float:
        return float(torch.sqrt(error.square().mean()) / torch.sqrt(reference.square().mean()).clamp_min(1e-8))

    return {
        "mean_additivity": relative(mapb - ma - mb, mapb),
        "mean_scaling": relative(msa - scale * ma, scale * ma),
        "variance_scaling": relative(vsa - scale.square() * va, scale.square() * va),
        "variance_polarization": relative(vapb + vamb - 2 * va - 2 * vb, vapb + vamb),
    }


def run_seed(seed: int, device: torch.device) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    projective, direct, training = train(seed, device)
    domains = empirical_domains(device)
    domains = {"gaussian": None, **domains}
    rows = []
    for domain, pool in domains.items():
        for task_family in TRAIN_FAMILIES:
            for query_family in QUERY_FAMILIES:
                for name, model in (("projective", projective), ("direct", direct)):
                    rows.append(
                        evaluate_cell(
                            model, name, seed, task_family, query_family,
                            domain, device, pool,
                        )
                    )
    pd.DataFrame(rows).to_csv(OUT / f"cells_seed{seed}.csv", index=False)
    payload = {
        "seed": seed,
        "protocol_sha256": protocol_hash(),
        "wall_seconds": time.perf_counter() - started,
        "training": training,
        "identities": {
            "projective": identities(projective, seed, device),
            "direct": identities(direct, seed, device),
        },
    }
    (OUT / f"audit_seed{seed}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    torch.save(projective.state_dict(), OUT / f"projective_seed{seed}.pt")
    torch.save(direct.state_dict(), OUT / f"direct_seed{seed}.pt")
    print(json.dumps(payload, indent=2, sort_keys=True))


def analyze() -> dict:
    cell_paths = [OUT / f"cells_seed{seed}.csv" for seed in SEEDS]
    audit_paths = [OUT / f"audit_seed{seed}.json" for seed in SEEDS]
    missing = [str(path) for path in cell_paths + audit_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"incomplete projective panel: {missing}")
    cells = pd.concat([pd.read_csv(path) for path in cell_paths], ignore_index=True)
    audits = [json.loads(path.read_text()) for path in audit_paths]
    cells.to_csv(OUT / "cells.csv", index=False)
    paired = cells.pivot(
        index=["seed", "task_family", "query_family", "domain"],
        columns="model", values="nll",
    ).reset_index()
    paired["projective_advantage"] = paired.direct - paired.projective
    paired.to_csv(OUT / "paired.csv", index=False)
    ood = paired[paired.query_family.isin(OOD_QUERIES)]
    point = paired[paired.query_family == "point"]
    empirical = ood[ood.domain != "gaussian"].groupby("domain").projective_advantage.mean()
    identity_names = ("mean_additivity", "mean_scaling", "variance_scaling", "variance_polarization")
    projective_identity = {
        key: float(np.mean([audit["identities"]["projective"][key] for audit in audits]))
        for key in identity_names
    }
    direct_identity = {
        key: float(np.mean([audit["identities"]["direct"][key] for audit in audits]))
        for key in identity_names
    }
    maximum_projective_identity = max(
        audit["identities"]["projective"][key]
        for audit in audits for key in identity_names
    )
    metrics = {
        "ood_mean_nll_advantage": float(ood.projective_advantage.mean()),
        "ood_cell_win_rate": float((ood.projective_advantage > 0).mean()),
        "empirical_domains_won": int((empirical > 0).sum()),
        "empirical_domain_advantage": {key: float(value) for key, value in empirical.items()},
        "point_mean_nll_advantage": float(point.projective_advantage.mean()),
        "projective_identity_mean": projective_identity,
        "direct_identity_mean": direct_identity,
        "maximum_projective_identity": float(maximum_projective_identity),
        "projective_parameters": int(audits[0]["training"]["projective_parameters"]),
        "direct_parameters": int(audits[0]["training"]["direct_parameters"]),
        "total_wall_seconds": float(sum(audit["wall_seconds"] for audit in audits)),
    }
    integrity = bool(
        len(cells) == len(SEEDS) * 5 * len(TRAIN_FAMILIES) * len(QUERY_FAMILIES) * 2
        and np.isfinite(cells[["nll", "rmse", "coverage90"]]).all().all()
        and all(audit["protocol_sha256"] == protocol_hash() for audit in audits)
    )
    gates = {
        "integrity": integrity,
        "direct_capacity_at_least_projective": metrics["direct_parameters"] >= metrics["projective_parameters"],
        "ood_nll_advantage_at_least_0_05": metrics["ood_mean_nll_advantage"] >= 0.05,
        "ood_win_rate_at_least_70pct": metrics["ood_cell_win_rate"] >= 0.70,
        "at_least_3_empirical_domains_won": metrics["empirical_domains_won"] >= 3,
        "point_nll_degradation_at_most_0_02": metrics["point_mean_nll_advantage"] >= -0.02,
        "projective_identities_below_1e_5": metrics["maximum_projective_identity"] < 1e-5,
        "direct_exposes_at_least_2_violations": sum(value > 0.01 for value in direct_identity.values()) >= 2,
    }
    result = {
        "status": "complete_static_semisynthetic_replay",
        "protocol_sha256": protocol_hash(),
        "metrics": metrics,
        "gates": gates,
        "passed": all(gates.values()),
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.analyze:
        analyze()
    elif args.seed is not None:
        run_seed(args.seed, torch.device(args.device))
    else:
        parser.error("choose --seed or --analyze")


if __name__ == "__main__":
    main()
