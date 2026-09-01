#!/usr/bin/env python3
"""Post-gate stress tests for static projectivity's direct comparator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_projective as base  # noqa: E402


OUT = HERE / "results" / "direct_controls"
CONTROLS = ("direct_long", "direct_broad", "direct_moment")
EVAL_QUERIES = ("point", "dense", "scaled_dense")


def protocol_hash() -> str:
    return hashlib.sha256((HERE / "STRESS_CONTROL_PROTOCOL.md").read_bytes()).hexdigest()


class DirectMomentModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.context = base.ContextEncoder()
        self.first = nn.Sequential(
            nn.Linear(base.DIM, 160), nn.GELU(), nn.Linear(160, 160), nn.GELU()
        )
        self.second = nn.Sequential(
            nn.Linear(base.DIM, 160), nn.GELU(), nn.Linear(160, 160), nn.GELU()
        )
        self.output = nn.Sequential(
            nn.Linear(base.WIDTH + 320 + 4, 192), nn.GELU(),
            nn.Linear(192, 192), nn.GELU(), nn.Linear(192, 2),
        )

    def forward(self, xc: Tensor, yc: Tensor, xq: Tensor, a: Tensor) -> tuple[Tensor, Tensor]:
        context = self.context(xc, yc)
        first = (a[:, :, None] * self.first(xq)).sum(dim=1) / math.sqrt(base.QUERIES)
        second = (a.square()[:, :, None] * self.second(xq)).sum(dim=1) / math.sqrt(base.QUERIES)
        stats = torch.stack((a.sum(1), a.norm(dim=1), a.max(1).values, a.min(1).values), dim=1)
        output = self.output(torch.cat((context, first, second, stats), dim=1))
        return output[:, 0], nn.functional.softplus(output[:, 1]) + 1e-6


def train_control(name: str, seed: int, device: torch.device) -> tuple[nn.Module, dict]:
    base.seed_everything(seed + {"direct_long": 11, "direct_broad": 22, "direct_moment": 33}[name])
    generator = torch.Generator(device=device).manual_seed(seed + 101)
    model: nn.Module = DirectMomentModel() if name == "direct_moment" else base.DirectModel()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    steps = 20_000 if name == "direct_long" else base.TRAIN_STEPS
    started = time.perf_counter()
    tail = []
    for step in range(steps):
        family = base.TRAIN_FAMILIES[step % len(base.TRAIN_FAMILIES)]
        if name == "direct_broad":
            query_family = base.QUERY_FAMILIES[step % len(base.QUERY_FAMILIES)]
        else:
            query_family = base.QUERY_FAMILIES[step % 3]
        xc, yc, xq, yq = base.episode(base.BATCH, family, generator, device)
        a = base.coefficients(base.BATCH, query_family, generator, device)
        target = torch.sum(a * yq, dim=1)
        optimizer.zero_grad(set_to_none=True)
        mean, variance = model(xc, yc, xq, a)
        loss = base.gaussian_nll(mean, variance, target).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step >= steps - 100:
            tail.append(float(loss.detach()))
        if time.perf_counter() - started > base.MAX_SECONDS * 0.85:
            raise TimeoutError(f"{name} exceeded stress-control time reserve")
    return model, {
        "updates": steps,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_seconds": time.perf_counter() - started,
        "final_nll": float(np.mean(tail)),
    }


def run_seed(seed: int, device: torch.device) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    domains = base.empirical_domains(device)
    domains = {"gaussian": None, **domains}
    rows = []
    training = {}
    started = time.perf_counter()
    for name in CONTROLS:
        model, meta = train_control(name, seed, device)
        training[name] = meta
        for domain, pool in domains.items():
            for task_family in base.TRAIN_FAMILIES:
                for query_family in EVAL_QUERIES:
                    rows.append(
                        base.evaluate_cell(
                            model, name, seed, task_family, query_family,
                            domain, device, pool,
                        )
                    )
        torch.save(model.state_dict(), OUT / f"{name}_seed{seed}.pt")
    pd.DataFrame(rows).to_csv(OUT / f"cells_seed{seed}.csv", index=False)
    audit = {
        "seed": seed,
        "protocol_sha256": protocol_hash(),
        "wall_seconds": time.perf_counter() - started,
        "training": training,
    }
    (OUT / f"audit_seed{seed}.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


def analyze() -> dict:
    paths = [OUT / f"cells_seed{seed}.csv" for seed in base.SEEDS]
    audits = [OUT / f"audit_seed{seed}.json" for seed in base.SEEDS]
    missing = [str(path) for path in paths + audits if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    controls = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    primary = pd.read_csv(base.OUT / "cells.csv")
    primary = primary[
        (primary.model == "projective") & primary.query_family.isin(EVAL_QUERIES)
    ]
    keys = ["seed", "task_family", "query_family", "domain"]
    table = controls.pivot(index=keys, columns="model", values="nll").join(
        primary.set_index(keys).nll.rename("projective")
    ).reset_index()
    for control in CONTROLS:
        table[f"advantage_vs_{control}"] = table[control] - table.projective
    table["best_control"] = table[list(CONTROLS)].min(axis=1)
    table["advantage_vs_best_control"] = table.best_control - table.projective
    table.to_csv(OUT / "paired.csv", index=False)
    comparisons = {}
    for control in (*CONTROLS, "best_control"):
        column = f"advantage_vs_{control}"
        comparisons[control] = {
            query: {
                "mean_projective_nll_advantage": float(group[column].mean()),
                "projective_win_rate": float((group[column] > 0).mean()),
                "cells": len(group),
            }
            for query, group in table.groupby("query_family")
        }
    dense = table[table.query_family == "dense"].advantage_vs_best_control
    point = table[table.query_family == "point"].advantage_vs_best_control
    audits_loaded = [json.loads(path.read_text()) for path in audits]
    gates = {
        "integrity": bool(
            len(table) == len(base.SEEDS) * len(base.TRAIN_FAMILIES) * 5 * len(EVAL_QUERIES)
            and np.isfinite(table.select_dtypes(include=[np.number])).all().all()
            and all(audit["protocol_sha256"] == protocol_hash() for audit in audits_loaded)
        ),
        "dense_advantage_at_least_0_05": float(dense.mean()) >= 0.05,
        "dense_win_rate_at_least_70pct": float((dense > 0).mean()) >= 0.70,
        "point_degradation_at_most_0_02": float(point.mean()) >= -0.02,
    }
    result = {
        "status": "complete_postgate_stress_control",
        "protocol_sha256": protocol_hash(),
        "comparisons": comparisons,
        "gates": gates,
        "stress_robust": all(gates.values()),
        "total_wall_seconds": float(sum(audit["wall_seconds"] for audit in audits_loaded)),
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=base.SEEDS)
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
