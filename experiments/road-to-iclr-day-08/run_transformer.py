"""Bounded FT-Transformer geometry diagnostic (two synthetic, two real)."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch import Tensor, nn

from day8_core import HERE, ArrayData, BasisMap, evaluate_predictions, load_real_dataset, make_synthetic, seed_all


OUT = HERE / "raw/transformer"


class NumericTokenizer(nn.Module):
    def __init__(self, data: ArrayData, kind: str, d_token: int = 24, bins: int = 8) -> None:
        super().__init__()
        self.kind, self.n_num, self.d_token, self.bins = kind, data.n_num, d_token, bins
        self.map = BasisMap(kind, data.x_num["train"], 0, bins)
        width = bins if kind == "ple" else 1
        self.weight = nn.Parameter(torch.randn(self.n_num, width, d_token) * 0.08)
        self.bias = nn.Parameter(torch.zeros(self.n_num, d_token))

    def forward(self, x: Tensor) -> Tensor:
        if self.kind == "ple":
            basis = self.map._ple(x)
        else:
            basis = self.map(x, x.new_empty((len(x), 0)))[:, :, None]
        return torch.einsum("nfb,fbd->nfd", basis, self.weight) + self.bias[None]


class FTTransformer(nn.Module):
    def __init__(self, data: ArrayData, kind: str, d_token: int = 24) -> None:
        super().__init__()
        self.tokenizer = NumericTokenizer(data, kind, d_token)
        self.cat_projection = nn.Linear(data.n_cat, d_token) if data.n_cat else None
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_token, 4, d_token * 2, 0.1, batch_first=True, norm_first=True, activation="gelu")
            for _ in range(2)
        ])
        out = 1 if data.task == "regression" else data.n_classes
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.Linear(d_token, out))
        self.task = data.task

    def forward(self, x_num: Tensor, x_cat: Tensor, stages: bool = False):
        numeric = self.tokenizer(x_num)
        tokens = numeric
        if self.cat_projection is not None:
            tokens = torch.cat((tokens, self.cat_projection(x_cat)[:, None]), dim=1)
        initial = tokens
        z = torch.cat((self.cls.expand(len(tokens), -1, -1), tokens), dim=1)
        z = self.blocks[0](z)
        first = z[:, 1:]
        z = self.blocks[1](z)
        output = self.head(z[:, 0])
        return (output, initial, first, z[:, 0]) if stages else output


@torch.no_grad()
def predict(model: FTTransformer, data: ArrayData, part: str, device: torch.device, stages: bool = False):
    model.eval(); outs, a, b, c = [], [], [], []
    xn = torch.tensor(data.x_num[part], device=device); xc = torch.tensor(data.x_cat[part], device=device)
    for start in range(0, len(xn), 256):
        result = model(xn[start:start + 256], xc[start:start + 256], stages)
        if stages:
            out, x0, x1, x2 = result; a.append(x0.cpu()); b.append(x1.cpu()); c.append(x2.cpu())
        else: out = result
        outs.append(out.cpu())
    raw = torch.cat(outs).numpy()
    if data.task == "classification": raw = torch.softmax(torch.from_numpy(raw), dim=1).numpy()
    if stages: return raw, torch.cat(a).numpy(), torch.cat(b).numpy(), torch.cat(c).numpy()
    return raw


def train(data: ArrayData, kind: str, device: torch.device, seed: int = 20260831) -> tuple[FTTransformer, dict[str, float]]:
    seed_all(seed); model = FTTransformer(data, kind).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-5)
    xn = torch.tensor(data.x_num["train"], device=device); xc = torch.tensor(data.x_cat["train"], device=device); y = torch.tensor(data.y["train"], device=device)
    rng = np.random.default_rng(seed); best, best_loss, stale = None, math.inf, 0
    for epoch in range(20):
        model.train()
        for _ in range(min(16, max(8, math.ceil(len(y) / 128)))):
            idx = torch.tensor(rng.choice(len(y), min(128, len(y)), replace=False), device=device)
            out = model(xn[idx], xc[idx])
            loss = nn.functional.mse_loss(out[:, 0], y[idx].float()) if data.task == "regression" else nn.functional.cross_entropy(out, y[idx].long())
            optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        val = evaluate_predictions(predict(model, data, "validation", device), data.y["validation"], data)
        if val["loss"] < best_loss - 1e-5:
            best_loss, best, stale = val["loss"], copy.deepcopy(model.state_dict()), 0
        else: stale += 1
        if epoch >= 6 and stale >= 4: break
    if best is not None: model.load_state_dict(best)
    metrics = evaluate_predictions(predict(model, data, "test", device), data.y["test"], data)
    metrics.update(epochs=epoch + 1, parameters=sum(p.numel() for p in model.parameters()), best_validation_loss=best_loss)
    return model, metrics


def distance_correlation(initial: np.ndarray, first: np.ndarray, final: np.ndarray, seed: int = 20260831) -> dict[str, float]:
    rng = np.random.default_rng(seed); n = len(initial); pairs = rng.integers(0, n, size=(2, min(10000, n * 20)))
    keep = pairs[0] != pairs[1]; i, j = pairs[0, keep], pairs[1, keep]
    d0 = np.square(initial[i].reshape(len(i), -1) - initial[j].reshape(len(i), -1)).sum(axis=1)
    d1 = np.square(first[i].reshape(len(i), -1) - first[j].reshape(len(i), -1)).sum(axis=1)
    d2 = np.square(final[i] - final[j]).sum(axis=1)
    return {"token_to_first_distance_spearman": float(spearmanr(d0, d1).statistic), "token_to_final_distance_spearman": float(spearmanr(d0, d2).statistic), "first_to_final_distance_spearman": float(spearmanr(d1, d2).statistic)}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--device", default="cuda:0"); args = parser.parse_args()
    device = torch.device(args.device); OUT.mkdir(parents=True, exist_ok=True); rows = []
    tasks = ("S1_rotating", "S4_warp", "california", "higgs-small")
    for task in tasks:
        data = make_synthetic(task, 20260831)[0] if task.startswith("S") else load_real_dataset(task)
        for kind in ("raw", "ple", "localwarp"):
            path = OUT / f"{task}__{kind}.json"
            if path.exists(): rows.append(json.loads(path.read_text())); continue
            started = time.perf_counter(); print(f"transformer {task}/{kind}", flush=True)
            model, metrics = train(data, kind, device)
            _, initial, first, final = predict(model, data, "test", device, True)
            row = {"status": "complete", "task": task, "scope": "synthetic" if task.startswith("S") else "real", "representation": kind, "wall_seconds": time.perf_counter() - started, **metrics, **distance_correlation(initial, first, final)}
            path.write_text(json.dumps(row, indent=2)); rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False)


if __name__ == "__main__": main()
