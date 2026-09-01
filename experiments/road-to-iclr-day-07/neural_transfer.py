#!/usr/bin/env python3
"""Development test of Geometry Transfer with a fixed neural tabular base."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tabm
import torch
from rtdl_revisiting_models import FTTransformerBackbone
from scipy import sparse
from scipy.stats import spearmanr
from torch import nn


HERE = Path(__file__).resolve().parent
GEOMETRY = HERE.parent / "geometry_transfer"
MPE = HERE.parent / "mpe_iclr"
for path in (GEOMETRY, MPE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from geometry_transfer import empirical_gain, operator_family, stable_seed, state_means  # noqa: E402
from representations import load_task, split_state_indices  # noqa: E402
from ridge_benchmark import ordinary_design  # noqa: E402


TASKS = ["acs_occupation", "tlc_pickup_zone", "airline_origin_airport", "medical_charges"]
SPLITS = [0, 1]
OUT = HERE / "results" / "neural"


class BaseMLP(nn.Module):
    def __init__(self, width_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width_in, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ResidualBlock(nn.Module):
    def __init__(self, width: int = 128):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width * 2), nn.ReLU(),
            nn.Linear(width * 2, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class BaseResNet(nn.Module):
    def __init__(self, width_in: int):
        super().__init__()
        self.input = nn.Linear(width_in, 128)
        self.blocks = nn.Sequential(ResidualBlock(), ResidualBlock())
        self.output = nn.Sequential(nn.LayerNorm(128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.blocks(self.input(x))).squeeze(-1)


class BaseFTTransformer(nn.Module):
    def __init__(self, width_in: int):
        super().__init__()
        self.n_tokens = 8
        self.d_token = 32
        self.stem = nn.Linear(width_in, self.n_tokens * self.d_token)
        self.cls = nn.Parameter(torch.empty(1, 1, self.d_token))
        nn.init.normal_(self.cls, std=self.d_token**-0.5)
        self.backbone = FTTransformerBackbone(
            d_out=1, n_blocks=2, d_block=self.d_token,
            attention_n_heads=4, attention_dropout=0.0,
            ffn_d_hidden=3 * self.d_token, ffn_d_hidden_multiplier=None,
            ffn_dropout=0.0, ffn_activation="ReGLU", residual_dropout=0.0,
            n_tokens=None, linformer_kv_compression_ratio=None,
            linformer_kv_compression_sharing=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.stem(x).reshape(len(x), self.n_tokens, self.d_token)
        tokens = torch.cat([self.cls.expand(len(x), -1, -1), tokens], dim=1)
        return self.backbone(tokens).squeeze(-1)


class BaseTabM(nn.Module):
    is_tabm = True

    def __init__(self, width_in: int):
        super().__init__()
        self.stem = nn.Linear(width_in, 64)
        self.backbone = tabm.TabM.make(
            n_num_features=64, cat_cardinalities=[], d_out=1,
            num_embeddings=None, n_blocks=2, d_block=128, dropout=0.0, k=8,
        )

    def forward_members(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.stem(x), None).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_members(x).mean(dim=1)


def make_base(backbone: str, width_in: int) -> nn.Module:
    constructors = {
        "mlp": BaseMLP,
        "resnet": BaseResNet,
        "ft_transformer": BaseFTTransformer,
        "tabm": BaseTabM,
    }
    return constructors[backbone](width_in)


def backbone_output(backbone: str) -> Path:
    # Preserve the already-sealed MLP paths while isolating all new outcomes.
    return OUT if backbone == "mlp" else OUT / backbone


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def state_preserving_sample(rows: np.ndarray, states: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if len(rows) <= limit:
        return rows
    rng = np.random.default_rng(seed)
    first = [rng.choice(rows[states[rows] == state]) for state in np.unique(states[rows])]
    pool = np.setdiff1d(rows, np.asarray(first))
    extra = rng.choice(pool, limit - len(first), replace=False)
    return np.sort(np.concatenate([np.asarray(first), extra]))


def fit_predict(
    design: torch.Tensor, target: torch.Tensor, states: np.ndarray,
    fit_rows: np.ndarray, predict_rows: np.ndarray, seed: int, device: torch.device,
    backbone: str = "mlp",
) -> np.ndarray:
    seed_all(seed)
    fit_rows = state_preserving_sample(fit_rows, states, 50_000, seed)
    unique, counts = np.unique(states[fit_rows], return_counts=True)
    lookup = dict(zip(unique.tolist(), counts.tolist()))
    weights = torch.as_tensor([1.0 / lookup[states[i]] for i in fit_rows], dtype=torch.float32, device=device)
    weights *= len(weights) / weights.sum()
    fit_index = torch.as_tensor(fit_rows, dtype=torch.long, device=device)
    model = make_base(backbone, design.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = np.random.default_rng(seed + 17)
    for _ in range(40):
        order = generator.permutation(len(fit_rows))
        model.train()
        for start in range(0, len(order), 2048):
            pos = torch.as_tensor(order[start:start + 2048], dtype=torch.long, device=device)
            rows = fit_index.index_select(0, pos)
            features = design.index_select(0, rows)
            if getattr(model, "is_tabm", False):
                prediction = model.forward_members(features)
                loss_rows = ((prediction - target.index_select(0, rows)[:, None]) ** 2).mean(dim=1)
            else:
                prediction = model(features)
                loss_rows = (prediction - target.index_select(0, rows)) ** 2
            loss = (loss_rows * weights.index_select(0, pos)).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    model.eval(); output = []
    with torch.inference_mode():
        for start in range(0, len(predict_rows), 8192):
            idx = torch.as_tensor(predict_rows[start:start + 8192], dtype=torch.long, device=device)
            output.append(model(design.index_select(0, idx)).cpu().numpy())
    return np.concatenate(output).astype(np.float64)


def base_residuals(task, split: int, device: torch.device, backbone: str = "mlp") -> dict[str, np.ndarray]:
    path = backbone_output(backbone) / "base_cache" / f"{task.name}__split{split}.npz"
    if path.exists():
        with np.load(path) as data:
            return {key: data[key] for key in data.files}
    parts = split_state_indices(task, split)
    observed = np.sort(np.concatenate([parts["train"], parts["validation"]]))
    unseen = parts["test"]
    row_state = task.row_state_indices()
    tr = np.flatnonzero(np.isin(row_state, observed)); ur = np.flatnonzero(np.isin(row_state, unseen))
    raw_y = pd.to_numeric(task.rows.target, errors="raise").to_numpy(np.float64)
    center = raw_y[tr].mean(); scale = raw_y[tr].std() or 1.0; y = (raw_y - center) / scale
    design_sparse = ordinary_design(task, tr)
    if design_sparse.shape[1] == 0:
        design_sparse = sparse.csr_matrix(np.ones((len(y), 1), dtype=np.float32))
    design = torch.from_numpy(design_sparse.toarray().astype(np.float32, copy=False)).to(device)
    target = torch.from_numpy(y.astype(np.float32)).to(device)
    order = tr.copy(); np.random.default_rng(stable_seed("day7-neural-rowfold", task.name, split)).shuffle(order)
    folds = np.array_split(order, 3); oof = np.empty(len(tr)); position = {row: i for i, row in enumerate(tr.tolist())}
    for k, held in enumerate(folds):
        fit = np.setdiff1d(tr, held)
        pred = fit_predict(
            design, target, row_state, fit, held,
            stable_seed("day7-neural-oof", backbone, task.name, split, k), device, backbone,
        )
        oof[[position[row] for row in held]] = pred
    pred_u = fit_predict(
        design, target, row_state, tr, ur,
        stable_seed("day7-neural-full", backbone, task.name, split), device, backbone,
    )
    payload = {
        "t_states": observed, "u_states": unseen, "row_state_t": row_state[tr], "row_state_u": row_state[ur],
        "residual_t": y[tr] - oof, "residual_u": y[ur] - pred_u,
        "ordinary_dimensions": np.asarray([design.shape[1]]), "oof_finite": np.asarray([np.isfinite(oof).all()]),
    }
    path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(path, **payload)
    return payload


def local_index(rows: np.ndarray, ordered: np.ndarray) -> np.ndarray:
    lookup = {state: i for i, state in enumerate(ordered.tolist())}
    return np.asarray([lookup[state] for state in rows])


def inner_operator_scores(task, split: int, cache: dict[str, np.ndarray]) -> pd.DataFrame:
    states = cache["t_states"].copy(); residual = cache["residual_t"]; row_state = cache["row_state_t"]
    np.random.default_rng(stable_seed("day7-neural-statefold", task.name, split)).shuffle(states)
    folds = [np.sort(x) for x in np.array_split(states, 5) if len(x)]
    rows = []
    for k, held in enumerate(folds):
        train = np.setdiff1d(states, held); mu = state_means(residual, row_state, train)
        held_index = local_index(row_state[np.isin(row_state, held)], held)
        held_residual = residual[np.isin(row_state, held)]
        for name, matrix in operator_family(task.distance, train, held).items():
            gain = empirical_gain(held_residual, held_index, matrix @ mu)
            rows.append({"operator": name, "fold": k, "gain": gain})
    frame = pd.DataFrame(rows)
    return frame.groupby("operator").gain.agg(["mean", "std", "count"]).reset_index().assign(
        se=lambda x: x["std"] / np.sqrt(x["count"])
    )


def outer_operator_gains(task, cache: dict[str, np.ndarray]) -> dict[str, float]:
    train, test = cache["t_states"], cache["u_states"]
    mu = state_means(cache["residual_t"], cache["row_state_t"], train)
    index = local_index(cache["row_state_u"], test)
    return {
        name: empirical_gain(cache["residual_u"], index, matrix @ mu)
        for name, matrix in operator_family(task.distance, train, test).items()
    }


def run_cell(task_name: str, split: int, device: torch.device, backbone: str = "mlp") -> list[dict]:
    task = load_task(task_name); cache = base_residuals(task, split, device, backbone)
    inner = inner_operator_scores(task, split, cache); outer = outer_operator_gains(task, cache)
    winner = inner.sort_values(["mean", "operator"], ascending=[False, True]).iloc[0]
    rows = []
    for row in inner.itertuples(index=False):
        rows.append({
            "source": task.manifest["source_unit"], "task": task_name, "split": split,
            "backbone": backbone,
            "operator": row.operator, "predicted_gain": row.mean, "predicted_se": row.se,
            "actual_gain": outer[row.operator],
            "selected_mean": row.operator == winner["operator"] and winner["mean"] > 0,
            "selected_pessimistic": (
                row.operator == winner["operator"]
                and winner["mean"] - 1.96 * winner["se"] > 0
            ),
            "ordinary_dimensions": int(cache["ordinary_dimensions"][0]), "oof_finite": bool(cache["oof_finite"][0]),
        })
    return rows


def analyze(cells: pd.DataFrame) -> dict:
    aggregates = cells.groupby(["source", "operator"], as_index=False)[["predicted_gain", "actual_gain"]].mean()
    rho = float(spearmanr(aggregates.predicted_gain, aggregates.actual_gain).statistic)
    sign = float(np.mean((aggregates.predicted_gain > 0) == (aggregates.actual_gain > 0)))
    selectors = {}
    for column in ("selected_mean", "selected_pessimistic"):
        chosen = cells[cells[column]].groupby(["source", "task", "split"], as_index=False).actual_gain.sum()
        all_cells = cells[["source", "task", "split"]].drop_duplicates().merge(chosen, how="left").fillna({"actual_gain": 0.0})
        source = all_cells.groupby("source").actual_gain.mean()
        selectors[column] = {
            "source_balanced_gain": float(source.mean()), "harmful_cells": int((all_cells.actual_gain < 0).sum()),
            "min_source_gain": float(source.min()), "selected_cells": int(len(chosen)), "cells": int(len(all_cells)),
        }
    gates = {
        "aggregate_spearman_at_least_0p60": rho >= 0.60,
        "aggregate_sign_accuracy_at_least_0p75": sign >= 0.75,
        "pessimistic_positive": selectors["selected_pessimistic"]["source_balanced_gain"] > 0,
        "pessimistic_no_source_below_minus_0p002": selectors["selected_pessimistic"]["min_source_gain"] >= -0.002,
        "pessimistic_no_more_harm": selectors["selected_pessimistic"]["harmful_cells"] <= selectors["selected_mean"]["harmful_cells"],
        "integrity": bool(cells.oof_finite.all() and np.isfinite(cells[["predicted_gain", "actual_gain"]]).all().all()),
    }
    return {"aggregate_spearman": rho, "aggregate_sign_accuracy": sign, "selectors": selectors, "gates": gates, "passes": all(gates.values())}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--device", default="cuda:0"); parser.add_argument("--shard", type=int, default=0); parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--backbone", choices=["mlp", "resnet", "ft_transformer", "tabm"], default="mlp")
    args = parser.parse_args(); device = torch.device(args.device); out = backbone_output(args.backbone); out.mkdir(parents=True, exist_ok=True)
    matrix = [(task, split) for task in TASKS for split in SPLITS]
    rows = []
    for index, (task, split) in enumerate(matrix):
        if index % args.shards != args.shard: continue
        started = time.time(); rows.extend(run_cell(task, split, device, args.backbone)); print(args.backbone, task, split, round(time.time()-started, 1), flush=True)
    shard_path = out / f"cells_shard{args.shard}of{args.shards}.csv"; pd.DataFrame(rows).to_csv(shard_path, index=False)
    paths = [out / f"cells_shard{i}of{args.shards}.csv" for i in range(args.shards)]
    if all(path.exists() for path in paths):
        cells = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True).sort_values(["task", "split", "operator"])
        cells.to_csv(out / "cells.csv", index=False); summary = analyze(cells)
        summary["backbone"] = args.backbone
        (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n"); print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
