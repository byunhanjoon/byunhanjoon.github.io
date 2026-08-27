#!/usr/bin/env python3
"""Type-agnostic rank, exact/hash identity, and mass positional tokenizer."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from semantic_multiview_pilot import (
    MLPBackbone,
    PARTS,
    ResNetBackbone,
    load_tabred,
    ple_basis,
)
from support_identity_transfer_pilot import HybridAttentionMLP, MatchedFTTransformer


HERE = Path(__file__).resolve().parent


@dataclass
class UniversalData:
    rank: dict[str, np.ndarray]
    rank_lower: dict[str, np.ndarray]
    rank_upper: dict[str, np.ndarray]
    exact_code: dict[str, np.ndarray]
    bin_code: dict[str, np.ndarray]
    information: dict[str, np.ndarray]
    y: dict[str, np.ndarray]
    y_scale: float
    n_fields: int
    metadata: list[dict[str, object]]


def raw_fields(data) -> list[dict[str, np.ndarray]]:
    fields: list[dict[str, np.ndarray]] = []
    for source in (data.x_num, data.x_bin, data.x_cat):
        if source is None:
            continue
        for column in range(source["train"].shape[1]):
            fields.append({part: source[part][:, column] for part in PARTS})
    return fields


def encode_field(
    parts: dict[str, np.ndarray],
    top_levels: int,
    hash_buckets: int,
    rank_bins: int,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, object],
]:
    train = parts["train"]
    levels, counts = np.unique(train, return_counts=True)
    cumulative = np.cumsum(counts)
    midpoint_rank = (cumulative - 0.5 * counts) / len(train)
    probability = counts / len(train)
    top_order = np.lexsort((np.arange(len(levels)), -counts))[:top_levels]
    top_code = np.zeros(len(levels), dtype=np.int64)
    top_code[top_order] = np.arange(1, len(top_order) + 1)
    top_count = len(top_order)
    output_rank, output_lower, output_upper = {}, {}, {}
    output_exact, output_bin, output_information = {}, {}, {}
    for part, query in parts.items():
        positions = np.searchsorted(levels, query)
        clipped = np.minimum(positions, len(levels) - 1)
        known = (positions < len(levels)) & (levels[clipped] == query)
        rank = np.empty(len(query), dtype=np.float64)
        rank[known] = midpoint_rank[clipped[known]]
        unknown_positions = positions[~known]
        before_mass = np.where(
            unknown_positions > 0, cumulative[np.maximum(unknown_positions - 1, 0)], 0
        )
        rank[~known] = before_mass / len(train)
        lower = rank.copy()
        upper = rank.copy()
        lower[known] = (cumulative[clipped[known]] - counts[clipped[known]]) / len(train)
        upper[known] = cumulative[clipped[known]] / len(train)
        code = np.empty(len(query), dtype=np.int64)
        known_top = known & (top_code[clipped] > 0)
        code[known_top] = top_code[clipped[known_top]]
        hashed = ~known_top
        code[hashed] = top_count + 1 + (positions[hashed] % hash_buckets)
        prob = np.full(len(query), 0.5 / len(train), dtype=np.float64)
        prob[known] = probability[clipped[known]]
        information = np.clip(
            -np.log(prob) / max(math.log(2.0 * len(train)), 1.0), 0.0, 1.0
        )
        output_rank[part] = rank.astype(np.float32)
        output_lower[part] = lower.astype(np.float32)
        output_upper[part] = upper.astype(np.float32)
        output_exact[part] = code
        output_bin[part] = (
            1 + np.minimum((rank * rank_bins).astype(np.int64), rank_bins - 1)
        )
        output_information[part] = information.astype(np.float32)
    return output_rank, output_lower, output_upper, output_exact, output_bin, output_information, {
        "cardinality": int(len(levels)),
        "top_levels": int(top_count),
        "maximum_mass": float(probability.max()),
        "singleton_mass": float(counts[counts == 1].sum() / len(train)),
    }


def prepare(data, config: dict) -> UniversalData:
    fields = raw_fields(data)
    rank_columns = {part: [] for part in PARTS}
    lower_columns = {part: [] for part in PARTS}
    upper_columns = {part: [] for part in PARTS}
    exact_columns = {part: [] for part in PARTS}
    bin_columns = {part: [] for part in PARTS}
    information_columns = {part: [] for part in PARTS}
    metadata = []
    for field, parts in enumerate(fields):
        rank, lower, upper, exact, binned, information, field_meta = encode_field(
            parts,
            config["top_exact_levels"],
            config["rare_hash_buckets"],
            config["rank_ple_bins"],
        )
        field_meta["field"] = field
        metadata.append(field_meta)
        for part in PARTS:
            rank_columns[part].append(rank[part])
            lower_columns[part].append(lower[part])
            upper_columns[part].append(upper[part])
            exact_columns[part].append(exact[part])
            bin_columns[part].append(binned[part])
            information_columns[part].append(information[part])
    stack_float = lambda columns: {part: np.ascontiguousarray(np.column_stack(values), dtype=np.float32) for part, values in columns.items()}
    stack_int = lambda columns: {part: np.ascontiguousarray(np.column_stack(values), dtype=np.int64) for part, values in columns.items()}
    return UniversalData(
        rank=stack_float(rank_columns),
        rank_lower=stack_float(lower_columns),
        rank_upper=stack_float(upper_columns),
        exact_code=stack_int(exact_columns),
        bin_code=stack_int(bin_columns),
        information=stack_float(information_columns),
        y=data.y,
        y_scale=data.y_scale,
        n_fields=len(fields),
        metadata=metadata,
    )


def ple_interval_basis(lower: Tensor, upper: Tensor, edges: Tensor) -> Tensor:
    """Average the cumulative PLE basis over a closed empirical-CDF interval."""
    left, right = edges[:, :-1], edges[:, 1:]
    bin_width = right - left

    def antiderivative(x: Tensor) -> Tensor:
        offset = x[:, :, None] - left[None]
        return torch.where(
            x[:, :, None] <= left[None],
            torch.zeros((), device=x.device, dtype=x.dtype),
            torch.where(
                x[:, :, None] < right[None],
                offset.square() / (2.0 * bin_width[None]),
                x[:, :, None] - 0.5 * (left + right)[None],
            ),
        )

    width = upper - lower
    integrated = (antiderivative(upper) - antiderivative(lower)) / width.clamp_min(
        torch.finfo(width.dtype).eps
    )[:, :, None]
    midpoint = ple_basis(0.5 * (lower + upper), edges)
    return torch.where((width > torch.finfo(width.dtype).eps)[:, :, None], integrated, midpoint)


class UniversalTokenizer(nn.Module):
    def __init__(self, n_fields: int, config: dict, method: str) -> None:
        super().__init__()
        self.method = method
        self.rank_bins = config["rank_ple_bins"]
        d_token = config["d_token"]
        edges = np.tile(
            np.linspace(0.0, 1.0, self.rank_bins + 1, dtype=np.float32),
            (n_fields, 1),
        )
        self.register_buffer("rank_edges", torch.from_numpy(edges))
        self.rank_weight = nn.Parameter(torch.empty(n_fields, self.rank_bins, d_token))
        self.field_bias = nn.Parameter(torch.zeros(n_fields, d_token))
        nn.init.xavier_uniform_(self.rank_weight)
        vocabulary = config["top_exact_levels"] + config["rare_hash_buckets"] + 1
        self.identity = nn.ModuleList(
            nn.Embedding(vocabulary, d_token, padding_idx=0) for _ in range(n_fields)
        )
        for embedding in self.identity:
            nn.init.normal_(embedding.weight, std=1.0 / math.sqrt(d_token))
            with torch.no_grad(): embedding.weight[0].zero_()
        self.frequency_modes = config["frequency_modes"]
        self.frequency_weight = nn.Parameter(
            torch.empty(n_fields, 2 * self.frequency_modes, d_token)
        )
        nn.init.xavier_uniform_(self.frequency_weight)
        control_rng = np.random.default_rng(config["seed"] + 73)
        self.register_buffer(
            "control_frequencies",
            torch.from_numpy(
                control_rng.uniform(
                    0.5,
                    self.frequency_modes + 0.5,
                    size=(n_fields, self.frequency_modes),
                ).astype(np.float32)
            ),
        )
        self.identity_gate = nn.Parameter(torch.zeros(n_fields))
        cycle_start = float(config.get("cycle_gate_logit", -2.0))
        self.frequency_gate = nn.Parameter(
            torch.full(
                (n_fields,),
                cycle_start if method in {"rank_cycle", "rank_cycle_control"} else 0.0,
            )
        )

    def forward(
        self,
        rank: Tensor,
        rank_lower: Tensor,
        rank_upper: Tensor,
        code: Tensor,
        information: Tensor,
    ) -> Tensor:
        basis = (
            ple_interval_basis(rank_lower, rank_upper, self.rank_edges)
            if self.method == "interval_rank"
            else ple_basis(rank, self.rank_edges)
        )
        base = torch.einsum("nfb,fbd->nfd", basis, self.rank_weight) + self.field_bias
        if self.method in {"bin_identity", "exact_identity", "mass_identity"}:
            identity = torch.stack(
                [embedding(code[:, field]) for field, embedding in enumerate(self.identity)],
                dim=1,
            )
            base = base + self.identity_gate[None, :, None] * identity
        if self.method in {"mass_identity", "rank_cycle", "rank_cycle_control"}:
            if self.method == "mass_identity":
                position = information
                frequencies = torch.arange(
                    1,
                    self.frequency_modes + 1,
                    device=information.device,
                    dtype=information.dtype,
                )[None, None, :]
            elif self.method == "rank_cycle":
                position = rank
                frequencies = torch.arange(
                    1,
                    self.frequency_modes + 1,
                    device=rank.device,
                    dtype=rank.dtype,
                )[None, None, :]
            else:
                position = rank
                frequencies = self.control_frequencies[None, :, :].to(rank.dtype)
            angle = 2.0 * math.pi * position[:, :, None] * frequencies
            positional = torch.cat((torch.sin(angle), torch.cos(angle)), dim=-1)
            frequency = torch.einsum("nfm,fmd->nfd", positional, self.frequency_weight)
            gate = (
                torch.sigmoid(self.frequency_gate)
                if self.method in {"rank_cycle", "rank_cycle_control"}
                else self.frequency_gate
            )
            base = base + gate[None, :, None] * frequency
        return base


class UniversalModel(nn.Module):
    def __init__(self, data: UniversalData, config: dict, method: str, architecture: str, width: int, ff_width: int) -> None:
        super().__init__()
        self.tokenizer = UniversalTokenizer(data.n_fields, config, method)
        d_token, depth = config["d_token"], config["depth"]
        if architecture == "mlp": self.backbone = MLPBackbone(data.n_fields, d_token, width, depth)
        elif architecture == "resnet": self.backbone = ResNetBackbone(data.n_fields, d_token, width, depth)
        elif architecture == "ft_transformer": self.backbone = MatchedFTTransformer(data.n_fields, d_token, depth, ff_width, config["dropout"])
        elif architecture == "hybrid": self.backbone = HybridAttentionMLP(data.n_fields, d_token, width, depth, ff_width, config["dropout"])
        else: raise KeyError(architecture)

    def forward(
        self,
        rank: Tensor,
        rank_lower: Tensor,
        rank_upper: Tensor,
        code: Tensor,
        information: Tensor,
    ) -> Tensor:
        prediction, _ = self.backbone(
            self.tokenizer(rank, rank_lower, rank_upper, code, information)
        )
        return prediction


def count(model: nn.Module) -> int: return sum(p.numel() for p in model.parameters())


def codes(data: UniversalData, method: str) -> dict[str, np.ndarray]:
    return data.bin_code if method == "bin_identity" else data.exact_code


def loader(data: UniversalData, method: str, part: str, batch: int, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(
        TensorDataset(
            torch.from_numpy(data.rank[part]),
            torch.from_numpy(data.rank_lower[part]),
            torch.from_numpy(data.rank_upper[part]),
            torch.from_numpy(codes(data, method)[part]),
            torch.from_numpy(data.information[part]), torch.from_numpy(data.y[part]),
        ), batch_size=batch, shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed), pin_memory=True,
    )


@torch.inference_mode()
def evaluate(model: UniversalModel, stream: DataLoader, device: torch.device, scale: float) -> tuple[dict[str, float], np.ndarray]:
    model.eval(); predictions=[]; targets=[]
    for rank, rank_lower, rank_upper, code, information, target in stream:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction=model(
                rank.to(device),
                rank_lower.to(device),
                rank_upper.to(device),
                code.to(device),
                information.to(device),
            )
        predictions.append(prediction.float().cpu()); targets.append(target)
    pred=torch.cat(predictions).numpy(); truth=torch.cat(targets).numpy(); mse=float(np.mean((pred-truth)**2))
    return {"loss":mse,"rmse":math.sqrt(mse)*scale},pred


def train_one(data: UniversalData, config: dict, method: str, architecture: str, device: str) -> tuple[dict[str, object],np.ndarray,np.ndarray]:
    seed=config["seed"]; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model=UniversalModel(data,config,method,architecture,config["width"],config["ft_feedforward_width"])
    resolved=torch.device(device); model=model.to(resolved)
    batch=min(config["batch_size"],256) if architecture in {"ft_transformer","hybrid"} else config["batch_size"]
    streams={part:loader(data,method,part,batch if part=="train" else batch*2,part=="train",seed) for part in PARTS}
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"])
    best,epoch_best,stale,state=math.inf,0,0,None; started=time.perf_counter()
    for epoch in range(1,config["epochs"]+1):
        model.train()
        for rank,rank_lower,rank_upper,code,information,target in streams["train"]:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=resolved.type,dtype=torch.bfloat16,enabled=resolved.type=="cuda"):
                prediction=model(rank.to(resolved),rank_lower.to(resolved),rank_upper.to(resolved),code.to(resolved),information.to(resolved)); loss=torch.nn.functional.mse_loss(prediction,target.to(resolved))
            loss.backward(); optimizer.step()
        validation,_=evaluate(model,streams["val"],resolved,data.y_scale)
        if validation["loss"]<best:
            best,epoch_best,stale=validation["loss"],epoch,0; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: stale+=1
        if stale>config["patience"]: break
    assert state is not None; model.load_state_dict(state)
    validation,val_pred=evaluate(model,streams["val"],resolved,data.y_scale); test,test_pred=evaluate(model,streams["test"],resolved,data.y_scale)
    effective_frequency_gate = (
        model.tokenizer.frequency_gate.sigmoid()
        if method in {"rank_cycle", "rank_cycle_control"}
        else model.tokenizer.frequency_gate.abs()
    )
    return {"parameters":count(model),"best_epoch":epoch_best,"val_loss":validation["loss"],"val_rmse":validation["rmse"],"test_loss":test["loss"],"test_rmse":test["rmse"],"mean_identity_gate":float(model.tokenizer.identity_gate.abs().mean().cpu()),"mean_frequency_gate":float(effective_frequency_gate.mean().cpu()),"train_seconds":time.perf_counter()-started},val_pred,test_pred


def read(path: Path) -> list[dict[str,str]]:
    if not path.exists(): return []
    with path.open(newline="") as handle:return list(csv.DictReader(handle))


def write(path: Path,rows:list[dict[str,object]])->None:
    fields=list(dict.fromkeys(k for row in rows for k in row));path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("w",newline="") as handle:w=csv.DictWriter(handle,fieldnames=fields);w.writeheader();w.writerows(rows)
    tmp.replace(path)


def analyze(config:dict,output:Path)->dict[str,object]:
    frame=pd.read_csv(output);cells=[]
    for (dataset,model),group in frame.groupby(["dataset","model"]):
        s=group.set_index("method")
        if not set(config["methods"]).issubset(s.index):continue
        base=s.loc["rank_only","val_rmse"]; mass=s.loc["mass_identity","val_rmse"]
        gains={name:100*(s.loc[name,"val_rmse"]-mass)/s.loc[name,"val_rmse"] for name in ("rank_only","bin_identity","exact_identity")}
        cells.append({"dataset":dataset,"model":model,**{f"gain_vs_{k}_pct":v for k,v in gains.items()},"test_gain_vs_rank_pct":100*(s.loc["rank_only","test_rmse"]-s.loc["mass_identity","test_rmse"])/s.loc["rank_only","test_rmse"],"cell_gate":all(v>0 for v in gains.values())})
    c=pd.DataFrame(cells);c.to_csv(output.with_name(output.stem+"_cells.csv"),index=False)
    dev=c[c.dataset.isin(config["development_datasets"])];gates={d:int(g.cell_gate.sum())>=2 for d,g in dev.groupby("dataset")};passed=any(gates.values())
    decision={"method_gate_passed":passed,"dataset_gates":gates,"passing_cells":int(dev.cell_gate.sum()),"development_cells":int(len(dev)),"transfer_authorized":passed}
    output.with_name(output.stem+"_decision.json").write_text(json.dumps(decision,indent=2,sort_keys=True)+"\n");print(c.to_string(index=False));print(json.dumps(decision,indent=2,sort_keys=True));return decision


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--config",type=Path,default=HERE/"universal_mass_identity_config.json");parser.add_argument("--datasets",nargs="+");parser.add_argument("--models",nargs="+");parser.add_argument("--methods",nargs="+");parser.add_argument("--seed",type=int);parser.add_argument("--device",default="cuda:0");parser.add_argument("--output",type=Path,default=HERE/"results/universal_mass_identity.csv");parser.add_argument("--analyze-only",action="store_true");args=parser.parse_args();config=json.loads(args.config.read_text())
    if args.seed is not None:config["seed"]=args.seed
    if args.analyze_only:analyze(config,args.output);return
    rows:list[dict[str,object]]=list(read(args.output));done={(r["dataset"],r["model"],r["method"]) for r in rows};metadata={}
    for dataset_name in args.datasets or config["development_datasets"]:
        source=load_tabred(dataset_name,max_train_rows=config["max_train_rows"],max_eval_rows=config["max_eval_rows"],sample_seed=config["sample_seed"]);data=prepare(source,config);metadata[dataset_name]=data.metadata
        for model in args.models or config["architectures"]:
            for method in args.methods or config["methods"]:
                key=(dataset_name,model,method)
                if key in done:continue
                result,val,test=train_one(data,config,method,model,args.device);path=args.output.parent/f"{args.output.stem}_predictions"/f"{dataset_name}__{model}__{method}.npz";path.parent.mkdir(parents=True,exist_ok=True)
                with path.open("wb") as handle:np.savez_compressed(handle,validation=val,test=test)
                row={"dataset":dataset_name,"model":model,"method":method,"seed":config["seed"],"n_fields":data.n_fields,**result};rows.append(row);done.add(key);write(args.output,rows);print(json.dumps(row,sort_keys=True),flush=True)
    args.output.with_suffix(".metadata.json").write_text(json.dumps({"config":config,"fields":metadata},indent=2,sort_keys=True)+"\n")
    required={(d,m,x) for d in config["development_datasets"] for m in config["architectures"] for x in config["methods"]}
    ablation_methods={"rank_only","bin_identity","exact_identity","mass_identity"}
    if ablation_methods.issubset(config["methods"]) and required.issubset(done):analyze(config,args.output)


if __name__=="__main__":main()
