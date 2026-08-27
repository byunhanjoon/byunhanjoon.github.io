#!/usr/bin/env python3
"""Adult selection-by-interface identity mechanism audit."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
DAY1 = HERE.parent / "road-to-iclr-day-01"
sys.path.insert(0, str(DAY1))

import real_data_benchmark as day1  # noqa: E402
from semantic_multiview_pilot import SplitData, _encode_categories, quantile_edges  # noqa: E402
from support_identity_transfer_pilot import (  # noqa: E402
    Encodings,
    PARTS,
    SupportModel,
    exact_support_codes,
    parameter_count,
    quantile_bin_codes,
)


def load_adult() -> SplitData:
    source = day1.load_dataset(DAY1 / "data", "adult")
    assert source.x_num is not None
    x_num = day1._clean_numeric(source.x_num)
    x_num = {
        part: np.ascontiguousarray(values, dtype=np.float32)
        for part, values in x_num.items()
    }
    x_bin = (
        None
        if source.x_bin is None
        else {
            part: np.ascontiguousarray(values, dtype=np.float32)
            for part, values in day1._clean_numeric(source.x_bin).items()
        }
    )
    if source.x_cat is None:
        x_cat, cardinalities = None, []
    else:
        x_cat, cardinalities = _encode_categories(source.x_cat)
    y = {
        part: np.ascontiguousarray(values, dtype=np.float32)
        for part, values in source.y.items()
    }
    return SplitData(
        x_num=x_num,
        x_bin=x_bin,
        x_cat=x_cat,
        y=y,
        y_mean=0.0,
        y_scale=1.0,
        category_cardinalities=cardinalities,
        cyclic_columns=[],
        cyclic_names=[],
        cyclic_periods=[],
        cyclic_origins=[],
        split_sizes_full={part: len(values) for part, values in y.items()},
    )


def classifier_tree_edges(
    train: np.ndarray, target: np.ndarray, n_bins: int, min_samples_leaf: int
) -> np.ndarray:
    fallback = quantile_edges(train, n_bins).astype(np.float64)
    output = np.empty_like(fallback)
    for field in range(train.shape[1]):
        values = train[:, field]
        if np.all(values == values[0]):
            output[field] = fallback[field]
            continue
        tree = DecisionTreeClassifier(
            max_leaf_nodes=n_bins,
            min_samples_leaf=min_samples_leaf,
            random_state=0,
        ).fit(values[:, None], target).tree_
        thresholds = tree.threshold[tree.children_left != tree.children_right]
        knots = list(np.unique(np.r_[values.min(), thresholds, values.max()]))
        scale = max(float(np.ptp(values)), 1.0)
        step = np.finfo(np.float32).eps * scale * 8
        while len(knots) < n_bins + 1:
            knots.append(knots[-1] + step)
        output[field] = knots[: n_bins + 1]
    edges = output.astype(np.float32)
    if not np.all(np.diff(edges, axis=1) > 0):
        raise RuntimeError("classifier T-PLE edges are not strict")
    return edges


def subset_encoding(
    data: SplitData,
    qple: np.ndarray,
    tple: np.ndarray,
    all_columns: list[int],
    all_cardinalities: list[int],
    all_codes: dict[str, np.ndarray],
    columns: list[int],
) -> Encodings:
    positions = [all_columns.index(column) for column in columns]
    cardinalities = [all_cardinalities[position] for position in positions]
    exact = {
        part: values[:, positions].copy() for part, values in all_codes.items()
    }
    binned = quantile_bin_codes(data.x_num, qple, columns, cardinalities)
    return Encodings(qple, tple, columns, cardinalities, exact, binned)


def method_spec(name: str) -> tuple[str, str | None, str, str | None]:
    if name in {"qple", "tple"}:
        return name, None, "additive", None
    pieces = name.split("_")
    edge_kind, code_kind, selection, interface = pieces
    return edge_kind, code_kind, interface, selection


def method_names() -> list[str]:
    names = ["qple", "tple"]
    for selection in ("all", "supervised"):
        for interface in ("additive", "separate"):
            names.extend(
                (
                    f"qple_bin_{selection}_{interface}",
                    f"qple_exact_{selection}_{interface}",
                )
            )
    names.extend(
        ("tple_exact_supervised_additive", "tple_exact_supervised_separate")
    )
    return names


def build_model(
    data: SplitData,
    encoding: Encodings,
    config: dict,
    name: str,
    architecture: str,
    width: int,
    ff_width: int,
) -> SupportModel:
    edge_kind, code_kind, interface, _ = method_spec(name)
    internal_method = (
        edge_kind if code_kind is None else f"{edge_kind}_support"
    )
    return SupportModel(
        data=data,
        encoding=encoding,
        method=internal_method,
        architecture=architecture,
        d_token=config["d_token"],
        width=width,
        depth=config["depth"],
        ft_feedforward_width=ff_width,
        dropout=config["dropout"],
        support_interface=interface,
    )


def matched_model(
    data: SplitData,
    encoding: Encodings,
    config: dict,
    name: str,
    architecture: str,
    target: int,
) -> tuple[SupportModel, int, int]:
    base_width, base_ff = config["width"], config["ft_feedforward_width"]

    def closest_integer(low: int, high: int, count) -> int:
        while low < high:
            middle = (low + high) // 2
            if count(middle) < target:
                low = middle + 1
            else:
                high = middle
        candidates = {max(4, low - 1), low}
        return min(candidates, key=lambda value: abs(count(value) - target))

    if architecture == "ft_transformer":
        chosen = closest_integer(
            4,
            256,
            lambda value: parameter_count(
                build_model(data, encoding, config, name, architecture, base_width, value)
            ),
        )
        return (
            build_model(data, encoding, config, name, architecture, base_width, chosen),
            base_width,
            chosen,
        )
    chosen = closest_integer(
        8,
        512,
        lambda value: parameter_count(
            build_model(data, encoding, config, name, architecture, value, base_ff)
        ),
    )
    return (
        build_model(data, encoding, config, name, architecture, chosen, base_ff),
        chosen,
        base_ff,
    )


def codes_for(name: str, encoding: Encodings) -> dict[str, np.ndarray]:
    _, code_kind, _, _ = method_spec(name)
    if code_kind == "exact":
        return encoding.exact_codes
    if code_kind == "bin":
        return encoding.bin_codes
    return {
        part: np.empty((len(values), 0), dtype=np.int64)
        for part, values in encoding.exact_codes.items()
    }


def loader(
    data: SplitData,
    codes: dict[str, np.ndarray],
    part: str,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    rows = len(data.y[part])
    x_bin = data.x_bin[part] if data.x_bin is not None else np.empty((rows, 0), np.float32)
    x_cat = data.x_cat[part] if data.x_cat is not None else np.empty((rows, 0), np.int64)
    return DataLoader(
        TensorDataset(
            torch.from_numpy(data.x_num[part]),
            torch.from_numpy(x_bin),
            torch.from_numpy(x_cat),
            torch.from_numpy(codes[part]),
            torch.from_numpy(data.y[part]),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=True,
    )


@torch.inference_mode()
def evaluate(model: nn.Module, stream: DataLoader, device: torch.device) -> tuple[dict[str, float], np.ndarray]:
    model.eval()
    logits, targets = [], []
    for x_num, x_bin, x_cat, codes, target in stream:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction = model(x_num.to(device), x_bin.to(device), x_cat.to(device), codes.to(device))
        logits.append(prediction.float().cpu())
        targets.append(target)
    logit = torch.cat(logits).numpy()
    target = torch.cat(targets).numpy()
    probability = 1.0 / (1.0 + np.exp(-np.clip(logit, -40, 40)))
    return {
        "log_loss": float(log_loss(target, probability)),
        "auc": float(roc_auc_score(target, probability)),
        "accuracy": float(accuracy_score(target, probability >= 0.5)),
    }, logit


def train_one(
    data: SplitData,
    encoding: Encodings,
    config: dict,
    name: str,
    architecture: str,
    target_parameters: int,
    device: str,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    seed = config["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model, width, ff_width = matched_model(
        data, encoding, config, name, architecture, target_parameters
    )
    resolved = torch.device(device)
    model = model.to(resolved)
    codes = codes_for(name, encoding)
    batch = (
        min(config["batch_size"], 128)
        if architecture in {"ft_transformer", "hybrid"}
        else config["batch_size"]
    )
    streams = {
        part: loader(data, codes, part, batch if part == "train" else batch * 2, part == "train", seed)
        for part in PARTS
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    criterion = nn.BCEWithLogitsLoss()
    best_loss, best_epoch, stale, state = math.inf, 0, 0, None
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        for x_num, x_bin, x_cat, support_codes, target in streams["train"]:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=resolved.type, dtype=torch.bfloat16, enabled=resolved.type == "cuda"):
                prediction = model(
                    x_num.to(resolved, non_blocking=True), x_bin.to(resolved, non_blocking=True),
                    x_cat.to(resolved, non_blocking=True), support_codes.to(resolved, non_blocking=True),
                )
                loss = criterion(prediction, target.to(resolved, non_blocking=True))
            loss.backward(); optimizer.step()
        validation, _ = evaluate(model, streams["val"], resolved)
        if validation["log_loss"] < best_loss:
            best_loss, best_epoch, stale = validation["log_loss"], epoch, 0
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale > config["patience"]:
            break
    assert state is not None
    model.load_state_dict(state)
    validation, val_logit = evaluate(model, streams["val"], resolved)
    test, test_logit = evaluate(model, streams["test"], resolved)
    gates = model.tokenizer.support_gate_logits
    return {
        "parameters": parameter_count(model), "target_parameters": target_parameters,
        "matched_width": width, "matched_ft_feedforward_width": ff_width,
        "best_epoch": best_epoch, **{f"val_{k}": v for k, v in validation.items()},
        **{f"test_{k}": v for k, v in test.items()},
        "mean_support_gate": float(torch.sigmoid(gates).mean().cpu()) if gates is not None and len(gates) else 0.0,
        "train_seconds": time.perf_counter() - started,
    }, val_logit, test_logit


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists(): return []
    with path.open(newline="") as handle: return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def analyze(config: dict, output: Path) -> None:
    frame = pd.read_csv(output)
    rows = []
    for architecture, group in frame.groupby("model"):
        scores = group.set_index("method")
        for selection in ("all", "supervised"):
            for interface in ("additive", "separate"):
                exact = f"qple_exact_{selection}_{interface}"
                control = f"qple_bin_{selection}_{interface}"
                if exact not in scores.index: continue
                rows.append({
                    "model": architecture, "selection": selection, "interface": interface,
                    "exact_vs_qple_logloss": scores.loc[exact, "val_log_loss"] - scores.loc["qple", "val_log_loss"],
                    "exact_vs_bin_logloss": scores.loc[exact, "val_log_loss"] - scores.loc[control, "val_log_loss"],
                    "exact_vs_qple_auc_pp": 100 * (scores.loc[exact, "val_auc"] - scores.loc["qple", "val_auc"]),
                    "gate": bool(scores.loc[exact, "val_log_loss"] < scores.loc["qple", "val_log_loss"] and scores.loc[exact, "val_log_loss"] < scores.loc[control, "val_log_loss"]),
                })
    mechanisms = pd.DataFrame(rows)
    mechanisms.to_csv(output.with_name(output.stem + "_mechanisms.csv"), index=False)
    summary = mechanisms.groupby(["selection", "interface"]).agg(passes=("gate", "sum"), mean_exact_vs_bin=("exact_vs_bin_logloss", "mean")).reset_index()
    summary.to_csv(output.with_name(output.stem + "_summary.csv"), index=False)
    winner = summary.sort_values(["passes", "mean_exact_vs_bin"], ascending=[False, True]).iloc[0]
    winner_name = f"{winner.selection}/{winner.interface}"
    tple_passes = 0
    if winner.selection == "supervised":
        candidate = f"tple_exact_supervised_{winner.interface}"
        for _, group in frame.groupby("model"):
            scores = group.set_index("method")
            tple_passes += int(scores.loc[candidate, "val_log_loss"] < scores.loc["tple", "val_log_loss"])
    decision = {
        "winning_mechanism": winner_name,
        "qple_architecture_passes": int(winner.passes),
        "tple_architecture_passes": tple_passes,
        "mechanism_gate_passed": bool(winner.passes >= 2 and (winner.selection != "supervised" or tple_passes >= 2)),
    }
    output.with_name(output.stem + "_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(summary.to_string(index=False)); print(json.dumps(decision, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "adult_identity_mechanism_config.json")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--methods", nargs="+", choices=method_names())
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=HERE / "results/adult_identity_mechanism.csv")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args(); config = json.loads(args.config.read_text())
    if args.analyze_only: analyze(config, args.output); return
    data = load_adult()
    qple = quantile_edges(data.x_num["train"], config["qple_bins"])
    tple = classifier_tree_edges(data.x_num["train"], data.y["train"], config["tple_bins"], config["tple_min_samples_leaf"])
    all_columns, all_cards, all_codes = exact_support_codes(data.x_num, 128)
    assert all_columns == config["target_free_columns"]
    encodings = {
        "all": subset_encoding(data, qple, tple, all_columns, all_cards, all_codes, config["target_free_columns"]),
        "supervised": subset_encoding(data, qple, tple, all_columns, all_cards, all_codes, config["supervised_residual_columns"]),
    }
    rows: list[dict[str, object]] = list(read_rows(args.output))
    completed = {(row["model"], row["method"]) for row in rows}
    for architecture in args.models or config["architectures"]:
        reference = build_model(data, encodings["all"], config, "qple_exact_all_separate", architecture, config["width"], config["ft_feedforward_width"])
        target_parameters = parameter_count(reference); del reference
        for name in args.methods or method_names():
            if (architecture, name) in completed: continue
            _, _, _, selection = method_spec(name)
            encoding = encodings[selection or "all"]
            result, val_logit, test_logit = train_one(data, encoding, config, name, architecture, target_parameters, args.device)
            prediction_path = output_prediction = args.output.parent / f"{args.output.stem}_predictions" / f"{architecture}__{name}.npz"
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            with output_prediction.open("wb") as handle: np.savez_compressed(handle, validation=val_logit, test=test_logit)
            row = {"dataset": "adult", "model": architecture, "method": name, "seed": config["seed"], "selected_columns": ";".join(map(str, encoding.selected_columns)), **result}
            rows.append(row); completed.add((architecture, name)); write_rows(args.output, rows)
            print(json.dumps(row, sort_keys=True), flush=True)
    required = {(model, method) for model in config["architectures"] for method in method_names()}
    if required.issubset(completed): analyze(config, args.output)


if __name__ == "__main__": main()
