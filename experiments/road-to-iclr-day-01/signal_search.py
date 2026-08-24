"""Broad, validation-selected signal search for the four Day 1 datasets.

The search has three stages per dataset:

1. tune the schema + PLE backbone on seed 0;
2. test semantic routes around the two best backbone configurations;
3. freeze the selected single models and greedy ensemble, then confirm them on
   seeds 0--4.

Only validation predictions participate in selection. Test scores are reported
for the frozen confirmation set, not used to rank screening candidates.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

import real_data_benchmark as benchmark


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    phase: str
    method: str
    bins: int
    width: int
    depth: int
    dropout: float
    learning_rate: float
    weight_decay: float
    model: str
    activation: str
    low_cardinality: int = 32
    smoothing: float = 20.0
    identity_effect_threshold: float = 0.01
    gate_entropy_weight: float = 0.001


@dataclass
class CandidateRun:
    candidate: Candidate
    output: benchmark.TrainOutput


def base_candidates(count: int, seed: int) -> list[Candidate]:
    anchors = [
        (bins, 256, 3, 0.1, 1e-3, 1e-4, "mlp", "relu")
        for bins in (16, 32, 64, 128)
    ]
    grid = [
        (bins, width, depth, dropout, learning_rate, weight_decay, model, activation)
        for bins in (16, 32, 64, 128)
        for width in (128, 256, 384, 512)
        for depth in (2, 3, 4)
        for dropout in (0.0, 0.1, 0.2)
        for learning_rate in (3e-4, 1e-3)
        for weight_decay in (0.0, 1e-5, 1e-4)
        for model in ("mlp", "resnet")
        for activation in ("relu", "gelu", "silu")
    ]
    rng = random.Random(seed)
    rng.shuffle(grid)
    configurations = anchors[:count]
    for configuration in grid:
        if len(configurations) >= count:
            break
        if configuration not in configurations:
            configurations.append(configuration)
    return [
        Candidate(
            candidate_id=f"base_{index:03d}",
            phase="backbone",
            method="schema_ple",
            bins=configuration[0],
            width=configuration[1],
            depth=configuration[2],
            dropout=configuration[3],
            learning_rate=configuration[4],
            weight_decay=configuration[5],
            model=configuration[6],
            activation=configuration[7],
        )
        for index, configuration in enumerate(configurations)
    ]


def semantic_candidates(
    bases: list[Candidate], dataset: benchmark.Dataset
) -> list[Candidate]:
    has_categories = dataset.x_cat is not None
    has_low_cardinality_numeric = False
    if dataset.x_num is not None:
        has_low_cardinality_numeric = any(
            len(np.unique(dataset.x_num["train"][:, column])) <= 128
            for column in range(dataset.x_num["train"].shape[1])
        )

    candidates: list[Candidate] = []
    for base_index, base in enumerate(bases):
        variants: list[dict[str, object]] = [
            {"method": "gated_ple", "gate_entropy_weight": entropy}
            for entropy in (0.0, 1e-3)
        ]
        if has_low_cardinality_numeric:
            variants.extend(
                {"method": "numeric_identity", "low_cardinality": cardinality}
                for cardinality in (8, 32, 128)
            )
            variants.extend(
                {
                    "method": "diagnostic_residual",
                    "low_cardinality": cardinality,
                    "identity_effect_threshold": threshold,
                }
                for cardinality in (32, 128)
                for threshold in (0.001, 0.01)
            )
        if has_categories:
            variants.extend(
                {"method": method, "smoothing": smoothing}
                for method in ("cat_target", "cat_residual")
                for smoothing in (1.0, 20.0, 100.0)
            )
            variants.extend(
                {"method": method, "smoothing": smoothing}
                for method in ("multi_view", "multi_view_residual")
                for smoothing in (5.0, 50.0)
            )
        if has_categories or has_low_cardinality_numeric:
            variants.extend(
                {
                    "method": "sparse_gate",
                    "low_cardinality": cardinality,
                    "identity_effect_threshold": threshold,
                    "smoothing": smoothing,
                    "gate_entropy_weight": entropy,
                }
                for cardinality in (32, 128)
                for threshold in (0.001, 0.01)
                for smoothing, entropy in ((5.0, 0.0), (50.0, 1e-3))
            )
        for variant_index, values in enumerate(variants):
            candidates.append(
                replace(
                    base,
                    candidate_id=f"semantic_{base_index:02d}_{variant_index:03d}",
                    phase="semantic",
                    **values,
                )
            )
    return candidates


def validation_loss(dataset: benchmark.Dataset, output: benchmark.TrainOutput) -> float:
    return benchmark._prediction_loss(
        dataset.task, output.val_prediction, output_target(dataset, "val")
    )


def output_target(dataset: benchmark.Dataset, part: str) -> np.ndarray:
    target = dataset.y[part].astype(np.float32)
    if dataset.task == "regression":
        mean = float(dataset.y["train"].mean())
        scale = float(dataset.y["train"].std()) or 1.0
        target = (target - mean) / scale
    return target


def run_candidate(
    dataset: benchmark.Dataset,
    candidate: Candidate,
    seed: int,
    device: torch.device,
    max_epochs: int,
    patience: int,
) -> CandidateRun:
    cache: dict[str, object] = {}
    encoded = benchmark.encode_dataset(
        dataset,
        candidate.method,
        seed,
        candidate.bins,
        candidate.low_cardinality,
        candidate.smoothing,
        candidate.identity_effect_threshold,
        cache,
    )
    reference = benchmark.encode_dataset(
        dataset,
        "schema_ple",
        seed,
        candidate.bins,
        candidate.low_cardinality,
        candidate.smoothing,
        candidate.identity_effect_threshold,
        cache,
    )
    target_parameters = None
    if candidate.method != "schema_ple":
        target_parameters = benchmark._parameter_count(
            benchmark._make_model(
                reference,
                candidate.width,
                candidate.depth,
                candidate.dropout,
                False,
                candidate.model,
                candidate.activation,
            )
        )
    output = benchmark.train_one(
        encoded,
        seed,
        benchmark.BATCH_SIZES[dataset.name],
        device,
        candidate.width,
        candidate.depth,
        candidate.dropout,
        candidate.learning_rate,
        candidate.weight_decay,
        max_epochs,
        patience,
        gated=candidate.method in ("gated_ple", "sparse_gate"),
        gate_entropy_weight=candidate.gate_entropy_weight,
        target_parameters=target_parameters,
        model_type=candidate.model,
        activation=candidate.activation,
    )
    return CandidateRun(candidate, output)


def screen_row(dataset: benchmark.Dataset, run: CandidateRun) -> dict[str, object]:
    row: dict[str, object] = {
        "dataset": dataset.name,
        "task": dataset.task,
        "metric": "accuracy" if dataset.task == "binclass" else "rmse",
        "seed": 0,
        **asdict(run.candidate),
        **run.output.result,
        "selection_loss": validation_loss(dataset, run.output),
    }
    # The screen is ranked on validation loss. Keeping the test column out of
    # this file makes accidental test-based selection harder.
    row.pop("test_score", None)
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def greedy_ensemble(
    dataset: benchmark.Dataset,
    runs: list[CandidateRun],
    max_members: int,
) -> list[CandidateRun]:
    selected: list[CandidateRun] = []
    remaining = runs.copy()
    prediction: np.ndarray | None = None
    best_loss = float("inf")
    target = output_target(dataset, "val")
    while remaining and len(selected) < max_members:
        best_run = None
        best_prediction = None
        for run in remaining:
            proposal = run.output.val_prediction
            if prediction is not None:
                proposal = (prediction * len(selected) + proposal) / (len(selected) + 1)
            loss = benchmark._prediction_loss(dataset.task, proposal, target)
            if loss < best_loss - 1e-8:
                best_loss = loss
                best_run = run
                best_prediction = proposal
        if best_run is None:
            break
        selected.append(best_run)
        remaining.remove(best_run)
        prediction = best_prediction
    return selected


def prediction_scores(
    dataset: benchmark.Dataset,
    val_prediction: np.ndarray,
    test_prediction: np.ndarray,
) -> tuple[float, float]:
    val_score = benchmark._metric(dataset.task, val_prediction, output_target(dataset, "val"))
    test_score = benchmark._metric(
        dataset.task, test_prediction, output_target(dataset, "test")
    )
    if dataset.task == "regression":
        scale = float(dataset.y["train"].std()) or 1.0
        val_score *= scale
        test_score *= scale
    return val_score, test_score


def confirmation_rows(
    dataset: benchmark.Dataset,
    selected_roles: dict[str, Candidate],
    ensemble_members: list[Candidate],
    screen_runs: dict[str, CandidateRun],
    seeds: list[int],
    device: torch.device,
    max_epochs: int,
    patience: int,
) -> list[dict[str, object]]:
    needed = {candidate.candidate_id: candidate for candidate in selected_roles.values()}
    needed.update({candidate.candidate_id: candidate for candidate in ensemble_members})
    rows: list[dict[str, object]] = []
    for seed in seeds:
        runs: dict[str, CandidateRun] = {}
        for candidate_id, candidate in needed.items():
            if seed == 0:
                runs[candidate_id] = screen_runs[candidate_id]
            else:
                runs[candidate_id] = run_candidate(
                    dataset, candidate, seed, device, max_epochs, patience
                )
        for role, candidate in selected_roles.items():
            run = runs[candidate.candidate_id]
            rows.append(
                {
                    "dataset": dataset.name,
                    "task": dataset.task,
                    "metric": "accuracy" if dataset.task == "binclass" else "rmse",
                    "role": role,
                    "seed": seed,
                    **asdict(candidate),
                    **run.output.result,
                    "ensemble_members": candidate.candidate_id,
                }
            )
        val_prediction = np.mean(
            [runs[candidate.candidate_id].output.val_prediction for candidate in ensemble_members],
            axis=0,
        )
        test_prediction = np.mean(
            [runs[candidate.candidate_id].output.test_prediction for candidate in ensemble_members],
            axis=0,
        )
        val_score, test_score = prediction_scores(dataset, val_prediction, test_prediction)
        rows.append(
            {
                "dataset": dataset.name,
                "task": dataset.task,
                "metric": "accuracy" if dataset.task == "binclass" else "rmse",
                "role": "hp_ensemble",
                "seed": seed,
                **{key: "" for key in asdict(ensemble_members[0])},
                "input_features": sum(
                    int(runs[candidate.candidate_id].output.result["input_features"])
                    for candidate in ensemble_members
                ),
                "parameters": sum(
                    int(runs[candidate.candidate_id].output.result["parameters"])
                    for candidate in ensemble_members
                ),
                "width": 0,
                "best_epoch": 0,
                "val_score": val_score,
                "test_score": test_score,
                "train_seconds": sum(
                    float(runs[candidate.candidate_id].output.result["train_seconds"])
                    for candidate in ensemble_members
                ),
                "selected_numeric": "",
                "members": "",
                "model": "ensemble",
                "activation": "",
                "ensemble_members": ";".join(
                    candidate.candidate_id for candidate in ensemble_members
                ),
            }
        )
        print(f"{dataset.name}: confirmed seed {seed}", flush=True)
    return rows


def aggregate(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    roles = sorted({str(row["role"]) for row in rows})
    summary = {}
    for role in roles:
        scores = np.array(
            [float(row["test_score"]) for row in rows if row["role"] == role]
        )
        summary[role] = {
            "test_mean": float(scores.mean()),
            "test_std": float(scores.std()),
            "seeds": int(len(scores)),
        }
    return summary


def search_dataset(
    dataset_name: str,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    dataset = benchmark.load_dataset(args.data, dataset_name)
    screen_runs: list[CandidateRun] = []
    screen_rows: list[dict[str, object]] = []
    screen_path = args.output_dir / f"{dataset_name}_screen.csv"

    bases = base_candidates(args.base_candidates, args.search_seed)
    for index, candidate in enumerate(bases, 1):
        run = run_candidate(
            dataset, candidate, 0, device, args.max_epochs, args.patience
        )
        screen_runs.append(run)
        screen_rows.append(screen_row(dataset, run))
        write_csv(screen_path, screen_rows)
        print(
            f"{dataset_name}: backbone {index}/{len(bases)} "
            f"val_loss={validation_loss(dataset, run.output):.6g}",
            flush=True,
        )

    ranked_bases = sorted(screen_runs, key=lambda run: validation_loss(dataset, run.output))
    semantic = semantic_candidates(
        [run.candidate for run in ranked_bases[: args.top_backbones]], dataset
    )
    for index, candidate in enumerate(semantic, 1):
        run = run_candidate(
            dataset, candidate, 0, device, args.max_epochs, args.patience
        )
        screen_runs.append(run)
        screen_rows.append(screen_row(dataset, run))
        write_csv(screen_path, screen_rows)
        print(
            f"{dataset_name}: semantic {index}/{len(semantic)} "
            f"{candidate.method} val_loss={validation_loss(dataset, run.output):.6g}",
            flush=True,
        )

    ranked_all = sorted(screen_runs, key=lambda run: validation_loss(dataset, run.output))
    ranked_semantic = sorted(
        (run for run in screen_runs if run.candidate.phase == "semantic"),
        key=lambda run: validation_loss(dataset, run.output),
    )
    ensemble_runs = greedy_ensemble(dataset, screen_runs, args.max_ensemble_members)
    selected_roles = {
        "fixed_baseline": next(run.candidate for run in screen_runs if run.candidate.candidate_id == "base_000"),
        "tuned_backbone": ranked_bases[0].candidate,
        "best_semantic": ranked_semantic[0].candidate,
        "best_overall": ranked_all[0].candidate,
    }
    run_map = {run.candidate.candidate_id: run for run in screen_runs}
    confirmation = confirmation_rows(
        dataset,
        selected_roles,
        [run.candidate for run in ensemble_runs],
        run_map,
        args.confirm_seeds,
        device,
        args.max_epochs,
        args.patience,
    )
    confirmation_path = args.output_dir / f"{dataset_name}_confirmation.csv"
    write_csv(confirmation_path, confirmation)
    summary = {
        "dataset": dataset_name,
        "screen_candidates": len(screen_runs),
        "selection_seed": 0,
        "selected_roles": {
            role: asdict(candidate) for role, candidate in selected_roles.items()
        },
        "ensemble_members": [asdict(run.candidate) for run in ensemble_runs],
        "confirmation": aggregate(confirmation),
    }
    (args.output_dir / f"{dataset_name}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["confirmation"], indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=Path(__file__).with_name("data")
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=benchmark.PRIMARY_DATASETS,
        default=benchmark.PRIMARY_DATASETS,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).with_name("signal_search")
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--search-seed", type=int, default=20260824)
    parser.add_argument("--base-candidates", type=int, default=24)
    parser.add_argument("--top-backbones", type=int, default=2)
    parser.add_argument("--max-ensemble-members", type=int, default=5)
    parser.add_argument("--confirm-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    for dataset_name in args.datasets:
        search_dataset(dataset_name, args, device)


if __name__ == "__main__":
    main()
