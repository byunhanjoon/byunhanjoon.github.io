"""Paired function-space trajectory decomposition for the Day 3 extension.

The runner keeps the two models in lockstep: the same rows, labels, dropout
masks, and fixed update budget are used in both coordinate systems.  It is
resumable and records failures as scientific outcomes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .broad_data import (
    controlled_representation,
    load_broad_dataset,
    paired_natural_representations,
)
from .broad_models import (
    FirstLayerMatrixUpdater,
    covariance_initialize,
    make_model,
    member_loss,
    metrics,
    predictions,
)
from .core import Prepared, geometry, make_prepared


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "experiments/day3/configs/trajectory_decomposition_preregistered.json"
RESULTS = ROOT / "results/day3/trajectory_decomposition"
MATCHED_ARMS = {"matched_adamw", "matched_input_natural"}
NATURAL_ARMS = {"matched_input_natural", "ordinary_input_natural"}


@dataclass(frozen=True)
class PairSpec:
    family: str
    label: str
    target_kappa: float | None
    reference: dict[str, np.ndarray]
    changed: dict[str, np.ndarray]
    transform: np.ndarray
    metadata: dict[str, object]


def config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text())


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("highest")


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _pair_specs(dataset, families: list[str], kappas: list[float]) -> list[PairSpec]:
    output: list[PairSpec] = []
    if "controlled" in families:
        for kappa in kappas:
            changed = controlled_representation(dataset, kappa)
            if changed.reference is None or changed.basis_transform is None:
                raise AssertionError("Controlled representation did not retain its reference map")
            output.append(
                PairSpec(
                    family="controlled",
                    label=f"controlled_kappa_{kappa:g}",
                    target_kappa=float(kappa),
                    reference=changed.reference,
                    changed=changed.parts,
                    transform=changed.basis_transform,
                    metadata=changed.metadata,
                )
            )
    if "natural" in families:
        reference, changed, transform = paired_natural_representations(dataset)
        output.append(
            PairSpec(
                family="natural",
                label="cumulative_helmert__to__local_adjacent",
                target_kappa=None,
                reference=reference.parts,
                changed=changed.parts,
                transform=transform,
                metadata=changed.metadata,
            )
        )
    return output


def function_match_first_layer(first: nn.Linear, transform: np.ndarray) -> None:
    """Map a reference first layer to inputs ``X @ transform``."""

    inverse_transpose = np.linalg.inv(np.asarray(transform, dtype=np.float64)).T
    mapped = first.weight.detach().cpu().double().numpy() @ inverse_transpose
    with torch.no_grad():
        first.weight.copy_(torch.from_numpy(mapped).to(first.weight))


def symmetric_prediction_drift(reference: np.ndarray, changed: np.ndarray) -> float:
    reference64 = np.asarray(reference, dtype=np.float64)
    changed64 = np.asarray(changed, dtype=np.float64)
    numerator = float(np.sqrt(np.mean((reference64 - changed64) ** 2)))
    reference_rms = float(np.sqrt(np.mean(reference64**2)))
    changed_rms = float(np.sqrt(np.mean(changed64**2)))
    return numerator / max(0.5 * (reference_rms + changed_rms), 1e-8)


def mapped_weight_drift(reference: nn.Linear, changed: nn.Linear, transform: np.ndarray) -> float:
    reference_weight = reference.weight.detach().cpu().double().numpy()
    changed_weight = changed.weight.detach().cpu().double().numpy()
    mapped = changed_weight @ np.asarray(transform, dtype=np.float64).T
    return float(
        np.linalg.norm(reference_weight - mapped)
        / max(0.5 * (np.linalg.norm(reference_weight) + np.linalg.norm(mapped)), 1e-12)
    )


def _predict_numpy(
    model: nn.Module,
    x: np.ndarray,
    task: str,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    values = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).to(device)
            values.append(predictions(model, batch, task).float().cpu().numpy())
    return np.concatenate(values)


def _probe(values: np.ndarray, maximum: int) -> np.ndarray:
    if len(values) <= maximum:
        return values
    indices = np.linspace(0, len(values) - 1, maximum, dtype=np.int64)
    return values[indices]


def _rng_state(device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return cpu, cuda


def _restore_rng(device: torch.device, state: tuple[torch.Tensor, torch.Tensor | None]) -> None:
    torch.set_rng_state(state[0])
    if device.type == "cuda" and state[1] is not None:
        torch.cuda.set_rng_state(state[1], device)


def _initialize_pair(
    reference_data: Prepared,
    changed_data: Prepared,
    *,
    model_name: str,
    arm: str,
    transform: np.ndarray,
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, nn.Module]:
    _seed(seed)
    output_size = reference_data.n_classes if reference_data.task == "multiclass" else 1
    reference = make_model(model_name, reference_data.x["train"].shape[1], output_size).to(device)
    changed = make_model(model_name, changed_data.x["train"].shape[1], output_size).to(device)
    changed.load_state_dict(reference.state_dict())
    if arm == "covariance_adamw":
        covariance_initialize(reference.first, reference_data.x["train"], seed, ridge=1e-8)
        covariance_initialize(changed.first, changed_data.x["train"], seed, ridge=1e-8)
    elif arm in MATCHED_ARMS:
        function_match_first_layer(changed.first, transform)
    return reference, changed


def _optimizers(
    reference: nn.Module,
    changed: nn.Module,
    reference_train: np.ndarray,
    changed_train: np.ndarray,
    arm: str,
    training: dict[str, object],
) -> tuple[
    torch.optim.Optimizer,
    torch.optim.Optimizer,
    FirstLayerMatrixUpdater | None,
    FirstLayerMatrixUpdater | None,
]:
    if arm not in NATURAL_ARMS:
        options = {
            "lr": float(training["adamw_learning_rate"]),
            "weight_decay": float(training["adamw_weight_decay"]),
        }
        return (
            torch.optim.AdamW(reference.parameters(), **options),
            torch.optim.AdamW(changed.parameters(), **options),
            None,
            None,
        )

    def later_parameters(model: nn.Module):
        first_ids = {id(model.first.weight), id(model.first.bias)}
        return [parameter for parameter in model.parameters() if id(parameter) not in first_ids]

    reference_optimizer = torch.optim.AdamW(
        later_parameters(reference),
        lr=float(training["natural_later_adamw_learning_rate"]),
        weight_decay=float(training["adamw_weight_decay"]),
    )
    changed_optimizer = torch.optim.AdamW(
        later_parameters(changed),
        lr=float(training["natural_later_adamw_learning_rate"]),
        weight_decay=float(training["adamw_weight_decay"]),
    )
    learning_rate = float(training["natural_first_learning_rate"])
    return (
        reference_optimizer,
        changed_optimizer,
        FirstLayerMatrixUpdater(
            reference.first, reference_train, "input_natural", learning_rate, ridge=0.0
        ),
        FirstLayerMatrixUpdater(
            changed.first, changed_train, "input_natural", learning_rate, ridge=0.0
        ),
    )


def _trajectory_observation(
    reference: nn.Module,
    changed: nn.Module,
    reference_data: Prepared,
    changed_data: Prepared,
    transform: np.ndarray,
    *,
    step: int,
    device: torch.device,
    probe_rows: int,
    batch_size: int,
) -> list[dict[str, object]]:
    observations = []
    for part in ("train", "val"):
        reference_x = _probe(reference_data.x[part], probe_rows)
        changed_x = _probe(changed_data.x[part], probe_rows)
        reference_prediction = _predict_numpy(
            reference, reference_x, reference_data.task, device, batch_size
        )
        changed_prediction = _predict_numpy(
            changed, changed_x, changed_data.task, device, batch_size
        )
        observations.append(
            {
                "step": step,
                "probe_split": part,
                "probe_rows": len(reference_x),
                "prediction_drift": symmetric_prediction_drift(
                    reference_prediction, changed_prediction
                ),
                "reference_prediction_rms": float(
                    np.sqrt(np.mean(reference_prediction.astype(np.float64) ** 2))
                ),
                "changed_prediction_rms": float(
                    np.sqrt(np.mean(changed_prediction.astype(np.float64) ** 2))
                ),
                "mapped_first_weight_drift": mapped_weight_drift(
                    reference.first, changed.first, transform
                ),
            }
        )
    reference.train()
    changed.train()
    return observations


def train_pair(
    reference_data: Prepared,
    changed_data: Prepared,
    *,
    model_name: str,
    arm: str,
    transform: np.ndarray,
    seed: int,
    device: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    cfg = config()
    training = cfg["training"]
    resolved = torch.device(device)
    reference, changed = _initialize_pair(
        reference_data,
        changed_data,
        model_name=model_name,
        arm=arm,
        transform=transform,
        seed=seed,
        device=resolved,
    )
    reference_optimizer, changed_optimizer, reference_updater, changed_updater = _optimizers(
        reference,
        changed,
        reference_data.x["train"],
        changed_data.x["train"],
        arm,
        training,
    )
    batch_size = int(training["batch_size"])
    generator = torch.Generator().manual_seed(seed + 70000)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(reference_data.x["train"]),
            torch.from_numpy(changed_data.x["train"]),
            torch.from_numpy(reference_data.y["train"]),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=resolved.type == "cuda",
    )
    trajectory_steps = {int(value) for value in training["trajectory_steps"]}
    trajectory = _trajectory_observation(
        reference,
        changed,
        reference_data,
        changed_data,
        transform,
        step=0,
        device=resolved,
        probe_rows=int(training["probe_rows_per_split"]),
        batch_size=batch_size * 2,
    )
    if arm in MATCHED_ARMS:
        initial_max = max(row["prediction_drift"] for row in trajectory)
        if initial_max > float(cfg["analysis_gates"]["matched_step0_max_drift"]):
            raise AssertionError(f"Function-matched step-0 drift is {initial_max:.3e}")

    total_steps = int(training["updates"])
    step = 0
    reference_loss = math.nan
    changed_loss = math.nan
    try:
        while step < total_steps:
            for reference_x, changed_x, target in loader:
                if step >= total_steps:
                    break
                reference_x = reference_x.to(resolved, non_blocking=True)
                changed_x = changed_x.to(resolved, non_blocking=True)
                target = target.to(resolved, non_blocking=True)
                reference_optimizer.zero_grad(set_to_none=True)
                changed_optimizer.zero_grad(set_to_none=True)
                if reference_updater is not None:
                    reference.first.weight.grad = None
                    reference.first.bias.grad = None
                    changed.first.weight.grad = None
                    changed.first.bias.grad = None
                reference.train()
                changed.train()
                paired_rng = _rng_state(resolved)
                reference_objective = member_loss(
                    reference, reference_x, target, reference_data.task
                )
                _restore_rng(resolved, paired_rng)
                changed_objective = member_loss(changed, changed_x, target, changed_data.task)
                reference_objective.backward()
                changed_objective.backward()
                if reference_updater is not None and changed_updater is not None:
                    reference_updater.step(len(reference_x))
                    changed_updater.step(len(changed_x))
                reference_optimizer.step()
                changed_optimizer.step()
                step += 1
                reference_loss = float(reference_objective.detach().cpu())
                changed_loss = float(changed_objective.detach().cpu())
                if step in trajectory_steps:
                    trajectory.extend(
                        _trajectory_observation(
                            reference,
                            changed,
                            reference_data,
                            changed_data,
                            transform,
                            step=step,
                            device=resolved,
                            probe_rows=int(training["probe_rows_per_split"]),
                            batch_size=batch_size * 2,
                        )
                    )
        final = {}
        for side, model, data in (
            ("reference", reference, reference_data),
            ("changed", changed, changed_data),
        ):
            logits = _predict_numpy(model, data.x["test"], data.task, resolved, batch_size * 2)
            final.update(
                {f"{side}_test_{key}": value for key, value in metrics(data, logits, data.y["test"]).items()}
            )
        reference_primary = float(final["reference_test_primary"])
        changed_primary = float(final["changed_test_primary"])
        final["final_primary_harm"] = reference_primary - changed_primary
        final["final_normalized_harm"] = (reference_primary - changed_primary) / max(
            abs(reference_primary), 1e-12
        )
        final["reference_last_batch_loss"] = reference_loss
        final["changed_last_batch_loss"] = changed_loss
        final["updates"] = total_steps
        final["parameters_per_model"] = sum(parameter.numel() for parameter in reference.parameters())
        return final, trajectory
    finally:
        if reference_updater is not None:
            reference_updater.close()
        if changed_updater is not None:
            changed_updater.close()


def run(args: argparse.Namespace) -> None:
    rows = _read(args.output)
    trajectory_path = args.output.with_name(args.output.stem + "_trajectories.csv")
    trajectory_rows = _read(trajectory_path)

    def key(row: dict[str, str]) -> tuple[str, str, str, str, int]:
        return (
            row["dataset"],
            row["pair_label"],
            row["model"],
            row["arm"],
            int(row["seed"]),
        )

    complete = {key(row) for row in rows if not row.get("failure", "").strip()}
    for dataset_name in args.datasets:
        dataset = load_broad_dataset(dataset_name)
        for pair in _pair_specs(dataset, args.representation_pairs, args.kappas):
            relation_errors = {
                part: float(
                    np.linalg.norm(pair.reference[part] @ pair.transform - pair.changed[part])
                    / max(np.linalg.norm(pair.changed[part]), 1e-30)
                )
                for part in ("train", "val", "test")
            }
            reference_data = make_prepared(dataset, pair.reference, {})
            changed_data = make_prepared(dataset, pair.changed, {})
            for model_name in args.models:
                for arm in args.arms:
                    for seed in args.seeds:
                        candidate_key = (dataset_name, pair.label, model_name, arm, seed)
                        if candidate_key in complete:
                            continue
                        failure = ""
                        try:
                            result, observations = train_pair(
                                reference_data,
                                changed_data,
                                model_name=model_name,
                                arm=arm,
                                transform=pair.transform,
                                seed=seed,
                                device=args.device,
                            )
                        except Exception as error:  # failures are part of the result
                            failure = f"{type(error).__name__}: {error}"
                            result, observations = {}, []
                        base = {
                            "experiment": "function_matched_trajectory_decomposition",
                            "dataset": dataset_name,
                            "task": dataset.task,
                            "split_fingerprint": dataset.split_fingerprint,
                            "pair_family": pair.family,
                            "pair_label": pair.label,
                            "target_kappa": "" if pair.target_kappa is None else pair.target_kappa,
                            "realized_basis_condition": float(np.linalg.cond(pair.transform)),
                            "basis_relation_max_error": max(relation_errors.values()),
                            "model": model_name,
                            "arm": arm,
                            "seed": seed,
                            "failure": failure,
                            "pair_metadata": json.dumps(pair.metadata, sort_keys=True),
                            **{f"reference_{key}": value for key, value in geometry(pair.reference["train"]).items()},
                            **{f"changed_{key}": value for key, value in geometry(pair.changed["train"]).items()},
                        }
                        rows.append({**base, **result})
                        _write(args.output, rows)
                        trajectory_rows.extend({**base, **observation} for observation in observations)
                        _write(trajectory_path, trajectory_rows)
                        status = failure or (
                            f"harm={result['final_normalized_harm']:+.5f} "
                            f"drift={observations[-1]['prediction_drift']:.5f}"
                        )
                        print(
                            f"{dataset_name:12s} {pair.label:42s} {model_name:6s} "
                            f"{arm:26s} s{seed} {status}",
                            flush=True,
                        )


def main() -> None:
    cfg = config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=cfg["datasets"])
    parser.add_argument(
        "--representation-pairs",
        nargs="+",
        choices=("controlled", "natural"),
        default=list(cfg["representation_pairs"]),
    )
    parser.add_argument(
        "--kappas", nargs="+", type=float, default=cfg["representation_pairs"]["controlled"]["kappas"]
    )
    parser.add_argument("--models", nargs="+", default=cfg["models"])
    parser.add_argument("--arms", nargs="+", default=list(cfg["arms"]))
    parser.add_argument("--seeds", nargs="+", type=int, default=cfg["seeds"])
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=RESULTS / "runs.csv")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
