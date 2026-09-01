#!/usr/bin/env python3
"""Falsify the Day 1 atomic-identity interpretation against tuned T-PLE."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

import real_data_benchmark as benchmark
from adult_identity_schema_matched import (
    fit_path,
    fixed_identity_encoded,
    load_fit,
    metrics,
    paired_test_bootstrap,
    save_fit,
    sigmoid,
)


ROOT = Path(__file__).resolve().parent


def _encode_edges(
    parts: dict[str, np.ndarray], edges: list[np.ndarray]
) -> dict[str, np.ndarray]:
    transformed: dict[str, list[np.ndarray]] = {part: [] for part in parts}
    for column, knots in enumerate(edges):
        left, right = knots[:-1], knots[1:]
        width = np.maximum(right - left, 1e-12)
        for part, values in parts.items():
            transformed[part].append(
                np.clip(
                    (values[:, column, None] - left[None, :]) / width[None, :],
                    0.0,
                    1.0,
                )
            )
    return {
        part: np.column_stack(columns).astype(np.float32)
        for part, columns in transformed.items()
    }


def tree_piecewise_linear(
    parts: dict[str, np.ndarray],
    target: np.ndarray,
    n_bins: int,
    min_samples_leaf: int,
    min_impurity_decrease: float,
) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    """Paper-faithful T-PLE: one supervised 1D tree per numerical feature."""

    train = parts["train"]
    edges = []
    for column in range(train.shape[1]):
        values = train[:, column]
        tree = DecisionTreeClassifier(
            max_leaf_nodes=n_bins,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            random_state=0,
        ).fit(values[:, None], target).tree_
        thresholds = tree.threshold[tree.children_left != tree.children_right]
        knots = np.unique(np.r_[values.min(), thresholds, values.max()])
        if len(knots) < 2:
            knots = np.array([knots[0], knots[0] + 1.0])
        edges.append(knots.astype(np.float64))
    return _encode_edges(parts, edges), edges


def high_mass_atoms(
    train: np.ndarray, selected: tuple[int, ...], min_count: int
) -> dict[int, np.ndarray]:
    atoms = {}
    for column in selected:
        values, counts = np.unique(train[:, column], return_counts=True)
        atoms[column] = values[counts >= min_count]
    return atoms


def atomic_indicators(
    parts: dict[str, np.ndarray], atoms: dict[int, np.ndarray]
) -> dict[str, np.ndarray]:
    output = {}
    for part, values in parts.items():
        columns = [
            (values[:, column, None] == atom_values[None, :]).astype(np.float32)
            for column, atom_values in atoms.items()
            if len(atom_values)
        ]
        output[part] = (
            np.column_stack(columns).astype(np.float32)
            if columns
            else np.empty((len(values), 0), dtype=np.float32)
        )
    return output


def atom_bracketed_piecewise_linear(
    parts: dict[str, np.ndarray], bins: int, atoms: dict[int, np.ndarray]
) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    """Q-PLE plus train-support midpoints around atoms, without equality flags."""

    train = parts["train"]
    edges = []
    for column in range(train.shape[1]):
        knots = list(
            np.unique(
                np.quantile(train[:, column], np.linspace(0.0, 1.0, bins + 1))
            )
        )
        support = np.unique(train[:, column])
        for atom in atoms.get(column, np.empty(0)):
            position = int(np.searchsorted(support, atom))
            if position > 0:
                knots.append(float((support[position - 1] + atom) / 2.0))
            if position + 1 < len(support):
                knots.append(float((atom + support[position + 1]) / 2.0))
        feature_edges = np.unique(knots)
        if len(feature_edges) < 2:
            feature_edges = np.array([feature_edges[0], feature_edges[0] + 1.0])
        edges.append(feature_edges.astype(np.float64))
    return _encode_edges(parts, edges), edges


def replace_ple(
    baseline: benchmark.EncodedDataset,
    ple: dict[str, np.ndarray],
    name: str,
) -> benchmark.EncodedDataset:
    schema_size = baseline.view_sizes[0]
    x = {
        part: np.ascontiguousarray(
            np.column_stack([baseline.x[part][:, :schema_size], ple[part]]),
            dtype=np.float32,
        )
        for part in baseline.x
    }
    return benchmark.EncodedDataset(
        x=x,
        y=baseline.y,
        task=baseline.task,
        y_mean=baseline.y_mean,
        y_scale=baseline.y_scale,
        view_names=(baseline.view_names[0], name),
        view_sizes=(schema_size, ple["train"].shape[1]),
        selected_numeric=(),
    )


def append_view(
    baseline: benchmark.EncodedDataset,
    view: dict[str, np.ndarray],
    name: str,
    selected: tuple[int, ...] = (),
) -> benchmark.EncodedDataset:
    x = {
        part: np.ascontiguousarray(
            np.column_stack([baseline.x[part], view[part]]), dtype=np.float32
        )
        for part in baseline.x
    }
    return benchmark.EncodedDataset(
        x=x,
        y=baseline.y,
        task=baseline.task,
        y_mean=baseline.y_mean,
        y_scale=baseline.y_scale,
        view_names=(*baseline.view_names, name),
        view_sizes=(*baseline.view_sizes, view["train"].shape[1]),
        selected_numeric=selected,
    )


def grid_key(candidate: dict[str, float | int]) -> str:
    return (
        f"b{candidate['n_bins']}_leaf{candidate['min_samples_leaf']}"
        f"_gain{float(candidate['min_impurity_decrease']):g}"
    )


def train_model(
    encoded: benchmark.EncodedDataset,
    config: dict,
    output: Path,
    device: torch.device,
    family: str,
    seed: int,
    variant: str,
    target_parameters: int,
) -> None:
    path = fit_path(output, family, seed, variant)
    if path.exists() and path.with_suffix(".npz").exists():
        print(f"skip {path.name}", flush=True)
        return
    result = benchmark.train_one(
        encoded,
        seed,
        benchmark.BATCH_SIZES[config["dataset"]],
        device,
        config["width"],
        config["depth"],
        config["dropout"],
        config["learning_rate"],
        config["weight_decay"],
        config["max_epochs"],
        config["patience"],
        gated=False,
        gate_entropy_weight=0.0,
        target_parameters=target_parameters if config["match_parameters"] else None,
        model_type=config["model"],
        activation=config["activation"],
    )
    save_fit(path, result, encoded.selected_numeric)
    print(
        f"{path.name}: val_accuracy={result.result['val_score']:.4f} "
        f"features={result.result['input_features']} "
        f"parameters={result.result['parameters']}",
        flush=True,
    )


def prepare(config: dict, seed: int) -> tuple[
    benchmark.Dataset,
    benchmark.EncodedDataset,
    dict[str, object],
    dict[int, np.ndarray],
    int,
]:
    dataset = benchmark.load_dataset(ROOT / "data", config["dataset"])
    cache: dict[str, object] = {}
    qple = benchmark.encode_dataset(
        dataset,
        "schema_ple",
        seed,
        config["quantile_bins"],
        128,
        20.0,
        0.001,
        cache,
    )
    clean = cast(dict[str, np.ndarray], cache["clean_num"])
    min_count = math.ceil(len(clean["train"]) / config["quantile_bins"])
    atoms = high_mass_atoms(
        clean["train"], tuple(config["selected_numeric"]), min_count
    )
    target_parameters = benchmark._parameter_count(
        benchmark._make_model(
            qple,
            config["width"],
            config["depth"],
            config["dropout"],
            False,
            config["model"],
            config["activation"],
        )
    )
    return dataset, qple, cache, atoms, target_parameters


def tune(config: dict, output: Path, device: torch.device) -> dict:
    rows = []
    for seed in config["tuning_seeds"]:
        dataset, qple, cache, _, target_parameters = prepare(config, seed)
        clean = cast(dict[str, np.ndarray], cache["clean_num"])
        for order, candidate in enumerate(config["tple_grid"]):
            key = grid_key(candidate)
            ple, edges = tree_piecewise_linear(
                clean,
                dataset.y["train"],
                int(candidate["n_bins"]),
                int(candidate["min_samples_leaf"]),
                float(candidate["min_impurity_decrease"]),
            )
            encoded = replace_ple(qple, ple, "numeric_tple")
            train_model(
                encoded, config, output, device, "tple_tune", seed, key,
                target_parameters,
            )
            val_logits, _, record = load_fit(
                fit_path(output, "tple_tune", seed, key)
            )
            rows.append(
                {
                    "order": order,
                    "seed": seed,
                    "key": key,
                    **candidate,
                    "total_tree_bins": sum(len(x) - 1 for x in edges),
                    "val_log_loss": log_loss(
                        dataset.y["val"], sigmoid(val_logits)
                    ),
                    "val_auc": roc_auc_score(dataset.y["val"], sigmoid(val_logits)),
                    "val_accuracy": record["val_score"],
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "tple_grid_runs.csv", index=False)
    ranked = (
        frame.groupby(
            [
                "order", "key", "n_bins", "min_samples_leaf",
                "min_impurity_decrease", "total_tree_bins",
            ],
            as_index=False,
        )[["val_log_loss", "val_auc", "val_accuracy"]]
        .mean()
        .sort_values(["val_log_loss", "total_tree_bins", "order"])
    )
    ranked.to_csv(output / "tple_grid_summary.csv", index=False)
    winner = ranked.iloc[0].to_dict()
    selected = {
        "key": winner["key"],
        "n_bins": int(winner["n_bins"]),
        "min_samples_leaf": int(winner["min_samples_leaf"]),
        "min_impurity_decrease": float(winner["min_impurity_decrease"]),
        "mean_val_log_loss": float(winner["val_log_loss"]),
        "mean_val_auc": float(winner["val_auc"]),
        "total_tree_bins": int(winner["total_tree_bins"]),
    }
    (output / "selected_tple.json").write_text(
        json.dumps(selected, indent=2, sort_keys=True) + "\n"
    )
    print(f"selected T-PLE: {selected}", flush=True)
    return selected


def confirm(config: dict, output: Path, device: torch.device, selected: dict) -> None:
    selected_numeric = tuple(config["selected_numeric"])
    for seed in config["confirmation_seeds"]:
        dataset, qple, cache, atoms, target_parameters = prepare(config, seed)
        clean = cast(dict[str, np.ndarray], cache["clean_num"])
        tple_values, _ = tree_piecewise_linear(
            clean,
            dataset.y["train"],
            selected["n_bins"],
            selected["min_samples_leaf"],
            selected["min_impurity_decrease"],
        )
        tple = replace_ple(qple, tple_values, "numeric_tple")
        full_identity = fixed_identity_encoded(tple, cache, selected_numeric)
        atom_view = atomic_indicators(clean, atoms)
        atom_identity = append_view(
            qple, atom_view, "numeric_atom_identity", selected_numeric
        )
        bracket_values, _ = atom_bracketed_piecewise_linear(
            clean, config["quantile_bins"], atoms
        )
        bracketed = replace_ple(qple, bracket_values, "numeric_atom_bracketed_ple")
        for family, encoded in (
            ("tple", tple),
            ("tple_full_identity", full_identity),
            ("qple_atom_identity", atom_identity),
            ("atom_bracketed_ple", bracketed),
        ):
            train_model(
                encoded, config, output, device, family, seed, "SELECTED",
                target_parameters,
            )


def baseline_path(family: str, seed: int) -> Path:
    old_family = "ple" if family == "qple" else "identity"
    variant = "RAW" if family == "qple" else "SELECTED"
    return ROOT / "adult_identity_schema_matched" / "run_records" / f"{old_family}__seed{seed}__{variant}.json"


def analyze(config: dict, output: Path, selected: dict) -> None:
    dataset = benchmark.load_dataset(ROOT / "data", config["dataset"])
    labels = {
        "qple": "Q-PLE",
        "qple_full_identity": "Q-PLE + full identity",
        "tple": "T-PLE",
        "tple_full_identity": "T-PLE + full identity",
        "qple_atom_identity": "Q-PLE + atom-only indicators",
        "atom_bracketed_ple": "Atom-bracketed Q-PLE",
    }
    predictions: dict[str, dict[str, list[np.ndarray]]] = {}
    rows = []
    for family in labels:
        predictions[family] = {"val": [], "test": []}
        for seed in config["confirmation_seeds"]:
            path = (
                baseline_path(family, seed)
                if family in ("qple", "qple_full_identity")
                else fit_path(output, family, seed, "SELECTED")
            )
            val_logit, test_logit, record = load_fit(path)
            for part, logit in (("val", val_logit), ("test", test_logit)):
                probability = sigmoid(logit)
                predictions[family][part].append(probability)
                rows.append(
                    {
                        "system": labels[family],
                        "family": family,
                        "seed": seed,
                        "part": part,
                        **metrics(probability, dataset.y[part]),
                        "input_features": record["input_features"],
                        "parameters": record["parameters"],
                    }
                )
    members = pd.DataFrame(rows)
    members.to_csv(output / "members.csv", index=False)
    ensemble_rows = []
    for family, label in labels.items():
        for part in ("val", "test"):
            probability = np.mean(predictions[family][part], axis=0)
            ensemble_rows.append(
                {
                    "system": label,
                    "family": family,
                    "part": part,
                    **metrics(probability, dataset.y[part]),
                }
            )
    ensembles = pd.DataFrame(ensemble_rows)
    ensembles.to_csv(output / "ensembles.csv", index=False)

    def contrast(candidate: str, reference: str, part: str) -> dict[str, float]:
        cand_members = members[(members.family == candidate) & (members.part == part)]
        ref_members = members[(members.family == reference) & (members.part == part)]
        cand_ensemble = ensembles[(ensembles.family == candidate) & (ensembles.part == part)].iloc[0]
        ref_ensemble = ensembles[(ensembles.family == reference) & (ensembles.part == part)].iloc[0]
        return {
            "mean_member_log_loss_delta": float(
                cand_members.log_loss.mean() - ref_members.log_loss.mean()
            ),
            "mean_member_auc_delta_pp": float(
                100 * (cand_members.auc.mean() - ref_members.auc.mean())
            ),
            "ensemble_accuracy_delta_pp": float(
                100 * (cand_ensemble.accuracy - ref_ensemble.accuracy)
            ),
            "ensemble_auc_delta_pp": float(
                100 * (cand_ensemble.auc - ref_ensemble.auc)
            ),
            "ensemble_log_loss_delta": float(
                cand_ensemble.log_loss - ref_ensemble.log_loss
            ),
        }

    contrast_rows = []
    for name, candidate, reference in (
        ("full identity over Q-PLE", "qple_full_identity", "qple"),
        ("T-PLE over Q-PLE", "tple", "qple"),
        ("full identity over T-PLE", "tple_full_identity", "tple"),
        ("atom-only indicators over Q-PLE", "qple_atom_identity", "qple"),
        ("atom-bracketing over Q-PLE", "atom_bracketed_ple", "qple"),
    ):
        for part in ("val", "test"):
            contrast_rows.append(
                {"contrast": name, "part": part, **contrast(candidate, reference, part)}
            )
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(output / "contrasts.csv", index=False)

    uncertainty_rows = []
    for name, candidate, reference in (
        ("T-PLE over Q-PLE", "tple", "qple"),
        ("full identity over T-PLE", "tple_full_identity", "tple"),
        ("atom-only indicators over Q-PLE", "qple_atom_identity", "qple"),
    ):
        candidate_prediction = np.mean(predictions[candidate]["test"], axis=0)
        reference_prediction = np.mean(predictions[reference]["test"], axis=0)
        uncertainty_rows.append(
            {
                "contrast": name,
                **paired_test_bootstrap(
                    candidate_prediction,
                    reference_prediction,
                    dataset.y["test"],
                ),
            }
        )
    uncertainty = pd.DataFrame(uncertainty_rows)
    uncertainty.to_csv(output / "test_bootstrap.csv", index=False)

    primary_val = contrasts[
        (contrasts.contrast == "full identity over T-PLE")
        & (contrasts.part == "val")
    ].iloc[0]
    primary_test = contrasts[
        (contrasts.contrast == "full identity over T-PLE")
        & (contrasts.part == "test")
    ].iloc[0]
    atom_val = contrasts[
        (contrasts.contrast == "atom-only indicators over Q-PLE")
        & (contrasts.part == "val")
    ].iloc[0]
    transfer = bool(
        primary_val.mean_member_log_loss_delta < 0
        and primary_val.mean_member_auc_delta_pp > 0
        and primary_test.ensemble_auc_delta_pp > 0
    )
    atom_specific = bool(
        atom_val.mean_member_log_loss_delta < 0
        and atom_val.mean_member_auc_delta_pp > 0
        and contrasts[
            (contrasts.contrast == "atom-only indicators over Q-PLE")
            & (contrasts.part == "test")
        ].iloc[0].ensemble_auc_delta_pp > 0
    )
    decision = {
        "broader_transfer_gate_passed": transfer,
        "atom_specific_gate_passed": atom_specific,
        "atom_effect_assessment": "directionally positive but practically negligible on Adult",
        "interpretation": (
            "identity survives tuned T-PLE and merits a frozen transfer test"
            if transfer
            else "identity does not robustly survive tuned T-PLE on the predeclared gate"
        ),
    }
    (output / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )

    test_table = ensembles[ensembles.part == "test"]
    lines = [
        "# Adult T-PLE atomic falsification",
        "",
        config["development_status"],
        "",
        f"Validation selected `{selected['key']}` with {selected['total_tree_bins']} total numerical bins (mean validation log loss {selected['mean_val_log_loss']:.6f}). All neural networks were matched to the original Q-PLE ResNet parameter budget.",
        "",
        "## Four-seed ensemble results",
        "",
        "| System | Test accuracy | Test AUC | Test log loss |",
        "| --- | ---: | ---: | ---: |",
    ]
    for _, row in test_table.iterrows():
        lines.append(
            f"| {row['system']} | {row['accuracy']:.4f} | {row['auc']:.4f} | {row['log_loss']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Predeclared contrasts",
            "",
            "Negative log-loss deltas and positive AUC deltas favor the candidate.",
            "",
            "| Contrast | Part | Mean-member log-loss delta | Mean-member AUC delta | Ensemble AUC delta |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for _, row in contrasts.iterrows():
        lines.append(
            f"| {row['contrast']} | {row['part']} | {row['mean_member_log_loss_delta']:+.6f} | "
            f"{row['mean_member_auc_delta_pp']:+.4f} pp | {row['ensemble_auc_delta_pp']:+.4f} pp |"
        )
    lines.extend(
        [
            "",
            "## Fixed-test-row uncertainty",
            "",
            "These paired bootstrap intervals condition on the already-used Adult test split and do not measure training-seed or dataset uncertainty.",
            "",
            "| Contrast | Accuracy delta (95% CI) | AUC delta (95% CI) |",
            "| --- | ---: | ---: |",
        ]
    )
    for _, row in uncertainty.iterrows():
        lines.append(
            f"| {row['contrast']} | {row['accuracy_difference_pp']:+.4f} "
            f"[{row['accuracy_ci_low_pp']:+.4f}, {row['accuracy_ci_high_pp']:+.4f}] pp | "
            f"{row['auc_difference_pp']:+.4f} [{row['auc_ci_low_pp']:+.4f}, "
            f"{row['auc_ci_high_pp']:+.4f}] pp |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Broader-transfer gate: **{'PASS' if transfer else 'FAIL'}**.",
            f"- Directional atom-specific gate: **{'PASS' if atom_specific else 'FAIL'}**, but the observed atom-only effect is practically negligible.",
            "",
            "The full-identity view encodes every observed value in numerical columns 3, 4, and 5. The atom-only view encodes only values occurring at least `ceil(n_train / 128)` times. Atom-bracketed Q-PLE adds train-support midpoints around those atoms but no equality indicator. T-PLE uses one target-aware one-dimensional decision tree per numerical feature, exactly the construction used by the official PLE implementation.",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "adult_tple_atomic_falsification.json",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "adult_tple_atomic_falsification"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--stage", choices=("all", "tune", "confirm", "analyze"), default="all"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    selected_path = args.output / "selected_tple.json"
    if args.stage in ("all", "tune"):
        selected = tune(config, args.output, torch.device(args.device))
    else:
        selected = json.loads(selected_path.read_text())
    if args.stage in ("all", "confirm"):
        confirm(config, args.output, torch.device(args.device), selected)
    if args.stage in ("all", "analyze"):
        analyze(config, args.output, selected)


if __name__ == "__main__":
    main()
