"""Analyze the frozen Day-5 Tier-1 orbit and held-out action transfer."""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
IDEA = HERE.parent / "road-to-iclr-idea-search"
sys.path.insert(0, str(IDEA))
from orbit_anova import budgeted_marginalization_frontier, decompose  # noqa: E402


def proper_loss(y: np.ndarray, predictions: np.ndarray) -> float:
    if predictions.shape[-1] == 1:
        return float(np.mean((predictions[..., 0] - y) ** 2))
    targets = np.eye(predictions.shape[-1])[y.astype(int)]
    return float(np.mean(np.sum((predictions - targets) ** 2, axis=-1)))


def schema_risk(predictions: np.ndarray) -> float:
    flat = predictions.reshape((-1,) + predictions.shape[-2:]).astype(np.float64)
    centroid = flat.mean(axis=0)
    return float(np.mean(np.sum((flat - centroid) ** 2, axis=-1)))


def action_frontier(predictions: np.ndarray, budget: int) -> dict[tuple[str, ...], dict[str, Any]]:
    names = ("feature", "category", "class")
    return {
        tuple(item["factors"]): item
        for item in budgeted_marginalization_frontier(
            predictions.astype(np.float64), names, budget
        )
    }


def average_frontier_by_seed(predictions: np.ndarray, budget: int) -> dict[tuple[str, ...], dict[str, float]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for seed in range(predictions.shape[3]):
        for key, item in action_frontier(predictions[..., seed, :, :], budget).items():
            grouped.setdefault(key, []).append(item)
    output = {}
    for key, items in grouped.items():
        total = np.mean([float(item["residual_risk"]) for item in items])
        expected = np.mean([float(item["expected_residual_risk"]) for item in items])
        output[key] = {
            "expected_residual": float(expected),
            "total_risk": float(total),
            "expected_fraction": float(expected / total) if total else 0.0,
            "realized_cost": float(np.mean([int(item["realized_cost"]) for item in items])),
        }
    return output


def action_proper_loss(predictions: np.ndarray, y: np.ndarray, budget: int, key: tuple[str, ...]) -> float:
    values = []
    for seed in range(predictions.shape[3]):
        seed_predictions = predictions[..., seed, :, :].astype(np.float64)
        item = action_frontier(seed_predictions, budget)[key]
        centroid = seed_predictions.reshape((-1,) + seed_predictions.shape[-2:]).mean(axis=0)
        values.append(proper_loss(y, centroid) + float(item["expected_residual_risk"]))
    return float(np.mean(values))


def seed_ensemble_metrics(predictions: np.ndarray, y: np.ndarray, budget: int) -> dict[str, float]:
    seed_count = predictions.shape[3]
    if budget > seed_count:
        return {"residual": np.nan, "proper_loss": np.nan}
    groups = [tuple(range(start, start + budget)) for start in range(0, seed_count - budget + 1, budget)]
    residuals = []
    losses = []
    for group in groups:
        ensemble = predictions[..., list(group), :, :].mean(axis=3).astype(np.float64)
        residuals.append(schema_risk(ensemble))
        flat = ensemble.reshape((-1,) + ensemble.shape[-2:])
        losses.append(float(np.mean([proper_loss(y, member) for member in flat])))
    return {"residual": float(np.mean(residuals)), "proper_loss": float(np.mean(losses))}


def analyze_cell(npz_path: Path, manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    archive = np.load(npz_path)
    results: dict[str, Any] = {}
    for split in ("validation", "test"):
        predictions = archive[f"{split}_predictions"].astype(np.float64)
        y = archive[f"{split}_y"]
        persistent = predictions.mean(axis=3)
        persistent_anova = decompose(persistent, ("feature", "category", "class"))
        conditional_anovas = [
            decompose(predictions[..., seed, :, :], ("feature", "category", "class"))
            for seed in range(predictions.shape[3])
        ]
        conditional_total = float(np.mean([item["total"] for item in conditional_anovas]))
        flat = predictions.reshape((-1,) + predictions.shape[-2:])
        centroid = flat.mean(axis=0)
        joint_risk = float(np.mean(np.sum((flat - centroid) ** 2, axis=-1)))
        mean_member_loss = float(np.mean([proper_loss(y, member) for member in flat]))
        centroid_loss = proper_loss(y, centroid)
        risk = schema_risk(persistent)
        if predictions.shape[-1] > 1:
            hard = np.argmax(persistent.reshape((-1,) + persistent.shape[-2:]), axis=-1)
            hard_flip = float(np.mean(np.any(hard != hard[0:1], axis=0)))
        else:
            hard_flip = np.nan
        results[split] = {
            "persistent_schema_risk": risk,
            "same_seed_conditional_schema_risk": conditional_total,
            "persistent_fraction_of_member_loss": risk / mean_member_loss if mean_member_loss else np.nan,
            "conditional_fraction_of_member_loss": conditional_total / mean_member_loss if mean_member_loss else np.nan,
            "mean_member_proper_loss": mean_member_loss,
            "quotient_centroid_proper_loss": centroid_loss,
            "joint_schema_seed_risk": joint_risk,
            "ambiguity_identity_error": abs(mean_member_loss - centroid_loss - joint_risk),
            "hard_flip_fraction": hard_flip,
            "persistent_anova": persistent_anova,
            "conditional_anova_mean": {
                key: float(np.mean([item[key] for item in conditional_anovas]))
                for key in conditional_anovas[0]
            },
            "maximum_probability_deviation": float(np.max(np.abs(predictions - predictions[0, 0, 0, 0]))),
        }
    return results, {
        "validation_predictions": archive["validation_predictions"],
        "test_predictions": archive["test_predictions"],
        "validation_y": archive["validation_y"],
        "test_y": archive["test_y"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "tier1_config.json")
    parser.add_argument("--input-dir", type=Path, default=HERE / "results" / "tier1")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    arrays: dict[tuple[str, str], dict[str, Any]] = {}
    missing = []
    for dataset, model in itertools.product(config["datasets"], config["models"]):
        stem = f"{dataset}__{model}"
        manifest_path = args.input_dir / f"{stem}.json"
        npz_path = args.input_dir / f"{stem}.npz"
        if not manifest_path.exists() or not npz_path.exists():
            missing.append(stem)
            continue
        manifest = json.loads(manifest_path.read_text())
        analysis, cell_arrays = analyze_cell(npz_path, manifest)
        cells[(dataset, model)] = {"manifest": manifest, **analysis}
        arrays[(dataset, model)] = cell_arrays
    if missing:
        raise RuntimeError(f"missing cells: {missing}")

    action_rows = []
    classification = [
        dataset for dataset in config["datasets"]
        if cells[(dataset, config["models"][0])]["manifest"]["task"] == "binclass"
    ]
    for model in config["models"]:
        for heldout in classification:
            development = [dataset for dataset in classification if dataset != heldout]
            for budget in config["action_budgets"]:
                development_frontiers = [
                    average_frontier_by_seed(arrays[(dataset, model)]["validation_predictions"], int(budget))
                    for dataset in development
                ]
                keys = sorted(set.intersection(*(set(frontier) for frontier in development_frontiers)))
                scores = {
                    key: float(np.mean([frontier[key]["expected_fraction"] for frontier in development_frontiers]))
                    for key in keys
                }
                selected = min(keys, key=lambda key: (scores[key], len(key), key))
                heldout_validation = average_frontier_by_seed(
                    arrays[(heldout, model)]["validation_predictions"], int(budget)
                )
                oracle = min(keys, key=lambda key: (heldout_validation[key]["expected_fraction"], len(key), key))
                test_frontier = average_frontier_by_seed(
                    arrays[(heldout, model)]["test_predictions"], int(budget)
                )
                generic = test_frontier[()]
                chosen = test_frontier[selected]
                seed = seed_ensemble_metrics(
                    arrays[(heldout, model)]["test_predictions"],
                    arrays[(heldout, model)]["test_y"],
                    int(budget),
                )
                chosen_loss = action_proper_loss(
                    arrays[(heldout, model)]["test_predictions"],
                    arrays[(heldout, model)]["test_y"],
                    int(budget), selected,
                )
                generic_loss = action_proper_loss(
                    arrays[(heldout, model)]["test_predictions"],
                    arrays[(heldout, model)]["test_y"],
                    int(budget), (),
                )
                reference_loss = float(cells[(heldout, model)]["test"]["mean_member_proper_loss"])
                material = bool(cells[(heldout, model)]["test"]["conditional_fraction_of_member_loss"] >= 0.005)
                action_rows.append({
                    "dataset": heldout, "model": model, "budget": int(budget),
                    "selected_factors": "+".join(selected) if selected else "iid",
                    "heldout_validation_oracle_factors": "+".join(oracle) if oracle else "iid",
                    "development_expected_residual_fraction": scores[selected],
                    "test_action_residual": chosen["expected_residual"],
                    "test_iid_schema_residual": generic["expected_residual"],
                    "test_seed_ensemble_residual": seed["residual"],
                    "action_vs_iid_residual_reduction": 1.0 - chosen["expected_residual"] / generic["expected_residual"] if generic["expected_residual"] else 0.0,
                    "action_vs_seed_residual_reduction": 1.0 - chosen["expected_residual"] / seed["residual"] if seed["residual"] else 0.0,
                    "action_expected_proper_loss": chosen_loss,
                    "iid_expected_proper_loss": generic_loss,
                    "seed_ensemble_proper_loss": seed["proper_loss"],
                    "action_relative_loss_vs_iid": (chosen_loss - generic_loss) / reference_loss if reference_loss else np.nan,
                    "action_relative_loss_vs_seed": (chosen_loss - seed["proper_loss"]) / reference_loss if reference_loss else np.nan,
                    "material_cell": material,
                })

    rows = []
    for (dataset, model), cell in cells.items():
        for split in ("validation", "test"):
            result = cell[split]
            row = {
                "dataset": dataset, "model": model, "task": cell["manifest"]["task"], "split": split,
                **{key: value for key, value in result.items() if not isinstance(value, dict)},
            }
            for key, value in result["conditional_anova_mean"].items():
                if key not in {"component_sum_error", "prediction_reconstruction_max_error"}:
                    row[f"anova_{key}"] = value
            rows.append(row)
    frame = pd.DataFrame(rows)
    actions = pd.DataFrame(action_rows)
    material_actions = actions[actions.material_cell]
    rope = float(config["proper_loss_rope_relative"])
    pass_iid = material_actions.action_vs_iid_residual_reduction > 0
    pass_seed = material_actions.action_vs_seed_residual_reduction > 0
    pass_loss = material_actions.action_relative_loss_vs_iid <= rope
    gate = bool(
        len(material_actions)
        and (pass_iid & pass_seed & pass_loss).sum() > len(material_actions) / 2
        and material_actions.action_relative_loss_vs_iid.mean() <= rope
    )
    summary = {
        "status": "complete",
        "cells": len(cells),
        "classification_action_rows": len(actions),
        "material_action_rows": len(material_actions),
        "material_cells_beating_iid_and_seed_without_loss_violation": int((pass_iid & pass_seed & pass_loss).sum()),
        "action_gate_passed": gate,
        "test_material_cells": int((frame[(frame.split == "test")].conditional_fraction_of_member_loss >= 0.005).sum()),
        "maximum_ambiguity_identity_error": float(frame.ambiguity_identity_error.max()),
        "maximum_anova_component_sum_error": float(max(
            cell[split]["persistent_anova"]["component_sum_error"]
            for cell in cells.values() for split in ("validation", "test")
        )),
        "mean_action_vs_iid_residual_reduction_material": float(material_actions.action_vs_iid_residual_reduction.mean()) if len(material_actions) else np.nan,
        "mean_action_vs_seed_residual_reduction_material": float(material_actions.action_vs_seed_residual_reduction.mean()) if len(material_actions) else np.nan,
        "mean_action_relative_loss_vs_iid_material": float(material_actions.action_relative_loss_vs_iid.mean()) if len(material_actions) else np.nan,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "tier1_cell_summary.csv", index=False)
    actions.to_csv(args.output_dir / "tier1_action_transfer.csv", index=False)
    (args.output_dir / "tier1_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
