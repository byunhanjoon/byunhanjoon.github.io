#!/usr/bin/env python3
"""Integrity-first analysis of the frozen small E2 PriorDial phase diagram."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import total_variation
from src.stats import mean_interval


MODELS = ("tabicl_v2_single", "tabicl_v2_default", "mitra_default")
TASKS = ("classification", "regression")
RHOS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
MECHANISMS = {"linear", "additive", "threshold", "interaction", "partition", "periodic"}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-hash-prefix", default="d1e07cf837")
    parser.add_argument("--tag", default="v1")
    return parser.parse_args()


def normalize_probabilities(prediction: np.ndarray) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.ndim != 3 or prediction.shape[-1] != 2:
        raise AssertionError(f"expected binary probability tensor, got {prediction.shape}")
    if not np.all(np.isfinite(prediction)) or np.any(prediction < 0):
        raise AssertionError("classification predictions must be finite and nonnegative")
    denominator = prediction.sum(axis=-1, keepdims=True)
    if np.any(denominator <= 0):
        raise AssertionError("classification predictions have a zero-sum row")
    normalized = prediction / denominator
    return np.clip(normalized, 1e-15, 1 - 1e-15)


def binary_log_loss(y: np.ndarray, probability: np.ndarray) -> float:
    rows = np.arange(y.size)
    return float(-np.log(probability[rows, y.astype(int)]).mean())


def exclusive_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, mode="x")


def main() -> None:
    args = arguments()
    cell_rows: list[dict] = []
    integrity_rows: list[dict] = []
    expected = {(model, task) for model in MODELS for task in TASKS}
    found: set[tuple[str, str]] = set()

    for model, task in sorted(expected):
        stem = f"e2_{model}_{task}_{args.config_hash_prefix}"
        raw_path = ROOT / "results/raw" / f"{stem}.npz"
        metadata_path = ROOT / "results/raw" / f"{stem}.metadata.json"
        metrics_path = ROOT / "results/processed" / f"{stem}_metrics.csv"
        if not all(path.exists() for path in (raw_path, metadata_path, metrics_path)):
            missing = [str(path.relative_to(ROOT)) for path in (raw_path, metadata_path, metrics_path) if not path.exists()]
            raise FileNotFoundError(f"incomplete E2 bundle for {model}/{task}: {missing}")
        metadata = json.loads(metadata_path.read_text())
        metrics = pd.read_csv(metrics_path)
        with np.load(raw_path, allow_pickle=False) as bundle:
            arrays = {key: bundle[key] for key in bundle.files}
        required = {
            "prediction_clean", "prediction_matched", "prediction_identity", "y_query",
            "rho", "mechanism", "warp", "transform_states",
        }
        if set(arrays) != required:
            raise AssertionError(f"unexpected arrays for {stem}: {set(arrays)}")
        if len(metrics) != 42 or arrays["y_query"].shape != (42, 64):
            raise AssertionError(f"unexpected episode/query count for {stem}")
        if set(arrays["mechanism"].tolist()) != MECHANISMS or set(arrays["rho"].tolist()) != set(RHOS):
            raise AssertionError(f"incomplete phase grid for {stem}")
        if not np.array_equal(metrics["rho"].to_numpy(), arrays["rho"]):
            raise AssertionError(f"rho order mismatch for {stem}")
        if not np.array_equal(metrics["mechanism"].to_numpy(), arrays["mechanism"]):
            raise AssertionError(f"mechanism order mismatch for {stem}")
        if not all(isinstance(json.loads(state), list) for state in arrays["transform_states"]):
            raise AssertionError(f"invalid transform state serialization for {stem}")
        if not all(np.all(np.isfinite(arrays[key])) for key in ("prediction_clean", "prediction_matched", "prediction_identity", "y_query")):
            raise AssertionError(f"nonfinite raw values for {stem}")

        if task == "classification":
            raw_predictions = [arrays[key] for key in ("prediction_clean", "prediction_matched", "prediction_identity")]
            maximum_row_sum_error = max(float(np.max(np.abs(value.sum(-1) - 1))) for value in raw_predictions)
            clean, matched, identity = map(normalize_probabilities, raw_predictions)
        else:
            maximum_row_sum_error = np.nan
            clean, matched, identity = (arrays[key].astype(np.float64) for key in ("prediction_clean", "prediction_matched", "prediction_identity"))
            if clean.shape != (42, 64) or matched.shape != clean.shape or identity.shape != clean.shape:
                raise AssertionError(f"unexpected regression prediction shape for {stem}")

        for index, source in metrics.iterrows():
            y = arrays["y_query"][index]
            if task == "classification":
                clean_loss = binary_log_loss(y, clean[index])
                matched_loss = binary_log_loss(y, matched[index])
                disagreement = float(total_variation(clean[index], matched[index]).mean())
                identity_disagreement = float(total_variation(clean[index], identity[index]).mean())
            else:
                clean_loss = float(np.mean((clean[index] - y) ** 2))
                matched_loss = float(np.mean((matched[index] - y) ** 2))
                # The run CSV stores the context-label scale, which is not duplicated in
                # the immutable prediction bundle. Its disagreement fields are therefore
                # retained after verifying all prediction-derived losses exactly.
                disagreement = float(source["matched_disagreement"])
                identity_disagreement = float(source["identity_disagreement"])
            if not np.isclose(clean_loss, source["clean_loss"], rtol=2e-5, atol=2e-8):
                raise AssertionError(f"clean loss mismatch for {stem} episode {index}")
            if not np.isclose(matched_loss, source["matched_loss"], rtol=2e-5, atol=2e-8):
                raise AssertionError(f"matched loss mismatch for {stem} episode {index}")
            cell_rows.append({
                **source.to_dict(),
                "clean_loss_corrected": clean_loss,
                "matched_loss_corrected": matched_loss,
                "matched_loss_gap_corrected": matched_loss - clean_loss,
                "matched_disagreement_corrected": disagreement,
                "identity_disagreement_corrected": identity_disagreement,
                "excess_disagreement_corrected": disagreement - identity_disagreement,
            })
        found.add((model, task))
        integrity_rows.append({
            "model": model,
            "task_type": task,
            "episodes": len(metrics),
            "raw_values_finite": True,
            "prediction_loss_recomputed": True,
            "grid_complete": True,
            "maximum_probability_row_sum_error": maximum_row_sum_error,
            "wall_clock_seconds": metadata["wall_clock_seconds"],
            "checkpoint": metadata.get("checkpoint"),
        })

    if found != expected:
        raise AssertionError(f"model/task coverage mismatch: {found}")
    cells = pd.DataFrame(cell_rows)
    summaries: list[dict] = []
    for group_index, ((model, task, rho), group) in enumerate(cells.groupby(["model", "task_type", "rho"], sort=True)):
        if len(group) != 6 or set(group.mechanism) != MECHANISMS:
            raise AssertionError(f"unbalanced cell: {model}/{task}/{rho}")
        loss = mean_interval(group.clean_loss_corrected.to_numpy(), draws=10000, seed=1000 + group_index)
        disagreement = mean_interval(group.excess_disagreement_corrected.to_numpy(), draws=10000, seed=2000 + group_index)
        gap = mean_interval(group.matched_loss_gap_corrected.to_numpy(), draws=10000, seed=3000 + group_index)
        summaries.append({
            "model": model, "task_type": task, "rho": rho, "episodes": len(group),
            "clean_loss_mean": loss[0], "clean_loss_ci_low": loss[1], "clean_loss_ci_high": loss[2],
            "excess_disagreement_mean": disagreement[0],
            "excess_disagreement_ci_low": disagreement[1], "excess_disagreement_ci_high": disagreement[2],
            "matched_loss_gap_mean": gap[0], "matched_loss_gap_ci_low": gap[1], "matched_loss_gap_ci_high": gap[2],
            "identity_disagreement_mean": float(group.identity_disagreement_corrected.mean()),
        })
    summary = pd.DataFrame(summaries)
    integrity = pd.DataFrame(integrity_rows)

    contrasts: list[dict] = []
    pairing = ["task_type", "rho", "mechanism", "episode_seed"]
    wide = cells.pivot(
        index=pairing, columns="model", values="excess_disagreement_corrected"
    )
    for contrast_index, task in enumerate(TASKS):
        task_wide = wide.xs(task, level="task_type")
        for tabicl in ("tabicl_v2_single", "tabicl_v2_default"):
            difference = (task_wide[tabicl] - task_wide["mitra_default"]).to_numpy()
            estimate = mean_interval(difference, draws=10000, seed=4000 + contrast_index)
            contrasts.append({
                "task_type": task,
                "contrast": f"{tabicl}_minus_mitra_default",
                "paired_episodes": len(difference),
                "excess_disagreement_difference": estimate[0],
                "difference_ci_low": estimate[1],
                "difference_ci_high": estimate[2],
                "tabicl_to_mitra_ratio": float(task_wide[tabicl].mean() / task_wide["mitra_default"].mean()),
                "fraction_tabicl_larger": float(np.mean(difference > 0)),
            })
    contrast_frame = pd.DataFrame(contrasts)

    output = ROOT / "results/processed"
    exclusive_csv(cells, output / f"e2_cells_{args.tag}.csv")
    exclusive_csv(summary, output / f"e2_phase_summary_{args.tag}.csv")
    exclusive_csv(integrity, output / f"e2_integrity_{args.tag}.csv")
    exclusive_csv(contrast_frame, output / f"e2_family_contrasts_{args.tag}.csv")

    colors = {"tabicl_v2_single": "#2667ff", "tabicl_v2_default": "#4cc9f0", "mitra_default": "#ef476f"}
    labels = {"tabicl_v2_single": "TabICL single", "tabicl_v2_default": "TabICL default", "mitra_default": "Mitra"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex="col")
    for column, task in enumerate(TASKS):
        for model in MODELS:
            cell = summary[(summary.model == model) & (summary.task_type == task)].sort_values("rho")
            for row, (metric, low, high, ylabel) in enumerate((
                ("clean_loss_mean", "clean_loss_ci_low", "clean_loss_ci_high", "clean log loss" if task == "classification" else "clean MSE"),
                ("excess_disagreement_mean", "excess_disagreement_ci_low", "excess_disagreement_ci_high", "TV minus identity floor" if task == "classification" else "normalized disagreement minus identity floor"),
            )):
                axis = axes[row, column]
                axis.plot(cell.rho, cell[metric], marker="o", color=colors[model], label=labels[model])
                axis.fill_between(cell.rho, cell[low], cell[high], color=colors[model], alpha=0.14)
                axis.set_ylabel(ylabel)
                axis.grid(alpha=0.2)
        axes[0, column].set_title(task.capitalize())
        axes[1, column].axhline(0, color="black", lw=1)
        axes[1, column].set_xlabel("PriorDial coupling rho")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("E2 exploratory current-TFM phase diagram (six mechanisms per rho)")
    fig.tight_layout()
    figure_path = ROOT / "figures" / f"e2_tfm_phase_{args.tag}.png"
    if figure_path.exists():
        raise FileExistsError(figure_path)
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(summary.to_string(index=False))
    print("\nPaired family contrasts:\n" + contrast_frame.to_string(index=False))
    print("\nIntegrity:\n" + integrity.to_string(index=False))


if __name__ == "__main__":
    main()
