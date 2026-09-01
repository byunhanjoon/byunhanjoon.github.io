"""Aggregate and audit the frozen PhaseCover confirmation."""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_confirmation import (
    BACKBONES,
    CELLS,
    DATASETS,
    HERE,
    MODEL_SEEDS,
    PATCH,
    PHASECOVER,
    PREDICTIONS,
    PROTOCOL_SHA256,
    TRAIN_MODES,
    integrity_check,
)


FIGURES = HERE / "figures"
FIGURES.mkdir(exist_ok=True)


def load_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    method_rows: list[dict[str, Any]] = []
    design_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    paths = sorted(CELLS.glob("*.json"))
    if len(paths) != 24:
        raise AssertionError(f"expected 24 cells, found {len(paths)}")
    for path in paths:
        payload = json.loads(path.read_text())
        if payload["status"] != "complete" or payload["protocol_sha256"] != PROTOCOL_SHA256:
            raise AssertionError(path)
        fixed = {
            "dataset": payload["dataset"],
            "backbone": payload["backbone"],
            "train_mode": payload["train_mode"],
            "seed": int(payload["seed"]),
            "phase_materiality": payload["test"]["phase_materiality"],
            "phase_spread_rms": payload["test"]["phase_spread_rms"],
            "epochs": payload["fit"]["epochs"],
            "fit_wall_seconds": payload["fit"]["wall_seconds"],
            "parameters_total": payload["fit"]["parameters_total"],
            "parameters_trainable": payload["fit"]["parameters_trainable"],
        }
        for method in ("canonical", "exact_iid4", "phasecover4", "full8"):
            item = payload["test"][method]
            method_rows.append({
                **fixed,
                "method": method,
                "rmse": item["rmse"],
                "mae": item["mae"],
                "quotient_mse": item["quotient_mse"],
            })
        for index, item in enumerate(payload["test"]["exact_iid4"]["designs"]):
            design_rows.append({
                **{key: fixed[key] for key in ("dataset", "backbone", "train_mode", "seed")},
                "design": index,
                "phases": "-".join(map(str, item["phases"])),
                "rmse": item["rmse"],
                "mae": item["mae"],
                "quotient_mse": item["quotient_mse"],
            })
        arrays = np.load(HERE / payload["prediction_file"])
        predictions, target = arrays["predictions"], arrays["target"]
        if predictions.shape != (PATCH, len(target), 24, 8) or not np.isfinite(predictions).all():
            raise AssertionError((path, predictions.shape))
        full = predictions.mean(axis=0)
        for phase in range(PATCH):
            error = predictions[phase] - target
            phase_rows.append({
                **{key: fixed[key] for key in ("dataset", "backbone", "train_mode", "seed")},
                "phase": phase,
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "mae": float(np.mean(np.abs(error))),
                "quotient_mse": float(np.mean(np.square(predictions[phase] - full))),
            })
    methods = pd.DataFrame(method_rows)
    designs = pd.DataFrame(design_rows)
    phases = pd.DataFrame(phase_rows)
    if methods.duplicated(["dataset", "backbone", "train_mode", "seed", "method"]).any():
        raise AssertionError("duplicated method cell")
    if len(methods) != 96 or len(designs) != 24 * 70 or len(phases) != 24 * PATCH:
        raise AssertionError((len(methods), len(designs), len(phases)))
    return methods, designs, phases


def summarize(methods: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = ["dataset", "backbone", "train_mode", "seed"]
    rmse = methods.pivot(index=index, columns="method", values="rmse").reset_index()
    quotient = methods.pivot(index=index, columns="method", values="quotient_mse").reset_index()
    fixed = methods.drop_duplicates(index)[
        index + [
            "phase_materiality", "phase_spread_rms", "epochs", "fit_wall_seconds",
            "parameters_total", "parameters_trainable",
        ]
    ]
    cells = fixed.merge(rmse, on=index, validate="one_to_one")
    cells = cells.merge(
        quotient[index + ["exact_iid4", "phasecover4"]].rename(columns={
            "exact_iid4": "exact_iid4_quotient_mse",
            "phasecover4": "phasecover4_quotient_mse",
        }),
        on=index,
        validate="one_to_one",
    )
    cells["cover_minus_iid_rmse"] = cells.phasecover4 - cells.exact_iid4
    cells["full_gain_vs_canonical"] = cells.canonical - cells.full8
    cells["cover_quotient_ratio"] = cells.phasecover4_quotient_mse / cells.exact_iid4_quotient_mse
    summary = cells.groupby(["dataset", "backbone", "train_mode"], as_index=False).agg(
        seeds=("seed", "nunique"),
        phase_materiality=("phase_materiality", "mean"),
        phase_materiality_sd=("phase_materiality", "std"),
        canonical_rmse=("canonical", "mean"),
        exact_iid4_rmse=("exact_iid4", "mean"),
        phasecover4_rmse=("phasecover4", "mean"),
        full8_rmse=("full8", "mean"),
        cover_minus_iid_rmse=("cover_minus_iid_rmse", "mean"),
        full_gain_vs_canonical=("full_gain_vs_canonical", "mean"),
        exact_iid4_quotient_mse=("exact_iid4_quotient_mse", "mean"),
        phasecover4_quotient_mse=("phasecover4_quotient_mse", "mean"),
        cover_quotient_ratio=("cover_quotient_ratio", "mean"),
        epochs=("epochs", "mean"),
        fit_wall_seconds=("fit_wall_seconds", "sum"),
        parameters_total=("parameters_total", "first"),
        parameters_trainable=("parameters_trainable", "first"),
    )
    return cells, summary


def decisions(summary: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {
        "phase_sensitivity": {},
        "phase_augmentation": {},
        "quotient_efficiency": {},
        "forecast_improvement": {},
    }
    for backbone in BACKBONES:
        canonical = summary[(summary.backbone == backbone) & (summary.train_mode == "canonical_train")]
        augmented = summary[(summary.backbone == backbone) & (summary.train_mode == "phase_aug_train")]
        material_datasets = int((canonical.phase_materiality >= 0.05).sum())
        result["phase_sensitivity"][backbone] = {
            "datasets_passing": material_datasets,
            "passed": material_datasets >= 2,
        }
        material_reduction = 1.0 - float(augmented.phase_materiality.mean() / canonical.phase_materiality.mean())
        canonical_rmse_change = float(augmented.canonical_rmse.mean() / canonical.canonical_rmse.mean() - 1.0)
        result["phase_augmentation"][backbone] = {
            "relative_materiality_reduction": material_reduction,
            "relative_canonical_rmse_change": canonical_rmse_change,
            "passed": material_reduction >= 0.15 and canonical_rmse_change <= 0.02,
        }
        for train_mode in TRAIN_MODES:
            group = summary[(summary.backbone == backbone) & (summary.train_mode == train_mode)]
            label = f"{backbone}/{train_mode}"
            quotient_wins = int((group.phasecover4_quotient_mse < group.exact_iid4_quotient_mse).sum())
            quotient_ratio = float(group.cover_quotient_ratio.mean())
            result["quotient_efficiency"][label] = {
                "dataset_wins": quotient_wins,
                "dataset_balanced_ratio": quotient_ratio,
                "passed": quotient_wins >= 2 and quotient_ratio <= 0.80,
            }
            forecast_wins = int((group.cover_minus_iid_rmse <= 0).sum())
            result["forecast_improvement"][label] = {
                "dataset_wins": forecast_wins,
                "dataset_balanced_cover_minus_iid_rmse": float(group.cover_minus_iid_rmse.mean()),
                "passed": forecast_wins >= 2,
            }
    result["phase_sensitivity"]["overall_passed"] = all(
        result["phase_sensitivity"][backbone]["passed"] for backbone in BACKBONES
    )
    result["phase_augmentation"]["overall_passed"] = all(
        result["phase_augmentation"][backbone]["passed"] for backbone in BACKBONES
    )
    quotient_passes = sum(item["passed"] for item in result["quotient_efficiency"].values())
    forecast_passes = sum(item["passed"] for item in result["forecast_improvement"].values())
    result["quotient_efficiency"]["groups_passing"] = quotient_passes
    result["quotient_efficiency"]["overall_passed"] = quotient_passes >= 3
    result["forecast_improvement"]["groups_passing"] = forecast_passes
    result["forecast_improvement"]["overall_passed"] = forecast_passes >= 3
    return result


def posthoc_anchor_coset_control() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Audit forecast gains against a phase-0-matched baseline and the other cover coset."""
    all_designs = list(combinations(range(PATCH), 4))
    anchored_designs = [phases for phases in all_designs if 0 in phases]
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for backbone in BACKBONES:
            for train_mode in TRAIN_MODES:
                for seed in MODEL_SEEDS:
                    path = PREDICTIONS / f"{dataset}__{backbone}__{train_mode}__seed-{seed}.npz"
                    arrays = np.load(path)
                    predictions, target = arrays["predictions"], arrays["target"]
                    full = predictions.mean(axis=0)

                    def rmse(prediction: np.ndarray) -> float:
                        return float(np.sqrt(np.mean(np.square(prediction - target))))

                    cover = predictions[[0, 2, 4, 6]].mean(axis=0)
                    complement = predictions[[1, 3, 5, 7]].mean(axis=0)
                    exact_iid_rmse = float(np.mean([
                        rmse(predictions[list(phases)].mean(axis=0)) for phases in all_designs
                    ]))
                    anchored_iid_rmse = float(np.mean([
                        rmse(predictions[list(phases)].mean(axis=0)) for phases in anchored_designs
                    ]))
                    rows.append({
                        "dataset": dataset,
                        "backbone": backbone,
                        "train_mode": train_mode,
                        "seed": seed,
                        "anchored_designs": len(anchored_designs),
                        "cover_rmse": rmse(cover),
                        "complement_rmse": rmse(complement),
                        "exact_iid4_rmse": exact_iid_rmse,
                        "anchored_iid4_rmse": anchored_iid_rmse,
                        "cover_minus_anchored_iid_rmse": rmse(cover) - anchored_iid_rmse,
                        "complement_minus_exact_iid_rmse": rmse(complement) - exact_iid_rmse,
                        "complement_minus_cover_rmse": rmse(complement) - rmse(cover),
                        "cover_quotient_mse": float(np.mean(np.square(cover - full))),
                        "complement_quotient_mse": float(np.mean(np.square(complement - full))),
                    })
    cells = pd.DataFrame(rows)
    summary = cells.groupby(["dataset", "backbone", "train_mode"], as_index=False).agg(
        seeds=("seed", "nunique"),
        anchored_designs=("anchored_designs", "first"),
        cover_rmse=("cover_rmse", "mean"),
        complement_rmse=("complement_rmse", "mean"),
        exact_iid4_rmse=("exact_iid4_rmse", "mean"),
        anchored_iid4_rmse=("anchored_iid4_rmse", "mean"),
        cover_minus_anchored_iid_rmse=("cover_minus_anchored_iid_rmse", "mean"),
        complement_minus_exact_iid_rmse=("complement_minus_exact_iid_rmse", "mean"),
        complement_minus_cover_rmse=("complement_minus_cover_rmse", "mean"),
        cover_quotient_mse=("cover_quotient_mse", "mean"),
        complement_quotient_mse=("complement_quotient_mse", "mean"),
    )
    audit: dict[str, Any] = {"groups": {}}
    anchor_group_passes = 0
    complement_group_passes = 0
    for backbone in BACKBONES:
        for train_mode in TRAIN_MODES:
            group = summary[(summary.backbone == backbone) & (summary.train_mode == train_mode)]
            label = f"{backbone}/{train_mode}"
            anchor_wins = int((group.cover_minus_anchored_iid_rmse <= 0).sum())
            complement_wins = int((group.complement_minus_exact_iid_rmse <= 0).sum())
            anchor_pass = anchor_wins >= 2
            complement_pass = complement_wins >= 2
            anchor_group_passes += int(anchor_pass)
            complement_group_passes += int(complement_pass)
            audit["groups"][label] = {
                "anchored_dataset_wins": anchor_wins,
                "anchored_mean_delta": float(group.cover_minus_anchored_iid_rmse.mean()),
                "anchored_passed": anchor_pass,
                "complement_dataset_wins": complement_wins,
                "complement_mean_delta": float(group.complement_minus_exact_iid_rmse.mean()),
                "complement_passed": complement_pass,
                "mean_coset_gap": float(group.complement_minus_cover_rmse.mean()),
            }
    audit["anchored_groups_passing"] = anchor_group_passes
    audit["complement_groups_passing"] = complement_group_passes
    audit["forecast_robustness_supported"] = anchor_group_passes >= 3 and complement_group_passes >= 3
    audit["maximum_coset_quotient_mse_difference"] = float(
        np.max(np.abs(cells.cover_quotient_mse - cells.complement_quotient_mse))
    )
    return cells, summary, audit


def save_figures(summary: pd.DataFrame, phases: pd.DataFrame) -> None:
    ordered_groups = [(backbone, mode) for backbone in BACKBONES for mode in TRAIN_MODES]
    labels = [f"{backbone}\n{'canonical' if mode == 'canonical_train' else 'phase aug'}" for backbone, mode in ordered_groups]
    datasets = list(DATASETS)
    materiality = np.zeros((len(datasets), len(ordered_groups)))
    quotient = np.zeros_like(materiality)
    forecast = np.zeros_like(materiality)
    for row_index, dataset in enumerate(datasets):
        for column_index, (backbone, mode) in enumerate(ordered_groups):
            row = summary[
                (summary.dataset == dataset) & (summary.backbone == backbone) & (summary.train_mode == mode)
            ].iloc[0]
            materiality[row_index, column_index] = row.phase_materiality
            quotient[row_index, column_index] = row.cover_quotient_ratio
            forecast[row_index, column_index] = row.cover_minus_iid_rmse
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    image = axes[0].imshow(materiality * 100, cmap="YlOrRd", aspect="auto", vmin=0)
    axes[0].set_title("Patch-origin materiality (%)")
    axes[0].set_xticks(range(len(labels)), labels)
    axes[0].set_yticks(range(len(datasets)), datasets)
    for i in range(len(datasets)):
        for j in range(len(labels)):
            axes[0].text(j, i, f"{materiality[i,j]*100:.1f}", ha="center", va="center")
    fig.colorbar(image, ax=axes[0], fraction=.046)
    x = np.arange(len(datasets))
    width = .19
    for j, label in enumerate(labels):
        axes[1].bar(x + (j - 1.5) * width, quotient[:, j], width, label=label.replace("\n", " "))
    axes[1].axhline(.8, color="black", linestyle="--", linewidth=1, label="frozen threshold")
    axes[1].set_xticks(x, datasets)
    axes[1].set_ylabel("PhaseCover / exact IID4 quotient MSE")
    axes[1].set_title("Deterministic quotient efficiency")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(w_pad=3)
    fig.savefig(FIGURES / "figure_1_mechanism_confirmation.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIGURES / "figure_1_mechanism_confirmation.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for j, label in enumerate(labels):
        offsets = x + (j - 1.5) * width
        colors = np.where(forecast[:, j] <= 0, "#315da8", "#d97432")
        ax.bar(offsets, forecast[:, j], width, label=label.replace("\n", " "), color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x, datasets)
    ax.set_ylabel("PhaseCover4 RMSE − exact IID4 RMSE")
    ax.set_title("Forecast comparison: negative favors PhaseCover")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_2_forecast_comparison.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIGURES / "figure_2_forecast_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def save_posthoc_figure(posthoc_summary: pd.DataFrame) -> None:
    ordered_groups = [(backbone, mode) for backbone in BACKBONES for mode in TRAIN_MODES]
    labels = [f"{backbone} {'canonical' if mode == 'canonical_train' else 'phase aug'}" for backbone, mode in ordered_groups]
    datasets = list(DATASETS)
    anchor = np.zeros((len(datasets), len(ordered_groups)))
    complement = np.zeros_like(anchor)
    for i, dataset in enumerate(datasets):
        for j, (backbone, mode) in enumerate(ordered_groups):
            row = posthoc_summary[
                (posthoc_summary.dataset == dataset)
                & (posthoc_summary.backbone == backbone)
                & (posthoc_summary.train_mode == mode)
            ].iloc[0]
            anchor[i, j] = row.cover_minus_anchored_iid_rmse
            complement[i, j] = row.complement_minus_exact_iid_rmse
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    x = np.arange(len(datasets))
    width = .19
    for j, label in enumerate(labels):
        axes[0].bar(x + (j - 1.5) * width, anchor[:, j], width, label=label)
        axes[1].bar(x + (j - 1.5) * width, complement[:, j], width, label=label)
    for ax in axes:
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(x, datasets)
    axes[0].set_title("{0,2,4,6} vs IID4 containing phase 0")
    axes[1].set_title("{1,3,5,7} vs unconditioned IID4")
    axes[0].set_ylabel("cover RMSE − comparator RMSE")
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Post-hoc anchor and complementary-coset controls")
    fig.tight_layout(rect=(0, 0, 1, .94), w_pad=3)
    fig.savefig(FIGURES / "figure_3_anchor_coset_controls.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIGURES / "figure_3_anchor_coset_controls.pdf", bbox_inches="tight")
    plt.close(fig)

def write_report(
    summary: pd.DataFrame,
    posthoc_summary: pd.DataFrame,
    decision: dict[str, Any],
    audit: dict[str, Any],
) -> None:
    phase_pass = decision["phase_sensitivity"]["overall_passed"]
    aug_pass = decision["phase_augmentation"]["overall_passed"]
    quotient_pass = decision["quotient_efficiency"]["overall_passed"]
    forecast_pass = decision["forecast_improvement"]["overall_passed"]
    lines = [
        "# PHASECOVER PUBLISHED-BACKBONE CONFIRMATION RESULTS",
        "",
        "## Decision",
        "",
        f"- Phase sensitivity transfers across backbones: **{'PASS' if phase_pass else 'FAIL'}**.",
        f"- Random-phase training is a remedy: **{'PASS' if aug_pass else 'FAIL'}**.",
        f"- PhaseCover is an efficient quotient design: **{'PASS' if quotient_pass else 'FAIL'}**.",
        f"- PhaseCover improves forecasting over exact IID4: **{'PASS' if forecast_pass else 'FAIL'}**.",
        f"- Forecast advantage survives anchor/coset controls: **{'PASS' if audit['posthoc_anchor_coset']['forecast_robustness_supported'] else 'FAIL'}**.",
        "",
        "## Dataset means over two seeds",
        "",
        "| Dataset | Backbone | Training | materiality | canonical | exact IID4 | PhaseCover4 | full8 | quotient ratio | cover−IID |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.sort_values(["dataset", "backbone", "train_mode"]).itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.backbone} | {row.train_mode} | {row.phase_materiality:.1%} | "
            f"{row.canonical_rmse:.5f} | {row.exact_iid4_rmse:.5f} | {row.phasecover4_rmse:.5f} | "
            f"{row.full8_rmse:.5f} | {row.cover_quotient_ratio:.3f} | {row.cover_minus_iid_rmse:+.5f} |"
        )
    lines.extend([
        "",
        "`cover−IID` is PhaseCover4 RMSE minus the exact mean over all 70 IID4 designs; negative is better.",
        "",
        "## Frozen-gate details",
        "",
        "### Phase sensitivity",
        "",
    ])
    for backbone in BACKBONES:
        item = decision["phase_sensitivity"][backbone]
        lines.append(f"- {backbone}: {item['datasets_passing']}/3 material datasets — {'PASS' if item['passed'] else 'FAIL'}.")
    lines.extend(["", "### Phase augmentation", ""])
    for backbone in BACKBONES:
        item = decision["phase_augmentation"][backbone]
        lines.append(
            f"- {backbone}: materiality reduction {item['relative_materiality_reduction']:.1%}; "
            f"canonical RMSE change {item['relative_canonical_rmse_change']:+.1%} — "
            f"{'PASS' if item['passed'] else 'FAIL'}."
        )
    lines.extend(["", "### Quotient and forecast groups", ""])
    for label, item in decision["quotient_efficiency"].items():
        if not isinstance(item, dict):
            continue
        forecast_item = decision["forecast_improvement"][label]
        lines.append(
            f"- {label}: quotient {item['dataset_wins']}/3, ratio {item['dataset_balanced_ratio']:.3f} "
            f"({'PASS' if item['passed'] else 'FAIL'}); forecast {forecast_item['dataset_wins']}/3, "
            f"mean delta {forecast_item['dataset_balanced_cover_minus_iid_rmse']:+.5f} "
            f"({'PASS' if forecast_item['passed'] else 'FAIL'})."
        )
    lines.extend([
        "",
        "## Post-hoc anchor and complementary-coset audit",
        "",
        "The frozen IID4 comparison includes phase 0 in only half of its designs, while PhaseCover `{0,2,4,6}`",
        "always includes it. The table therefore adds an exact 35-design IID4 comparator conditioned on phase 0",
        "and evaluates the equally spaced complementary coset `{1,3,5,7}`.",
        "",
        "| Dataset | Backbone | Training | cover−anchored IID | complement−IID | complement−cover |",
        "|---|---|---|---:|---:|---:|",
    ])
    for row in posthoc_summary.sort_values(["dataset", "backbone", "train_mode"]).itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.backbone} | {row.train_mode} | "
            f"{row.cover_minus_anchored_iid_rmse:+.5f} | {row.complement_minus_exact_iid_rmse:+.5f} | "
            f"{row.complement_minus_cover_rmse:+.5f} |"
        )
    posthoc = audit["posthoc_anchor_coset"]
    lines.extend([
        "",
        f"Only {posthoc['anchored_groups_passing']}/4 groups pass against phase-0-matched IID4; "
        f"{posthoc['complement_groups_passing']}/4 pass for the complementary coset. The two cosets have identical",
        f"quotient MSE up to {posthoc['maximum_coset_quotient_mse_difference']:.2e}, but sharply different forecast",
        "accuracy. Thus spacing explains quotient efficiency, while the apparent accuracy win depends on the",
        "arbitrary choice of the phase-0-anchored coset.",
    ])
    lines.extend([
        "",
        "## Integrity and scope",
        "",
        f"- Protocol SHA-256 `{PROTOCOL_SHA256}` matched: {audit['protocol_hash_matches']}.",
        f"- Exact reconstruction maximum error: {audit['phase_integrity']['maximum_reconstruction_error']:.1f}.",
        "- Complete cells 24/24; method rows 96; exact design rows 1,680; phase rows 192.",
        f"- Summed fit time: {summary.fit_wall_seconds.sum():.1f} seconds.",
        "- Published implementations were used, but the experiment remains an eight-channel, mean-boundary-fill screen.",
        "- MOMENT is a frozen-encoder linear probe; this is not a full TSFM fine-tuning comparison.",
        "- Jena's finite `-9999` missing-value sentinel required the documented data-cleaning deviation in",
        "  `PROTOCOL_DEVIATION.md`; the original invalid artifacts were retained in quarantine.",
        "",
        "## Paper decision",
        "",
    ])
    forecast_robust = audit["posthoc_anchor_coset"]["forecast_robustness_supported"]
    if quotient_pass and (not forecast_pass or not forecast_robust):
        lines.extend([
            "The representation/quotient phenomenon transfers, but deterministic phase coverage still lacks a reliable",
            "forecasting advantage. Continue only if the paper is reframed around invariance measurement or certified",
            "quotient approximation; do not sell PhaseCover as an accuracy method.",
        ])
    elif quotient_pass and forecast_pass:
        lines.extend([
            "Both quotient efficiency and forecast utility transfer. Proceed to a broader benchmark with confidence",
            "intervals, longer horizons, patch sizes, and full TSFM fine-tuning.",
        ])
    else:
        lines.extend([
            "The core quotient claim did not transfer under the frozen published-backbone test. Kill or fundamentally",
            "reformulate PhaseCover before allocating an ICLR-scale benchmark.",
        ])
    (HERE / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    methods, designs, phases = load_results()
    cells, summary = summarize(methods)
    decision = decisions(summary)
    posthoc_cells, posthoc_summary, posthoc_audit = posthoc_anchor_coset_control()
    phase_integrity = integrity_check()
    protocol_digest = hashlib.sha256((HERE / "PROTOCOL.md").read_bytes()).hexdigest()
    audit = {
        "status": "complete",
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_hash_matches": protocol_digest == PROTOCOL_SHA256,
        "phase_integrity": phase_integrity,
        "cells": 24,
        "method_rows": len(methods),
        "design_rows": len(designs),
        "phase_rows": len(phases),
        "decision": decision,
        "posthoc_anchor_coset": posthoc_audit,
    }
    if not audit["protocol_hash_matches"] or not phase_integrity["passed"]:
        raise AssertionError(audit)
    methods.to_csv(HERE / "table_methods.csv", index=False)
    designs.to_csv(HERE / "table_exact_designs.csv", index=False)
    phases.to_csv(HERE / "table_phases.csv", index=False)
    cells.to_csv(HERE / "table_cells.csv", index=False)
    summary.to_csv(HERE / "table_dataset_summary.csv", index=False)
    posthoc_cells.to_csv(HERE / "table_posthoc_anchor_coset_cells.csv", index=False)
    posthoc_summary.to_csv(HERE / "table_posthoc_anchor_coset_summary.csv", index=False)
    save_figures(summary, phases)
    save_posthoc_figure(posthoc_summary)
    write_report(summary, posthoc_summary, decision, audit)
    (HERE / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
