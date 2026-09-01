"""Aggregate and audit the frozen PhaseCover preliminary screen."""

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

from run_phasecover import (
    DATASETS,
    HERE,
    IID_SEED,
    MODEL_SEEDS,
    OUTPUT,
    PATCH,
    PHASECOVER,
    PREDICTIONS,
    PROTOCOL_SHA256,
    integrity_check,
)


FIGURES = HERE / "figures"
FIGURES.mkdir(exist_ok=True)


def load_cells() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    iid_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    paths = sorted(OUTPUT.glob("*.json"))
    if len(paths) != 9:
        raise AssertionError(f"expected 9 cells, found {len(paths)}")
    for path in paths:
        payload = json.loads(path.read_text())
        if payload.get("status") != "complete" or payload.get("protocol_sha256") != PROTOCOL_SHA256:
            raise AssertionError(path)
        dataset, seed = payload["dataset"], int(payload["seed"])
        fixed = {
            "dataset": dataset,
            "seed": seed,
            **payload["metadata"],
            "epochs": payload["fit"]["epochs"],
            "fit_wall_seconds": payload["fit"]["wall_seconds"],
            "parameters": payload["fit"]["parameters"],
            "phase_spread_rms": payload["test"]["phase_spread_rms"],
            "phase_materiality": payload["test"]["phase_materiality"],
        }
        for method in ("canonical", "iid4", "phasecover4", "full16"):
            item = payload["test"][method]
            rows.append({
                **fixed,
                "method": method,
                "rmse": item["rmse"],
                "mae": item["mae"],
                "quotient_mse": item["quotient_mse"],
                "rmse_sd_design": item.get("rmse_sd", np.nan),
            })
        for draw, item in enumerate(payload["test"]["iid4"]["subsets"]):
            iid_rows.append({
                "dataset": dataset,
                "seed": seed,
                "draw": draw,
                "phases": "-".join(map(str, item["phases"])),
                "rmse": item["rmse"],
                "mae": item["mae"],
                "quotient_mse": item["quotient_mse"],
            })
        prediction_path = HERE / payload["prediction_file"]
        arrays = np.load(prediction_path)
        predictions, target = arrays["predictions"], arrays["target"]
        if predictions.shape[0] != PATCH or predictions.shape[1:] != target.shape:
            raise AssertionError((path, predictions.shape, target.shape))
        full = predictions.mean(axis=0)
        for phase in range(PATCH):
            phase_rows.append({
                "dataset": dataset,
                "seed": seed,
                "phase": phase,
                "rmse": float(np.sqrt(np.mean(np.square(predictions[phase] - target)))),
                "mae": float(np.mean(np.abs(predictions[phase] - target))),
                "quotient_mse": float(np.mean(np.square(predictions[phase] - full))),
            })
    cells = pd.DataFrame(rows)
    iid = pd.DataFrame(iid_rows)
    phases = pd.DataFrame(phase_rows)
    if cells.duplicated(["dataset", "seed", "method"]).any() or len(cells) != 36:
        raise AssertionError("cell rows incomplete or duplicated")
    if len(iid) != 9 * 64 or len(phases) != 9 * PATCH:
        raise AssertionError("design/phase rows incomplete")
    return cells, iid, phases


def dataset_summary(cells: pd.DataFrame, iid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric = cells.pivot(index=["dataset", "seed"], columns="method", values="rmse").reset_index()
    quotient = cells.pivot(index=["dataset", "seed"], columns="method", values="quotient_mse").reset_index()
    metadata = cells.drop_duplicates(["dataset", "seed"])[
        ["dataset", "seed", "phase_materiality", "phase_spread_rms", "epochs", "fit_wall_seconds"]
    ]
    frame = metadata.merge(metric, on=["dataset", "seed"], validate="one_to_one")
    frame = frame.merge(
        quotient[["dataset", "seed", "iid4", "phasecover4"]].rename(
            columns={"iid4": "iid4_quotient_mse", "phasecover4": "phasecover4_quotient_mse"}
        ),
        on=["dataset", "seed"],
        validate="one_to_one",
    )
    frame["full_gain_vs_canonical"] = frame.canonical - frame.full16
    frame["cover_gain_vs_iid"] = frame.iid4 - frame.phasecover4
    frame["cover_quotient_ratio"] = frame.phasecover4_quotient_mse / frame.iid4_quotient_mse
    cover = cells[cells.method == "phasecover4"][["dataset", "seed", "rmse", "quotient_mse"]]
    ranks = []
    for row in cover.itertuples(index=False):
        draws = iid[(iid.dataset == row.dataset) & (iid.seed == row.seed)]
        ranks.append({
            "dataset": row.dataset,
            "seed": row.seed,
            "cover_better_quotient_than_iid_fraction": float(np.mean(draws.quotient_mse > row.quotient_mse)),
            "cover_better_rmse_than_iid_fraction": float(np.mean(draws.rmse > row.rmse)),
        })
    frame = frame.merge(pd.DataFrame(ranks), on=["dataset", "seed"], validate="one_to_one")
    summary = frame.groupby("dataset", as_index=False).agg(
        seeds=("seed", "nunique"),
        phase_materiality=("phase_materiality", "mean"),
        phase_materiality_sd=("phase_materiality", "std"),
        canonical_rmse=("canonical", "mean"),
        iid4_rmse=("iid4", "mean"),
        phasecover4_rmse=("phasecover4", "mean"),
        full16_rmse=("full16", "mean"),
        full_gain_vs_canonical=("full_gain_vs_canonical", "mean"),
        cover_gain_vs_iid=("cover_gain_vs_iid", "mean"),
        iid4_quotient_mse=("iid4_quotient_mse", "mean"),
        phasecover4_quotient_mse=("phasecover4_quotient_mse", "mean"),
        cover_quotient_ratio=("cover_quotient_ratio", "mean"),
        cover_better_quotient_fraction=("cover_better_quotient_than_iid_fraction", "mean"),
        cover_better_rmse_fraction=("cover_better_rmse_than_iid_fraction", "mean"),
        epochs=("epochs", "mean"),
        fit_wall_seconds=("fit_wall_seconds", "sum"),
    )
    return frame, summary


def exhaustive4_diagnostic() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Post-hoc audit against the exact uniform distribution over four-phase subsets."""
    designs = list(combinations(range(PATCH), 4))
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in MODEL_SEEDS:
            arrays = np.load(PREDICTIONS / f"{dataset}__seed-{seed}.npz")
            predictions, target = arrays["predictions"], arrays["target"]
            full = predictions.mean(axis=0)
            cover = predictions[list(PHASECOVER)].mean(axis=0)
            cover_rmse = float(np.sqrt(np.mean(np.square(cover - target))))
            cover_quotient_mse = float(np.mean(np.square(cover - full)))
            iid_rmse = []
            iid_quotient_mse = []
            for phases in designs:
                prediction = predictions[list(phases)].mean(axis=0)
                iid_rmse.append(float(np.sqrt(np.mean(np.square(prediction - target)))))
                iid_quotient_mse.append(float(np.mean(np.square(prediction - full))))
            iid_rmse_array = np.asarray(iid_rmse)
            iid_quotient_array = np.asarray(iid_quotient_mse)
            rows.append({
                "dataset": dataset,
                "seed": seed,
                "designs": len(designs),
                "exact_iid4_rmse": float(iid_rmse_array.mean()),
                "phasecover4_rmse": cover_rmse,
                "cover_minus_exact_iid_rmse": cover_rmse - float(iid_rmse_array.mean()),
                "exact_iid4_quotient_mse": float(iid_quotient_array.mean()),
                "phasecover4_quotient_mse": cover_quotient_mse,
                "cover_quotient_ratio": cover_quotient_mse / float(iid_quotient_array.mean()),
                "cover_better_quotient_fraction": float(np.mean(iid_quotient_array > cover_quotient_mse)),
                "cover_better_rmse_fraction": float(np.mean(iid_rmse_array > cover_rmse)),
            })
    cells = pd.DataFrame(rows)
    summary = cells.groupby("dataset", as_index=False).agg(
        seeds=("seed", "nunique"),
        designs=("designs", "first"),
        exact_iid4_rmse=("exact_iid4_rmse", "mean"),
        phasecover4_rmse=("phasecover4_rmse", "mean"),
        cover_minus_exact_iid_rmse=("cover_minus_exact_iid_rmse", "mean"),
        exact_iid4_quotient_mse=("exact_iid4_quotient_mse", "mean"),
        phasecover4_quotient_mse=("phasecover4_quotient_mse", "mean"),
        cover_quotient_ratio=("cover_quotient_ratio", "mean"),
        cover_better_quotient_fraction=("cover_better_quotient_fraction", "mean"),
        cover_better_rmse_fraction=("cover_better_rmse_fraction", "mean"),
    )
    return cells, summary


def frozen_gates(summary: pd.DataFrame) -> dict[str, Any]:
    material = summary.phase_materiality >= 0.05
    full_wins = summary.full_gain_vs_canonical > 0
    cover_q_wins = summary.phasecover4_quotient_mse < summary.iid4_quotient_mse
    cover_rmse_wins = summary.cover_gain_vs_iid >= 0
    balanced_full_gain = float(summary.full_gain_vs_canonical.mean())
    balanced_q_ratio = float(summary.cover_quotient_ratio.mean())
    gates = {
        "phase_materiality": {
            "datasets_passing": int(material.sum()),
            "required": 2,
            "passed": bool(material.sum() >= 2),
        },
        "full16_forecast": {
            "dataset_wins": int(full_wins.sum()),
            "dataset_balanced_rmse_gain": balanced_full_gain,
            "passed": bool(full_wins.sum() >= 2 and balanced_full_gain > 0),
        },
        "phasecover_quotient": {
            "dataset_wins": int(cover_q_wins.sum()),
            "dataset_balanced_ratio": balanced_q_ratio,
            "passed": bool(cover_q_wins.sum() >= 2 and balanced_q_ratio <= 0.80),
        },
        "phasecover_forecast": {
            "dataset_wins_or_ties": int(cover_rmse_wins.sum()),
            "dataset_balanced_rmse_gain": float(summary.cover_gain_vs_iid.mean()),
            "passed": bool(cover_rmse_wins.sum() >= 2),
        },
    }
    gates["all_passed"] = all(item["passed"] for item in gates.values())
    return gates


def save_figures(summary: pd.DataFrame, phases: pd.DataFrame, exhaustive_summary: pd.DataFrame) -> None:
    phase_summary = phases.groupby(["dataset", "phase"], as_index=False).agg(
        rmse=("rmse", "mean"), rmse_sd=("rmse", "std")
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True)
    for ax, dataset in zip(axes, DATASETS):
        x = phase_summary[phase_summary.dataset == dataset]
        ax.plot(x.phase, x.rmse, marker="o", color="#315da8")
        ax.fill_between(x.phase, x.rmse - x.rmse_sd, x.rmse + x.rmse_sd, color="#315da8", alpha=.18)
        for phase in PHASECOVER:
            ax.axvline(phase, color="#d97432", alpha=.18, linewidth=2)
        ax.set_title(dataset)
        ax.set_xlabel("patch phase")
        ax.set_ylabel("standardized RMSE")
    fig.suptitle("Forecast error depends on arbitrary patch origin")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_1_phase_sensitivity.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIGURES / "figure_1_phase_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)

    x = np.arange(len(exhaustive_summary))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    axes[0].bar(
        x - .18, exhaustive_summary.exact_iid4_quotient_mse, .36,
        label="exact IID4", color="#8e9aaf",
    )
    axes[0].bar(
        x + .18, exhaustive_summary.phasecover4_quotient_mse, .36,
        label="PhaseCover4", color="#315da8",
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("MSE to full16 quotient (log)")
    forecast_delta = exhaustive_summary.cover_minus_exact_iid_rmse
    axes[1].bar(x, forecast_delta, color=np.where(forecast_delta <= 0, "#315da8", "#d97432"))
    axes[1].axhline(0, color="black", linewidth=.8)
    axes[1].set_ylabel("PhaseCover4 RMSE − exact IID4 RMSE")
    for ax in axes:
        ax.set_xticks(x, exhaustive_summary.dataset)
        ax.legend(frameon=False) if ax is axes[0] else None
    fig.suptitle("Exact four-phase audit over all 1,820 designs")
    fig.tight_layout(rect=(0, 0, 1, .94), w_pad=5.0)
    fig.savefig(FIGURES / "figure_2_phasecover_vs_iid.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIGURES / "figure_2_phasecover_vs_iid.pdf", bbox_inches="tight")
    plt.close(fig)


def write_report(
    summary: pd.DataFrame,
    exhaustive_summary: pd.DataFrame,
    cells_by_seed: pd.DataFrame,
    gates: dict[str, Any],
    audit: dict[str, Any],
) -> None:
    verdict = "MECHANISM PRELIMINARILY SUPPORTED; FORECAST-GAIN CLAIM NOT SUPPORTED"
    lines = [
        "# PHASECOVER PRELIMINARY RESULTS",
        "",
        "## Verdict",
        "",
        f"**{verdict}.** The frozen 64-draw screen passed all gates, but exact post-hoc enumeration",
        "shows that its marginal forecast gate was sampling-sensitive. The result applies only to one compact",
        "phase-augmented patch Transformer and is not a forecasting SOTA claim.",
        "",
        "## Frozen gates",
        "",
        "| Gate | Result | Requirement | Verdict |",
        "|---|---:|---:|---:|",
        f"| phase materiality | {gates['phase_materiality']['datasets_passing']}/3 datasets | >=2 | {'PASS' if gates['phase_materiality']['passed'] else 'FAIL'} |",
        f"| full16 vs canonical | {gates['full16_forecast']['dataset_wins']}/3; mean gain {gates['full16_forecast']['dataset_balanced_rmse_gain']:+.5f} | >=2 and positive | {'PASS' if gates['full16_forecast']['passed'] else 'FAIL'} |",
        f"| PhaseCover quotient vs IID4 | {gates['phasecover_quotient']['dataset_wins']}/3; mean ratio {gates['phasecover_quotient']['dataset_balanced_ratio']:.3f} | >=2 and <=0.80 | {'PASS' if gates['phasecover_quotient']['passed'] else 'FAIL'} |",
        f"| PhaseCover forecast vs IID4 | {gates['phasecover_forecast']['dataset_wins_or_ties']}/3; mean gain {gates['phasecover_forecast']['dataset_balanced_rmse_gain']:+.5f} | >=2 | {'PASS' if gates['phasecover_forecast']['passed'] else 'FAIL'} |",
        "",
        "## Dataset means over three model seeds",
        "",
        "| Dataset | phase materiality | canonical | IID4 | PhaseCover4 | full16 | cover/IID quotient | cover−IID forecast |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.phase_materiality:.1%} | {row.canonical_rmse:.5f} | "
            f"{row.iid4_rmse:.5f} | {row.phasecover4_rmse:.5f} | {row.full16_rmse:.5f} | "
            f"{row.cover_quotient_ratio:.3f} | {-row.cover_gain_vs_iid:+.5f} |"
        )
    lines.extend([
        "",
        "`cover−IID forecast` is PhaseCover4 RMSE minus expected IID4 RMSE, so negative is better.",
        "",
        "## Exhaustive post-hoc robustness check",
        "",
        "This diagnostic enumerates all `C(16,4)=1,820` four-phase subsets. It does not alter the frozen gates.",
        "",
        "| Dataset | exact IID4 | PhaseCover4 | cover−exact IID | quotient ratio | quotient percentile |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in exhaustive_summary.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.exact_iid4_rmse:.5f} | {row.phasecover4_rmse:.5f} | "
            f"{row.cover_minus_exact_iid_rmse:+.5f} | {row.cover_quotient_ratio:.3f} | "
            f"{row.cover_better_quotient_fraction:.1%} |"
        )
    lines.extend([
        "",
        f"The quotient advantage survives on {audit['posthoc_exhaustive4']['quotient_dataset_wins']}/3 datasets "
        f"(dataset-balanced ratio {audit['posthoc_exhaustive4']['dataset_balanced_quotient_ratio']:.3f}). "
        f"The forecast comparison wins only {audit['posthoc_exhaustive4']['forecast_dataset_wins']}/3 and is "
        f"{audit['posthoc_exhaustive4']['dataset_balanced_cover_minus_iid_rmse']:+.5f} RMSE worse on average.",
        "",
        "## Interpretation",
        "",
        "- ETTh1 and Solar exhibit material patch-origin sensitivity; Exchange is a useful near-null control.",
        "- Full phase averaging improves the canonical representation on the frozen dataset-level criterion.",
        "- Four equally spaced phases estimate the all-phase quotient substantially better than expected IID4.",
        "- Exact enumeration rejects a reliable forecast-gain claim. The clean result is efficient quotient",
        "  estimation, not demonstrated universal accuracy improvement.",
        "",
        "## Integrity and compute",
        "",
        f"- Protocol SHA-256: `{PROTOCOL_SHA256}` (matched: {audit['protocol_hash_matches']}).",
        f"- Exact context reconstruction maximum error: {audit['phase_integrity']['maximum_reconstruction_error']:.1f}.",
        f"- Cells: 9/9; method rows: 36; IID design rows: 576; phase rows: 144.",
        f"- Summed fit time: {cells_by_seed.fit_wall_seconds.sum():.1f} seconds; mean epochs: {cells_by_seed.epochs.mean():.1f}.",
        "",
        "## What would falsify the paper direction next",
        "",
        "The next study must use frozen published implementations (at least PatchTST and one pretrained TSFM),",
        "include canonical-trained and phase-augmented training controls, and repeat on untouched datasets.",
        "Kill the direction if materiality or quotient efficiency does not transfer. Do not tune offsets per dataset.",
        "Channel permutation, adaptive patch size, and TabPFN-on-lags are not claimed as new.",
        "",
        "## Readiness",
        "",
        "Novelty potential: **3.5/5**. Empirical readiness: **1.5/5**. Status: **promising preliminary",
        "mechanism, failed forecast-gain robustness check; not yet an ICLR result.**",
    ])
    (HERE / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    cells, iid, phases = load_cells()
    cells_by_seed, summary = dataset_summary(cells, iid)
    exhaustive_cells, exhaustive_summary = exhaustive4_diagnostic()
    gates = frozen_gates(summary)
    phase_integrity = integrity_check()
    digest = hashlib.sha256((HERE / "PROTOCOL.md").read_bytes()).hexdigest()
    audit = {
        "status": "complete",
        "decision": "mechanism_supported_forecast_gain_not_supported",
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_hash_matches": digest == PROTOCOL_SHA256,
        "cells": 9,
        "method_rows": len(cells),
        "iid_design_rows": len(iid),
        "phase_rows": len(phases),
        "phase_integrity": phase_integrity,
        "gates": gates,
        "iid_seed": IID_SEED,
        "phasecover_offsets": list(PHASECOVER),
        "posthoc_exhaustive4": {
            "designs": int(exhaustive_summary.designs.iloc[0]),
            "quotient_dataset_wins": int(
                (exhaustive_summary.phasecover4_quotient_mse < exhaustive_summary.exact_iid4_quotient_mse).sum()
            ),
            "dataset_balanced_quotient_ratio": float(exhaustive_summary.cover_quotient_ratio.mean()),
            "forecast_dataset_wins": int((exhaustive_summary.cover_minus_exact_iid_rmse <= 0).sum()),
            "dataset_balanced_cover_minus_iid_rmse": float(
                exhaustive_summary.cover_minus_exact_iid_rmse.mean()
            ),
        },
    }
    if not audit["protocol_hash_matches"] or not phase_integrity["passed"]:
        raise AssertionError(audit)
    cells.to_csv(HERE / "table_cells.csv", index=False)
    cells_by_seed.to_csv(HERE / "table_cell_comparisons.csv", index=False)
    summary.to_csv(HERE / "table_dataset_summary.csv", index=False)
    iid.to_csv(HERE / "table_iid_designs.csv", index=False)
    phases.to_csv(HERE / "table_phase_metrics.csv", index=False)
    exhaustive_cells.to_csv(HERE / "table_exhaustive4_cells.csv", index=False)
    exhaustive_summary.to_csv(HERE / "table_exhaustive4_summary.csv", index=False)
    save_figures(summary, phases, exhaustive_summary)
    write_report(summary, exhaustive_summary, cells_by_seed, gates, audit)
    (HERE / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
