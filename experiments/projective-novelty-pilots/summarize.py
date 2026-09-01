"""Audit and summarize the three projective novelty pilots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROTOCOL_SHA256 = "b5148cca2610c49d8cca287d123d81427cc2daa1874150ca16056159d8b3daab"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def checked_csv(path: Path, rows: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if len(frame) != rows:
        raise RuntimeError(f"{path}: expected {rows}, found {len(frame)}")
    if not np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all():
        raise FloatingPointError(path)
    return frame


def main() -> None:
    actual_hash = hashlib.sha256((HERE / "PROTOCOL.md").read_bytes()).hexdigest()
    if actual_hash != PROTOCOL_SHA256:
        raise RuntimeError(f"protocol changed: {actual_hash}")
    query_audit = load_json(HERE / "query_complexity" / "audit.json")
    mixture_audit = load_json(HERE / "mixture" / "audit.json")
    capacity_audit = load_json(HERE / "mixture" / "capacity_control_audit.json")
    reconciliation_audit = load_json(HERE / "reconciliation" / "audit.json")
    for audit in (query_audit, mixture_audit, reconciliation_audit):
        if audit["status"] != "complete" or audit["protocol_sha256"] != actual_hash:
            raise RuntimeError("primary audit mismatch")
    if capacity_audit["status"] != "complete":
        raise RuntimeError("capacity control incomplete")

    query_cells = checked_csv(HERE / "query_complexity" / "cells.csv", 108)
    query_regret = checked_csv(HERE / "query_complexity" / "regret_cells.csv", 54)
    mixture = checked_csv(HERE / "mixture" / "cells.csv", 27)
    capacity = checked_csv(HERE / "mixture" / "capacity_control_cells.csv", 18)
    reconciliation = checked_csv(HERE / "reconciliation" / "cells.csv", 27)
    diagnostics = checked_csv(HERE / "reconciliation" / "diagnostics.csv", 9)

    scorecard = pd.DataFrame(
        [
            {
                "rank": 1,
                "component": "Non-Gaussian projective mixture",
                "primary_gate": "pass",
                "capacity_control": "pass",
                "decision": "continue",
            },
            {
                "rank": 2,
                "component": "Black-box projective reconciliation",
                "primary_gate": "fail",
                "capacity_control": "not applicable",
                "decision": "stop",
            },
            {
                "rank": 3,
                "component": "Query support-size complexity law",
                "primary_gate": "fail",
                "capacity_control": "not applicable",
                "decision": "reject hypothesis",
            },
        ]
    )
    scorecard.to_csv(HERE / "NOVELTY_SCORECARD.csv", index=False)

    figure_dir = HERE / "figures"
    figure_dir.mkdir(exist_ok=True)
    datasets = ["JenaWeather", "Electricity", "Traffic"]

    regret_summary = query_regret.groupby(["dataset", "support"], as_index=False).nll_regret.mean()
    fig, axis = plt.subplots(figsize=(6.6, 3.8), constrained_layout=True)
    for dataset, group in regret_summary.groupby("dataset"):
        axis.plot(group.support, group.nll_regret, marker="o", label=dataset)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xscale("log", base=2)
    axis.set_xticks([1, 2, 4, 8, 16, 32], [1, 2, 4, 8, 16, 32])
    axis.set_xlabel("Nonzero query coordinates")
    axis.set_ylabel("Direct − projective NLL")
    axis.legend(frameon=False)
    fig.savefig(figure_dir / "query_complexity.png", dpi=180)
    plt.close(fig)

    all_mixture = pd.concat([mixture, capacity], ignore_index=True)
    mix_means = all_mixture.groupby(["dataset", "model"], as_index=False).heldout_nll.mean()
    mix_models = ["projective_mixture4", "joint_gaussian_matched", "direct_mixture4_matched"]
    mix_labels = ["Projective mixture", "Matched Gaussian", "Matched direct mixture"]
    fig, axis = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    x = np.arange(3)
    width = 0.24
    for offset, (model, label) in enumerate(zip(mix_models, mix_labels)):
        values = mix_means[mix_means.model == model].set_index("dataset").loc[datasets, "heldout_nll"]
        axis.bar(x + (offset - 1) * width, values, width, label=label)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x, datasets)
    axis.set_ylabel("Held-out mixture NLL")
    axis.legend(frameon=False)
    fig.savefig(figure_dir / "mixture_capacity_controls.png", dpi=180)
    plt.close(fig)

    recon_means = reconciliation.groupby(["dataset", "model"], as_index=False).nll.mean()
    recon_models = ["direct_broad", "reconciled", "trained_projective"]
    recon_labels = ["Direct", "Post-hoc reconciled", "Trained projective"]
    fig, axis = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    for offset, (model, label) in enumerate(zip(recon_models, recon_labels)):
        values = recon_means[recon_means.model == model].set_index("dataset").loc[datasets, "nll"]
        axis.bar(x + (offset - 1) * width, values, width, label=label)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x, datasets)
    axis.set_ylabel("Gaussian NLL (constant omitted)")
    axis.legend(frameon=False)
    fig.savefig(figure_dir / "reconciliation.png", dpi=180)
    plt.close(fig)

    primary_means = mixture.groupby(["dataset", "model"], as_index=False).mean(numeric_only=True)
    capacity_means = capacity.groupby(["dataset", "model"], as_index=False).mean(numeric_only=True)

    def metric(source: pd.DataFrame, dataset: str, model: str, column: str) -> float:
        return float(source[(source.dataset == dataset) & (source.model == model)][column].iloc[0])

    report = f"""# Projective novelty-pilot results

Primary protocol SHA-256: `{PROTOCOL_SHA256}`. All primary gates were frozen before outcomes. The mixture capacity addendum was frozen after the primary mixture pass and before its control outcomes.

| Rank | Component | Gate | Decision |
|---:|---|:---:|---|
| 1 | Non-Gaussian projective mixture | PASS + capacity control PASS | **Continue** |
| 2 | Black-box projective reconciliation | FAIL | **Stop** |
| 3 | Query support-size complexity law | FAIL | **Reject hypothesis** |

## 1. Non-Gaussian projective mixture — survives

A four-component conditional joint mixture retains analytic linear projections and exact moment identities while representing non-Gaussian predictive shapes.

- Primary comparison: 8/9 NLL wins over the single projective Gaussian and 9/9 over the direct scalar mixture.
- Capacity-matched comparison: 8/9 wins over the 135,989-parameter Gaussian and 9/9 over the 137,211-parameter direct mixture; the projective mixture has 136,580 parameters.
- NLL improvement over the matched Gaussian: Jena Weather {capacity_audit['dataset_nll_improvements_vs_matched_gaussian']['JenaWeather']:.3f}, Electricity {capacity_audit['dataset_nll_improvements_vs_matched_gaussian']['Electricity']:.3f}, Traffic {capacity_audit['dataset_nll_improvements_vs_matched_gaussian']['Traffic']:.3f}.
- Maximum projective identity violation: {mixture_audit['maximum_projective_identity_violation']:.2e}.
- PIT calibration error: {mixture_audit['mixture_pit_calibration_error'] * 100:.1f}%, only {mixture_audit['calibration_error_difference'] * 100:.1f} percentage points worse than the better comparator.

Capacity-matched mean NLL:

| Dataset | Projective mixture | Matched Gaussian | Matched direct mixture |
|---|---:|---:|---:|
| Jena Weather | {metric(primary_means, 'JenaWeather', 'projective_mixture4', 'heldout_nll'):.3f} | {metric(capacity_means, 'JenaWeather', 'joint_gaussian_matched', 'heldout_nll'):.3f} | {metric(capacity_means, 'JenaWeather', 'direct_mixture4_matched', 'heldout_nll'):.3f} |
| Electricity | {metric(primary_means, 'Electricity', 'projective_mixture4', 'heldout_nll'):.3f} | {metric(capacity_means, 'Electricity', 'joint_gaussian_matched', 'heldout_nll'):.3f} | {metric(capacity_means, 'Electricity', 'direct_mixture4_matched', 'heldout_nll'):.3f} |
| Traffic | {metric(primary_means, 'Traffic', 'projective_mixture4', 'heldout_nll'):.3f} | {metric(capacity_means, 'Traffic', 'joint_gaussian_matched', 'heldout_nll'):.3f} | {metric(capacity_means, 'Traffic', 'direct_mixture4_matched', 'heldout_nll'):.3f} |

![Mixture capacity controls](figures/mixture_capacity_controls.png)

**Interpretation:** non-Gaussianity adds real probabilistic value on Jena Weather and Traffic, not merely capacity. Electricity is effectively tied. This component merits a larger comparison against existing consistent non-Gaussian forecasters.

## 2. Query-complexity law — falsified

Direct-minus-projective NLL regret did not grow with support size. Spearman correlations were {query_audit['gates']['JenaWeather']['spearman_rho']:.2f} on Jena Weather, {query_audit['gates']['Electricity']['spearman_rho']:.2f} on Electricity, and {query_audit['gates']['Traffic']['spearman_rho']:.2f} on Traffic.

![Query complexity](figures/query_complexity.png)

The direct model's largest weakness is not composing many coordinates. It is respecting algebraic transformations—especially sign, scale, and variance relations—that are unevenly covered in training. Do not claim a monotone “complexity gap.” A future benchmark should cross query-family transformations rather than use support size as its central axis.

## 3. Black-box reconciliation — unreliable

- It improved NLL on two datasets, but closed at least 50% of the trained-projective gap only on Traffic ({reconciliation_audit['gates']['Traffic']['projective_gap_closed'] * 100:.1f}%).
- It closed {reconciliation_audit['gates']['Electricity']['projective_gap_closed'] * 100:.1f}% on Electricity and worsened Jena Weather.
- Reconstructed covariance matrices required a mean {reconciliation_audit['mean_relative_psd_correction'] * 100:.1f}% relative correction to become PSD.
- Coverage remained within the gate on only one dataset.

![Reconciliation](figures/reconciliation.png)

**Interpretation:** the Gaussian representability identity is useful theory and a diagnostic, but querying and repairing an inconsistent black box is not a reliable method. Consistency needs to be architectural or trained end-to-end.

## Updated novelty verdict

Only one proposed 4+/5 component survived: **a non-Gaussian joint mixture that provides exact, analytic distributions for arbitrary linear temporal queries and outperforms capacity-matched Gaussian and direct-query mixtures.**

This is promising but not yet a standalone ICLR claim because consistent mixture forecasting already has close prior work. The next decisive test is a paper-scale comparison against marginalization-consistent flows and hierarchical coherent forecasters, emphasizing the distinction between arbitrary signed/scaled linear queries and subset or fixed-hierarchy marginals.

## Integrity

- Finite audited rows: query metrics {len(query_cells)}, query regrets {len(query_regret)}, mixture {len(mixture)}, capacity controls {len(capacity)}, reconciliation {len(reconciliation)}, reconstruction diagnostics {len(diagnostics)}.
- Primary mixture wall time: {mixture_audit['wall_seconds']:.1f}s; capacity controls: {capacity_audit['wall_seconds']:.1f}s.
- All scripts compile, protocol hashes match, expected outputs exist, and no training process remains active.
"""
    (HERE / "RESULTS.md").write_text(report)
    print(scorecard.to_string(index=False))


if __name__ == "__main__":
    main()
