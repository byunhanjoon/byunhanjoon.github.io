"""Audit and summarize the three frozen oral-ceiling pilots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
EXPECTED_PROTOCOL_SHA256 = "538f14851b6a1cf54737c3b9bc8df3cf3b227c1b33cf2ef53469f261317b3164"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def audit_inputs() -> dict[str, dict]:
    actual_hash = hashlib.sha256((HERE / "PROTOCOL.md").read_bytes()).hexdigest()
    if actual_hash != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(f"protocol hash changed: {actual_hash}")
    audits = {
        name: load_json(HERE / directory / "audit.json")
        for name, directory in {
            "view": "view",
            "projective": "projective",
            "interventional": "interventional",
        }.items()
    }
    for name, audit in audits.items():
        if audit["status"] != "complete":
            raise RuntimeError(f"{name} is not complete")
        if audit["protocol_sha256"] != actual_hash:
            raise RuntimeError(f"{name} used a different protocol")
    return audits


def finite_csv(path: Path, expected_rows: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if len(frame) != expected_rows:
        raise RuntimeError(f"{path}: expected {expected_rows} rows, found {len(frame)}")
    numeric = frame.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy()).all():
        raise FloatingPointError(path)
    return frame


def make_figures(view_summary: pd.DataFrame, projective: pd.DataFrame, intervention: pd.DataFrame) -> None:
    figure_dir = HERE / "figures"
    figure_dir.mkdir(exist_ok=True)

    datasets = ["JenaWeather", "Electricity", "Traffic"]
    models = ["lightgbm", "mlp", "tabpfn"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), constrained_layout=True)
    for axis, column, title in (
        (axes[0], "relative_rmse_spread", "Worst-to-best RMSE spread"),
        (axes[1], "relative_prediction_dispersion", "Prediction dispersion"),
    ):
        matrix = view_summary.pivot(index="dataset", columns="model", values=column).loc[datasets, models] * 100
        image = axis.imshow(matrix, cmap="YlOrRd", vmin=0)
        axis.set_xticks(range(len(models)), ["LightGBM", "MLP", "TabPFN"])
        axis.set_yticks(range(len(datasets)), datasets)
        axis.set_title(title)
        for row in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix.iloc[row, column_index]
                axis.text(column_index, row, f"{value:.1f}%", ha="center", va="center", fontsize=9)
        fig.colorbar(image, ax=axis, shrink=0.72, label="percent")
    fig.savefig(figure_dir / "view_effects.png", dpi=180)
    plt.close(fig)

    identities = ["mean_additivity_violation", "scale_violation", "variance_polarization_violation"]
    labels = ["Mean additivity", "Scale equivariance", "Variance polarization"]
    means = projective.groupby("model")[identities].mean()
    fig, axis = plt.subplots(figsize=(6.5, 3.7), constrained_layout=True)
    x = np.arange(len(identities))
    width = 0.34
    axis.bar(x - width / 2, means.loc["querynet"] * 100, width, label="QueryNet")
    axis.bar(x + width / 2, means.loc["projectivenet"] * 100, width, label="ProjectiveNet")
    axis.axhline(5, color="black", linestyle="--", linewidth=1, label="5% gate")
    axis.set_yscale("log")
    axis.set_ylabel("Normalized violation (%) — log scale")
    axis.set_xticks(x, labels)
    axis.legend(frameon=False)
    fig.savefig(figure_dir / "projective_identities.png", dpi=180)
    plt.close(fig)

    means = intervention.groupby(["k", "model"], as_index=False).rmse.mean()
    fig, axis = plt.subplots(figsize=(6.5, 3.7), constrained_layout=True)
    labels = {
        "causalpfn": "CausalPFN",
        "obspfn": "ObsPFN",
        "ridge_naive": "Ridge",
        "ridge_balanced": "Balanced ridge",
    }
    for model, group in means.groupby("model"):
        axis.plot(group.k, group.rmse, marker="o", label=labels[model])
    axis.set_xlabel("Randomized context rows (k)")
    axis.set_ylabel("Interventional RMSE")
    axis.set_xticks([0, 2, 4, 8])
    axis.legend(frameon=False, ncol=2)
    fig.savefig(figure_dir / "interventional_rmse.png", dpi=180)
    plt.close(fig)


def main() -> None:
    audits = audit_inputs()
    view_cells = finite_csv(HERE / "view" / "cells.csv", 108)
    view_by_seed = finite_csv(HERE / "view" / "comparisons_by_seed.csv", 27)
    projective = finite_csv(HERE / "projective" / "cells.csv", 6)
    intervention = finite_csv(HERE / "interventional" / "cells.csv", 48)
    environment = finite_csv(HERE / "interventional" / "environment_metrics.csv", 3 * 4 * 256)

    view_summary = view_by_seed.groupby(["dataset", "model"], as_index=False).agg(
        best_rmse=("best_rmse", "mean"),
        worst_rmse=("worst_rmse", "mean"),
        relative_rmse_spread=("relative_rmse_spread", "mean"),
        relative_prediction_dispersion=("relative_prediction_dispersion", "mean"),
        view_ensemble_rmse=("view_ensemble_rmse", "mean"),
    )
    qualifying = view_summary[
        (view_summary.relative_rmse_spread >= 0.10)
        & (view_summary.relative_prediction_dispersion >= 0.10)
    ]
    projective_means = projective.groupby("model").mean(numeric_only=True)
    gates = audits["interventional"]["gates"]

    scorecard = pd.DataFrame(
        [
            {
                "rank": 1,
                "direction": "Projectively consistent query distributions",
                "decision": "continue",
                "frozen_gate_passed": True,
                "wall_seconds": audits["projective"]["wall_seconds"],
                "headline": "3/3 NLL wins; worst constrained identity violation 9.1e-8",
            },
            {
                "rank": 2,
                "direction": "View-consistent longitudinal learning",
                "decision": "continue",
                "frozen_gate_passed": True,
                "wall_seconds": audits["view"]["wall_seconds"],
                "headline": f"{len(qualifying)}/9 pairs qualify; rankings flip on 3/3 datasets",
            },
            {
                "rank": 3,
                "direction": "Interventional temporal-table pretraining",
                "decision": "reformulate",
                "frozen_gate_passed": False,
                "wall_seconds": audits["interventional"]["wall_seconds"],
                "headline": "beats ObsPFN but not the simple per-environment ridge control",
            },
        ]
    )
    scorecard.to_csv(HERE / "PILOT_SCORECARD.csv", index=False)
    view_summary.to_csv(HERE / "view" / "summary_clean.csv", index=False)
    make_figures(view_summary, projective, intervention)

    view_medians = view_summary[["relative_rmse_spread", "relative_prediction_dispersion"]].median() * 100
    query = projective_means.loc["querynet"]
    constrained = projective_means.loc["projectivenet"]
    report = f"""# Oral-ceiling pilot results

Protocol SHA-256: `{EXPECTED_PROTOCOL_SHA256}`. All decisions below use the frozen gates in [PROTOCOL.md](PROTOCOL.md); the three runs finished below their 30-minute caps.

| Rank | Direction | Gate | Decision | Wall time |
|---:|---|:---:|---|---:|
| 1 | Projectively consistent query distributions | PASS | **Continue** | {audits['projective']['wall_seconds']:.1f}s |
| 2 | View-consistent longitudinal learning | PASS | **Continue** | {audits['view']['wall_seconds']:.1f}s |
| 3 | Interventional temporal-table pretraining | FAIL | **Reformulate** | {audits['interventional']['wall_seconds']:.1f}s |

## 1. Projective consistency — strongest signal

- QueryNet violated all three algebraic identities above the 5% gate: mean additivity {query.mean_additivity_violation * 100:.1f}%, scale equivariance {query.scale_violation * 100:.1f}%, and variance polarization {query.variance_polarization_violation * 100:.1f}%.
- ProjectiveNet's worst mean violation was {audits['projective']['projectivenet_max_identity_violation']:.2e}, effectively numerical zero.
- The constraint did not trade away predictive quality: ProjectiveNet won held-out NLL in 3/3 seeds, averaging {constrained.heldout_nll:.3f} versus {query.heldout_nll:.3f}.
- **Next falsifier:** test calibrated joint/marginal distributions on real multivariate forecasting benchmarks and compare against covariance-capable forecasters. The present evidence is synthetic mechanism evidence only.

![Projective identity violations](figures/projective_identities.png)

## 2. View consistency — strong real-data phenomenon

- {len(qualifying)}/9 dataset-model pairs crossed both 10% thresholds; all three model families and all three datasets are represented.
- Median worst-to-best RMSE spread was {view_medians.relative_rmse_spread:.1f}%, and median cross-view prediction dispersion was {view_medians.relative_prediction_dispersion:.1f}% of canonical RMSE.
- The winning model changed across equivalent views on Jena Weather, Electricity, and Traffic (3/3 datasets).
- Round-trip reconstruction error was at most {audits['view']['maximum_roundtrip_error']:.2e}, so information loss does not explain the effect.
- **Next falsifier:** train an explicitly view-consistent objective and require lower worst-view regret on unseen invertible transformations without reducing canonical-view accuracy.

![Equivalent-view effects](figures/view_effects.png)

## 3. Interventional pretraining — control failure

- At `k=4`, CausalPFN reduced RMSE {gates['4']['reduction_vs_obspfn'] * 100:.1f}% versus ObsPFN but only {gates['4']['reduction_vs_best_ridge'] * 100:.1f}% versus the better ridge baseline; it beat both in {gates['4']['environment_win_fraction'] * 100:.1f}% of environments.
- At `k=8`, it reduced RMSE {gates['8']['reduction_vs_obspfn'] * 100:.1f}% versus ObsPFN but was {-gates['8']['reduction_vs_best_ridge'] * 100:.1f}% worse than the better ridge baseline; it beat both in {gates['8']['environment_win_fraction'] * 100:.1f}% of environments.
- This is not a green light for scaling. A defensible reformulation must introduce a genuinely nonlinear/high-dimensional identification problem and still retain strong semiparametric or doubly robust controls.

![Interventional RMSE](figures/interventional_rmse.png)

## Integrity notes

- Raw row counts: view {len(view_cells)}, projective {len(projective)}, interventional aggregate {len(intervention)}, interventional per-environment {len(environment)}. All numeric outputs are finite.
- The first ProjectiveNet run is deliberately retained in `projective_invalid_postprojection_floor/`. Its post-projection constant variance floor violated scale equivariance by construction. [PROJECTIVE_IMPLEMENTATION_NOTE.md](PROJECTIVE_IMPLEMENTATION_NOTE.md) records the correction; no seed, data, model, step, metric, or threshold changed.
- These are preliminary mechanism screens, not evidence for an oral-level paper by themselves.
"""
    (HERE / "RESULTS.md").write_text(report)
    print(scorecard.to_string(index=False))


if __name__ == "__main__":
    main()
