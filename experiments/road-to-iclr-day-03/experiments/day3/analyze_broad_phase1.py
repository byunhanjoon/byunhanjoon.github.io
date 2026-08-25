"""Analyze broad tiers and freeze the validation-selected confirmation methods."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .broad_data import config


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/day3/broad_benchmark"
FIGURES = RESULTS / "figures"


def load_phase1() -> pd.DataFrame:
    paths = sorted(
        path
        for path in RESULTS.glob("phase1_shard*.csv")
        if not path.stem.endswith("_curves")
    )
    if not paths:
        raise FileNotFoundError("No phase1 shard files")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    key = [
        "dataset",
        "representation",
        "target_kappa",
        "model",
        "remedy",
        "seed",
        "learning_rate_requested",
        "ridge_requested",
        "precondition_frequency_requested",
    ]
    return frame.drop_duplicates(key, keep="last")


def controlled_pairs(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    valid = frame[
        frame.representation.eq("controlled")
        & frame.failure.fillna("").eq("")
        & frame[metric].notna()
    ]
    keys = ["dataset", "task", "model", "remedy", "seed"]
    wide = valid.pivot_table(index=keys, columns="target_kappa", values=metric).reset_index()
    wide = wide.dropna(subset=[1.0, 1000.0])
    wide["sensitivity"] = wide[1000.0] - wide[1.0]
    adam = wide[wide.remedy.eq("adamw")][["dataset", "model", "seed", 1.0, 1000.0, "sensitivity"]].rename(
        columns={1.0: "adam_k1", 1000.0: "adam_k1000", "sensitivity": "adam_sensitivity"}
    )
    wide = wide.merge(adam, on=["dataset", "model", "seed"], how="left")
    wide["scale"] = np.where(wide.task.eq("regression"), wide.adam_k1.abs().clip(lower=1e-12), 1.0)
    wide["sensitivity_normalized"] = wide.sensitivity / wide.scale
    wide["k1_gain_normalized"] = (wide[1.0] - wide.adam_k1) / wide.scale
    wide["endpoint_gain_normalized"] = (wide[1000.0] - wide.adam_k1000) / wide.scale
    harmful_adam = (-wide.adam_sensitivity / wide.scale).clip(lower=0)
    harmful_remedy = (-wide.sensitivity_normalized).clip(lower=0)
    wide["sensitivity_reduction"] = np.where(
        harmful_adam > 1e-12, 1 - harmful_remedy / harmful_adam, np.nan
    )
    return wide


def bootstrap(values: np.ndarray, samples: int = 10000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20260825)
    means = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def source_group_values(pairs: pd.DataFrame, remedy: str) -> pd.Series:
    cfg = config()
    part = pairs[pairs.remedy.eq(remedy)].copy()
    part["source_group"] = part.dataset.map(cfg["source_groups"]).fillna(part.dataset)
    return part.groupby(["source_group", "model"]).sensitivity_normalized.mean()


def select_confirmation(validation_pairs: pd.DataFrame) -> dict[str, object]:
    # Exact canonicalizers are proofs/controls and are always confirmed
    # separately. Select four deployable/preprocessing comparisons by frozen
    # validation criteria.
    candidates = [
        "diagonal_adamw",
        "whiten_adamw",
        "input_natural",
        "first_layer_kfac",
        "shampoo",
        "soap",
    ]
    rows = []
    mlp = validation_pairs[validation_pairs.model.eq("mlp")]
    for remedy in candidates:
        part = mlp[mlp.remedy.eq(remedy)]
        rows.append(
            {
                "remedy": remedy,
                "datasets": int(part.dataset.nunique()),
                "mean_k1_gain_normalized": float(part.k1_gain_normalized.mean()),
                "mean_sensitivity_normalized": float(part.sensitivity_normalized.mean()),
                "mean_sensitivity_reduction": float(part.sensitivity_reduction.mean()),
                "mean_endpoint_gain_normalized": float(part.endpoint_gain_normalized.mean()),
            }
        )
    table = pd.DataFrame(rows)
    eligible = table[
        table.datasets.eq(len(config()["datasets"]))
        & (
            table.mean_k1_gain_normalized
            >= -float(
                config()["confirmation_selection"][
                    "maximum_mean_unperturbed_normalized_loss"
                ]
            )
        )
    ]
    ordered = eligible.sort_values(
        ["mean_sensitivity_reduction", "mean_endpoint_gain_normalized"], ascending=False
    )
    required = int(config()["confirmation_selection"]["count"])
    selected = ordered.head(required).remedy.tolist()
    if len(selected) != required:
        raise RuntimeError(
            f"Only {len(selected)} remedies have complete validation coverage; "
            f"{required} are required for frozen confirmation"
        )
    payload = {
        "selection_metric": "validation-only mean paired sensitivity reduction, then endpoint gain",
        "selected_deployable_comparisons": selected,
        "always_confirmed_controls": [
            "adamw",
            "anchor_whiten_adamw",
            "sketch_anchor_whiten_adamw",
        ],
        "table": rows,
    }
    table.to_csv(RESULTS / "confirmation_selection_table.csv", index=False)
    (RESULTS / "confirmation_selection.json").write_text(json.dumps(payload, indent=2))
    return payload


def natural_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame[
        frame.representation.isin(["cumulative_helmert", "local_adjacent"])
        & frame.failure.fillna("").eq("")
    ]
    keys = ["dataset", "task", "model", "seed"]
    wide = valid.pivot_table(index=keys, columns="representation", values="test_primary").reset_index()
    wide["local_minus_cumulative"] = wide.local_adjacent - wide.cumulative_helmert
    wide["scale"] = np.where(
        wide.task.eq("regression"), wide.cumulative_helmert.abs().clip(lower=1e-12), 1.0
    )
    wide["local_minus_cumulative_normalized"] = (
        wide.local_minus_cumulative / wide.scale
    )
    wide.to_csv(RESULTS / "natural_encoding_pairs.csv", index=False)
    return wide


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    frame = load_phase1()
    frame.to_csv(RESULTS / "phase1_all.csv", index=False)
    test_pairs = controlled_pairs(frame, "test_primary")
    validation_pairs = controlled_pairs(frame, "val_primary")
    test_pairs.to_csv(RESULTS / "phase1_test_pairs.csv", index=False)
    validation_pairs.to_csv(RESULTS / "phase1_validation_pairs.csv", index=False)
    selection = select_confirmation(validation_pairs)
    natural = natural_analysis(frame)

    summaries = []
    for remedy in sorted(test_pairs.remedy.unique()):
        part = test_pairs[test_pairs.remedy.eq(remedy)]
        grouped = source_group_values(test_pairs, remedy)
        ci = bootstrap(grouped.to_numpy()) if len(grouped) else (math.nan, math.nan)
        nonzero = grouped[grouped.abs() > 1e-12]
        pvalue = float(wilcoxon(nonzero).pvalue) if len(nonzero) >= 3 else math.nan
        summaries.append(
            {
                "remedy": remedy,
                "pairs": len(part),
                "datasets": part.dataset.nunique(),
                "models": part.model.nunique(),
                "mean_sensitivity_normalized": part.sensitivity_normalized.mean(),
                "median_sensitivity_normalized": part.sensitivity_normalized.median(),
                "harmful_fraction": (part.sensitivity_normalized < 0).mean(),
                "mean_k1_gain_normalized": part.k1_gain_normalized.mean(),
                "mean_sensitivity_reduction": part.sensitivity_reduction.mean(),
                "source_group_mean": grouped.mean(),
                "source_group_ci_low": ci[0],
                "source_group_ci_high": ci[1],
                "source_group_wilcoxon_p": pvalue,
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(RESULTS / "phase1_summary.csv", index=False)

    mlp = summary[summary.remedy.ne("adamw")].sort_values("mean_sensitivity_normalized")
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.barh(mlp.remedy, 100 * mlp.mean_sensitivity_normalized, color="#2a9d8f")
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Mean κ=1000 − κ=1 utility (%)")
    axis.set_title("Broad MLP remedy sensitivity")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "phase1_remedy_sensitivity.png", dpi=190)
    plt.close(fig)

    failures = frame.failure.fillna("").ne("")
    payload = {
        "runs": len(frame),
        "datasets": int(frame.dataset.nunique()),
        "failures": int(failures.sum()),
        "confirmation_selection": selection,
        "natural_pairs": len(natural),
    }
    (RESULTS / "phase1_summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
