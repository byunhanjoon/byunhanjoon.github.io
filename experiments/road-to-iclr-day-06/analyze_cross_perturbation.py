"""Analyze frozen H5 transfer from early schema shadows to seed fragility."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent


def mean_pairwise_mse(predictions: list[np.ndarray]) -> float:
    pairs = [
        float(np.mean((predictions[left] - predictions[right]) ** 2))
        for left, right in itertools.combinations(range(len(predictions)), 2)
    ]
    if not pairs:
        raise ValueError("seed fragility requires at least two predictions")
    return float(np.mean(pairs))


def safe_spearman(left: pd.Series | np.ndarray, right: pd.Series | np.ndarray) -> float:
    left_array, right_array = np.asarray(left), np.asarray(right)
    if np.unique(left_array).size < 2 or np.unique(right_array).size < 2:
        return 0.0
    value = spearmanr(left_array, right_array).statistic
    return float(value) if np.isfinite(value) else float("nan")


def top_quartile_labels(values: pd.Series, count: int = 3) -> pd.Series:
    """Label the frozen top-count target with the protocol's average tie ranks."""
    order = values.rank(method="average", ascending=False)
    return (order <= count).astype(int)


def analyze(input_dir: Path, output_dir: Path) -> dict:
    config = json.loads((HERE / "hypothesis_04_config.json").read_text())
    artifacts = sorted(input_dir.glob("*.npz"))
    if not artifacts:
        raise FileNotFoundError(f"no H4/H5 artifacts under {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_rows: list[dict] = []
    maximum_initial_gap = 0.0
    for artifact in artifacts:
        bundle = np.load(artifact)
        manifest = json.loads(artifact.with_suffix(".json").read_text())
        validation = bundle["validation_predictions"].astype(np.float64)
        test = bundle["test_predictions"].astype(np.float64)
        checkpoints = bundle["checkpoints"].astype(int)
        shadow_mse = np.mean(
            (validation[1:] - validation[0][None, ...]) ** 2,
            axis=(0, 2, 3),
        )
        row = {
            "dataset": manifest["dataset"], "model": manifest["model"],
            "seed": int(manifest["seed"]),
            "config_id": manifest["optimizer"]["id"],
        }
        row.update({
            f"shadow_epoch_{epoch}": float(value)
            for epoch, value in zip(checkpoints, shadow_mse)
        })
        final_index = int(np.flatnonzero(checkpoints == 20)[0])
        row["canonical_test_epoch_20"] = test[0, final_index]
        seed_rows.append(row)
        maximum_initial_gap = max(
            maximum_initial_gap, float(manifest["maximum_initial_gap"])
        )

    shadow_columns = [f"shadow_epoch_{epoch}" for epoch in config["checkpoints"]]
    scalar_rows = [
        {key: value for key, value in row.items() if key != "canonical_test_epoch_20"}
        for row in seed_rows
    ]
    pd.DataFrame(scalar_rows).to_csv(output_dir / "h5_seed_scores.csv", index=False)

    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in seed_rows:
        key = (row["dataset"], row["model"], row["config_id"])
        grouped.setdefault(key, []).append(row)
    cells = []
    for (dataset, model, config_id), rows in sorted(grouped.items()):
        current = {"dataset": dataset, "model": model, "config_id": config_id}
        for column in shadow_columns:
            current[column] = float(np.mean([row[column] for row in rows]))
        current["seed_fragility_epoch_20"] = mean_pairwise_mse(
            [row["canonical_test_epoch_20"] for row in rows]
        )
        current["seeds"] = len(rows)
        cells.append(current)
    cells_frame = pd.DataFrame(cells)
    cells_frame.to_csv(output_dir / "h5_config_cells.csv", index=False)

    correlations = []
    auc_rows = []
    for (dataset, model), current in cells_frame.groupby(["dataset", "model"]):
        target = np.log10(current.seed_fragility_epoch_20.clip(lower=1e-30))
        row = {"dataset": dataset, "model": model}
        for epoch in config["checkpoints"]:
            score = np.log10(current[f"shadow_epoch_{epoch}"].clip(lower=1e-30))
            row[f"rho_epoch_{epoch}"] = safe_spearman(score, target)
        correlations.append(row)
        material = top_quartile_labels(current.seed_fragility_epoch_20)
        score = np.log10(current.shadow_epoch_2.clip(lower=1e-30))
        auc_rows.append({
            "dataset": dataset, "model": model,
            "auroc_epoch_2_top_quartile": float(roc_auc_score(material, score))
            if material.nunique() == 2 else float("nan"),
            "positive": int(material.sum()), "total": int(len(material)),
        })
    correlation_frame = pd.DataFrame(correlations)
    auc_frame = pd.DataFrame(auc_rows)
    correlation_frame.to_csv(output_dir / "h5_correlations.csv", index=False)
    auc_frame.to_csv(output_dir / "h5_top_quartile_auc.csv", index=False)

    ft = cells_frame[cells_frame.model == "ft_transformer"].copy()
    rank_parts = []
    for dataset, current in ft.groupby("dataset"):
        rank_parts.append(pd.DataFrame({
            "dataset": dataset,
            "early_rank": rankdata(current.shadow_epoch_2, method="average"),
            "initial_rank": rankdata(current.shadow_epoch_0, method="average"),
            "target_rank": rankdata(current.seed_fragility_epoch_20, method="average"),
        }))
    ranks = pd.concat(rank_parts, ignore_index=True) if rank_parts else pd.DataFrame()
    pooled_epoch2 = safe_spearman(ranks.early_rank, ranks.target_rank) if len(ranks) else float("nan")
    pooled_epoch0 = safe_spearman(ranks.initial_rank, ranks.target_rank) if len(ranks) else float("nan")
    ft_correlations = correlation_frame[correlation_frame.model == "ft_transformer"]
    ft_auc = auc_frame[auc_frame.model == "ft_transformer"]
    ft_passing = int((ft_correlations.rho_epoch_2 >= 0.60).sum())
    equal_dataset_auc = float(ft_auc.auroc_epoch_2_top_quartile.mean())
    expected = (
        len(config["datasets"]) * len(config["models"]) * len(config["seeds"])
        * len(config["learning_rates"]) * len(config["weight_decays"])
        * len(config["batch_sizes"])
    )
    complete = len(artifacts) == expected and bool((cells_frame.seeds == 3).all())
    gates = {
        "ft_dataset_correlations": {
            "value": ft_passing, "required": 2, "pass": ft_passing >= 2,
            "rho_by_dataset": {
                row.dataset: float(row.rho_epoch_2)
                for row in ft_correlations.itertuples()
            },
        },
        "equal_dataset_pooled_spearman": {
            "value": pooled_epoch2, "required": 0.60,
            "pass": bool(pooled_epoch2 >= 0.60),
        },
        "early_over_initial_improvement": {
            "value": pooled_epoch2 - pooled_epoch0, "required": 0.20,
            "epoch2_spearman": pooled_epoch2, "epoch0_spearman": pooled_epoch0,
            "pass": bool(pooled_epoch2 - pooled_epoch0 >= 0.20),
        },
        "top_quartile_auroc": {
            "value": equal_dataset_auc, "required": 0.80,
            "pass": bool(equal_dataset_auc >= 0.80 and len(ft_auc) == 3),
            "by_dataset": {
                row.dataset: float(row.auroc_epoch_2_top_quartile)
                for row in ft_auc.itertuples()
            },
        },
        "complete_and_matched": {
            "artifacts": len(artifacts), "required_artifacts": expected,
            "maximum_initial_gap": maximum_initial_gap,
            "pass": bool(complete and maximum_initial_gap <= 1e-6),
        },
    }
    summary = {
        "status": "complete" if complete else "in_progress",
        "artifacts": len(artifacts), "expected_artifacts": expected,
        "gates": gates,
        "hypothesis_supported": bool(complete and all(item["pass"] for item in gates.values())),
    }
    (output_dir / "h5_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=HERE / "results" / "h4")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    analyze(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
