"""Analyze H4 early semantic-shadow forecasting gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent


def safe_spearman(left: pd.Series | np.ndarray, right: pd.Series | np.ndarray) -> float:
    left_array, right_array = np.asarray(left), np.asarray(right)
    if np.unique(left_array).size < 2 or np.unique(right_array).size < 2:
        return 0.0
    value = spearmanr(left_array, right_array).statistic
    return float(value) if np.isfinite(value) else float("nan")


def analyze(input_dir: Path, output_dir: Path) -> dict:
    config = json.loads((HERE / "hypothesis_04_config.json").read_text())
    rows = []
    artifacts = sorted(input_dir.glob("*.npz"))
    if not artifacts:
        raise FileNotFoundError(f"no H4 artifacts under {input_dir}")
    maximum_initial_gap = 0.0
    for artifact in artifacts:
        bundle = np.load(artifact)
        manifest = json.loads(artifact.with_suffix(".json").read_text())
        predictions = bundle["validation_predictions"].astype(np.float64)
        checkpoints = bundle["checkpoints"].astype(int)
        mse = np.mean((predictions[1:] - predictions[0][None, ...]) ** 2, axis=(0, 2, 3))
        row = {
            "dataset": manifest["dataset"], "model": manifest["model"],
            "seed": manifest["seed"], "config_id": manifest["optimizer"]["id"],
            "learning_rate": manifest["optimizer"]["learning_rate"],
            "weight_decay": manifest["optimizer"]["weight_decay"],
            "batch_size": manifest["optimizer"]["batch_size"],
        }
        row.update({f"mse_epoch_{epoch}": float(value) for epoch, value in zip(checkpoints, mse)})
        rows.append(row)
        maximum_initial_gap = max(maximum_initial_gap, float(manifest["maximum_initial_gap"]))
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "h4_seed_bundles.csv", index=False)
    mse_columns = [f"mse_epoch_{epoch}" for epoch in config["checkpoints"]]
    cells = frame.groupby(["dataset", "model", "config_id"], as_index=False)[mse_columns].mean()
    cells.to_csv(output_dir / "h4_config_cells.csv", index=False)

    ft = cells[cells.model == "ft_transformer"]
    per_dataset = {}
    ft_passing = 0
    for dataset in config["datasets"]:
        current = ft[ft.dataset == dataset]
        rho = safe_spearman(
            np.log10(current.mse_epoch_2.clip(lower=1e-30)),
            np.log10(current.mse_epoch_20.clip(lower=1e-30)),
        ) if len(current) >= 3 else float("nan")
        per_dataset[dataset] = rho
        ft_passing += int(rho >= 0.70)
    rank_parts = []
    for dataset, current in ft.groupby("dataset"):
        rank_parts.append(pd.DataFrame({
            "dataset": dataset,
            "epoch_2_rank": rankdata(current.mse_epoch_2, method="average"),
            "epoch_0_rank": rankdata(current.mse_epoch_0, method="average"),
            "epoch_20_rank": rankdata(current.mse_epoch_20, method="average"),
            "material": (
                current.mse_epoch_20 > float(config["material_final_mse"])
            ).astype(int).to_numpy(),
        }))
    ranks = pd.concat(rank_parts, ignore_index=True) if rank_parts else pd.DataFrame()
    pooled_epoch2 = safe_spearman(
        ranks.epoch_2_rank, ranks.epoch_20_rank,
    ) if len(ranks) else float("nan")
    pooled_epoch0 = safe_spearman(
        ranks.epoch_0_rank, ranks.epoch_20_rank,
    ) if len(ranks) else float("nan")
    target = (ft.mse_epoch_20 > float(config["material_final_mse"])).astype(int)
    auroc = float(roc_auc_score(
        ranks.material, ranks.epoch_2_rank
    )) if len(ranks) and ranks.material.nunique() == 2 else float("nan")
    stable = cells[cells.model.isin(["mlp", "resnet"])]
    stable_fraction = float(
        (stable.mse_epoch_20 < float(config["stable_final_mse"])).mean()
    ) if len(stable) else 0.0
    expected = (
        len(config["datasets"]) * len(config["models"]) * len(config["seeds"])
        * len(config["learning_rates"]) * len(config["weight_decays"])
        * len(config["batch_sizes"])
    )
    complete = len(artifacts) == expected
    improvement = pooled_epoch2 - pooled_epoch0
    gates = {
        "ft_dataset_correlations": {
            "value": ft_passing, "required": 2, "pass": ft_passing >= 2,
            "spearman_by_dataset": per_dataset,
        },
        "early_over_initial_improvement": {
            "value": improvement, "required": 0.20,
            "epoch2_spearman": pooled_epoch2, "epoch0_spearman": pooled_epoch0,
            "pass": bool(improvement >= 0.20),
        },
        "material_configuration_auroc": {
            "value": auroc, "required": 0.85,
            "material": int(target.sum()), "total": int(len(target)),
            "score": "within_dataset_epoch2_rank",
            "pass": bool(auroc >= 0.85),
        },
        "stable_control_fraction": {
            "value": stable_fraction, "required": 0.90,
            "pass": stable_fraction >= 0.90,
        },
        "initial_match": {
            "value": maximum_initial_gap,
            "required_maximum": float(config["initial_match_tolerance"]),
            "pass": maximum_initial_gap <= float(config["initial_match_tolerance"]),
        },
    }
    summary = {
        "status": "complete" if complete else "in_progress",
        "artifacts": len(artifacts), "expected_artifacts": expected,
        "gates": gates,
        "hypothesis_supported": bool(complete and all(value["pass"] for value in gates.values())),
    }
    (output_dir / "h4_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
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
