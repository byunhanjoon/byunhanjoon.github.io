"""Independent-candidate randomization for unbiased cross-score selection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_robust_model_selection as RMS
from analyze_cross_quotient_selection import PANELS, cross_and_mean_scores, iid_u_scores
from analyze_strength2_cover import proper_loss


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DRAWS = 512
METHODS = ("strength2_cross32", "iid_u32")


def analyze_dataset(panel: str, dataset: str, models: list[str], directory: Path):
    validation, test = [], []
    val_y = test_y = None
    shape = None
    task = None
    for model in models:
        archive = np.load(directory / f"{dataset}__{model}.npz")
        manifest = json.loads((directory / f"{dataset}__{model}.json").read_text())
        val_y, test_y = archive["validation_y"], archive["test_y"]
        shape = tuple(archive["validation_predictions"].shape[:4])
        task = manifest["task"]
        validation.append(archive["validation_predictions"].reshape(
            (-1,) + archive["validation_predictions"].shape[-2:]
        ).astype(np.float64))
        test.append(archive["test_predictions"].reshape(
            (-1,) + archive["test_predictions"].shape[-2:]
        ).astype(np.float64))
    assert val_y is not None and test_y is not None and shape is not None
    quotient_val = np.asarray([
        proper_loss(val_y, values.mean(axis=0)) for values in validation
    ])
    quotient_test = np.asarray([
        proper_loss(test_y, values.mean(axis=0)) for values in test
    ])
    winner = int(np.argmin(quotient_val))
    val_scores = {method: [] for method in METHODS}
    test_scores = {method: [] for method in METHODS}
    for model, val_flat, test_flat in zip(models, validation, test):
        first = RMS.action_ids(
            shape, RMS.stable_seed("cross-independent-a", panel, dataset, model)
        )
        second = RMS.action_ids(
            shape, RMS.stable_seed("cross-independent-b", panel, dataset, model)
        )
        s2_cross, _ = cross_and_mean_scores(
            val_y, val_flat,
            first["strength2"][:DRAWS], second["strength2"][:DRAWS]
        )
        iid_ids = np.concatenate([
            first["iid16"][:DRAWS], second["iid16"][:DRAWS]
        ], axis=1)
        iid_score = iid_u_scores(val_y, val_flat, iid_ids)
        _, s2_test = cross_and_mean_scores(
            test_y, test_flat,
            first["strength2"][:DRAWS], second["strength2"][:DRAWS]
        )
        iid_test = RMS.batched_losses(test_y, test_flat, iid_ids, batch=8)
        val_scores["strength2_cross32"].append(s2_cross)
        val_scores["iid_u32"].append(iid_score)
        test_scores["strength2_cross32"].append(s2_test)
        test_scores["iid_u32"].append(iid_test)
    rows = []
    for method in METHODS:
        scores = np.stack(val_scores[method], axis=1)
        realized = np.stack(test_scores[method], axis=1)
        selected = np.argmin(scores, axis=1)
        for draw, chosen in enumerate(selected):
            rows.append({
                "panel": panel, "dataset": dataset, "task": task,
                "method": method, "draw": draw,
                "selection_agreement": bool(chosen == winner),
                "validation_quotient_regret": float(
                    quotient_val[chosen] - quotient_val[winner]
                ),
                "selected_quotient_test_loss": float(quotient_test[chosen]),
                "selected_realized_test_loss": float(realized[draw, chosen]),
            })
    return rows


def main() -> None:
    rows = []
    for panel, config_name, directory_name in PANELS:
        config = json.loads((HERE / config_name).read_text())
        for dataset in config["datasets"]:
            rows.extend(analyze_dataset(
                panel, dataset, config["models"], RESULTS / directory_name
            ))
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "cross_score_coupling_draws.csv", index=False)
    cells = frame.groupby(["panel", "dataset", "method"], as_index=False).agg(
        selection_agreement=("selection_agreement", "mean"),
        validation_quotient_regret=("validation_quotient_regret", "mean"),
        selected_quotient_test_loss=("selected_quotient_test_loss", "mean"),
        selected_realized_test_loss=("selected_realized_test_loss", "mean"),
    )
    cells.to_csv(RESULTS / "cross_score_coupling_cells.csv", index=False)
    summary: dict[str, object] = {"status": "complete", "panels": {}}
    agreement_passes = regret_passes = 0
    for panel, current in cells.groupby("panel"):
        means = current.groupby("method").mean(numeric_only=True)
        agreement = bool(
            means.loc["strength2_cross32", "selection_agreement"]
            > means.loc["iid_u32", "selection_agreement"]
        )
        regret = bool(
            means.loc["strength2_cross32", "validation_quotient_regret"]
            < means.loc["iid_u32", "validation_quotient_regret"]
        )
        agreement_passes += agreement; regret_passes += regret
        test_pivot = current.pivot(
            index="dataset", columns="method", values="selected_realized_test_loss"
        )
        difference = test_pivot.strength2_cross32 - test_pivot.iid_u32
        summary["panels"][panel] = {
            "agreement_above_iid": agreement,
            "validation_regret_below_iid": regret,
            "method_means": means.reset_index().to_dict(orient="records"),
            "strength2_minus_iid_realized_test_loss_mean": float(difference.mean()),
            "sources_strength2_test_lower": int((difference < 0).sum()),
            "sources": len(difference),
        }
    summary["panels_agreement_above_iid"] = int(agreement_passes)
    summary["panels_regret_below_iid"] = int(regret_passes)
    summary["independent_candidate_gate_passed"] = bool(
        agreement_passes >= 4 and regret_passes >= 4
    )
    (RESULTS / "cross_score_coupling_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
