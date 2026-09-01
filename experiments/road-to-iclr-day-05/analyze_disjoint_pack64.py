"""Four mutually disjoint covers versus two disjoint pairs at 64 fits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_cross_quotient_selection as CQS
import analyze_robust_model_selection as RMS
from analyze_cross_score_budget_frontier import cover_block_scores
from analyze_disjoint_pair_cross import DRAWS, cover_graph, sample_independent_blocks
from analyze_strength2_cover import proper_loss


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def sample_pack_and_pairs(shape: tuple[int, ...], panel: str, dataset: str):
    ids, _, adjacency, neighbors = cover_graph(shape)
    rng = np.random.default_rng(RMS.stable_seed("disjoint-pack64", panel, dataset))
    pack = np.empty((DRAWS, 4, 16), dtype=int)
    pairs = np.empty_like(pack)
    attempts = []
    if int(np.prod(shape)) <= 32:
        first = rng.integers(0, len(ids), size=DRAWS)
        if len(ids) == 1:
            second = first
        else:
            second = np.fromiter(
                (neighbors[index][rng.integers(0, len(neighbors[index]))] for index in first),
                dtype=int, count=DRAWS,
            )
        pack[:, 0] = pairs[:, 0] = ids[first]
        pack[:, 1] = pairs[:, 1] = ids[second]
        pack[:, 2] = pairs[:, 2] = ids[first]
        pack[:, 3] = pairs[:, 3] = ids[second]
        return pack, pairs, 1.0

    for draw in range(DRAWS):
        tries = 0
        while True:
            tries += 1
            a = int(rng.integers(0, len(ids)))
            b = int(rng.choice(neighbors[a]))
            common_ab = np.flatnonzero(adjacency[a] & adjacency[b])
            if not len(common_ab):
                continue
            c = int(rng.choice(common_ab))
            common_abc = np.flatnonzero(adjacency[a] & adjacency[b] & adjacency[c])
            if not len(common_abc):
                continue
            d = int(rng.choice(common_abc))
            pack[draw] = ids[[a, b, c, d]]
            e = int(rng.integers(0, len(ids)))
            f = int(rng.choice(neighbors[e]))
            pairs[draw] = ids[[a, b, e, f]]
            attempts.append(tries)
            break
    return pack, pairs, float(np.mean(attempts))


def prediction_residuals(flat: np.ndarray, blocks: np.ndarray) -> np.ndarray:
    quotient = flat.mean(axis=0)
    output = np.empty(len(blocks), dtype=np.float64)
    for start in range(0, len(blocks), 8):
        stop = min(start + 8, len(blocks))
        prediction = np.mean(np.stack([
            flat[blocks[start:stop, block]].mean(axis=1) for block in range(4)
        ], axis=1), axis=1)
        output[start:stop] = np.mean(
            np.sum((prediction - quotient) ** 2, axis=-1), axis=1
        )
    return output


def analyze_dataset(panel: str, dataset: str, models: list[str], directory: Path):
    validation, test = [], []
    validation_y = test_y = None
    shape = None
    task = ""
    for model in models:
        archive = np.load(directory / f"{dataset}__{model}.npz")
        manifest = json.loads((directory / f"{dataset}__{model}.json").read_text())
        validation_y, test_y = archive["validation_y"], archive["test_y"]
        shape = tuple(int(value) for value in archive["validation_predictions"].shape[:4])
        task = manifest["task"]
        validation.append(archive["validation_predictions"].reshape(
            (-1,) + archive["validation_predictions"].shape[-2:]
        ).astype(np.float64))
        test.append(archive["test_predictions"].reshape(
            (-1,) + archive["test_predictions"].shape[-2:]
        ).astype(np.float64))
    assert validation_y is not None and test_y is not None and shape is not None
    pack, pairs, attempts = sample_pack_and_pairs(shape, panel, dataset)
    independent = sample_independent_blocks(
        shape, np.random.default_rng(RMS.stable_seed("pack-independent64", panel, dataset))
    )
    actions = {
        "mutually_disjoint_pack64": pack,
        "two_disjoint_pairs64": pairs,
        "independent_four64": independent,
    }
    quotient_val = np.asarray([
        proper_loss(validation_y, values.mean(axis=0)) for values in validation
    ])
    quotient_test = np.asarray([
        proper_loss(test_y, values.mean(axis=0)) for values in test
    ])
    winner = int(np.argmin(quotient_val))
    scores = {method: [] for method in actions}
    test_losses = {method: [] for method in actions}
    calibration = []
    for model, val_flat, test_flat, exact in zip(models, validation, test, quotient_val):
        for method, blocks in actions.items():
            _, val_loss = cover_block_scores(validation_y, val_flat, blocks)
            _, test_loss = cover_block_scores(test_y, test_flat, blocks)
            residual = prediction_residuals(val_flat, blocks)
            scores[method].append(val_loss); test_losses[method].append(test_loss)
            bias = float(val_loss.mean() - exact)
            calibration.append({
                "panel": panel, "dataset": dataset, "task": task, "model": model,
                "method": method, "product_cells": int(np.prod(shape)),
                "score_bias": bias,
                "score_rmse": float(np.sqrt(val_loss.var(ddof=1) + bias ** 2)),
                "prediction_residual": float(residual.mean()),
                "max_absolute_score_error": float(np.max(np.abs(val_loss - exact))),
                "mean_pack_sampling_attempts": attempts,
            })
    rows = []
    for method in actions:
        score_matrix = np.stack(scores[method], axis=1)
        test_matrix = np.stack(test_losses[method], axis=1)
        selected = np.argmin(score_matrix, axis=1)
        for draw, chosen in enumerate(selected):
            rows.append({
                "panel": panel, "dataset": dataset, "task": task,
                "method": method, "draw": draw,
                "selection_agreement": bool(chosen == winner),
                "validation_quotient_regret": float(quotient_val[chosen] - quotient_val[winner]),
                "selected_quotient_test_loss": float(quotient_test[chosen]),
                "selected_realized_test_loss": float(test_matrix[draw, chosen]),
            })
    return rows, calibration


def main() -> None:
    rows, calibration_rows = [], []
    for panel, config_name, directory_name in CQS.PANELS:
        config = json.loads((HERE / config_name).read_text())
        for dataset in config["datasets"]:
            current, calibration = analyze_dataset(
                panel, dataset, config["models"], RESULTS / directory_name
            )
            rows.extend(current); calibration_rows.extend(calibration)
    draws = pd.DataFrame(rows)
    draws.to_csv(RESULTS / "disjoint_pack64_draws.csv", index=False)
    cells = draws.groupby(["panel", "dataset", "task", "method"], as_index=False).mean(numeric_only=True)
    cells.to_csv(RESULTS / "disjoint_pack64_cells.csv", index=False)
    calibration = pd.DataFrame(calibration_rows)
    calibration.to_csv(RESULTS / "disjoint_pack64_calibration.csv", index=False)

    exact = calibration[
        (calibration.method == "mutually_disjoint_pack64") &
        (calibration.product_cells <= 64)
    ]
    summary: dict[str, object] = {
        "status": "complete", "exact_partition_candidates": len(exact),
        "exact_partition_max_absolute_error": float(exact.max_absolute_score_error.max()),
        "panels": {},
    }
    counts = {"rmse_nolower": 0, "rmse_strict": 0, "residual_nolower": 0,
              "residual_strict": 0, "agreement": 0, "regret": 0}
    for panel, current in cells.groupby("panel"):
        means = current.groupby("method").mean(numeric_only=True)
        cal = calibration[calibration.panel == panel].groupby("method").mean(numeric_only=True)
        action, control = means.loc["mutually_disjoint_pack64"], means.loc["two_disjoint_pairs64"]
        armse, crmse = cal.loc["mutually_disjoint_pack64", "score_rmse"], cal.loc["two_disjoint_pairs64", "score_rmse"]
        ares, cres = cal.loc["mutually_disjoint_pack64", "prediction_residual"], cal.loc["two_disjoint_pairs64", "prediction_residual"]
        clauses = {
            "score_rmse_nohigher": bool(armse <= crmse + 1e-15),
            "score_rmse_strictly_lower": bool(armse < crmse - 1e-15),
            "prediction_residual_nohigher": bool(ares <= cres + 1e-15),
            "prediction_residual_strictly_lower": bool(ares < cres - 1e-15),
            "agreement_nolower": bool(action.selection_agreement >= control.selection_agreement),
            "regret_nohigher": bool(action.validation_quotient_regret <= control.validation_quotient_regret),
        }
        for key, value in clauses.items():
            counts[{"score_rmse_nohigher": "rmse_nolower", "score_rmse_strictly_lower": "rmse_strict",
                    "prediction_residual_nohigher": "residual_nolower", "prediction_residual_strictly_lower": "residual_strict",
                    "agreement_nolower": "agreement", "regret_nohigher": "regret"}[key]] += int(value)
        summary["panels"][panel] = {
            "clauses": clauses, "score_rmse": cal.score_rmse.to_dict(),
            "prediction_residual": cal.prediction_residual.to_dict(),
            "method_means": means.reset_index().to_dict(orient="records"),
        }
    summary["panels_passing_by_clause"] = counts
    summary["frozen_gate_passed"] = bool(
        counts["rmse_nolower"] == 5 and counts["rmse_strict"] >= 3
        and counts["residual_nolower"] == 5 and counts["residual_strict"] >= 3
        and counts["agreement"] >= 4 and counts["regret"] >= 4
        and summary["exact_partition_max_absolute_error"] < 1e-12
    )
    (RESULTS / "disjoint_pack64_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
