"""Antithetic disjoint-cover pairs with an unbiased outer cross-score."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_cross_quotient_selection as CQS
import analyze_robust_model_selection as RMS
from analyze_cross_score_budget_frontier import cover_block_scores
from analyze_strength2_cover import component_coefficients, proper_loss, strength2_family


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DRAWS = 1_024


@lru_cache(maxsize=None)
def cover_graph(shape: tuple[int, ...]):
    family = strength2_family(*shape[1:])
    ids = np.sort(np.ravel_multi_index(family.transpose(2, 0, 1), shape), axis=1)
    ids = np.unique(ids, axis=0)
    membership = np.zeros((len(ids), int(np.prod(shape))), dtype=np.uint8)
    membership[np.arange(len(ids))[:, None], ids] = 1
    adjacency = membership @ membership.T == 0
    degree = adjacency.sum(axis=1)
    if len(ids) > 1 and not np.all(degree == degree[0]):
        raise AssertionError("disjointness graph is not regular")
    neighbors = tuple(np.flatnonzero(row) for row in adjacency)
    return ids, membership, adjacency, neighbors


def sample_disjoint_blocks(
    shape: tuple[int, ...], rng: np.random.Generator
) -> np.ndarray:
    ids, _, _, neighbors = cover_graph(shape)
    output = np.empty((DRAWS, 4, 16), dtype=int)
    for pair in range(2):
        first = rng.integers(0, len(ids), size=DRAWS)
        if len(ids) == 1:
            second = first
        else:
            degree = len(neighbors[0])
            positions = rng.integers(0, degree, size=DRAWS)
            second = np.fromiter(
                (neighbors[index][position] for index, position in zip(first, positions)),
                dtype=int, count=DRAWS,
            )
        output[:, 2 * pair] = ids[first]
        output[:, 2 * pair + 1] = ids[second]
    return output


def sample_independent_blocks(
    shape: tuple[int, ...], rng: np.random.Generator
) -> np.ndarray:
    ids = cover_graph(shape)[0]
    chosen = rng.integers(0, len(ids), size=(DRAWS, 4))
    return ids[chosen]


def antithetic_score(
    y: np.ndarray, flat: np.ndarray, blocks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    left_ids = np.concatenate((blocks[:, 0], blocks[:, 1]), axis=1)
    right_ids = np.concatenate((blocks[:, 2], blocks[:, 3]), axis=1)
    return CQS.cross_and_mean_scores(y, flat, left_ids, right_ids)


def graph_theory(shape: tuple[int, ...]) -> dict[str, object]:
    ids, membership_u8, adjacency, _ = cover_graph(shape)
    membership = membership_u8.astype(np.float64)
    cells = membership.shape[1]
    uniform = np.full(cells, 1 / cells)
    single_second = membership.T @ membership / (len(ids) * 16 ** 2)
    single_cov = single_second - np.outer(uniform, uniform)
    degree = int(adjacency.sum(axis=1)[0]) if len(ids) > 1 else 0
    if degree:
        cross_second = (
            membership.T @ (adjacency.astype(np.float64) @ membership)
            / (len(ids) * degree * 32 ** 2)
        )
        pair_second = 2 * single_second / 4 + 2 * cross_second
        pair_cov = pair_second - np.outer(uniform, uniform)
    else:
        pair_cov = single_cov
    single = component_coefficients(single_cov, shape)
    pair = component_coefficients(pair_cov, shape)
    return {
        "shape": list(shape), "unique_covers": len(ids), "disjoint_degree": degree,
        "single_cover_coefficients": single,
        "disjoint_pair_mean_coefficients": pair,
        "pair_to_single_ratios": {
            key: (pair[key] / value if abs(value) > 1e-15 else 0.0)
            for key, value in single.items()
        },
    }


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
    anti_blocks = sample_disjoint_blocks(
        shape, np.random.default_rng(RMS.stable_seed("disjoint-pairs", panel, dataset))
    )
    independent_blocks = sample_independent_blocks(
        shape, np.random.default_rng(RMS.stable_seed("independent-four", panel, dataset))
    )
    quotient_val = np.asarray([
        proper_loss(validation_y, values.mean(axis=0)) for values in validation
    ])
    quotient_test = np.asarray([
        proper_loss(test_y, values.mean(axis=0)) for values in test
    ])
    winner = int(np.argmin(quotient_val))
    methods = ("disjoint_pair_cross64", "independent_block_u64")
    val_scores = {method: [] for method in methods}
    prediction_losses = {method: [] for method in methods}
    test_losses = {method: [] for method in methods}
    calibration = []
    for model, val_flat, test_flat, exact in zip(models, validation, test, quotient_val):
        anti_score, anti_mean = antithetic_score(validation_y, val_flat, anti_blocks)
        independent_score, independent_mean = cover_block_scores(
            validation_y, val_flat, independent_blocks
        )
        _, anti_test = antithetic_score(test_y, test_flat, anti_blocks)
        _, independent_test = cover_block_scores(test_y, test_flat, independent_blocks)
        for method, score, mean_loss, test_loss in (
            (methods[0], anti_score, anti_mean, anti_test),
            (methods[1], independent_score, independent_mean, independent_test),
        ):
            val_scores[method].append(score)
            prediction_losses[method].append(mean_loss)
            test_losses[method].append(test_loss)
            bias = float(score.mean() - exact)
            calibration.append({
                "panel": panel, "dataset": dataset, "task": task, "model": model,
                "method": method, "score_bias": bias,
                "score_rmse": float(np.sqrt(score.var(ddof=1) + bias ** 2)),
                "prediction_residual": float(mean_loss.mean() - exact),
            })
    rows = []
    for method in methods:
        score_matrix = np.stack(val_scores[method], axis=1)
        prediction_matrix = np.stack(prediction_losses[method], axis=1)
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
                "mean_prediction_residual": float(np.mean(prediction_matrix[draw] - quotient_val)),
            })
    return rows, calibration, shape


def main() -> None:
    rows, calibration_rows = [], []
    shapes: set[tuple[int, ...]] = set()
    for panel, config_name, directory_name in CQS.PANELS:
        config = json.loads((HERE / config_name).read_text())
        for dataset in config["datasets"]:
            current, calibration, shape = analyze_dataset(
                panel, dataset, config["models"], RESULTS / directory_name
            )
            rows.extend(current); calibration_rows.extend(calibration); shapes.add(shape)
    draws = pd.DataFrame(rows)
    draws.to_csv(RESULTS / "disjoint_pair_cross_draws.csv", index=False)
    cells = draws.groupby(["panel", "dataset", "task", "method"], as_index=False).mean(numeric_only=True)
    cells.to_csv(RESULTS / "disjoint_pair_cross_cells.csv", index=False)
    calibration = pd.DataFrame(calibration_rows)
    calibration.to_csv(RESULTS / "disjoint_pair_cross_calibration.csv", index=False)

    summary: dict[str, object] = {
        "status": "complete", "draws_per_dataset": DRAWS,
        "cover_graphs": [graph_theory(shape) for shape in sorted(shapes)],
        "panels": {},
    }
    counts = {"rmse": 0, "agreement": 0, "regret": 0, "residual": 0}
    for panel, current in cells.groupby("panel"):
        means = current.groupby("method").mean(numeric_only=True)
        rmses = calibration[calibration.panel == panel].groupby("method").score_rmse.mean()
        anti, control = means.loc["disjoint_pair_cross64"], means.loc["independent_block_u64"]
        clauses = {
            "score_rmse_lower": bool(rmses.disjoint_pair_cross64 < rmses.independent_block_u64),
            "agreement_nolower": bool(anti.selection_agreement >= control.selection_agreement),
            "regret_nohigher": bool(anti.validation_quotient_regret <= control.validation_quotient_regret),
            "prediction_residual_lower": bool(anti.mean_prediction_residual < control.mean_prediction_residual),
        }
        for key, value in clauses.items():
            counts[{"score_rmse_lower": "rmse", "agreement_nolower": "agreement",
                    "regret_nohigher": "regret", "prediction_residual_lower": "residual"}[key]] += int(value)
        summary["panels"][panel] = {
            "clauses": clauses, "score_rmse": rmses.to_dict(),
            "method_means": means.reset_index().to_dict(orient="records"),
        }
    summary["panels_passing_by_clause"] = counts
    summary["frozen_gate_passed"] = bool(all(value >= 4 for value in counts.values()))
    (RESULTS / "disjoint_pair_cross_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
