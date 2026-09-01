"""Cross-score versus IID-U selection over paired evaluation repartitions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_cross_quotient_selection as CQS
import analyze_repeated_holdout_shift as RHS
import analyze_robust_model_selection as RMS


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DRAWS = 1_024
BATCH = 8
METHODS = ("strength2_cross32", "iid_u32")
PANELS = RHS.PANELS


def make_masks(
    pooled_y: np.ndarray, validation_y: np.ndarray, classification: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    masks = np.zeros((DRAWS, len(pooled_y)), dtype=bool)
    all_indices = np.arange(len(pooled_y))
    for draw in range(DRAWS):
        if classification:
            selected = RHS.stratified_indices(pooled_y, validation_y, rng)
        else:
            selected = rng.choice(all_indices, size=len(validation_y), replace=False)
        masks[draw, selected] = True
    return masks


def target_array(y: np.ndarray, width: int) -> np.ndarray:
    if width == 1:
        return y.astype(np.float64)[:, None]
    return np.eye(width, dtype=np.float64)[y.astype(int)]


def validation_action_scores(
    y: np.ndarray, flat: np.ndarray, masks: np.ndarray,
    left: np.ndarray, right: np.ndarray | None = None,
) -> np.ndarray:
    """Paired-draw validation means for a cross score or complete IID U-score."""
    output = np.empty(len(masks), dtype=np.float64)
    target = target_array(y, flat.shape[-1])
    counts = masks.sum(axis=1)
    for start in range(0, len(masks), BATCH):
        stop = min(start + BATCH, len(masks))
        if right is not None:
            pred_left = flat[left[start:stop]].mean(axis=1)
            pred_right = flat[right[start:stop]].mean(axis=1)
            row_score = np.sum(
                (target[None] - pred_left) * (target[None] - pred_right), axis=-1
            )
        else:
            members = flat[left[start:stop]]
            residual = target[None, None] - members
            summed = residual.sum(axis=1)
            numerator = np.sum(summed ** 2, axis=-1) - np.sum(
                residual ** 2, axis=(1, 3)
            )
            count = members.shape[1]
            row_score = numerator / (count * (count - 1))
        output[start:stop] = np.sum(row_score * masks[start:stop], axis=1) / counts[start:stop]
    return output


def load_dataset(
    directory: Path, dataset: str, models: list[str]
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, str, tuple[int, ...]]:
    candidates: list[np.ndarray] = []
    validation_y = test_y = None
    task = ""
    shape = None
    for model in models:
        archive = np.load(directory / f"{dataset}__{model}.npz")
        manifest = json.loads((directory / f"{dataset}__{model}.json").read_text())
        if validation_y is not None and not np.array_equal(validation_y, archive["validation_y"]):
            raise AssertionError("validation labels differ")
        if test_y is not None and not np.array_equal(test_y, archive["test_y"]):
            raise AssertionError("test labels differ")
        validation_y, test_y = archive["validation_y"], archive["test_y"]
        task = str(manifest["task"])
        current_shape = tuple(int(value) for value in archive["validation_predictions"].shape[:4])
        if shape is not None and current_shape != shape:
            raise AssertionError("factor shape mismatch")
        shape = current_shape
        validation = archive["validation_predictions"].reshape(
            (-1,) + archive["validation_predictions"].shape[-2:]
        ).astype(np.float64)
        test = archive["test_predictions"].reshape(
            (-1,) + archive["test_predictions"].shape[-2:]
        ).astype(np.float64)
        candidates.append(np.concatenate((validation, test), axis=1))
    assert validation_y is not None and test_y is not None and shape is not None
    return candidates, validation_y, test_y, task, shape


def analyze_dataset(
    panel: str, dataset: str, models: list[str], directory: Path
) -> list[dict[str, object]]:
    candidates, validation_y, test_y, task, shape = load_dataset(directory, dataset, models)
    pooled_y = np.concatenate((validation_y, test_y))
    masks = make_masks(
        pooled_y, validation_y, task in {"binclass", "multiclass"},
        np.random.default_rng(RMS.stable_seed("repartitioned-cross", panel, dataset)),
    )
    ids = CQS.action_ids(shape, panel, dataset)
    iid_ids = np.concatenate(ids["iid"], axis=1)
    scores = {method: [] for method in METHODS}
    quotient_rows = []
    for flat in candidates:
        scores["strength2_cross32"].append(
            validation_action_scores(pooled_y, flat, masks, *ids["strength2"])
        )
        scores["iid_u32"].append(
            validation_action_scores(pooled_y, flat, masks, iid_ids)
        )
        quotient_rows.append(RHS.row_losses(pooled_y, flat.mean(axis=0)))
    score_matrices = {key: np.stack(value, axis=1) for key, value in scores.items()}
    quotient_rows_array = np.stack(quotient_rows, axis=1)
    validation_counts = masks.sum(axis=1)
    test_counts = len(pooled_y) - validation_counts
    exact_validation = masks.astype(np.float64) @ quotient_rows_array / validation_counts[:, None]
    exact_test = (
        quotient_rows_array.sum(axis=0)[None] - masks.astype(np.float64) @ quotient_rows_array
    ) / test_counts[:, None]
    validation_winner = np.argmin(exact_validation, axis=1)
    test_winner = np.argmin(exact_test, axis=1)
    rows: list[dict[str, object]] = []
    for method in METHODS:
        selected = np.argmin(score_matrices[method], axis=1)
        for draw, candidate in enumerate(selected):
            candidate = int(candidate)
            rows.append({
                "panel": panel, "dataset": dataset, "task": task,
                "method": method, "draw": draw,
                "selected_model": models[candidate],
                "validation_agreement": bool(candidate == validation_winner[draw]),
                "test_agreement": bool(candidate == test_winner[draw]),
                "validation_quotient_regret": float(
                    exact_validation[draw, candidate] - exact_validation[draw, validation_winner[draw]]
                ),
                "test_quotient_regret": float(
                    exact_test[draw, candidate] - exact_test[draw, test_winner[draw]]
                ),
                "selected_test_quotient_loss": float(exact_test[draw, candidate]),
                "validation_test_winner_agreement": bool(validation_winner[draw] == test_winner[draw]),
            })
    return rows


def main() -> None:
    rows: list[dict[str, object]] = []
    for panel, config_name, directory_name in PANELS:
        config = json.loads((HERE / config_name).read_text())
        for dataset in config["datasets"]:
            rows.extend(analyze_dataset(
                panel, dataset, config["models"], RESULTS / directory_name
            ))
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "repartitioned_cross_selection_draws.csv", index=False)
    cells = frame.groupby(["panel", "dataset", "task", "method"], as_index=False).agg(
        validation_agreement=("validation_agreement", "mean"),
        test_agreement=("test_agreement", "mean"),
        validation_quotient_regret=("validation_quotient_regret", "mean"),
        test_quotient_regret=("test_quotient_regret", "mean"),
        selected_test_quotient_loss=("selected_test_quotient_loss", "mean"),
        validation_test_winner_agreement=("validation_test_winner_agreement", "mean"),
    )
    cells.to_csv(RESULTS / "repartitioned_cross_selection_cells.csv", index=False)
    panels: dict[str, object] = {}
    clauses = {"validation": 0, "test": 0, "source": 0}
    for panel, current in cells.groupby("panel"):
        means = current.groupby("method").mean(numeric_only=True)
        cover = means.loc["strength2_cross32"]
        iid = means.loc["iid_u32"]
        pivot = current.pivot(index="dataset", columns="method", values="test_quotient_regret")
        favorable_sources = int((pivot.strength2_cross32 < pivot.iid_u32).sum())
        validation_better = bool(cover.validation_quotient_regret < iid.validation_quotient_regret)
        test_better = bool(cover.test_quotient_regret < iid.test_quotient_regret)
        source_clause = bool(favorable_sources >= 4)
        clauses["validation"] += validation_better
        clauses["test"] += test_better
        clauses["source"] += source_clause
        panels[panel] = {
            "means": means.reset_index().to_dict(orient="records"),
            "cover_validation_regret_below_iid": validation_better,
            "cover_test_regret_below_iid": test_better,
            "sources_with_lower_cover_test_regret": favorable_sources,
            "source_clause": source_clause,
        }
    if clauses["validation"] < 2:
        interpretation = "method_failure"
    elif clauses["test"] == 2 and clauses["source"] == 2:
        interpretation = "transfer_pass"
    else:
        interpretation = "validation_only_pass"
    summary = {
        "status": "complete", "draws_per_dataset": DRAWS,
        "frozen_interpretation": interpretation, "panels": panels,
    }
    (RESULTS / "repartitioned_cross_selection_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
