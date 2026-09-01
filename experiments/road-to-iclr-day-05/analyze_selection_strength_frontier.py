"""Strength-1/2/3 nuisance-cover frontier for downstream model selection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_robust_model_selection as RMS
from analyze_strength2_cover import proper_loss, strength1_family, strength2_family
from analyze_strength3_cover import strength3_family


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DRAWS = 512
METHODS = (
    ("strength1_b4", 4), ("iid_b4", 4), ("seed_blocks_b4", 4),
    ("strength2_b16", 16), ("iid_b16", 16), ("four_strength1_b16", 16),
    ("seed_blocks_b16", 16),
    ("strength3_b64", 64), ("iid_b64", 64), ("four_strength2_b64", 64),
    ("sixteen_strength1_b64", 64), ("seed_blocks_b64", 64),
)
PANELS = RMS.PANELS + ((
    "openml_taskbalanced", "openml_taskbalanced_cover_config.json",
    "openml_taskbalanced_cover",
),)
ORIGINAL_PANELS = {panel for panel, _, _ in RMS.PANELS}


def flatten(coordinates: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    return np.ravel_multi_index(np.moveaxis(coordinates, -1, 0), shape)


def seed_blocks(shape: tuple[int, ...], draws: int, blocks: int, rng: np.random.Generator) -> np.ndarray:
    seed_levels = shape[-1]
    schema = np.column_stack([
        rng.integers(0, size, size=draws * blocks) for size in shape[:3]
    ]).reshape(draws, blocks, 3)
    values = np.empty((draws, blocks, seed_levels, 4), dtype=int)
    values[..., :3] = schema[:, :, None, :]
    values[..., 3] = np.arange(seed_levels)[None, None, :]
    return flatten(values.reshape(draws, blocks * seed_levels, 4), shape)


def frontier_actions(shape: tuple[int, int, int, int], seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    key = shape[1:]
    families = {
        1: strength1_family(*key), 2: strength2_family(*key), 3: strength3_family(*key)
    }
    def sample_family(strength: int, blocks: int) -> np.ndarray:
        family = families[strength]
        coordinates = family[rng.integers(0, len(family), size=(DRAWS, blocks))]
        return flatten(coordinates.reshape(DRAWS, -1, 4), shape)
    return {
        "strength1_b4": sample_family(1, 1),
        "iid_b4": rng.integers(0, np.prod(shape), size=(DRAWS, 4)),
        "seed_blocks_b4": seed_blocks(shape, DRAWS, 1, rng),
        "strength2_b16": sample_family(2, 1),
        "iid_b16": rng.integers(0, np.prod(shape), size=(DRAWS, 16)),
        "four_strength1_b16": sample_family(1, 4),
        "seed_blocks_b16": seed_blocks(shape, DRAWS, 4, rng),
        "strength3_b64": sample_family(3, 1),
        "iid_b64": rng.integers(0, np.prod(shape), size=(DRAWS, 64)),
        "four_strength2_b64": sample_family(2, 4),
        "sixteen_strength1_b64": sample_family(1, 16),
        "seed_blocks_b64": seed_blocks(shape, DRAWS, 16, rng),
    }


def analyze_dataset(panel: str, dataset: str, models: list[str], directory: Path) -> list[dict]:
    validation, test = [], []
    val_y = test_y = None
    shape = None
    for model in models:
        archive = np.load(directory / f"{dataset}__{model}.npz")
        current = tuple(archive["validation_predictions"].shape[:4])
        shape = current if shape is None else shape
        if current != shape:
            raise AssertionError("candidate factor shapes differ")
        val_y = archive["validation_y"] if val_y is None else val_y
        test_y = archive["test_y"] if test_y is None else test_y
        validation.append(archive["validation_predictions"].reshape((-1,) + archive["validation_predictions"].shape[-2:]).astype(np.float64))
        test.append(archive["test_predictions"].reshape((-1,) + archive["test_predictions"].shape[-2:]).astype(np.float64))
    assert shape is not None and val_y is not None and test_y is not None
    quotient_val = np.asarray([proper_loss(val_y, values.mean(axis=0)) for values in validation])
    quotient_test = np.asarray([proper_loss(test_y, values.mean(axis=0)) for values in test])
    winner = int(np.argmin(quotient_val))
    actions = frontier_actions(shape, RMS.stable_seed("frontier", panel, dataset))
    rows = []
    for method, budget in METHODS:
        ids = actions[method]
        val_losses = np.stack([RMS.batched_losses(val_y, values, ids) for values in validation], axis=1)
        selected = np.argmin(val_losses, axis=1)
        test_losses = np.stack([RMS.batched_losses(test_y, values, ids) for values in test], axis=1)
        rows.append({
            "panel": panel, "dataset": dataset, "method": method, "budget": budget,
            "strength": 1 if method == "strength1_b4" else 2 if method == "strength2_b16" else 3 if method == "strength3_b64" else 0,
            "selection_agreement": float(np.mean(selected == winner)),
            "validation_quotient_regret": float(np.mean(quotient_val[selected] - quotient_val[winner])),
            "selected_quotient_test_loss": float(np.mean(quotient_test[selected])),
            "selected_realized_test_loss": float(np.mean(test_losses[np.arange(DRAWS), selected])),
            "selection_entropy_bits": RMS.entropy(selected, len(models)),
        })
    full_budget = int(np.prod(shape))
    rows.append({
        "panel": panel, "dataset": dataset, "method": "full_quotient", "budget": full_budget,
        "strength": 4, "selection_agreement": 1.0, "validation_quotient_regret": 0.0,
        "selected_quotient_test_loss": float(quotient_test[winner]),
        "selected_realized_test_loss": float(quotient_test[winner]), "selection_entropy_bits": 0.0,
    })
    return rows


def main() -> None:
    rows = []
    for panel, config_name, directory in PANELS:
        config = json.loads((HERE / config_name).read_text())
        for dataset in config["datasets"]:
            rows.extend(analyze_dataset(panel, dataset, config["models"], RESULTS / directory))
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "selection_strength_frontier_cells.csv", index=False)
    # Full-enumeration budgets differ across sources when nuisance factors are
    # degenerate.  Keep those exact rows in the cell table, but do not average
    # incomparable source subsets into the cross-source frontier summary.
    means = frame[frame.method != "full_quotient"].groupby(
        ["panel", "method", "budget", "strength"], as_index=False
    ).mean(numeric_only=True)
    means.to_csv(RESULTS / "selection_strength_frontier_means.csv", index=False)
    panels = {}
    for panel, current in means.groupby("panel"):
        indexed = current.set_index("method")
        cover_names = ["strength1_b4", "strength2_b16", "strength3_b64"]
        iid_names = ["iid_b4", "iid_b16", "iid_b64"]
        agreements = indexed.loc[cover_names, "selection_agreement"].to_numpy()
        regrets = indexed.loc[cover_names, "validation_quotient_regret"].to_numpy()
        iid_regrets = indexed.loc[iid_names, "validation_quotient_regret"].to_numpy()
        clauses = {
            "cover_agreement_nondecreasing": bool(np.all(np.diff(agreements) >= -1e-15)),
            "cover_regret_nonincreasing": bool(np.all(np.diff(regrets) <= 1e-15)),
            "each_cover_regret_lower_than_iid": bool(np.all(regrets < iid_regrets)),
        }
        panels[panel] = {
            "clauses": clauses, "passed": bool(all(clauses.values())),
            "method_means": current.to_dict(orient="records"),
        }
    panel_passes = sum(
        value["passed"] for panel, value in panels.items() if panel in ORIGINAL_PANELS
    )
    summary = {
        "status": "complete", "draws_per_dataset": DRAWS, "panels": panels,
        "panels_passing_all_clauses": panel_passes,
        "frozen_hierarchy_gate_passed": panel_passes >= 2,
        "taskbalanced_postgate_addendum_passed": panels["openml_taskbalanced"]["passed"],
    }
    (RESULTS / "selection_strength_frontier_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
