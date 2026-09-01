"""Equal-compute model selection under randomized nuisance-cover actions."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from analyze_strength2_cover import proper_loss, strength1_family, strength2_family


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DRAWS = 1024
BOOTSTRAP_DRAWS = 100_000
METHODS = ("strength2", "iid16", "srswor16", "four_strength1", "four_seed_blocks")
QMC_METHODS = ("sobol16", "lhs16")
ALL_METHODS = METHODS + QMC_METHODS
PANELS = (
    ("confirmation", "tier1_confirmation_config.json", "tier1_confirmation"),
    ("menu_repeat", "tier1_menu_repeat_config.json", "tier1_menu_repeat"),
    ("subsample_repeat", "tier1_subsample_repeat_config.json", "tier1_subsample_repeat"),
)


def stable_seed(*parts: str) -> int:
    payload = ":".join(parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def action_ids(shape: tuple[int, int, int, int], seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    category, label, seed_levels = shape[1:]
    family2 = strength2_family(category, label, seed_levels)
    family1 = strength1_family(category, label, seed_levels)
    selected2 = family2[rng.integers(0, len(family2), size=DRAWS)]
    selected1 = family1[rng.integers(0, len(family1), size=(DRAWS, 4))]
    iid = rng.integers(0, math.prod(shape), size=(DRAWS, 16))
    blocks_per_draw = 16 // seed_levels
    schema = np.column_stack([
        rng.integers(0, size, size=DRAWS * blocks_per_draw) for size in shape[:3]
    ]).reshape(DRAWS, blocks_per_draw, 3)
    blocks = np.empty((DRAWS, blocks_per_draw, seed_levels, 4), dtype=int)
    blocks[..., :3] = schema[:, :, None, :]
    blocks[..., 3] = np.arange(seed_levels)[None, None, :]
    # A separate stream preserves every pre-addendum action exactly.
    srs_rng = np.random.default_rng(seed ^ 0x535253574F52)
    population = math.prod(shape)
    srswor = np.stack([srs_rng.choice(population, size=16, replace=False) for _ in range(DRAWS)])

    def qmc_ids(kind: str) -> np.ndarray:
        output = np.empty((DRAWS, 16), dtype=int)
        salt = 0x534F424F4C if kind == "sobol16" else 0x4C4853
        levels = np.asarray(shape)
        for draw in range(DRAWS):
            qseed = int((seed ^ salt ^ draw) & 0xFFFFFFFF)
            points = (
                qmc.Sobol(d=4, scramble=True, seed=qseed).random_base2(4)
                if kind == "sobol16"
                else qmc.LatinHypercube(d=4, scramble=True, seed=qseed).random(16)
            )
            coordinates = np.minimum((points * levels[None]).astype(int), levels - 1)
            output[draw] = np.ravel_multi_index(coordinates.T, shape)
        return output

    def flatten(coordinates: np.ndarray) -> np.ndarray:
        reshaped = coordinates.reshape(DRAWS, 16, 4)
        return np.ravel_multi_index(np.moveaxis(reshaped, -1, 0), shape)

    return {
        "strength2": flatten(selected2),
        "iid16": iid,
        "srswor16": srswor,
        "four_strength1": flatten(selected1),
        "four_seed_blocks": flatten(blocks),
        "sobol16": qmc_ids("sobol16"),
        "lhs16": qmc_ids("lhs16"),
    }


def batched_losses(y: np.ndarray, flat: np.ndarray, ids: np.ndarray, batch: int = 32) -> np.ndarray:
    output = np.empty(len(ids), dtype=np.float64)
    for start in range(0, len(ids), batch):
        stop = min(len(ids), start + batch)
        predictions = flat[ids[start:stop]].mean(axis=1)
        if predictions.shape[-1] == 1:
            output[start:stop] = np.mean((predictions[..., 0] - y[None, :]) ** 2, axis=1)
        else:
            targets = np.eye(predictions.shape[-1])[y.astype(int)]
            output[start:stop] = np.mean(np.sum((predictions - targets[None, ...]) ** 2, axis=-1), axis=1)
    return output


def entropy(values: np.ndarray, levels: int) -> float:
    probabilities = np.bincount(values, minlength=levels) / len(values)
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def ranking_fidelity(estimated: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stable-tie ranking correlation and pairwise-order accuracy per draw."""
    levels = len(truth)
    truth_ranks = np.argsort(np.argsort(truth, kind="stable"), kind="stable").astype(float)
    estimated_ranks = np.argsort(
        np.argsort(estimated, axis=1, kind="stable"), axis=1, kind="stable"
    ).astype(float)
    centered_truth = truth_ranks - truth_ranks.mean()
    centered_estimated = estimated_ranks - estimated_ranks.mean(axis=1, keepdims=True)
    denominator = np.sqrt(
        np.sum(centered_truth ** 2) * np.sum(centered_estimated ** 2, axis=1)
    )
    correlation = np.sum(centered_estimated * centered_truth[None], axis=1) / denominator
    pair_correct = np.zeros(len(estimated), dtype=float)
    pairs = 0
    for left in range(levels):
        for right in range(left + 1, levels):
            pair_correct += (
                (estimated[:, left] < estimated[:, right])
                == (truth[left] < truth[right])
            )
            pairs += 1
    return correlation, pair_correct / pairs


def analyze_dataset(panel: str, dataset: str, models: list[str], input_dir: Path) -> list[dict]:
    validation, test, validation_y, test_y = [], [], None, None
    shape = None
    task = None
    for model in models:
        archive = np.load(input_dir / f"{dataset}__{model}.npz")
        manifest = json.loads((input_dir / f"{dataset}__{model}.json").read_text())
        current_shape = tuple(int(value) for value in archive["validation_predictions"].shape[:4])
        if shape is None:
            shape = current_shape
            validation_y = archive["validation_y"]
            test_y = archive["test_y"]
            task = manifest["task"]
        if current_shape != shape:
            raise AssertionError(f"factor shape mismatch for {panel}/{dataset}/{model}")
        if not np.array_equal(validation_y, archive["validation_y"]):
            raise AssertionError("validation labels differ across candidates")
        if not np.array_equal(test_y, archive["test_y"]):
            raise AssertionError("test labels differ across candidates")
        validation.append(archive["validation_predictions"].reshape((-1,) + archive["validation_predictions"].shape[-2:]).astype(np.float64))
        test.append(archive["test_predictions"].reshape((-1,) + archive["test_predictions"].shape[-2:]).astype(np.float64))
    assert shape is not None and validation_y is not None and test_y is not None and task is not None
    quotient_validation_loss = np.asarray([proper_loss(validation_y, values.mean(axis=0)) for values in validation])
    quotient_test_loss = np.asarray([proper_loss(test_y, values.mean(axis=0)) for values in test])
    validation_winner = int(np.argmin(quotient_validation_loss))
    test_oracle_loss = float(np.min(quotient_test_loss))
    actions = action_ids(shape, stable_seed("model-selection", panel, dataset))
    rows = []
    for method, ids in actions.items():
        candidate_validation = np.stack([
            batched_losses(validation_y, values, ids) for values in validation
        ], axis=1)
        rank_correlation, pairwise_accuracy = ranking_fidelity(
            candidate_validation, quotient_validation_loss
        )
        selected = np.argmin(candidate_validation, axis=1)
        candidate_test = np.stack([
            batched_losses(test_y, values, ids) for values in test
        ], axis=1)
        realized_test = candidate_test[np.arange(DRAWS), selected]
        for draw in range(DRAWS):
            chosen = int(selected[draw])
            rows.append({
                "panel": panel, "dataset": dataset, "task": task, "method": method,
                "draw": draw, "selected_model": models[chosen],
                "selected_realized_validation_loss": float(candidate_validation[draw, chosen]),
                "selected_realized_test_loss": float(realized_test[draw]),
                "selected_quotient_validation_loss": float(quotient_validation_loss[chosen]),
                "selected_quotient_test_loss": float(quotient_test_loss[chosen]),
                "validation_quotient_regret": float(quotient_validation_loss[chosen] - quotient_validation_loss[validation_winner]),
                "test_quotient_regret": float(quotient_test_loss[chosen] - test_oracle_loss),
                "agrees_validation_quotient_winner": chosen == validation_winner,
                "validation_rank_spearman": float(rank_correlation[draw]),
                "validation_pairwise_order_accuracy": float(pairwise_accuracy[draw]),
            })
        rows[-1]["selection_entropy_bits"] = entropy(selected, len(models))
    return rows


def cluster_interval(differences: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(differences), size=(BOOTSTRAP_DRAWS, len(differences)))
    values = differences[draws].mean(axis=1)
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def main() -> None:
    rows = []
    for panel, config_name, directory in PANELS:
        config = json.loads((HERE / config_name).read_text())
        for dataset in config["datasets"]:
            rows.extend(analyze_dataset(panel, dataset, config["models"], RESULTS / directory))
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "robust_model_selection_draws.csv", index=False)
    aggregate = frame.groupby(["panel", "dataset", "task", "method"], as_index=False).agg(
        selected_realized_test_loss=("selected_realized_test_loss", "mean"),
        selected_quotient_test_loss=("selected_quotient_test_loss", "mean"),
        validation_quotient_regret=("validation_quotient_regret", "mean"),
        test_quotient_regret=("test_quotient_regret", "mean"),
        selection_agreement=("agrees_validation_quotient_winner", "mean"),
    )
    entropies = frame.dropna(subset=["selection_entropy_bits"])[
        ["panel", "dataset", "method", "selection_entropy_bits"]
    ]
    aggregate = aggregate.merge(entropies, on=["panel", "dataset", "method"], how="left")
    aggregate.to_csv(RESULTS / "robust_model_selection_datasets.csv", index=False)
    summary = {"status": "complete", "draws_per_dataset": DRAWS, "panels": {}}
    for panel, current in aggregate.groupby("panel"):
        pivot = current.pivot(index="dataset", columns="method", values="selected_realized_test_loss")
        records = {}
        for control in ALL_METHODS[1:]:
            differences = (pivot.strength2 - pivot[control]).to_numpy()
            records[control] = {
                "mean_strength2_minus_control_test_loss": float(differences.mean()),
                "datasets_strength2_lower": int(np.sum(differences < 0)),
                "datasets": len(differences),
                "dataset_cluster_bootstrap_95_interval": cluster_interval(
                    differences, stable_seed("bootstrap", panel, control)
                ),
            }
        means = current.groupby("method").mean(numeric_only=True).to_dict(orient="index")
        gate = all(
            record["mean_strength2_minus_control_test_loss"] < 0
            and record["datasets_strength2_lower"] / record["datasets"] >= 0.6
            for control, record in records.items() if control in METHODS[1:]
        )
        qmc_gate = all(
            records[control]["mean_strength2_minus_control_test_loss"] < 0
            and records[control]["datasets_strength2_lower"] / records[control]["datasets"] >= 0.6
            for control in QMC_METHODS
        )
        summary["panels"][panel] = {
            "datasets": int(current.dataset.nunique()),
            "mean_metrics_by_method": means,
            "strength2_comparisons": records,
            "task_stratified_strength2_vs_iid16": {
                task: {
                    "datasets": int(task_frame.dataset.nunique()),
                    "mean_strength2_minus_iid16_test_loss": float(
                        task_frame.pivot(index="dataset", columns="method", values="selected_realized_test_loss")
                        .eval("strength2 - iid16").mean()
                    ),
                    "datasets_strength2_lower": int((
                        task_frame.pivot(index="dataset", columns="method", values="selected_realized_test_loss")
                        .eval("strength2 - iid16") < 0
                    ).sum()),
                }
                for task, task_frame in current.groupby("task")
            },
            "frozen_panel_gate_passed": bool(gate),
            "frozen_qmc_selection_addendum_passed": bool(qmc_gate),
        }
    (RESULTS / "robust_model_selection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
