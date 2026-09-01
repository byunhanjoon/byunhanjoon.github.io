"""Consolidated exact-risk, ranking, selection, and matched-control analysis."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest, qmc, spearmanr
from analyze_strength2_cover import strength2_family as classical_strength2_family


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
NEURAL = RESULTS / "completion_neural"
CLASSICAL = RESULTS / "completion_classical"
CONFIG = json.loads((HERE / "completion_config.json").read_text())
MUL4 = np.asarray([[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]])
DRAWS = 512


def trace4(value: int) -> int:
    return int(value) ^ int(MUL4[value, value])


def strength2_base(cards: tuple[int, ...]) -> np.ndarray:
    rows = []
    for u in range(4):
        for v in range(4):
            rows.append((
                u,
                v if cards[1] == 4 else 0,
                trace4(u ^ v) if cards[2] == 2 else 0,
                trace4(int(MUL4[2, u]) ^ v) if cards[3] == 2 else 0,
                trace4(int(MUL4[3, u]) ^ int(MUL4[2, v])) if cards[4] == 2 else 0,
            ))
    base = np.asarray(rows, dtype=np.int16)
    assert_strength(base, cards, 2)
    return base


def strength3_base(cards: tuple[int, ...]) -> np.ndarray:
    rows = []
    for u in range(4):
        for v in range(4):
            for w in range(4):
                rows.append((
                    u,
                    v if cards[1] == 4 else 0,
                trace4(w) if cards[2] == 2 else 0,
                trace4(u ^ v ^ w) if cards[3] == 2 else 0,
                trace4(int(MUL4[2, u]) ^ int(MUL4[3, v]) ^ int(MUL4[2, w])) if cards[4] == 2 else 0,
                ))
    base = np.asarray(rows, dtype=np.int16)
    assert_strength(base, cards, 3)
    return base


def strength1_base(cards: tuple[int, ...]) -> np.ndarray:
    columns = [np.arange(4)]
    for card in cards[1:]:
        columns.append(np.arange(4) % card if card > 1 else np.zeros(4, dtype=int))
    base = np.stack(columns, axis=1).astype(np.int16)
    assert_strength(base, cards, 1)
    return base


def assert_strength(design: np.ndarray, cards: tuple[int, ...], strength: int) -> None:
    for order in range(1, strength + 1):
        for factors in itertools.combinations(range(len(cards)), order):
            if any(cards[index] == 1 for index in factors):
                continue
            counts = np.zeros(tuple(cards[index] for index in factors), dtype=int)
            for row in design:
                counts[tuple(row[index] for index in factors)] += 1
            if np.unique(counts).size != 1:
                raise AssertionError(f"unbalanced design {factors}: {np.unique(counts)}")


def randomize(base: np.ndarray, cards: tuple[int, ...], draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = np.empty((draws, len(base), len(cards)), dtype=np.int16)
    for draw in range(draws):
        current = base.copy()
        for factor, card in enumerate(cards):
            permutation = rng.permutation(card)
            current[:, factor] = permutation[current[:, factor]]
        rng.shuffle(current, axis=0)
        output[draw] = current
    return output


def action_ids(designs: np.ndarray, cards: tuple[int, ...]) -> np.ndarray:
    return np.ravel_multi_index(np.moveaxis(designs, -1, 0), cards)


def strength2_cover_family(cards: tuple[int, ...]) -> np.ndarray:
    """Enumerate the distinct level-relabelings of the frozen strength-2 OA."""
    base = strength2_base(cards)
    permutations = [list(itertools.permutations(range(card))) for card in cards]
    covers = []
    for current in itertools.product(*permutations):
        transformed = np.empty_like(base)
        for factor, permutation in enumerate(current):
            transformed[:, factor] = np.asarray(permutation)[base[:, factor]]
        covers.append(np.sort(action_ids(transformed, cards)))
    family = np.unique(np.asarray(covers, dtype=np.int64), axis=0)
    if not len(family):
        raise AssertionError("empty strength-2 cover family")
    return family


def packed_families(cards: tuple[int, ...], seed: int) -> dict[str, np.ndarray]:
    """Draw independent covers, disjoint pairs, and four-cover disjoint packs."""
    population = math.prod(cards)
    family = strength2_cover_family(cards)
    membership = np.zeros((len(family), population), dtype=np.uint8)
    membership[np.arange(len(family))[:, None], family] = 1
    adjacency = membership @ membership.T == 0
    neighbors = [np.flatnonzero(row) for row in adjacency]
    rng = np.random.default_rng(seed)
    independent32 = family[rng.integers(0, len(family), size=(DRAWS, 2))].reshape(DRAWS, 32)
    independent64 = family[rng.integers(0, len(family), size=(DRAWS, 4))].reshape(DRAWS, 64)
    if population <= 32:
        closure = np.tile(np.arange(population), math.ceil(32 / population))[:32]
        pair32 = np.tile(closure, (DRAWS, 1))
    else:
        pair32 = np.empty((DRAWS, 32), dtype=np.int64)
        for draw in range(DRAWS):
            first = int(rng.integers(0, len(family)))
            if not len(neighbors[first]):
                raise AssertionError(f"no disjoint cover pair for {cards}")
            second = int(rng.choice(neighbors[first]))
            pair32[draw] = np.concatenate((family[first], family[second]))
    if population <= 64:
        closure = np.tile(np.arange(population), math.ceil(64 / population))[:64]
        pack64 = np.tile(closure, (DRAWS, 1))
    else:
        pack64 = np.empty((DRAWS, 64), dtype=np.int64)
        for draw in range(DRAWS):
            for _ in range(10_000):
                chosen = [int(rng.integers(0, len(family)))]
                while len(chosen) < 4:
                    valid = np.ones(len(family), dtype=bool)
                    for old in chosen:
                        valid &= adjacency[old]
                    candidates = np.flatnonzero(valid)
                    if not len(candidates):
                        break
                    chosen.append(int(rng.choice(candidates)))
                if len(chosen) == 4:
                    pack64[draw] = family[chosen].reshape(-1)
                    break
            else:
                raise AssertionError(f"could not construct four-cover pack for {cards}")
    return {
        "independent_cover32": independent32,
        "disjoint_pair32": pair32,
        "independent_cover64": independent64,
        "disjoint_pack64": pack64,
    }


def random_families(cards: tuple[int, ...], seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    population = math.prod(cards)
    families: dict[str, np.ndarray] = {}
    for budget in (1, 2, 4, 8, 16, 32, 64):
        families[f"iid{budget}"] = rng.integers(0, population, size=(DRAWS, budget))
        actual = min(budget, population)
        families[f"srswor{budget}"] = np.stack([
            rng.choice(population, actual, replace=False) for _ in range(DRAWS)
        ])
    randomized_s1 = [
        action_ids(randomize(strength1_base(cards), cards, DRAWS, seed + 10 + block), cards)
        for block in range(4)
    ]
    families["strength1_4"] = randomized_s1[0]
    s1_blocks = []
    for block in range(4):
        s1_blocks.append(randomized_s1[block])
    families["strength1_16"] = np.concatenate(s1_blocks, axis=1)
    families["strength2_16"] = action_ids(randomize(strength2_base(cards), cards, DRAWS, seed + 20), cards)
    families["strength3_64"] = action_ids(randomize(strength3_base(cards), cards, DRAWS, seed + 30), cards)
    lhs = np.empty((DRAWS, 16), dtype=np.int64)
    sobol = np.empty((DRAWS, 16), dtype=np.int64)
    levels = np.asarray(cards)
    for draw in range(DRAWS):
        lhs_points = qmc.LatinHypercube(d=len(cards), scramble=True, seed=seed + 1000 + draw).random(16)
        sobol_points = qmc.Sobol(d=len(cards), scramble=True, seed=seed + 2000 + draw).random_base2(4)
        lhs[draw] = action_ids(np.minimum((lhs_points * levels).astype(int), levels - 1), cards)
        sobol[draw] = action_ids(np.minimum((sobol_points * levels).astype(int), levels - 1), cards)
    families["lhs16"] = lhs; families["sobol16"] = sobol
    # Four schema representatives, each crossed with all init/order combinations.
    schema_cards = cards[:3]
    schema_population = math.prod(schema_cards)
    seed_only = np.empty((DRAWS, 16), dtype=np.int64)
    schema_only = np.empty((DRAWS, 16), dtype=np.int64)
    for draw in range(DRAWS):
        chosen_schema = rng.choice(schema_population, 4, replace=schema_population < 4)
        rows = []
        for schema_id in chosen_schema:
            schema = np.unravel_index(schema_id, schema_cards)
            for init in range(cards[3]):
                for order in range(cards[4]):
                    rows.append((*schema, init, order))
        seed_only[draw] = np.ravel_multi_index(np.asarray(rows).T, cards)
        schema_ids = rng.choice(schema_population, 16, replace=schema_population < 16)
        schema_rows = [(*np.unravel_index(value, schema_cards), 0, 0) for value in schema_ids]
        schema_only[draw] = np.ravel_multi_index(np.asarray(schema_rows).T, cards)
    families["seed_only16"] = seed_only; families["schema_only16"] = schema_only
    return families


def proper_loss(y: np.ndarray, prediction: np.ndarray) -> float:
    if prediction.shape[-1] == 1:
        return float(np.mean((prediction[:, 0] - y) ** 2))
    target = np.eye(prediction.shape[-1])[y.astype(int)]
    return float(np.mean(np.sum((prediction - target) ** 2, axis=-1)))


def loss_draws(y: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    if predictions.shape[-1] == 1:
        return np.mean((predictions[..., 0] - y) ** 2, axis=-1)
    target = np.eye(predictions.shape[-1])[y.astype(int)]
    return np.mean(np.sum((predictions - target[None]) ** 2, axis=-1), axis=-1)


def residual_draws(flat: np.ndarray, ids: np.ndarray, quotient: np.ndarray) -> np.ndarray:
    estimates = flat[ids].mean(axis=1)
    return np.mean(np.sum((estimates - quotient[None]) ** 2, axis=-1), axis=-1)


def decompose_array(values: np.ndarray) -> dict[tuple[int, ...], float]:
    grand = values.mean(axis=tuple(range(values.ndim - 2)), keepdims=True)
    components: dict[tuple[int, ...], np.ndarray] = {(): grand}
    factors = values.ndim - 2
    for order in range(1, factors + 1):
        for subset in itertools.combinations(range(factors), order):
            average_axes = tuple(index for index in range(factors) if index not in subset)
            marginal = values.mean(axis=average_axes, keepdims=True) if average_axes else values.copy()
            component = marginal.copy()
            for suborder in range(order):
                for lower in itertools.combinations(subset, suborder):
                    component = component - components[lower]
            components[subset] = component
    return {
        subset: float(np.mean(np.sum(component**2, axis=-1)))
        for subset, component in components.items() if subset
    }


def secondary_metrics(y: np.ndarray, prediction: np.ndarray, task: str) -> dict[str, float]:
    if task == "regression":
        residual = prediction[:, 0] - y
        denominator = float(np.sum((y - y.mean()) ** 2))
        return {
            "mse": float(np.mean(residual**2)), "rmse": float(np.sqrt(np.mean(residual**2))),
            "mae": float(np.mean(np.abs(residual))),
            "r2": float(1 - np.sum(residual**2) / denominator) if denominator else float("nan"),
        }
    probability = np.clip(prediction[:, 1], 1e-12, 1 - 1e-12)
    accuracy = float(np.mean((probability >= 0.5) == y))
    logloss = float(-np.mean(y * np.log(probability) + (1 - y) * np.log(1 - probability)))
    bins = np.minimum((probability * 10).astype(int), 9)
    ece = 0.0
    for index in range(10):
        mask = bins == index
        if mask.any():
            ece += mask.mean() * abs(probability[mask].mean() - y[mask].mean())
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y, probability))
    except ValueError:
        auc = float("nan")
    return {"brier": proper_loss(y, prediction), "logloss": logloss, "accuracy": accuracy, "auc": auc, "ece": float(ece)}


def load_neural(dataset: str, model: str, split_seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, ...], dict[str, Any]]:
    stem = f"{dataset}__{model}__split{split_seed}__broad"
    archive = np.load(NEURAL / f"{stem}.npz")
    manifest = json.loads((NEURAL / f"{stem}.json").read_text())
    actions = archive["actions"].astype(int)
    cards = tuple(int(actions[:, index].max() + 1) for index in range(actions.shape[1]))
    expected = math.prod(cards)
    if len(actions) != expected or len(np.unique(np.ravel_multi_index(actions.T, cards))) != expected:
        raise AssertionError(f"{stem} is not a complete product")
    order = np.argsort(np.ravel_multi_index(actions.T, cards))
    return (
        archive["validation_predictions"][order].astype(np.float64),
        archive["test_predictions"][order].astype(np.float64),
        archive["validation_y"], archive["test_y"], cards, manifest,
    )


def analyze_neural() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    risk_rows = []; metric_rows = []
    family_cache: dict[tuple[int, ...], dict[str, np.ndarray]] = {}
    packing_cache: dict[tuple[int, ...], dict[str, np.ndarray]] = {}
    tensors: dict[tuple[str, int], dict[str, Any]] = {}
    for dataset in CONFIG["datasets"]:
        for split_seed in CONFIG["split_seeds"]:
            candidates = {}
            for model in CONFIG["models"]:
                val, test, val_y, test_y, cards, manifest = load_neural(dataset, model, split_seed)
                shaped_test = test.reshape(cards + test.shape[-2:])
                flat_val = val.reshape((-1,) + val.shape[-2:]); flat_test = test.reshape((-1,) + test.shape[-2:])
                qv = flat_val.mean(0); qt = flat_test.mean(0)
                components = decompose_array(shaped_test)
                total = sum(components.values())
                by_order = {order: sum(value for subset, value in components.items() if len(subset) == order) for order in range(1, len(cards) + 1)}
                schema_factors = {0, 1, 2}
                stochastic_factors = {3, 4}
                schema_only = sum(
                    value for subset, value in components.items()
                    if set(subset) <= schema_factors
                )
                seed_only = sum(
                    value for subset, value in components.items()
                    if set(subset) <= stochastic_factors
                )
                schema_seed_interaction = total - schema_only - seed_only
                cache_key = cards
                if cache_key not in family_cache:
                    family_cache[cache_key] = random_families(cards, 2026082851 + sum(cards))
                    packing_cache[cache_key] = packed_families(cards, 2026082871 + sum(cards))
                families = family_cache[cache_key]
                record: dict[str, Any] = {
                    "dataset": dataset, "split_seed": split_seed, "model": model,
                    "task": manifest["task"], "population": math.prod(cards), "total_nuisance_variance": total,
                    "schema_only_variance": schema_only,
                    "seed_only_variance": seed_only,
                    "schema_seed_interaction_variance": schema_seed_interaction,
                    "main_fraction": by_order[1] / total if total else 0.0,
                    "main_pair_fraction": (by_order[1] + by_order[2]) / total if total else 0.0,
                    "triple_fraction": by_order[3] / total if total else 0.0,
                    "higher_fraction": sum(by_order[o] for o in range(4, len(cards) + 1)) / total if total else 0.0,
                    "quotient_validation_loss": proper_loss(val_y, qv), "quotient_test_loss": proper_loss(test_y, qt),
                    "wall_seconds": manifest["wall_seconds"], "peak_device_bytes": manifest["maximum_peak_device_bytes"],
                }
                for method, ids in families.items():
                    residual = residual_draws(flat_test, ids, qt)
                    record[f"{method}_residual_mean"] = float(residual.mean())
                    record[f"{method}_residual_median"] = float(np.median(residual))
                    record[f"{method}_residual_se"] = float(residual.std(ddof=1) / np.sqrt(len(residual)))
                for method, ids in packing_cache[cache_key].items():
                    residual = residual_draws(flat_test, ids, qt)
                    record[f"{method}_residual_mean"] = float(residual.mean())
                    record[f"{method}_residual_median"] = float(np.median(residual))
                    record[f"{method}_residual_se"] = float(residual.std(ddof=1) / np.sqrt(len(residual)))
                risk_rows.append(record)
                for split, y, quotient in (("validation", val_y, qv), ("test", test_y, qt)):
                    metric_rows.append({
                        "dataset": dataset, "split_seed": split_seed, "model": model,
                        "task": manifest["task"], "split": split,
                        "metric_scale": "standardized_target" if manifest["task"] == "regression" else "probability",
                        **secondary_metrics(y, quotient, manifest["task"]),
                    })
                candidates[model] = {"val": flat_val, "test": flat_test, "val_y": val_y, "test_y": test_y, "qv": qv, "qt": qt, "cards": cards, "families": families}
            tensors[(dataset, split_seed)] = candidates

    selection_rows = []; cross_rows = []
    for (dataset, split_seed), candidates in tensors.items():
        models = list(CONFIG["models"])
        exact_val = np.asarray([proper_loss(candidates[m]["val_y"], candidates[m]["qv"]) for m in models])
        exact_test = np.asarray([proper_loss(candidates[m]["test_y"], candidates[m]["qt"]) for m in models])
        winner = int(np.argmin(exact_val)); test_winner = int(np.argmin(exact_test))
        for method in (
            "iid4", "iid16", "iid64", "srswor16", "strength1_4", "strength1_16",
            "strength2_16", "strength3_64", "lhs16", "sobol16", "seed_only16", "schema_only16",
        ):
            score_matrix = []
            rank_correlations = []
            for model in models:
                current = candidates[model]
                estimates = current["val"][current["families"][method]].mean(axis=1)
                score_matrix.append(loss_draws(current["val_y"], estimates))
            scores = np.stack(score_matrix, axis=1)
            selected = scores.argmin(axis=1)
            for row in scores:
                rank_correlations.append(spearmanr(row, exact_val).statistic)
            pairwise = []
            for left, right in itertools.combinations(range(len(models)), 2):
                truth = np.sign(exact_val[left] - exact_val[right])
                pairwise.append(np.mean(np.sign(scores[:, left] - scores[:, right]) == truth))
            selection_rows.append({
                "dataset": dataset, "split_seed": split_seed, "method": method,
                "validation_winner": models[winner], "test_winner": models[test_winner],
                "winner_agreement": float(np.mean(selected == winner)),
                "pairwise_accuracy": float(np.mean(pairwise)),
                "spearman": float(np.nanmean(rank_correlations)),
                "validation_regret": float(np.mean(exact_val[selected] - exact_val[winner])),
                "selected_test_regret": float(np.mean(exact_test[selected] - exact_test[test_winner])),
                "exact_validation_test_winner_agreement": bool(winner == test_winner),
            })
        # Two independent 16-fit blocks for unbiased quadratic cross-score.
        for base_method, label in (("iid16", "iid_cross32"), ("strength2_16", "strength2_cross32")):
            for model in models:
                current = candidates[model]
                ids_a = current["families"][base_method]
                ids_b = np.roll(ids_a, 1, axis=0)
                a = current["val"][ids_a].mean(axis=1)
                b = current["val"][ids_b].mean(axis=1)
                y = current["val_y"]
                if a.shape[-1] == 1:
                    scores = np.mean((y - a[..., 0]) * (y - b[..., 0]), axis=1)
                else:
                    target = np.eye(a.shape[-1])[y.astype(int)]
                    scores = np.mean(np.sum((target[None] - a) * (target[None] - b), axis=-1), axis=1)
                target_loss = proper_loss(y, current["qv"])
                ordinary = loss_draws(y, (a + b) / 2)
                cross_rows.append({
                    "dataset": dataset, "split_seed": split_seed, "model": model, "method": label,
                    "target_loss": target_loss, "mean_score": float(scores.mean()),
                    "bias": float(scores.mean() - target_loss),
                    "rmse": float(np.sqrt(np.mean((scores - target_loss) ** 2))),
                    "ordinary_bias": float(ordinary.mean() - target_loss),
                })
    return pd.DataFrame(risk_rows), pd.DataFrame(metric_rows), pd.DataFrame(selection_rows), pd.DataFrame(cross_rows)


def analyze_matched() -> pd.DataFrame:
    rows = []
    for dataset in CONFIG["exact_datasets"]:
        for model in CONFIG["models"]:
            stem = f"{dataset}__{model}__matched"
            archive = np.load(NEURAL / f"{stem}.npz")
            manifest = json.loads((NEURAL / f"{stem}.json").read_text())
            predictions = archive["test_predictions"].astype(np.float64)
            variances = []
            for arm in range(2):
                quotient = predictions[arm].mean(0, keepdims=True)
                variances.append(float(np.mean(np.sum((predictions[arm] - quotient) ** 2, axis=-1))))
            rows.append({
                "dataset": dataset, "model": model, "task": manifest["task"],
                "ordinary_variance": variances[0], "matched_variance": variances[1],
                "fraction_removed": 1 - variances[1] / variances[0] if variances[0] else np.nan,
                "maximum_matched_initial_gap": manifest["maximum_matched_initial_gap"],
            })
    return pd.DataFrame(rows)


def analyze_classical() -> pd.DataFrame:
    config = json.loads((HERE / "completion_classical_config.json").read_text())
    rows = []
    for dataset in CONFIG["datasets"]:
        for model in config["models"]:
            stem = f"{dataset}__{model}"
            archive = np.load(CLASSICAL / f"{stem}.npz")
            manifest = json.loads((CLASSICAL / f"{stem}.json").read_text())
            predictions = archive["test_predictions"].astype(np.float64)
            actions = archive["actions"].astype(int)
            cards = tuple(int(actions[:, index].max() + 1) for index in range(actions.shape[1]))
            action_order = np.argsort(np.ravel_multi_index(actions.T, cards))
            predictions = predictions[action_order]
            quotient = predictions.mean(0)
            components = decompose_array(predictions.reshape(cards + predictions.shape[-2:]))
            total = sum(components.values())
            schema_only = sum(value for subset, value in components.items() if set(subset) <= {0, 1, 2})
            seed_only = sum(value for subset, value in components.items() if set(subset) <= {3})
            factor_totals = {
                label: sum(value for subset, value in components.items() if factor in subset)
                for factor, label in enumerate(("feature", "category", "class", "seed"))
            }
            main_pair = sum(value for subset, value in components.items() if len(subset) <= 2)
            flat = predictions.reshape((-1,) + predictions.shape[-2:])
            rng = np.random.default_rng(2026082881 + sum(dataset.encode()) + sum(model.encode()))
            iid_ids = rng.integers(0, len(flat), size=(DRAWS, 16))
            srs_ids = np.stack([rng.choice(len(flat), min(16, len(flat)), replace=False) for _ in range(DRAWS)])
            cover_designs = classical_strength2_family(cards[1], cards[2], cards[3])
            cover_ids = action_ids(cover_designs[rng.integers(0, len(cover_designs), size=DRAWS)], cards)
            iid_residual = residual_draws(flat, iid_ids, quotient)
            srs_residual = residual_draws(flat, srs_ids, quotient)
            cover_residual = residual_draws(flat, cover_ids, quotient)
            rows.append({
                "dataset": dataset, "model": model, "task": manifest["task"],
                "total_nuisance_variance": total, "quotient_test_loss": proper_loss(archive["test_y"], quotient),
                "schema_only_variance": schema_only, "seed_only_variance": seed_only,
                "schema_seed_interaction_variance": total - schema_only - seed_only,
                **{f"{label}_total_variance": value for label, value in factor_totals.items()},
                "main_pair_fraction": main_pair / total if total else 0.0,
                "higher_order_fraction": 1 - main_pair / total if total else 0.0,
                "iid16_residual_mean": float(iid_residual.mean()),
                "srswor16_residual_mean": float(srs_residual.mean()),
                "strength2_16_residual_mean": float(cover_residual.mean()),
                "strength2_vs_iid16_reduction": float(
                    1 - cover_residual.mean() / iid_residual.mean()
                ) if iid_residual.mean() else 0.0,
                "wall_seconds": manifest["wall_seconds"], "actions": manifest["actions"],
            })
    return pd.DataFrame(rows)


def source_comparisons(risk: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    methods = {
        "iid16": "iid16_residual_mean", "srswor16": "srswor16_residual_mean",
        "lhs16": "lhs16_residual_mean", "sobol16": "sobol16_residual_mean",
        "strength1_16": "strength1_16_residual_mean",
    }
    material = risk[risk.total_nuisance_variance > 1e-12]
    rng = np.random.default_rng(2026082891)
    rows = []; summary = {}
    for label, column in methods.items():
        grouped = material.groupby("dataset")[[column, "strength2_16_residual_mean"]].mean()
        grouped["control"] = label
        grouped["strength2_wins"] = grouped.strength2_16_residual_mean < grouped[column]
        grouped["fractional_reduction"] = 1 - grouped.strength2_16_residual_mean / grouped[column]
        rows.extend(grouped.reset_index()[
            ["dataset", "control", column, "strength2_16_residual_mean", "strength2_wins", "fractional_reduction"]
        ].rename(columns={column: "control_residual"}).to_dict(orient="records"))
        values = grouped[[column, "strength2_16_residual_mean"]].to_numpy()
        boot = np.empty(10_000)
        for draw in range(len(boot)):
            chosen = rng.integers(0, len(values), len(values))
            current = values[chosen]
            boot[draw] = 1 - current[:, 1].mean() / current[:, 0].mean()
        wins = int(grouped.strength2_wins.sum())
        summary[label] = {
            "sources": len(grouped), "source_wins": wins,
            "equal_source_fractional_reduction": float(1 - values[:, 1].mean() / values[:, 0].mean()),
            "source_bootstrap_95_interval": np.quantile(boot, [0.025, 0.975]).tolist(),
            "two_sided_sign_test_p": float(binomtest(wins, len(grouped), 0.5).pvalue),
        }
    return pd.DataFrame(rows), summary


def main() -> None:
    risk, metrics, selection, cross = analyze_neural()
    matched = analyze_matched(); classical = analyze_classical()
    source_groups, source_summary = source_comparisons(risk)
    for name, frame in (
        ("completion_neural_risk_cells", risk), ("completion_neural_metrics", metrics),
        ("completion_neural_selection", selection), ("completion_neural_cross_score", cross),
        ("completion_matched_function", matched), ("completion_classical_risk", classical),
        ("completion_neural_source_groups", source_groups),
    ):
        frame.to_csv(RESULTS / f"{name}.csv", index=False)
    material = risk[risk.total_nuisance_variance > 1e-12]
    summary = {
        "status": "complete", "neural_cells": len(risk), "datasets": risk.dataset.nunique(),
        "splits": risk.split_seed.nunique(), "models": risk.model.nunique(),
        "material_cells": len(material),
        "strength2_vs_iid16_wins": int((material.strength2_16_residual_mean < material.iid16_residual_mean).sum()),
        "strength2_vs_srswor16_wins": int((material.strength2_16_residual_mean < material.srswor16_residual_mean).sum()),
        "mean_strength2_vs_iid16_reduction": float(1 - material.strength2_16_residual_mean.mean() / material.iid16_residual_mean.mean()),
        "mean_strength2_vs_srswor16_reduction": float(1 - material.strength2_16_residual_mean.mean() / material.srswor16_residual_mean.mean()),
        "mean_main_pair_fraction": float(material.main_pair_fraction.mean()),
        "matched_pooled_fraction_removed": float(1 - matched.matched_variance.mean() / matched.ordinary_variance.mean()),
        "matched_cells_with_residual_above_1e_10": int((matched.matched_variance > 1e-10).sum()),
        "packing_population128_cells": int((risk.population == 128).sum()),
        "disjoint_pair32_vs_independent_cover32_wins_population128": int((
            risk.loc[risk.population == 128, "disjoint_pair32_residual_mean"]
            < risk.loc[risk.population == 128, "independent_cover32_residual_mean"]
        ).sum()),
        "disjoint_pack64_vs_independent_cover64_wins_population128": int((
            risk.loc[risk.population == 128, "disjoint_pack64_residual_mean"]
            < risk.loc[risk.population == 128, "independent_cover64_residual_mean"]
        ).sum()),
        "selection_method_means": selection.groupby("method")[["winner_agreement", "pairwise_accuracy", "spearman", "validation_regret", "selected_test_regret"]].mean().to_dict(orient="index"),
        "cross_score_method_means": cross.groupby("method")[["bias", "rmse", "ordinary_bias"]].mean().to_dict(orient="index"),
        "exact_validation_test_winner_agreement": float(selection.drop_duplicates(["dataset", "split_seed"]).exact_validation_test_winner_agreement.mean()),
        "classical_cells": len(classical),
        "classical_strength2_vs_iid16_wins": int((
            classical.strength2_16_residual_mean < classical.iid16_residual_mean
        ).sum()),
        "classical_strength2_vs_srswor16_wins": int((
            classical.strength2_16_residual_mean < classical.srswor16_residual_mean
        ).sum()),
        "represented_neural_fits": int(sum(json.loads(path.read_text()).get("represented_fits", 0) for path in NEURAL.glob("*__broad.json"))),
        "represented_matched_fits": int(sum(json.loads(path.read_text()).get("represented_fits", 0) for path in NEURAL.glob("*__matched.json"))),
        "represented_classical_fits": int(classical.actions.sum()),
        "monte_carlo_draws_per_method": DRAWS,
        "source_clustered_comparisons": source_summary,
    }
    (RESULTS / "completion_panel_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
