from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from src.core import load_dataset, parse_indices  # noqa: E402


DATASETS = ("credit-g", "diamonds")
EPS = 1e-12


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), EPS))


def _dist_summary(point: np.ndarray, cloud: np.ndarray) -> list[float]:
    distances = np.linalg.norm(cloud - point, axis=1)
    return [
        float(distances.min()),
        float(np.quantile(distances, 0.25)),
        float(np.median(distances)),
        float(distances.mean()),
        float(distances.max()),
    ]


def _entropy(counts: np.ndarray) -> float:
    probabilities = counts[counts > 0] / max(counts.sum(), 1)
    return float(-(probabilities * np.log(probabilities)).sum())


def _feature_names(d: int) -> tuple[list[str], list[str]]:
    row = []
    for block in ("out", "in", "diff", "absdiff", "product"):
        row.extend(f"{block}_z{j}" for j in range(d))
    row.extend(
        [
            "in_out_distance",
            "in_out_cosine",
            "same_target_bin",
            "target_bin_delta",
        ]
    )
    context = []
    for block in ("set_mean", "set_std", "query_mean", "query_std"):
        context.extend(f"{block}_z{j}" for j in range(d))
    context.extend(
        [
            "out_to_set_mean_distance",
            "in_to_set_mean_distance",
            "out_to_set_mean_cosine",
            "in_to_set_mean_cosine",
            "out_to_query_mean_distance",
            "in_to_query_mean_distance",
            "out_to_query_mean_cosine",
            "in_to_query_mean_cosine",
        ]
    )
    for point in ("out", "in"):
        context.extend(f"{point}_to_set_{stat}" for stat in ("min", "q25", "median", "mean", "max"))
    context.extend(
        [
            "out_bin_count",
            "in_bin_count",
            "bin_entropy_before",
            "bin_entropy_after",
            "bin_entropy_delta",
        ]
    )
    return row, context


def build_neighborhoods(dataset: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    bundle = load_dataset(dataset)
    trace = pd.read_csv(ROOT / "results" / "processed" / "direct_search" / f"{dataset}.csv")
    raw = pd.read_csv(ROOT / "results" / "raw" / "direct_context_evaluations" / f"{dataset}.csv")
    learned = np.load(ROOT / "results" / "processed" / "selector_models" / f"{dataset}_k32.npz")

    utility_by_context = {
        tuple(parse_indices(row.indices)): float(row.utility)
        for row in raw.itertuples()
        if int(row.context_size) == 32
    }
    z = np.asarray(bundle.feature_z, dtype=float)
    query_z = np.asarray(bundle.selector_feature_z, dtype=float)
    target_bins = np.asarray(bundle.target_bins, dtype=int)
    additive = np.asarray(learned["additive"], dtype=float)
    fm_additive = np.asarray(learned["fm_additive"], dtype=float)
    fm_pair = np.asarray(learned["fm_pair"], dtype=float)
    row_names, context_names = _feature_names(z.shape[1])
    records: list[dict[str, float | int | str]] = []

    for anchor in trace.iloc[:-1].itertuples():
        current = np.asarray(parse_indices(anchor.indices), dtype=int)
        selected = set(map(int, current))
        current_utility = float(anchor.selector_utility)
        set_z = z[current]
        set_mean = set_z.mean(axis=0)
        set_std = set_z.std(axis=0)
        query_mean = query_z.mean(axis=0)
        query_std = query_z.std(axis=0)
        counts_before = np.bincount(target_bins[current], minlength=int(target_bins.max()) + 1)

        for old in current:
            if bundle.task == "classification" and np.sum(bundle.y_candidate[current] == bundle.y_candidate[old]) <= 1:
                continue
            cloud_without_old = z[current[current != old]]
            for new in sorted(set(range(len(z))) - selected):
                proposal = tuple(sorted((selected - {int(old)}) | {int(new)}))
                if proposal not in utility_by_context:
                    raise KeyError(f"Missing cached neighborhood: {dataset}, round={anchor.round}, {old}->{new}")
                out_z, in_z = z[old], z[new]
                row_features = np.concatenate(
                    [out_z, in_z, in_z - out_z, np.abs(in_z - out_z), in_z * out_z]
                ).tolist()
                row_features.extend(
                    [
                        float(np.linalg.norm(in_z - out_z)),
                        _safe_cosine(in_z, out_z),
                        float(target_bins[old] == target_bins[new]),
                        float(target_bins[new] - target_bins[old]),
                    ]
                )
                counts_after = counts_before.copy()
                counts_after[target_bins[old]] -= 1
                counts_after[target_bins[new]] += 1
                context_features = np.concatenate([set_mean, set_std, query_mean, query_std]).tolist()
                context_features.extend(
                    [
                        float(np.linalg.norm(out_z - set_mean)),
                        float(np.linalg.norm(in_z - set_mean)),
                        _safe_cosine(out_z, set_mean),
                        _safe_cosine(in_z, set_mean),
                        float(np.linalg.norm(out_z - query_mean)),
                        float(np.linalg.norm(in_z - query_mean)),
                        _safe_cosine(out_z, query_mean),
                        _safe_cosine(in_z, query_mean),
                    ]
                )
                context_features.extend(_dist_summary(out_z, cloud_without_old))
                context_features.extend(_dist_summary(in_z, cloud_without_old))
                entropy_before = _entropy(counts_before)
                entropy_after = _entropy(counts_after)
                context_features.extend(
                    [
                        float(counts_before[target_bins[old]]),
                        float(counts_before[target_bins[new]]),
                        entropy_before,
                        entropy_after,
                        entropy_after - entropy_before,
                    ]
                )
                proposal_array = np.asarray(proposal, dtype=int)
                fm_current = fm_additive[current].sum() + np.triu(fm_pair[np.ix_(current, current)], 1).sum()
                fm_proposal = fm_additive[proposal_array].sum() + np.triu(
                    fm_pair[np.ix_(proposal_array, proposal_array)], 1
                ).sum()
                record: dict[str, float | int | str] = {
                    "dataset": dataset,
                    "round": int(anchor.round),
                    "swap_out": int(old),
                    "swap_in": int(new),
                    "utility": utility_by_context[proposal],
                    "gain": utility_by_context[proposal] - current_utility,
                    "additive_delta": float(additive[new] - additive[old]),
                    "fm_delta": float(fm_proposal - fm_current),
                }
                record.update(zip(row_names, row_features))
                record.update(zip(context_names, context_features))
                records.append(record)

    frame = pd.DataFrame.from_records(records)
    return frame, row_names, context_names


def choice_metrics(frame: pd.DataFrame, score: np.ndarray) -> dict[str, float]:
    actual = frame["gain"].to_numpy()
    chosen = int(np.argmax(score))
    oracle = float(actual.max())
    picked = float(actual[chosen])
    rank = float((actual <= picked).mean())
    scale = max(float(actual.std()), EPS)
    return {
        "r2": float(r2_score(actual, score)),
        "spearman": float(spearmanr(actual, score).statistic),
        "choice_percentile": rank,
        "chosen_gain": picked,
        "oracle_gain": oracle,
        "regret": oracle - picked,
        "normalized_regret": (oracle - picked) / scale,
        "improving_choice": float(picked > 0),
    }


def model_factories(seed: int = 0):
    return {
        "ridge": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 20))),
        "hist_gbdt": lambda: HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, random_state=seed
        ),
        "extra_trees": lambda: ExtraTreesRegressor(
            n_estimators=200, min_samples_leaf=8, max_features=0.7, n_jobs=-1, random_state=seed
        ),
    }


def evaluate_dataset(frame: pd.DataFrame, row_features: list[str], context_features: list[str]) -> pd.DataFrame:
    outputs: list[dict[str, float | int | str]] = []
    rounds = sorted(frame["round"].unique())
    for heldout in rounds:
        train = frame[frame["round"] != heldout]
        test = frame[frame["round"] == heldout]
        base = {"dataset": test.dataset.iloc[0], "split": "leave_one_round_out", "round": int(heldout)}
        for baseline in ("additive_delta", "fm_delta"):
            outputs.append({**base, "model": baseline, **choice_metrics(test, test[baseline].to_numpy())})

        means = train.groupby(["swap_out", "swap_in"])["gain"].mean()
        fallback = float(train["gain"].mean())
        empirical = np.asarray(
            [means.get((row.swap_out, row.swap_in), fallback) for row in test.itertuples()], dtype=float
        )
        outputs.append({**base, "model": "empirical_static_swap", **choice_metrics(test, empirical)})

        for feature_set, features in (("row", row_features), ("contextual", row_features + context_features)):
            for model_name, factory in model_factories().items():
                model = factory()
                model.fit(train[features], train["gain"])
                prediction = model.predict(test[features])
                outputs.append(
                    {**base, "model": f"{feature_set}_{model_name}", **choice_metrics(test, prediction)}
                )

    # A stricter online-style test: only earlier neighborhoods may be used.
    for heldout in rounds[1:]:
        train = frame[frame["round"] < heldout]
        test = frame[frame["round"] == heldout]
        base = {"dataset": test.dataset.iloc[0], "split": "past_to_next", "round": int(heldout)}
        for baseline in ("additive_delta", "fm_delta"):
            outputs.append({**base, "model": baseline, **choice_metrics(test, test[baseline].to_numpy())})
        for feature_set, features in (("row", row_features), ("contextual", row_features + context_features)):
            for model_name in ("hist_gbdt", "extra_trees"):
                model = model_factories()[model_name]()
                model.fit(train[features], train["gain"])
                outputs.append(
                    {
                        **base,
                        "model": f"{feature_set}_{model_name}",
                        **choice_metrics(test, model.predict(test[features])),
                    }
                )
    return pd.DataFrame(outputs)


def stability_diagnostics(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, float | int]] = []
    rounds = sorted(frame["round"].unique())
    for i, left_round in enumerate(rounds):
        left = frame[frame["round"] == left_round][["swap_out", "swap_in", "gain"]]
        for right_round in rounds[i + 1 :]:
            right = frame[frame["round"] == right_round][["swap_out", "swap_in", "gain"]]
            joined = left.merge(right, on=["swap_out", "swap_in"], suffixes=("_left", "_right"))
            rows.append(
                {
                    "round_left": int(left_round),
                    "round_right": int(right_round),
                    "shared_swaps": len(joined),
                    "spearman": float(spearmanr(joined.gain_left, joined.gain_right).statistic),
                    "sign_flip_rate": float((np.sign(joined.gain_left) != np.sign(joined.gain_right)).mean()),
                    "mean_absolute_gain_change": float(np.abs(joined.gain_left - joined.gain_right).mean()),
                }
            )
    pairwise = pd.DataFrame(rows)
    repeated = frame.groupby(["swap_out", "swap_in"]).filter(lambda group: len(group) >= 3)
    per_swap = repeated.groupby(["swap_out", "swap_in"])["gain"].agg(["std", "mean"])
    summary = {
        "mean_cross_round_spearman": float(pairwise.spearman.mean()),
        "min_cross_round_spearman": float(pairwise.spearman.min()),
        "mean_sign_flip_rate": float(pairwise.sign_flip_rate.mean()),
        "median_within_swap_std": float(per_swap["std"].median()),
        "median_absolute_within_swap_mean": float(per_swap["mean"].abs().median()),
    }
    return pairwise, summary


def budget_simulation(frame: pd.DataFrame, features: list[str], repetitions: int = 5) -> pd.DataFrame:
    """One-step, within-neighborhood call-budget simulation.

    A random warm start is evaluated, then a contextual model chooses one additional
    swap. This isolates whether learned structure improves over best-of-budget random
    search without pretending to evaluate a full adaptive trajectory.
    """
    rows: list[dict[str, float | int | str]] = []
    budgets = (16, 32, 64, 128, 256)
    for round_id, neighborhood in frame.groupby("round"):
        values = neighborhood["gain"].to_numpy()
        X = neighborhood[features]
        n = len(neighborhood)
        for budget in budgets:
            for repetition in range(repetitions):
                rng = np.random.default_rng(10_000 * int(round_id) + 100 * budget + repetition)
                observed = rng.choice(n, size=budget, replace=False)
                unseen_mask = np.ones(n, dtype=bool)
                unseen_mask[observed] = False
                unseen = np.flatnonzero(unseen_mask)
                random_best = float(values[observed].max())
                model = ExtraTreesRegressor(
                    n_estimators=30,
                    min_samples_leaf=max(2, budget // 32),
                    max_features=0.8,
                    n_jobs=4,
                    random_state=repetition,
                )
                model.fit(X.iloc[observed], values[observed])
                proposed = int(unseen[np.argmax(model.predict(X.iloc[unseen]))])
                learned_best = max(random_best, float(values[proposed]))
                oracle = float(values.max())
                for method, achieved in (("random_best", random_best), ("contextual_plus_one", learned_best)):
                    rows.append(
                        {
                            "dataset": neighborhood.dataset.iloc[0],
                            "round": int(round_id),
                            "budget": budget,
                            "repetition": repetition,
                            "method": method,
                            "achieved_gain": achieved,
                            "oracle_gain": oracle,
                            "regret": oracle - achieved,
                            "improving": float(achieved > 0),
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    all_evaluations = []
    all_budgets = []
    audit: dict[str, dict[str, float | int]] = {}
    for dataset in DATASETS:
        neighborhood_path = HERE / f"{dataset}_neighborhoods.parquet"
        row_features, context_features = _feature_names(16)
        if neighborhood_path.exists():
            print(f"Loading cached {dataset} swap neighborhoods", flush=True)
            frame = pd.read_parquet(neighborhood_path)
        else:
            print(f"Building {dataset} swap neighborhoods", flush=True)
            frame, row_features, context_features = build_neighborhoods(dataset)
            frame.to_parquet(neighborhood_path, index=False)
        pairwise, summary = stability_diagnostics(frame)
        pairwise.insert(0, "dataset", dataset)
        pairwise.to_csv(HERE / f"{dataset}_cross_round_stability.csv", index=False)
        evaluations = evaluate_dataset(frame, row_features, context_features)
        evaluations.to_csv(HERE / f"{dataset}_predictive_results.csv", index=False)
        budgets = budget_simulation(frame, row_features + context_features)
        budgets.to_csv(HERE / f"{dataset}_budget_simulation.csv", index=False)
        all_evaluations.append(evaluations)
        all_budgets.append(budgets)
        audit[dataset] = {
            "rows": len(frame),
            "rounds": int(frame["round"].nunique()),
            "row_features": len(row_features),
            "context_features": len(context_features),
            **summary,
        }
        print(f"Finished {dataset}: {json.dumps(audit[dataset], sort_keys=True)}", flush=True)

    evaluation = pd.concat(all_evaluations, ignore_index=True)
    budget = pd.concat(all_budgets, ignore_index=True)
    evaluation.to_csv(HERE / "predictive_results.csv", index=False)
    budget.to_csv(HERE / "budget_simulation.csv", index=False)
    summary = (
        evaluation.groupby(["split", "model"])[
            ["r2", "spearman", "choice_percentile", "chosen_gain", "regret", "normalized_regret", "improving_choice"]
        ]
        .mean()
        .reset_index()
    )
    summary.to_csv(HERE / "predictive_summary.csv", index=False)
    budget_summary = (
        budget.groupby(["budget", "method"])[["achieved_gain", "regret", "improving"]]
        .mean()
        .reset_index()
    )
    budget_summary.to_csv(HERE / "budget_summary.csv", index=False)
    with (HERE / "audit.json").open("w") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
    print("\nPredictive summary\n", summary.to_string(index=False), flush=True)
    print("\nBudget summary\n", budget_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
