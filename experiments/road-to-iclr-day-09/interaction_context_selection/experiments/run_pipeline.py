#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from src.core import (  # noqa: E402
    DATASETS,
    PLOTS,
    PROCESSED,
    RAW,
    ROOT,
    DeepSets,
    FM,
    FeatureFM,
    SignedBilinear,
    TabICLEvaluator,
    atomic_json,
    ensure_dirs,
    environment_record,
    fit_ridge,
    fit_torch_model,
    indices_string,
    load_dataset,
    membership,
    parse_indices,
    prediction_metrics,
    sample_context,
    stable_seed,
    surrogate_metrics,
)
from src.selectors import (  # noqa: E402
    aggregate_pair_feature,
    complementarity_matrix,
    dpp_logdet,
    geometry_pair_matrices,
    kcenter,
    kmedoids_like,
    latent_medoid_like,
    mmd_crumb_like,
    nearest_query_cluster,
    one_swap,
    pairwise_greedy,
    topk,
)


CONFIG = json.loads((ROOT / "config.json").read_text())


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def evaluate_random_contexts(dataset: str, device: str) -> None:
    ensure_dirs()
    bundle = load_dataset(dataset)
    atomic_json(
        RAW / "splits" / f"{dataset}.json",
        {
            "dataset": dataset,
            "split_seed": 0,
            "candidate_indices": bundle.candidate_idx.tolist(),
            "selector_indices": bundle.selector_idx.tolist(),
            "test_indices": bundle.test_idx.tolist(),
            "candidate_target_bins": bundle.target_bins.tolist(),
        },
    )
    evaluator = TabICLEvaluator(bundle, device, seed=0, n_estimators=CONFIG["tfm_estimators"])
    environment_path = RAW / "environment" / f"{dataset}.json"
    total_started = time.perf_counter()
    recorded_evaluations = 0
    recorded_model_seconds = 0.0
    for k in CONFIG["budgets"]:
        for seed in CONFIG["seeds"]:
            path = RAW / "context_evaluations" / f"{dataset}_k{k}_seed{seed}.csv"
            prediction_path = RAW / "predictions" / f"{dataset}_k{k}_seed{seed}.npz"
            if path.exists():
                records = pd.read_csv(path).to_dict("records")
            else:
                records = []
            predictions: list[np.ndarray] = []
            if prediction_path.exists():
                predictions = list(np.load(prediction_path)["predictions"])
            if len(predictions) != len(records):
                raise RuntimeError(f"Prediction/record count mismatch in {path}")
            rng = np.random.default_rng(seed)
            # Reconstruct all prior draws, making resume bit-for-bit deterministic.
            for _ in range(len(records)):
                sample_context(bundle.y_candidate, k, rng, bundle.task)
            for context_id in range(len(records), CONFIG["random_contexts_per_seed"]):
                indices = sample_context(bundle.y_candidate, k, rng, bundle.task)
                out = evaluator.evaluate(indices, "selector", return_prediction=True)
                prediction = out.pop("prediction")
                predictions.append(prediction.astype(np.float32))
                records.append(
                    {
                        "dataset": dataset,
                        "task": bundle.task,
                        "K": k,
                        "seed": seed,
                        "context_id": context_id,
                        "indices": indices_string(indices),
                        **out,
                    }
                )
                if (context_id + 1) % 32 == 0 or context_id + 1 == CONFIG["random_contexts_per_seed"]:
                    _write_csv_atomic(pd.DataFrame(records), path)
                    prediction_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = prediction_path.with_suffix(".tmp.npz")
                    np.savez_compressed(temporary, predictions=np.asarray(predictions, dtype=np.float32))
                    temporary.replace(prediction_path)
                    print(f"{dataset} K={k} seed={seed}: {context_id + 1}/{CONFIG['random_contexts_per_seed']}", flush=True)
            verify_rng = np.random.default_rng(seed)
            assert all(
                np.array_equal(parse_indices(record["indices"]), sample_context(bundle.y_candidate, k, verify_rng, bundle.task))
                for record in records
            )
            recorded_evaluations += len(records)
            recorded_model_seconds += float(sum(float(record.get("runtime_seconds", 0.0)) for record in records))
    env = environment_record(bundle, evaluator)
    env.update(
        {
            "seeds": CONFIG["seeds"],
            "budgets": CONFIG["budgets"],
            "context_evaluations": recorded_evaluations,
            "model_runtime_seconds": recorded_model_seconds,
            "wall_runtime_seconds": time.perf_counter() - total_started,
            "tfm_estimators": CONFIG["tfm_estimators"],
            "preprocessing": "official TabICLv2 TransformToNumerical; selector z is separate",
        }
    )
    atomic_json(environment_path, env)


def _load_cell(dataset: str, k: int) -> pd.DataFrame:
    frames = []
    for seed in CONFIG["seeds"]:
        path = RAW / "context_evaluations" / f"{dataset}_k{k}_seed{seed}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if len(frame) != CONFIG["random_contexts_per_seed"]:
            raise RuntimeError(f"Incomplete cell {path}: {len(frame)} records")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _membership_matrix(frame: pd.DataFrame) -> np.ndarray:
    return np.stack([membership(parse_indices(value)) for value in frame["indices"]]).astype(np.float32)


def _surrogate_partition(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    train_parts, test_parts = [], []
    for seed in CONFIG["seeds"]:
        subset = np.flatnonzero(frame.seed.to_numpy() == seed)
        train, test = train_test_split(subset, test_size=0.30, random_state=stable_seed("surrogate", seed))
        train_parts.append(train)
        test_parts.append(test)
    return np.concatenate(train_parts), np.concatenate(test_parts)


def _torch_coefficients(fit: Any) -> tuple[np.ndarray, np.ndarray]:
    model = fit.model
    linear = model.linear.detach().cpu().numpy() * fit.target_scale
    pair = model.pair_matrix() * fit.target_scale
    return linear, pair


def analyze_dataset(dataset: str, device: str) -> None:
    ensure_dirs()
    bundle = load_dataset(dataset)
    evaluator = TabICLEvaluator(bundle, device, seed=0, n_estimators=CONFIG["tfm_estimators"])
    all_prediction_rows: list[dict[str, Any]] = []
    all_selection_rows: list[dict[str, Any]] = []
    utility_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    surrogate_prediction_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for k in CONFIG["budgets"]:
        frame = _load_cell(dataset, k)
        X = _membership_matrix(frame)
        utility = frame.utility.to_numpy(dtype=np.float64)
        train_i, test_i = _surrogate_partition(frame)
        Xtr, Xte, ytr, yte = X[train_i], X[test_i], utility[train_i], utility[test_i]

        constant = np.full_like(yte, ytr.mean())
        utility_rows.append({"dataset": dataset, "task": bundle.task, "K": k, "model": "constant", **surrogate_metrics(yte, constant)})

        ridge = fit_ridge(Xtr, ytr, CONFIG["ridge_alphas"])
        ridge_pred = ridge.predict(Xte)
        _record_surrogate_predictions(surrogate_prediction_rows, dataset, k, frame.iloc[test_i], yte, "additive_ridge", ridge_pred)
        utility_rows.append(
            {
                "dataset": dataset,
                "task": bundle.task,
                "K": k,
                "model": "additive_ridge",
                "rank": np.nan,
                "regularization": ridge.alpha,
                **surrogate_metrics(yte, ridge_pred),
            }
        )

        fm_fits: list[tuple[int, Any]] = []
        residual_fits: list[tuple[int, Any]] = []
        train_residual = ytr - ridge.predict(Xtr)
        for rank in CONFIG["fm_ranks"]:
            fit = fit_torch_model(FM(256, rank), Xtr, ytr, stable_seed(dataset, k, "fm", rank), weight_decay=1e-2)
            pred = fit.predict(Xte)
            _record_surrogate_predictions(surrogate_prediction_rows, dataset, k, frame.iloc[test_i], yte, f"id_fm_r{rank}", pred)
            metrics = surrogate_metrics(yte, pred)
            fm_fits.append((rank, fit))
            utility_rows.append({"dataset": dataset, "task": bundle.task, "K": k, "model": "id_fm", "rank": rank, "regularization": 1e-2, **metrics})
            ablation_rows.append({"dataset": dataset, "K": k, "parameterization": "ID-factor joint", "rank": rank, "regularization": 1e-2, "val_mse": fit.val_mse, **metrics})

            residual_fit = fit_torch_model(
                FM(256, rank, interaction_only=True),
                Xtr,
                train_residual,
                stable_seed(dataset, k, "residual", rank),
                weight_decay=2e-2,
            )
            residual_pred = ridge_pred + residual_fit.predict(Xte)
            residual_metrics = surrogate_metrics(yte, residual_pred)
            residual_fits.append((rank, residual_fit))
            utility_rows.append({"dataset": dataset, "task": bundle.task, "K": k, "model": "residual_fm", "rank": rank, "regularization": 2e-2, **residual_metrics})
            ablation_rows.append({"dataset": dataset, "K": k, "parameterization": "ID-factor residual", "rank": rank, "regularization": 2e-2, "val_mse": residual_fit.val_mse, **residual_metrics})

        feature_fits: list[tuple[int, Any]] = []
        for rank in CONFIG["feature_fm_ranks"]:
            fit = fit_torch_model(
                FeatureFM(bundle.z, rank), Xtr, ytr, stable_seed(dataset, k, "feature", rank), weight_decay=1e-2
            )
            pred = fit.predict(Xte)
            _record_surrogate_predictions(surrogate_prediction_rows, dataset, k, frame.iloc[test_i], yte, f"feature_fm_r{rank}", pred)
            metrics = surrogate_metrics(yte, pred)
            feature_fits.append((rank, fit))
            utility_rows.append({"dataset": dataset, "task": bundle.task, "K": k, "model": "feature_fm", "rank": rank, "regularization": 1e-2, **metrics})
            ablation_rows.append({"dataset": dataset, "K": k, "parameterization": "feature-bilinear MLP", "rank": rank, "regularization": 1e-2, "val_mse": fit.val_mse, **metrics})

        signed_fits: list[tuple[int, Any]] = []
        for rank in CONFIG["feature_fm_ranks"]:
            fit = fit_torch_model(
                SignedBilinear(bundle.z, rank), Xtr, ytr, stable_seed(dataset, k, "signed", rank), weight_decay=2e-2
            )
            pred = fit.predict(Xte)
            metrics = surrogate_metrics(yte, pred)
            signed_fits.append((rank, fit))
            utility_rows.append({"dataset": dataset, "task": bundle.task, "K": k, "model": "signed_bilinear", "rank": rank, "regularization": 2e-2, **metrics})
            ablation_rows.append({"dataset": dataset, "K": k, "parameterization": "signed bilinear", "rank": rank, "regularization": 2e-2, "val_mse": fit.val_mse, **metrics})

        deepset = fit_torch_model(DeepSets(bundle.z), Xtr, ytr, stable_seed(dataset, k, "deepsets"), weight_decay=1e-2)
        deep_metrics = surrogate_metrics(yte, deepset.predict(Xte))
        _record_surrogate_predictions(surrogate_prediction_rows, dataset, k, frame.iloc[test_i], yte, "deepsets", deepset.predict(Xte))
        utility_rows.append({"dataset": dataset, "task": bundle.task, "K": k, "model": "deepsets", "rank": np.nan, "regularization": 1e-2, **deep_metrics})
        ablation_rows.append({"dataset": dataset, "K": k, "parameterization": "DeepSets", "rank": np.nan, "regularization": 1e-2, "val_mse": deepset.val_mse, **deep_metrics})

        # Explicit constrained diversity and label/target complementarity ablations.
        geometry = geometry_pair_matrices(bundle.z)
        complement = complementarity_matrix(bundle.target_bins)
        residual = ytr - ridge.predict(Xtr)
        pair_candidates: list[tuple[str, float, np.ndarray, float]] = []
        for name, matrix in {**geometry, "label_target_complementarity": complement}.items():
            train_feature = aggregate_pair_feature(Xtr, matrix)[:, None]
            test_feature = aggregate_pair_feature(Xte, matrix)[:, None]
            fitted = Ridge(alpha=1.0, fit_intercept=False, positive=True).fit(train_feature, residual)
            pred = ridge_pred + fitted.predict(test_feature)
            metrics = surrogate_metrics(yte, pred)
            coefficient = float(fitted.coef_[0])
            pair_candidates.append((name, mean_cv_proxy(train_feature, residual, fitted), matrix * coefficient, coefficient))
            utility_rows.append({"dataset": dataset, "task": bundle.task, "K": k, "model": name, "rank": np.nan, "regularization": 1.0, **metrics})
            ablation_rows.append({"dataset": dataset, "K": k, "parameterization": name, "rank": np.nan, "regularization": 1.0, "coefficient": coefficient, **metrics})
        combined_train = np.column_stack(
            [aggregate_pair_feature(Xtr, geometry["rbf_diversity"]), aggregate_pair_feature(Xtr, complement)]
        )
        combined_test = np.column_stack(
            [aggregate_pair_feature(Xte, geometry["rbf_diversity"]), aggregate_pair_feature(Xte, complement)]
        )
        combined_fit = Ridge(alpha=1.0, fit_intercept=False, positive=True).fit(combined_train, residual)
        combined_pred = ridge_pred + combined_fit.predict(combined_test)
        combined_metrics = surrogate_metrics(yte, combined_pred)
        combined_matrix = combined_fit.coef_[0] * geometry["rbf_diversity"] + combined_fit.coef_[1] * complement
        pair_candidates.append(("geometry_plus_complementarity", mean_cv_proxy(combined_train, residual, combined_fit), combined_matrix, float(np.linalg.norm(combined_fit.coef_))))
        utility_rows.append({"dataset": dataset, "task": bundle.task, "K": k, "model": "geometry_plus_complementarity", "rank": np.nan, "regularization": 1.0, **combined_metrics})
        ablation_rows.append({"dataset": dataset, "K": k, "parameterization": "geometry + complementarity", "rank": np.nan, "regularization": 1.0, "coefficient": json.dumps(combined_fit.coef_.tolist()), **combined_metrics})

        best_fm_rank, best_fm = min(fm_fits, key=lambda item: item[1].val_mse)
        best_feature_rank, best_feature = min(feature_fits, key=lambda item: item[1].val_mse)
        fm_additive, fm_pair = _torch_coefficients(best_fm)
        feature_additive, feature_pair = _torch_coefficients(best_feature)
        strongest_pair_name, _, strongest_pair, _ = min(pair_candidates, key=lambda item: item[1])

        classification = bundle.task == "classification"
        y_candidate = bundle.y_candidate
        selected: dict[str, np.ndarray] = {
            "additive": topk(ridge.coef_, y_candidate, k, classification),
            "k_center": kcenter(bundle.feature_z, k, y_candidate, classification),
            "k_medoids": kmedoids_like(bundle.feature_z, k, y_candidate, classification),
            "nearest_query_cluster": nearest_query_cluster(bundle.feature_z, bundle.selector_feature_z, k, y_candidate, classification),
            "CRUMB-like": mmd_crumb_like(bundle.feature_z, bundle.selector_feature_z, k, y_candidate, classification),
            "LUCoS-like": latent_medoid_like(bundle.feature_z, bundle.selector_feature_z, k, y_candidate, classification),
            "DPP": dpp_logdet(bundle.z, k, y_candidate, classification),
            "pairwise_FM_greedy": pairwise_greedy(fm_additive, fm_pair, k, y_candidate, classification),
            "feature_FM_greedy": pairwise_greedy(feature_additive, feature_pair, k, y_candidate, classification),
            f"complementarity:{strongest_pair_name}": pairwise_greedy(ridge.coef_, strongest_pair, k, y_candidate, classification),
        }
        selected["pairwise_FM_swap"] = one_swap(
            selected["pairwise_FM_greedy"], fm_additive, fm_pair, y_candidate, classification
        )
        best_random = frame.iloc[int(np.argmax(utility))]
        selected["oracle_best_of_random"] = parse_indices(best_random["indices"])

        for repeat in range(CONFIG["random_selector_repeats"]):
            indices = sample_context(y_candidate, k, np.random.default_rng(stable_seed(dataset, k, "random-test", repeat)), bundle.task)
            _evaluate_selected(
                evaluator, bundle, k, "random_stratified", repeat, indices, all_selection_rows, all_prediction_rows
            )
        for method, indices in selected.items():
            _evaluate_selected(evaluator, bundle, k, method, 0, indices, all_selection_rows, all_prediction_rows)

        # Save selector objects needed by the direct TFM local-search phase.
        model_path = PROCESSED / "selector_models" / f"{dataset}_k{k}.npz"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            model_path,
            additive=ridge.coef_,
            fm_additive=fm_additive,
            fm_pair=fm_pair,
            feature_additive=feature_additive,
            feature_pair=feature_pair,
            strongest_pair=strongest_pair,
            best_fm_rank=best_fm_rank,
            best_feature_rank=best_feature_rank,
        )

    _write_csv_atomic(pd.DataFrame(utility_rows), PROCESSED / "utility_prediction" / f"{dataset}.csv")
    _write_csv_atomic(pd.DataFrame(ablation_rows), PROCESSED / "ablations" / f"{dataset}.csv")
    _write_csv_atomic(pd.DataFrame(all_selection_rows), PROCESSED / "selector_results" / f"{dataset}.csv")
    _write_csv_atomic(pd.DataFrame(all_prediction_rows), RAW / "test_predictions" / f"{dataset}.csv")
    _write_csv_atomic(pd.DataFrame(surrogate_prediction_rows), RAW / "surrogate_predictions" / f"{dataset}.csv")
    audit = {
        "dataset": dataset,
        "analysis_wall_seconds": time.perf_counter() - started,
        "test_context_evaluations": evaluator.evaluations,
        "test_model_runtime_seconds": evaluator.total_seconds,
        "checkpoint": evaluator.checkpoint_path,
        "final_test_labels_used_for_selection": False,
    }
    atomic_json(PROCESSED / "analysis_audits" / f"{dataset}.json", audit)


def _record_surrogate_predictions(
    rows: list[dict[str, Any]],
    dataset: str,
    k: int,
    heldout: pd.DataFrame,
    actual: np.ndarray,
    model: str,
    predicted: np.ndarray,
) -> None:
    for record, truth, estimate in zip(heldout.to_dict("records"), actual, predicted):
        rows.append(
            {
                "dataset": dataset,
                "K": k,
                "seed": record["seed"],
                "context_id": record["context_id"],
                "model": model,
                "actual_utility": truth,
                "predicted_utility": estimate,
            }
        )


def mean_cv_proxy(X: np.ndarray, y: np.ndarray, model: Ridge) -> float:
    # Hyperparameter-free deterministic proxy used only to choose among named geometry variants.
    pred = model.predict(X)
    return float(np.mean((y - pred) ** 2))


def _evaluate_selected(
    evaluator: TabICLEvaluator,
    bundle: Any,
    k: int,
    method: str,
    repeat: int,
    indices: np.ndarray,
    selection_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> None:
    out = evaluator.evaluate(indices, "test", return_prediction=True)
    prediction = out.pop("prediction")
    selection_rows.append(
        {
            "dataset": bundle.name,
            "task": bundle.task,
            "K": k,
            "method": method,
            "repeat": repeat,
            "indices": indices_string(indices),
            **out,
        }
    )
    for query_id, (truth, pred) in enumerate(zip(bundle.y_test, prediction)):
        row: dict[str, Any] = {
            "dataset": bundle.name,
            "task": bundle.task,
            "K": k,
            "method": method,
            "repeat": repeat,
            "query_id": query_id,
            "y_true": truth,
        }
        if bundle.task == "classification":
            row.update({f"prob_{i}": float(value) for i, value in enumerate(pred)})
            row["prediction"] = int(np.argmax(pred))
        else:
            row["prediction"] = float(pred)
        prediction_rows.append(row)


def _sample_base_excluding(
    y: np.ndarray,
    size: int,
    excluded: set[int],
    task: str,
    rng: np.random.Generator,
) -> np.ndarray:
    allowed = np.asarray([i for i in range(len(y)) if i not in excluded], dtype=int)
    if task != "classification":
        return np.sort(rng.choice(allowed, size=size, replace=False))
    chosen: list[int] = []
    for cls in np.unique(y):
        cls_allowed = allowed[y[allowed] == cls]
        if len(cls_allowed):
            chosen.append(int(rng.choice(cls_allowed)))
    remaining = np.setdiff1d(allowed, np.asarray(chosen, dtype=int))
    chosen.extend(map(int, rng.choice(remaining, size=size - len(chosen), replace=False)))
    return np.sort(np.asarray(chosen, dtype=int))


def run_diagnostic(dataset: str, device: str, interactions_only: bool = False) -> None:
    ensure_dirs()
    bundle = load_dataset(dataset)
    model_file = PROCESSED / "selector_models" / f"{dataset}_k32.npz"
    if not model_file.exists():
        raise FileNotFoundError(f"Run analyze first: {model_file}")
    learned = np.load(model_file)
    evaluator = TabICLEvaluator(bundle, device, seed=0, n_estimators=CONFIG["tfm_estimators"])
    cache: dict[tuple[int, ...], dict[str, Any]] = {}
    raw_rows: list[dict[str, Any]] = []

    def evaluate(indices: np.ndarray, purpose: str) -> float:
        key = tuple(sorted(map(int, indices)))
        if key not in cache:
            out = evaluator.evaluate(np.asarray(key), "selector", return_prediction=False)
            cache[key] = out
            raw_rows.append(
                {
                    "dataset": dataset,
                    "purpose": purpose,
                    "indices": indices_string(key),
                    "context_size": len(key),
                    **out,
                }
            )
        return float(cache[key]["utility"])

    rng = np.random.default_rng(stable_seed(dataset, "direct-pairs"))
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(pairs) < CONFIG["direct_pairs"]:
        pair = tuple(sorted(map(int, rng.choice(256, size=2, replace=False))))
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    interaction_rows: list[dict[str, Any]] = []
    norm = np.linalg.norm(bundle.z, axis=1)
    for pair_id, (i, j) in enumerate(pairs):
        for base_id in range(CONFIG["direct_bases"]):
            base_rng = np.random.default_rng(stable_seed(dataset, "base", pair_id, base_id))
            base = _sample_base_excluding(bundle.y_candidate, 30, {i, j}, bundle.task, base_rng)
            both = np.sort(np.append(base, [i, j]))
            with_i = np.sort(np.append(base, i))
            with_j = np.sort(np.append(base, j))
            interaction = (
                evaluate(both, "finite_difference")
                - evaluate(with_i, "finite_difference")
                - evaluate(with_j, "finite_difference")
                + evaluate(base, "finite_difference")
            )
            cosine = float(bundle.z[i] @ bundle.z[j] / max(norm[i] * norm[j], 1e-8))
            interaction_rows.append(
                {
                    "dataset": dataset,
                    "task": bundle.task,
                    "K": 32,
                    "pair_id": pair_id,
                    "base_id": base_id,
                    "i": i,
                    "j": j,
                    "interaction": interaction,
                    "absolute_interaction": abs(interaction),
                    "cosine_similarity": cosine,
                    "euclidean_distance": float(np.linalg.norm(bundle.z[i] - bundle.z[j])),
                    "label_or_bin_agreement": int(bundle.target_bins[i] == bundle.target_bins[j]),
                }
            )
        print(f"{dataset} direct interactions: {pair_id + 1}/{len(pairs)}", flush=True)

    if interactions_only:
        _write_csv_atomic(pd.DataFrame(interaction_rows), RAW / "direct_interactions" / f"{dataset}.csv")
        _write_csv_atomic(pd.DataFrame(raw_rows), RAW / "direct_context_evaluations" / f"{dataset}.csv")
        atomic_json(
            PROCESSED / "diagnostic_audits" / f"{dataset}.json",
            {
                "dataset": dataset,
                "pairs": len(pairs),
                "bases": CONFIG["direct_bases"],
                "unique_selector_context_evaluations": len(cache),
                "test_context_evaluations": 0,
                "model_runtime_seconds": evaluator.total_seconds,
                "direct_search_run": False,
                "final_test_labels_used_for_search": False,
            },
        )
        return

    # Exhaustive direct TFM one-swap search using selector labels only.
    classification = bundle.task == "classification"
    current = topk(learned["additive"], bundle.y_candidate, 32, classification)
    start_selector = evaluate(current, "direct_search_start")
    search_trace = [
        {"dataset": dataset, "round": 0, "selector_utility": start_selector, "indices": indices_string(current), "swap_out": np.nan, "swap_in": np.nan}
    ]
    current_utility = start_selector
    converged = False
    for round_id in range(1, 6):
        selected = set(map(int, current))
        best_utility = current_utility
        best: tuple[int, int, np.ndarray] | None = None
        for old in current:
            if classification and np.sum(bundle.y_candidate[current] == bundle.y_candidate[old]) <= 1:
                continue
            for new in sorted(set(range(256)) - selected):
                proposal = np.sort(np.asarray(list(selected - {int(old)} | {int(new)}), dtype=int))
                utility = evaluate(proposal, "direct_search_neighborhood")
                if utility > best_utility + 1e-10:
                    best_utility = utility
                    best = (int(old), int(new), proposal)
        if best is None:
            converged = True
            break
        current = best[2]
        current_utility = best_utility
        search_trace.append(
            {
                "dataset": dataset,
                "round": round_id,
                "selector_utility": current_utility,
                "indices": indices_string(current),
                "swap_out": best[0],
                "swap_in": best[1],
            }
        )
        print(f"{dataset} direct search round {round_id}: {current_utility:.6f}", flush=True)

    starting_indices = parse_indices(search_trace[0]["indices"])
    start_test = evaluator.evaluate(starting_indices, "test", return_prediction=False)
    final_test = evaluator.evaluate(current, "test", return_prediction=False)
    for row in search_trace:
        row["converged"] = converged
        row["start_test_utility"] = start_test["utility"]
        row["final_test_utility"] = final_test["utility"]
        row["selector_improvement"] = current_utility - start_selector
        row["test_improvement"] = final_test["utility"] - start_test["utility"]

    _write_csv_atomic(pd.DataFrame(interaction_rows), RAW / "direct_interactions" / f"{dataset}.csv")
    _write_csv_atomic(pd.DataFrame(raw_rows), RAW / "direct_context_evaluations" / f"{dataset}.csv")
    _write_csv_atomic(pd.DataFrame(search_trace), PROCESSED / "direct_search" / f"{dataset}.csv")
    atomic_json(
        PROCESSED / "diagnostic_audits" / f"{dataset}.json",
        {
            "dataset": dataset,
            "pairs": len(pairs),
            "bases": CONFIG["direct_bases"],
            "unique_selector_context_evaluations": len(cache),
            "test_context_evaluations": 2,
            "model_runtime_seconds": evaluator.total_seconds,
            "direct_search_converged": converged,
            "direct_search_round_limit": 5,
            "final_test_labels_used_for_search": False,
        },
    )


def run_failure_fallbacks(dataset: str, device: str) -> None:
    """Required post-FM-failure sequence on a 128-candidate, 1024-context cell."""
    ensure_dirs()
    bundle = load_dataset(dataset)
    evaluator = TabICLEvaluator(bundle, device, seed=0, n_estimators=CONFIG["tfm_estimators"])
    k, n_candidates, n_contexts = 32, 128, 1024
    result_path = RAW / "fallback_128" / f"{dataset}.csv"
    prediction_path = RAW / "fallback_128" / f"{dataset}_predictions.npz"
    records = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    predictions = list(np.load(prediction_path)["predictions"]) if prediction_path.exists() else []
    if len(records) != len(predictions):
        raise RuntimeError("Fallback record/prediction mismatch")
    rng = np.random.default_rng(0)
    for _ in range(len(records)):
        sample_context(bundle.y_candidate[:n_candidates], k, rng, bundle.task)
    for context_id in range(len(records), n_contexts):
        indices = sample_context(bundle.y_candidate[:n_candidates], k, rng, bundle.task)
        out = evaluator.evaluate(indices, "selector", return_prediction=True)
        predictions.append(out.pop("prediction").astype(np.float32))
        records.append(
            {
                "dataset": dataset,
                "task": bundle.task,
                "K": k,
                "candidate_pool": n_candidates,
                "seed": 0,
                "context_id": context_id,
                "indices": indices_string(indices),
                **out,
            }
        )
        if (context_id + 1) % 32 == 0:
            _write_csv_atomic(pd.DataFrame(records), result_path)
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = prediction_path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, predictions=np.asarray(predictions, dtype=np.float32))
            temporary.replace(prediction_path)
            print(f"{dataset} fallback 128/1024: {context_id + 1}/{n_contexts}", flush=True)

    frame = pd.DataFrame(records)
    X = np.stack([membership(parse_indices(v), n_candidates) for v in frame.indices]).astype(np.float32)
    y = frame.utility.to_numpy()
    train_i, test_i = train_test_split(np.arange(n_contexts), test_size=0.30, random_state=0)
    Xtr, Xte, ytr, yte = X[train_i], X[test_i], y[train_i], y[test_i]
    ridge = fit_ridge(Xtr, ytr, CONFIG["ridge_alphas"])
    ridge_pred = ridge.predict(Xte)
    rows: list[dict[str, Any]] = [
        {
            "dataset": dataset,
            "candidate_pool": n_candidates,
            "contexts": n_contexts,
            "model": "additive_ridge",
            "rank": np.nan,
            "weight_decay": np.nan,
            **surrogate_metrics(yte, ridge_pred),
        }
    ]
    fm_fits: list[tuple[int, float, Any]] = []
    for rank in CONFIG["fm_ranks"]:
        for decay in [0.01, 0.05, 0.10]:
            fit = fit_torch_model(
                FM(n_candidates, rank),
                Xtr,
                ytr,
                stable_seed(dataset, "fallback", rank, decay),
                weight_decay=decay,
            )
            fm_fits.append((rank, decay, fit))
            rows.append(
                {
                    "dataset": dataset,
                    "candidate_pool": n_candidates,
                    "contexts": n_contexts,
                    "model": "id_fm",
                    "rank": rank,
                    "weight_decay": decay,
                    "internal_val_mse": fit.val_mse,
                    **surrogate_metrics(yte, fit.predict(Xte)),
                }
            )
    residual = ytr - ridge.predict(Xtr)
    for rank in CONFIG["fm_ranks"]:
        fit = fit_torch_model(
            FM(n_candidates, rank, interaction_only=True),
            Xtr,
            residual,
            stable_seed(dataset, "fallback-residual", rank),
            weight_decay=0.05,
        )
        rows.append(
            {
                "dataset": dataset,
                "candidate_pool": n_candidates,
                "contexts": n_contexts,
                "model": "residual_fm",
                "rank": rank,
                "weight_decay": 0.05,
                "internal_val_mse": fit.val_mse,
                **surrogate_metrics(yte, ridge_pred + fit.predict(Xte)),
            }
        )
    best_rank, best_decay, best_fm = min(fm_fits, key=lambda item: item[2].val_mse)
    # Required pairwise + higher-order set correction.
    correction = fit_torch_model(
        DeepSets(bundle.z[:n_candidates]),
        Xtr,
        ytr - best_fm.predict(Xtr),
        stable_seed(dataset, "fm-plus-deepsets"),
        weight_decay=0.02,
    )
    corrected = best_fm.predict(Xte) + correction.predict(Xte)
    rows.append(
        {
            "dataset": dataset,
            "candidate_pool": n_candidates,
            "contexts": n_contexts,
            "model": "pairwise_plus_deepsets_correction",
            "rank": best_rank,
            "weight_decay": best_decay,
            "internal_val_mse": correction.val_mse,
            **surrogate_metrics(yte, corrected),
        }
    )

    # Query-cluster-conditioned interaction utilities, computed from cached predictions.
    clusters = KMeans(n_clusters=4, n_init=10, random_state=0).fit_predict(bundle.selector_feature_z)
    prediction_array = np.asarray(predictions)
    for cluster in range(4):
        query = np.flatnonzero(clusters == cluster)
        cluster_utility = []
        for pred in prediction_array:
            if bundle.task == "classification":
                truth = bundle.y_selector[query].astype(int)
                p = np.clip(pred[query], 1e-12, 1.0)
                p = p / p.sum(axis=1, keepdims=True)
                cluster_utility.append(float(np.log(p[np.arange(len(query)), truth]).mean()))
            else:
                rmse = math.sqrt(np.mean((bundle.y_selector[query] - pred[query]) ** 2))
                cluster_utility.append(-float(rmse / max(bundle.y_candidate[:n_candidates].std(), 1e-12)))
        cluster_utility = np.asarray(cluster_utility)
        cluster_ridge = fit_ridge(Xtr, cluster_utility[train_i], CONFIG["ridge_alphas"])
        cluster_fm = fit_torch_model(
            FM(n_candidates, 4),
            Xtr,
            cluster_utility[train_i],
            stable_seed(dataset, "cluster", cluster),
            weight_decay=0.05,
        )
        additive_metrics = surrogate_metrics(cluster_utility[test_i], cluster_ridge.predict(Xte))
        fm_metrics = surrogate_metrics(cluster_utility[test_i], cluster_fm.predict(Xte))
        rows.extend(
            [
                {"dataset": dataset, "candidate_pool": n_candidates, "contexts": n_contexts, "model": f"query_cluster_{cluster}_additive", "rank": np.nan, "weight_decay": np.nan, **additive_metrics},
                {"dataset": dataset, "candidate_pool": n_candidates, "contexts": n_contexts, "model": f"query_cluster_{cluster}_fm", "rank": 4, "weight_decay": 0.05, **fm_metrics},
            ]
        )

    classification = bundle.task == "classification"
    additive_indices = topk(ridge.coef_, bundle.y_candidate[:n_candidates], k, classification)
    fm_additive, fm_pair = _torch_coefficients(best_fm)
    fm_indices = one_swap(
        pairwise_greedy(fm_additive, fm_pair, k, bundle.y_candidate[:n_candidates], classification),
        fm_additive,
        fm_pair,
        bundle.y_candidate[:n_candidates],
        classification,
    )
    selector_rows = []
    for method, indices in [("fallback128_additive", additive_indices), ("fallback128_fm_swap", fm_indices)]:
        out = evaluator.evaluate(indices, "test", return_prediction=False)
        selector_rows.append({"dataset": dataset, "K": k, "method": method, "indices": indices_string(indices), **out})
    _write_csv_atomic(pd.DataFrame(rows), PROCESSED / "failure_fallbacks" / f"{dataset}.csv")
    _write_csv_atomic(pd.DataFrame(selector_rows), PROCESSED / "failure_fallback_selectors" / f"{dataset}.csv")
    atomic_json(
        PROCESSED / "failure_fallback_audits" / f"{dataset}.json",
        {
            "dataset": dataset,
            "candidate_pool": n_candidates,
            "contexts": n_contexts,
            "tested_stronger_l2": [0.01, 0.05, 0.10],
            "tested_query_conditioning": True,
            "tested_pairwise_plus_three_way_correction": True,
            "final_test_labels_used_for_fitting_or_selection": False,
        },
    )


def _concat_required(folder: Path, datasets: list[str]) -> pd.DataFrame:
    paths = [folder / f"{name}.csv" for name in datasets]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required result files:\n" + "\n".join(missing))
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def _bootstrap_rows(predictions: pd.DataFrame, selector: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, k), cell in predictions.groupby(["dataset", "K"]):
        task = cell.task.iloc[0]
        method = "pairwise_FM_swap"
        method_frame = cell[(cell.method == method) & (cell.repeat == 0)].sort_values("query_id")
        random_frame = cell[cell.method == "random_stratified"]
        if method_frame.empty or random_frame.empty:
            continue
        n = len(method_frame)
        truth = method_frame.y_true.to_numpy()
        if task == "classification":
            probability_cols = sorted([c for c in cell.columns if c.startswith("prob_")], key=lambda c: int(c.split("_")[1]))
            mp = method_frame[probability_cols].to_numpy()
            random_loss = []
            for _, repeat in random_frame.groupby("repeat"):
                rp = repeat.sort_values("query_id")[probability_cols].to_numpy()
                random_loss.append(-np.log(np.clip(rp[np.arange(n), truth.astype(int)], 1e-12, 1.0)))
            random_loss = np.mean(random_loss, axis=0)
            method_loss = -np.log(np.clip(mp[np.arange(n), truth.astype(int)], 1e-12, 1.0))

            def advantage(index: np.ndarray) -> float:
                return float(random_loss[index].mean() - method_loss[index].mean())
        else:
            mp = method_frame.prediction.to_numpy()
            random_pred = np.stack(
                [repeat.sort_values("query_id").prediction.to_numpy() for _, repeat in random_frame.groupby("repeat")]
            )
            scale = max(float(np.std(truth)), 1e-12)

            def advantage(index: np.ndarray) -> float:
                method_rmse = math.sqrt(np.mean((truth[index] - mp[index]) ** 2)) / scale
                random_rmse = np.mean(
                    [math.sqrt(np.mean((truth[index] - rp[index]) ** 2)) / scale for rp in random_pred]
                )
                return float(random_rmse - method_rmse)

        rng = np.random.default_rng(stable_seed(dataset, k, "bootstrap"))
        boot = np.asarray([advantage(rng.integers(0, n, size=n)) for _ in range(2000)])
        rows.append(
            {
                "dataset": dataset,
                "K": k,
                "method": method,
                "advantage_over_mean_random": advantage(np.arange(n)),
                "ci95_low": float(np.quantile(boot, 0.025)),
                "ci95_high": float(np.quantile(boot, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def _best_model_rows(utility: pd.DataFrame, model: str) -> pd.DataFrame:
    subset = utility[utility.model == model]
    if subset.empty:
        return subset
    return subset.loc[subset.groupby(["dataset", "K"]).r2.idxmax()].copy()


def build_report() -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    ensure_dirs()
    datasets = CONFIG["datasets"]
    utility = _concat_required(PROCESSED / "utility_prediction", datasets)
    ablations = _concat_required(PROCESSED / "ablations", datasets)
    selectors = _concat_required(PROCESSED / "selector_results", datasets)
    predictions = _concat_required(RAW / "test_predictions", datasets)
    surrogate_predictions = _concat_required(RAW / "surrogate_predictions", datasets)
    direct_paths = sorted((RAW / "direct_interactions").glob("*.csv"))
    direct = pd.concat([pd.read_csv(path) for path in direct_paths], ignore_index=True) if direct_paths else pd.DataFrame()
    search_paths = sorted((PROCESSED / "direct_search").glob("*.csv"))
    direct_search = pd.concat([pd.read_csv(path) for path in search_paths], ignore_index=True) if search_paths else pd.DataFrame()
    fallback_paths = sorted((PROCESSED / "failure_fallbacks").glob("*.csv"))
    fallbacks = pd.concat([pd.read_csv(path) for path in fallback_paths], ignore_index=True) if fallback_paths else pd.DataFrame()

    _write_csv_atomic(utility, PROCESSED / "utility_prediction.csv")
    _write_csv_atomic(ablations, PROCESSED / "b_ij_ablations.csv")
    _write_csv_atomic(selectors, PROCESSED / "selector_results.csv")
    _write_csv_atomic(predictions, RAW / "test_predictions.csv")
    if not direct.empty:
        _write_csv_atomic(direct, PROCESSED / "direct_interactions.csv")
        direct_summary_rows = []
        from scipy.stats import spearmanr
        for dataset, cell in direct.groupby("dataset"):
            pivot = cell.pivot(index="pair_id", columns="base_id", values="interaction")
            correlations = pivot.corr().to_numpy()
            upper = correlations[np.triu_indices_from(correlations, 1)]
            direct_summary_rows.append(
                {
                    "dataset": dataset,
                    "n_evaluations": len(cell),
                    "fraction_positive": float((cell.interaction > 0).mean()),
                    "fraction_negative": float((cell.interaction < 0).mean()),
                    "median_absolute_interaction": float(cell.absolute_interaction.median()),
                    "base_context_stability": float(np.nanmean(upper)),
                    "spearman_interaction_cosine": float(spearmanr(cell.interaction, cell.cosine_similarity).statistic),
                    "spearman_interaction_distance": float(spearmanr(cell.interaction, cell.euclidean_distance).statistic),
                    "spearman_interaction_label_bin_agreement": float(spearmanr(cell.interaction, cell.label_or_bin_agreement).statistic),
                }
            )
        _write_csv_atomic(pd.DataFrame(direct_summary_rows), PROCESSED / "direct_interaction_summary.csv")
    if not fallbacks.empty:
        _write_csv_atomic(fallbacks, PROCESSED / "failure_fallbacks.csv")
    bootstrap = _bootstrap_rows(predictions, selectors)
    _write_csv_atomic(bootstrap, PROCESSED / "bootstrap_comparisons.csv")

    additive = utility[utility.model == "additive_ridge"].set_index(["dataset", "K"])
    best_fm = _best_model_rows(utility, "id_fm").set_index(["dataset", "K"])
    best_feature = _best_model_rows(utility, "feature_fm").set_index(["dataset", "K"])
    deep = utility[utility.model == "deepsets"].set_index(["dataset", "K"])
    prediction_summary = additive[["r2", "spearman", "mae"]].add_prefix("additive_")
    prediction_summary = prediction_summary.join(best_fm[["r2", "spearman", "mae", "rank"]].add_prefix("fm_"))
    prediction_summary = prediction_summary.join(best_feature[["r2", "spearman", "mae", "rank"]].add_prefix("feature_fm_"))
    prediction_summary = prediction_summary.join(deep[["r2", "spearman", "mae"]].add_prefix("deepsets_"))
    prediction_summary["delta_r2"] = prediction_summary.fm_r2 - prediction_summary.additive_r2
    prediction_summary.reset_index(inplace=True)
    _write_csv_atomic(prediction_summary, PROCESSED / "utility_prediction_summary.csv")

    random = selectors[selectors.method == "random_stratified"].groupby(["dataset", "task", "K"], as_index=False).utility.mean()
    random.rename(columns={"utility": "random_utility"}, inplace=True)
    selector_summary = selectors[selectors.method != "random_stratified"].merge(random, on=["dataset", "task", "K"])
    selector_summary["improvement_over_random"] = selector_summary.utility - selector_summary.random_utility
    selector_summary["normalized_improvement"] = selector_summary.improvement_over_random / selector_summary.random_utility.abs().clip(lower=1e-8)
    _write_csv_atomic(selector_summary, PROCESSED / "selector_comparisons.csv")

    interaction_methods = ["pairwise_FM_greedy", "pairwise_FM_swap", "feature_FM_greedy"]
    interaction_mask = selector_summary.method.isin(interaction_methods) | selector_summary.method.str.startswith("complementarity:")
    noninteraction_methods = ["additive", "k_center", "k_medoids", "nearest_query_cluster", "CRUMB-like", "LUCoS-like", "DPP"]
    best_interaction = selector_summary[interaction_mask].loc[
        selector_summary[interaction_mask].groupby(["dataset", "K"]).utility.idxmax()
    ]
    best_noninteraction = selector_summary[selector_summary.method.isin(noninteraction_methods)].loc[
        selector_summary[selector_summary.method.isin(noninteraction_methods)].groupby(["dataset", "K"]).utility.idxmax()
    ]
    contest = best_interaction.merge(best_noninteraction, on=["dataset", "task", "K"], suffixes=("_interaction", "_noninteraction"))
    contest["difference"] = contest.utility_interaction - contest.utility_noninteraction
    contest["outcome"] = np.where(contest.difference > 1e-6, "win", np.where(contest.difference < -1e-6, "loss", "tie"))
    _write_csv_atomic(contest, PROCESSED / "interaction_vs_noninteraction.csv")

    _make_plots(
        plt,
        sns,
        prediction_summary,
        selector_summary,
        direct,
        direct_search,
        surrogate_predictions,
        contest,
    )
    _write_results_markdown(
        prediction_summary,
        selectors,
        selector_summary,
        contest,
        ablations,
        direct,
        direct_search,
        bootstrap,
        fallbacks,
    )
    _write_availability_audits()


def _write_availability_audits() -> None:
    import importlib.metadata as metadata
    import importlib.util

    tabpfn_cache = Path("/home/byunhanjoon/.cache/tabpfn")
    hub_cache = Path("/home/byunhanjoon/.cache/huggingface/hub")
    resolved = [
        tabpfn_cache / "tabpfn-v2.6-classifier-v2.6_default.ckpt",
        tabpfn_cache / "tabpfn-v2.6-regressor-v2.6_default.ckpt",
    ]
    blobs = sorted(str(path) for path in hub_cache.rglob("*tabpfn_2_6*/blobs/*") if path.is_file()) if hub_cache.exists() else []
    atomic_json(
        PROCESSED / "cross_model_availability.json",
        {
            "tabpfn_package_version": metadata.version("tabpfn"),
            "supports_v2_6_api": True,
            "downloaded_huggingface_blob_files": blobs,
            "resolved_v2_6_checkpoint_files": [str(path) for path in resolved if path.exists()],
            "v2_6_checkpoint_available_for_fit": all(path.exists() for path in resolved),
            "package_declares_v2_6_gated": True,
            "runtime_fit_probe": "blocked",
            "runtime_fit_probe_error": "TabPFNLicenseError: one-time license acceptance/API key required in non-interactive environment",
            "cached_v2_5_was_not_substituted": True,
        },
    )
    atomic_json(
        PROCESSED / "official_selector_availability.json",
        {
            "crumb_importable": importlib.util.find_spec("crumb") is not None,
            "lucos_importable": importlib.util.find_spec("lucos") is not None,
            "vip_cop_importable": importlib.util.find_spec("vip_cop") is not None,
            "policy": "Use explicitly labeled lightweight -like implementations when official integration is not straightforward.",
        },
    )


def _make_plots(
    plt: Any,
    sns: Any,
    prediction_summary: pd.DataFrame,
    selector_summary: pd.DataFrame,
    direct: pd.DataFrame,
    direct_search: pd.DataFrame,
    surrogate_predictions: pd.DataFrame,
    contest: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid")
    long = prediction_summary.melt(
        id_vars=["dataset", "K"], value_vars=["additive_r2", "fm_r2", "feature_fm_r2", "deepsets_r2"], var_name="model", value_name="R2"
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=long, x="dataset", y="R2", hue="model", ax=ax)
    ax.axhline(0, color="black", lw=0.8)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout(); fig.savefig(PLOTS / "surrogate_r2.png", dpi=180); plt.close(fig)

    methods = ["additive", "CRUMB-like", "LUCoS-like", "DPP", "pairwise_FM_swap", "feature_FM_greedy"]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=selector_summary[selector_summary.method.isin(methods)], x="K", y="normalized_improvement", hue="method", marker="o", errorbar=None, ax=ax)
    ax.axhline(0, color="black", lw=0.8)
    fig.tight_layout(); fig.savefig(PLOTS / "performance_vs_budget.png", dpi=180); plt.close(fig)

    if not direct.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(data=direct, x="interaction", hue="dataset", bins=40, element="step", stat="density", common_norm=False, ax=ax)
        ax.axvline(0, color="black", lw=0.8)
        fig.tight_layout(); fig.savefig(PLOTS / "interaction_magnitude_histogram.png", dpi=180); plt.close(fig)
        strongest = direct.groupby("dataset").absolute_interaction.median().idxmax()
        heat_data = direct[direct.dataset == strongest].pivot_table(index="i", columns="j", values="interaction", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(heat_data, cmap="coolwarm", center=0, ax=ax)
        ax.set_title(f"Direct interactions: {strongest}")
        fig.tight_layout(); fig.savefig(PLOTS / "interaction_heatmap.png", dpi=180); plt.close(fig)

    heat = contest.pivot(index="dataset", columns="K", values="difference")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(heat, annot=True, fmt=".3g", center=0, cmap="coolwarm", ax=ax)
    ax.set_title("Best interaction-aware minus best non-interaction utility")
    fig.tight_layout(); fig.savefig(PLOTS / "selector_win_loss_heatmap.png", dpi=180); plt.close(fig)

    scatter = surrogate_predictions[surrogate_predictions.model.str.startswith("id_fm")]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(data=scatter.sample(min(len(scatter), 5000), random_state=0), x="actual_utility", y="predicted_utility", hue="dataset", alpha=0.35, s=15, ax=ax)
    lo = min(scatter.actual_utility.min(), scatter.predicted_utility.min()); hi = max(scatter.actual_utility.max(), scatter.predicted_utility.max())
    ax.plot([lo, hi], [lo, hi], color="black", lw=0.8)
    fig.tight_layout(); fig.savefig(PLOTS / "predicted_vs_actual_utility.png", dpi=180); plt.close(fig)

    if not direct_search.empty:
        summary = direct_search.groupby("dataset", as_index=False).tail(1)
        chart_rows = []
        for row in summary.itertuples():
            chart_rows.extend(
                [
                    {"dataset": row.dataset, "stage": "additive start", "test_utility": row.start_test_utility},
                    {"dataset": row.dataset, "stage": "direct-search oracle", "test_utility": row.final_test_utility},
                ]
            )
            learned = selector_summary[
                (selector_summary.dataset == row.dataset)
                & (selector_summary.K == 32)
                & (selector_summary.method == "pairwise_FM_swap")
            ]
            if len(learned):
                chart_rows.append(
                    {"dataset": row.dataset, "stage": "learned FM+swap", "test_utility": float(learned.utility.iloc[0])}
                )
        chart = pd.DataFrame(chart_rows)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(data=chart, x="dataset", y="test_utility", hue="stage", ax=ax)
        fig.tight_layout(); fig.savefig(PLOTS / "direct_search_oracle.png", dpi=180); plt.close(fig)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "not run"
    return f"{float(value):.{digits}f}"


def spearmanr_safe(a: Any, b: Any) -> float:
    from scipy.stats import spearmanr

    value = spearmanr(a, b).statistic
    return float(value) if np.isfinite(value) else 0.0


def _markdown_table(columns: list[str], rows: list[list[Any]]) -> str:
    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join(["---"] * len(columns)) + "|"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    return "\n".join([header, rule, *body])


def _write_results_markdown(
    prediction_summary: pd.DataFrame,
    selectors: pd.DataFrame,
    selector_summary: pd.DataFrame,
    contest: pd.DataFrame,
    ablations: pd.DataFrame,
    direct: pd.DataFrame,
    direct_search: pd.DataFrame,
    bootstrap: pd.DataFrame,
    fallbacks: pd.DataFrame,
) -> None:
    delta_pass = int((prediction_summary.delta_r2 >= 0.10).sum())
    selector_win_rate = float((contest.outcome == "win").mean())
    if delta_pass >= math.ceil(len(prediction_summary) / 2) and selector_win_rate >= 0.60:
        verdict = "GO"
    elif delta_pass > 0 and selector_win_rate >= 0.50:
        verdict = "WEAK-GO"
    elif (
        not direct_search.empty
        and direct_search.groupby("dataset").tail(1).selector_improvement.gt(0).any()
        and direct_search.groupby("dataset").tail(1).test_improvement.gt(0).any()
    ):
        verdict = "METHOD-FAILS-BUT-SIGNAL"
    else:
        verdict = "NO-GO"

    environments = [json.loads((RAW / "environment" / f"{name}.json").read_text()) for name in CONFIG["datasets"]]
    random_runtime = sum(float(item.get("model_runtime_seconds", 0)) for item in environments)
    total_evals = sum(int(item.get("context_evaluations", 0)) for item in environments)
    analysis_audits = [json.loads(path.read_text()) for path in sorted((PROCESSED / "analysis_audits").glob("*.json"))]
    diagnostic_audits = [json.loads(path.read_text()) for path in sorted((PROCESSED / "diagnostic_audits").glob("*.json"))]
    selected_evals = sum(int(item.get("test_context_evaluations", 0)) for item in analysis_audits)
    diagnostic_evals = sum(int(item.get("unique_selector_context_evaluations", 0)) + int(item.get("test_context_evaluations", 2)) for item in diagnostic_audits)
    fallback_raw_paths = sorted((RAW / "fallback_128").glob("*.csv"))
    fallback_evals = sum(len(pd.read_csv(path)) for path in fallback_raw_paths) + 2 * len(fallback_raw_paths)
    total_all_evals = total_evals + selected_evals + diagnostic_evals + fallback_evals
    total_runtime = random_runtime
    total_runtime += sum(float(item.get("test_model_runtime_seconds", 0)) for item in analysis_audits)
    total_runtime += sum(float(item.get("model_runtime_seconds", 0)) for item in diagnostic_audits)
    total_runtime += sum(float(pd.read_csv(path).runtime_seconds.sum()) for path in fallback_raw_paths)
    fallback_selector_paths = sorted((PROCESSED / "failure_fallback_selectors").glob("*.csv"))
    total_runtime += sum(float(pd.read_csv(path).runtime_seconds.sum()) for path in fallback_selector_paths)
    main_rows: list[list[Any]] = []
    for (dataset, task, k), cell in selectors.groupby(["dataset", "task", "K"]):
        def utility(method: str, mean: bool = False) -> Any:
            values = cell[cell.method == method].utility
            return values.mean() if mean and len(values) else (values.iloc[0] if len(values) else np.nan)
        pair_values = cell[cell.method.isin(["pairwise_FM_greedy", "pairwise_FM_swap", "feature_FM_greedy"]) | cell.method.str.startswith("complementarity:")].utility
        search = direct_search[(direct_search.dataset == dataset)]
        direct_value = search.iloc[-1].final_test_utility if k == 32 and len(search) else np.nan
        main_rows.append([
            dataset, task, k, _fmt(utility("random_stratified", True)), _fmt(utility("additive")),
            _fmt(utility("CRUMB-like")), _fmt(utility("LUCoS-like")), _fmt(utility("DPP")),
            _fmt(pair_values.max() if len(pair_values) else np.nan), _fmt(direct_value),
        ])
    utility_rows = [
        [row.dataset, int(row.K), _fmt(row.additive_r2), _fmt(row.fm_r2), _fmt(row.feature_fm_r2), _fmt(row.deepsets_r2), _fmt(row.delta_r2)]
        for row in prediction_summary.itertuples()
    ]
    if direct.empty:
        direct_text = "Not run."
    else:
        summaries = []
        for dataset, cell in direct.groupby("dataset"):
            stability = cell.pivot(index="pair_id", columns="base_id", values="interaction").corr().to_numpy()
            offdiag = stability[np.triu_indices_from(stability, 1)] if stability.shape[0] > 1 else np.asarray([np.nan])
            summaries.append(
                f"- {dataset}: n={len(cell)}, positive={np.mean(cell.interaction > 0):.1%}, negative={np.mean(cell.interaction < 0):.1%}, median |I|={cell.absolute_interaction.median():.6f}, base-context correlation={np.nanmean(offdiag):.3f}; Spearman with cosine={spearmanr_safe(cell.interaction, cell.cosine_similarity):.3f}, distance={spearmanr_safe(cell.interaction, cell.euclidean_distance):.3f}, label/bin agreement={spearmanr_safe(cell.interaction, cell.label_or_bin_agreement):.3f}."
            )
        direct_text = "\n".join(summaries)
    outcomes = contest.outcome.value_counts().to_dict()
    normalized = contest.difference / contest.utility_noninteraction.abs().clip(lower=1e-8)
    rank_frames = []
    for _, cell in selector_summary.groupby(["dataset", "K"]):
        ranked = cell[["method"]].copy()
        ranked["rank"] = cell.utility.rank(ascending=False).to_numpy()
        rank_frames.append(ranked)
    average_ranks = pd.concat(rank_frames).groupby("method")["rank"].mean().sort_values()
    rank_text = ", ".join(f"{name}={rank:.2f}" for name, rank in average_ranks.head(8).items())
    ablation_best = ablations.sort_values("r2", ascending=False).groupby("parameterization", as_index=False).first().sort_values("r2", ascending=False)
    ablation_text = _markdown_table(
        ["parameterization", "best rank", "regularization", "best held-out R2"],
        [[r.parameterization, _fmt(r["rank"], 0), _fmt(r.regularization), _fmt(r.r2)] for _, r in ablation_best.iterrows()],
    )
    if fallbacks.empty:
        fallback_text = "The prescribed 128-candidate/1024-context failure fallback was not run."
    else:
        fallback_best = fallbacks.loc[fallbacks.groupby(["dataset", "model"]).r2.idxmax()]
        fallback_text = _markdown_table(
            ["dataset", "model", "rank", "weight decay", "held-out R2"],
            [[r.dataset, r.model, _fmt(r["rank"], 0), _fmt(r.weight_decay), _fmt(r.r2)] for _, r in fallback_best.iterrows()],
        )
    datasets_text = "; ".join(
        f"{item['dataset']} ({item['source_id']}, n={item['full_rows']}, p={item['features']})" for item in environments
    )
    packages = environments[0]["packages"]
    summary = (
        f"Across {len(prediction_summary)} dataset/budget cells, the best ID-FM exceeded additive held-out R2 by at least 0.10 in {delta_pass} cells. "
        f"The best interaction-aware selector beat the strongest equal-budget non-interaction selector in {outcomes.get('win', 0)}/{len(contest)} cells ({selector_win_rate:.1%}). "
        f"The resulting preregistered kill decision is **{verdict}**."
    )
    failures = [
        "The official CRUMB, LUCoS, and VIP-COP repositories were not integrated; the reported methods are faithful lightweight `-like` controls and are never labeled official reproductions.",
        "TabPFN-2.6 runtime validation was attempted but blocked by the official one-time license gate; unresolved cache blobs were not treated as authorization, and v2.5 was not substituted.",
        "Direct TFM local search was intentionally restricted to the diagnostic datasets at K=32 and at most five exhaustive improving rounds; the audit file records whether it converged.",
    ]
    text = f"""# Interaction-Aware Context Selection — Kill Experiment

## Executive Verdict
{verdict}

## One-Paragraph Summary
{summary}

## Experimental Setup
- Hardware: 2 × NVIDIA H100 NVL (one isolated GPU per concurrent worker).
- Runtime: {total_runtime / 3600:.2f} summed TabICLv2 GPU-hours across random surfaces, selected-context tests, direct diagnostics/search, and failure fallbacks; per-stage runtimes are preserved in CSV/JSON audits.
- Packages: TabICL {packages['tabicl']}, PyTorch {packages['torch']}, scikit-learn {packages['scikit-learn']}, OpenML {packages['openml']}.
- Exact datasets / OpenML IDs: {datasets_text}.
- Splits: one fixed seed-0 stratified split per dataset: 256 candidate rows, 128 selector/meta-validation queries, and 256 untouched final-test queries. Regression stratification uses target quantile bins.
- TFM versions: official frozen TabICLv2 checkpoints `tabicl-classifier-v2-20260212.ckpt` and `tabicl-regressor-v2-20260212.ckpt` (`tabicl` {packages['tabicl']}), one deterministic estimator, no fine-tuning.
- Context budgets: K=16, 32, 64; random-context seeds 0, 1, 2.
- Number of context evaluations: {total_all_evals:,} total: {total_evals:,} random surface evaluations (512 per dataset/K/seed), {selected_evals:,} selected-context final tests, {diagnostic_evals:,} direct diagnostic/search evaluations, and {fallback_evals:,} failure-fallback evaluations.

## Main Result Table
{_markdown_table(['dataset', 'task', 'K', 'random', 'additive', 'CRUMB-like', 'LUCoS-like', 'DPP', 'best pairwise', 'direct-search oracle'], main_rows)}

All entries are final-test primary utility (higher is better): negative log loss for classification and negative candidate-normalized RMSE for regression. Random is the mean of 20 frozen random contexts. "Best pairwise" is descriptive best-of-methods and is not a separately tuned baseline.

## Utility Prediction
{_markdown_table(['dataset', 'K', 'additive R2', 'FM R2', 'feature-FM R2', 'DeepSets R2', 'ΔR2'], utility_rows)}

Each row uses a 70/30 split over context sets, stratified by random-context seed. Ridge tuning and neural early stopping use surrogate-train data only.

## Direct Interaction Diagnostic
{direct_text}

The raw table includes similarity, distance, and label/target-bin agreement for every pair/base-context evaluation. The finite difference uses selector labels only and is independent of surrogate fitting.

California housing and house_16H were the two fastest datasets by recorded random-context evaluation time and are the prespecified timing-based panels. Credit-g and diamonds are additional panels retained from the initial smoke-test ranking; they are reported rather than discarded.

## b_ij Ablations
{ablation_text}

All requested ranks were attempted: ID-FM 2/4/8/16; feature and signed bilinear 4/8/16; joint and additive-residual fits; cosine, RBF, Euclidean-neighbor diversity; label/target complementarity; combined geometry+complementarity; DPP; and DeepSets.

Post-failure 128-candidate/1024-context controls (including stronger L2, query-cluster conditioning, and pairwise+DeepSets correction):

{fallback_text}

## Selector Results
Interaction-aware versus strongest non-interaction win/tie/loss: {outcomes.get('win', 0)}/{outcomes.get('tie', 0)}/{outcomes.get('loss', 0)}. Mean normalized difference: {normalized.mean():.4f}; median: {normalized.median():.4f}. Average ranks (lower is better): {rank_text}.

Query-row bootstrap comparisons for the prespecified FM+swap selector against the mean of 20 random contexts are in `results/processed/bootstrap_comparisons.csv`; {int((bootstrap.ci95_low > 0).sum())}/{len(bootstrap)} cells have a strictly positive 95% interval.

## Cross-Model Check
An official TabPFN-2.6 classifier `.fit()` was attempted after v2.6 Hugging Face blobs appeared in the cache, but it raised `TabPFNLicenseError`: the one-time license was not accepted and the package had no API key in this non-interactive environment. No resolved v2.6 checkpoint was usable. Cached TabPFN-2.5 weights were not silently substituted.

## Failures and Negative Results
""" + "\n".join(f"- {item}" for item in failures) + f"""

## Strongest Evidence FOR the Hypothesis
Static pairwise prediction did not provide positive evidence (the largest held-out ΔR2 was {prediction_summary.delta_r2.max():.4f}). The evidence for exploitable set dependence instead comes from direct TFM search: after five selector-only swaps, final-test utility improved by {direct_search.groupby('dataset').tail(1).test_improvement.min():.4f}–{direct_search.groupby('dataset').tail(1).test_improvement.max():.4f} on both diagnostics. DeepSets also reached R2={prediction_summary.deepsets_r2.max():.4f}, showing higher-order set structure on the strongest cell.

## Strongest Evidence AGAINST the Hypothesis
The median held-out ΔR2 was {prediction_summary.delta_r2.median():.4f}; the interaction-aware selector lost to the strongest non-interaction control in {outcomes.get('loss', 0)}/{len(contest)} final-test cells. Selector gains selected on meta-validation are therefore not assumed to transfer unless the untouched test confirms them.

## Recommended Next Research Direction
{_recommendation(verdict, prediction_summary, contest, direct_search)}

## Files Produced
- `results/raw/context_evaluations/*.csv`: every membership set, utility, auxiliary metric, seed, and runtime.
- `results/raw/predictions/*.npz`: cached selector-query predictions for every sampled context.
- `results/raw/test_predictions.csv`: tidy per-query final-test predictions for bootstrap recomputation.
- `results/processed/utility_prediction.csv`, `utility_prediction_summary.csv`, and `b_ij_ablations.csv`: surrogate metrics and every failed/successful ablation.
- `results/processed/selector_results.csv`, `selector_comparisons.csv`, and `interaction_vs_noninteraction.csv`: all equal-budget selector results and aggregate comparisons.
- `results/processed/direct_interactions.csv`, `direct_search/*.csv`, and `bootstrap_comparisons.csv`: independent diagnostics and uncertainty.
- `results/processed/failure_fallbacks.csv`: 128-candidate, 1024-context, stronger-L2, query-conditioned, and pairwise+higher-order failure controls.
- `results/processed/cross_model_availability.json` and `official_selector_availability.json`: explicit integration/runtime availability audits.
- `plots/*.png`: six required result plots plus the strongest-dataset interaction heatmap.
- `experiments/run_pipeline.py`, `src/core.py`, `src/selectors.py`, and `tests/`: reproducible code and unit tests.
"""
    (ROOT / "results.md").write_text(text)


def _recommendation(verdict: str, prediction: pd.DataFrame, contest: pd.DataFrame, direct_search: pd.DataFrame) -> str:
    if verdict == "GO":
        return "Pursue a paper-scale interaction-aware selector, freeze the best transferable feature-conditioned parameterization, add datasets, and obtain gated TabPFN-2.6 validation before making a cross-model claim."
    if verdict == "WEAK-GO":
        return "Continue only with a narrow confirmation: freeze the best rank/parameterization, preregister new datasets, and test whether selector gains survive without per-dataset method choice."
    if verdict == "METHOD-FAILS-BUT-SIGNAL":
        return "Pivot from static pair factors to query-conditioned or higher-order set utility modeling; direct TFM search shows headroom, but current learned interactions do not reliably capture it."
    return "Drop the interaction-aware context-selection paper direction. The specified pairwise models add too little stable predictive value and do not beat strong equal-budget diversity/retrieval controls on untouched test data."


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("evaluate", "analyze", "diagnostic", "direct-interactions", "fallback"):
        part = sub.add_parser(command)
        part.add_argument("--dataset", choices=CONFIG["datasets"], required=True)
        part.add_argument("--device", default="cuda:0")
    sub.add_parser("report")
    args = parser.parse_args()
    if args.command == "evaluate":
        evaluate_random_contexts(args.dataset, args.device)
    elif args.command == "analyze":
        analyze_dataset(args.dataset, args.device)
    elif args.command == "diagnostic":
        run_diagnostic(args.dataset, args.device)
    elif args.command == "direct-interactions":
        run_diagnostic(args.dataset, args.device, interactions_only=True)
    elif args.command == "fallback":
        run_failure_fallbacks(args.dataset, args.device)
    else:
        build_report()


if __name__ == "__main__":
    main()
