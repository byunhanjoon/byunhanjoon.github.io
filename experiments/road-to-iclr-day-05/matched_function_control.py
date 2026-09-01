"""Exact matched-initial-function control for Day-5 schema nuisances."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from tier1_orbit import encode_categories, load_dataset


HERE = Path(__file__).resolve().parent
CONFIG = HERE / "matched_function_config.json"


class EmbeddingMLP(nn.Module):
    def __init__(self, numerical: int, cardinalities: list[int], embedding_dim: int, hidden: int) -> None:
        super().__init__()
        self.numerical = numerical
        self.embedding_dim = embedding_dim
        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality + 1, embedding_dim) for cardinality in cardinalities
        )
        width = numerical + embedding_dim * len(cardinalities)
        self.hidden = nn.Linear(width, hidden)
        self.output = nn.Linear(hidden, 2)

    def forward(self, numerical: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        parts = [numerical]
        parts.extend(embedding(categorical[:, index]) for index, embedding in enumerate(self.embeddings))
        return self.output(torch.relu(self.hidden(torch.cat(parts, dim=1))))


def protocol_digest(config: dict[str, Any]) -> str:
    return hashlib.sha256((HERE / config["protocol"]).read_bytes()).hexdigest()


def make_views(numerical: int, cardinalities: list[int], count: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    views = [{
        "numerical": np.arange(numerical),
        "categories": [np.arange(cardinality) for cardinality in cardinalities],
        "classes": np.arange(2),
    }]
    while len(views) < count:
        candidate = {
            "numerical": rng.permutation(numerical),
            "categories": [rng.permutation(cardinality) for cardinality in cardinalities],
            "classes": rng.permutation(2),
        }
        signature = (
            tuple(candidate["numerical"]),
            tuple(tuple(value) for value in candidate["categories"]),
            tuple(candidate["classes"]),
        )
        prior = {
            (
                tuple(view["numerical"]),
                tuple(tuple(value) for value in view["categories"]),
                tuple(view["classes"]),
            )
            for view in views
        }
        if signature not in prior:
            views.append(candidate)
    return views


def matched_state(
    canonical: dict[str, torch.Tensor],
    numerical_permutation: np.ndarray,
    category_maps: list[np.ndarray],
    class_map: np.ndarray,
    numerical: int,
) -> dict[str, torch.Tensor]:
    state = copy.deepcopy(canonical)
    numeric_index = torch.as_tensor(numerical_permutation, dtype=torch.long)
    state["hidden.weight"][:, :numerical] = canonical["hidden.weight"][:, numeric_index]
    for index, mapping in enumerate(category_maps):
        key = f"embeddings.{index}.weight"
        transformed = canonical[key].clone()
        target = torch.as_tensor(mapping + 1, dtype=torch.long)
        transformed[target] = canonical[key][1:]
        state[key] = transformed
    target = torch.as_tensor(class_map, dtype=torch.long)
    output_weight = canonical["output.weight"].clone()
    output_bias = canonical["output.bias"].clone()
    output_weight[target] = canonical["output.weight"]
    output_bias[target] = canonical["output.bias"]
    state["output.weight"] = output_weight
    state["output.bias"] = output_bias
    return state


def transform_inputs(
    numerical: np.ndarray,
    categorical: np.ndarray,
    view: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    transformed_categorical = np.zeros_like(categorical, dtype=np.int64)
    for index, mapping in enumerate(view["categories"]):
        known = categorical[:, index] >= 0
        transformed_categorical[known, index] = mapping[categorical[known, index].astype(int)] + 1
    return numerical[:, view["numerical"]], transformed_categorical


def initialize(
    seed: int,
    numerical: int,
    cardinalities: list[int],
    training: dict[str, Any],
) -> EmbeddingMLP:
    torch.manual_seed(seed)
    return EmbeddingMLP(
        numerical,
        cardinalities,
        int(training["embedding_dim"]),
        int(training["hidden_dim"]),
    ).double()


def probabilities(
    model: EmbeddingMLP,
    numerical: np.ndarray,
    categorical: np.ndarray,
    class_map: np.ndarray,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(numerical), torch.from_numpy(categorical))
        raw = torch.softmax(logits, dim=1).cpu().numpy()
    return raw[:, class_map]


def train(
    model: EmbeddingMLP,
    numerical: np.ndarray,
    categorical: np.ndarray,
    target: np.ndarray,
    training: dict[str, Any],
) -> None:
    model.train()
    x_num = torch.from_numpy(numerical)
    x_cat = torch.from_numpy(categorical)
    y = torch.from_numpy(target.astype(np.int64))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    for _ in range(int(training["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(x_num, x_cat), y)
        loss.backward()
        optimizer.step()


def run_dataset(name: str, config: dict[str, Any], output: Path) -> list[dict[str, Any]]:
    data = load_dataset(Path(config["data_root"]) / name, config)
    if data.task not in {"binclass", "multiclass"} or len(np.unique(data.train_y)) != 2:
        raise ValueError(f"{name} is not binary classification")
    encoded, cardinalities = encode_categories(data)
    means = np.nanmean(data.train_n, axis=0)
    scales = np.nanstd(data.train_n, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    scales = np.where(np.isfinite(scales) & (scales > 0), scales, 1.0)
    numerical = {
        "train": np.nan_to_num((data.train_n - means) / scales).astype(np.float64),
        "validation": np.nan_to_num((data.validation_n - means) / scales).astype(np.float64),
        "test": np.nan_to_num((data.test_n - means) / scales).astype(np.float64),
    }
    views = make_views(
        data.train_n.shape[1], cardinalities, int(config["representatives"]),
        int(config["view_seed"]) + sum(name.encode()),
    )
    arms = ("ordinary", "matched")
    shape = (len(arms), len(config["seeds"]), len(views))
    validation = np.empty(shape + (len(data.validation_y), 2), dtype=np.float64)
    test = np.empty(shape + (len(data.test_y), 2), dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(config["seeds"]):
        canonical_model = initialize(int(seed), data.train_n.shape[1], cardinalities, config["training"])
        canonical_state = copy.deepcopy(canonical_model.state_dict())
        canonical_initial = probabilities(
            canonical_model,
            *transform_inputs(numerical["validation"], encoded["validation"], views[0]),
            views[0]["classes"],
        )
        for view_index, view in enumerate(views):
            split_inputs = {
                split: transform_inputs(numerical[split], encoded[split], view)
                for split in ("train", "validation", "test")
            }
            for arm_index, arm in enumerate(arms):
                model = initialize(int(seed), data.train_n.shape[1], cardinalities, config["training"])
                if arm == "matched":
                    model.load_state_dict(matched_state(
                        canonical_state, view["numerical"], view["categories"],
                        view["classes"], data.train_n.shape[1],
                    ))
                initial = probabilities(model, *split_inputs["validation"], view["classes"])
                initial_gap = float(np.max(np.abs(initial - canonical_initial))) if arm == "matched" else float("nan")
                if arm == "matched" and initial_gap > float(config["initial_tolerance"]):
                    raise AssertionError(f"matched initial gap {initial_gap}")
                transformed_y = view["classes"][data.train_y]
                train(model, *split_inputs["train"], transformed_y, config["training"])
                validation[arm_index, seed_index, view_index] = probabilities(
                    model, *split_inputs["validation"], view["classes"]
                )
                test[arm_index, seed_index, view_index] = probabilities(
                    model, *split_inputs["test"], view["classes"]
                )
                rows.append({
                    "dataset": name,
                    "seed": int(seed),
                    "representative": view_index,
                    "arm": arm,
                    "initial_max_aligned_gap": initial_gap,
                })
                print(f"{name} seed={seed} view={view_index} arm={arm}", flush=True)
        identity_gap = float(np.max(np.abs(validation[0, seed_index, 0] - validation[1, seed_index, 0])))
        if identity_gap != 0.0:
            raise AssertionError(f"identity arm mismatch {identity_gap}")
    tolerance = float(config["probability_tolerance"])
    for values in (validation, test):
        if not np.isfinite(values).all() or values.min() < -tolerance or values.max() > 1 + tolerance:
            raise AssertionError("invalid probabilities")
        if float(np.max(np.abs(values.sum(axis=-1) - 1))) > tolerance:
            raise AssertionError("probabilities do not sum to one")
    np.savez_compressed(
        output / f"{name}.npz", validation_predictions=validation, test_predictions=test,
        validation_y=data.validation_y, test_y=data.test_y,
    )
    manifest = {
        "status": "complete", "dataset": name, "arms": list(arms),
        "seeds": config["seeds"], "representatives": len(views),
        "rows": {"train": len(data.train_y), "validation": len(data.validation_y), "test": len(data.test_y)},
        "features": {"numerical": data.train_n.shape[1], "categorical": data.train_c.shape[1]},
        "cardinalities": cardinalities,
    }
    (output / f"{name}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return rows


def prediction_variance(values: np.ndarray) -> float:
    center = values.mean(axis=0, keepdims=True)
    return float(np.mean(np.sum((values - center) ** 2, axis=-1)))


def analyze(config: dict[str, Any], output: Path, fit_rows: list[dict[str, Any]]) -> dict[str, Any]:
    cell_rows = []
    for dataset in config["datasets"]:
        with np.load(output / f"{dataset}.npz") as archive:
            validation = archive["validation_predictions"]
            test = archive["test_predictions"]
        for seed_index, seed in enumerate(config["seeds"]):
            ordinary = prediction_variance(test[0, seed_index])
            matched = prediction_variance(test[1, seed_index])
            cell_rows.append({
                "dataset": dataset, "seed": int(seed),
                "ordinary_schema_variance": ordinary,
                "matched_function_schema_variance": matched,
                "fraction_removed": 1 - matched / ordinary if ordinary > 0 else float("nan"),
                "matched_post_training_max_gap": float(np.max(
                    np.abs(test[1, seed_index] - test[1, seed_index, 0:1])
                )),
            })
    cells = pd.DataFrame(cell_rows)
    cells.to_csv(output.parent / "matched_function_cells.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(output.parent / "matched_function_fits.csv", index=False)
    ordinary = float(cells.ordinary_schema_variance.mean())
    matched = float(cells.matched_function_schema_variance.mean())
    summary = {
        "status": "complete",
        "protocol_sha256": protocol_digest(config),
        "datasets": len(config["datasets"]),
        "seeds": len(config["seeds"]),
        "representatives": int(config["representatives"]),
        "model_fits": len(fit_rows),
        "mean_ordinary_schema_variance": ordinary,
        "mean_matched_function_schema_variance": matched,
        "pooled_fraction_removed": 1 - matched / ordinary if ordinary > 0 else None,
        "maximum_matched_initial_gap": float(pd.DataFrame(fit_rows).initial_max_aligned_gap.max()),
        "maximum_matched_post_training_gap": float(cells.matched_post_training_max_gap.max()),
        "dataset_means": cells.groupby("dataset")[[
            "ordinary_schema_variance", "matched_function_schema_variance", "fraction_removed"
        ]].mean().to_dict(orient="index"),
        "gate_f_residual_supported": bool(matched > max(ordinary * 1e-6, 1e-14)),
        "interpretation": "residual optimization-path support only if the matched variance is materially above numerical error",
    }
    (output.parent / "matched_function_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "matched_function")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if protocol_digest(config) != config["protocol_sha256"]:
        raise AssertionError("frozen protocol hash mismatch")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in config["datasets"]:
        rows.extend(run_dataset(dataset, config, args.output_dir))
    print(json.dumps(analyze(config, args.output_dir, rows), indent=2))


if __name__ == "__main__":
    main()
