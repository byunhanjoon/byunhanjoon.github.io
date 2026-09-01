"""Controlled pilot for Native Feature Geometry.

Protocols and gates live in PILOT_PROTOCOL.md.  Artifacts are write-once unless
--force is explicitly supplied.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from scipy.linalg import expm
from torch import nn


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "pilot_config.json"
N_CATEGORIES = 16


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_offset(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def center(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    h = np.eye(n) - np.ones((n, n)) / n
    return h @ matrix @ h


def graph_laplacian(domain: str) -> np.ndarray:
    adjacency = np.zeros((N_CATEGORIES, N_CATEGORIES), dtype=np.float64)
    for index in range(N_CATEGORIES - 1):
        adjacency[index, index + 1] = adjacency[index + 1, index] = 1.0
    if domain == "cycle16":
        adjacency[0, -1] = adjacency[-1, 0] = 1.0
    return np.diag(adjacency.sum(axis=1)) - adjacency


def tree_distance() -> np.ndarray:
    distance = np.zeros((N_CATEGORIES, N_CATEGORIES), dtype=np.float64)
    for left in range(N_CATEGORIES):
        for right in range(N_CATEGORIES):
            if left == right:
                continue
            xor = left ^ right
            common_prefix = 4 - xor.bit_length()
            distance[left, right] = 2.0 * (4 - common_prefix)
    return distance


def semantic_kernel(domain: str) -> np.ndarray:
    if domain in {"cycle16", "ordinal16"}:
        kernel = expm(-0.8 * graph_laplacian(domain))
    elif domain == "tree16":
        kernel = np.exp(-tree_distance() / 3.0)
    elif domain == "nominal16":
        kernel = np.eye(N_CATEGORIES, dtype=np.float64)
    else:
        raise ValueError(f"unknown domain: {domain}")
    return (kernel + kernel.T) / 2.0


def native_embedding(domain: str, rank: int = 15) -> tuple[np.ndarray, np.ndarray]:
    gram = center(semantic_kernel(domain))
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    vectors = vectors[:, order]
    positive = values > 1e-12
    values, vectors = values[positive], vectors[:, positive]
    if len(values) < rank:
        raise AssertionError(f"{domain} has only {len(values)} positive centered modes")
    embedding = vectors[:, :rank] * np.sqrt(values[:rank])[None, :]
    # A common global scale removes an irrelevant kernel-amplitude confound.
    scale = math.sqrt(rank / np.mean(np.sum(embedding**2, axis=1)))
    embedding = embedding * scale
    return embedding.astype(np.float64), embedding @ embedding.T


def centered_cka(left: np.ndarray, right_gram: np.ndarray) -> float:
    left_gram = center(np.asarray(left, dtype=np.float64) @ np.asarray(left, dtype=np.float64).T)
    right_gram = center(np.asarray(right_gram, dtype=np.float64))
    denom = np.linalg.norm(left_gram) * np.linalg.norm(right_gram)
    return float(np.sum(left_gram * right_gram) / denom) if denom > 0 else float("nan")


def charts(seed: int, count: int) -> list[np.ndarray]:
    found = [np.arange(N_CATEGORIES, dtype=np.int64)]
    rng = np.random.default_rng(seed + 91_003)
    while len(found) < count:
        candidate = rng.permutation(N_CATEGORIES).astype(np.int64)
        if not any(np.array_equal(candidate, previous) for previous in found):
            found.append(candidate)
    return found


def native_effects(domain: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if domain == "cycle16":
        theta = 2 * np.pi * np.arange(N_CATEGORIES) / N_CATEGORIES
        main = 1.10 * np.sin(theta) + 0.55 * np.cos(2 * theta)
        interaction = 0.55 * np.sin(theta + np.pi / 4) + 0.20 * np.cos(3 * theta)
    elif domain == "ordinal16":
        value = np.linspace(-1.0, 1.0, N_CATEGORIES)
        main = 1.10 * value + 0.65 * value**2 - 0.35 * value**3
        interaction = 0.50 * np.cos(np.pi * value) + 0.15 * value
    elif domain == "tree16":
        main = np.zeros(N_CATEGORIES)
        interaction = np.zeros(N_CATEGORIES)
        for leaf in range(N_CATEGORIES):
            bits = np.array([(leaf >> shift) & 1 for shift in (3, 2, 1, 0)])
            signs = 2.0 * bits - 1.0
            main[leaf] = np.dot(signs, np.array([0.85, 0.48, 0.27, 0.15]))
            interaction[leaf] = np.dot(signs, np.array([0.42, -0.24, 0.14, -0.08]))
    elif domain == "nominal16":
        rng = np.random.default_rng(seed + 170_021)
        main = rng.normal(size=N_CATEGORIES)
        interaction = 0.45 * rng.normal(size=N_CATEGORIES)
    else:
        raise ValueError(domain)
    main = (main - main.mean()) / main.std()
    interaction = interaction - interaction.mean()
    return main.astype(np.float32), interaction.astype(np.float32)


@dataclass
class Dataset:
    category: dict[str, np.ndarray]
    continuous: dict[str, np.ndarray]
    target: dict[str, np.ndarray]
    seen: np.ndarray
    held: np.ndarray
    target_mean: float
    target_std: float


def make_dataset(domain: str, regime: str, seed: int, config: dict[str, Any]) -> Dataset:
    holdouts = np.asarray(config["holdouts"][domain], dtype=np.int64)
    held = holdouts if regime == "category_holdout" else np.array([], dtype=np.int64)
    seen = np.asarray([i for i in range(N_CATEGORIES) if i not in set(held)], dtype=np.int64)
    main, interaction = native_effects(domain, seed)
    rng = np.random.default_rng(seed + stable_offset(domain) + stable_offset(regime))
    category: dict[str, np.ndarray] = {}
    continuous: dict[str, np.ndarray] = {}
    target: dict[str, np.ndarray] = {}
    for part in ("train", "validation", "test"):
        size = int(config["sizes"][part])
        if part == "test":
            values = np.tile(np.arange(N_CATEGORIES), math.ceil(size / N_CATEGORIES))[:size]
            values = values[rng.permutation(size)]
        else:
            values = rng.choice(seen, size=size, replace=True)
        z = rng.normal(size=(size, 3)).astype(np.float32)
        noise = 0.10 * rng.normal(size=size)
        y = (
            main[values]
            + interaction[values] * z[:, 0]
            + 0.35 * (z[:, 1] ** 2 - 1.0)
            - 0.25 * z[:, 2]
            + noise
        )
        category[part] = values.astype(np.int64)
        continuous[part] = z
        target[part] = y.astype(np.float32)
    target_mean = float(target["train"].mean())
    target_std = float(target["train"].std())
    if target_std <= 0:
        raise AssertionError("degenerate training target")
    return Dataset(category, continuous, target, seen, held, target_mean, target_std)


class FeatureMLP(nn.Module):
    def __init__(
        self,
        interface: str,
        code_table: np.ndarray | None,
        embedding_dim: int,
        hidden: int,
    ) -> None:
        super().__init__()
        self.interface = interface
        if interface == "label":
            self.lift = nn.Linear(1, embedding_dim)
            self.embedding = None
            self.register_buffer("fixed_table", None)
        elif interface in {"learned", "native_tuned"}:
            self.embedding = nn.Embedding(N_CATEGORIES, embedding_dim)
            if code_table is not None:
                with torch.no_grad():
                    self.embedding.weight.copy_(torch.as_tensor(code_table, dtype=torch.float32))
            self.lift = None
            self.register_buffer("fixed_table", None)
        else:
            if code_table is None:
                raise ValueError(f"{interface} needs a fixed table")
            self.embedding = None
            self.lift = None
            self.register_buffer("fixed_table", torch.as_tensor(code_table, dtype=torch.float32))
        self.network = nn.Sequential(
            nn.Linear(embedding_dim + 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def representation(self, code: torch.Tensor) -> torch.Tensor:
        if self.interface == "label":
            scalar = code.float().unsqueeze(1) / (N_CATEGORIES - 1)
            return self.lift(scalar)
        if self.embedding is not None:
            return self.embedding(code)
        return self.fixed_table[code]

    def forward(self, code: torch.Tensor, continuous: torch.Tensor) -> torch.Tensor:
        value = torch.cat([self.representation(code), continuous], dim=1)
        return self.network(value).squeeze(1)


def code_table(semantic_table: np.ndarray, chart: np.ndarray) -> np.ndarray:
    result = np.empty_like(semantic_table)
    result[chart] = semantic_table
    return result


def aligned_table(model: FeatureMLP, chart: np.ndarray, device: torch.device) -> np.ndarray:
    code = torch.as_tensor(chart, dtype=torch.long, device=device)
    with torch.no_grad():
        table = model.representation(code).cpu().numpy()
    return table.astype(np.float64)


def predict(
    model: FeatureMLP,
    category: np.ndarray,
    continuous: np.ndarray,
    chart: np.ndarray,
    data: Dataset,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    code = torch.as_tensor(chart[category], dtype=torch.long, device=device)
    z = torch.as_tensor(continuous, dtype=torch.float32, device=device)
    with torch.no_grad():
        normalized = model(code, z).cpu().numpy()
    return (normalized * data.target_std + data.target_mean).astype(np.float32)


def train_model(
    model: FeatureMLP,
    data: Dataset,
    chart: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
) -> None:
    training = config["training"]
    model.to(device)
    code = torch.as_tensor(chart[data.category["train"]], dtype=torch.long, device=device)
    z = torch.as_tensor(data.continuous["train"], dtype=torch.float32, device=device)
    y = torch.as_tensor(
        (data.target["train"] - data.target_mean) / data.target_std,
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    model.train()
    for _ in range(int(training["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(code, z) - y) ** 2)
        loss.backward()
        optimizer.step()


def mse(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(len(target), dtype=bool)
    return float(np.mean((prediction[mask].astype(np.float64) - target[mask]) ** 2))


def affine_transport(source: np.ndarray, destination: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=np.float64)
    destination = np.asarray(destination, dtype=np.float64)
    augmented = np.column_stack([source, np.ones(len(source))])
    penalty = np.eye(augmented.shape[1]) * ridge
    penalty[-1, -1] = 0.0
    coefficient = np.linalg.solve(augmented.T @ augmented + penalty, augmented.T @ destination)
    return coefficient[:-1], coefficient[-1]


def patch_predictions(
    model: FeatureMLP,
    interface: str,
    chart: np.ndarray,
    data: Dataset,
    native: np.ndarray,
    corrupt_native: np.ndarray,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    if model.embedding is None:
        raise ValueError("patching requires a trainable lookup")
    patch_names = ["original", "native_transport", "mean", "random", "shuffled_transport"]
    original_prediction = predict(
        model, data.category["test"], data.continuous["test"], chart, data, device
    )
    original_aligned = aligned_table(model, chart, device)
    candidates: dict[str, np.ndarray] = {"original": original_aligned[data.held].copy()}
    matrix, bias = affine_transport(
        native[data.seen], original_aligned[data.seen], float(config["transport_ridge"])
    )
    candidates["native_transport"] = native[data.held] @ matrix + bias
    candidates["mean"] = np.repeat(
        original_aligned[data.seen].mean(axis=0, keepdims=True), len(data.held), axis=0
    )
    rng = np.random.default_rng(seed + stable_offset(interface) + 311_009)
    centered_seen = original_aligned[data.seen] - original_aligned[data.seen].mean(axis=0)
    scale = np.sqrt(np.mean(centered_seen**2, axis=0) + 1e-8)
    candidates["random"] = original_aligned[data.seen].mean(axis=0) + rng.normal(
        size=(len(data.held), original_aligned.shape[1])
    ) * scale
    shuffled_matrix, shuffled_bias = affine_transport(
        corrupt_native[data.seen], original_aligned[data.seen], float(config["transport_ridge"])
    )
    candidates["shuffled_transport"] = corrupt_native[data.held] @ shuffled_matrix + shuffled_bias

    all_predictions = []
    seen_changes = []
    original_weights = model.embedding.weight.detach().clone()
    seen_test = np.isin(data.category["test"], data.seen)
    for name in patch_names:
        with torch.no_grad():
            model.embedding.weight.copy_(original_weights)
            code_rows = chart[data.held]
            model.embedding.weight[torch.as_tensor(code_rows, dtype=torch.long, device=device)] = torch.as_tensor(
                candidates[name], dtype=torch.float32, device=device
            )
        current = predict(model, data.category["test"], data.continuous["test"], chart, data, device)
        all_predictions.append(current)
        seen_changes.append(float(np.max(np.abs(current[seen_test] - original_prediction[seen_test]))))
    with torch.no_grad():
        model.embedding.weight.copy_(original_weights)
    return patch_names, np.asarray(all_predictions), np.asarray(seen_changes, dtype=np.float64)


def run_bundle(
    domain: str,
    regime: str,
    seed: int,
    device_name: str,
    output_dir: Path,
    force: bool = False,
) -> Path:
    config = json.loads(CONFIG_PATH.read_text())
    if domain not in config["domains"] or regime not in config["regimes"] or seed not in config["seeds"]:
        raise ValueError("bundle outside frozen matrix")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{domain}__{regime}__seed{seed}"
    artifact = output_dir / f"{stem}.npz"
    manifest_path = output_dir / f"{stem}.json"
    if artifact.exists() and manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") == "complete":
            print(f"complete: {stem}", flush=True)
            return artifact

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(device_name)
    data = make_dataset(domain, regime, seed, config)
    native, native_gram = native_embedding(domain, int(config["embedding_dim"]))
    rng = np.random.default_rng(seed + stable_offset(domain) + 230_003)
    random_fixed = rng.normal(size=native.shape)
    random_fixed *= math.sqrt(config["embedding_dim"] / np.mean(np.sum(random_fixed**2, axis=1)))
    corruption = rng.permutation(N_CATEGORIES)
    corrupt_native = native[corruption]
    chart_list = charts(seed + stable_offset(domain), int(config["nonidentity_charts"]) + 1)
    interfaces = list(config["interfaces"])
    predictions = np.empty((len(interfaces), len(chart_list), len(data.target["test"])), dtype=np.float32)
    initial_tables = np.empty((len(interfaces), len(chart_list), N_CATEGORIES, config["embedding_dim"]), dtype=np.float32)
    final_tables = np.empty_like(initial_tables)
    metrics: list[dict[str, Any]] = []
    transport_interfaces = ["learned", "native_tuned"]
    patch_names = ["original", "native_transport", "mean", "random", "shuffled_transport"]
    patch_values = np.full(
        (len(transport_interfaces), len(chart_list), len(patch_names), len(data.target["test"])),
        np.nan,
        dtype=np.float32,
    )
    patch_seen_changes = np.full(
        (len(transport_interfaces), len(chart_list), len(patch_names)), np.nan, dtype=np.float64
    )
    started = time.perf_counter()

    for interface_index, interface in enumerate(interfaces):
        for chart_index, chart in enumerate(chart_list):
            seed_all(seed + 701)
            semantic_table: np.ndarray | None
            if interface == "native_fixed" or interface == "native_tuned":
                semantic_table = native
            elif interface == "random_fixed":
                semantic_table = random_fixed
            elif interface == "corrupt_fixed":
                semantic_table = corrupt_native
            else:
                semantic_table = None
            table = None if semantic_table is None else code_table(semantic_table, chart)
            model = FeatureMLP(
                interface,
                table,
                int(config["embedding_dim"]),
                int(config["training"]["hidden_width"]),
            ).to(device)
            initial = aligned_table(model, chart, device)
            train_model(model, data, chart, config, device)
            final = aligned_table(model, chart, device)
            current = predict(model, data.category["test"], data.continuous["test"], chart, data, device)
            predictions[interface_index, chart_index] = current
            initial_tables[interface_index, chart_index] = initial
            final_tables[interface_index, chart_index] = final
            seen_mask = np.isin(data.category["test"], data.seen)
            held_mask = np.isin(data.category["test"], data.held)
            row = {
                "domain": domain,
                "regime": regime,
                "seed": seed,
                "interface": interface,
                "chart": chart_index,
                "test_mse": mse(current, data.target["test"]),
                "seen_mse": mse(current, data.target["test"], seen_mask),
                "held_mse": None if not held_mask.any() else mse(current, data.target["test"], held_mask),
                "initial_native_cka": centered_cka(initial, native_gram),
                "final_native_cka": centered_cka(final, native_gram),
                "final_corrupt_cka": centered_cka(final, corrupt_native @ corrupt_native.T),
            }
            metrics.append(row)
            if regime == "category_holdout" and interface in transport_interfaces:
                names, values, changes = patch_predictions(
                    model, interface, chart, data, native, corrupt_native, config, seed, device
                )
                if names != patch_names:
                    raise AssertionError("patch menu changed")
                patch_index = transport_interfaces.index(interface)
                patch_values[patch_index, chart_index] = values
                patch_seen_changes[patch_index, chart_index] = changes

    payload = {
        "interfaces": np.asarray(interfaces),
        "charts": np.asarray(chart_list, dtype=np.int64),
        "test_category": data.category["test"],
        "test_continuous": data.continuous["test"],
        "test_target": data.target["test"],
        "seen": data.seen,
        "held": data.held,
        "native_embedding": native.astype(np.float32),
        "native_gram": native_gram.astype(np.float64),
        "corrupt_native": corrupt_native.astype(np.float32),
        "corruption": corruption,
        "predictions": predictions,
        "initial_tables": initial_tables,
        "final_tables": final_tables,
        "transport_interfaces": np.asarray(transport_interfaces),
        "patch_names": np.asarray(patch_names),
        "patch_predictions": patch_values,
        "patch_seen_changes": patch_seen_changes,
    }
    temporary = artifact.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(artifact)
    manifest = {
        "status": "complete",
        "domain": domain,
        "regime": regime,
        "seed": seed,
        "config_sha256": sha256(CONFIG_PATH),
        "artifact_sha256": sha256(artifact),
        "paths": len(interfaces) * len(chart_list),
        "elapsed_seconds": time.perf_counter() - started,
        "metrics": metrics,
    }
    manifest_tmp = manifest_path.with_suffix(".json.tmp")
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest_tmp.replace(manifest_path)
    print(f"wrote {artifact} ({manifest['elapsed_seconds']:.2f}s)", flush=True)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True)
    parser.add_argument("--regime", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "pilot")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_bundle(args.domain, args.regime, args.seed, args.device, args.output_dir, args.force)


if __name__ == "__main__":
    main()

