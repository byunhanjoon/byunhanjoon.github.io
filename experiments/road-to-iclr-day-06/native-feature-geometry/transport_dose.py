"""Outcome-informed, prospectively gated H6 metric-corruption dose response."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import native_geometry as ng


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "pilot_config.json"
ALPHAS = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float64)
INTERFACES = ("learned", "native_tuned")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def embedding_from_gram(gram: np.ndarray, rank: int) -> np.ndarray:
    gram = ng.center((gram + gram.T) / 2.0)
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    vectors = vectors[:, order]
    positive = values > 1e-12
    values, vectors = values[positive], vectors[:, positive]
    if len(values) < rank:
        raise AssertionError(f"only {len(values)} positive modes")
    embedding = vectors[:, :rank] * np.sqrt(values[:rank])[None, :]
    embedding *= math.sqrt(rank / np.mean(np.sum(embedding**2, axis=1)))
    return embedding


def run_bundle(
    domain: str, seed: int, device_name: str, output_dir: Path, force: bool = False
) -> Path:
    config = json.loads(CONFIG_PATH.read_text())
    if domain not in config["domains"] or seed not in config["seeds"]:
        raise ValueError("bundle outside frozen H6 matrix")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{domain}__seed{seed}"
    artifact = output_dir / f"{stem}.npz"
    manifest_path = output_dir / f"{stem}.json"
    if artifact.exists() and manifest_path.exists() and not force:
        if json.loads(manifest_path.read_text()).get("status") == "complete":
            print(f"complete: {stem}", flush=True)
            return artifact

    pilot_path = HERE / "results" / "pilot" / f"{domain}__category_holdout__seed{seed}.npz"
    if not pilot_path.exists():
        raise FileNotFoundError(pilot_path)
    pilot = np.load(pilot_path, allow_pickle=False)
    pilot_interfaces = pilot["interfaces"].tolist()
    pilot_transport = pilot["transport_interfaces"].tolist()
    native_patch_index = pilot["patch_names"].tolist().index("native_transport")
    shuffled_patch_index = pilot["patch_names"].tolist().index("shuffled_transport")

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(device_name)
    data = ng.make_dataset(domain, "category_holdout", seed, config)
    native, _ = ng.native_embedding(domain, int(config["embedding_dim"]))
    base_gram = native @ native.T
    corruption = pilot["corruption"]
    corrupt_gram = base_gram[np.ix_(corruption, corruption)]
    dose_embeddings = np.asarray([
        embedding_from_gram(
            (1.0 - alpha) * base_gram + alpha * corrupt_gram,
            int(config["embedding_dim"]),
        )
        for alpha in ALPHAS
    ])
    chart_list = pilot["charts"]
    predictions = np.empty(
        (len(INTERFACES), len(chart_list), len(ALPHAS), len(data.target["test"])),
        dtype=np.float32,
    )
    seen_changes = np.empty(
        (len(INTERFACES), len(chart_list), len(ALPHAS)), dtype=np.float64
    )
    replay_errors: list[float] = []
    endpoint_errors: list[float] = []
    started = time.perf_counter()

    for interface_index, interface in enumerate(INTERFACES):
        source_interface_index = pilot_interfaces.index(interface)
        source_transport_index = pilot_transport.index(interface)
        for chart_index, chart in enumerate(chart_list):
            ng.seed_all(seed + 701)
            table = ng.code_table(native, chart) if interface == "native_tuned" else None
            model = ng.FeatureMLP(
                interface,
                table,
                int(config["embedding_dim"]),
                int(config["training"]["hidden_width"]),
            ).to(device)
            ng.train_model(model, data, chart, config, device)
            original = ng.predict(
                model, data.category["test"], data.continuous["test"], chart, data, device
            )
            replay_errors.append(float(np.max(np.abs(
                original - pilot["predictions"][source_interface_index, chart_index]
            ))))
            aligned = ng.aligned_table(model, chart, device)
            original_weights = model.embedding.weight.detach().clone()
            seen_mask = np.isin(data.category["test"], data.seen)
            for alpha_index, coordinates in enumerate(dose_embeddings):
                matrix, bias = ng.affine_transport(
                    coordinates[data.seen], aligned[data.seen], float(config["transport_ridge"])
                )
                held_values = coordinates[data.held] @ matrix + bias
                with torch.no_grad():
                    model.embedding.weight.copy_(original_weights)
                    rows = torch.as_tensor(chart[data.held], dtype=torch.long, device=device)
                    model.embedding.weight[rows] = torch.as_tensor(
                        held_values, dtype=torch.float32, device=device
                    )
                current = ng.predict(
                    model, data.category["test"], data.continuous["test"], chart, data, device
                )
                predictions[interface_index, chart_index, alpha_index] = current
                seen_changes[interface_index, chart_index, alpha_index] = float(
                    np.max(np.abs(current[seen_mask] - original[seen_mask]))
                )
            with torch.no_grad():
                model.embedding.weight.copy_(original_weights)
            endpoint_errors.append(float(np.max(np.abs(
                predictions[interface_index, chart_index, 0]
                - pilot["patch_predictions"][source_transport_index, chart_index, native_patch_index]
            ))))
            endpoint_errors.append(float(np.max(np.abs(
                predictions[interface_index, chart_index, -1]
                - pilot["patch_predictions"][source_transport_index, chart_index, shuffled_patch_index]
            ))))

    temporary = artifact.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            domain=np.asarray(domain),
            seed=np.asarray(seed),
            alphas=ALPHAS,
            interfaces=np.asarray(INTERFACES),
            charts=chart_list,
            test_category=data.category["test"],
            test_target=data.target["test"],
            seen=data.seen,
            held=data.held,
            predictions=predictions,
            seen_changes=seen_changes,
            replay_errors=np.asarray(replay_errors),
            endpoint_errors=np.asarray(endpoint_errors),
        )
    temporary.replace(artifact)
    manifest = {
        "status": "complete",
        "domain": domain,
        "seed": seed,
        "config_sha256": sha256(CONFIG_PATH),
        "artifact_sha256": sha256(artifact),
        "trained_paths": len(INTERFACES) * len(chart_list),
        "interventions": len(INTERFACES) * len(chart_list) * len(ALPHAS),
        "elapsed_seconds": time.perf_counter() - started,
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary_manifest.replace(manifest_path)
    print(f"wrote {artifact} ({manifest['elapsed_seconds']:.2f}s)", flush=True)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "h6")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_bundle(args.domain, args.seed, args.device, args.output_dir, args.force)


if __name__ == "__main__":
    main()

