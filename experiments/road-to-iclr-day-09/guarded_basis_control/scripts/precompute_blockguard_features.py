#!/usr/bin/env python3
"""Precompute disjoint exact one-block caches for a slow BlockGuard unit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from guarded_basis.blockguard import gram_interface, target_free_descriptors  # noqa: E402
from guarded_basis.common import (  # noqa: E402
    development_base_predictions,
    development_specs,
    load_blocks,
    load_protocol,
    orthogonal_orbit,
)
from run_blockguard import one_block_interventions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--features", required=True, help="comma-separated exact feature names")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    protocol = load_protocol()
    specs = [spec for spec in development_specs(protocol) if spec["key"] == args.dataset]
    if len(specs) != 1:
        raise RuntimeError(f"unknown development dataset: {args.dataset}")
    blocks = load_blocks(specs[0], protocol)
    orbit = orthogonal_orbit(blocks, protocol)
    gram_orbit = [gram_interface(rep, blocks.dataset.key) for rep in orbit]
    raw, gram, _ = development_base_predictions(args.model, blocks, orbit, args.seed, args.device)
    requested = set(args.features.split(","))
    descriptors = [
        row for row in target_free_descriptors(orbit[0], orbit)
        if row["feature"] in requested
    ]
    if {row["feature"] for row in descriptors} != requested:
        raise RuntimeError(f"unknown features: {sorted(requested - {row['feature'] for row in descriptors})}")
    one_block_interventions(
        blocks=blocks,
        orbit=orbit,
        gram_orbit=gram_orbit,
        raw=raw,
        gram=gram,
        model=args.model,
        seed=args.seed,
        device=args.device,
        descriptors=descriptors,
    )


if __name__ == "__main__":
    main()
