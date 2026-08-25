"""Direct finite-precision audit of exact and sketched canonical coordinates."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .broad_data import controlled_representation, load_broad_dataset, sketched_anchor_canonicalize
from .core import PARTS, whiten
from .optimizer_remedies import anchor_canonicalize


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/day3/broad_benchmark"


def relative(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(left), 1e-30))


def audit(method, endpoint_parts):
    transformed, metadata, seconds = {}, {}, {}
    whitened = {}
    for kappa, parts in endpoint_parts.items():
        started = time.perf_counter()
        coordinates, meta = method(parts)
        seconds[str(kappa)] = time.perf_counter() - started
        transformed[kappa] = coordinates
        metadata[str(kappa)] = meta
        whitened[kappa] = whiten(coordinates)[0]
    left_anchors = set(metadata["1.0"]["anchor_rows"])
    right_anchors = set(metadata["1000.0"]["anchor_rows"])
    return {
        "anchor_count_k1": len(left_anchors),
        "anchor_count_k1000": len(right_anchors),
        "anchors_identical": left_anchors == right_anchors,
        "anchor_symmetric_difference": len(left_anchors.symmetric_difference(right_anchors)),
        "coordinate_relative_difference": {
            part: relative(transformed[1.0][part], transformed[1000.0][part])
            for part in PARTS
        },
        "post_whitening_relative_difference": {
            part: relative(whitened[1.0][part], whitened[1000.0][part])
            for part in PARTS
        },
        "runtime_seconds": seconds,
        "metadata": metadata,
    }


def main() -> None:
    dataset = load_broad_dataset("adult")
    endpoint_parts = {
        kappa: controlled_representation(dataset, kappa).parts
        for kappa in (1.0, 1000.0)
    }
    payload = {
        "dataset": "adult",
        "comparison": "same train-fitted information orbit at kappa 1 and 1000",
        "full_anchor": audit(anchor_canonicalize, endpoint_parts),
        "sketched_anchor": audit(sketched_anchor_canonicalize, endpoint_parts),
        "interpretation": (
            "Exact algebra predicts identical coordinates when numerical rank and pivot "
            "selection agree. Nonzero sketched differences expose finite-precision pivot "
            "switches, not information loss or a different hypothesis class."
        ),
    }
    output = RESULTS / "canonicalization_numerical_audit.json"
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
