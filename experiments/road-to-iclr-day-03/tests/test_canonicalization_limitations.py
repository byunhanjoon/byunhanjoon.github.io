import numpy as np

from experiments.day3.broad_data import sketched_anchor_canonicalize


def test_sketched_anchor_ties_are_not_claimed_pointwise_invariant():
    """Symmetric leverage-score ties may switch pivots in floating point."""

    dimension = 12
    train = np.tile(np.eye(dimension), (50, 1))
    parts = {
        "train": train,
        "val": np.eye(dimension),
        "test": np.eye(dimension)[::-1],
    }
    rng = np.random.default_rng(0)
    left, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    right, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    transform = left @ np.diag(np.geomspace(1e-3, 1.0, dimension)) @ right
    changed = {part: values @ transform for part, values in parts.items()}

    canonical, metadata = sketched_anchor_canonicalize(parts, initial_rows=128)
    recoded, recoded_metadata = sketched_anchor_canonicalize(changed, initial_rows=128)

    assert metadata["anchor_rows"] != recoded_metadata["anchor_rows"]
    relative_difference = np.linalg.norm(canonical["train"] - recoded["train"]) / np.linalg.norm(
        canonical["train"]
    )
    assert relative_difference > 0.1
    assert max(metadata["reconstruction_errors"].values()) < 1e-8
    assert max(recoded_metadata["reconstruction_errors"].values()) < 1e-8
