"""Shared locked data, cache, metric, and provenance helpers."""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DAY9 = ROOT.parent
SAFE_ROOT = DAY9 / "safe_basis_control"
TOURNAMENT_ROOT = DAY9 / "basis_controlled_tournament"
CONFIRMATION_ROOT = DAY9 / "basis_dependence_confirmation"
PROTOCOL_PATH = ROOT / "configs" / "GUARDED_PROTOCOL.json"
PANEL_PATH = ROOT / "configs" / "GUARDED_PROSPECTIVE_PANEL.json"
BLACKLIST_PATH = ROOT / "configs" / "PRIOR_DATASET_BLACKLIST.json"
FINALISTS_PATH = ROOT / "configs" / "GUARDED_FINALISTS.json"
FINALISTS_SHA_PATH = ROOT / "configs" / "GUARDED_FINALISTS.sha256"

for path in (SAFE_ROOT, TOURNAMENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from safe_basis.common import (  # noqa: E402
    bd,
    disagreement,
    load_prediction_bundle,
    mix_predictions,
    normalized_excess_risk,
    save_prediction_bundle,
    sha256_file,
    task_error,
    write_json,
)
from safe_basis.models import fit_predictions  # noqa: E402
from safe_basis.rankgram import build_rank_adaptive_interface  # noqa: E402
from tournament.representations import build_interface  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_protocol() -> dict[str, Any]:
    return load_json(PROTOCOL_PATH)


def frozen_hashes(*, finalist_hash: str | None = None) -> dict[str, str]:
    result = {
        "protocol": sha256_file(PROTOCOL_PATH),
        "panel": sha256_file(PANEL_PATH),
        "blacklist": sha256_file(BLACKLIST_PATH),
    }
    if finalist_hash is not None:
        result["finalists"] = finalist_hash
    return result


def data_config(protocol: dict[str, Any], *, basis_dimension: int | None = None) -> dict[str, Any]:
    return {
        "split_seed": int(protocol["split_seed"]),
        "max_train_rows": int(protocol["max_train_rows"]),
        "max_validation_rows": int(protocol["max_validation_rows"]),
        "max_test_rows": int(protocol["max_test_rows"]),
        "minimum_continuous_unique_values": int(protocol["minimum_continuous_unique_values"]),
        "basis_dimension": int(protocol["basis_dimension"] if basis_dimension is None else basis_dimension),
    }


def _panel_index(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["key"]): dict(row) for row in load_json(path)["datasets"]}


def development_specs(protocol: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    protocol = load_protocol() if protocol is None else protocol
    records: dict[str, dict[str, Any]] = {}
    for path in (
        CONFIRMATION_ROOT / "configs" / "dataset_panel.json",
        TOURNAMENT_ROOT / "configs" / "NEW_PROSPECTIVE_PANEL.json",
        SAFE_ROOT / "configs" / "NEW_TAIL_PROSPECTIVE_PANEL.json",
    ):
        records.update(_panel_index(path))
    missing = set(protocol["development_datasets"]) - set(records)
    if missing:
        raise RuntimeError(f"guarded development specifications missing: {sorted(missing)}")
    return [{**records[key], "panel": "guarded_development"} for key in protocol["development_datasets"]]


def load_blocks(
    spec: dict[str, Any], protocol: dict[str, Any] | None = None, *, basis_dimension: int | None = None
) -> Any:
    protocol = load_protocol() if protocol is None else protocol
    config = data_config(protocol, basis_dimension=basis_dimension)
    if spec["key"] == "SoilKsatDB" and spec.get("panel") == "guarded_new_untouched_prospective":
        dataset = _load_soilksat_with_observed_target(spec, config)
    else:
        dataset = bd.load_dataset(spec, config)
    matrix_config = dict(config)
    if spec["key"] == "2dplanes" and spec.get("panel") == "guarded_new_untouched_prospective":
        # All ten frozen inputs are genuinely numeric but take only 2--3 values.
        # Keep them as low-rank RBF blocks rather than declaring the locked
        # dataset unusable after prospective access.
        matrix_config["minimum_continuous_unique_values"] = 2
    return bd.build_rbf_feature_matrix(dataset, matrix_config)


def _load_soilksat_with_observed_target(spec: dict[str, Any], config: dict[str, Any]) -> Any:
    """Mirror the frozen loader after dropping OpenML rows with missing ksat_lab.

    OpenML task 42332 designates ``ksat_lab`` as its target, but 4,369 rows
    have no observed value.  ``fetch_openml`` rejects such a target before the
    shared loader can preprocess it, so fetch all columns, retain the 8,703
    observed target rows, and then apply the exact frozen split/subsample logic.
    """

    import pandas as pd
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split

    bunch = fetch_openml(
        data_id=int(spec["openml_id"]), as_frame=True, parser="auto", target_column=None
    )
    version = int(bunch.details["version"])
    if version != int(spec["openml_version"]):
        raise RuntimeError(f"OpenML version drift for {spec['key']}: {version}")
    frame = bunch.data.copy()
    frame.columns = [str(column) for column in frame.columns]
    target = pd.to_numeric(frame.pop("ksat_lab"), errors="coerce")
    observed = target.notna().to_numpy()
    if int(observed.sum()) != 8703 or int((~observed).sum()) != 4369:
        raise RuntimeError(
            "SoilKsatDB observed-target count drift: "
            f"observed={int(observed.sum())}, missing={int((~observed).sum())}"
        )
    X = frame.loc[observed].reset_index(drop=True)
    raw_y = target.loc[observed].to_numpy(dtype=float)
    all_indices = np.arange(len(X), dtype=np.int64)
    outer, test = train_test_split(
        all_indices, test_size=0.2, random_state=int(config["split_seed"])
    )
    train, validation = train_test_split(
        outer, test_size=0.2, random_state=int(config["split_seed"]) + 1
    )
    train = bd._subsample(
        train, raw_y, int(config["max_train_rows"]), int(config["split_seed"]) + 2, False
    )
    validation = bd._subsample(
        validation,
        raw_y,
        int(config["max_validation_rows"]),
        int(config["split_seed"]) + 3,
        False,
    )
    test = bd._subsample(
        test, raw_y, int(config["max_test_rows"]), int(config["split_seed"]) + 4, False
    )
    nominal = [
        column
        for column in X.columns
        if isinstance(X[column].dtype, pd.CategoricalDtype)
        or X[column].dtype == object
        or X[column].dtype.name == "string"
    ]
    numerical = [column for column in X.columns if column not in nominal]
    prepared = X.copy()
    for column in nominal:
        prepared[column] = prepared[column].astype("string").fillna("__MISSING__")
    for column in numerical:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce").astype(float)
    frames = [prepared.iloc[index].reset_index(drop=True) for index in (train, validation, test)]
    targets = [raw_y[index].astype(float) for index in (train, validation, test)]
    return bd.DatasetData(
        key=str(spec["key"]),
        openml_id=int(spec["openml_id"]),
        openml_version=version,
        panel=str(spec["panel"]),
        problem_type=str(spec["problem_type"]),
        X_train_raw=frames[0],
        X_validation_raw=frames[1],
        X_test_raw=frames[2],
        y_train=targets[0],
        y_validation=targets[1],
        y_test=targets[2],
        train_indices=train,
        validation_indices=validation,
        test_indices=test,
        nominal_columns=nominal,
        numerical_columns=numerical,
        cyclic_periods={str(key): int(value) for key, value in spec.get("cyclic_periods", {}).items()},
    )


def orthogonal_orbit(blocks: Any, protocol: dict[str, Any] | None = None) -> list[Any]:
    protocol = load_protocol() if protocol is None else protocol
    reps = bd.build_primary_representations(blocks, int(protocol["orbit_members"]))
    selected = [rep for rep in reps if rep.is_reference or rep.variant == "orthogonal_all"]
    selected.sort(key=lambda rep: (-int(rep.is_reference), rep.member))
    if len(selected) != int(protocol["orbit_members"]) + 1:
        raise RuntimeError(f"orthogonal orbit incomplete for {blocks.dataset.key}: {len(selected)}")
    return selected


def _validate_prediction_shapes(
    blocks: Any, orbit: list[Any], raw: dict[str, dict[str, np.ndarray]], gram: dict[str, np.ndarray]
) -> None:
    expected_ids = {rep.representation_id for rep in orbit}
    if set(raw) != expected_ids:
        raise RuntimeError(f"raw orbit ID mismatch for {blocks.dataset.key}")
    lengths = {"validation": len(blocks.dataset.y_validation), "test": len(blocks.dataset.y_test)}
    for rep_id, values in raw.items():
        for split, length in lengths.items():
            if len(values[split]) != length:
                raise RuntimeError(f"prediction length mismatch at {blocks.dataset.key}/{rep_id}/{split}")
    for split, length in lengths.items():
        if len(gram[split]) != length:
            raise RuntimeError(f"Gram prediction length mismatch at {blocks.dataset.key}/{split}")


def _safe_prospective_predictions(
    model: str, dataset: str, seed: int
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray], dict[str, Any]] | None:
    root = SAFE_ROOT / "results" / "raw" / "prospective" / model / dataset / f"seed_{seed}"
    raw_paths = sorted((root / "Raw").glob("*.npz"))
    gram_path = root / "GramAnchor-m16.npz"
    if len(raw_paths) != 9 or not gram_path.exists():
        return None
    raw = {path.stem: load_prediction_bundle(path)[0] for path in raw_paths}
    gram = load_prediction_bundle(gram_path)[0]
    return raw, gram, {"source": "safe_basis_prospective", "root": str(root)}


def _earlier_frozen_predictions(
    model: str, dataset: str, seed: int
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray], dict[str, Any]] | None:
    try:
        from safe_basis.common import load_frozen_development_predictions

        return load_frozen_development_predictions(model, dataset, seed)
    except (FileNotFoundError, RuntimeError, KeyError):
        return None


def _cached_fit(
    path: Path,
    *,
    model: str,
    blocks: Any,
    rep: Any,
    seed: int,
    device: str,
    definition: dict[str, Any],
    locked_hashes: dict[str, str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    hashes = frozen_hashes() if locked_hashes is None else dict(locked_hashes)
    if path.exists() and path.with_suffix(".json").exists():
        arrays, metadata = load_prediction_bundle(path)
        if metadata.get("frozen_hashes") != hashes or metadata.get("definition") != definition:
            raise RuntimeError(f"guarded development cache drift at {path}")
        return arrays, metadata
    prediction, telemetry = fit_predictions(
        model,
        blocks.dataset.problem_type,
        rep,
        blocks.dataset.y_train,
        blocks.dataset.y_validation,
        seed,
        device,
    )
    metadata = {
        "status": "COMPLETE",
        "dataset": blocks.dataset.key,
        "model": model,
        "seed": int(seed),
        "definition": definition,
        "frozen_hashes": hashes,
        "telemetry": telemetry,
    }
    save_prediction_bundle(path, None, prediction["validation"], prediction["test"], metadata)
    return prediction, metadata


def cached_representation_predictions(
    path: Path,
    *,
    model: str,
    blocks: Any,
    rep: Any,
    seed: int,
    device: str,
    definition: dict[str, Any],
    finalist_hash: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit or load one protocol-bound representation prediction bundle.

    Method scripts use this public wrapper so every new cache is checked against
    the three pre-outcome locks in exactly the same way as the shared Raw/Gram
    development cache.
    """

    return _cached_fit(
        path,
        model=model,
        blocks=blocks,
        rep=rep,
        seed=seed,
        device=device,
        definition=definition,
        locked_hashes=frozen_hashes(finalist_hash=finalist_hash),
    )


def development_base_predictions(
    model: str,
    blocks: Any,
    orbit: list[Any],
    seed: int,
    device: str,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray], dict[str, Any]]:
    prior = _safe_prospective_predictions(model, blocks.dataset.key, seed)
    if prior is None:
        prior = _earlier_frozen_predictions(model, blocks.dataset.key, seed)
    if prior is not None:
        raw, gram, source = prior
        _validate_prediction_shapes(blocks, orbit, raw, gram)
        return raw, gram, source

    root = ROOT / "results" / "raw" / "development_base" / model / blocks.dataset.key / f"seed_{seed}"
    raw: dict[str, dict[str, np.ndarray]] = {}
    raw_seconds = 0.0
    for rep in orbit:
        prediction, metadata = _cached_fit(
            root / "Raw" / f"{rep.representation_id}.npz",
            model=model,
            blocks=blocks,
            rep=rep,
            seed=seed,
            device=device,
            definition={"method": "Raw", "representation_id": rep.representation_id},
        )
        raw[rep.representation_id] = prediction
        raw_seconds += float(metadata["telemetry"].get("fit_seconds", 0.0))
    gram_rep = build_interface(
        orbit[0], "gram_anchor", blocks.dataset.key, anchors=16, selection="gram_pivot", normalize=True
    )
    gram, gram_meta = _cached_fit(
        root / "GramAnchor-m16.npz",
        model=model,
        blocks=blocks,
        rep=gram_rep,
        seed=seed,
        device=device,
        definition={
            "method": "GramAnchor-m16",
            "anchors": 16,
            "selection": "gram_pivot",
            "normalize": True,
            "coordinate_standardization": True,
        },
    )
    _validate_prediction_shapes(blocks, orbit, raw, gram)
    return raw, gram, {
        "source": "guarded_development_fit",
        "root": str(root),
        "raw_fit_seconds": raw_seconds,
        "gram_fit_seconds": float(gram_meta["telemetry"].get("fit_seconds", 0.0)),
    }


def rank_prediction(
    model: str,
    blocks: Any,
    reference: Any,
    seed: int,
    device: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    config = {
        "relative_threshold": 1e-4,
        "anchor_rule": "rank",
        "normalization": "N1_anchor_norm",
        "standardize": True,
    }
    rep = build_rank_adaptive_interface(reference, blocks.dataset.key, **config)
    path = ROOT / "results" / "raw" / "development_rank" / model / blocks.dataset.key / f"seed_{seed}.npz"
    return _cached_fit(
        path,
        model=model,
        blocks=blocks,
        rep=rep,
        seed=seed,
        device=device,
        definition={"method": "RankAdaptiveGram", **config},
    )


def prospective_base_predictions(
    model: str,
    blocks: Any,
    orbit: list[Any],
    seed: int,
    device: str,
    finalist_hash: str,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray], dict[str, Any]]:
    """Fit/load the finalist-hash-bound Raw orbit and invariant Gram model."""

    root = ROOT / "results" / "raw" / "prospective" / "general" / model / blocks.dataset.key / f"seed_{seed}"
    raw: dict[str, dict[str, np.ndarray]] = {}
    raw_seconds = 0.0
    for rep in orbit:
        prediction, metadata = cached_representation_predictions(
            root / "Raw" / f"{rep.representation_id}.npz",
            model=model,
            blocks=blocks,
            rep=rep,
            seed=seed,
            device=device,
            finalist_hash=finalist_hash,
            definition={"method": "Raw", "representation_id": rep.representation_id},
        )
        raw[rep.representation_id] = prediction
        raw_seconds += float(metadata["telemetry"].get("fit_seconds", 0.0))
    gram_rep = build_interface(
        orbit[0], "gram_anchor", blocks.dataset.key, anchors=16, selection="gram_pivot", normalize=True
    )
    gram, gram_meta = cached_representation_predictions(
        root / "GramAnchor-m16.npz",
        model=model,
        blocks=blocks,
        rep=gram_rep,
        seed=seed,
        device=device,
        finalist_hash=finalist_hash,
        definition={
            "method": "GramAnchor-m16",
            "anchors": 16,
            "selection": "gram_pivot",
            "normalize": True,
            "coordinate_standardization": True,
        },
    )
    _validate_prediction_shapes(blocks, orbit, raw, gram)
    return raw, gram, {
        "source": "guarded_locked_prospective_fit",
        "root": str(root),
        "raw_fit_seconds": raw_seconds,
        "gram_fit_seconds": float(gram_meta["telemetry"].get("fit_seconds", 0.0)),
    }


def prospective_rank_prediction(
    model: str,
    blocks: Any,
    reference: Any,
    seed: int,
    device: str,
    finalist_hash: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit/load the frozen SafeRank representation under the finalist hash."""

    config = {
        "relative_threshold": 1e-4,
        "anchor_rule": "rank",
        "normalization": "N1_anchor_norm",
        "standardize": True,
    }
    rep = build_rank_adaptive_interface(reference, blocks.dataset.key, **config)
    path = (
        ROOT / "results" / "raw" / "prospective" / "general" / model / blocks.dataset.key
        / f"seed_{seed}" / "RankAdaptiveGram.npz"
    )
    return cached_representation_predictions(
        path,
        model=model,
        blocks=blocks,
        rep=rep,
        seed=seed,
        device=device,
        finalist_hash=finalist_hash,
        definition={"method": "RankAdaptiveGram", **config},
    )


def finalist_lock() -> tuple[dict[str, Any], str]:
    for path in (PANEL_PATH, PROTOCOL_PATH, BLACKLIST_PATH, FINALISTS_PATH, FINALISTS_SHA_PATH):
        if not path.exists():
            raise RuntimeError(f"guarded prospective access refused: missing {path.name}")
    expected = FINALISTS_SHA_PATH.read_text().split()[0]
    actual = sha256_file(FINALISTS_PATH)
    if expected != actual:
        raise RuntimeError("guarded prospective access refused: finalist SHA256 mismatch")
    config = load_json(FINALISTS_PATH)
    if config.get("status") != "FROZEN_BEFORE_GUARDED_PROSPECTIVE_DATA_ACCESS":
        raise RuntimeError("guarded prospective access refused: finalist status is not frozen")
    if not 1 <= len(config.get("finalists", [])) <= 4:
        raise RuntimeError("guarded prospective access refused: finalist count must be 1--4")
    if config.get("prospective_panel_sha256") != sha256_file(PANEL_PATH):
        raise RuntimeError("guarded prospective access refused: panel hash drift")
    if config.get("protocol_sha256") != sha256_file(PROTOCOL_PATH):
        raise RuntimeError("guarded prospective access refused: protocol hash drift")
    for path in (PANEL_PATH, PROTOCOL_PATH, BLACKLIST_PATH):
        sidecar = path.with_suffix(".sha256")
        if not sidecar.exists() or sidecar.read_text().split()[0] != sha256_file(path):
            raise RuntimeError(f"guarded prospective access refused: {path.name} sidecar drift")
    prospective_root = ROOT / "results" / "raw" / "prospective"
    artifacts = [path for path in prospective_root.rglob("*") if path.is_file()]
    finalist_mtime = FINALISTS_PATH.stat().st_mtime_ns
    if any(path.stat().st_mtime_ns <= finalist_mtime for path in artifacts):
        raise RuntimeError(
            "guarded prospective access refused: a prospective artifact does not postdate the finalist lock"
        )
    return config, actual


def prospective_specs() -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    finalists, digest = finalist_lock()
    panel = load_json(PANEL_PATH)
    if panel.get("status") != "LOCKED_BEFORE_GUARDED_DEVELOPMENT_OUTCOME_ACCESS":
        raise RuntimeError("guarded prospective access refused: panel status drift")
    specs = [
        {
            **{
                key: value
                for key, value in row.items()
                if key in {"key", "openml_id", "openml_version", "problem_type", "cyclic_periods"}
            },
            "panel": "guarded_new_untouched_prospective",
        }
        for row in panel["datasets"]
    ]
    return specs, finalists, digest


def with_test(rep: Any, test: np.ndarray) -> Any:
    """Return a representation with a diagnostic prediction matrix as test."""

    return dataclasses.replace(rep, X_test=np.asarray(test))
