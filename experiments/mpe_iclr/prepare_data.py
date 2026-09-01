#!/usr/bin/env python3
"""Prepare the frozen MPE field panel without consulting prediction outcomes.

Raw archives live outside the repository by default.  Every runnable task is
written in one small, common format under ``processed/<task>``:

* ``rows.parquet``: stable row ID, metric-field state, target, and frozen
  ordinary covariates;
* ``states.parquet``: the complete target-independent state codebook;
* ``distance_primary.npy``: the declared metric in codebook order;
* ``splits.json``: five prospectively seeded state-disjoint partitions; and
* ``manifest.json`` plus a dataset leakage audit.

The target is carried into the row table for later training, but is never
passed to any metric, state, sampling, split, landmark, or representation
constructor in this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import networkx as nx
import numpy as np
import pandas as pd
import shapefile
from pyproj import Transformer
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from shapely.geometry import shape
from sklearn.datasets import fetch_openml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mpe import (  # noqa: E402
    EARTH_RADIUS_KM,
    assert_disjoint_states,
    haversine_distance,
    make_state_partition,
    normalize_string,
    stable_seed,
    validate_distance_matrix,
)


GLOBAL_SEED = 20260829
SPLIT_SEEDS = [20261001, 20261002, 20261003, 20261004, 20261005]


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_values(values: Iterable[object], namespace: str) -> np.ndarray:
    prefix = f"{GLOBAL_SEED}|{namespace}|".encode()
    return np.fromiter(
        (
            int.from_bytes(
                hashlib.sha256(prefix + str(value).encode("utf-8", "surrogatepass")).digest()[:8],
                "big",
            )
            for value in values
        ),
        dtype=np.uint64,
    )


def deterministic_take(frame: pd.DataFrame, cap: int, namespace: str) -> pd.DataFrame:
    """Keep the smallest frozen SHA-256 row hashes, independent of target."""
    if len(frame) <= cap:
        return frame.copy()
    hashes = sha256_values(frame["row_id"].astype(str), namespace)
    selected = np.argpartition(hashes, cap - 1)[:cap]
    selected = selected[np.lexsort((frame.iloc[selected]["row_id"].astype(str), hashes[selected]))]
    return frame.iloc[selected].copy()


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def clean_code(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    if text.endswith(".0"):
        text = text[:-2]
    return re.sub(r"[^0-9A-Z]", "", text)


def json_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def frame_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    ordered = frame.sort_values("row_id", kind="stable")
    for row in ordered.itertuples(index=False, name=None):
        digest.update("\x1f".join(map(str, row)).encode("utf-8", "surrogatepass"))
        digest.update(b"\n")
    return digest.hexdigest()


def hierarchy_geometry(
    states: Sequence[str], prefix_lengths: Sequence[int], root_name: str
) -> tuple[np.ndarray, dict[str, list[str]], dict[str, str]]:
    """Create an unweighted prefix hierarchy and leaf-to-leaf path metric."""
    graph = nx.Graph()
    graph.add_node(root_name)
    paths: dict[str, list[str]] = {}
    parents: dict[str, str] = {}
    state_nodes: list[str] = []
    for state in states:
        code = clean_code(state)
        lengths = [length for length in prefix_lengths if length <= len(code)]
        if not lengths:
            raise ValueError(f"state has no hierarchy prefix: {state!r}")
        nodes: list[str] = []
        previous = root_name
        for length in lengths:
            node = f"L{length}:{code[:length]}"
            graph.add_edge(previous, node)
            nodes.append(node)
            previous = node
        leaf = f"STATE:{state}"
        if previous != leaf:
            graph.add_edge(previous, leaf)
        state_nodes.append(leaf)
        paths[str(state)] = [root_name, *nodes, leaf]
        parents[str(state)] = nodes[-2] if len(nodes) >= 2 else root_name
    distance = np.asarray(
        [[nx.shortest_path_length(graph, left, right) for right in state_nodes] for left in state_nodes],
        dtype=np.float64,
    )
    return distance, paths, parents


def hierarchy_from_paths(
    states: Sequence[str], published_paths: Sequence[Sequence[str]]
) -> tuple[np.ndarray, dict[str, list[str]], dict[str, str]]:
    graph = nx.Graph()
    root = "AMAZON_ROOT"
    graph.add_node(root)
    all_paths: dict[str, list[str]] = {}
    for path in published_paths:
        clean = [normalize_string(item) for item in path if normalize_string(item)]
        previous = root
        nodes = [root]
        for depth, label in enumerate(clean, 1):
            node = f"L{depth}:{label}"
            graph.add_edge(previous, node)
            nodes.append(node)
            previous = node
        if clean:
            all_paths[clean[-1]] = nodes
    state_nodes = []
    parents: dict[str, str] = {}
    for state in states:
        key = normalize_string(state)
        if key not in all_paths:
            raise KeyError(f"missing Amazon hierarchy path for {state}")
        node = all_paths[key][-1]
        state_nodes.append(node)
        parents[str(state)] = all_paths[key][-2]
    distance = np.asarray(
        [[nx.shortest_path_length(graph, left, right) for right in state_nodes] for left in state_nodes],
        dtype=np.float64,
    )
    return distance, all_paths, parents


def string_jaccard_distance(states: Sequence[str], n: int = 3) -> np.ndarray:
    """Dense character n-gram Jaccard metric via sparse intersections."""
    vocabulary: dict[str, int] = {}
    rows: list[int] = []
    cols: list[int] = []
    for row, state in enumerate(states):
        text = f"  {normalize_string(state)}  "
        grams = {text[index : index + n] for index in range(max(1, len(text) - n + 1))}
        for gram in grams:
            column = vocabulary.setdefault(gram, len(vocabulary))
            rows.append(row)
            cols.append(column)
    matrix = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(states), len(vocabulary)))
    intersections = (matrix @ matrix.T).toarray().astype(np.float64)
    sizes = np.asarray(matrix.sum(axis=1)).ravel()
    unions = sizes[:, None] + sizes[None, :] - intersections
    similarity = np.divide(intersections, unions, out=np.ones_like(intersections), where=unions > 0)
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    return distance


def _split_payload(states: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split_index, seed in enumerate(SPLIT_SEEDS):
        parts = make_state_partition(states, seed)
        assert_disjoint_states(parts)
        result[str(split_index)] = {
            "seed": seed,
            **{name: [str(item) for item in values] for name, values in parts.items()},
        }
    return result


def write_leakage_audit(
    task: str,
    source: str,
    metric_definition: str,
    metric_inputs: Sequence[str],
    unavailable: Sequence[str],
    status: str = "RUN",
    notes: str = "",
) -> None:
    text = f"""# Leakage Audit — {task}

- Status: `{status}`
- Source: {source}
- Metric definition: {metric_definition}
- Information used to define the metric: {', '.join(metric_inputs)}.
- Was any prediction target used in metric construction? **No.**
- Are held-out states known structurally at inference? **Yes.** State identifiers and
  externally published ontology/coordinate/string metadata are transductively known;
  their outcomes are not.
- Are held-out labels used in representation construction, landmark selection,
  bandwidth selection, preprocessing, or splitting? **No.** Landmarks and learned
  preprocessing are fit later from training states/rows only.
- Information unavailable at prediction time and therefore excluded: {', '.join(unavailable)}.
- The state partitions and row cap use identifiers and frozen hashes only. Target
  values are retained solely for downstream fitting/evaluation after construction.
{notes}
"""
    (HERE / f"LEAKAGE_AUDIT_{task}.md").write_text(text)


def write_task(
    task: str,
    source: str,
    frame: pd.DataFrame,
    distance: np.ndarray,
    state_metadata: pd.DataFrame,
    *,
    metric_name: str,
    metric_inputs: Sequence[str],
    ordinary_covariates: Sequence[str],
    unavailable: Sequence[str],
    input_paths: Sequence[Path],
    minimum_rows_per_state: int,
    auxiliary_columns: Sequence[str] = (),
    extra_arrays: dict[str, np.ndarray] | None = None,
    extra_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    required = ["row_id", "field_state", "target", *ordinary_covariates, *auxiliary_columns]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise KeyError(f"{task}: missing output columns {missing}")
    frame = frame[required].copy()
    frame["field_state"] = frame["field_state"].astype(str)
    frame = frame.dropna(subset=required)
    counts = frame["field_state"].value_counts()
    eligible_states = sorted(counts[counts >= minimum_rows_per_state].index.astype(str), key=str)
    frame = frame[frame["field_state"].isin(eligible_states)].copy()
    frame = frame.sort_values("row_id", kind="stable").reset_index(drop=True)
    metadata = state_metadata.copy()
    metadata["state_id"] = metadata["state_id"].astype(str)
    metadata = metadata.drop_duplicates("state_id").set_index("state_id").loc[eligible_states].reset_index()
    original_state_ids = state_metadata["state_id"].astype(str).tolist()
    lookup = {state: index for index, state in enumerate(original_state_ids)}
    indices = [lookup[state] for state in eligible_states]
    d = np.asarray(distance, dtype=np.float64)[np.ix_(indices, indices)]
    metric_audit = validate_distance_matrix(d, triangle=True)
    if set(frame["field_state"]) != set(metadata["state_id"]):
        raise AssertionError(f"{task}: rows and codebook states disagree")
    if len(metadata) < 15:
        raise ValueError(f"{task}: only {len(metadata)} eligible states")

    output = HERE / "processed" / task
    output.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output / "rows.parquet", index=False, compression="zstd")
    metadata.to_parquet(output / "states.parquet", index=False, compression="zstd")
    np.save(output / "distance_primary.npy", d.astype(np.float32))
    arrays_written: list[str] = []
    for name, array in (extra_arrays or {}).items():
        array_np = np.asarray(array)
        if array_np.ndim >= 2 and array_np.shape[0] == len(original_state_ids):
            array_np = array_np[indices]
            if array_np.ndim == 2 and array_np.shape[1] == len(original_state_ids):
                array_np = array_np[:, indices]
        np.save(output / f"{name}.npy", array_np)
        arrays_written.append(name)
    splits = _split_payload(eligible_states)
    json_dump(splits, output / "splits.json")

    target_finite = bool(np.isfinite(numeric(frame["target"])).all())
    if not target_finite:
        raise AssertionError(f"{task}: non-finite target survived preparation")
    manifest = {
        "task": task,
        "source_unit": source,
        "status": "RUN",
        "rows": len(frame),
        "states": len(metadata),
        "minimum_rows_per_state": minimum_rows_per_state,
        "min_state_rows": int(frame["field_state"].value_counts().min()),
        "max_state_rows": int(frame["field_state"].value_counts().max()),
        "metric": metric_name,
        "metric_inputs": list(metric_inputs),
        "metric_target_independent": True,
        "ordinary_covariates": list(ordinary_covariates),
        "target_column": "target",
        "row_table_sha256": frame_sha256(frame),
        "metric_audit": metric_audit,
        "input_files": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in input_paths
        ],
        "extra_arrays": arrays_written,
        "split_seeds": SPLIT_SEEDS,
        "target_used_for_sampling_or_geometry": False,
    }
    manifest.update(extra_manifest or {})
    json_dump(manifest, output / "manifest.json")
    write_leakage_audit(
        task,
        source,
        metric_name,
        metric_inputs,
        unavailable,
    )
    return manifest


def write_not_run(task: str, source: str, reason: str, evidence: dict[str, Any]) -> dict[str, Any]:
    output = HERE / "processed" / task
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"task": task, "source_unit": source, "status": "NOT RUN", "reason": reason, **evidence}
    json_dump(manifest, output / "manifest.json")
    write_leakage_audit(
        task,
        source,
        "unavailable",
        [],
        [],
        status="NOT RUN",
        notes=f"\nSchema failure: {reason}\n",
    )
    return manifest


def prepare_acs(downloads: Path) -> list[dict[str, Any]]:
    paths = [downloads / f"acs_{state}_2024.zip" for state in ("ca", "ny", "tx")]
    columns = [
        "SERIALNO", "SPORDER", "AGEP", "SCHL", "WKHP", "WKWN", "SEX",
        "COW", "ESR", "MAR", "RAC1P", "WAGP", "SOCP", "NAICSP", "STATE",
    ]
    chunks: list[pd.DataFrame] = []
    for archive in paths:
        with zipfile.ZipFile(archive) as handle:
            member = next(name for name in handle.namelist() if Path(name).name.startswith("psam_p"))
            with handle.open(member) as stream:
                for chunk in pd.read_csv(stream, usecols=columns, dtype={"SERIALNO": str, "SOCP": str, "NAICSP": str}, chunksize=250_000):
                    chunk = chunk.rename(columns={"STATE": "ST"})
                    age, wage, esr = numeric(chunk["AGEP"]), numeric(chunk["WAGP"]), numeric(chunk["ESR"])
                    valid = age.between(18, 70) & wage.gt(0) & esr.isin([1, 2])
                    chunk = chunk.loc[valid].copy()
                    if chunk.empty:
                        continue
                    chunk["row_id"] = chunk["ST"].astype(str) + "|" + chunk["SERIALNO"] + "|" + chunk["SPORDER"].astype(str)
                    chunk["target"] = np.log1p(numeric(chunk["WAGP"]))
                    chunks.append(chunk)
    base = pd.concat(chunks, ignore_index=True)
    ordinary = ["AGEP", "SCHL", "WKHP", "WKWN", "ST", "SEX", "COW", "ESR", "MAR", "RAC1P"]
    results = []
    for task, field, levels in [
        ("acs_occupation", "SOCP", [2, 3, 5, 6]),
        ("acs_industry", "NAICSP", [2, 3, 4, 5, 6]),
    ]:
        frame = base.copy()
        frame["field_state"] = frame[field].map(clean_code)
        frame = frame[frame["field_state"].str.len() >= 2]
        frame = deterministic_take(frame, 300_000, task)
        counts = frame["field_state"].value_counts()
        states = sorted(counts[counts >= 50].index.tolist(), key=str)
        distance, paths_by_state, parents = hierarchy_geometry(states, levels, f"{task.upper()}_ROOT")
        metadata = pd.DataFrame(
            {
                "state_id": states,
                "parent": [parents[state] for state in states],
                "path_json": [json.dumps(paths_by_state[state]) for state in states],
            }
        )
        results.append(
            write_task(
                task,
                "ACS",
                frame,
                distance,
                metadata,
                metric_name="unweighted shortest path on frozen official-code prefix hierarchy",
                metric_inputs=[field, "official Census crosswalk"],
                ordinary_covariates=ordinary,
                unavailable=["PERNP", "PINCP", "other earnings", "target-field aliases", "allocation flags"],
                input_paths=[
                    *paths,
                    downloads / ("census_2018_occupation_crosswalk.xlsx" if field == "SOCP" else "census_2022_industry_crosswalk.xlsx"),
                ],
                minimum_rows_per_state=50,
                extra_manifest={"field_source_column": field, "target_transform": "log1p(WAGP)"},
            )
        )
    return results


def taxi_geometry(downloads: Path, extract_root: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    archive = downloads / "tlc_taxi_zones.zip"
    destination = extract_root / "taxi_zones"
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(destination)
    shp_path = next(destination.rglob("*.shp"))
    reader = shapefile.Reader(str(shp_path))
    fields = [field[0] for field in reader.fields[1:]]
    transformer = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True)
    records: list[dict[str, Any]] = []
    geometries = []
    for shape_record in reader.iterShapeRecords():
        record = dict(zip(fields, shape_record.record))
        geometry = shape(shape_record.shape.__geo_interface__)
        centroid = geometry.centroid
        longitude, latitude = transformer.transform(centroid.x, centroid.y)
        records.append(
            {
                "state_id": str(int(record["LocationID"])),
                "zone_name": str(record.get("zone", "")),
                "borough": str(record.get("borough", "")),
                "latitude": latitude,
                "longitude": longitude,
            }
        )
        geometries.append(geometry)
    metadata = pd.DataFrame(records).sort_values("state_id", key=lambda s: s.astype(int)).reset_index(drop=True)
    coordinates = metadata[["latitude", "longitude"]].to_numpy(float)
    primary = haversine_distance(coordinates)
    adjacency = np.zeros((len(metadata), len(metadata)), dtype=np.float32)
    for left in range(len(geometries)):
        for right in range(left + 1, len(geometries)):
            if geometries[left].touches(geometries[right]) or geometries[left].distance(geometries[right]) <= 1e-7:
                adjacency[left, right] = adjacency[right, left] = 1.0
    component_count, labels = connected_components(csr_matrix(adjacency), directed=False)
    audit = {
        "polygon_count": len(metadata),
        "adjacency_edges": int(adjacency.sum() // 2),
        "adjacency_components": int(component_count),
        "component_sizes": dict(Counter(map(int, labels))),
        "source_crs": "EPSG:2263",
        "centroid_crs": "EPSG:4326",
    }
    return metadata, primary, adjacency, audit


def prepare_tlc(downloads: Path, extract_root: Path) -> list[dict[str, Any]]:
    zone_metadata, primary, adjacency, geometry_audit = taxi_geometry(downloads, extract_root)
    valid_zones = set(zone_metadata["state_id"])
    paths = [downloads / f"tlc_yellow_2024_{month:02d}.parquet" for month in (1, 4, 7, 10)]
    columns = [
        "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count",
        "RatecodeID", "PULocationID", "DOLocationID", "payment_type",
    ]
    periods: list[pd.DataFrame] = []
    for month, path in zip((1, 4, 7, 10), paths):
        frame = pd.read_parquet(path, columns=columns)
        pickup = pd.to_datetime(frame["tpep_pickup_datetime"], errors="coerce")
        dropoff = pd.to_datetime(frame["tpep_dropoff_datetime"], errors="coerce")
        duration = (dropoff - pickup).dt.total_seconds()
        pu = numeric(frame["PULocationID"]).astype("Int64").astype(str)
        do = numeric(frame["DOLocationID"]).astype("Int64").astype(str)
        valid = (
            duration.between(60, 10800)
            & pu.isin(valid_zones)
            & do.isin(valid_zones)
            & numeric(frame["passenger_count"]).notna()
            & numeric(frame["VendorID"]).notna()
            & numeric(frame["RatecodeID"]).notna()
        )
        frame = frame.loc[valid].copy()
        pickup = pickup.loc[valid]
        duration = duration.loc[valid]
        frame["pickup_hour"] = pickup.dt.hour.astype(int)
        frame["pickup_day_of_week"] = pickup.dt.dayofweek.astype(int)
        frame["target"] = np.log1p(duration)
        frame["pickup_zone"] = numeric(frame["PULocationID"]).astype(int).astype(str)
        frame["dropoff_zone"] = numeric(frame["DOLocationID"]).astype(int).astype(str)
        frame["row_id"] = (
            f"2024-{month:02d}|" + frame["VendorID"].astype(str) + "|" + pickup.astype(str)
            + "|" + frame["pickup_zone"] + "|" + frame["dropoff_zone"] + "|" + frame.index.astype(str)
        )
        periods.append(deterministic_take(frame, 75_000, f"tlc-{month:02d}"))
    base = pd.concat(periods, ignore_index=True)
    results = []
    for task, field, other in [
        ("tlc_pickup_zone", "pickup_zone", "dropoff_zone"),
        ("tlc_dropoff_zone", "dropoff_zone", "pickup_zone"),
    ]:
        frame = base.copy()
        frame["field_state"] = frame[field]
        frame["other_endpoint_zone"] = frame[other]
        ordinary = ["pickup_hour", "pickup_day_of_week", "passenger_count", "VendorID", "RatecodeID", "other_endpoint_zone"]
        results.append(
            write_task(
                task,
                "NYC_TLC",
                frame,
                primary,
                zone_metadata,
                metric_name="haversine distance between official taxi-zone polygon centroids",
                metric_inputs=["official taxi zone polygon", "EPSG:2263 to EPSG:4326 transform"],
                ordinary_covariates=ordinary,
                unavailable=["actual trip distance", "fares", "tips", "tolls", "payment type", "dropoff time components"],
                input_paths=[*paths, downloads / "tlc_taxi_zones.zip"],
                minimum_rows_per_state=50,
                auxiliary_columns=["payment_type"],
                extra_arrays={"coordinates": zone_metadata[["latitude", "longitude"]].to_numpy(float), "adjacency": adjacency},
                extra_manifest={
                    "target_transform": "log1p(trip duration seconds)",
                    "geometry_audit": geometry_audit,
                    "secondary_adjacency_metric_status": "RUN" if geometry_audit["adjacency_components"] == 1 else "NOT RUN — official polygon-touch graph disconnected",
                },
            )
        )
    return results


def iter_citibike_chunks(archive: Path, chunksize: int = 250_000):
    columns = [
        "ride_id", "rideable_type", "started_at", "ended_at", "start_station_id",
        "end_station_id", "start_lat", "start_lng", "end_lat", "end_lng", "member_casual",
    ]
    with zipfile.ZipFile(archive) as handle:
        for member in sorted(handle.namelist()):
            if not member.lower().endswith(".csv"):
                continue
            with handle.open(member) as stream:
                yield from pd.read_csv(stream, usecols=columns, chunksize=chunksize, low_memory=False)


def prepare_citibike(downloads: Path) -> list[dict[str, Any]]:
    specs = [("2024-01", "2024_01"), ("2024-07", "2024_07"), ("2025-01", "2025_01")]
    paths = [downloads / f"citibike_{suffix}.zip" for _, suffix in specs]
    sampled_periods: list[pd.DataFrame] = []
    for (period, _), path in zip(specs, paths):
        candidates: list[pd.DataFrame] = []
        for chunk_index, chunk in enumerate(iter_citibike_chunks(path)):
            started = pd.to_datetime(chunk["started_at"], errors="coerce")
            ended = pd.to_datetime(chunk["ended_at"], errors="coerce")
            duration = (ended - started).dt.total_seconds()
            valid = (
                duration.between(60, 10800)
                & chunk["ride_id"].notna()
                & chunk["start_station_id"].notna()
                & chunk["end_station_id"].notna()
                & numeric(chunk["start_lat"]).between(-90, 90)
                & numeric(chunk["start_lng"]).between(-180, 180)
            )
            frame = chunk.loc[valid].copy()
            started = started.loc[valid]
            frame["row_id"] = frame["ride_id"].astype(str)
            frame["field_state"] = frame["start_station_id"].astype(str)
            frame["target"] = np.log1p(duration.loc[valid])
            frame["start_hour"] = started.dt.hour.astype(int)
            frame["start_day_of_week"] = started.dt.dayofweek.astype(int)
            frame["period"] = period
            candidates.append(deterministic_take(frame, 150_000, f"citibike-{period}-chunk-{chunk_index}"))
        period_frame = deterministic_take(pd.concat(candidates, ignore_index=True), 150_000, f"citibike-{period}")
        sampled_periods.append(period_frame)
    base = pd.concat(sampled_periods, ignore_index=True)
    counts = base["field_state"].value_counts()
    states = sorted(counts[counts >= 50].index.tolist(), key=str)
    coordinate_rows = base[base["field_state"].isin(states)].copy()
    metadata = (
        coordinate_rows.groupby("field_state", sort=False)[["start_lat", "start_lng"]]
        .median()
        .rename(columns={"start_lat": "latitude", "start_lng": "longitude"})
        .reindex(states)
        .reset_index()
        .rename(columns={"field_state": "state_id"})
    )
    distance = haversine_distance(metadata[["latitude", "longitude"]].to_numpy(float))
    period_sets = {period: set(frame["field_state"]) for (period, _), frame in zip(specs, sampled_periods)}
    natural_validation = sorted(period_sets["2024-07"] - period_sets["2024-01"])
    natural_test = sorted(period_sets["2025-01"] - period_sets["2024-01"] - period_sets["2024-07"])
    metadata["first_period"] = metadata["state_id"].map(
        lambda state: next(period for period, _ in specs if state in period_sets[period])
    )
    ordinary = ["start_hour", "start_day_of_week", "member_casual", "rideable_type", "end_station_id"]
    manifest = write_task(
        "citibike_start_station",
        "CITI_BIKE",
        base,
        distance,
        metadata,
        metric_name="haversine distance between target-independent pooled published station coordinates",
        metric_inputs=["start_station_id", "published start_lat/start_lng across frozen periods"],
        ordinary_covariates=ordinary,
        unavailable=["end time", "duration", "future target values", "post-trip measurements"],
        input_paths=paths,
        minimum_rows_per_state=50,
        auxiliary_columns=["period"],
        extra_arrays={"coordinates": metadata[["latitude", "longitude"]].to_numpy(float)},
        extra_manifest={
            "target_transform": "log1p(trip duration seconds)",
            "natural_validation_states_before_frequency_filter": natural_validation,
            "natural_test_states_before_frequency_filter": natural_test,
            "network_metric_rule": "edges from January-2024 endpoint pairs only; constructed per split downstream",
        },
    )
    prepared_folder = HERE / "processed" / "citibike_start_station"
    prepared_states = pd.read_parquet(prepared_folder / "states.parquet")
    natural = {
        "rule": "train=January-2024 old stations; validation=July-2024 stations absent in January; test=January-2025 stations absent in both earlier windows",
        "train": prepared_states.loc[prepared_states["first_period"] == "2024-01", "state_id"].astype(str).tolist(),
        "validation": prepared_states.loc[prepared_states["first_period"] == "2024-07", "state_id"].astype(str).tolist(),
        "test": prepared_states.loc[prepared_states["first_period"] == "2025-01", "state_id"].astype(str).tolist(),
        "row_periods": {"train": "2024-01", "validation": "2024-07", "test": "2025-01"},
    }
    assert_disjoint_states({key: natural[key] for key in ("train", "validation", "test")})
    json_dump(natural, prepared_folder / "natural_split.json")
    return [manifest]


def bts_columns(header: Sequence[str]) -> dict[str, str]:
    normalized = {re.sub(r"[^a-z0-9]", "", value.lower()): value for value in header}
    aliases = {
        "flight_date": ["flightdate"],
        "month": ["month"],
        "day_of_week": ["dayofweek"],
        "reporting_airline": ["reportingairline", "opuniquecarrier"],
        "flight_number": ["flightnumberreportingairline", "opcarriernum"],
        "origin": ["origin"],
        "dest": ["dest"],
        "crs_dep_time": ["crsdeptime"],
        "crs_arr_time": ["crsarrtime"],
        "crs_elapsed_time": ["crselapsedtime"],
        "distance": ["distance"],
        "cancelled": ["cancelled"],
        "diverted": ["diverted"],
        "arr_delay": ["arrdelay"],
        "arr_del15": ["arrdel15"],
    }
    result = {}
    for key, choices in aliases.items():
        result[key] = next((normalized[item] for item in choices if item in normalized), "")
        if not result[key]:
            raise KeyError(f"BTS missing {key}; normalized columns include {sorted(normalized)[:30]}")
    return result


def iter_bts_chunks(archive: Path, chunksize: int = 250_000):
    with zipfile.ZipFile(archive) as handle:
        member = next(name for name in handle.namelist() if name.lower().endswith(".csv"))
        with handle.open(member) as stream:
            header = pd.read_csv(stream, nrows=0).columns.tolist()
        mapping = bts_columns(header)
        usecols = sorted(set(mapping.values()))
        with handle.open(member) as stream:
            for chunk in pd.read_csv(stream, usecols=usecols, chunksize=chunksize, low_memory=False):
                yield chunk.rename(columns={value: key for key, value in mapping.items()})


def prepare_bts(downloads: Path) -> list[dict[str, Any]]:
    paths = [downloads / f"bts_on_time_2024_{month:02d}.zip" for month in (1, 4, 7, 10)]
    month_frames: list[pd.DataFrame] = []
    for month, path in zip((1, 4, 7, 10), paths):
        candidates = []
        for chunk_index, frame in enumerate(iter_bts_chunks(path)):
            valid = (
                numeric(frame["cancelled"]).eq(0)
                & numeric(frame["diverted"]).eq(0)
                & numeric(frame["arr_delay"]).notna()
                & frame["origin"].notna()
                & frame["dest"].notna()
            )
            frame = frame.loc[valid].copy()
            frame["row_id"] = (
                frame["flight_date"].astype(str) + "|" + frame["reporting_airline"].astype(str)
                + "|" + frame["flight_number"].astype(str) + "|" + frame["origin"].astype(str)
                + "|" + frame["dest"].astype(str) + "|" + frame["crs_dep_time"].astype(str)
            )
            frame["target"] = numeric(frame["arr_delay"])
            candidates.append(deterministic_take(frame, 75_000, f"bts-{month:02d}-chunk-{chunk_index}"))
        month_frames.append(deterministic_take(pd.concat(candidates, ignore_index=True), 75_000, f"bts-{month:02d}"))
    base = pd.concat(month_frames, ignore_index=True)
    faa_path = downloads / "faa_aviation_facilities.csv"
    faa = pd.read_csv(faa_path, low_memory=False)
    required = ["ARPT_ID", "LAT_DECIMAL", "LONG_DECIMAL"]
    if not set(required).issubset(faa.columns):
        raise KeyError(f"FAA schema does not contain {required}")
    faa = faa[required + (["REGION_CODE"] if "REGION_CODE" in faa.columns else [])].copy()
    faa["state_id"] = faa["ARPT_ID"].astype(str).str.strip()
    faa["latitude"] = numeric(faa["LAT_DECIMAL"])
    faa["longitude"] = numeric(faa["LONG_DECIMAL"])
    faa = faa.dropna(subset=["latitude", "longitude"]).drop_duplicates("state_id")
    airport_set = set(faa["state_id"])
    base = base[base["origin"].astype(str).isin(airport_set) & base["dest"].astype(str).isin(airport_set)].copy()
    results = []
    ordinary_base = ["month", "day_of_week", "reporting_airline", "flight_number", "crs_dep_time", "crs_arr_time", "crs_elapsed_time", "distance"]
    for task, field, other in [
        ("airline_origin_airport", "origin", "dest"),
        ("airline_destination_airport", "dest", "origin"),
    ]:
        frame = base.copy()
        frame["field_state"] = frame[field].astype(str)
        frame["other_endpoint_airport"] = frame[other].astype(str)
        counts = frame["field_state"].value_counts()
        states = sorted(counts[counts >= 50].index.tolist(), key=str)
        metadata = faa.set_index("state_id").loc[states].reset_index()
        coordinates = metadata[["latitude", "longitude"]].to_numpy(float)
        primary = haversine_distance(coordinates)
        results.append(
            write_task(
                task,
                "BTS",
                frame,
                primary,
                metadata,
                metric_name="haversine distance between official FAA airport coordinates",
                metric_inputs=["Origin/Dest airport code", "FAA LAT_DECIMAL/LONG_DECIMAL"],
                ordinary_covariates=[*ordinary_base, "other_endpoint_airport"],
                unavailable=["actual departure/arrival", "taxi/wheels times", "cancellation/diversion", "delay causes"],
                input_paths=[*paths, faa_path],
                minimum_rows_per_state=50,
                auxiliary_columns=["arr_del15"],
                extra_arrays={"coordinates": coordinates},
                extra_manifest={
                    "secondary_target_column_raw": "arr_del15",
                    "route_metric_rule": "unweighted scheduled route graph from training rows only; built per split",
                },
            )
        )
    return results


def prepare_amazon(downloads: Path) -> list[dict[str, Any]]:
    path = downloads / "amazon_raw_meta_all_beauty.parquet"
    columns = [
        "main_category", "title", "average_rating", "rating_number", "features",
        "description", "price", "store", "categories", "parent_asin",
    ]
    frame = pd.read_parquet(path, columns=columns)
    price = pd.to_numeric(frame["price"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False), errors="coerce")
    category_length = frame["categories"].map(lambda value: len(value) if value is not None else 0)
    evidence = {
        "input_file": {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)},
        "raw_rows": len(frame),
        "positive_finite_price_rows": int((price > 0).sum()),
        "nonempty_category_path_rows": int((category_length > 0).sum()),
        "jointly_eligible_rows": int(((price > 0) & (category_length > 0)).sum()),
    }
    if evidence["jointly_eligible_rows"] == 0:
        return [
            write_not_run(
                "amazon_leaf_category",
                "AMAZON_2023",
                "Frozen raw_meta_All_Beauty snapshot contains no nonempty categories path, so the declared external hierarchy does not exist.",
                evidence,
            )
        ]
    valid = (price > 0) & (category_length > 0) & frame["parent_asin"].notna()
    frame = frame.loc[valid].copy()
    paths = [list(map(str, value)) for value in frame["categories"]]
    frame["field_state"] = [normalize_string(value[-1]) for value in paths]
    frame["target"] = np.log1p(price.loc[valid])
    frame["row_id"] = frame["parent_asin"].astype(str)
    frame["title_length"] = frame["title"].fillna("").astype(str).str.len()
    frame["feature_count"] = frame["features"].map(lambda value: len(value) if value is not None else 0)
    frame["description_length"] = frame["description"].map(
        lambda value: sum(len(str(item)) for item in value) if value is not None else 0
    )
    frame = deterministic_take(frame, 300_000, "amazon_leaf_category")
    states = sorted(frame["field_state"].value_counts().loc[lambda x: x >= 50].index)
    distance, all_paths, parents = hierarchy_from_paths(states, paths)
    metadata = pd.DataFrame({"state_id": states, "parent": [parents[state] for state in states], "path_json": [json.dumps(all_paths[state]) for state in states]})
    ordinary = ["main_category", "store", "average_rating", "rating_number", "title_length", "feature_count", "description_length"]
    return [
        write_task(
            "amazon_leaf_category", "AMAZON_2023", frame, distance, metadata,
            metric_name="unweighted shortest path on published product category paths",
            metric_inputs=["categories"], ordinary_covariates=ordinary,
            unavailable=["category ancestors as ordinary covariates", "review text", "price bands"],
            input_paths=[path], minimum_rows_per_state=50,
            extra_manifest={"target_transform": "log1p(price)"},
        )
    ]


def openml_frame(data_id: int, cache_root: Path) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    bunch = fetch_openml(data_id=data_id, as_frame=True, data_home=str(cache_root), parser="auto")
    frame = bunch.data.copy()
    target = pd.Series(bunch.target, index=frame.index, name=str(bunch.target.name))
    details = {"requested_data_id": data_id, "name": bunch.details.get("name"), "version": bunch.details.get("version"), "url": bunch.details.get("url")}
    return frame, target, details


def prepare_strings(downloads: Path, cache_root: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    employee, employee_target, employee_details = openml_frame(42125, cache_root)
    employee.columns = [normalize_string(column).replace(" ", "_") for column in employee.columns]
    employee_target_numeric = pd.to_numeric(employee_target.astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False), errors="coerce")
    employee["target"] = employee_target_numeric
    employee["field_state"] = employee["employee_position_title"].map(normalize_string)
    employee["row_id"] = employee.index.map(lambda value: f"openml42125|{value}")
    ordinary = ["gender", "department_name", "division", "assignment_category", "year_first_hired"]
    employee = employee.dropna(subset=["target", "field_state", *ordinary])
    states = sorted(employee["field_state"].value_counts().loc[lambda x: x >= 20].index)
    distance = string_jaccard_distance(states)
    metadata = pd.DataFrame({"state_id": states, "normalized_string": states})
    results.append(
        write_task(
            "employee_salaries", "STRING_BENCHMARK", employee, distance, metadata,
            metric_name="one minus padded character-trigram Jaccard similarity",
            metric_inputs=["employee_position_title string"], ordinary_covariates=ordinary,
            unavailable=["full name", "gross pay received", "overtime pay", "underfilled job title"],
            input_paths=[], minimum_rows_per_state=20,
            extra_manifest={"openml": employee_details, "target_transform": "none"},
        )
    )

    medical, medical_target, medical_details = openml_frame(42130, cache_root)
    medical.columns = [normalize_string(column).replace(" ", "_") for column in medical.columns]
    medical["target"] = numeric(medical_target)
    medical["field_state"] = medical["drg_definition"].map(normalize_string)
    medical["row_id"] = medical.index.map(lambda value: f"openml42130|{value}")
    medical = deterministic_take(medical, 100_000, "medical_charges")
    ordinary = ["provider_state", "average_covered_charges"]
    medical = medical.dropna(subset=["target", "field_state", *ordinary])
    states = sorted(medical["field_state"].value_counts().loc[lambda x: x >= 20].index)
    distance = string_jaccard_distance(states)
    metadata = pd.DataFrame({"state_id": states, "normalized_string": states})
    results.append(
        write_task(
            "medical_charges", "STRING_BENCHMARK", medical, distance, metadata,
            metric_name="one minus padded character-trigram Jaccard similarity",
            metric_inputs=["drg_definition string"], ordinary_covariates=ordinary,
            unavailable=["provider identity/address/city/zip", "Medicare payments", "total discharges"],
            input_paths=[], minimum_rows_per_state=20,
            extra_manifest={
                "openml": medical_details,
                "frozen_openml_id": 41444,
                "schema_deviation": "Used active canonical OpenML version 42130 because frozen version 41444 is inactive and lacks the declared target column.",
                "target_transform": "none",
            },
        )
    )

    payments, payments_target, payments_details = openml_frame(42738, cache_root)
    payments.columns = [normalize_string(column).replace(" ", "_") for column in payments.columns]
    amount_aliases = [column for column in payments if "amount" in column and "payment" in column]
    evidence = {
        "frozen_openml_id": 41442,
        "active_openml_id_inspected": 42738,
        "active_openml": payments_details,
        "available_columns": payments.columns.tolist(),
        "rows": len(payments),
        "required_amount_columns_found": amount_aliases,
    }
    if not amount_aliases:
        results.append(
            write_not_run(
                "open_payments", "STRING_BENCHMARK",
                "Frozen and active OpenML snapshots omit Total Amount of Payment, which the prospective manifest declared mandatory.",
                evidence,
            )
        )
    return results


def validate_archives(downloads: Path) -> dict[str, Any]:
    audit: dict[str, Any] = {}
    for path in sorted(downloads.iterdir()):
        if path.name.endswith(".part"):
            continue
        if path.suffix.lower() in {".zip", ".xlsx"}:
            with zipfile.ZipFile(path) as handle:
                bad = handle.testzip()
            if bad is not None:
                raise IOError(f"corrupt archive {path}: {bad}")
        audit[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    amazon = audit.get("amazon_raw_meta_all_beauty.parquet", {})
    expected_amazon = "14815cf1312a0d847364866e6876c8c73738993469067242870774c372c04387"
    if amazon and amazon["sha256"] != expected_amazon:
        raise AssertionError(f"Amazon published checksum mismatch: {amazon['sha256']}")
    json_dump(audit, HERE / "raw" / "source_checksums.json")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/data/mpe_iclr_data"))
    parser.add_argument("--only", choices=["acs", "tlc", "citibike", "bts", "amazon", "strings", "all"], default="all")
    args = parser.parse_args()
    downloads = args.data_root / "downloads"
    extract_root = args.data_root / "extracts"
    cache_root = args.data_root / "openml_cache"
    extract_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    (HERE / "processed").mkdir(exist_ok=True)
    (HERE / "raw").mkdir(exist_ok=True)

    validate_archives(downloads)
    builders = {
        "acs": lambda: prepare_acs(downloads),
        "tlc": lambda: prepare_tlc(downloads, extract_root),
        "citibike": lambda: prepare_citibike(downloads),
        "bts": lambda: prepare_bts(downloads),
        "amazon": lambda: prepare_amazon(downloads),
        "strings": lambda: prepare_strings(downloads, cache_root),
    }
    selected = list(builders) if args.only == "all" else [args.only]
    manifests: list[dict[str, Any]] = []
    for name in selected:
        print(f"prepare {name}", flush=True)
        manifests.extend(builders[name]())
        print(f"completed {name}", flush=True)
    json_dump(manifests, HERE / "raw" / f"preparation_summary_{args.only}.json")
    print(json.dumps([{key: item.get(key) for key in ("task", "status", "rows", "states")} for item in manifests], indent=2))


if __name__ == "__main__":
    main()
