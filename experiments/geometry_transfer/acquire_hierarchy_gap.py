#!/usr/bin/env python3
"""Materialize the frozen Census CBP hierarchy addendum from its bulk archive."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
FOLDER = HERE / "prospective_data" / "census_cbp_naics"
ARCHIVE = FOLDER / "source" / "cbp23st.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prefix_distance(a: str, b: str) -> int:
    levels = (2, 3, 4, 5, 6)
    shared = sum(a[:level] == b[:level] for level in levels)
    return 2 * (len(levels) - shared)


def main() -> None:
    with zipfile.ZipFile(ARCHIVE) as archive:
        with archive.open("cbp23st.txt") as stream:
            frame = pd.read_csv(
                stream,
                dtype={"fipstate": "string", "naics": "string", "lfo": "string"},
                usecols=["fipstate", "naics", "lfo", "emp", "ap", "est"],
            )

    # Bulk-file '-' is the all-legal-forms record corresponding to API LFO=001;
    # the top-level emp/ap/est columns are the all-size totals (API EMPSZES=001).
    frame = frame.loc[frame.lfo.eq("-")].copy()
    frame = frame.loc[frame.naics.str.fullmatch(r"[0-9]{6}", na=False)].copy()
    for column in ["emp", "ap", "est"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[(frame[["emp", "ap", "est"]] > 0).all(axis=1)].copy()
    counts = frame.groupby("naics").fipstate.nunique()
    keep = counts[counts >= 10].index
    frame = frame.loc[frame.naics.isin(keep)].copy()
    frame = frame.sort_values(["naics", "fipstate"], kind="stable").reset_index(drop=True)

    states = pd.DataFrame({"state_id": sorted(frame.naics.unique())})
    codes = states.state_id.tolist()
    distance = np.asarray([[prefix_distance(a, b) for b in codes] for a in codes], dtype=np.float32)
    rows = pd.DataFrame(
        {
            "target": np.log1p(frame.ap.astype(float)),
            "field_state": frame.naics.astype(str),
            "geo_state": frame.fipstate.astype(str),
            "log1p_emp": np.log1p(frame.emp.astype(float)),
            "log1p_estab": np.log1p(frame.est.astype(float)),
        }
    )

    if len(states) < 8 or len(rows) < 500:
        raise RuntimeError(f"frozen minimum not met: states={len(states)}, rows={len(rows)}")
    rows.to_parquet(FOLDER / "rows.parquet", index=False)
    states.to_csv(FOLDER / "states.csv", index=False)
    np.save(FOLDER / "distance_primary.npy", distance)
    np.save(FOLDER / "distance_domain.npy", distance)
    manifest = {
        "status": "RUN",
        "source": "census_cbp_naics",
        "family": "official_industry_hierarchy",
        "rows": len(rows),
        "states": len(states),
        "ordinary_covariates": ["geo_state", "log1p_emp", "log1p_estab"],
        "target": "log1p(ap)",
        "target_units": "log1p annual payroll in thousands of dollars",
        "archive_url": "https://www2.census.gov/programs-surveys/cbp/datasets/2023/cbp23st.zip",
        "archive_sha256": sha256(ARCHIVE),
        "bulk_to_api_mapping": {"lfo=-": "LFO=001", "top-level emp/ap/est": "EMPSZES=001"},
        "filters": {"naics": "six ASCII digits", "minimum_geographies": 10, "positive": ["emp", "ap", "est"]},
        "geometry": "NAICS prefix-tree path over prefix lengths 2,3,4,5,6",
    }
    (FOLDER / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
