#!/usr/bin/env python3
"""Run the separately frozen Census CBP hierarchy confirmation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import run_prospective as prospective


HERE = Path(__file__).resolve().parent
RAW = HERE / "raw" / "hierarchy_gap"


def main() -> None:
    gap = json.loads((HERE / "GAP_CONFIG.json").read_text())
    prospective.CONFIG = {
        "outer_seeds": gap["outer_seeds"],
        "operators": gap["operators"],
        "base_model": gap["base_model"],
    }
    prospective.RAW = RAW
    source = prospective.load(gap["source"]["name"])
    rows = prospective.run_source(source)
    RAW.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RAW / "cells.csv", index=False)
    (RAW / "availability.json").write_text(
        json.dumps([{"source": source.name, "status": "RUN"}], indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
