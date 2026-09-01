"""Check deterministic identity and summarize timed late-panel refits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CONFIG = json.loads((HERE / "openml_late_source_cover_config.json").read_text())
ORIGINAL = RESULTS / "openml_late_source_cover"
TIMED = RESULTS / "openml_late_source_timed_refit"


def main() -> None:
    rows, mismatches = [], []
    for dataset in CONFIG["datasets"]:
        for model in CONFIG["models"]:
            stem = f"{dataset}__{model}"
            manifest = json.loads((TIMED / f"{stem}.json").read_text())
            with np.load(ORIGINAL / f"{stem}.npz") as original, np.load(TIMED / f"{stem}.npz") as timed:
                keys = sorted(original.files)
                exact = all(np.array_equal(original[key], timed[key]) for key in keys)
                if not exact:
                    mismatches.append(stem)
                tensor_bytes = int(sum(timed[key].nbytes for key in keys))
            rows.append({
                "dataset": dataset, "model": model, "fits": manifest["fits"],
                "exact_artifact_match": exact, "tensor_bytes": tensor_bytes,
                **manifest["timing"],
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "timed_refit_cells.csv", index=False)
    by_model = frame.groupby("model").agg(
        cells=("dataset", "size"),
        median_end_to_end_seconds=("end_to_end_seconds", "median"),
        median_fit_loop_seconds=("fit_loop_seconds", "median"),
        median_fits_per_second=("fits_per_fit_loop_second", "median"),
        maximum_end_to_end_seconds=("end_to_end_seconds", "max"),
    ).reset_index()
    summary = {
        "status": "complete", "timed_cells": int(len(frame)),
        "fits_per_cell": 128, "total_timed_fits": int(frame.fits.sum()),
        "concurrent_process_cap": 4,
        "exact_artifact_matches": int(frame.exact_artifact_match.sum()),
        "mismatches": mismatches,
        "model_family_timing": by_model.to_dict(orient="records"),
        "median_end_to_end_seconds_all_cells": float(frame.end_to_end_seconds.median()),
        "maximum_end_to_end_seconds_all_cells": float(frame.end_to_end_seconds.max()),
        "portable_runtime_claimed": False,
    }
    (RESULTS / "timed_refit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
