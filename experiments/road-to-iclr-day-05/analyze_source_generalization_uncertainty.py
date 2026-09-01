"""Source-cluster bootstrap intervals for validation-screened cover gains."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DRAWS = 100_000


def stable_seed(panel: str, comparator: str) -> int:
    return int.from_bytes(hashlib.sha256(f"source-bootstrap:{panel}:{comparator}".encode()).digest()[:8], "little")


def panel_frame(study: str, cells_path: Path, groups: dict[str, str]) -> pd.DataFrame:
    selected = pd.read_csv(RESULTS / "validation_screened_cover_cells.csv")
    chosen = selected[selected.study == study][["dataset", "model"]]
    frame = pd.read_csv(cells_path)
    frame = frame[frame.split == "test"].merge(chosen, on=["dataset", "model"])
    frame["source_group"] = frame.dataset.map(groups)
    populations = pd.read_csv(RESULTS / "without_replacement_baseline_cells.csv")
    populations = populations[populations.panel == study][
        ["dataset", "model", "srswor_b16_residual"]
    ]
    return frame.merge(populations, on=["dataset", "model"])


def main() -> None:
    confirmation = json.loads((HERE / "tier1_confirmation_config.json").read_text())["source_groups"]
    external_config = json.loads((HERE / "openml_external_cover_config.json").read_text())
    external = external_config["source_groups"]
    taskbalanced_config = json.loads((HERE / "openml_taskbalanced_cover_config.json").read_text())
    taskbalanced = taskbalanced_config["source_groups"]
    panels = {
        "strength2_confirmation": panel_frame(
            "strength2_confirmation", RESULTS / "strength2_confirmation_cells.csv", confirmation
        ),
        "strength2_openml_external": panel_frame(
            "strength2_openml_external", RESULTS / "strength2_openml_external_cells.csv", external
        ),
        "strength2_openml_taskbalanced": panel_frame(
            "strength2_openml_taskbalanced",
            RESULTS / "strength2_openml_taskbalanced_cells.csv", taskbalanced
        ),
    }
    comparators = (
        "iid16_residual", "srswor_b16_residual",
        "four_strength1_residual", "four_seed_blocks_residual",
    )
    summaries = {}
    group_outputs = []
    for panel, frame in panels.items():
        group = frame.groupby("source_group", as_index=False)[["strength2_residual", *comparators]].mean()
        group.insert(0, "panel", panel)
        group_outputs.append(group)
        records = {}
        for comparator in comparators:
            rng = np.random.default_rng(stable_seed(panel, comparator))
            ids = rng.integers(0, len(group), size=(DRAWS, len(group)))
            action = group.strength2_residual.to_numpy()[ids].mean(axis=1)
            control = group[comparator].to_numpy()[ids].mean(axis=1)
            valid = control > 1e-30
            reductions = 1 - action[valid] / control[valid]
            records[comparator] = {
                "point_reduction_equal_source": float(1 - group.strength2_residual.mean() / group[comparator].mean()),
                "source_cluster_bootstrap_95_interval": [float(x) for x in np.quantile(reductions, [0.025, 0.975])],
                "valid_bootstrap_draws": int(valid.sum()),
                "source_groups": len(group),
                "source_groups_strength2_lower": int((group.strength2_residual < group[comparator]).sum()),
            }
        summaries[panel] = records
    pd.concat(group_outputs).to_csv(RESULTS / "source_generalization_groups.csv", index=False)
    summary = {
        "status": "complete", "bootstrap_draws": DRAWS, "panels": summaries,
        "posthoc_scope_extension_panels": ["strength2_openml_taskbalanced"],
    }
    (RESULTS / "source_generalization_uncertainty_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
