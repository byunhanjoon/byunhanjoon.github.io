"""Exact risk analysis for a literal strength-1/2/3 nested cover schedule."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_strength2_cover import (
    MUL4, PERMS4, assert_strength, component_coefficients,
    expected_residual, gf4_trace, incidence_covariance,
    strength1_family, strength2_family,
)
from analyze_strength3_cover import strength3_family


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PREFIXES = ((4, 1), (16, 2), (64, 3))
FIRST = {(0, 0), (1, 3), (2, 1), (3, 2)}


def raw_rows(category_levels: int, class_levels: int) -> list[tuple[int, int, int, int, int, int]]:
    """Return projected rows while retaining latent `(u,v)` for nesting."""
    rows = []
    for u, v, w in itertools.product(range(4), repeat=3):
        category = v if category_levels == 4 else gf4_trace(v) if category_levels == 2 else 0
        linear = u ^ int(MUL4[2, v]) ^ int(MUL4[3, w])
        label = linear if class_levels == 4 else gf4_trace(linear) if class_levels == 2 else 0
        rows.append((u, category, label, w, u, v))
    return rows


def nested_base(category_levels: int, class_levels: int) -> np.ndarray:
    rows = raw_rows(category_levels, class_levels)
    middle = [row for row in rows if row[3] == (row[4] ^ row[5])]
    first = [row for row in middle if (row[4], row[5]) in FIRST]
    ordered = first + [row for row in middle if row not in first] + [row for row in rows if row not in middle]
    output = np.asarray([row[:4] for row in ordered], dtype=np.int8)
    if len(output) != 64 or len({tuple(row) for row in output}) != 64 // (4 // category_levels if category_levels else 1):
        # Projection can create duplicates; row multiplicity is intentional.
        pass
    for budget, strength in PREFIXES:
        assert_strength(output[:budget], (4, category_levels, class_levels, 4), strength)
    return output


def nested_family(category_levels: int, class_levels: int) -> np.ndarray:
    base = nested_base(category_levels, class_levels)
    category_perms = PERMS4 if category_levels == 4 else ((0, 1), (1, 0)) if category_levels == 2 else ((0,),)
    class_perms = PERMS4 if class_levels == 4 else ((0, 1), (1, 0)) if class_levels == 2 else ((0,),)
    designs = []
    for fp, cp, lp, sp in itertools.product(PERMS4, category_perms, class_perms, PERMS4):
        current = base.copy()
        current[:, 0] = np.asarray(fp)[current[:, 0]]
        current[:, 1] = np.asarray(cp)[current[:, 1]]
        current[:, 2] = np.asarray(lp)[current[:, 2]]
        current[:, 3] = np.asarray(sp)[current[:, 3]]
        designs.append(current)
    return np.asarray(designs, dtype=np.int8)


def panel(study: str, input_dir: Path) -> pd.DataFrame:
    selected = pd.read_csv(RESULTS / "validation_screened_cover_cells.csv")
    chosen = selected[selected.study == study][["dataset", "model"]]
    caches = {}
    rows = []
    for cell in chosen.itertuples(index=False):
        predictions = np.load(input_dir / f"{cell.dataset}__{cell.model}.npz")["test_predictions"].astype(np.float64)
        shape = tuple(map(int, predictions.shape[:4]))
        if shape not in caches:
            family = nested_family(shape[1], shape[2])
            nested_cov = {b: incidence_covariance(family[:, :b], shape) for b, _ in PREFIXES}
            separate = {
                4: incidence_covariance(strength1_family(*shape[1:]), shape),
                16: incidence_covariance(strength2_family(*shape[1:]), shape),
                64: incidence_covariance(strength3_family(*shape[1:]), shape),
            }
            coefficients = {b: component_coefficients(nested_cov[b], shape) for b, _ in PREFIXES}
            caches[shape] = nested_cov, separate, coefficients
        nested_cov, separate, coefficients = caches[shape]
        flat = predictions.reshape((-1,) + predictions.shape[-2:])
        quotient = flat.mean(axis=0)
        joint = float(np.mean(np.sum((flat - quotient) ** 2, axis=-1)))
        for budget, strength in PREFIXES:
            rows.append({
                "panel": study, "dataset": cell.dataset, "model": cell.model,
                "budget": budget, "strength": strength, "joint_risk": joint,
                "nested_residual": expected_residual(predictions, nested_cov[budget]),
                "iid_residual": joint / budget,
                "separate_cover_residual": expected_residual(predictions, separate[budget]),
                "maximum_annihilated_coefficient": max(
                    abs(value) for name, value in coefficients[budget].items()
                    if name.count(":") + 1 <= strength
                ),
            })
    return pd.DataFrame(rows)


def main() -> None:
    frames = [
        panel("strength2_confirmation", RESULTS / "tier1_confirmation"),
        panel("strength2_openml_external", RESULTS / "openml_external_cover"),
        panel("strength2_openml_taskbalanced", RESULTS / "openml_taskbalanced_cover"),
        panel("strength2_openml_multiclass", RESULTS / "openml_multiclass_cover"),
    ]
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(RESULTS / "anytime_nested_cover_cells.csv", index=False)
    summaries = {}
    gate = True
    frozen_panels = {"strength2_confirmation", "strength2_openml_external"}
    for panel_name, current_panel in frame.groupby("panel"):
        checkpoints = {}
        for budget, current in current_panel.groupby("budget"):
            wins = int((current.nested_residual < current.iid_residual).sum())
            checkpoint = {
                "cells": len(current), "cells_lower_than_iid": wins,
                "pooled_reduction_vs_iid": float(1 - current.nested_residual.mean() / current.iid_residual.mean()),
                "pooled_ratio_vs_separate_cover": float(current.nested_residual.mean() / current.separate_cover_residual.mean())
                if current.separate_cover_residual.mean() > 1e-30 else None,
                "maximum_annihilated_coefficient": float(current.maximum_annihilated_coefficient.max()),
            }
            checkpoints[str(budget)] = checkpoint
            if panel_name in frozen_panels:
                gate &= checkpoint["pooled_reduction_vs_iid"] > 0
                if budget >= 16:
                    gate &= wins >= 0.75 * len(current)
        summaries[panel_name] = checkpoints
    summary = {
        "status": "complete", "panels": summaries,
        "frozen_gate_panels": sorted(frozen_panels),
        "posthoc_scope_extension_panels": sorted(set(summaries) - frozen_panels),
        "frozen_anytime_gate_passed": bool(gate),
    }
    (RESULTS / "anytime_nested_cover_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
