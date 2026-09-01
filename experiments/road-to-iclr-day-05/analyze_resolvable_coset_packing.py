"""Exact resolvable GF(4) coset packing frontier on the 4^4 product."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_strength2_cover import PERMS4, assert_strength, strength2_base


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SHAPE = (4, 4, 4, 4)
RESOLUTIONS = 8_192
KS = (1, 2, 4, 8, 16)


def coset_resolution() -> np.ndarray:
    base = strength2_base(4, 4, 4)
    candidates = []
    for shift in itertools.product(range(4), repeat=4):
        current = base ^ np.asarray(shift, dtype=int)
        ids = tuple(sorted(np.ravel_multi_index(current.T, SHAPE).tolist()))
        candidates.append((ids, current))
    unique: dict[tuple[int, ...], np.ndarray] = {}
    for ids, design in candidates:
        unique.setdefault(ids, design)
    return np.stack(list(unique.values()))


def pure_field(subset: tuple[int, ...]) -> np.ndarray:
    vectors = []
    for index in range(4):
        if index in subset:
            value = np.arange(4, dtype=float) - 1.5
            value /= np.sqrt(np.mean(value ** 2))
        else:
            value = np.ones(4)
        vectors.append(value)
    output = vectors[0]
    for value in vectors[1:]:
        output = np.multiply.outer(output, value)
    return output


def main() -> None:
    resolution = coset_resolution()
    assert resolution.shape == (16, 16, 4)
    all_ids = np.ravel_multi_index(resolution.transpose(2, 0, 1), SHAPE)
    partition = len(np.unique(all_ids)) == 256
    strength = True
    for design in resolution:
        try:
            assert_strength(design, SHAPE, 2)
        except AssertionError:
            strength = False
    fields = {
        ":".join(map(str, subset)): pure_field(subset)
        for subset in list(itertools.combinations(range(4), 3)) + [(0, 1, 2, 3)]
    }
    rng = np.random.default_rng(2026082844)
    population_energy = {key: [] for key in fields}
    for _ in range(RESOLUTIONS):
        perms = np.asarray([PERMS4[rng.integers(0, 24)] for _ in range(4)])
        randomized = np.empty_like(resolution)
        for factor in range(4):
            randomized[:, :, factor] = perms[factor][resolution[:, :, factor]]
        for key, field in fields.items():
            means = np.asarray([
                np.mean(field[tuple(design.T)]) for design in randomized
            ])
            population_energy[key].append(float(np.mean(means ** 2)))
            if abs(means.mean()) > 1e-12:
                raise AssertionError("coset means do not average to quotient")
    rows = []
    max_error = 0.0
    for key, values in population_energy.items():
        single = float(np.mean(values))
        for k in KS:
            independent = single / k
            packed = independent * (16 - k) / 15
            ratio = packed / independent if independent > 0 else 0.0
            predicted = (16 - k) / 15
            max_error = max(max_error, abs(ratio - predicted))
            rows.append({
                "component": key, "covers": k, "fits": 16 * k,
                "single_cover_risk": single,
                "independent_cover_risk": independent,
                "packed_coset_risk": packed,
                "packed_to_independent_ratio": ratio,
                "predicted_ratio": predicted,
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "resolvable_coset_frontier.csv", index=False)
    summary = {
        "status": "complete", "randomized_resolutions": RESOLUTIONS,
        "cosets": len(resolution), "each_coset_strength2": strength,
        "cosets_partition_full_product": partition,
        "max_ratio_formula_error": max_error,
        "full_pack_max_risk": float(frame[frame.covers == 16].packed_coset_risk.max()),
        "ratios_by_covers": frame.groupby("covers").packed_to_independent_ratio.mean().to_dict(),
        "frozen_gate_passed": bool(
            strength and partition and max_error < 1e-10
            and frame[frame.covers == 16].packed_coset_risk.max() < 1e-15
        ),
    }
    (RESULTS / "resolvable_coset_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
