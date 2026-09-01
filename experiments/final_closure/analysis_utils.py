"""Statistical and fANOVA helpers shared by final-closure analyzers."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def squared_residual(prediction: np.ndarray, reference: np.ndarray) -> float:
    return float(np.mean(np.sum((prediction - reference) ** 2, axis=-1)))


def full_factor_components(
    values: np.ndarray, cards: tuple[int, ...]
) -> dict[tuple[int, ...], float]:
    """Exact functional-ANOVA component energies on a balanced full product."""

    product = int(np.prod(cards))
    if values.shape[0] != product:
        raise ValueError(f"expected full product {product}, got {values.shape[0]}")
    field_shape = values.shape[1:]
    tensor = np.asarray(values, dtype=np.float64).reshape((*cards, *field_shape))
    factor_count = len(cards)
    components: dict[tuple[int, ...], np.ndarray] = {}
    energies: dict[tuple[int, ...], float] = {}
    for order in range(1, factor_count + 1):
        for subset in itertools.combinations(range(factor_count), order):
            complement = tuple(index for index in range(factor_count) if index not in subset)
            conditional = tensor.mean(axis=complement, keepdims=True) if complement else tensor.copy()
            component = conditional - tensor.mean(axis=tuple(range(factor_count)), keepdims=True)
            for lower_order in range(1, order):
                for lower in itertools.combinations(subset, lower_order):
                    component = component - components[lower]
            components[subset] = component
            energies[subset] = float(np.mean(component**2))
    return energies


def oa_factor_components(
    values: np.ndarray, design: np.ndarray, cards: tuple[int, ...], maximum_order: int = 3
) -> dict[tuple[int, ...], float]:
    """Orthogonal-contrast energies on a strength-t fractional design."""

    data = np.asarray(values, dtype=np.float64)
    grand = data.mean(axis=0)
    row_components: dict[tuple[int, ...], np.ndarray] = {}
    energies: dict[tuple[int, ...], float] = {}
    for order in range(1, maximum_order + 1):
        for subset in itertools.combinations(range(len(cards)), order):
            if any(cards[index] == 1 for index in subset):
                continue
            conditional = np.empty_like(data)
            groups: dict[tuple[int, ...], list[int]] = {}
            for row_index, row in enumerate(design):
                key = tuple(int(row[index]) for index in subset)
                groups.setdefault(key, []).append(row_index)
            for members in groups.values():
                conditional[members] = data[members].mean(axis=0)
            component = conditional - grand
            for lower_order in range(1, order):
                for lower in itertools.combinations(subset, lower_order):
                    if lower in row_components:
                        component -= row_components[lower]
            row_components[subset] = component
            energies[subset] = float(np.mean(component**2))
    return energies


def summarize_orders(
    energies: dict[tuple[int, ...], float], total: float | None = None
) -> dict[str, float]:
    by_order: dict[int, float] = {}
    for subset, energy in energies.items():
        by_order[len(subset)] = by_order.get(len(subset), 0.0) + float(energy)
    reconstructed = sum(by_order.values())
    actual_total = reconstructed if total is None else float(total)
    if total is not None and actual_total > reconstructed:
        by_order[max(by_order.keys(), default=0) + 1] = actual_total - reconstructed
    denominator = actual_total if actual_total > 0 else 1.0
    main = by_order.get(1, 0.0)
    pair = by_order.get(2, 0.0)
    triple = by_order.get(3, 0.0)
    higher = max(actual_total - main - pair - triple, 0.0)
    effective = (
        sum(order * value for order, value in by_order.items()) / actual_total
        if actual_total > 0 else 0.0
    )
    return {
        "total_nuisance_variance": actual_total,
        "main_fraction": main / denominator,
        "pair_fraction": pair / denominator,
        "main_pair_fraction": (main + pair) / denominator,
        "triple_fraction": triple / denominator,
        "higher_fraction": higher / denominator,
        "effective_interaction_order": effective,
        "fanova_reconstruction_error": actual_total - reconstructed,
    }


def dataset_cluster_bootstrap(
    frame: pd.DataFrame, value: str, *, draws: int, seed: int,
    cluster: str = "dataset",
) -> tuple[float, float]:
    grouped = frame.groupby(cluster, sort=True)[value].mean().to_numpy(dtype=float)
    if not len(grouped):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = grouped[rng.integers(0, len(grouped), size=(draws, len(grouped)))].mean(axis=1)
    return tuple(float(value) for value in np.quantile(samples, [0.025, 0.975]))


def markdown_text(frame: pd.DataFrame) -> str:
    """Render a compact GFM table without pandas' optional tabulate dependency."""

    def render(value: object) -> str:
        if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
            return ""
        if isinstance(value, (float, np.floating)):
            text = f"{float(value):.10g}"
        else:
            text = str(value)
        return text.replace("|", "\\|").replace("\r\n", "<br>").replace("\n", "<br>")

    header = "| " + " | ".join(render(column) for column in frame.columns) + " |"
    separator = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def markdown_table(frame: pd.DataFrame, path: Path) -> None:
    """Write a compact GFM table without pandas' optional tabulate dependency."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_text(frame) + "\n", encoding="utf-8")


def write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
