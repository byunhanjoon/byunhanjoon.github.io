"""Frozen mixed-level design constructors used by final-closure analyses."""

from __future__ import annotations

import itertools
import math
from typing import Iterable

import numpy as np


MUL4 = np.asarray(
    [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]],
    dtype=np.int16,
)


def trace4(value: int) -> int:
    value = int(value)
    return value ^ int(MUL4[value, value])


def assert_levels(design: np.ndarray, cards: tuple[int, ...]) -> None:
    if design.ndim != 2 or design.shape[1] != len(cards):
        raise AssertionError("design shape/cardinality mismatch")
    for factor, card in enumerate(cards):
        if np.any(design[:, factor] < 0) or np.any(design[:, factor] >= card):
            raise AssertionError(f"factor {factor} outside 0..{card - 1}")


def assert_strength(design: np.ndarray, cards: tuple[int, ...], strength: int) -> None:
    assert_levels(design, cards)
    for order in range(1, strength + 1):
        for factors in itertools.combinations(range(len(cards)), order):
            if any(cards[factor] == 1 for factor in factors):
                continue
            counts = np.zeros(tuple(cards[factor] for factor in factors), dtype=np.int64)
            for row in design:
                counts[tuple(int(row[factor]) for factor in factors)] += 1
            if len(np.unique(counts)) != 1:
                raise AssertionError(
                    f"strength-{strength} imbalance on {factors}: {np.unique(counts)}"
                )


def randomize_levels(
    design: np.ndarray, cards: tuple[int, ...], rng: np.random.Generator
) -> np.ndarray:
    output = np.asarray(design, dtype=np.int16).copy()
    for factor, card in enumerate(cards):
        permutation = rng.permutation(card)
        output[:, factor] = permutation[output[:, factor]]
    rng.shuffle(output, axis=0)
    return output


def strength1_schema_base(cards: tuple[int, int, int]) -> np.ndarray:
    runs = math.lcm(*cards)
    design = np.stack(
        [np.arange(runs, dtype=np.int16) % card for card in cards], axis=1
    )
    assert_strength(design, cards, 1)
    return design


def strength2_schema_base(cards: tuple[int, int, int]) -> np.ndarray:
    if cards[0] != 4 or cards[1] not in (1, 4) or cards[2] not in (1, 2):
        raise ValueError(f"unsupported frozen schema cards {cards}")
    rows = []
    for u in range(4):
        for v in range(4):
            rows.append(
                (
                    u,
                    v if cards[1] == 4 else 0,
                    trace4(u ^ v) if cards[2] == 2 else 0,
                )
            )
    design = np.asarray(rows, dtype=np.int16)
    assert_strength(design, cards, 2)
    return design


def nested_blocks(
    base: np.ndarray, cards: tuple[int, ...], budget: int,
    rng: np.random.Generator,
) -> np.ndarray:
    blocks = []
    while sum(len(block) for block in blocks) < budget:
        blocks.append(randomize_levels(base, cards, rng))
    return np.concatenate(blocks, axis=0)[:budget]


def sample_schema_design(
    method: str, cards: tuple[int, int, int], budget: int,
    rng: np.random.Generator,
) -> np.ndarray:
    population = math.prod(cards)
    if method == "IID-JOINT":
        ids = rng.integers(0, population, size=budget)
        return np.stack(np.unravel_index(ids, cards), axis=1).astype(np.int16)
    if method == "SRS-JOINT":
        if budget <= population:
            ids = rng.choice(population, size=budget, replace=False)
        else:
            full = [rng.permutation(population) for _ in range(math.ceil(budget / population))]
            ids = np.concatenate(full)[:budget]
        return np.stack(np.unravel_index(ids, cards), axis=1).astype(np.int16)
    if method == "OC1-INDEPENDENT":
        return nested_blocks(strength1_schema_base(cards), cards, budget, rng)
    if method == "OC2-INDEPENDENT":
        return nested_blocks(strength2_schema_base(cards), cards, budget, rng)
    raise ValueError(method)


def trajectory_strength3(cards: tuple[int, int, int, int]) -> np.ndarray:
    """OA-128 over feature(4), category(4/1), class(2/1), seed(8).

    The first, second, and fourth coordinates form the complete product.  The
    binary class column is a trace contrast involving all three and is balanced
    conditional on every pair, yielding formal strength three.
    """

    if cards[0] != 4 or cards[1] not in (1, 4) or cards[2] not in (1, 2) or cards[3] != 8:
        raise ValueError(f"unsupported trajectory cards {cards}")
    rows = []
    for feature in range(4):
        for category_raw in range(4):
            category = category_raw if cards[1] == 4 else 0
            for seed in range(8):
                seed_gf4, seed_bit = seed % 4, seed // 4
                class_level = (
                    trace4(
                        feature
                        ^ int(MUL4[2, category_raw])
                        ^ int(MUL4[3, seed_gf4])
                    )
                    ^ seed_bit
                    if cards[2] == 2
                    else 0
                )
                rows.append((feature, category, class_level, seed))
    design = np.asarray(rows, dtype=np.int16)
    assert_strength(design, cards, 3)
    expected_unique = min(len(design), math.prod(cards))
    if len(np.unique(design, axis=0)) != expected_unique:
        raise AssertionError("trajectory design lacks maximum row uniqueness")
    return design


def finite_strength2_base(cards: tuple[int, int, int, int, int]) -> np.ndarray:
    """The repaired 16-run GF(4) design for schema×init×order factors."""

    if cards[0] != 4 or cards[1] not in (1, 4) or cards[2] not in (1, 2):
        raise ValueError(cards)
    if cards[3] not in (1, 2, 4) or cards[4] not in (1, 2, 4):
        raise ValueError(cards)
    rows = []
    for u in range(4):
        for v in range(4):
            init_gf = u ^ int(MUL4[2, v])
            order_gf = u ^ int(MUL4[3, v])
            raw = [u, v, trace4(u ^ v), init_gf, order_gf]
            # Binary collapsed stochastic factors use independent trace
            # functionals; four-level factors retain the GF(4) coordinate.
            if cards[3] == 2:
                raw[3] = trace4(init_gf)
            if cards[4] == 2:
                raw[4] = trace4(order_gf)
            rows.append(tuple(0 if cards[index] == 1 else raw[index] for index in range(5)))
    design = np.asarray(rows, dtype=np.int16)
    assert_strength(design, cards, 2)
    return design


def mechanism_design(
    method: str, cards: tuple[int, int, int, int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Return the frozen B=16 coupling-ablation design."""

    budget = 16
    output = np.empty((budget, 5), dtype=np.int16)
    for factor, card in enumerate(cards):
        output[:, factor] = rng.integers(0, card, size=budget)
    if method == "none":
        return output
    if method in {"schema", "schema_initialization", "schema_order", "all_factors"}:
        selected = [0, 1, 2]
        if method in {"schema_initialization", "all_factors"}:
            selected.append(3)
        if method in {"schema_order", "all_factors"}:
            selected.append(4)
        reduced_cards = tuple(cards[index] for index in selected)
        padded = list(reduced_cards) + [1] * (5 - len(reduced_cards))
        base = finite_strength2_base(tuple(padded))[:, : len(selected)]
        base = randomize_levels(base, reduced_cards, rng)
        output[:, selected] = base
        return output
    if method == "initialization":
        output[:, 3] = np.tile(rng.permutation(cards[3]), budget // cards[3])
        return output
    if method == "order":
        output[:, 4] = np.tile(rng.permutation(cards[4]), budget // cards[4])
        return output
    if method == "initialization_order":
        pair = np.asarray(list(itertools.product(range(cards[3]), range(cards[4]))), dtype=np.int16)
        if len(pair) != budget:
            raise AssertionError("frozen init×order design requires 4×4 levels")
        rng.shuffle(pair, axis=0)
        output[:, 3:5] = pair
        return output
    raise ValueError(method)


def rows_to_ids(rows: np.ndarray, cards: tuple[int, ...]) -> np.ndarray:
    assert_levels(rows, cards)
    return np.ravel_multi_index(np.moveaxis(rows, -1, 0), cards)
