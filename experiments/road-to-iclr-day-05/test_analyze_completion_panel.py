import numpy as np
import pandas as pd

from analyze_completion_panel import (
    assert_strength, packed_families, strength1_base, strength2_base,
    strength2_cover_family, strength3_base,
    source_comparisons,
)


def test_completion_design_strengths():
    for cards in ((4, 4, 2, 2, 2), (4, 4, 1, 2, 2), (4, 1, 2, 2, 2), (4, 1, 1, 2, 2)):
        assert_strength(strength1_base(cards), cards, 1)
        assert_strength(strength2_base(cards), cards, 2)
        assert_strength(strength3_base(cards), cards, 3)


def test_strength2_has_sixteen_rows_and_strength3_sixty_four():
    for cards in ((4, 4, 2, 2, 2), (4, 4, 1, 2, 2), (4, 1, 2, 2, 2), (4, 1, 1, 2, 2)):
        second = strength2_base(cards)
        third = strength3_base(cards)
        assert second.shape == (16, 5)
        assert third.shape == (64, 5)
        second_ids = np.ravel_multi_index(second.T, cards)
        third_ids = np.ravel_multi_index(third.T, cards)
        assert len(np.unique(second_ids)) == min(16, np.prod(cards))
        assert len(np.unique(third_ids)) == min(64, np.prod(cards))
        assert np.unique(np.bincount(third_ids, minlength=np.prod(cards))).size <= 2


def test_completion_disjoint_cover_packing_and_closure():
    cards = (4, 4, 2, 2, 2)
    family = strength2_cover_family(cards)
    assert family.shape[1] == 16
    designs = packed_families(cards, 81)
    assert designs["disjoint_pair32"].shape == (512, 32)
    assert designs["disjoint_pack64"].shape == (512, 64)
    for row in designs["disjoint_pair32"][:8]:
        assert len(np.unique(row)) == 32
    for row in designs["disjoint_pack64"][:8]:
        assert len(np.unique(row)) == 64
    closed = packed_families((4, 1, 1, 2, 2), 82)
    assert np.array_equal(np.unique(closed["disjoint_pair32"][0]), np.arange(16))
    assert np.array_equal(np.unique(closed["disjoint_pack64"][0]), np.arange(16))


def test_source_comparisons_clusters_before_bootstrap():
    rows = []
    for dataset, scale in (("a", 1.0), ("b", 2.0)):
        for repeat in range(3):
            rows.append({
                "dataset": dataset, "total_nuisance_variance": 1.0,
                "strength2_16_residual_mean": 0.5 * scale,
                "iid16_residual_mean": 1.0 * scale,
                "srswor16_residual_mean": 1.0 * scale,
                "lhs16_residual_mean": 1.0 * scale,
                "sobol16_residual_mean": 1.0 * scale,
                "strength1_16_residual_mean": 1.0 * scale,
            })
    cells, summary = source_comparisons(pd.DataFrame(rows))
    assert len(cells) == 10
    assert summary["iid16"]["sources"] == 2
    assert np.isclose(summary["iid16"]["equal_source_fractional_reduction"], 0.5)
