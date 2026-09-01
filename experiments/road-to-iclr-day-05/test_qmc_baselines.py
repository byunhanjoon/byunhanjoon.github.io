import numpy as np

import analyze_qmc_baselines as qmc


def test_discrete_qmc_families_are_budget16_and_marginally_balanced():
    old = qmc.DESIGNS
    qmc.DESIGNS = 8
    try:
        for kind in ("sobol16", "lhs16"):
            family = qmc.design_family(kind)
            assert family.shape == (8, 16, 4)
            for design in family:
                for column, levels in enumerate(qmc.SHAPE):
                    np.testing.assert_array_equal(np.bincount(design[:, column], minlength=levels), 16 // levels)
    finally:
        qmc.DESIGNS = old
