import numpy as np

from analyze_interaction_phase_diagram import SHAPE, pure_component


def test_pure_components_are_centered_and_unit_energy():
    for subset in ((0,), (1, 2), (0, 2, 3), (0, 1, 2, 3)):
        tensor = pure_component(subset)
        assert tensor.shape == SHAPE
        assert abs(tensor.mean()) < 1e-12
        assert np.isclose(np.mean(tensor**2), 1.0)
        for axis in subset:
            assert np.allclose(tensor.mean(axis=axis), 0.0)
