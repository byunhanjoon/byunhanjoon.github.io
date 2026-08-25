import numpy as np

from experiments.day3.core import real_fourier_basis


def test_full_real_fourier_is_orthonormal_nonconstant_basis():
    for k in (7, 24):
        basis = real_fourier_basis(k)
        assert basis.shape == (k, k - 1)
        assert np.allclose(basis.T @ basis, np.eye(k - 1), atol=1e-12)
        assert np.allclose(basis.sum(axis=0), 0.0, atol=1e-12)


def test_integer_phase_is_an_orthogonal_rotation():
    base = real_fourier_basis(24, 0)
    shifted = real_fourier_basis(24, 5)
    transform = np.linalg.lstsq(base, shifted, rcond=None)[0]
    assert np.allclose(base @ transform, shifted, atol=1e-11)
    assert np.allclose(transform.T @ transform, np.eye(23), atol=1e-11)
