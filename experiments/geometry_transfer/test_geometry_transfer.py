import numpy as np

from geometry_transfer import decompose, empirical_gain, rbf_operator


def test_expanded_identity_and_full_covariance():
    rng = np.random.default_rng(1)
    mu_u, mu_t = rng.normal(size=5), rng.normal(size=7)
    a = rng.normal(size=(5, 7))
    c = rng.normal(size=(7, 7)); sigma = c @ c.T / 100
    q = np.arange(1, 6, dtype=float); q /= q.sum()
    d = decompose(mu_u, mu_t, a, sigma, q)
    expanded = 2 * mu_u @ (q[:, None] * a) @ mu_t - mu_t @ a.T @ (q[:, None] * a) @ mu_t - np.trace(np.diag(q) @ a @ sigma @ a.T)
    assert np.allclose(d.delta, expanded)


def test_irreducible_noise_cancels_monte_carlo():
    rng = np.random.default_rng(2)
    mu_u = np.array([1.0, -0.5]); mu_t = np.array([.8, -.2])
    a = np.array([[1., 0.], [.25, .75]]); sigma = np.array([.04, .09])
    truth = decompose(mu_u, mu_t, a, sigma).delta
    for test_sd in (0.0, 1.0, 10.0):
        gains = []
        for _ in range(30000):
            mh = mu_t + rng.normal(0, np.sqrt(sigma))
            y = mu_u + rng.normal(0, test_sd, 2)
            gains.append(np.mean(y*y - (y-a@mh)**2))
        assert abs(np.mean(gains) - truth) < .08


def test_realized_oracle_arithmetic():
    residual = np.array([1., 3., -2., 0.]); state = np.array([0, 0, 1, 1]); g = np.array([.5, -.25])
    means = np.array([2., -1.])
    assert np.allclose(empirical_gain(residual, state, g), np.mean(2*means*g-g*g))


def test_spectral_corollary():
    rng = np.random.default_rng(3); n = 8
    z = rng.normal(size=(n, n)); v, _ = np.linalg.qr(z)
    h = np.linspace(0, 1, n); coeff = rng.normal(size=n); sigma2 = .2
    H = v @ np.diag(h) @ v.T; mu = v @ coeff
    matrix = decompose(mu, mu, H, np.full(n, sigma2), np.ones(n)).delta
    spectral = np.sum((2*h-h*h)*coeff*coeff-sigma2*h*h)
    assert np.allclose(matrix, spectral)


def test_mpe_factorization():
    rng = np.random.default_rng(4)
    w = rng.normal(size=(13, 17)); v = rng.normal(size=(17, 5)); stem = rng.normal(size=(5, 9))
    assert np.max(np.abs((w@v)@stem-w@(v@stem))) < 1e-12


def test_rbf_rows_sum_to_one():
    x = np.arange(10); d = np.abs(x[:,None]-x[None,:]).astype(float)
    a = rbf_operator(d, np.arange(0,10,2), np.arange(1,10,2), 2.)
    assert np.allclose(a.sum(axis=1), 1)
