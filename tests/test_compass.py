"""Unit tests. Run with: python -m pytest -q"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import compass as cp  # noqa: E402


def test_equicorr_reduces_to_pairwise():
    # k = 2: P_correct = Phi(c sqrt((1-r)/2)) = 1 - pairwise_error
    for c, r in [(2, 0.5), (5, 0.9), (8, 0.99)]:
        a = c * np.sqrt(1 - r)
        assert abs(cp.p_correct_equicorr(2, a) - (1 - cp.pairwise_error(c, r))) < 1e-6


def test_perfect_ld_is_chance():
    assert abs(cp.p_correct_equicorr(5, 0.0) - 0.2) < 1e-6
    assert cp.p_correct_equicorr_abs(4, 6.0, 1.0) == 0.25


def test_quadrature_law():
    assert abs(cp.combined_snr(3.0, 4.0) - 5.0) < 1e-12
    assert abs(cp.combined_snr(3.0, 4.0, 0.0, optimal=False) - 5.0) < 1e-12
    # correlated channels: optimal combination is never worse than the naive sum
    for rho in (0.2, 0.5, 0.8):
        assert cp.combined_snr(2.0, 1.0, rho) >= cp.combined_snr(2.0, 1.0, rho, optimal=False) - 1e-12


def test_auc_mu_roundtrip():
    for auc in (0.55, 0.7, 0.9):
        assert abs(cp.auc_from_mu(cp.mu_from_auc(auc)) - auc) < 1e-12


def test_monte_carlo_matches_theorem2():
    rng = np.random.default_rng(0)
    k, r, c, n = 5, 0.9, 5.0, 40000
    W = rng.standard_normal((n, 1))
    U = rng.standard_normal((n, k))
    mean = np.full(k, c * r)
    mean[0] = c
    z = mean + np.sqrt(r) * W + np.sqrt(1 - r) * U
    sim = np.mean(np.argmax(np.abs(z), 1) == 0)
    assert abs(sim - cp.p_correct_equicorr_abs(k, c, r)) < 0.01


def test_bounds_bracket_error():
    R = np.array([[1, .9, .8], [.9, 1, .85], [.8, .85, 1]])
    lo, hi = cp.error_bounds(5.0, R, 0)
    assert 0 <= lo <= hi <= 1


def test_srfm_recovers_temperature():
    rng = np.random.default_rng(1)
    mu, G, k = 1.0, 3000, 8
    pips, scores, groups, truth = [], [], [], []
    for g in range(G):
        j = rng.integers(k)
        s = rng.standard_normal(k)
        s[j] += mu
        z = rng.standard_normal(k)
        z[j] += 1.5
        lbf = 1.5 * z                     # exact log-likelihood ratio of "i is causal"
        p = np.exp(lbf - lbf.max())
        p /= p.sum()
        pips += list(p); scores += list(s); groups += [g] * k; truth += list(np.arange(k) == j)
    lam, _ = cp.fit_lambda(np.array(pips), np.array(scores), np.array(groups))
    assert abs(lam - mu) < 0.15
    q = cp.srfm_posterior(pips, scores, groups, lam)
    assert np.allclose(np.bincount(groups, weights=q), 1)


def test_conformal_coverage():
    rng = np.random.default_rng(2)
    G, k = 2000, 10
    q, truth, groups, fold = [], [], [], []
    for g in range(G):
        j = rng.integers(k)
        a = rng.standard_normal(k)
        a[j] += 1.0
        p = np.exp(a) / np.exp(a).sum()
        q += list(p); truth += list(np.arange(k) == j); groups += [g] * k; fold += [g % 2] * k
    for alpha in (0.2, 0.1):
        cov, size = cp.evaluate_cross_fit(q, truth, groups, fold, alpha)
        assert cov >= 1 - alpha - 0.025
        assert 1 <= size <= k


def test_metrics():
    s = np.array([3, 1, 2, 0, 5, 4])
    t = np.array([1, 0, 0, 0, 1, 0], bool)
    g = np.array([0, 0, 0, 1, 1, 1])
    assert cp.within_set_auc(s, t, g) == 1.0
    assert cp.top1_accuracy(s, t, g) == 1.0
    assert list(cp.credible_set([0.5, 0.3, 0.2], 0.75)) == [0, 1]


def test_kappa_transform():
    R = np.array([[1, .9], [.9, 1]])
    Rk = cp.kappa_transform(R, 0.2)
    assert abs(Rk[0, 1] - 0.98) < 1e-3
    assert np.allclose(np.diag(Rk), 1)


def test_c_required_and_multiancestry():
    r = np.sqrt(0.99)
    c = cp.c_required(r, 0.9)
    assert abs(cp.pairwise_correct_abs(c, r) - 0.9) < 1e-6
    assert abs(c - 25.6) < 0.1
    assert np.isinf(cp.c_required(1.0))
    # two populations with the same c and r equal one population with twice the sample
    assert abs(cp.multiancestry_snr([5, 5], [0.9, 0.9]) - np.sqrt(2) * 5 * np.sqrt(0.1)) < 1e-12
