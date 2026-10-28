"""Closed-form resolution theory for single-effect fine-mapping.

Summary-statistic model: z ~ N(c R e_j, R), where c = sqrt(n) * beta is the
non-centrality of the causal variant j and R is the LD (correlation) matrix.
A sequence channel s_i = mu * 1[i = j] + noise has within-set AUC
A_s = Phi(mu / sqrt(2)).
"""
import numpy as np
from scipy import stats, integrate
from scipy.special import logsumexp

Phi, phi, Phiinv = stats.norm.cdf, stats.norm.pdf, stats.norm.ppf

__all__ = [
    "pairwise_error", "pairwise_correct_abs", "p_correct_equicorr",
    "p_correct_equicorr_abs", "error_bounds", "mu_from_auc", "auc_from_mu",
    "combined_snr", "neff_ratio", "expected_max_pip", "kappa_transform",
    "c_required", "n_required", "multiancestry_snr", "c_debiased",
]


def pairwise_error(c, r):
    """Theorem 1: Bayes error for two candidates with correlation r (known sign)."""
    return Phi(-np.abs(c) * np.sqrt((1 - np.abs(r)) / 2))


def pairwise_correct_abs(c, r):
    """Probability that the |z| rule ranks the causal variant above one partner.

    With a = c sqrt((1-r)/2) and b = c sqrt((1+r)/2) the |z| rule is correct with
    probability Phi(a)Phi(b) + Phi(-a)Phi(-b).
    """
    r = np.clip(np.abs(r), 0, 1)
    a = np.abs(c) * np.sqrt((1 - r) / 2)
    b = np.abs(c) * np.sqrt((1 + r) / 2)
    return Phi(a) * Phi(b) + Phi(-a) * Phi(-b)


def p_correct_equicorr(k, snr):
    """Theorem 2: top-1 accuracy for k exchangeable candidates.

    snr is c * sqrt(1 - r) for association data alone, or the combined value
    from `combined_snr` when a sequence channel is added (Theorem 3).
    """
    if k <= 1:
        return 1.0
    f = lambda x: phi(x - snr) * Phi(x) ** (k - 1)
    v, _ = integrate.quad(f, -10 + min(0.0, snr), 10 + snr, limit=200)
    return float(v)


_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(80)
_GH_W = _GH_W / _GH_W.sum()


def p_correct_equicorr_abs(k, c, r):
    """Sign-agnostic top-1 accuracy of argmax |z| (single-effect regression).

    Evaluated by two-dimensional Gauss-Hermite quadrature over the shared LD
    component W and the causal variant's own noise U1.
    """
    if k <= 1:
        return 1.0
    if r >= 1:
        return 1.0 / k
    s = np.sqrt(1 - r)
    W = _GH_X[:, None]
    U1 = _GH_X[None, :]
    wts = _GH_W[:, None] * _GH_W[None, :]
    A = np.abs(c + np.sqrt(r) * W + s * U1)
    m = c * r + np.sqrt(r) * W
    pj = Phi((A - m) / s) - Phi((-A - m) / s)
    return float((wts * pj ** (k - 1)).sum())


def error_bounds(c, R, j):
    """Theorem 2 bounds for a general LD matrix: max_l P_l <= P_err <= sum_l P_l."""
    rs = np.delete(np.asarray(R)[j], j)
    pe = Phi(-np.abs(c) * np.sqrt(np.clip(1 - np.abs(rs), 0, None) / 2))
    if len(pe) == 0:
        return 0.0, 0.0
    return float(pe.max()), float(min(1.0, pe.sum()))


def mu_from_auc(auc):
    """Binormal separation mu = sqrt(2) * Phi^{-1}(AUC)."""
    return np.sqrt(2) * Phiinv(auc)


def auc_from_mu(mu):
    return Phi(mu / np.sqrt(2))


def combined_snr(a, b, rho=0.0, optimal=True):
    """Theorem 3: combine two Gaussian evidence channels.

    a, b are per-channel separations (for association data a = c sqrt(1 - r),
    for a sequence model b = mu) and rho is the correlation of their noise.
    optimal=True gives the Mahalanobis combination
        sqrt((a^2 + b^2 - 2 rho a b) / (1 - rho^2)),
    optimal=False the unit-weight sum (a^2 + b^2) / sqrt(a^2 + b^2 + 2 rho a b).
    With rho = 0 both reduce to the quadrature law sqrt(a^2 + b^2).
    """
    if optimal:
        return np.sqrt((a * a + b * b - 2 * rho * a * b) / (1 - rho * rho))
    return (a * a + b * b) / np.sqrt(a * a + b * b + 2 * rho * a * b)


def neff_ratio(auc, c, r):
    """Corollary 1: n_eff / n = 1 + mu^2 / (c^2 (1 - |r|))."""
    return 1 + mu_from_auc(auc) ** 2 / (c ** 2 * (1 - np.abs(r)))


def expected_max_pip(z_lead, R, n_draws=300, rng=None, prior_var=None):
    """Monte Carlo expected maximum PIP and top-1 accuracy of a single-effect
    regression, with variant 0 causal, c = |z_lead| and LD matrix R."""
    rng = np.random.default_rng(rng)
    R = np.asarray(R, float)
    m = len(R)
    c = abs(z_lead)
    W = max(c * c, 1.0) if prior_var is None else prior_var
    L = np.linalg.cholesky(R + 1e-3 * np.eye(m))
    z = c * R[:, 0][None, :] + (L @ rng.standard_normal((m, n_draws))).T
    lbf = 0.5 * z * z * W / (1 + W)
    a = np.exp(lbf - logsumexp(lbf, 1, keepdims=True))
    return float(a.max(1).mean()), float(np.mean(np.argmax(a, 1) == 0))


def kappa_transform(R, kappa):
    """Effective-resolution transform R -> 11^T - kappa (11^T - R), projected to
    the nearest correlation matrix (eigenvalue clipping and rescaling)."""
    R = np.asarray(R, float)
    Rk = 1 - kappa * (1 - R)
    w, V = np.linalg.eigh((Rk + Rk.T) / 2)
    Rk = (V * np.clip(w, 1e-6, None)) @ V.T
    d = np.sqrt(np.diag(Rk))
    return Rk / np.outer(d, d)


def c_required(r, target=0.9, k=2, sign_agnostic=True):
    """Non-centrality c needed for top-1 accuracy `target` among k candidates
    with correlation r (equicorrelated). Infinite when |r| = 1."""
    from scipy.optimize import brentq
    r = abs(float(r))
    if r >= 1:
        return np.inf
    if target <= 1.0 / k:
        return 0.0
    if sign_agnostic:
        f = (lambda c: pairwise_correct_abs(c, r) - target) if k == 2 else \
            (lambda c: p_correct_equicorr_abs(k, c, r) - target)
    else:
        f = lambda c: p_correct_equicorr(k, c * np.sqrt(1 - r)) - target
    hi = 1.0
    while f(hi) < 0:
        hi *= 2
        if hi > 1e7:
            return np.inf
    return float(brentq(f, 1e-9, hi))


def n_required(z, n, r, target=0.9, k=2):
    """Sample size at which a signal with observed |z| at sample size n reaches
    top-1 accuracy `target` against a partner with correlation r (c scales as sqrt(n))."""
    cs = c_required(r, target, k)
    return np.inf if not np.isfinite(cs) else n * (cs / max(abs(z), 1e-12)) ** 2


def multiancestry_snr(c, r, mu=0.0):
    """Separation for one candidate pair when independent studies in several
    populations are combined: sqrt(sum_p c_p^2 (1 - |r_p|) + mu^2)."""
    c, r = np.asarray(c, float), np.abs(np.asarray(r, float))
    return float(np.sqrt(np.sum(c ** 2 * (1 - r)) + mu ** 2))


def c_debiased(z_lead, R, n_draws=400, rng=None, tol=0.02):
    """Winner's-curse-corrected non-centrality: the c at which the expected maximum |z| among the
    candidates (variant 0 causal, LD matrix R) equals the observed |z_lead| (method of moments)."""
    rng = np.random.default_rng(rng)
    R = np.asarray(R, float)
    m = len(R)
    L = np.linalg.cholesky(R + 1e-3 * np.eye(m))
    E = (L @ rng.standard_normal((m, n_draws))).T            # common random numbers
    target = abs(z_lead)
    f = lambda c: np.abs(c * R[:, 0][None, :] + E).max(1).mean() - target
    lo, hi = 0.0, target
    if f(lo) >= 0:
        return 0.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
