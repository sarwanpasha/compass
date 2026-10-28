"""Sum of single effects regression on summary statistics (SuSiE-RSS, IBSS algorithm).

A compact NumPy implementation with per-effect prior variance estimated by
maximum likelihood, fixed residual variance 1 (z-score input) and purity-filtered
95% credible sets. It follows the sufficient-statistic form of Wang et al. (2020)
and Zou et al. (2022): XtX = (n - 1) R, Xty = sqrt(n - 1) z, y'y = n - 1.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

__all__ = ["susie_rss"]


def _ser_lbf(b, s2, V):
    if V <= 0:
        return np.zeros_like(b)
    return 0.5 * np.log(s2 / (V + s2)) + 0.5 * b * b * V / (s2 * (V + s2))


def susie_rss(z, R, n, L=10, max_iter=100, tol=1e-4, coverage=0.95, min_abs_corr=0.5,
              estimate_prior_variance=True, prior_variance=0.2):
    z = np.asarray(z, float)
    z = z * np.sqrt((n - 1.0) / (z * z + n - 2.0))      # PVE adjustment used by susieR::susie_rss
    R = np.asarray(R, float)
    p = len(z)
    d = np.full(p, n - 1.0)                       # diag(XtX)
    XtX = (n - 1.0) * R
    Xty = np.sqrt(n - 1.0) * z
    alpha = np.full((L, p), 1.0 / p)
    mu = np.zeros((L, p))
    V = np.full(L, prior_variance)
    XtXb = np.zeros(p)
    for it in range(max_iter):
        old = alpha.copy()
        for l in range(L):
            bl = alpha[l] * mu[l]
            XtXb -= XtX @ bl
            r = Xty - XtXb
            bhat, s2 = r / d, 1.0 / d
            if estimate_prior_variance:
                # log marginal likelihood ratio of the SER against the null, with uniform prior weights
                f = lambda lv: -(logsumexp(_ser_lbf(bhat, s2, np.exp(lv))) - np.log(p))
                res = minimize_scalar(f, bounds=(-20, 5), method="bounded")
                V[l] = np.exp(res.x) if -res.fun > 0 else 0.0
            lbf = _ser_lbf(bhat, s2, V[l])
            alpha[l] = np.exp(lbf - logsumexp(lbf))
            if V[l] > 0:
                pv = 1.0 / (1.0 / V[l] + d)
                mu[l] = pv * r
            else:
                mu[l] = 0.0
            XtXb += XtX @ (alpha[l] * mu[l])
        if np.abs(alpha - old).max() < tol:
            break
    pip = 1 - np.prod(1 - alpha[V > 0], axis=0) if (V > 0).any() else np.zeros(p)
    sets = []
    for l in range(L):
        if V[l] <= 0:
            continue
        o = np.argsort(-alpha[l])
        m = int(np.searchsorted(np.cumsum(alpha[l][o]), coverage) + 1)
        cs = o[:m]
        pur = np.abs(R[np.ix_(cs, cs)]).min() if len(cs) > 1 else 1.0
        if pur >= min_abs_corr:
            others = sum((alpha[m] * mu[m] for m in range(L) if m != l), np.zeros(p))
            z_cond = (Xty - XtX @ others) / np.sqrt(n - 1.0)          # lead z conditional on the other effects
            sets.append(dict(effect=l, members=cs, alpha=alpha[l][cs], purity=pur, V=V[l], z_cond=z_cond[cs]))
    # drop duplicated sets (same lead)
    seen, uniq = set(), []
    for s in sets:
        if s["members"][0] not in seen:
            seen.add(s["members"][0])
            uniq.append(s)
    return dict(pip=pip, alpha=alpha, mu=mu, V=V, sets=uniq, niter=it + 1)
