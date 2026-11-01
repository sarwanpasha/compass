"""Study design from resolution theory: two linked causal variants, selection-corrected
non-centrality and resolution-optimal allocation of samples across populations.

Notation follows compass.theory. For candidates u, v with LD r_uv and mean z-scores
eta_u, eta_v, the Bayes rule ranks u above v with probability
    Phi((eta_u - eta_v) / sqrt(2 (1 - r_uv))).
With one causal variant j of non-centrality c, eta_u = c r_uj (Theorem 1). With two causal
variants (Theorem 4), eta = R (c1 e_1 + c2 e_2).
"""
import numpy as np
from scipy import optimize, stats

Phi, Phiinv = stats.norm.cdf, stats.norm.ppf

__all__ = [
    "mean_z", "rank_prob", "two_causal_separation", "phantom_variants", "phantom_prone",
    "c_truncated_mle", "required_separation", "optimal_allocation", "auc_needed",
    "design_signal",
]


def mean_z(R, causal, c):
    """Expected z-scores eta = R @ (sum_k c_k e_{causal_k})."""
    R = np.asarray(R, float)
    beta = np.zeros(len(R))
    beta[np.asarray(causal)] = np.asarray(c, float)
    return R @ beta


def rank_prob(eta_u, eta_v, r_uv):
    """Probability that z_u > z_v when (z_u, z_v) are Gaussian with means eta and unit variances."""
    s = np.sqrt(np.clip(2 * (1 - np.asarray(r_uv, float)), 1e-12, None))
    return Phi((np.asarray(eta_u) - np.asarray(eta_v)) / s)


def two_causal_separation(c1, c2, r1l, r12, r2l):
    """Mean of z_1 - z_l with causal variants 1 and 2 (Theorem 4):
    c1 (1 - r1l) + c2 (r12 - r2l). The single-causal value is c1 (1 - r1l)."""
    return c1 * (1 - r1l) + c2 * (r12 - r2l)


def phantom_variants(R, causal, c):
    """Indices of non-causal variants whose expected z exceeds that of every causal variant.
    Such a variant is ranked first with probability tending to one as the sample grows."""
    eta = mean_z(R, causal, c)
    mask = np.ones(len(eta), bool)
    mask[np.asarray(causal)] = False
    return np.flatnonzero(mask & (eta > eta[np.asarray(causal)].max() + 1e-12))


def phantom_prone(R, max_r_causal=0.8):
    """For equal same-sign effects, a variant l is a phantom for causal pair (i, j) when
    r_il + r_jl > 1 + r_ij. Returns True if any pair with r_ij < max_r_causal has a phantom."""
    R = np.asarray(R, float)
    m = len(R)
    for i in range(m):
        for j in range(i + 1, m):
            if R[i, j] >= max_r_causal:
                continue
            s = R[i] + R[j]
            s[[i, j]] = -np.inf
            if (s > 1 + R[i, j]).any():
                return True
    return False


def c_truncated_mle(z, t):
    """Selection-corrected non-centrality (conditional maximum likelihood) for |z| reported
    only when |z| >= t: maximizes phi(|z| - c) / {Phi(c - t) + Phi(-c - t)} over c >= 0."""
    z = abs(float(z))
    if z < t:
        return np.nan
    nll = lambda c: -(stats.norm.logpdf(z - c) - np.log(Phi(c - t) + Phi(-c - t)))
    res = optimize.minimize_scalar(nll, bounds=(0.0, z), method="bounded")
    return float(res.x)


def required_separation(tau=0.9):
    """Bayes-rule pairwise accuracy Phi(sqrt(S / 2)) reaches tau when S >= 2 Phi^{-1}(tau)^2."""
    return 2 * Phiinv(tau) ** 2


def optimal_allocation(a, S0=None, tau=0.9, eps=1e-12):
    """Smallest number of additional individuals, split across populations, that resolves every
    partner (Proposition 4). a[p, l] = per-individual separation c_p^2 (1 - |r_pl|) / n_p for
    population p and partner l; S0[l] is the separation already achieved. Solves
        min sum_p n_p  s.t.  S0_l + sum_p a_pl n_p >= S*,  n_p >= 0
    and returns (N, n) with N = inf when some partner cannot be separated in any population."""
    a = np.asarray(a, float)
    P, L = a.shape
    S0 = np.zeros(L) if S0 is None else np.asarray(S0, float)
    need = required_separation(tau) - S0
    keep = need > 0
    if not keep.any():
        return 0.0, np.zeros(P)
    a, need = a[:, keep], need[keep]
    if (a.max(0) <= eps).any():
        return np.inf, np.full(P, np.nan)
    if a.shape[1] == 1:                                    # closed form: best single population
        p = int(np.argmax(a[:, 0]))
        n = np.zeros(P)
        n[p] = need[0] / a[p, 0]
        return float(n[p]), n
    res = optimize.linprog(np.ones(P), A_ub=-a.T, b_ub=-need, bounds=[(0, None)] * P, method="highs")
    if not res.success:
        return np.inf, np.full(P, np.nan)
    return float(res.fun), res.x


def auc_needed(S0, tau=0.9):
    """Within-set AUC a sequence channel needs so that the weakest partner reaches accuracy tau
    now: mu^2 = S* - min_l S0_l (Theorem 3), AUC = Phi(mu / sqrt 2)."""
    deficit = required_separation(tau) - np.min(S0)
    if deficit <= 0:
        return 0.5
    return float(Phi(np.sqrt(deficit) / np.sqrt(2)))


def design_signal(z, n0, het, r, disc="EUR", tau=0.9, max_factor=10.0, z_threshold=None):
    """Per-signal design summary.

    z: lead z-score in the discovery study; n0: its sample size; het[p]: heterozygosity 2f(1-f)
    of the lead in population p; r[p]: array of |r| between the lead and each partner in p.
    The discovery study is assumed to sample population `disc`. Returns a dict with the
    selection-corrected c, the fold increase needed with the discovery population alone, the
    fold increase with the optimal multi-population allocation and its shares, the AUC a
    sequence channel would need now, and a design class:
      'resolved'     accuracy tau against every partner already;
      'more_samples' resolvable with <= max_factor x n0 more individuals of `disc`;
      'ancestry'     not, but resolvable with <= max_factor x n0 individuals optimally allocated;
      'large'        resolvable only with more than max_factor x n0 individuals;
      'functional'   some partner has |r| = 1 in every population (no sample size suffices).
    """
    pops = list(r)
    c = abs(z) if z_threshold is None else c_truncated_mle(z, z_threshold)
    if not np.isfinite(c):
        c = abs(z)
    per = c * c / n0 / het[disc]                         # non-centrality^2 per individual per unit het
    a = np.vstack([per * het[p] * (1 - np.clip(np.abs(r[p]), 0, 1)) for p in pops])
    S0 = a[pops.index(disc)] * n0
    Sstar = required_separation(tau)
    out = dict(c=c, min_S0=float(S0.min()), auc_needed=auc_needed(S0, tau))
    deficit = Sstar - S0
    if (deficit <= 0).all():
        out.update(cls="resolved", fold_disc=0.0, fold_opt=0.0)
        return out
    ad = a[pops.index(disc)]
    with np.errstate(divide="ignore"):
        n_disc = np.max(np.where(deficit > 0, deficit / ad, 0))
    N, n = optimal_allocation(a, S0, tau)
    out.update(fold_disc=float(n_disc / n0), fold_opt=float(N / n0))
    if np.isfinite(N) and N > 0:
        out.update({f"share_{p}": float(n[i] / N) for i, p in enumerate(pops)})
    if not np.isfinite(N):
        out["cls"] = "functional"
    elif n_disc <= max_factor * n0:
        out["cls"] = "more_samples"
    elif N <= max_factor * n0:
        out["cls"] = "ancestry"
    else:
        out["cls"] = "large"
    return out
