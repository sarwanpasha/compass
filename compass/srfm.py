"""Sequence-resolved fine-mapping (SR-FM).

Within each credible set g with PIPs p_i and standardized sequence scores s_i,
the sequence prior is pi_i(lambda) = softmax(lambda * s)_i and the SR-FM
posterior is q_i proportional to p_i * pi_i(lambda). The temperature lambda is
an empirical Bayes estimate that maximizes
    sum_g log sum_{i in g} p_i pi_i(lambda),
so no functional labels are used.
"""
import numpy as np
from scipy.special import logsumexp

from .metrics import group_indices

__all__ = ["sequence_score", "standardize", "fit_lambda", "srfm_posterior"]


def sequence_score(delta, track_kind, use=("DNASE", "ATAC", "CAGE")):
    """Label-free sequence score from predicted alt-minus-ref effects.

    delta : array (n_variants, n_tracks) of predicted effects.
    track_kind : array (n_tracks,) of assay names, e.g. 'DNASE', 'CAGE'.
    Returns log of the root mean square of track-standardized effects over the
    selected tracks, plus 1e-3.
    """
    delta = np.asarray(delta, float)
    Z = delta / (delta.std(0) + 1e-6)
    sel = np.isin(np.char.upper(np.asarray(track_kind, str)), list(use))
    if not sel.any():
        raise ValueError("no tracks of the requested kinds")
    return np.log(np.sqrt((Z[:, sel] ** 2).mean(1)) + 1e-3)


def standardize(x):
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / np.nanstd(x)


def _padded(pips, scores, groups):
    idx = group_indices(groups)
    K = max(len(i) for i in idx)
    LP = np.full((len(idx), K), -np.inf)
    S = np.zeros((len(idx), K))
    M = np.zeros((len(idx), K), bool)
    for r, i in enumerate(idx):
        LP[r, :len(i)] = np.log(pips[i])
        S[r, :len(i)] = scores[i]
        M[r, :len(i)] = True
    return LP, S, M


def fit_lambda(pips, scores, groups, grid=np.linspace(0, 4, 81)):
    """Empirical Bayes temperature by grid search over the marginal likelihood.

    pips, scores, groups : 1-D arrays of equal length (groups = credible-set id).
    Returns (lambda_hat, log-likelihood values on the grid).
    """
    pips = np.clip(np.asarray(pips, float), 1e-6, 1)
    scores = np.asarray(scores, float)
    LP, S, M = _padded(pips, scores, groups)
    ll = []
    for lam in grid:
        prior = np.where(M, lam * S, -np.inf)
        prior = prior - logsumexp(prior, axis=1, keepdims=True)
        ll.append(logsumexp(LP + prior, axis=1).sum())
    ll = np.array(ll)
    return float(grid[int(np.argmax(ll))]), ll


def srfm_posterior(pips, scores, groups, lam):
    """SR-FM posterior q_i within each credible set (sums to one per set)."""
    pips = np.clip(np.asarray(pips, float), 1e-6, 1)
    a = np.log(pips) + lam * np.asarray(scores, float)
    q = np.empty_like(a)
    for m in group_indices(groups):
        q[m] = np.exp(a[m] - logsumexp(a[m]))
    return q
