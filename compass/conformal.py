"""Split-conformal functional credible sets (least ambiguous set-valued classifier).

For a calibration credible set with known causal variants T_g the score is
S_g = 1 - max_{i in T_g} q_i. The threshold tau is the
ceil((m + 1)(1 - alpha))-th smallest calibration score, and the set for a new
credible set is {i : q_i >= 1 - tau}. Under exchangeability the set contains a
causal variant with probability at least 1 - alpha.
"""
import numpy as np

from .metrics import group_indices

__all__ = ["conformal_threshold", "conformal_sets", "evaluate_cross_fit"]


def _split(q, truth, groups):
    q, truth = np.asarray(q, float), np.asarray(truth, bool)
    return [(q[m], truth[m]) for m in group_indices(groups)]


def conformal_threshold(q, truth, groups, alpha):
    """Calibrate tau on credible sets that contain at least one known causal variant."""
    S = np.array([1 - qq[t].max() for qq, t in _split(q, truth, groups) if t.any()])
    m = len(S)
    if m == 0:
        raise ValueError("no calibration sets with a causal variant")
    level = min(1.0, np.ceil((m + 1) * (1 - alpha)) / m)
    return float(np.quantile(S, level, method="higher"))


def conformal_sets(q, groups, tau):
    """Boolean membership of each variant in its conformal set (at least one per set)."""
    q = np.asarray(q, float)
    keep = q >= 1 - tau - 1e-12
    for m in group_indices(groups):
        if not keep[m].any():
            keep[m[np.argmax(q[m])]] = True
    return keep


def evaluate_cross_fit(q, truth, groups, fold, alpha):
    """Calibrate on fold == 0 and test on fold == 1, then the reverse.

    fold is a per-variant 0/1 label that is constant within a credible set
    (for example odd versus even chromosome). Returns pooled coverage and mean size.
    """
    q, truth, groups, fold = map(np.asarray, (q, truth, groups, fold))
    cov, size = [], []
    for f in (0, 1):
        cal, te = fold == f, fold != f
        tau = conformal_threshold(q[cal], truth[cal], groups[cal], alpha)
        qt, tt, gt = q[te], truth[te], groups[te]
        keep = conformal_sets(qt, gt, tau)
        for m in group_indices(gt):
            if not tt[m].any():
                continue
            cov.append(bool((keep[m] & tt[m]).any()))
            size.append(int(keep[m].sum()))
    return float(np.mean(cov)), float(np.mean(size))
