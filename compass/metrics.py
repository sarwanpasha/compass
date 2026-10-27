"""Within-credible-set evaluation metrics."""
import numpy as np

__all__ = ["group_indices", "within_set_auc", "top1_accuracy", "credible_set", "ece"]


def group_indices(groups):
    """List of index arrays, one per distinct group label, in sorted label order."""
    groups = np.asarray(groups)
    order = np.argsort(groups, kind="stable")
    g = groups[order]
    cuts = np.flatnonzero(g[1:] != g[:-1]) + 1
    return np.split(order, cuts)


_groups = group_indices


def within_set_auc(score, truth, groups):
    """Fraction of (causal, non-causal) pairs in the same set where the causal
    variant scores higher; ties count one half; pooled over sets."""
    score, truth = np.asarray(score, float), np.asarray(truth, bool)
    num = den = 0.0
    for idx in _groups(groups):
        a, b = score[idx][truth[idx]], score[idx][~truth[idx]]
        if len(a) == 0 or len(b) == 0:
            continue
        num += (a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()
        den += len(a) * len(b)
    return num / den if den else np.nan


def top1_accuracy(score, truth, groups):
    """Fraction of sets whose highest-scoring member is causal."""
    score, truth = np.asarray(score, float), np.asarray(truth, bool)
    return float(np.mean([truth[idx][np.argmax(score[idx])] for idx in _groups(groups)]))


def credible_set(pip, level=0.95):
    """Indices of the smallest set whose PIPs sum to at least `level`."""
    pip = np.asarray(pip, float)
    o = np.argsort(-pip)
    m = int(np.searchsorted(np.cumsum(pip[o]), level) + 1)
    return o[:m]


def ece(pred, obs, n_bins=10):
    """Mean absolute difference between predicted and observed values over
    quantile bins of the prediction."""
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    ranks = np.argsort(np.argsort(pred, kind="stable"))
    b = (ranks * n_bins) // len(pred)
    return float(np.mean([abs(pred[b == k].mean() - obs[b == k].mean()) for k in range(n_bins) if (b == k).any()]))
