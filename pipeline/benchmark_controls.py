"""Controls for the SR-FM benchmark and the conformal sets.

Usage: python benchmark_controls.py MODEL   (after benchmark_v2.py MODEL)

(1) Negative control: the sequence score is permuted across credible sets among variants in the same bin of
    LD stratum x indel status (20 permutations); SR-FM is refitted with the permuted score.
    This keeps the association of the score with generic features and removes variant-specific information.
(2) Strata by discovery-study lead |z| and by credible-set size (tertiles).
(3) Within-set correlation of the sequence score among non-causal members, by distance between members
    (the quadrature law assumes independent noise across candidates).
(4) Conformal sets: coverage conditional on LD stratum and set size for standard split conformal and for
    LD-stratified (Mondrian) conformal; calibration and test split by donor cohort instead of chromosome;
    selection-weighted conformal, in which calibration sets are reweighted to the distribution of all
    unresolved credible sets (set size, minimum r^2, lead |z|, sample size) and coverage is estimated with
    the same importance weights.
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression

from common import R, D

model = sys.argv[1]
rng = np.random.default_rng(0)
OUT = {"model": model}


def eb_lambda(G, col, grid=np.linspace(0, 4, 81)):
    lp = [np.log(np.clip(g.pip.values, 1e-6, 1)) for g in G]
    sc = [g[col].values for g in G]
    ll = [sum(logsumexp(a + l * s - logsumexp(l * s)) for a, s in zip(lp, sc)) for l in grid]
    return float(grid[int(np.argmax(ll))])


def srfm(G, col, lam):
    return [np.exp(g.lpip.values + lam * g[col].values - logsumexp(g.lpip.values + lam * g[col].values)) for g in G]


def top1(G, P):
    return float(np.mean([g.truth.values[np.argmax(p)] for g, p in zip(G, P)]))


def auc(G, P):
    num = den = 0.0
    for g, p in zip(G, P):
        t = g.truth.values
        if t.all() or not t.any():
            continue
        a, b = p[t], p[~t]
        num += (a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()
        den += len(a) * len(b)
    return num / den


def conf_eval(S, test_mask, cal_mask, alpha, PP, weights=None, strata=None, w_test=None):
    """Split conformal with optional Mondrian strata or likelihood-ratio weights. Returns per-set coverage
    indicator and size for the test sets."""
    cov = np.full(len(S), np.nan)
    size = np.full(len(S), np.nan)
    for i in np.flatnonzero(test_mask):
        cm = cal_mask if strata is None else (cal_mask & (strata == strata[i]))
        Sc = S[cm]
        if weights is None:
            m = len(Sc)
            q = np.quantile(Sc, min(1, np.ceil((m + 1) * (1 - alpha)) / m), method="higher")
        else:
            w = weights[cm]
            o = np.argsort(Sc)
            p = np.append(w[o], w_test[i]) / (w.sum() + w_test[i])
            k = np.searchsorted(np.cumsum(p[:-1]), 1 - alpha)
            q = Sc[o][k] if k < len(Sc) else np.inf
        cov[i] = S[i] <= q + 1e-12
        size[i] = max(1, int((PP[i] >= 1 - q - 1e-12).sum()))
    return cov, size


for truth in ["repl", "repl_strict", "mage"]:
    A = pd.read_parquet(R + f"bench_v2_{model}_{truth}.parquet")
    A["lpip"] = np.log(np.clip(A.pip, 1e-6, 1))
    G = [g for _, g in A.groupby("cs_id")]
    res = {}
    lam = eb_lambda(G, "seq")
    P0 = srfm(G, "seq", lam)
    res["observed"] = dict(lam=lam, top1=top1(G, P0), auc=auc(G, P0), top1_pip=top1(G, [g.pip.values for g in G]))
    # (1) permuted-score control
    bins = A.ld_stratum.astype(str) + "_" + A.indel.astype(str)
    null = []
    for b in range(20):
        A["seq_perm"] = A.groupby(bins).seq.transform(lambda x: rng.permutation(x.values))
        Gp = [g for _, g in A.groupby("cs_id")]
        lp = eb_lambda(Gp, "seq_perm")
        Pp = srfm(Gp, "seq_perm", lp)
        null.append(dict(lam=lp, top1=top1(Gp, Pp), auc=auc(Gp, Pp)))
    N = pd.DataFrame(null)
    res["permuted"] = dict(lam_mean=float(N.lam.mean()), top1_mean=float(N.top1.mean()), top1_max=float(N.top1.max()),
                           auc_mean=float(N.auc.mean()), auc_max=float(N.auc.max()))
    # (2) strata by lead |z| and set size
    zc = "z" if "z" in A else None
    st = {}
    if zc:
        lz = A.groupby("cs_id")[zc].apply(lambda x: x.abs().max())
        tert = pd.qcut(lz.rank(method="first"), 3, labels=["low", "mid", "high"])
        for k in ["low", "mid", "high"]:
            ids = set(tert[tert == k].index)
            Gs = [g for g in G if g.cs_id.iloc[0] in ids]
            Ps = [p for g, p in zip(G, P0) if g.cs_id.iloc[0] in ids]
            st[f"leadz_{k}"] = dict(n=len(Gs), lead_z_median=float(lz[list(ids)].median()), top1_pip=top1(Gs, [g.pip.values for g in Gs]),
                                    top1_srfm=top1(Gs, Ps), auc_seq=auc(Gs, [g.seq.values for g in Gs]), auc_srfm=auc(Gs, Ps))
    sz = pd.Series({g.cs_id.iloc[0]: len(g) for g in G})
    tsz = pd.qcut(sz.rank(method="first"), 3, labels=["small", "medium", "large"])
    for k in ["small", "medium", "large"]:
        ids = set(tsz[tsz == k].index)
        Gs = [g for g in G if g.cs_id.iloc[0] in ids]
        Ps = [p for g, p in zip(G, P0) if g.cs_id.iloc[0] in ids]
        st[f"size_{k}"] = dict(n=len(Gs), size_median=float(sz[list(ids)].median()), top1_pip=top1(Gs, [g.pip.values for g in Gs]),
                               top1_srfm=top1(Gs, Ps), auc_seq=auc(Gs, [g.seq.values for g in Gs]), auc_srfm=auc(Gs, Ps))
    res["strata"] = st
    # (3) within-set correlation of the score among non-causal members, by distance
    pos = A.variant.str.split("_").str[1].astype(int)
    A["pos"] = pos
    dd, prod = [], []
    for g in G:
        h = A.loc[g.index]
        h = h[~h.truth]
        if len(h) < 2:
            continue
        s = h.seq.values - h.seq.values.mean()
        p = h.pos.values
        iu = np.triu_indices(len(h), 1)
        dd.append(np.abs(p[iu[0]] - p[iu[1]]))
        prod.append(np.vstack([h.seq.values[iu[0]], h.seq.values[iu[1]]]).T)
    dd, prod = np.concatenate(dd), np.vstack(prod)
    corr = {}
    for lo, hi in [(0, 1000), (1000, 10000), (10000, 100000), (100000, 10 ** 9)]:
        m = (dd >= lo) & (dd < hi)
        if m.sum() > 50:
            corr[f"{lo}-{hi}"] = dict(n_pairs=int(m.sum()), r=float(np.corrcoef(prod[m, 0], prod[m, 1])[0, 1]))
    nc = A[~A.truth]
    within = nc.groupby("cs_id").seq.transform(lambda x: x - x.mean())
    res["noise_correlation"] = dict(by_distance=corr, icc=float(1 - within.var() / nc.seq.var()))
    # (4) conformal
    ids = [g.cs_id.iloc[0] for g in G]
    S = np.array([1 - p[g.truth.values].max() for g, p in zip(G, P0)])
    S_pip = np.array([1 - (g.pip.values / g.pip.sum())[g.truth.values].max() for g in G])
    PP_pip = [g.pip.values / g.pip.sum() for g in G]
    ldst = np.array([str(g.ld_stratum.iloc[0]) for g in G])
    szt = tsz.reindex(ids).astype(str).values
    odd = np.array([g.chrom.iloc[0].isdigit() and int(g.chrom.iloc[0]) % 2 == 1 for g in G])
    conf = {}
    for alpha in [0.2, 0.1]:
        for nm, scores, posts in [("srfm", S, P0), ("pip", S_pip, PP_pip)]:
            for mode, strata in [("split", None), ("mondrian_ld", ldst), ("mondrian_size", szt),
                                 ("mondrian_ld_size", np.char.add(ldst.astype(str), szt.astype(str)))]:
                cov = np.full(len(S), np.nan)
                size = np.full(len(S), np.nan)
                for f in (odd, ~odd):
                    c_, s_ = conf_eval(scores, ~f, f, alpha, posts, strata=strata)
                    m = ~np.isnan(c_)
                    cov[m], size[m] = c_[m], s_[m]
                rec = dict(cov=float(np.nanmean(cov)), size=float(np.nanmean(size)))
                rec["by_ld"] = {k: dict(n=int((ldst == k).sum()), cov=float(np.nanmean(cov[ldst == k])), size=float(np.nanmean(size[ldst == k])))
                                for k in np.unique(ldst)}
                rec["by_size"] = {k: dict(n=int((szt == k).sum()), cov=float(np.nanmean(cov[szt == k])), size=float(np.nanmean(size[szt == k])))
                                  for k in np.unique(szt)}
                conf[f"{nm}_{mode}_{alpha}"] = rec
    # donor-cohort split
    if "cohort" in A:
        coh = np.array([str(g.cohort.iloc[0]) for g in G])
        uc = np.unique(coh)
        fold = np.isin(coh, rng.choice(uc, len(uc) // 2, replace=False))
        for alpha in [0.2, 0.1]:
            cov, size = [], []
            for f in (fold, ~fold):
                c_, s_ = conf_eval(S, ~f, f, alpha, P0)
                cov += list(c_[~np.isnan(c_)])
                size += list(s_[~np.isnan(s_)])
            conf[f"srfm_cohortsplit_{alpha}"] = dict(cov=float(np.mean(cov)), size=float(np.mean(size)), n_cohorts=len(uc))
    res["conformal"] = conf
    # selection-weighted conformal: reweight to all unresolved credible sets
    try:
        Call = pd.read_parquet(D + "eqtlcat/cs_ge_r7.parquet", columns=["dataset_id", "gene_id", "cs_id", "pip", "z", "cs_size", "cs_min_r2"])
        meta = pd.read_csv(D + "eqtlcat/dataset_metadata_r7.tsv", sep="\t")[["dataset_id", "sample_size"]]
        Fs = Call.groupby("cs_id").agg(maxpip=("pip", "max"), lz=("z", lambda x: x.abs().max()), size=("cs_size", "first"),
                                       minr2=("cs_min_r2", "first"), dataset_id=("dataset_id", "first")).reset_index()
        Fs = Fs.merge(meta, on="dataset_id")
        Fs = Fs[(Fs.maxpip < 0.5) & (Fs["size"] <= 50) & (Fs["size"] >= 2)]
        Fs["in_truth"] = Fs.cs_id.isin(set(ids))
        X = np.column_stack([np.log(Fs["size"]), Fs.minr2.clip(0, 1), np.log(Fs.lz), np.log(Fs.sample_size)])
        lr = LogisticRegression(max_iter=2000).fit(X, Fs.in_truth)
        p = pd.Series(lr.predict_proba(X)[:, 1], index=Fs.cs_id)
        w_all = ((1 - p) / p).reindex(ids).values
        ok = np.isfinite(w_all)
        w_all = np.where(ok, w_all, np.nanmedian(w_all))
        wsel = {}
        for alpha in [0.2, 0.1]:
            for nm, wt in [("unweighted", None), ("weighted", w_all)]:
                cov = np.full(len(S), np.nan)
                size = np.full(len(S), np.nan)
                for f in (odd, ~odd):
                    c_, s_ = conf_eval(S, ~f, f, alpha, P0, weights=wt, w_test=w_all)
                    m = ~np.isnan(c_)
                    cov[m], size[m] = c_[m], s_[m]
                # coverage on the target population estimated with importance weights
                wsel[f"{nm}_{alpha}"] = dict(cov_target=float(np.sum(w_all * cov) / np.sum(w_all)),
                                             size_target=float(np.sum(w_all * size) / np.sum(w_all)),
                                             cov_calibration_population=float(np.mean(cov)))
        wsel["n_unresolved_sets"] = int(len(Fs))
        wsel["weight_cv"] = float(np.std(w_all) / np.mean(w_all))
        res["selection_weighted"] = wsel
    except Exception as e:  # noqa: BLE001
        res["selection_weighted"] = dict(error=str(e))
    OUT[truth] = res
    print(truth, json.dumps(res, default=float)[:2500], flush=True)
json.dump(OUT, open(R + f"bench_controls_{model}.json", "w"), indent=1, default=float)
print("DONE")
