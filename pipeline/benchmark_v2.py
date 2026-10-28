"""Extended benchmark of SR-FM against PIP, sequence scores and published-style baselines.

Usage: python benchmark_v2.py MODEL   (MODEL in enformer, borzoi, alphagenome; after scoring and ld_truthsets.py)

Truths: replication (resolved in an independent donor cohort), strict replication (two independent
cohorts) and multi-ancestry (MAGE, discovery cohort not from 1000 Genomes donors).
Scores, all computed within credible sets:
  pip           SuSiE PIP
  seq           label-free sequence score (RMS of DNase/ATAC/CAGE effects at the variant)
  seq_tss       label-free gene-specific score (RMS of CAGE/RNA effects at the TSS of the eQTL gene)
  srfm          PIP x softmax(lambda * seq), lambda by empirical Bayes (one value)
  srfm_tss      same with seq_tss (seq where the TSS lies outside the model window)
  srfm_strat    lambda estimated separately in each LD stratum
  srfm_sup      SR-FM with a supervised sequence score (logistic regression, leave-chromosome-out)
  fipip         PIP x classifier probability renormalized in the set (gradient boosting on sequence
                features trained on truth labels of other chromosomes; Sweeney and Kang form)
  emslike       PIP x classifier probability, classifier trained on PIP >= 0.9 vs PIP < 0.01 members of
                other credible sets (EMS-style training signal, no truth labels)
  polyfun_eb    PIP x softmax(w' a) with 16 annotations (sequence-feature principal components and TSS
                distance) and w by empirical Bayes across credible sets (PolyFun-style functional prior)
"""
import json
import os
import sys
import glob

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from common import R, D

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from compass.metrics import group_indices  # noqa: E402

model = sys.argv[1]
rng = np.random.default_rng(0)
OUT = {"model": model}


# ------------------------------------------------------------------ features
def load_scores(m):
    metas, locs, tss, tmeta = [], [], [], []
    for f in [f for f in sorted(glob.glob(R + f"scores/{m}_*_meta.parquet")) if not f.endswith("_tss_meta.parquet")]:
        mt = pd.read_parquet(f)
        l = np.load(f.replace("_meta.parquet", "_loc.npy")).astype(np.float32)
        off = sum(len(x) for x in metas)
        metas.append(mt)
        locs.append(l)
        tf = f.replace("_meta.parquet", "_tss.npy")
        if os.path.exists(tf):
            tm = pd.read_parquet(f.replace("_meta.parquet", "_tss_meta.parquet"))
            tm["row"] += off
            tss.append(np.load(tf).astype(np.float32))
            tmeta.append(tm)
    M = pd.concat(metas, ignore_index=True)
    X = np.vstack(locs)
    M["variant"] = "chr" + M.chrom.astype(str).str.replace("chr", "") + "_" + M.pos.astype(str) + "_" + M.ref + "_" + M.alt
    T = np.vstack(tss) if tss else None
    TM = pd.concat(tmeta, ignore_index=True) if tmeta else None
    return M, X, T, TM


M, X, T, TM = load_scores(model)
ok = M.ok.values
kinds = pd.read_csv(D + f"models/{model}_targets_human.txt", sep="\t").description.astype(str)
kind = kinds.str.split(":").str[0].str.upper().values[:X.shape[1]]
Z = X / (X[ok].std(0) + 1e-6)
sel = np.isin(kind, ["DNASE", "ATAC", "CAGE"])
M["seq_raw"] = np.where(ok, np.log(np.sqrt((Z[:, sel] ** 2).mean(1)) + 1e-3), np.nan)
F = np.sign(X) * np.log1p(np.abs(Z))
pca = PCA(32, random_state=0).fit(F[ok][rng.choice(ok.sum(), min(60000, ok.sum()), replace=False)])
PC = pca.transform(F)
PCF = np.hstack([PC, np.abs(PC)])
pc_cols = [f"pc{i}" for i in range(PCF.shape[1])]
M = pd.concat([M, pd.DataFrame(PCF, columns=pc_cols)], axis=1)
feat = M[["variant", "seq_raw"] + pc_cols].drop_duplicates("variant")

tss_feat = None
if T is not None and len(TM):
    TZ = T / (T.std(0) + 1e-6)
    TM["seq_tss_raw"] = np.log(np.sqrt((TZ ** 2).mean(1)) + 1e-3)
    TM["variant"] = M.variant.values[TM.row.values]
    tss_feat = TM.groupby(["variant", "gene_id"]).seq_tss_raw.max().reset_index()


# ------------------------------------------------------------------ helpers
def ws_auc(G, col):
    num = den = 0.0
    for g in G:
        y = g.truth.values.astype(bool)
        s = g[col].values
        a, b = s[y], s[~y]
        num += (a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()
        den += len(a) * len(b)
    return num / max(den, 1)


def ws_auc_perset(G, col):
    v = []
    for g in G:
        y = g.truth.values.astype(bool)
        s = g[col].values
        a, b = s[y], s[~y]
        if not len(a) or not len(b):
            continue
        v.append(((a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()) / (len(a) * len(b)))
    return float(np.mean(v))


def top1_vec(G, col):
    return np.array([g.truth.values[np.argmax(g[col].values)] for g in G], float)


def eb_lambda(G, col, grid=np.linspace(0, 4, 81)):
    lp = [np.log(np.clip(g.pip.values, 1e-6, 1)) for g in G]
    sc = [g[col].values for g in G]
    ll = [sum(logsumexp(a + l * s - logsumexp(l * s)) for a, s in zip(lp, sc)) for l in grid]
    return float(grid[int(np.argmax(ll))])


def post(G, logprior_col):
    out = []
    for g in G:
        a = np.log(np.clip(g.pip.values, 1e-6, 1)) + g[logprior_col].values
        out.append(np.exp(a - logsumexp(a)))
    return out


def chrom_of(df):
    return df.variant.str.split("_").str[0].str.replace("chr", "")


# ------------------------------------------------------------------ truth sets
LDT = pd.read_parquet(R + "truthsets_ld.parquet") if os.path.exists(R + "truthsets_ld.parquet") else None
results = {}
for truth_name, f, tcol in [("repl", "truth_repl_fullsets.parquet", "truth"),
                            ("repl_strict", "truth_repl_fullsets.parquet", "truth_strict"),
                            ("mage", "truth_mage_fullsets.parquet", "truth")]:
    A = pd.read_parquet(R + f)
    A["truth"] = A[tcol].astype(bool)
    A = A[A.cs_id.isin(A.groupby("cs_id").truth.sum().pipe(lambda t: t[t >= 1].index))]
    A = A.merge(feat, on="variant", how="left")
    if tss_feat is not None:
        A = A.merge(tss_feat, on=["variant", "gene_id"], how="left")
    else:
        A["seq_tss_raw"] = np.nan
    full = A.groupby("cs_id").seq_raw.apply(lambda x: x.notna().all())
    A = A[A.cs_id.isin(full[full].index)].copy()
    A["seq"] = (A.seq_raw - A.seq_raw.mean()) / A.seq_raw.std()
    t = A.seq_tss_raw
    A["seq_tss"] = np.where(t.notna(), (t - t.mean()) / t.std(), A.seq)
    A["tss_available"] = t.notna()
    A["chrom"] = chrom_of(A)
    A["indel"] = (A.ref.str.len() != 1) | (A.alt.str.len() != 1)
    A["lpip"] = np.log(np.clip(A.pip, 1e-6, 1))
    A["ld_stratum"] = pd.cut(A.cs_min_r2, [-0.01, 0.5, 0.8, 0.95, 1.0], labels=["<0.5", "0.5-0.8", "0.8-0.95", ">0.95"])
    if LDT is not None:
        A = A.merge(LDT[["cs_id", "eur_irreducible", "truth_max_r2_eur"]].drop_duplicates("cs_id"), on="cs_id", how="left")
    res = dict(n_sets=int(A.cs_id.nunique()), n_members=len(A), indel_frac=float(A.indel.mean()),
               tss_available=float(A.tss_available.mean()))

    # --- label-free SR-FM variants
    G = [g for _, g in A.groupby("cs_id")]
    for c in ["seq", "seq_tss"]:
        lam = eb_lambda(G, c)
        res[f"lambda_{c}"] = lam
        A[f"srfm_{c}"] = A.lpip + lam * A[c]
    A["srfm"] = A["srfm_seq"]
    A["srfm_tss"] = A["srfm_seq_tss"]
    A["srfm_strat"] = np.nan
    res["lambda_by_stratum"] = {}
    for st, idx in A.groupby("ld_stratum", observed=True).groups.items():
        Gs = [g for _, g in A.loc[idx].groupby("cs_id")]
        lam = eb_lambda(Gs, "seq")
        A.loc[idx, "srfm_strat"] = A.loc[idx, "lpip"] + lam * A.loc[idx, "seq"]
        # within-set SD of the score and binormal mu from the within-set AUC
        sd_w = float(np.sqrt(np.mean(np.concatenate([(g.seq - g.seq.mean()).values ** 2 for g in Gs]))))
        auc = ws_auc(Gs, "seq")
        mu = float(np.sqrt(2) * norm.ppf(auc))
        res["lambda_by_stratum"][str(st)] = dict(n=len(Gs), lam=lam, auc_seq=auc, mu=mu, sd_within=sd_w,
                                                 mu_over_sd=mu / sd_w)

    # --- supervised variants (leave-one-chromosome-out on truth labels)
    A["sup_score"] = np.nan
    A["fipip_prob"] = np.nan
    try:
        from xgboost import XGBClassifier
    except Exception:
        XGBClassifier = None
    feats = pc_cols
    for c in A.chrom.unique():
        te = (A.chrom == c).values
        tr = ~te
        lr = LogisticRegression(C=0.1, max_iter=3000).fit(A.loc[tr, feats], A.loc[tr, "truth"])
        A.loc[te, "sup_score"] = lr.decision_function(A.loc[te, feats])
        if XGBClassifier is not None:
            xg = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                               n_jobs=32, verbosity=0).fit(A.loc[tr, feats + ["seq", "seq_tss"]], A.loc[tr, "truth"])
            A.loc[te, "fipip_prob"] = xg.predict_proba(A.loc[te, feats + ["seq", "seq_tss"]])[:, 1]
        else:
            A.loc[te, "fipip_prob"] = 1 / (1 + np.exp(-A.loc[te, "sup_score"]))
    A["sup_score"] = (A.sup_score - A.sup_score.mean()) / A.sup_score.std()
    G = [g for _, g in A.groupby("cs_id")]
    lam = eb_lambda(G, "sup_score")
    res["lambda_sup"] = lam
    A["srfm_sup"] = A.lpip + lam * A.sup_score
    A["fipip"] = A.lpip + np.log(np.clip(A.fipip_prob, 1e-6, 1))

    # --- EMS-like: classifier trained on confidently fine-mapped variants of other credible sets
    E = pd.read_parquet(R + "ems_training.parquet").merge(feat, on="variant").dropna(subset=["seq_raw"])
    E["chrom"] = chrom_of(E)
    A["emslike_prob"] = 0.5
    for c in (A.chrom.unique() if E.label.nunique() == 2 and len(E) > 1000 else []):
        tr = E[(E.chrom != c) & ~E.variant.isin(set(A.variant))]
        lr = LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced").fit(tr[pc_cols], tr.label)
        te = (A.chrom == c).values
        A.loc[te, "emslike_prob"] = lr.predict_proba(A.loc[te, pc_cols])[:, 1]
    A["emslike"] = A.lpip + np.log(np.clip(A.emslike_prob, 1e-6, 1))

    # --- PolyFun-style multi-annotation empirical Bayes prior
    ann = pc_cols[:15]
    Aa = A[ann].values
    Aa = (Aa - Aa.mean(0)) / Aa.std(0)
    Aa = np.hstack([Aa, A[["seq"]].values])
    idx = group_indices(A.cs_id.values)
    lp = A.lpip.values

    def negll(w):
        e = Aa @ w
        return -sum(logsumexp(lp[i] + e[i] - logsumexp(e[i])) for i in idx) + 0.5 * 1e-2 * (w @ w)

    w = minimize(negll, np.zeros(Aa.shape[1]), method="L-BFGS-B").x
    A["polyfun_eb"] = A.lpip + Aa @ w
    res["polyfun_eb_weights_norm"] = float(np.linalg.norm(w))

    # --- metrics
    G = [g for _, g in A.groupby("cs_id")]
    cols = ["pip", "seq", "seq_tss", "srfm", "srfm_tss", "srfm_strat", "srfm_sup", "fipip", "emslike", "polyfun_eb", "sup_score"]
    T1 = {c: top1_vec(G, c) for c in cols}
    res["metrics"] = {c: dict(top1=float(T1[c].mean()), auc_pooled=ws_auc(G, c), auc_perset=ws_auc_perset(G, c)) for c in cols}
    res["random_top1"] = float(np.mean([g.truth.mean() for g in G]))
    B = np.array([rng.choice(len(G), len(G)) for _ in range(2000)])
    ci = lambda a, b: np.percentile([(T1[a][bb] - T1[b][bb]).mean() for bb in B], [2.5, 97.5]).tolist()
    res["ci_top1"] = {f"{a}-{b}": ci(a, b) for a, b in [("srfm", "pip"), ("srfm", "seq"), ("srfm_tss", "seq_tss"),
                                                        ("srfm_tss", "srfm"), ("fipip", "srfm"), ("fipip", "srfm_tss"),
                                                        ("srfm_strat", "srfm"), ("emslike", "srfm"), ("polyfun_eb", "srfm")]}

    # --- strata: LD, SNV vs indel truth, EUR-irreducible class
    strata = {}
    keyf = {"ld": lambda g: str(g.ld_stratum.iloc[0]),
            "truth_type": lambda g: "indel" if g[g.truth].indel.any() else "snv"}
    if "eur_irreducible" in A:
        keyf["eur_irreducible"] = lambda g: str(bool(g.eur_irreducible.iloc[0])) if pd.notna(g.eur_irreducible.iloc[0]) else "NA"
    for nm, kf in keyf.items():
        lab = np.array([kf(g) for g in G])
        strata[nm] = {}
        for v in np.unique(lab):
            Gs = [g for g, l in zip(G, lab) if l == v]
            if len(Gs) < 30:
                continue
            strata[nm][v] = dict(n=len(Gs), **{c: dict(top1=float(top1_vec(Gs, c).mean()), auc=ws_auc(Gs, c))
                                               for c in ["pip", "seq", "seq_tss", "srfm", "srfm_tss", "srfm_strat", "fipip"]})
            ap, as_ = ws_auc(Gs, "pip"), ws_auc(Gs, "seq")
            strata[nm][v]["pred_quadrature"] = float(norm.cdf(np.sqrt(max(norm.ppf(ap), 0) ** 2 + max(norm.ppf(as_), 0) ** 2)))
            strata[nm][v]["oracle"] = max(ws_auc([g.assign(t=g.lpip + w_ * g.seq) for g in Gs], "t") for w_ in [0, .25, .5, 1, 2, 4])
    res["strata"] = strata

    # --- channel models: equal-variance binormal, unequal-variance binormal and empirical
    mixed = lambda gs: [g for g in gs if 0 < g.truth.sum() < len(g)]
    cen = pd.concat([g.assign(sc=g.seq - g.seq[~g.truth].mean()) for g in mixed(G)])
    s1, s0 = cen.sc[cen.truth].values, cen.sc[~cen.truth].values
    res["channel"] = dict(mean_causal=float(s1.mean()), sd_causal=float(s1.std()), sd_noncausal=float(s0.std()),
                          skew_noncausal=float(pd.Series(s0).skew()), kurt_noncausal=float(pd.Series(s0).kurt()))
    chan = {}
    for st, idx_ in A.groupby("ld_stratum", observed=True).groups.items():
        Gs = mixed([g for _, g in A.loc[idx_].groupby("cs_id")])
        ap, as_ = ws_auc(Gs, "pip"), ws_auc(Gs, "seq")
        a = np.sqrt(2) * norm.ppf(max(ap, 0.5))
        Cs = pd.concat([g.assign(sc=g.seq - g.seq[~g.truth].mean()) for g in Gs])
        c1, c0 = Cs.sc[Cs.truth].values, Cs.sc[~Cs.truth].values
        m1, sd1, sd0 = c1.mean(), c1.std(), c0.std()
        n_mc = 200000
        xd1, xd0 = rng.normal(a, 1, n_mc), rng.normal(0, 1, n_mc)
        preds = {}
        for nm_, (u1, u0) in {"binormal_equal": (rng.normal(m1 / sd0, 1, n_mc), rng.normal(0, 1, n_mc)),
                              "binormal_unequal": (rng.normal(m1 / sd0, sd1 / sd0, n_mc), rng.normal(0, 1, n_mc)),
                              "empirical": (rng.choice(c1, n_mc) / sd0, rng.choice(c0, n_mc) / sd0)}.items():
            best = 0
            for w_ in np.linspace(0, 3, 31):
                best = max(best, float(np.mean(xd1 + w_ * u1 > xd0 + w_ * u0)))
            preds[nm_] = best
        chan[str(st)] = dict(n=len(Gs), auc_pip=ap, auc_seq=as_, observed_srfm=ws_auc(Gs, "srfm"),
                             observed_oracle=max(ws_auc([g.assign(t=g.lpip + w_ * g.seq) for g in Gs], "t") for w_ in [0, .25, .5, 1, 2, 4]),
                             quadrature=float(norm.cdf(np.sqrt(max(norm.ppf(ap), 0) ** 2 + max(norm.ppf(as_), 0) ** 2))), **preds)
    res["channel_by_ld"] = chan

    # --- conformal sets within this truth (odd/even chromosome cross-fit)
    odd = np.array([int(g.chrom.iloc[0]) % 2 == 1 if g.chrom.iloc[0].isdigit() else False for g in G])
    conf = {}
    for col in ["pip", "srfm", "srfm_tss", "fipip"]:
        lc = "lpip" if col == "pip" else col
        PP = [np.exp(g[lc].values - logsumexp(g[lc].values)) for g in G]
        sc = np.array([1 - p[g.truth.values].max() for p, g in zip(PP, G)])
        for alpha in [0.2, 0.1]:
            cov, size = [], []
            for calm in [odd, ~odd]:
                m = calm.sum()
                q = np.quantile(sc[calm], min(1, np.ceil((m + 1) * (1 - alpha)) / m), method="higher")
                for i in np.flatnonzero(~calm):
                    cov.append(sc[i] <= q + 1e-12)
                    size.append(max(1, int((PP[i] >= 1 - q - 1e-12).sum())))
            conf[f"{col}_{alpha}"] = dict(cov=float(np.mean(cov)), size=float(np.mean(size)))
        conf[f"{col}_scores"] = sc.tolist()
        conf[f"{col}_posts"] = PP
    res["_conformal_raw"] = conf
    res["conformal"] = {k: v for k, v in conf.items() if not (k.endswith("_scores") or k.endswith("_posts"))}
    results[truth_name] = res
    A.drop(columns=pc_cols).to_parquet(R + f"bench_v2_{model}_{truth_name}.parquet")
    print(truth_name, json.dumps({k: v for k, v in res.items() if k in ["n_sets", "metrics", "ci_top1"]}, default=float)[:3000], flush=True)

# --- conformal transfer across truth sources (calibrate on one, test on the other)
transfer = {}
for cal_t, test_t in [("repl", "mage"), ("mage", "repl")]:
    for col in ["pip", "srfm"]:
        sc_cal = np.array(results[cal_t]["_conformal_raw"][f"{col}_scores"])
        sc_te = np.array(results[test_t]["_conformal_raw"][f"{col}_scores"])
        PP = results[test_t]["_conformal_raw"][f"{col}_posts"]
        for alpha in [0.2, 0.1]:
            m = len(sc_cal)
            q = np.quantile(sc_cal, min(1, np.ceil((m + 1) * (1 - alpha)) / m), method="higher")
            transfer[f"{cal_t}->{test_t}_{col}_{alpha}"] = dict(
                cov=float(np.mean(sc_te <= q + 1e-12)),
                size=float(np.mean([max(1, int((p >= 1 - q - 1e-12).sum())) for p in PP])))
OUT["conformal_transfer"] = transfer
for k in results:
    results[k].pop("_conformal_raw", None)
OUT.update(results)
json.dump(OUT, open(R + f"bench_v2_{model}.json", "w"), indent=1, default=float)
print("TRANSFER", json.dumps(transfer, indent=1))
print("DONE")
