"""Semi-synthetic benchmark with real LD and a real sequence channel.

Uses eQTL credible sets with at least three reporter-tested members. In each set an
emVar is the causal variant, z is simulated from z ~ N(c R e_j, R) with EUR LD among
the tested members and c in {3, 5, 8, 12} (four replicates each), and the real
sequence score of every member (default Enformer) is the sequence channel. A
synthetic Gaussian channel with the same within-set AUC is the theoretical reference.
Single-effect regressions are fitted with uniform, correctly weighted and threefold
overweighted sequence priors. Reports top-1 accuracy by LD between the causal
variant and its nearest competitor, coverage and size of 95% credible sets, and
split-conformal sets (alpha = 0.05) calibrated on half of the credible sets.

Usage: python semisynthetic.py [MODEL]   (after benchmark.py MODEL)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from common import R, load_haplotypes, region_columns, hap_corr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from compass.theory import mu_from_auc  # noqa: E402
from compass.metrics import within_set_auc, credible_set  # noqa: E402

model = sys.argv[1] if len(sys.argv) > 1 else "enformer"
F = pd.read_parquet(R + f"features_{model}.parquet")[["variant", f"{model}_unsup"]].rename(columns={f"{model}_unsup": "seq"})
T = pd.read_parquet(R + "labeled_cs_members.parquet").merge(F, on="variant").dropna(subset=["seq"])
T = T.drop_duplicates(["cs_id", "variant"])
n = T.groupby("cs_id").variant.transform("size")
T = T[n >= 3]
T["seq"] = (T.seq - T.seq.mean()) / T.seq.std()
A_seq = within_set_auc(T.seq, T.emvar.astype(bool), T.cs_id)
mu = max(mu_from_auc(A_seq), 0.0)
print(f"within-set AUC of the {model} score: {A_seq:.3f} (mu = {mu:.3f})")

rng = np.random.default_rng(0)
rows = []
for chrom, Tc in T.groupby("chromosome"):
    idx, H = load_haplotypes(chrom, Tc.position, tmp_tag="semi")
    eur = region_columns(chrom, ["EUR"])["EUR"]
    for cs, g in Tc.groupby("cs_id"):
        g = g[g.variant.isin(idx)]
        if len(g) < 3 or not g.emvar.any():
            continue
        Rm, maf = hap_corr(H, [idx[v] for v in g.variant], eur)
        if not np.isfinite(Rm).all():
            keep = np.isfinite(np.diag(Rm))
            g, Rm = g[keep], Rm[np.ix_(keep, keep)]
            if len(g) < 3 or not g.emvar.any():
                continue
        k = len(g)
        Lc = np.linalg.cholesky(Rm + 1e-4 * np.eye(k))
        s_real = g.seq.values
        causal_pool = np.flatnonzero(g.emvar.values.astype(bool))
        for c in (3, 5, 8, 12):
            for rep in range(4):
                j = rng.choice(causal_pool)
                z = c * Rm[:, j] + Lc @ rng.standard_normal(k)
                Wv = max(c * c, 1.0)
                lbf = 0.5 * z * z * Wv / (1 + Wv)
                s_gauss = rng.standard_normal(k)
                s_gauss[j] += mu
                rj = np.abs(np.delete(Rm[j], j)).max()
                rec = dict(cs_id=cs, chrom=str(chrom), c=c, k=k, rmax=rj)
                for name, lp in [("uniform", 0 * s_real), ("seq", mu * s_real), ("seq_overconf", 3 * mu * s_real),
                                 ("gauss", mu * s_gauss)]:
                    a = lbf + lp
                    pip = np.exp(a - logsumexp(a))
                    cs95 = credible_set(pip, 0.95)
                    rec[f"top1_{name}"] = int(np.argmax(pip) == j)
                    rec[f"cs_cov_{name}"] = int(j in cs95)
                    rec[f"cs_size_{name}"] = len(cs95)
                    rec[f"score_{name}"] = 1 - pip[j]
                    rec[f"pips_{name}"] = pip.astype(np.float32).tobytes()
                rows.append(rec)
D = pd.DataFrame(rows)
print(len(D), "simulations in", D.cs_id.nunique(), "credible sets")
D["ld"] = pd.cut(D.rmax, [0, 0.8, 0.95, 0.99, 1.0001], right=False)
by = D.groupby(["c", "ld"], observed=True)[["top1_uniform", "top1_seq", "top1_gauss", "top1_seq_overconf"]].mean()
by["n"] = D.groupby(["c", "ld"], observed=True).size()
print(by.round(3).to_string())
diff = (by.top1_seq - by.top1_gauss).abs()
out = dict(n=len(D), n_sets=int(D.cs_id.nunique()), auc_seq=A_seq, mu=mu,
           mean_abs_diff_real_vs_gauss=float(diff.mean()), max_abs_diff=float(diff.max()),
           by_c_ld=by.reset_index().astype({"ld": str}).to_dict("records"))
# 95% credible sets and split-conformal sets (calibrate on odd chromosomes, test on even, and the reverse)
odd = D.chrom.str.replace("chr", "").map(lambda x: int(x) % 2 == 1 if x.isdigit() else False).values
for name in ["uniform", "seq", "seq_overconf"]:
    cov, size = [], []
    for cal in (odd, ~odd):
        sc = D[f"score_{name}"].values[cal]
        m = len(sc)
        q = np.quantile(sc, min(1, np.ceil((m + 1) * 0.95) / m), method="higher")
        te = np.flatnonzero(~cal)
        cov += list(D[f"score_{name}"].values[te] <= q + 1e-12)
        size += [max(1, int((np.frombuffer(D[f"pips_{name}"].values[i], np.float32) >= 1 - q - 1e-6).sum())) for i in te]
    out[name] = dict(cs95_cov=float(D[f"cs_cov_{name}"].mean()), cs95_size=float(D[f"cs_size_{name}"].mean()),
                     conformal_cov=float(np.mean(cov)), conformal_size=float(np.mean(size)))
    print(name, out[name])
json.dump(out, open(R + f"semisynthetic_{model}.json", "w"), indent=1, default=float)
