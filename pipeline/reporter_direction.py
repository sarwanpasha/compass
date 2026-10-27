"""Direction of reporter-assay effects on eQTL haplotypes.

(a) For emVars in eQTL credible sets, the fraction whose allelic skew agrees in
    sign with the eQTL z-score, for PIP > 0.5 and PIP < 0.1 (two-sided binomial tests).
(b) For pairs of emVars in the same credible set, haplotype concordance: whether
    the alleles carried on the same haplotype (aligned by the sign of EUR LD) shift
    reporter activity in the same direction, by LD bin, with a permutation null
    that shuffles skew directions across emVars.

Usage: python reporter_direction.py   (after build_sets.py and hgdp_samples.py)
"""
import json
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from common import R, load_haplotypes, region_columns, hap_corr

V = pd.read_parquet(R + "mpra_variants_hg38.parquet")
T = pd.read_parquet(R + "labeled_cs_members.parquet")
E = T.merge(V[["variant", "skew"]], on="variant")
E = E[E.emvar.astype(bool) & E.skew.notna() & (E.skew != 0) & E.z.notna()]
out = {}

E["agree"] = np.sign(E.skew) == np.sign(E.z)
for name, m in [("pip_gt_0.5", E.pip > 0.5), ("pip_lt_0.1", E.pip < 0.1)]:
    g = E[m].drop_duplicates(["cs_id", "variant"])
    k, n = int(g.agree.sum()), len(g)
    out[name] = dict(n=n, agree=k / n, p=binomtest(k, n, 0.5).pvalue)
print(json.dumps(out, indent=1))

pairs = []
for chrom, Ec in E.groupby("chromosome"):
    idx, H = load_haplotypes(chrom, Ec.position, tmp_tag="dir")
    eur = region_columns(chrom, ["EUR"])["EUR"]
    for cs, g in Ec.groupby("cs_id"):
        g = g[g.variant.isin(idx)].drop_duplicates("variant")
        if len(g) < 2:
            continue
        Rm, _ = hap_corr(H, [idx[v] for v in g.variant], eur)
        sk = np.sign(g.skew.values)
        for a, b in combinations(range(len(g)), 2):
            if np.isfinite(Rm[a, b]) and Rm[a, b] != 0:
                pairs.append((cs, g.variant.iloc[a], g.variant.iloc[b], Rm[a, b], sk[a], sk[b]))
P = pd.DataFrame(pairs, columns=["cs_id", "v1", "v2", "r", "s1", "s2"])
P["concordant"] = np.sign(P.r) * P.s1 * P.s2 > 0
P["absr"] = P.r.abs()
P["bin"] = pd.cut(P.absr, [0, 0.2, 0.5, 0.8, 0.95, 1.0001], right=False)
out["pairs"] = len(P)
tight = P[P.absr >= 0.8]
obs = float(tight.concordant.mean())
rng = np.random.default_rng(0)
signs = dict(zip(E.variant, np.sign(E.skew)))
keys = np.array(list(signs))
null = []
for _ in range(1000):
    perm = dict(zip(keys, rng.permutation(np.array(list(signs.values())))))
    null.append(np.mean(np.sign(tight.r) * tight.v1.map(perm) * tight.v2.map(perm) > 0))
null = np.array(null)
out["tight_ld"] = dict(n=len(tight), concordance=obs, null_mean=float(null.mean()),
                       p=float((np.abs(null - null.mean()) >= abs(obs - null.mean())).mean()))
out["by_ld"] = P.groupby("bin", observed=True).concordant.agg(["size", "mean"]).reset_index().astype({"bin": str}).to_dict("records")
print(json.dumps({k: v for k, v in out.items() if k != "by_ld"}, indent=1))
json.dump(out, open(R + "reporter_direction.json", "w"), indent=1, default=float)
