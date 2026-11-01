"""Two linked causal variants in real LD (Theorem 4).

Windows of 150 consecutive common EUR SNVs (MAF >= 5%) are taken from HGDP+1kG, and their haplotype
correlation matrix R is computed from all unrelated EUR individuals. In each window, pairs of causal
variants with 0.3 <= r < 0.8 are drawn and given equal same-sign non-centralities c in {5, 10, 20, 40}
(a 64-fold range of sample size). For each configuration the script records
  * whether a phantom exists: a non-causal variant whose expected z exceeds that of both causal
    variants (Theorem 4);
  * for 200 draws z ~ N(R (c e_i + c e_j), R): the frequency with which argmax |z| is a causal variant
    and with which it is a phantom, and, for 10 random pairs of members, the simulated and predicted
    probabilities that one ranks above the other;
  * for 20 draws, SuSiE-RSS (compass.susie_rss, L = 5, n = 10,000 scaled to c) top-PIP calls: causal,
    phantom or other, and whether a 95% credible set contains a phantom but no causal variant.

Usage: python simulations/two_causal.py CHROM N_WINDOWS OUT.parquet
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))
from compass.design import mean_z, rank_prob, phantom_variants  # noqa: E402
from compass.susie import susie_rss  # noqa: E402
from common import bcf_path, region_columns  # noqa: E402

chrom, n_win, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
c_ = chrom.replace("chr", "")
rng = np.random.default_rng(int(c_) if c_.isdigit() else 0)
eur = region_columns(c_, ["EUR"])["EUR"]
P, CS = 150, [5.0, 10.0, 20.0, 40.0]


def window(start):
    txt = subprocess.run(["bcftools", "query", "-r", f"chr{c_}:{start}-{start + 2_000_000}", "-i", 'TYPE="snp" && N_ALT=1',
                          "-f", "%POS[\t%GT]\n", bcf_path(c_)], capture_output=True, text=True).stdout
    rows = []
    for line in txt.splitlines():
        f = line.split("\t")
        g = np.array("\t".join(f[1:]).replace("|", "\t").replace("/", "\t").replace(".", "0").split("\t"),
                     dtype=np.int8).reshape(-1, 2)[eur].ravel()
        m = g.mean()
        if 0.05 <= m <= 0.95:
            rows.append(g)
        if len(rows) == P:
            break
    if len(rows) < P:
        return None
    X = np.stack(rows).astype(np.float32)
    Z = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
    return Z @ Z.T / Z.shape[1]


GRCH38 = [248956422, 242193529, 198295559, 190214555, 181538259, 170805979, 159345973, 145138636, 138394717, 133797422,
          135086622, 133275309, 114364328, 107043718, 101991189, 90338345, 83257441, 80373285, 58617616, 64444167,
          46709983, 50818468]
length = GRCH38[int(c_) - 1] if c_.isdigit() else 50_000_000
recs, pairs = [], []
for w in range(n_win):
    R = window(int(rng.integers(1_000_000, max(2_000_000, length - 3_000_000))))
    if R is None:
        continue
    L = np.linalg.cholesky(R + 1e-4 * np.eye(P))
    iu = np.argwhere((np.triu(R, 1) >= 0.3) & (np.triu(R, 1) < 0.8))
    if not len(iu):
        continue
    for i, j in iu[rng.choice(len(iu), min(5, len(iu)), replace=False)]:
        ph = phantom_variants(R, [i, j], [1.0, 1.0])
        for c in CS:
            eta = mean_z(R, [i, j], [c, c])
            z = eta + (L @ rng.standard_normal((P, 200))).T
            top = np.argmax(np.abs(z), 1)
            rec = dict(window=w, i=int(i), j=int(j), r12=float(R[i, j]), c=c, phantom=len(ph) > 0, n_phantom=len(ph),
                       top_causal=float(np.isin(top, [i, j]).mean()), top_phantom=float(np.isin(top, ph).mean()))
            for _ in range(10):
                u, v = rng.choice(P, 2, replace=False)
                pairs.append(dict(c=c, sim=float((z[:, u] > z[:, v]).mean()), pred=float(rank_prob(eta[u], eta[v], R[u, v]))))
            n = 10_000
            calls = []
            for d in range(20):
                fit = susie_rss(z[d], R, n, L=5)
                pip = fit["pip"]
                tp = int(np.argmax(pip))
                cs_ph = any(np.isin(s["members"], ph).any() and not np.isin(s["members"], [i, j]).any() for s in fit["sets"])
                calls.append((tp in (i, j), tp in set(ph), cs_ph))
            rec["susie_top_causal"] = float(np.mean([a for a, _, _ in calls]))
            rec["susie_top_phantom"] = float(np.mean([b for _, b, _ in calls]))
            rec["susie_cs_phantom_only"] = float(np.mean([x for _, _, x in calls]))
            recs.append(rec)
    print("window", w, len(recs), flush=True)
pd.DataFrame(recs).to_parquet(out)
pd.DataFrame(pairs).to_parquet(out.replace(".parquet", "_pairs.parquet"))
