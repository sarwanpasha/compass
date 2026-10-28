"""Calibration of the resolution theory against SuSiE, with in-sample and reference LD.

GEUVADIS (eQTL Catalogue r7) and MAGE (r8 beta, QTD000764) donors are 1000 Genomes
samples, so LD can be computed from the same populations as each study. For every
credible set with 2 to 60 members, the expected maximum PIP of a single-effect
regression (c = |z_lead|) is computed by Monte Carlo with
  (i) in-sample LD (GEUVADIS: CEU, FIN, GBR, TSI, YRI; MAGE: all 1000 Genomes samples),
  (ii) HGDP+1kG EUR reference LD,
and, for each, the effective-resolution transform R -> 11' - kappa (11' - R).

Usage: python calibration.py CHROM
"""
import os
import sys

import numpy as np
import pandas as pd

from common import R, D, load_haplotypes, sample_table, bcf_samples

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from compass.theory import expected_max_pip, kappa_transform, c_debiased  # noqa: E402

KAPPAS = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]
N_DRAWS = 300
c = sys.argv[1].replace("chr", "")
os.makedirs(R + "calib", exist_ok=True)

meta = sample_table()
pos_in = {s: i for i, s in enumerate(bcf_samples(c))}
groups = {
    "geuv_insample": meta[(meta.project == "1000 Genomes") & meta.population.isin(["CEU", "FIN", "GBR", "TSI", "YRI"])].s,
    "mage_insample": meta[meta.project == "1000 Genomes"].s,
    "eur_ref": meta[meta.region == "EUR"].s,
}
cols = {k: np.array([pos_in[s] for s in v if s in pos_in]) for k, v in groups.items()}

COLS = ["dataset_id", "cs_id", "variant", "chromosome", "pip", "z", "cs_size"]
C7 = pd.read_parquet(D + "eqtlcat/cs_ge_r7.parquet", columns=COLS)
M7 = pd.read_csv(D + "eqtlcat/dataset_metadata_r7.tsv", sep="\t")
geu = set(M7[M7.study_label == "GEUVADIS"].dataset_id)
A = C7[C7.dataset_id.isin(geu) & (C7.chromosome.astype(str) == c)].assign(src="geuv")
B = pd.read_parquet(D + "eqtlcat/cs_ge_r8b.parquet", columns=COLS)
B = B[(B.dataset_id == "QTD000764") & (B.chromosome.astype(str) == c)].assign(src="mage")
M8 = pd.read_csv(D + "eqtlcat/dataset_metadata_r8_beta.tsv", sep="\t")
NSIZE = {**dict(zip(M7.dataset_id, M7.sample_size)), **dict(zip(M8.dataset_id, M8.sample_size))}
X = pd.concat([A, B])
X = X[X.cs_size.between(2, 60)]

idx, H = load_haplotypes(c, [int(v.split("_")[1]) for v in X.variant], tmp_tag="cal")
rng = np.random.default_rng(int(c) if c.isdigit() else 23)
out = []
for (src, cs), g in X.groupby(["src", "cs_id"]):
    g = g[g.variant.isin(idx)].sort_values("pip", ascending=False)
    if len(g) < 2:
        continue
    vs = list(g.variant)
    zz = abs(g.z.iloc[0])
    nn = float(NSIZE.get(g.dataset_id.iloc[0], np.nan))
    zadj = zz * np.sqrt((nn - 1) / (zz * zz + nn - 2)) if np.isfinite(nn) else zz
    rec = dict(src=src, cs_id=cs, k=len(vs), lead_pip=float(g.pip.iloc[0]), lead_z=float(zz), n=nn, lead_z_adj=float(zadj))
    for grp in ([f"{src}_insample", "eur_ref"]):
        Hs = H[[idx[v] for v in vs]][:, cols[grp]].reshape(len(vs), -1).astype(np.float32)
        sd = Hs.std(1)
        if (sd == 0).any():
            continue
        Zs = (Hs - Hs.mean(1, keepdims=True)) / sd[:, None]
        Rm = Zs @ Zs.T / Zs.shape[1]
        key = "insample" if grp != "eur_ref" else "eurref"
        rec[f"rmax_{key}"] = float(np.abs(Rm[0, 1:]).max())
        cdb = c_debiased(zz, Rm, rng=rng)
        rec[f"c_debiased_{key}"] = cdb
        rec[f"pred_maxpip_{key}_debiased"] = expected_max_pip(cdb, Rm, n_draws=N_DRAWS, rng=rng, prior_var=max(zz * zz, 1.0))[0]
        for kap in KAPPAS:
            Rk = Rm if kap == 1.0 else kappa_transform(Rm, kap)
            mp, t1 = expected_max_pip(zz, Rk, n_draws=N_DRAWS, rng=rng)
            sfx = "" if kap == 1.0 else f"_k{kap}"
            rec[f"pred_maxpip_{key}{sfx}"] = mp
            rec[f"pred_top1_{key}{sfx}"] = t1
            rec[f"pred_maxpip_{key}_adj{sfx}"] = expected_max_pip(zadj, Rk, n_draws=N_DRAWS, rng=rng, prior_var=max(zadj * zadj, 1.0))[0]
    out.append(rec)
pd.DataFrame(out).to_parquet(R + f"calib/chr{c}.parquet")
print("chr", c, len(out))
