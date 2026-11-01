"""Per-locus study design for the resolution atlas (one chromosome).

Uses the same signals as atlas.py (one credible set per gene or trait and lead SNV). For each signal
it records, from HGDP+1kG haplotypes:
  * the selection-corrected non-centrality: conditional maximum likelihood given |z| >= t, with t the
    smallest lead |z| reported by the same dataset (eQTL) or 5.45 (GWAS, P < 5e-8), and the
    method-of-moments value that corrects for taking the maximum over correlated candidates;
  * the design class of compass.design.design_signal (resolved, more samples of the discovery
    population, another ancestry, very large studies, functional evidence only) with and without
    the selection correction, the fold increase needed with EUR samples alone and with the optimal
    allocation across AFR, AMR, CSA, EAS and EUR, the allocation shares, and the within-set AUC a
    sequence model would need to resolve the signal now;
  * multi-ancestry resolution at 1e5 and 1e6 individuals (equal split) under heterogeneous effects
    across populations (multiplicative factor 1 + e, e ~ N(0, tau^2), tau = 0.25 and 0.5), and when
    the design uses LD from 100 reference individuals instead of the full panel;
  * whether the EUR credible set contains a phantom-prone configuration (a non-causal member whose
    expected z exceeds both members of a weakly linked causal pair; Theorem 4);
  * for every 10th signal, partners within 250 kb of the lead outside the credible set (all panel
    SNVs with MAF >= 1% in the region), which removes the dependence on EUR-defined sets;
  * for every 5th signal, the r^2 >= 0.99 flag in 30 bootstrap resamples of individuals per region.

Usage: python atlas_design.py CHROM [eqtl|gwas]
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

from common import R, D, MAJOR, REGIONS, load_haplotypes, region_columns, bcf_path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from compass.design import design_signal, phantom_prone  # noqa: E402
from compass.theory import c_debiased  # noqa: E402

c = sys.argv[1].replace("chr", "")
mode = sys.argv[2] if len(sys.argv) > 2 else "eqtl"
LIGHT = os.environ.get("ATLAS_LIGHT") == "1"      # design columns only (skip bootstrap, windows, max correction)
OUTDIR = R + (f"atlas_design_{mode}_light" if LIGHT else f"atlas_design_{mode}")
os.makedirs(OUTDIR, exist_ok=True)
rng = np.random.default_rng(1000 + (int(c) if c.isdigit() else 23))
GWAS_N, GWAS_T = 361194, 5.45
B, WIN = 30, 250_000

if mode == "eqtl":
    C = pd.read_parquet(D + "eqtlcat/cs_ge_r7.parquet",
                        columns=["dataset_id", "gene_id", "cs_id", "variant", "chromosome", "position", "ref", "alt", "pip", "z"])
    meta = pd.read_csv(D + "eqtlcat/dataset_metadata_r7.tsv", sep="\t")[["dataset_id", "sample_size"]]
    C = C.merge(meta, on="dataset_id").rename(columns={"gene_id": "unit", "sample_size": "n0"})
    C["absz"] = C.z.abs()
    lead_all = C.sort_values(["cs_id", "absz"], ascending=[True, False]).drop_duplicates("cs_id")
    tmin = lead_all.groupby("dataset_id").absz.min()          # smallest reported lead |z| per dataset
    C["t_sel"] = C.dataset_id.map(tmin)
else:
    C = pd.read_parquet(R + "gwas_credible_sets.parquet")
    if "n0" not in C:
        C["n0"] = GWAS_N
    C["absz"] = C.z.abs()
    C["t_sel"] = GWAS_T
C = C[(C.chromosome.astype(str).str.replace("chr", "") == c) & (C.ref.str.len() == 1) & (C.alt.str.len() == 1)]
lead = C.sort_values(["cs_id", "absz"], ascending=[True, False]).drop_duplicates("cs_id")
lead = lead.sort_values("absz", ascending=False).drop_duplicates(["unit", "variant"])
C = C[C.cs_id.isin(set(lead.cs_id))]

idx, H = load_haplotypes(c, C.position, tmp_tag=f"design_{mode}")
cols = region_columns(c)
ref100 = {p: rng.choice(cols[p], min(100, len(cols[p])), replace=False) for p in MAJOR}
boot = {p: [rng.choice(cols[p], len(cols[p]), replace=True) for _ in range(B)] for p in REGIONS}


def corr_rows(X):
    """Correlation of every row of X (m x haplotypes) with row 0; MAF of every row."""
    X = X.astype(np.float32)
    f = X.mean(1)
    sd = X.std(1)
    Z = (X - f[:, None]) / np.where(sd > 0, sd, np.nan)[:, None]
    return np.nan_to_num((Z[1:] @ Z[0]) / X.shape[1]), np.minimum(f, 1 - f)


def window_haplotypes(pos):
    lo, hi = max(1, pos - WIN), pos + WIN
    txt = subprocess.run(["bcftools", "query", "-r", f"chr{c}:{lo}-{hi}", "-i", 'TYPE="snp" && N_ALT=1',
                          "-f", "%POS\t%REF\t%ALT[\t%GT]\n", bcf_path(c)], capture_output=True, text=True).stdout
    keys, rows = [], []
    for line in txt.splitlines():
        f = line.split("\t")
        gt = "\t".join(f[3:]).replace("|", "\t").replace("/", "\t").replace(".", "0")
        keys.append(f"chr{c}_{f[0]}_{f[1]}_{f[2]}")
        rows.append(np.array(gt.split("\t"), dtype=np.int8).reshape(-1, 2))
    return keys, (np.stack(rows) if rows else None)


def resolved(c2n, het, r, ntot, mult=None):
    """Equal split over MAJOR populations; Bayes-rule accuracy >= 0.9 against every partner."""
    S = 0.0
    for i, p in enumerate(MAJOR):
        m = 1.0 if mult is None else mult[i] ** 2
        S = S + c2n * m * (ntot / len(MAJOR)) * het[p] * (1 - r[p])
    return bool(norm.cdf(np.sqrt(np.min(S) / 2)) >= 0.9)


lead_by = lead.set_index("cs_id")
out = []
for k_sig, (cs, g) in enumerate(C.groupby("cs_id")):
    L = lead_by.loc[cs]
    if L.variant not in idx:
        continue
    others = [v for v in g.variant if v != L.variant and v in idx]
    if not others:
        continue
    rows = [idx[L.variant]] + [idx[v] for v in others]
    Hs = H[rows]
    rec = dict(cs_id=cs, unit=L.unit, lead=L.variant, lead_z=float(L.absz), n0=float(L.n0), t_sel=float(L.t_sel),
               n_members=len(g))
    r, het = {}, {}
    for p in REGIONS:
        rr, maf = corr_rows(Hs[:, cols[p]].reshape(len(rows), -1))
        if maf[0] < 0.01:
            continue
        r[p], het[p] = np.round(np.clip(np.abs(rr), 0, 1), 4), 2 * maf[0] * (1 - maf[0])   # |r| to 4 decimals, as in atlas.py
        if k_sig % 5 == 0 and not LIGHT:                          # bootstrap over individuals
            fl = []
            for bc in boot[p]:
                rb, mb = corr_rows(Hs[:, bc].reshape(len(rows), -1))
                fl.append(bool((rb ** 2 >= 0.99).any()) if mb[0] >= 0.01 else np.nan)
            rec[f"{p}_boot"] = fl
    if "EUR" not in r:
        continue
    rec["EUR_r2_099"] = bool((r["EUR"] ** 2 >= 0.99).any())
    # selection-corrected non-centrality
    X = Hs[:, cols["EUR"]].reshape(len(rows), -1).astype(np.float32)
    sd = X.std(1)
    ok = sd > 0
    Z = (X[ok] - X[ok].mean(1, keepdims=True)) / sd[ok, None]
    Reur = Z @ Z.T / Z.shape[1]
    if len(Reur) <= 60 and not LIGHT:
        rec["c_max_corrected"] = c_debiased(rec["lead_z"], Reur, n_draws=200, rng=rng)
        rec["phantom_prone"] = phantom_prone(Reur)
    have5 = all(p in r for p in MAJOR)
    if have5:
        rM = {p: r[p] for p in MAJOR}
        hM = {p: het[p] for p in MAJOR}
        for tag, thr, kap in [("raw", None, 1.0), ("sel", rec["t_sel"], 1.0), ("selk05", rec["t_sel"], 0.5),
                              ("selk03", rec["t_sel"], 0.3)]:
            rK = {p: 1 - kap * (1 - rM[p]) for p in MAJOR}        # effective resolution (Section 3.7)
            d = design_signal(rec["lead_z"], rec["n0"], hM, rK, disc="EUR", z_threshold=thr)
            for kk, vv in d.items():
                rec[f"{tag}_{kk}"] = vv
        c2n = rec["lead_z"] ** 2 / rec["n0"] / het["EUR"]
        for ntot in (1e5, 1e6):
            rec[f"res_all5_{int(ntot)}"] = resolved(c2n, hM, rM, ntot)
            for tau in (0.25, 0.5):
                rec[f"res_all5_{int(ntot)}_het{tau}"] = float(np.mean(
                    [resolved(c2n, hM, rM, ntot, mult=1 + tau * rng.standard_normal(5)) for _ in range(10)]))
            # reference-LD mismatch: design with 100 reference individuals per population
            r100 = {}
            for p in MAJOR:
                rb, mb = corr_rows(Hs[:, ref100[p]].reshape(len(rows), -1))
                r100[p] = np.round(np.clip(np.abs(rb), 0, 1), 4)
            rec[f"res_all5_{int(ntot)}_ref100"] = resolved(c2n, hM, r100, ntot)
    if k_sig % 10 == 0 and not LIGHT:                             # credible-set-independent partners
        keys, Hw = window_haplotypes(int(L.variant.split("_")[1]))
        if Hw is not None and L.variant in keys:
            j = keys.index(L.variant)
            order = [j] + [i for i in range(len(keys)) if i != j]
            members = set(g.variant)
            outside = np.array([keys[i] not in members for i in order[1:]])
            for p in MAJOR + ["MID"]:
                rw, mw = corr_rows(Hw[order][:, cols[p]].reshape(len(order), -1))
                if mw[0] < 0.01:
                    continue
                okm = mw[1:] >= 0.01
                hit = (rw ** 2 >= 0.99) & okm
                rec[f"{p}_win_any"] = bool(hit.any())
                rec[f"{p}_win_outside"] = bool((hit & outside).any())
                rec[f"{p}_win_n"] = int(hit.sum())
    out.append(rec)
    if k_sig % 1000 == 0:
        print(k_sig, len(out), flush=True)

pd.DataFrame(out).to_parquet(OUTDIR + f"/chr{c}.parquet")
print(mode, "chr", c, len(out), "signals")
