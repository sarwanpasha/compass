"""Resolution atlas for one chromosome.

Signals: one credible set per gene (or trait) and lead variant, the set with the largest lead |z|;
SNVs only. For each HGDP+1kG genetic region, the haplotype correlation r between the lead and every
other member is computed. Per signal and region the output records:
  * lead MAF; maximum partner r^2; number of partners with r^2 >= 0.99 and with |r| = 1;
  * the non-centrality c* needed for 90% pairwise accuracy of the |z| rule against the closest partner
    (Theorem 1, sign-agnostic form), the accuracy at the observed c = |z_lead|, and the factor (c*/c)^2
    by which the study would have to grow;
  * panel-size downsampling (25, 50, 100, 200, 400 individuals) of the r^2 >= 0.99 and |r| = 1 flags.
Multi-ancestry resolution: for total sample sizes of 1e5 and 1e6, split equally across AFR, AMR, CSA,
EAS and EUR or placed in EUR alone, the pairwise separation against each partner is
sqrt(sum_p c_p^2 (1 - |r_p|)) with c_p^2 = c^2 (n_p / n_0) (H_p / H_0), H = 2f(1 - f); a signal is
resolved when the |z| rule reaches 90% accuracy against every partner.
Leads are also clumped (EUR r^2 >= 0.5 within 1 Mb) to give locus-level counts.

Usage: python atlas.py CHROM [eqtl|gwas]
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

from common import R, D, MAJOR, REGIONS, load_haplotypes, region_columns

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from compass.theory import expected_max_pip, pairwise_correct_abs, c_required  # noqa: E402

c = sys.argv[1].replace("chr", "")
mode = sys.argv[2] if len(sys.argv) > 2 else "eqtl"
os.makedirs(R + f"atlas_{mode}", exist_ok=True)
rng = np.random.default_rng(int(c) if c.isdigit() else 23)
GWAS_N = 361194
DS = [25, 50, 100, 200, 400]

if mode == "eqtl":
    C = pd.read_parquet(D + "eqtlcat/cs_ge_r7.parquet",
                        columns=["dataset_id", "gene_id", "cs_id", "variant", "chromosome", "position", "ref", "alt", "pip", "z"])
    meta = pd.read_csv(D + "eqtlcat/dataset_metadata_r7.tsv", sep="\t")[["dataset_id", "sample_size"]]
    C = C.merge(meta, on="dataset_id").rename(columns={"gene_id": "unit", "sample_size": "n0"})
else:
    C = pd.read_parquet(R + "gwas_credible_sets.parquet")
    if "n0" not in C:
        C["n0"] = GWAS_N
C = C[(C.chromosome.astype(str).str.replace("chr", "") == c) & (C.ref.str.len() == 1) & (C.alt.str.len() == 1)]
C = C.assign(absz=C.z.abs())
lead = C.sort_values(["cs_id", "absz"], ascending=[True, False]).drop_duplicates("cs_id")
lead = lead.sort_values("absz", ascending=False).drop_duplicates(["unit", "variant"])
C = C[C.cs_id.isin(set(lead.cs_id))]

idx, H = load_haplotypes(c, C.position, tmp_tag=f"atlas_{mode}")
cols = region_columns(c)
ds_cols = {reg: {m: rng.choice(cols[reg], min(m, len(cols[reg])), replace=False) for m in DS} for reg in ["EUR", "AFR"]}

rate_map = None
try:
    import stdpopsim
    rate_map = stdpopsim.get_species("HomSap").get_genetic_map("HapMapII_GRCh38").get_chromosome_map(f"chr{c}")
except Exception as e:
    print("no recombination map:", e)

CSTAR = {}


def cstar(r):
    key = round(float(r), 4)
    if key not in CSTAR:
        CSTAR[key] = c_required(key, 0.9)
    return CSTAR[key]


def corr_to_lead(Hs):
    """Hs: (m, haplotypes) float; returns r of every row with row 0 and MAF of each row."""
    f = Hs.mean(1)
    sd = Hs.std(1)
    Z = (Hs - f[:, None]) / np.where(sd > 0, sd, np.nan)[:, None]
    r = (Z[1:] @ Z[0]) / Hs.shape[1]
    return r, np.minimum(f, 1 - f)


lead_by = lead.set_index("cs_id")
out = []
for cs, g in C.groupby("cs_id"):
    L = lead_by.loc[cs]
    rec = dict(cs_id=cs, unit=L.unit, lead=L.variant, position=int(L.position), lead_z=float(L.absz),
               lead_pip=float(L.pip), n0=float(L.n0), n_members=len(g), lead_in_panel=L.variant in idx)
    others = [v for v in g.variant if v != L.variant and v in idx]
    rec["n_partners_in_panel"] = len(others)
    if rate_map is not None:
        lo, hi = max(0, rec["position"] - 50_000), min(int(rate_map.sequence_length) - 1, rec["position"] + 50_000)
        try:
            m_ = rate_map.get_cumulative_mass([lo, hi])
            rec["recomb_cM_Mb"] = float((m_[1] - m_[0]) / max(hi - lo, 1) * 1e8)
        except Exception:
            rec["recomb_cM_Mb"] = np.nan
    if not rec["lead_in_panel"] or not others:
        out.append(rec)
        continue
    rows = [idx[L.variant]] + [idx[v] for v in others]
    Hsel = H[rows]
    r_reg, het = {}, {}
    for reg in REGIONS:
        Hs = Hsel[:, cols[reg]].reshape(len(rows), -1).astype(np.float32)
        r, maf = corr_to_lead(Hs)
        rec[f"{reg}_lead_maf"] = float(maf[0])
        if maf[0] < 0.01:
            continue
        r = np.nan_to_num(r)
        r2 = r ** 2
        rmax = float(np.abs(r).max())
        rec[f"{reg}_max_r2"] = float(r2.max())
        rec[f"{reg}_n_r2_099"] = int((r2 >= 0.99).sum())
        rec[f"{reg}_n_exact"] = int((np.abs(r) >= 1 - 1e-6).sum())
        rec[f"{reg}_cstar"] = cstar(rmax)
        rec[f"{reg}_acc_now"] = float(pairwise_correct_abs(rec["lead_z"], rmax))
        rec[f"{reg}_n_factor"] = float((rec[f"{reg}_cstar"] / rec["lead_z"]) ** 2) if np.isfinite(rec[f"{reg}_cstar"]) else np.inf
        if reg in MAJOR:
            r_reg[reg] = np.abs(r)
            het[reg] = 2 * maf[0] * (1 - maf[0])
        if reg in ds_cols:
            for m, cc in ds_cols[reg].items():
                rd, md = corr_to_lead(Hsel[:, cc].reshape(len(rows), -1).astype(np.float32))
                if md[0] >= 0.01:
                    rd = np.nan_to_num(rd)
                    rec[f"{reg}_ds{m}_irr"] = bool((rd ** 2 >= 0.99).any())
                    rec[f"{reg}_ds{m}_exact"] = bool((np.abs(rd) >= 1 - 1e-6).any())
        if reg == "EUR":
            Hs_all = Hs
            f = Hs_all.mean(1)
            sd = Hs_all.std(1)
            okm = sd > 0
            Zs = (Hs_all[okm] - f[okm, None]) / sd[okm, None]
            Rs = Zs @ Zs.T / Zs.shape[1]
            if okm[0] and len(Rs) > 1:
                rec["pred_maxpip_EUR"], rec["pred_top1_EUR"] = expected_max_pip(rec["lead_z"], Rs, n_draws=200, rng=rng)
    if len(r_reg) == len(MAJOR):
        Rr = np.clip(np.vstack([r_reg[p] for p in MAJOR]), 0, 1)           # 5 x partners
        rec["pan_irreducible"] = bool((Rr.min(0) ** 2 >= 0.99).any())
        rec["common_all5"] = bool(min(het[p] for p in MAJOR) >= 2 * 0.05 * 0.95)
        H0 = het.get("EUR", np.mean(list(het.values())))
        c2_per_n = rec["lead_z"] ** 2 / rec["n0"] / H0                     # c^2 per sample per unit heterozygosity
        for ntot in (1e5, 1e6):
            for scen, alloc in [("eur", {"EUR": 1.0}), ("all5", {p: 0.2 for p in MAJOR})]:
                s2 = sum(c2_per_n * ntot * w * het[p] * (1 - Rr[MAJOR.index(p)]) for p, w in alloc.items())
                b2 = sum(c2_per_n * ntot * w * het[p] * (1 + Rr[MAJOR.index(p)]) for p, w in alloc.items())
                a_, b_ = np.sqrt(s2 / 2), np.sqrt(b2 / 2)
                acc = norm.cdf(a_) * norm.cdf(b_) + norm.cdf(-a_) * norm.cdf(-b_)
                rec[f"resolved_{scen}_{int(ntot):d}"] = bool(acc.min() >= 0.9)
    out.append(rec)
A = pd.DataFrame(out)

# clump leads by EUR LD (r^2 >= 0.5 within 1 Mb)
cl = np.arange(len(A))
okl = A.lead_in_panel.values & A.lead.isin(idx).values
L_idx = np.flatnonzero(okl)
if len(L_idx):
    Hl = H[[idx[v] for v in A.lead.values[L_idx]]][:, cols["EUR"]].reshape(len(L_idx), -1).astype(np.float32)
    f = Hl.mean(1)
    sd = Hl.std(1)
    Zl = (Hl - f[:, None]) / np.where(sd > 0, sd, np.nan)[:, None]
    pos = A.position.values[L_idx]
    order = np.argsort(pos)
    parent = {i: i for i in L_idx}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a_i in range(len(order)):
        ia = order[a_i]
        b_i = a_i + 1
        while b_i < len(order) and pos[order[b_i]] - pos[ia] <= 1_000_000:
            ib = order[b_i]
            r_ = np.nansum(Zl[ia] * Zl[ib]) / Zl.shape[1]
            if r_ * r_ >= 0.5:
                parent[find(L_idx[ia])] = find(L_idx[ib])
            b_i += 1
    for i in L_idx:
        cl[i] = find(i)
A["clump"] = [f"chr{c}_{x}" for x in cl]
A.to_parquet(R + f"atlas_{mode}/chr{c}.parquet")
print(mode, "chr", c, len(A), "signals", A.clump.nunique(), "clumps")
