"""Resolution atlas for one chromosome.

For each signal (one credible set per gene or trait and lead variant, the set with
the largest lead |z|; SNVs only) and each HGDP+1kG genetic region, compute the
haplotype correlation between the lead and every other member present in the
panel. Records, per region: whether the lead is common (MAF >= 1%), the maximum
partner r^2, the number of partners with r^2 >= 0.99 and with |r| = 1. A signal
is pan-human irreducible when one partner has r^2 >= 0.99 with the lead in each of
AFR, AMR, CSA, EAS and EUR. Also records the local recombination rate (mean
HapMap rate within 50 kb of the lead) and the theory-predicted top-1 accuracy of
a single-effect regression using EUR LD and c = |z_lead|.

Usage: python atlas.py CHROM [eqtl|gwas]
Input for gwas: results/gwas_credible_sets.parquet from gwas_prep.py.
"""
import os
import sys

import numpy as np
import pandas as pd

from common import R, D, MAJOR, REGIONS, load_haplotypes, region_columns, hap_corr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from compass.theory import expected_max_pip  # noqa: E402

c = sys.argv[1].replace("chr", "")
mode = sys.argv[2] if len(sys.argv) > 2 else "eqtl"
os.makedirs(R + f"atlas_{mode}", exist_ok=True)
rng = np.random.default_rng(int(c) if c.isdigit() else 23)

if mode == "eqtl":
    C = pd.read_parquet(D + "eqtlcat/cs_ge_r7.parquet",
                        columns=["gene_id", "cs_id", "variant", "chromosome", "position", "ref", "alt", "pip", "z"])
    C = C.rename(columns={"gene_id": "unit"})
else:
    C = pd.read_parquet(R + "gwas_credible_sets.parquet")
C = C[(C.chromosome.astype(str).str.replace("chr", "") == c) & (C.ref.str.len() == 1) & (C.alt.str.len() == 1)]
C = C.assign(absz=C.z.abs())
lead = C.sort_values(["cs_id", "absz"], ascending=[True, False]).drop_duplicates("cs_id")
lead = lead.sort_values("absz", ascending=False).drop_duplicates(["unit", "variant"])
C = C[C.cs_id.isin(set(lead.cs_id))]

idx, H = load_haplotypes(c, C.position, tmp_tag=f"atlas_{mode}")
cols = region_columns(c)

# recombination map (HapMap II, GRCh38, as distributed with stdpopsim)
rate_map = None
try:
    import stdpopsim
    gm = stdpopsim.get_species("HomSap").get_genetic_map("HapMapII_GRCh38")
    rate_map = gm.get_chromosome_map(f"chr{c}")
except Exception as e:  # map is optional
    print("no recombination map:", e)

lead_by = lead.set_index("cs_id")
out = []
for cs, g in C.groupby("cs_id"):
    L = lead_by.loc[cs]
    rec = dict(cs_id=cs, unit=L.unit, lead=L.variant, lead_z=float(L.absz), lead_pip=float(L.pip),
               n_members=len(g), lead_in_panel=L.variant in idx)
    others = [v for v in g.variant if v != L.variant and v in idx]
    rec["n_partners_in_panel"] = len(others)
    if rate_map is not None:
        p = int(L.position)
        lo, hi = max(0, p - 50_000), min(int(rate_map.sequence_length), p + 50_000)
        try:
            mass = rate_map.get_cumulative_mass([lo, hi])
            rec["recomb_cM_Mb"] = float((mass[1] - mass[0]) / max(hi - lo, 1) * 1e8)
        except Exception:
            rec["recomb_cM_Mb"] = np.nan
    if not rec["lead_in_panel"] or not others:
        out.append(rec)
        continue
    rows = [idx[L.variant]] + [idx[v] for v in others]
    r2_major = []
    for reg in REGIONS:
        Rm, maf = hap_corr(H, rows, cols[reg])
        r = Rm[0, 1:]
        common = maf[0] >= 0.01
        rec[f"{reg}_common"] = bool(common)
        if common:
            r2 = np.nan_to_num(r ** 2)
            rec[f"{reg}_max_r2"] = float(r2.max())
            rec[f"{reg}_n_r2_099"] = int((r2 >= 0.99).sum())
            rec[f"{reg}_n_exact"] = int((np.abs(np.nan_to_num(r)) >= 1 - 1e-6).sum())
            if reg in MAJOR:
                r2_major.append(r2)
        if reg == "EUR" and common:
            members = [0] + [i + 1 for i in range(len(others)) if np.isfinite(Rm[0, i + 1])]
            Rs = np.nan_to_num(Rm[np.ix_(members, members)])
            np.fill_diagonal(Rs, 1.0)
            if len(members) > 1:
                rec["pred_maxpip_EUR"], rec["pred_top1_EUR"] = expected_max_pip(L.absz, Rs, n_draws=200, rng=rng)
    if len(r2_major) == len(MAJOR):
        rec["pan_irreducible"] = bool((np.min(np.vstack(r2_major), 0) >= 0.99).any())
    out.append(rec)
pd.DataFrame(out).to_parquet(R + f"atlas_{mode}/chr{c}.parquet")
print(mode, "chr", c, len(out), "signals")
