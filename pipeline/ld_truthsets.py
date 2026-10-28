"""LD between the causal (truth) variant and the other members of each benchmark credible set.

For every set in the replication and multi-ancestry truths, computes haplotype r^2 in HGDP+1kG EUR
and AFR between each truth variant and every other SNV member. A set is EUR-irreducible when a truth
variant has a partner with r^2 >= 0.99 in EUR. Output: results/truthsets_ld.parquet

Usage: python ld_truthsets.py
"""
import numpy as np
import pandas as pd

from common import R, load_haplotypes, region_columns, hap_corr

A = pd.concat([pd.read_parquet(R + "truth_repl_fullsets.parquet").assign(src="repl"),
               pd.read_parquet(R + "truth_mage_fullsets.parquet").assign(src="mage")])
A = A[(A.ref.str.len() == 1) & (A.alt.str.len() == 1)]
rows = []
for chrom, Ac in A.groupby(A.chromosome.astype(str)):
    idx, H = load_haplotypes(chrom, Ac.position, tmp_tag="truthld")
    cols = region_columns(chrom, ["EUR", "AFR"])
    for cs, g in Ac.groupby("cs_id"):
        g = g[g.variant.isin(idx)].drop_duplicates("variant")
        if len(g) < 2 or not g.truth.any():
            continue
        rec = dict(cs_id=cs, src=g.src.iloc[0], n_in_panel=len(g))
        rows_ = [idx[v] for v in g.variant]
        tmask = g.truth.values.astype(bool)
        for reg in ["EUR", "AFR"]:
            Rm, maf = hap_corr(H, rows_, cols[reg])
            r2 = np.nan_to_num(Rm ** 2)
            np.fill_diagonal(r2, 0)
            rec[f"truth_max_r2_{reg.lower()}"] = float(r2[tmask].max())
            rec[f"truth_maf_{reg.lower()}"] = float(maf[tmask].max())
        rec["eur_irreducible"] = rec["truth_max_r2_eur"] >= 0.99
        rec["afr_irreducible"] = rec["truth_max_r2_afr"] >= 0.99
        rows.append(rec)
D = pd.DataFrame(rows)
D.to_parquet(R + "truthsets_ld.parquet")
print(D.groupby("src")[["eur_irreducible", "afr_irreducible"]].mean().round(3).to_string(), len(D))
