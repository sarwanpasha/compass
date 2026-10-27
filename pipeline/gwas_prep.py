"""Format UK Biobank 94-trait SuSiE credible sets (GRCh37) for atlas.py.

Input: data/ukbb/UKBB_94traits_release1.bed.gz (release 1.1 of the fine-mapping resource).
Output: results/gwas_credible_sets.parquet with GRCh38 coordinates.
"""
import numpy as np
import pandas as pd
from pyliftover import LiftOver

from common import R, D

U = pd.read_csv(D + "ukbb/UKBB_94traits_release1.bed.gz", sep="\t", low_memory=False)
U.columns = [c.lstrip("#") for c in U.columns]
U = U[(U.method == "SUSIE") & (U.cs_id >= 0)].copy()
U["chrom"] = U.chromosome.astype(str).str.replace("chr", "")
lo = LiftOver(D + "ref/hg19ToHg38.over.chain.gz")
pos38 = []
for ch, p in zip(U.chrom, U.end):
    r = lo.convert_coordinate("chr" + ch, int(p) - 1)
    pos38.append(r[0][1] + 1 if r and r[0][0] == "chr" + ch else -1)
U["position"] = pos38
U = U[U.position > 0]
U["ref"], U["alt"] = U.allele1, U.allele2
U["variant"] = "chr" + U.chrom + "_" + U.position.astype(str) + "_" + U.ref + "_" + U.alt
U["z"] = U.beta_marginal / U.se_marginal
U["cs_id"] = U.trait + ":" + U.region.astype(str) + ":" + U.cs_id.astype(str)
U["unit"] = U.trait
U["chromosome"] = U.chrom
U[["unit", "cs_id", "variant", "chromosome", "position", "ref", "alt", "pip", "z"]].to_parquet(R + "gwas_credible_sets.parquet")
print(U.cs_id.nunique(), "credible sets;", len(U), "rows;", np.round(U.pip.mean(), 3), "mean PIP")
