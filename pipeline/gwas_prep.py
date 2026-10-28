"""Format UK Biobank 94-trait SuSiE credible sets (GRCh37) for atlas.py.

Input: data/ukbb/UKBB_94traits_release1.bed.gz (release 1.1 of the fine-mapping resource).
Output: results/gwas_credible_sets.parquet with GRCh38 coordinates.
"""
import numpy as np
import pandas as pd
from pyliftover import LiftOver

from common import R, D

import glob
bed = glob.glob(D + "ukbb/**/UKBB_94traits_release1.bed.gz", recursive=True)[0]
names = [l.split()[0].lstrip("#") for l in open(bed.replace(".bed.gz", ".cols")) if l.strip()]
U = pd.read_csv(bed, sep="\t", header=None, low_memory=False)
U.columns = names[:U.shape[1]] if len(names) >= U.shape[1] else (
    "chromosome start end variant rsid allele1 allele2 minorallele cohort model_marginal method trait region maf "
    "beta_marginal se_marginal chisq_marginal pip cs_id beta_posterior sd_posterior LD_HWE LD_SV").split()
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
tr = pd.read_csv(bed.replace(".bed.gz", ".traits"), sep="\t")
U["n0"] = U.trait.map(dict(zip(tr.trait, tr.n))).fillna(361194).astype(float)
U["unit"] = U.trait
U["chromosome"] = U.chrom
U[["unit", "cs_id", "variant", "chromosome", "position", "ref", "alt", "pip", "z", "n0"]].to_parquet(R + "gwas_credible_sets.parquet")
print(U.cs_id.nunique(), "credible sets;", len(U), "rows;", np.round(U.pip.mean(), 3), "mean PIP")
