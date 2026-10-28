"""Shared paths and helpers for the real-data pipeline.

All inputs and outputs live under one working directory, set with the
environment variable COMPASS_DIR (default: ./work). Layout:
    $COMPASS_DIR/data/{eqtlcat,ref,mpra,hgdp1kg,ukbb,models}
    $COMPASS_DIR/results/
"""
import os
import subprocess

import numpy as np
import pandas as pd

WORK = os.path.abspath(os.environ.get("COMPASS_DIR", "work"))
D = os.path.join(WORK, "data") + os.sep
R = os.path.join(WORK, "results") + os.sep
os.makedirs(R, exist_ok=True)

REGIONS = ["AFR", "AMR", "CSA", "EAS", "EUR", "MID"]
MAJOR = ["AFR", "AMR", "CSA", "EAS", "EUR"]


def bcf_path(chrom):
    c = str(chrom).replace("chr", "")
    return D + f"hgdp1kg/hgdp1kgp_chr{c}.filtered.SNV_INDEL.phased.shapeit5.bcf"


def sample_table():
    """Unrelated HGDP+1kG samples with genetic region (written by hgdp_samples.py)."""
    return pd.read_csv(D + "hgdp1kg/samples_unrelated_pop.tsv", sep="\t")


def bcf_samples(chrom):
    out = subprocess.run(["bcftools", "query", "-l", bcf_path(chrom)], capture_output=True, text=True, check=True)
    return out.stdout.split()


def load_haplotypes(chrom, positions, tmp_tag="q"):
    """Phased haplotypes of biallelic SNVs at the given GRCh38 positions.

    Returns (index, H) where index maps 'chrC_POS_REF_ALT' to a row of H and
    H has shape (n_variants, n_samples, 2) with int8 alleles.
    """
    c = str(chrom).replace("chr", "")
    if not os.path.exists(bcf_path(c)):                      # e.g. chrX: no phased panel available
        return {}, np.zeros((0, 0, 2), np.int8)
    reg = os.path.join(R, f"tmp_{tmp_tag}_chr{c}.tsv")
    pd.DataFrame({"c": f"chr{c}", "p": sorted(set(int(p) for p in positions))}).to_csv(
        reg, sep="\t", header=False, index=False)
    txt = subprocess.run(["bcftools", "query", "-R", reg, "-i", 'TYPE="snp" && N_ALT=1',
                          "-f", "%POS\t%REF\t%ALT[\t%GT]\n", bcf_path(c)],
                         capture_output=True, text=True, check=True).stdout
    os.remove(reg)
    index, rows = {}, []
    for line in txt.splitlines():
        f = line.split("\t")
        key = f"chr{c}_{f[0]}_{f[1]}_{f[2]}"
        if key in index:
            continue
        gt = "\t".join(f[3:]).replace("|", "\t").replace("/", "\t").replace(".", "0")
        index[key] = len(rows)
        rows.append(np.array(gt.split("\t"), dtype=np.int8).reshape(-1, 2))
    H = np.stack(rows) if rows else np.zeros((0, 0, 2), np.int8)
    return index, H


def region_columns(chrom, regions=REGIONS):
    """Column indices of unrelated samples from each genetic region in the BCF."""
    if not os.path.exists(bcf_path(chrom)):
        return {g: np.array([], int) for g in regions}
    meta = sample_table()
    pos = {s: i for i, s in enumerate(bcf_samples(chrom))}
    return {g: np.array([pos[s] for s in meta[meta.region == g].s if s in pos]) for g in regions}


def hap_corr(H, rows, cols):
    """Haplotype correlation matrix among `rows` using samples `cols`.

    Returns (R, maf); entries for monomorphic variants are NaN.
    """
    X = H[rows][:, cols].reshape(len(rows), -1).astype(np.float32)
    f = X.mean(1)
    sd = X.std(1)
    Z = np.where(sd[:, None] > 0, (X - f[:, None]) / np.where(sd > 0, sd, 1)[:, None], np.nan)
    Rm = Z @ Z.T / Z.shape[1]
    return Rm, np.minimum(f, 1 - f)
