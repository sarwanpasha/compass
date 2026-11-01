"""Robustness of the within-set reporter-assay result.

(1) emVar definitions of increasing stringency, from the per-cell DE tables:
      base        active allele (-log10 padj >= 2, log2FC >= 1) and skew -log10 FDR >= 1 (as before)
      fdr01       as base with skew -log10 FDR >= 2
      effect05    fdr01 and |log2 skew| >= 0.5
      effect1     fdr01 and |log2 skew| >= 1
      replicated  base in >= 2 cell lines with the same skew sign
    Negatives are tested members that are not base emVars in any cell line; members that pass base but
    not the stricter rule are excluded from that analysis.
(2) Chromatin context: emVars and negatives restricted to members inside ENCODE candidate cis-regulatory
    elements (any cell type) and, for matched pairs, inside DNase hypersensitive sites of the matched
    cell line (GM12878 for lymphoblastoid eQTL sets, HepG2 for liver sets).
(3) Cell-type-matched analysis with all lymphoblastoid credible sets (every dataset, not one per gene) and
    a gene-level cluster bootstrap; strict labels in GM12878.
(4) Between-set precision: among tested members with PIP >= 0.5 and < 0.1, the rate of emVars inside
    regulatory elements (the quantity reported by the assay's authors) next to the within-set AUC.
(5) Haplotype concordance restricted to pairs of variants that are both effect05 emVars in the same cell
    line, with alleles aligned by the sign of EUR LD.

Usage: python mpra_v3.py   (after build_sets.py; ENCODE beds under data/encode/)
"""
import glob
import json
import os
import re
from itertools import combinations

import numpy as np
import pandas as pd

from common import R, D, load_haplotypes, region_columns, hap_corr

out = {}


def ws_auc(df, y="emvar", score="pip"):
    num = den = 0.0
    n = 0
    for _, g in df.groupby("cs_id"):
        t = g[y].values.astype(bool)
        if t.all() or not t.any():
            continue
        a, b = g[score].values[t], g[score].values[~t]
        num += (a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()
        den += len(a) * len(b)
        n += 1
    return (num / den if den else np.nan), n


def cluster_boot(df, y="emvar", B=300, cluster="gene_id", seed=0):
    rng = np.random.default_rng(seed)
    ids = df[cluster].unique()
    grp = {k: g for k, g in df.groupby(cluster)}
    v = []
    for _ in range(B):
        s = rng.choice(ids, len(ids))
        v.append(ws_auc(pd.concat([grp[k].assign(cs_id=grp[k].cs_id + f"_{i}") for i, k in enumerate(s)]), y)[0])
    return np.nanpercentile(v, [2.5, 97.5]).tolist()


# ---------------------------------------------------------------- per-cell labels (hg19 -> hg38 via build_sets)
V = pd.read_parquet(R + "mpra_variants_hg38.parquet")          # variant, chr, pos (hg19), ref, alt, emvar
rows = []
for f in sorted(glob.glob(D + "mpra/data/MPRA/processed_results/DE/*emVAR_glm_paired*.out.gz")):
    lib, cell = re.search(r"DE/(OL[0-9A-Z_]+?)_([A-Za-z0-9]+)_emVAR", f).groups()
    d = pd.read_csv(f, sep="\t", usecols=["chr", "pos", "ref_allele", "alt_allele", "window", "A_logPadj_BH", "B_logPadj_BH",
                                         "A_log2FC", "B_log2FC", "Log2Skew", "Skew_logFDR_act"], low_memory=False)
    d = d[d.window == "center"]
    d["cell"] = cell
    rows.append(d)
L = pd.concat(rows)
L["chr"] = L.chr.astype(str).str.replace("chr", "")
L = L.rename(columns={"ref_allele": "ref", "alt_allele": "alt"})
L = L.groupby(["chr", "pos", "ref", "alt", "cell"]).agg(A=("A_logPadj_BH", "max"), B=("B_logPadj_BH", "max"),
                                                         fcA=("A_log2FC", "max"), fcB=("B_log2FC", "max"),
                                                         skew=("Log2Skew", "mean"), sfa=("Skew_logFDR_act", "max")).reset_index()
act = ((L.A >= 2) & (L.fcA >= 1)) | ((L.B >= 2) & (L.fcB >= 1))
L["base"] = act & (L.sfa >= 1)
L["fdr01"] = act & (L.sfa >= 2)
L["effect05"] = L.fdr01 & (L["skew"].abs() >= 0.5)
L["effect1"] = L.fdr01 & (L["skew"].abs() >= 1)
L = L.merge(V[["chr", "pos", "ref", "alt", "variant"]].assign(chr=V.chr.astype(str), pos=V.pos.astype(int)),
            on=["chr", "pos", "ref", "alt"])
agg = L.groupby("variant").agg(base=("base", "max"), fdr01=("fdr01", "max"), effect05=("effect05", "max"),
                               effect1=("effect1", "max"))
sg = L[L.base].groupby("variant")["skew"].agg(lambda s: (np.sign(s) > 0).sum() >= 2 or (np.sign(s) < 0).sum() >= 2)
agg["replicated"] = agg.index.map(sg).fillna(False).astype(bool)
out["n_emvars"] = {k: int(agg[k].sum()) for k in agg.columns}
out["n_tested"] = int(len(agg))
print(out, flush=True)

T = pd.read_parquet(R + "labeled_cs_members.parquet").drop(columns=["emvar"])
T = T.merge(agg, left_on="variant", right_index=True)
best = T.drop_duplicates("cs_id").sort_values("sample_size", ascending=False).drop_duplicates("gene_id").cs_id
P = T[T.cs_id.isin(set(best))].copy()


def labelled(df, rule):
    """Positives: rule emVars. Negatives: not base emVars. Others removed."""
    x = df[df[rule] | ~df.base].copy()
    x["y"] = x[rule]
    return x


out["stringency"] = {}
for rule in ["base", "fdr01", "effect05", "effect1", "replicated"]:
    X = labelled(P, rule)
    a, n = ws_auc(X, "y")
    out["stringency"][rule] = dict(auc=a, n_sets=n, ci=cluster_boot(X, "y", 200) if n >= 50 else None)
    X["ld"] = pd.cut(X.cs_min_r2, [-0.01, 0.5, 0.8, 1.0])
    out["stringency"][rule]["by_ld"] = {str(k): dict(zip(["auc", "n_sets"], ws_auc(g, "y"))) for k, g in X.groupby("ld", observed=True)}
print(json.dumps(out["stringency"], indent=1), flush=True)


# ---------------------------------------------------------------- chromatin context
def read_bed(path):
    b = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2], names=["c", "s", "e"], comment="#")
    return {c: (g.s.values, g.e.values) for c, g in b.sort_values(["c", "s"]).groupby("c")}


def in_bed(variants, bed):
    ch = variants.str.split("_").str[0]
    pos = variants.str.split("_").str[1].astype(int)
    hit = np.zeros(len(variants), bool)
    for c in ch.unique():
        if c not in bed:
            continue
        s, e = bed[c]
        m = (ch == c).values
        p = pos.values[m]
        i = np.searchsorted(s, p, side="right") - 1
        ok = i >= 0
        hit_m = np.zeros(m.sum(), bool)
        hit_m[ok] = p[ok] <= e[i[ok]]
        hit[m] = hit_m
    return hit


beds = {k: D + f"encode/{k}.bed.gz" for k in ["ccre", "dnase_GM12878", "dnase_HepG2"]}
beds = {k: read_bed(v) for k, v in beds.items() if os.path.exists(v)}
out["chromatin"] = {}
if "ccre" in beds:
    P["in_ccre"] = in_bed(P.variant, beds["ccre"])
    out["chromatin"]["frac_members_in_ccre"] = float(P.in_ccre.mean())
    for rule in ["base", "effect05"]:
        X = labelled(P[P.in_ccre], rule)
        a, n = ws_auc(X, "y")
        out["chromatin"][f"ccre_{rule}"] = dict(auc=a, n_sets=n, ci=cluster_boot(X, "y", 200) if n >= 50 else None)
        # emVars inside cCREs against all other members
        Y = P.copy()
        Y["y"] = Y[rule] & Y.in_ccre
        Y = Y[Y.y | ~Y.base]
        a, n = ws_auc(Y, "y")
        out["chromatin"][f"emvar_in_ccre_vs_rest_{rule}"] = dict(auc=a, n_sets=n)
print(json.dumps(out["chromatin"], indent=1), flush=True)

# ---------------------------------------------------------------- matched cell type, all sets, cluster bootstrap
Lc = L.groupby(["variant", "cell"]).agg(base=("base", "max"), effect05=("effect05", "max"), skew=("skew", "mean")).reset_index()
T["tissue"] = T.tissue_label.astype(str).str.lower()
out["matched_all_sets"] = {}
for cell, pat, bedk in [("GM12878", "lymphoblastoid|lcl", "dnase_GM12878"), ("HepG2", "liver|hepatocyte", "dnase_HepG2")]:
    lab = Lc[Lc.cell == cell][["variant", "base", "effect05"]].rename(columns={"base": "b_c", "effect05": "e_c"})
    X = T[T.tissue.str.contains(pat)].drop(columns=["base", "effect05", "fdr01", "effect1", "replicated"]).merge(lab, on="variant")
    for rule, col in [("base", "b_c"), ("effect05", "e_c")]:
        Y = X[X[col] | ~X.b_c].copy()
        Y["y"] = Y[col]
        a, n = ws_auc(Y, "y")
        res = dict(auc=a, n_sets=n, n_genes=int(Y.gene_id.nunique()), ci=cluster_boot(Y, "y", 300) if n >= 30 else None)
        if bedk in beds:
            Z = Y[in_bed(Y.variant, beds[bedk])]
            a2, n2 = ws_auc(Z, "y")
            res["in_dhs"] = dict(auc=a2, n_sets=n2, ci=cluster_boot(Z, "y", 300) if n2 >= 30 else None)
        out["matched_all_sets"][f"{cell}_{rule}"] = res
print(json.dumps(out["matched_all_sets"], indent=1), flush=True)

# ---------------------------------------------------------------- between-set precision
S = P.copy()
if "in_ccre" in S:
    for rule in ["base", "effect05"]:
        S["y"] = S[rule] & S.in_ccre
        out.setdefault("precision", {})[rule] = dict(
            pip_ge_05=float(S[S.pip >= 0.5].y.mean()), pip_lt_01=float(S[S.pip < 0.1].y.mean()),
            n_ge_05=int((S.pip >= 0.5).sum()), n_lt_01=int((S.pip < 0.1).sum()))
    print(json.dumps(out.get("precision"), indent=1), flush=True)

# ---------------------------------------------------------------- haplotype concordance for strong emVars in the same cell
E = Lc[Lc.effect05 & (Lc["skew"] != 0)]
M = T[["cs_id", "variant", "chromosome", "position"]].drop_duplicates().merge(E, on="variant")
pairs = []
for chrom, Mc in M.groupby("chromosome"):
    idx, H = load_haplotypes(chrom, Mc.position, tmp_tag="dir3")
    eur = region_columns(chrom, ["EUR"])["EUR"]
    for (cs, cell), g in Mc.groupby(["cs_id", "cell"]):
        g = g[g.variant.isin(idx)].drop_duplicates("variant")
        if len(g) < 2:
            continue
        Rm, _ = hap_corr(H, [idx[v] for v in g.variant], eur)
        sk = np.sign(g["skew"].values)
        for a, b in combinations(range(len(g)), 2):
            if np.isfinite(Rm[a, b]) and Rm[a, b] != 0:
                pairs.append((g.variant.iloc[a], g.variant.iloc[b], cell, Rm[a, b], sk[a], sk[b]))
Q = pd.DataFrame(pairs, columns=["v1", "v2", "cell", "r", "s1", "s2"]).drop_duplicates(["v1", "v2", "cell"])
if len(Q):
    Q["conc"] = np.sign(Q.r) * Q.s1 * Q.s2 > 0
    tight = Q[Q.r.abs() >= 0.8]
    rng = np.random.default_rng(0)
    sv = pd.concat([Q[["v1", "s1"]].set_axis(["v", "s"], axis=1), Q[["v2", "s2"]].set_axis(["v", "s"], axis=1)]).drop_duplicates("v")
    null = []
    for _ in range(1000):
        perm = dict(zip(sv.v, rng.permutation(sv.s.values)))
        null.append(np.mean(np.sign(tight.r) * tight.v1.map(perm) * tight.v2.map(perm) > 0))
    null = np.array(null)
    out["concordance_strong_same_cell"] = dict(
        n_pairs=len(Q), n_tight=len(tight), concordance=float(tight.conc.mean()), null_mean=float(null.mean()),
        p=float((np.abs(null - null.mean()) >= abs(tight.conc.mean() - null.mean())).mean()),
        by_cell={c: dict(n=len(g), conc=float(g.conc.mean())) for c, g in tight.groupby("cell")})
    print(json.dumps(out["concordance_strong_same_cell"], indent=1), flush=True)
json.dump(out, open(R + "mpra_v3.json", "w"), indent=1, default=float)
