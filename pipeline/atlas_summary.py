"""Summarize the resolution atlas across chromosomes.

Usage: python atlas_summary.py [eqtl|gwas]
Reports, per genetic region:
  * fraction of signals (lead MAF >= 1% in that region) whose lead has a partner with r^2 >= 0.99 and |r| = 1;
  * c-dependent resolution: fraction below 90% pairwise accuracy at the observed |z_lead|, and fractions that
    would need a >= 10-fold or >= 100-fold larger sample, or are not resolvable at any sample size (|r| = 1);
the same restricted to leads with MAF >= 5% in all five major groups (common denominator); pan-human
irreducible fractions at signal and LD-clump level; multi-ancestry resolution at 1e5 and 1e6; downsampling
curves of the r^2 >= 0.99 and exact-LD fractions against panel size; recombination quintiles; and the
Spearman correlation between predicted top-1 accuracy (EUR LD) and the reported lead PIP.
"""
import glob
import json
import sys

import numpy as np
import pandas as pd

from common import R, REGIONS, MAJOR

mode = sys.argv[1] if len(sys.argv) > 1 else "eqtl"
A = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(R + f"atlas_{mode}/chr*.parquet"))], ignore_index=True)
B = A[A.lead_in_panel & (A.n_partners_in_panel > 0)].copy()
out = dict(n_signals=len(A), n_lead_in_panel=int(A.lead_in_panel.sum()), n_with_partner=len(B),
           n_clumps=int(B.clump.nunique()))


def region_stats(g, reg):
    g = g[g[f"{reg}_max_r2"].notna()]
    if not len(g):
        return None
    nf = g[f"{reg}_n_factor"]
    return dict(n=len(g), frac_r2_099=float((g[f"{reg}_n_r2_099"] > 0).mean()),
                frac_exact=float((g[f"{reg}_n_exact"] > 0).mean()),
                frac_below90_now=float((g[f"{reg}_acc_now"] < 0.9).mean()),
                frac_need_10x=float((nf >= 10).mean()), frac_need_100x=float((nf >= 100).mean()),
                frac_never=float(np.isinf(nf).mean()),
                median_n_factor_r2_099=float(nf[g[f"{reg}_n_r2_099"] > 0].replace(np.inf, np.nan).median()))


out["by_region"] = {reg: region_stats(B, reg) for reg in REGIONS}
if "common_all5" in B:
    Cm = B[B.common_all5 == True]  # noqa: E712
    out["common_denominator"] = dict(n=len(Cm), **{reg: region_stats(Cm, reg) for reg in MAJOR})
P = B.dropna(subset=["pan_irreducible"])
out["pan_irreducible"] = dict(n=len(P), frac=float(P.pan_irreducible.astype(bool).mean()))
# locus level: a clump counts as irreducible when all its signals are
Pc = P.groupby("clump").pan_irreducible.agg(lambda x: bool(np.all(x.astype(bool))))
Pc_any = P.groupby("clump").pan_irreducible.agg(lambda x: bool(np.any(x.astype(bool))))
out["pan_irreducible_clumps"] = dict(n=len(Pc), frac_all=float(Pc.mean()), frac_any=float(Pc_any.mean()))
for reg in ["EUR", "AFR"]:
    g = B[B[f"{reg}_max_r2"].notna()]
    gc = g.groupby("clump").agg(irr=(f"{reg}_n_r2_099", lambda x: bool((x > 0).all())),
                                below=(f"{reg}_acc_now", lambda x: bool((x < 0.9).all())))
    out[f"clump_{reg}"] = dict(n=len(gc), frac_r2_099=float(gc.irr.mean()), frac_below90_now=float(gc.below.mean()))
res_cols = [c for c in B.columns if c.startswith("resolved_")]
out["multiancestry"] = {c: float(P[c].astype(float).mean()) for c in res_cols if c in P}
out["downsampling"] = {}
for reg in ["EUR", "AFR"]:
    curve = []
    for m in [25, 50, 100, 200, 400]:
        cc = f"{reg}_ds{m}_irr"
        if cc in B:
            g = B[B[cc].notna()]
            curve.append(dict(n_individuals=m, frac_r2_099=float(g[cc].astype(bool).mean()),
                              frac_exact=float(g[f"{reg}_ds{m}_exact"].astype(bool).mean()), n=len(g)))
    full = out["by_region"][reg]
    if full:
        curve.append(dict(n_individuals="all", frac_r2_099=full["frac_r2_099"], frac_exact=full["frac_exact"], n=full["n"]))
    out["downsampling"][reg] = curve
if "recomb_cM_Mb" in B:
    rq = pd.qcut(B.recomb_cM_Mb.rank(method="first"), 5, labels=False)
    for reg in ["EUR", "AFR"]:
        g = B.assign(q=rq)
        g = g[g[f"{reg}_max_r2"].notna()]
        t = g.groupby("q").agg(rate=("recomb_cM_Mb", "mean"),
                               irreducible=(f"{reg}_n_r2_099", lambda x: (x > 0).mean()),
                               partners=(f"{reg}_n_r2_099", "mean"),
                               below90=(f"{reg}_acc_now", lambda x: (x < 0.9).mean()))
        out[f"recomb_{reg}"] = t.round(4).to_dict("records")
# Corollary 1: effective sample-size gain of a sequence model with within-set AUC 0.65 against the closest partner
from scipy.stats import norm
mu2 = (np.sqrt(2) * norm.ppf(0.65)) ** 2
out["neff_auc065_median"] = {}
for reg in ["EUR", "EAS", "AFR"]:
    g = B[B[f"{reg}_max_r2"].notna()]
    r = np.sqrt(np.clip(g[f"{reg}_max_r2"], 0, 1))
    ratio = 1 + mu2 / (g.lead_z ** 2 * np.clip(1 - r, 1e-12, None))
    out["neff_auc065_median"][reg] = float(np.median(ratio))
if "pred_top1_EUR" in B:
    g = B.dropna(subset=["pred_top1_EUR"])
    out["spearman_pred_top1_vs_lead_pip"] = float(g[["pred_top1_EUR", "lead_pip"]].corr("spearman").iloc[0, 1])
print(json.dumps(out, indent=1))
json.dump(out, open(R + f"atlas_{mode}_summary.json", "w"), indent=1)
