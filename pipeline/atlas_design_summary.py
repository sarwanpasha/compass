"""Summarize atlas_design.py across chromosomes.

Usage: python atlas_design_summary.py [eqtl|gwas]
"""
import glob
import json
import sys

import numpy as np
import pandas as pd

from common import R, MAJOR, REGIONS

mode = sys.argv[1] if len(sys.argv) > 1 else "eqtl"
A = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(R + f"atlas_design_{mode}/chr*.parquet"))], ignore_index=True)
_light = sorted(glob.glob(R + f"atlas_design_{mode}_light/chr*.parquet"))
if _light:                                                  # design columns recomputed with |r| rounded to 4 decimals
    Lt = pd.concat([pd.read_parquet(f) for f in _light], ignore_index=True)
    keep = [c for c in Lt.columns if c.startswith(("raw_", "sel", "res_all5")) or c == "cs_id"]
    A = A.drop(columns=[c for c in keep if c != "cs_id" and c in A]).merge(Lt[keep], on="cs_id", how="left")
out = dict(n_signals=len(A))
CLS = ["resolved", "more_samples", "ancestry", "large", "functional"]


def classes(g, tag):
    col = f"{tag}_cls"
    g = g.dropna(subset=[col])
    return dict(n=len(g), **{k: float((g[col] == k).mean()) for k in CLS})


def med(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else None


hi = A[A.EUR_r2_099 == True]  # noqa: E712
for tag in ["raw", "sel", "selk05", "selk03"]:
    if f"{tag}_cls" not in A:
        continue
    out[f"classes_{tag}"] = classes(A, tag)
    out[f"classes_{tag}_r2_099"] = classes(hi, tag)
    unres = A[A[f"{tag}_cls"] != "resolved"]
    out[f"median_fold_eur_{tag}"] = med(unres[f"{tag}_fold_disc"])
    out[f"median_fold_opt_{tag}"] = med(unres[f"{tag}_fold_opt"])
    uh = hi[hi[f"{tag}_cls"] != "resolved"]
    out[f"median_fold_eur_{tag}_r2_099"] = med(uh[f"{tag}_fold_disc"])
    out[f"median_fold_opt_{tag}_r2_099"] = med(uh[f"{tag}_fold_opt"])
    an = A[A[f"{tag}_cls"] == "ancestry"]
    sh = [f"{tag}_share_{p}" for p in MAJOR if f"{tag}_share_{p}" in A]
    if len(an) and sh:
        top = an[sh].idxmax(axis=1).str.replace(f"{tag}_share_", "")
        out[f"ancestry_top_population_{tag}"] = top.value_counts(normalize=True).round(4).to_dict()
        out[f"ancestry_mixed_{tag}"] = float((an[sh].max(axis=1) < 0.9).mean())
    au = unres[f"{tag}_auc_needed"]
    out[f"auc_needed_{tag}"] = {f"le_{a}": float((au <= a).mean()) for a in [0.6, 0.65, 0.7, 0.8, 0.9]}
    out[f"auc_needed_{tag}"]["median"] = med(au)
# selection correction of c
for col in ["sel_c", "c_max_corrected"]:
    if col in A:
        ratio = A[col] / A.lead_z
        out[f"{col}_over_z"] = dict(median=med(ratio), q10=float(ratio.quantile(0.1)), q90=float(ratio.quantile(0.9)))
if "c_max_corrected" in A:
    g = hi.dropna(subset=["c_max_corrected"])
    out["median_nfactor_increase_maxcorr_r2_099"] = med((g.lead_z / g.c_max_corrected.clip(lower=1e-6)) ** 2)
# multi-ancestry sensitivity
for ntot in ["100000", "1000000"]:
    base = f"res_all5_{ntot}"
    if base in A:
        g = A.dropna(subset=[base])
        out[f"resolved_all5_{ntot}"] = dict(
            full=float(g[base].astype(float).mean()),
            het025=float(g[f"{base}_het0.25"].mean()), het05=float(g[f"{base}_het0.5"].mean()),
            ref100=float(g[f"{base}_ref100"].astype(float).mean()),
            ref100_false_resolved=float((g[f"{base}_ref100"].astype(bool) & ~g[base].astype(bool)).mean()))
if "phantom_prone" in A:
    out["phantom_prone"] = dict(n=int(A.phantom_prone.notna().sum()), frac=float(A.phantom_prone.dropna().astype(bool).mean()),
                                frac_r2_099=float(hi.phantom_prone.dropna().astype(bool).mean()))
# credible-set-independent partners within 250 kb
win = {}
for p in MAJOR + ["MID"]:
    c1 = f"{p}_win_any"
    if c1 in A:
        g = A.dropna(subset=[c1])
        win[p] = dict(n=len(g), any=float(g[c1].astype(bool).mean()), outside=float(g[f"{p}_win_outside"].astype(bool).mean()),
                      median_n=med(g[f"{p}_win_n"]))
out["window_250kb"] = win
# bootstrap confidence intervals over individuals for the r^2 >= 0.99 fraction
ci = {}
for p in REGIONS:
    c1 = f"{p}_boot"
    if c1 not in A:
        continue
    M = np.array([x for x in A[c1].dropna()], dtype=float)
    if not len(M):
        continue
    fr = np.nanmean(M, axis=0)
    ci[p] = dict(n=len(M), mean=float(np.mean(fr)), lo=float(np.percentile(fr, 2.5)), hi=float(np.percentile(fr, 97.5)))
out["bootstrap_r2_099"] = ci
print(json.dumps(out, indent=1))
json.dump(out, open(R + f"atlas_design_{mode}_summary.json", "w"), indent=1)
