"""Summarize the resolution atlas across chromosomes.

Usage: python atlas_summary.py [eqtl|gwas]
Reports, per genetic region, the fraction of signals with a common lead whose lead
has a partner with r^2 >= 0.99; the pan-human irreducible fraction; the fraction
with an exact-LD partner; irreducibility and partner counts by recombination
quintile; and the Spearman correlation between predicted top-1 accuracy (EUR LD)
and the reported lead PIP.
"""
import glob
import json
import sys

import pandas as pd

from common import R, REGIONS

mode = sys.argv[1] if len(sys.argv) > 1 else "eqtl"
A = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(R + f"atlas_{mode}/chr*.parquet"))], ignore_index=True)
out = dict(n_signals=len(A), n_lead_in_panel=int(A.lead_in_panel.sum()),
           n_with_partner=int((A.lead_in_panel & (A.n_partners_in_panel > 0)).sum()))
B = A[A.lead_in_panel & (A.n_partners_in_panel > 0)]
for reg in REGIONS:
    g = B[B.get(f"{reg}_common", False) == True]  # noqa: E712
    if len(g):
        out[reg] = dict(n=len(g), frac_r2_099=float((g[f"{reg}_n_r2_099"] > 0).mean()),
                        frac_exact=float((g[f"{reg}_n_exact"] > 0).mean()),
                        mean_partners_r2_099=float(g[f"{reg}_n_r2_099"].mean()))
P = B.dropna(subset=["pan_irreducible"]) if "pan_irreducible" in B else B.iloc[:0]
out["pan_irreducible"] = dict(n=len(P), frac=float(P.pan_irreducible.astype(bool).mean()) if len(P) else None)
if "recomb_cM_Mb" in B:
    rq = pd.qcut(B.recomb_cM_Mb.rank(method="first"), 5, labels=False)
    for reg in ["EUR", "AFR"]:
        g = B.assign(q=rq)[B.get(f"{reg}_common", False) == True]  # noqa: E712
        t = g.groupby("q").agg(rate=("recomb_cM_Mb", "mean"),
                               irreducible=(f"{reg}_n_r2_099", lambda x: (x > 0).mean()),
                               partners=(f"{reg}_n_r2_099", "mean"))
        out[f"recomb_{reg}"] = t.round(4).to_dict("records")
if "pred_top1_EUR" in B:
    g = B.dropna(subset=["pred_top1_EUR"])
    out["spearman_pred_top1_vs_lead_pip"] = float(g[["pred_top1_EUR", "lead_pip"]].corr("spearman").iloc[0, 1])
print(json.dumps(out, indent=1))
json.dump(out, open(R + f"atlas_{mode}_summary.json", "w"), indent=1)
