"""Summarize two_causal.py: phantom prevalence, ranking and SuSiE calls by non-centrality.

Usage: python simulations/two_causal_summary.py 'results/two_causal/chr*.parquet' OUT.json
"""
import glob
import json
import sys

import numpy as np
import pandas as pd

files = [f for f in glob.glob(sys.argv[1]) if not f.endswith("_pairs.parquet")]
D = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
Pp = pd.concat([pd.read_parquet(f.replace(".parquet", "_pairs.parquet")) for f in files], ignore_index=True)
cfg = D.drop_duplicates(["window", "i", "j", "r12"])
out = dict(n_configs=len(cfg), frac_phantom=float(cfg.phantom.mean()),
           pairwise=dict(n=len(Pp), r2=float(np.corrcoef(Pp.sim, Pp.pred)[0, 1] ** 2), max_abs_err=float((Pp.sim - Pp.pred).abs().max()),
                         mean_abs_err=float((Pp.sim - Pp.pred).abs().mean())))
cols = ["top_causal", "top_phantom", "susie_top_causal", "susie_top_phantom", "susie_cs_phantom_only"]
out["by_c"] = {f"{ph}": D[D.phantom == ph].groupby("c")[cols].mean().round(4).reset_index().to_dict("records") for ph in [True, False]}
r12b = pd.cut(cfg.r12, [0.3, 0.5, 0.65, 0.8])
out["phantom_by_r12"] = cfg.groupby(r12b, observed=True).phantom.mean().round(4).astype(float).rename(lambda x: str(x)).to_dict()
print(json.dumps(out, indent=1))
json.dump(out, open(sys.argv[2], "w"), indent=1)
