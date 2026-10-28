"""Summarize kappa_mechanisms.py: calibration error (ECE) of theory predictions against susieR lead PIPs.

Usage: python simulations/kappa_summary.py 'results/kappa_sim/chr*.parquet' OUT.json
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from compass.metrics import ece  # noqa: E402

D = pd.concat([pd.read_parquet(f) for f in glob.glob(sys.argv[1])], ignore_index=True)
KAPPAS = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]
out = {}
for mech, g in D.groupby("mech"):
    r = dict(n_sets=len(g), mean_obs=float(g.lead_pip.mean()), causal_in_cs=float(g.causal_in_cs.mean()))
    for pre, nm in [("pred", "plugin"), ("predadj", "finite_sample")]:
        e = {k: ece(g[f"{pre}_k{k}"], g.lead_pip) for k in KAPPAS}
        r[nm] = dict(ece=e, best_kappa=min(e, key=e.get), mean_pred_k1=float(g[f"{pre}_k1.0"].mean()))
    for col in ["pred_true_c", "pred_true_c_adj"]:
        gg = g.dropna(subset=[col])
        r[col] = dict(n=len(gg), ece=ece(gg[col], gg.lead_pip), mean_pred=float(gg[col].mean()), mean_obs=float(gg.lead_pip.mean()))
    out[mech] = r
    print(f"{mech:14s} n={r['n_sets']:5d} obs={r['mean_obs']:.3f} | plug-in ECE k=1 {r['plugin']['ece'][1.0]:.3f} best k {r['plugin']['best_kappa']} "
          f"| finite-sample ECE k=1 {r['finite_sample']['ece'][1.0]:.3f} best k {r['finite_sample']['best_kappa']} "
          f"| true c {r['pred_true_c']['ece']:.3f} true c adj {r['pred_true_c_adj']['ece']:.3f}", flush=True)
json.dump(out, open(sys.argv[2], "w"), indent=1)
