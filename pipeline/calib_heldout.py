"""Held-out evaluation of the effective-resolution factor kappa.

kappa is chosen to minimize the calibration error (ECE) on odd chromosomes and evaluated on even
chromosomes, and the reverse, for the real cohorts (calib/chr*.parquet) and for each simulated
mechanism (kappa_sim/chr*.parquet). Reports the held-out ECE at the selected kappa next to the
held-out ECE at kappa = 1.

Usage: python calib_heldout.py
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

from common import R

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from compass.metrics import ece  # noqa: E402


def load(pattern):
    parts = []
    for f in glob.glob(pattern):
        m = re.search(r"chr(\w+)\.parquet", f)
        d = pd.read_parquet(f)
        d["chrom_num"] = int(m.group(1)) if m and m.group(1).isdigit() else -1
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def heldout(g, cols, obs="lead_pip"):
    """cols: dict kappa -> prediction column."""
    res = []
    for train_odd in (True, False):
        tr = g[(g.chrom_num % 2 == 1) == train_odd]
        te = g[(g.chrom_num % 2 == 1) != train_odd]
        e_tr = {k: ece(tr[c], tr[obs]) for k, c in cols.items()}
        kbest = min(e_tr, key=e_tr.get)
        res.append(dict(kappa=kbest, ece_heldout=ece(te[cols[kbest]], te[obs]), ece_heldout_k1=ece(te[cols[1.0]], te[obs])))
    return dict(kappa=[r["kappa"] for r in res], ece_heldout=float(np.mean([r["ece_heldout"] for r in res])),
                ece_heldout_k1=float(np.mean([r["ece_heldout_k1"] for r in res])))


out = {}
T = load(R + "calib/chr*.parquet")
for src, g in T.groupby("src"):
    cols = {1.0: "pred_maxpip_insample_adj"}
    for c in g.columns:
        m = re.match(r"pred_maxpip_insample_adj_k([0-9.]+)$", c)
        if m:
            cols[float(m.group(1))] = c
    g = g.dropna(subset=list(cols.values()))
    out[src] = heldout(g, cols)
S = load(R + "kappa_sim/chr*.parquet")
for mech, g in S.groupby("mech"):
    cols = {float(re.search(r"k([0-9.]+)$", c).group(1)): c for c in g.columns if re.match(r"predadj_k[0-9.]+$", c)}
    out[f"sim_{mech}"] = heldout(g.dropna(subset=list(cols.values())), cols)
print(json.dumps(out, indent=1))
json.dump(out, open(R + "calib_heldout.json", "w"), indent=1)
