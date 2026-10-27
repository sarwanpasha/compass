"""Summarize calibration.py output: ECE and Spearman correlation per cohort, LD source and kappa."""
import glob
import json
import os
import sys

import pandas as pd

from common import R

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from compass.metrics import ece  # noqa: E402

T = pd.concat([pd.read_parquet(f) for f in glob.glob(R + "calib/chr*.parquet")], ignore_index=True)
out = {}
for src, g in T.groupby("src"):
    g = g.dropna(subset=["pred_maxpip_insample", "pred_maxpip_eurref"])
    res = dict(n=len(g))
    for col in [c for c in g.columns if c.startswith("pred_maxpip_")]:
        key = col.replace("pred_maxpip_", "")
        res[key] = dict(ece=ece(g[col], g.lead_pip),
                        spearman=float(g[[col, "lead_pip"]].corr("spearman").iloc[0, 1]),
                        mean_pred=float(g[col].mean()), mean_obs=float(g.lead_pip.mean()))
    out[src] = res
    print(src, "n =", res["n"])
    print(pd.DataFrame({k: v for k, v in res.items() if k != "n"}).T.round(3).to_string())
json.dump(out, open(R + "calibration.json", "w"), indent=1)
