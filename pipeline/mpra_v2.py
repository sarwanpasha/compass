"""Reporter-assay analyses requested in revision.

(1) Within-set AUC of PIP for emVars, by credible-set LD (minimum pairwise r^2 among members).
(2) Between-set view: emVar rate by PIP bin across all reporter-tested credible-set members, relative to
    tested variants that are in no eQTL credible set (enrichment), next to the within-set AUC.
(3) Tested vs untested members: PIP of members that were and were not reporter-tested, and the
    within-set AUC restricted to sets whose tested members carry >= 90% of the set's PIP.
(4) Cell-type matching: GM12878 emVars against lymphoblastoid (LCL) eQTL sets, HepG2 emVars against
    liver eQTL sets, compared with the same cells against unmatched tissues.

Usage: python mpra_v2.py   (after build_sets.py)
"""
import json

import numpy as np
import pandas as pd

from common import R, D


def ws_auc(df, score="pip", y="emvar"):
    num = den = 0.0
    for _, g in df.groupby("cs_id"):
        t = g[y].values.astype(bool)
        if t.all() or not t.any():
            continue
        a, b = g[score].values[t], g[score].values[~t]
        num += (a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()
        den += len(a) * len(b)
    return num / den if den else np.nan, int(df.cs_id.nunique())


def boot_auc(df, B=500, seed=0):
    rng = np.random.default_rng(seed)
    ids = df.cs_id.unique()
    grp = {k: g for k, g in df.groupby("cs_id")}
    v = []
    for _ in range(B):
        s = rng.choice(ids, len(ids))
        v.append(ws_auc(pd.concat([grp[k].assign(cs_id=f"{k}_{i}") for i, k in enumerate(s)]))[0])
    return np.nanpercentile(v, [2.5, 97.5]).tolist()


out = {}
T = pd.read_parquet(R + "labeled_cs_members.parquet")
best = T.drop_duplicates("cs_id").sort_values("sample_size", ascending=False).drop_duplicates("gene_id").cs_id
P = T[T.cs_id.isin(set(best))].copy()
out["all"] = dict(zip(["auc", "n_sets"], ws_auc(P)))
P["ld"] = pd.cut(P.cs_min_r2, [-0.01, 0.2, 0.5, 0.8, 0.95, 1.0])
out["by_ld"] = {}
for k, g in P.groupby("ld", observed=True):
    a, n = ws_auc(g)
    out["by_ld"][str(k)] = dict(auc=a, n_sets=n, ci=boot_auc(g, 300) if n >= 50 else None)
print(json.dumps(out, indent=1), flush=True)

# (2) between-set enrichment
V = pd.read_parquet(R + "mpra_variants_hg38.parquet")
C = pd.read_parquet(D + "eqtlcat/cs_ge_r7.parquet", columns=["variant", "pip", "cs_id"])
vmax = C.groupby("variant").pip.max()
V["max_pip"] = V.variant.map(vmax)
ctrl_rate = float(V[V.max_pip.isna()].emvar.mean())
bins = [0, 0.01, 0.1, 0.5, 0.9, 1.0001]
V["pip_bin"] = pd.cut(V.max_pip, bins, right=False)
tab = V.dropna(subset=["max_pip"]).groupby("pip_bin", observed=True).emvar.agg(["size", "mean"])
tab["enrichment_vs_not_in_cs"] = tab["mean"] / ctrl_rate
out["between_set"] = dict(control_rate=ctrl_rate, n_control=int(V.max_pip.isna().sum()),
                          by_pip=tab.reset_index().astype({"pip_bin": str}).to_dict("records"))
print(tab.round(4).to_string(), "control", round(ctrl_rate, 4), flush=True)

# (3) tested vs untested members
S = pd.read_parquet(R + "cs_tested_status.parquet")
S = S[S.cs_id.isin(set(P.cs_id))]
out["tested_vs_untested"] = dict(
    mean_pip_tested=float(S[S.tested].pip.mean()), mean_pip_untested=float(S[~S.tested].pip.mean()),
    median_frac_pip_tested=float(S.groupby("cs_id").apply(lambda g: g.pip[g.tested].sum() / g.pip.sum()).median()),
    frac_members_tested=float(S.tested.mean()))
fr = S.groupby("cs_id").apply(lambda g: g.pip[g.tested].sum() / g.pip.sum())
P90 = P[P.cs_id.isin(fr[fr >= 0.9].index)]
out["tested_vs_untested"]["auc_sets_90pct_tested"] = dict(zip(["auc", "n_sets"], ws_auc(P90)))
print(json.dumps(out["tested_vs_untested"], indent=1), flush=True)

# (4) cell-type matched
L = pd.read_parquet(R + "mpra_cell_level.parquet")
T2 = T.copy()
T2["tissue"] = T2.tissue_label.astype(str).str.lower()
is_lcl = T2.tissue.str.contains("lymphoblastoid|lcl")
is_liver = T2.tissue.str.contains("liver|hepatocyte")
out["matched"] = {}
for cell, mask, nm in [("GM12878", is_lcl, "LCL"), ("HepG2", is_liver, "liver")]:
    lab = L[L.cell == cell][["variant", "emvar"]].rename(columns={"emvar": "emvar_cell"})
    for tis_nm, m in [(nm, mask), ("other", ~mask & ~is_lcl & ~is_liver)]:
        X = T2[m].drop(columns=["emvar"]).merge(lab, on="variant")
        s = X.groupby("cs_id").agg(n=("emvar_cell", "size"), e=("emvar_cell", "sum"))
        X = X[X.cs_id.isin(s[(s.n >= 2) & (s.e >= 1) & (s.e < s.n)].index)]
        b_ = X.drop_duplicates("cs_id").sort_values("sample_size", ascending=False).drop_duplicates("gene_id").cs_id
        X = X[X.cs_id.isin(set(b_))]
        a, n = ws_auc(X, y="emvar_cell")
        out["matched"][f"{cell}_vs_{tis_nm}"] = dict(auc=a, n_sets=n, ci=boot_auc(X.rename(columns={"emvar_cell": "emvar"}), 200) if n >= 50 else None)
print(json.dumps(out["matched"], indent=1), flush=True)
json.dump(out, open(R + "mpra_v2.json", "w"), indent=1, default=float)
