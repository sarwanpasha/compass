"""Summarize individual-level simulations: accuracy by LD, calibration and conformal sets."""
import pandas as pd, numpy as np, json, argparse
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from compass.theory import Phi, phi
ap = argparse.ArgumentParser(); ap.add_argument('--loci', default='results/individual_level_loci.pkl'); ap.add_argument('--out', default='results/individual_level_summary.json'); args = ap.parse_args()
df = pd.read_pickle(args.loci)
out = {}
print('N', len(df), 'top1 data', df.top0.mean().round(3), 'seq-prior', df.top1.mean().round(3), 'overconf prior', df.top2.mean().round(3))
df['rbin'] = pd.cut(df.rmax, [0, .8, .9, .95, .99, .999, 1.0001], right=False)
g = df.groupby('rbin', observed=True).agg(n=('top0', 'size'), acc_data=('top0', 'mean'), pred_data=('pred_eq', 'mean'),
                                          acc_comb=('top1', 'mean'), pred_comb=('pred_eq_comb', 'mean'),
                                          cs0=('cs0_size', 'mean'), cs1=('cs1_size', 'mean'),
                                          cov0=('cs0_cov', 'mean'), cov1=('cs1_cov', 'mean'), cov2=('cs2_cov', 'mean'))
g['gain'] = g.acc_comb - g.acc_data; g['pred_gain'] = g.pred_comb - g.pred_data
print(g.round(3).to_string())
out['by_ld'] = g.reset_index().astype({'rbin': str}).to_dict('records')
for col in ['pred_eq', 'cert_mc', 'maxpip0']:
    b = pd.qcut(df[col].rank(method='first'), 10, labels=False)
    t = df.groupby(b).agg(pred=(col, 'mean'), obs=('top0', 'mean'))
    print(col, 'ECE=%.3f' % np.average((t.pred - t.obs).abs()))
    out['ece_' + col] = float(np.average((t.pred - t.obs).abs()))


def lac(v, alpha, cal, te, strata=None):
    """Split-conformal threshold sets {j: a_j >= 1-q}; score = 1 - a_causal. Mondrian if strata given."""
    sc = 1 - df['pip%s_causal' % v].values
    cov, size = [], []
    groups = [None] if strata is None else np.unique(strata)
    for gval in groups:
        c_ = cal if gval is None else cal[strata[cal] == gval]
        t_ = te if gval is None else te[strata[te] == gval]
        m = len(c_)
        if m < 20: continue
        q = np.quantile(sc[c_], min(1, np.ceil((m + 1) * (1 - alpha)) / m), method='higher')
        for i in t_:
            a = np.frombuffer(df['a' + v].values[i], dtype=np.float32)
            size.append(max(1, int((a >= 1 - q - 1e-9).sum()))); cov.append(sc[i] <= q + 1e-9)
    return np.mean(cov), np.mean(size)


rng = np.random.default_rng(0)
res = []
for rep in range(20):
    idx = rng.permutation(len(df)); cal, te = idx[:len(df) // 2], idx[len(df) // 2:]
    strata = pd.qcut(df.maxpip0.rank(method='first'), 4, labels=False).values
    for alpha in [0.1, 0.05]:
        for v, name in [('0', 'uniform'), ('1', 'seq'), ('2', 'seq_overconf')]:
            c1, s1 = lac(v, alpha, cal, te)
            c2, s2 = lac(v, alpha, cal, te, strata)
            res.append(dict(rep=rep, alpha=alpha, prior=name, cov=c1, size=s1, cov_mond=c2, size_mond=s2))
r = pd.DataFrame(res).groupby(['alpha', 'prior']).mean().drop(columns='rep')
print(r.round(3).to_string())
out['conformal'] = r.reset_index().to_dict('records')
cs = {nm: dict(cov=float(df['cs%s_cov' % v].mean()), size=float(df['cs%s_size' % v].mean())) for v, nm in [('0', 'uniform'), ('1', 'seq'), ('2', 'seq_overconf')]}
print('SuSiE 95% CS:', cs); out['susie_cs95'] = cs
# tight-LD only
t = df[df.rmax >= 0.95]
print('tight-LD (r>=.95) n=%d: CS cov uniform %.3f seq %.3f overconf %.3f; size %.2f %.2f %.2f' % (
    len(t), t.cs0_cov.mean(), t.cs1_cov.mean(), t.cs2_cov.mean(), t.cs0_size.mean(), t.cs1_size.mean(), t.cs2_size.mean()))
json.dump(out, open(args.out, 'w'), indent=1, default=float)
