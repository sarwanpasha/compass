"""Pairwise checks: (1) exact pairwise test of Theorem 1 (sign-agnostic closed form) and Theorem 3 quadrature law,
(2) empirical-Bayes (EM / marginal-likelihood) recovery of the sequence-prior temperature."""
import pandas as pd, numpy as np, glob, json, argparse
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from compass.theory import Phi
from scipy.special import logsumexp
ap = argparse.ArgumentParser(); ap.add_argument('loci', nargs='+', help='pickles written by individual_level.py'); ap.add_argument('--out', default='results/pairs_em_summary.json'); args = ap.parse_args()
df = pd.concat([pd.read_pickle(f) for f in args.loci], ignore_index=True)
F = lambda col: [np.frombuffer(x, dtype=np.float32).astype(float) for x in df[col]]
Z, S, Rr, LBF = F('zv'), F('sv'), F('rrow'), F('lbf')
rows = []
for i in range(len(df)):
    j = df.j[i]; c = df.c_true[i]; mu = df.mu[i]
    z, s, r, lbf = Z[i], S[i], Rr[i], LBF[i]
    for k in range(len(z)):
        if k == j: continue
        rk = abs(r[k]);
        if rk < 0.5: continue
        rows.append((c, rk, mu, int(abs(z[j]) > abs(z[k])), int(lbf[j] + mu * s[j] > lbf[k] + mu * s[k])))
P = pd.DataFrame(rows, columns=['c', 'r', 'mu', 'win_data', 'win_comb'])
a = P.c * np.sqrt((1 - P.r) / 2); b = P.c * np.sqrt((1 + P.r) / 2)
P['pred_data'] = Phi(a) * Phi(b) + Phi(-a) * Phi(-b)          # exact sign-agnostic pairwise (Theorem 1)
P['pred_comb'] = Phi(np.sqrt(P.c ** 2 * (1 - P.r) + P.mu ** 2) / np.sqrt(2))   # Theorem 3 quadrature
P['snr2'] = P.c ** 2 * (1 - P.r)
P['bin'] = pd.qcut(P.snr2, 15, duplicates='drop')
g = P.groupby('bin', observed=True)[['win_data', 'pred_data', 'win_comb', 'pred_comb']].mean()
g['n'] = P.groupby('bin', observed=True).size()
print(g.round(3).to_string())
res = {}
for v in ['data', 'comb']:
    e = g['win_' + v] - g['pred_' + v]
    res[v] = dict(max_abs=float(e.abs().max()), mean_abs=float(e.abs().mean()),
                  R2=float(1 - (e ** 2).sum() / ((g['win_' + v] - g['win_' + v].mean()) ** 2).sum()))
print(res)
# EM / marginal likelihood for prior temperature lambda (truth: mu)
lams = np.linspace(0, 3, 61)
ll = [sum(logsumexp(LBF[i] + l * S[i] - logsumexp(l * S[i])) for i in range(len(df))) for l in lams]
lam_hat = lams[int(np.argmax(ll))]
print('true mu=%.3f  ML lambda=%.3f' % (df.mu[0], lam_hat))
res['em'] = dict(true_mu=float(df.mu[0]), lambda_hat=float(lam_hat))
res['pairs_by_bin'] = g.reset_index().astype({'bin': str}).to_dict('records')
json.dump(res, open(args.out, 'w'), indent=1, default=float)
