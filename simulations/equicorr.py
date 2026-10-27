"""Equicorrelated grid: exact verification of Theorems 1 to 3 on equicorrelated LD blocks (summary-statistic model).
Model: z ~ N(c R e_j, R); seq channel s_i = mu*1[i=j] + noise, with noise correlated (rho)
with the z-noise component unique to each variant. Decision rules:
  data-only : argmax |z|                      (== SuSiE single-effect top-1, any prior variance)
  seq-only  : argmax s
  optimal   : argmax c*z_i + mu*s_i  (known c,mu; sign known)
  naive     : argmax c*z_i + mu*s_i ignoring rho
"""
import numpy as np, pandas as pd, itertools, argparse
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from compass.theory import p_correct_equicorr, combined_snr

ap = argparse.ArgumentParser(); ap.add_argument('--out', default='results/equicorr.csv'); args = ap.parse_args()
os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
rng = np.random.default_rng(1)
REPS = 4000
rows = []
for k, r, c, mu, rho in itertools.product([2, 5, 10, 25], [0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0],
                                           [1, 2, 3, 5, 8], [0.0, 0.5, 1.0, 1.5], [0.0, 0.4]):
    if mu == 0 and rho > 0:
        continue
    # z = c R e_1 + sqrt(r) W 1 + sqrt(1-r) U
    W = rng.standard_normal((REPS, 1))
    U = rng.standard_normal((REPS, k))
    mean = np.full(k, c * r); mean[0] = c
    z = mean + np.sqrt(r) * W + np.sqrt(1 - r) * U
    # seq noise correlated with U (per-variant unique LD-free part) with corr rho
    V = rho * U + np.sqrt(1 - rho ** 2) * rng.standard_normal((REPS, k))
    s = V.copy(); s[:, 0] += mu
    acc_d = np.mean(np.argmax(z, 1) == 0) if r < 1 else np.mean(rng.integers(0, k, REPS) == 0)
    acc_abs = np.mean(np.argmax(np.abs(z), 1) == 0) if r < 1 else 1/k
    acc_s = np.mean(np.argmax(s, 1) == 0) if mu > 0 else 1 / k
    a = c * np.sqrt(1 - r)
    # optimal weights in the whitened space: w = Sigma^{-1} m, Sigma=[[1,rho],[rho,1]], m=(a,mu)
    if mu > 0:
        Sig = np.array([[1, rho], [rho, 1]]); w = np.linalg.solve(Sig, [a, mu])
        zu = (z - np.sqrt(r) * W) / max(np.sqrt(1 - r), 1e-12) if r < 1 else np.zeros_like(z)
        # the unique part: zu = U + a/(sqrt(1-r))*... -> use statistic U + a*e1 directly
        zu = U.copy(); zu[:, 0] += a
        acc_o = np.mean(np.argmax(w[0] * zu + w[1] * s, 1) == 0)
        acc_n = np.mean(np.argmax(a * zu + mu * s, 1) == 0)
    else:
        acc_o = acc_n = acc_d
    pred_d = p_correct_equicorr(k, a)
    pred_s = p_correct_equicorr(k, mu)
    pred_o = p_correct_equicorr(k, combined_snr(a, mu, rho, True)) if mu > 0 else pred_d
    pred_n = p_correct_equicorr(k, combined_snr(a, mu, rho, False)) if mu > 0 else pred_d
    rows.append(dict(k=k, r=r, c=c, mu=mu, rho=rho, acc_abs=acc_abs, acc_data=acc_d, pred_data=pred_d,
                     acc_seq=acc_s, pred_seq=pred_s, acc_opt=acc_o, pred_opt=pred_o,
                     acc_naive=acc_n, pred_naive=pred_n))
df = pd.DataFrame(rows)
df.to_csv(args.out, index=False)
for a in ['data', 'seq', 'opt', 'naive']:
    e = df[f'acc_{a}'] - df[f'pred_{a}']
    ss = 1 - (e ** 2).sum() / ((df[f'acc_{a}'] - df[f'acc_{a}'].mean()) ** 2).sum()
    print(f'{a:6s} R2={ss:.4f} maxabs={e.abs().max():.4f} mean|e|={e.abs().mean():.4f}')
