"""Relation between the empirical Bayes temperature and the sequence-channel strength.

Within a credible set, s_i = m_g + sigma_w * e_i + delta * 1[i causal], with set effects m_g and the score
standardized across all variants. The within-set binormal separation is mu = delta / sigma_w, and the
likelihood ratio of "i is causal" is exp(delta * s_i / sigma_w^2), so the optimal temperature is
lambda* = mu / sigma_w, which exceeds mu whenever sigma_w < 1. The simulation checks this and shows how
over- or under-confident PIPs (log Bayes factors scaled by T) move the empirical Bayes estimate.

Usage: python simulations/lambda_mu.py OUT.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import logsumexp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from compass.srfm import fit_lambda  # noqa: E402
from compass.metrics import within_set_auc  # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 else "results/lambda_mu.csv"
rng = np.random.default_rng(3)
rows = []
for sigma_w in [0.5, 0.7, 0.85, 1.0]:
    for mu in [0.3, 0.55, 0.8]:
        for T in [0.5, 1.0, 2.0]:
            G, k = 3000, 12
            s, pips, grp = [], [], []
            sb = np.sqrt(max(1 - sigma_w ** 2, 0))
            for g in range(G):
                j = rng.integers(k)
                sc = rng.normal(0, sb) + sigma_w * rng.standard_normal(k)
                sc[j] += mu * sigma_w
                a = 1.0                                      # association separation per candidate
                z = rng.standard_normal(k)
                z[j] += a
                lbf = T * a * z                              # T = 1: exact log-likelihood ratio
                p = np.exp(lbf - logsumexp(lbf))
                s += list(sc); pips += list(p); grp += [g] * k
            s = np.array(s)
            s = (s - s.mean()) / s.std()
            lam, _ = fit_lambda(np.array(pips), s, np.array(grp), grid=np.linspace(0, 4, 161))
            truth = np.zeros(len(s), bool)
            sw_emp = np.sqrt(np.mean(np.concatenate([(s[i * k:(i + 1) * k] - s[i * k:(i + 1) * k].mean()) ** 2 for i in range(G)]) * k / (k - 1)))
            rows.append(dict(sigma_w=sigma_w, mu=mu, T=T, lambda_hat=lam, sd_within_emp=sw_emp,
                             mu_over_sdw=mu / sw_emp))
D = pd.DataFrame(rows)
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
D.to_csv(out, index=False)
print(D.round(3).to_string(index=False))
