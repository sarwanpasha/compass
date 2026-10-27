"""Minimal working example. Runs on a laptop CPU in under a minute.

Part 1 checks Theorems 1 to 3 against Monte Carlo simulation of the
summary-statistic model z ~ N(c R e_j, R).

Part 2 simulates credible sets with realistic LD structure and a sequence
model that predicts per-track effects, then runs the full SR-FM workflow:
label-free sequence score, empirical Bayes temperature, SR-FM posterior,
evaluation against the known causal variant and split-conformal sets.

Usage:
    python examples/demo.py            # default: 3,000 credible sets
    python examples/demo.py --sets 500 # faster
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import logsumexp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import compass as cp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sets", type=int, default=3000)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"))
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
rng = np.random.default_rng(args.seed)
t0 = time.time()

# --------------------------------------------------------------------------
print("Part 1. Closed-form accuracy against Monte Carlo (20,000 replicates each)")
REPS = 20000
rows = []
for k, r, c, auc in [(2, 0.9, 3, 0.70), (5, 0.8, 5, 0.70), (5, 0.95, 5, 0.75),
                     (10, 0.99, 8, 0.70), (25, 0.5, 3, 0.65), (5, 1.0, 5, 0.70)]:
    mu = cp.mu_from_auc(auc)
    W = rng.standard_normal((REPS, 1))
    U = rng.standard_normal((REPS, k))
    mean = np.full(k, c * r)
    mean[0] = c
    z = mean + np.sqrt(r) * W + np.sqrt(1 - r) * U
    s = rng.standard_normal((REPS, k))
    s[:, 0] += mu
    a = c * np.sqrt(1 - r)
    zu = U.copy()
    zu[:, 0] += a                                   # LD-free part of z
    sim_data = np.mean(np.argmax(z, 1) == 0) if r < 1 else 1 / k
    sim_abs = np.mean(np.argmax(np.abs(z), 1) == 0) if r < 1 else 1 / k
    sim_comb = np.mean(np.argmax(a * zu + mu * s, 1) == 0)
    rows.append(dict(k=k, r=r, c=c, AUC_seq=auc,
                     sim_data=sim_data, thm2=cp.p_correct_equicorr(k, a),
                     sim_abs=sim_abs, thm2_abs=cp.p_correct_equicorr_abs(k, c, r),
                     sim_comb=sim_comb, thm3=cp.p_correct_equicorr(k, cp.combined_snr(a, mu))))
T1 = pd.DataFrame(rows)
print(T1.round(3).to_string(index=False))
err = max((T1.sim_data - T1.thm2).abs().max(), (T1.sim_abs - T1.thm2_abs).abs().max(),
          (T1.sim_comb - T1.thm3).abs().max())
print(f"maximum absolute error: {err:.3f}")
z1 = rng.standard_normal((REPS, 2))
c_, r_ = 4.0, 0.9
L = np.linalg.cholesky(np.array([[1, r_], [r_, 1]]))
zz = np.array([c_, c_ * r_]) + z1 @ L.T
print(f"Theorem 1, c={c_}, r={r_}: simulated |z| rule {np.mean(abs(zz[:, 0]) > abs(zz[:, 1])):.3f}, "
      f"closed form {cp.pairwise_correct_abs(c_, r_):.3f}")
print(f"Corollary 1: AUC 0.70, c=5, 1-|r|=1e-3 -> n_eff/n = {cp.neff_ratio(0.70, 5, 1 - 1e-3):.1f}")
T1.to_csv(os.path.join(args.out, "part1_theory.csv"), index=False)

# --------------------------------------------------------------------------
print(f"\nPart 2. SR-FM and conformal sets on {args.sets:,} simulated credible sets")
KINDS = np.array(["DNASE"] * 8 + ["CAGE"] * 4 + ["RNA"] * 8)
AUC_TRUE = 0.72
recs = []
for g in range(args.sets):
    k = int(rng.integers(3, 31))
    r = rng.choice([0.5, 0.8, 0.9, 0.95, 0.99, 0.999])
    c = rng.uniform(5, 10)
    # block LD: equicorrelated with small perturbations
    R = np.full((k, k), r) + np.diag(np.full(k, 1 - r))
    j = int(rng.integers(k))
    z = c * R[:, j] + np.linalg.cholesky(R + 1e-9 * np.eye(k)) @ rng.standard_normal(k)
    Wv = 25.0
    lbf = 0.5 * z * z * Wv / (1 + Wv)
    pip = np.exp(lbf - logsumexp(lbf))
    # sequence model: per-track effects, causal variant larger on accessibility tracks
    delta = rng.standard_normal((k, len(KINDS))) * rng.lognormal(0, 0.5, (k, 1))
    boost = np.where(np.isin(KINDS, ["DNASE", "CAGE"]), 1.0, 0.0)
    delta[j] *= 1 + 0.6 * boost
    for i in range(k):
        recs.append(dict(cs_id=f"cs{g:05d}", fold=g % 2, variant=i, pip=pip[i], truth=i == j, ld_r=r,
                         **{f"track{t:02d}_{KINDS[t]}": delta[i, t] for t in range(len(KINDS))}))
D = pd.DataFrame(recs)
D.head(200).to_csv(os.path.join(args.out, "toy_credible_sets_head.csv"), index=False)
tracks = [col for col in D.columns if col.startswith("track")]
kinds = [col.split("_")[1] for col in tracks]

D["seq"] = cp.standardize(cp.sequence_score(D[tracks].values, kinds))
lam, _ = cp.fit_lambda(D.pip, D.seq, D.cs_id)
D["srfm"] = cp.srfm_posterior(D.pip, D.seq, D.cs_id, lam)
print(f"empirical Bayes temperature lambda = {lam:.2f}")

res = []
for name, col in [("PIP", "pip"), ("Sequence", "seq"), ("SR-FM", "srfm")]:
    res.append(dict(score=name, top1=cp.top1_accuracy(D[col], D.truth, D.cs_id),
                    within_set_AUC=cp.within_set_auc(D[col], D.truth, D.cs_id)))
R2 = pd.DataFrame(res)
print(R2.round(3).to_string(index=False))

auc_seq = cp.within_set_auc(D.seq, D.truth, D.cs_id)
print("\nQuadrature law by LD stratum (within-set AUC)")
rows = []
for r, d in D.groupby("ld_r"):
    ap_ = cp.within_set_auc(d.pip, d.truth, d.cs_id)
    as_ = cp.within_set_auc(d.seq, d.truth, d.cs_id)
    ac_ = cp.within_set_auc(d.srfm, d.truth, d.cs_id)
    pred = cp.auc_from_mu(cp.combined_snr(max(cp.mu_from_auc(ap_), 0), max(cp.mu_from_auc(as_), 0)))
    rows.append(dict(ld_r=r, n_sets=d.cs_id.nunique(), auc_pip=ap_, auc_seq=as_, auc_srfm=ac_, predicted=pred))
R3 = pd.DataFrame(rows)
print(R3.round(3).to_string(index=False))

print("\nSplit-conformal sets (calibrate on even sets, test on odd, and the reverse)")
rows = []
for alpha in (0.2, 0.1):
    for name, col in [("PIP", "pip"), ("SR-FM", "srfm")]:
        cov, size = cp.evaluate_cross_fit(D[col], D.truth, D.cs_id, D.fold, alpha)
        rows.append(dict(target=1 - alpha, score=name, coverage=cov, mean_size=size))
R4 = pd.DataFrame(rows)
print(R4.round(3).to_string(index=False))

R2.to_csv(os.path.join(args.out, "part2_benchmark.csv"), index=False)
R3.to_csv(os.path.join(args.out, "part2_ld_strata.csv"), index=False)
R4.to_csv(os.path.join(args.out, "part2_conformal.csv"), index=False)
print(f"\nfinished in {time.time() - t0:.1f} s; tables written to {os.path.relpath(args.out)}")
