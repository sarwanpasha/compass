"""Individual-level simulations on mosaic haplotypes with realistic LD (incl. perfect LD),
out-of-sample LD reference, unknown effect size. Tests:
 (i) Theorem 2 bounds / MC certificate vs realized SuSiE-SER top-1 accuracy
 (ii) Theorem 3 quadrature gain of a sequence prior, binned by LD with the causal
 (iii) conformal functional credible sets vs SuSiE 95% credible sets (coverage, size)
"""
import numpy as np, pandas as pd, argparse
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from compass.theory import mu_from_auc, p_correct_equicorr_abs, p_correct_equicorr

ap = argparse.ArgumentParser()
ap.add_argument('--loci', type=int, default=3000)
ap.add_argument('--out', default='results/individual_level_loci.pkl')
ap.add_argument('--seed', type=int, default=7)
args = ap.parse_args()
os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
rng = np.random.default_rng(args.seed)
P = 40


def haplos(m, founders, sw):
    F, p = founders.shape
    wf = rng.dirichlet(np.ones(F) * 0.7)
    cur = rng.choice(F, m, p=wf)
    H = np.empty((m, p), np.int8)
    for t in range(p):
        sw_mask = rng.random(m) < sw
        cur = np.where(sw_mask, rng.choice(F, m, p=wf), cur)
        H[:, t] = founders[cur, t]
    mut = rng.random(H.shape) < 0.0002
    return H ^ mut


def tree_founders(F, p):
    # Kingman tree over F founders; SNPs placed on branches prop. to length -> shared patterns (perfect LD)
    clades = [frozenset([i]) for i in range(F)]; t = 0.0; birth = {c: 0.0 for c in clades}; br = []
    while len(clades) > 1:
        kk = len(clades); t += rng.exponential(2.0 / (kk * (kk - 1)))
        i, j = rng.choice(kk, 2, replace=False)
        a, b = clades[i], clades[j]
        br += [(a, t - birth[a]), (b, t - birth[b])]
        new = a | b; birth[new] = t
        clades = [c for x, c in enumerate(clades) if x not in (i, j)] + [new]
    lens = np.array([l for _, l in br]); pr = lens / lens.sum()
    idx = rng.choice(len(br), p, p=pr)
    Fm = np.zeros((F, p), np.int8)
    for col, bi in enumerate(idx):
        Fm[list(br[bi][0]), col] = 1
    return Fm


def standardize(G):
    G = G - G.mean(0)
    sd = G.std(0); sd[sd == 0] = 1
    return G / sd


rows = []
for L in range(args.loci):
    F = int(rng.integers(4, 30))
    founders = tree_founders(F, P)
    sw = 10 ** rng.uniform(-3, -1)
    n = int(rng.choice([1000, 3000, 10000]))
    Hg = haplos(2 * n, founders, sw); G = (Hg[:n] + Hg[n:]).astype(float)
    Hr = haplos(2 * 500, founders, sw); Gr = (Hr[:500] + Hr[500:]).astype(float)
    maf = G.mean(0) / 2; maf = np.minimum(maf, 1 - maf)
    ok = np.where((maf > 0.02) & (Gr.std(0) > 0))[0]
    if len(ok) < 5:
        continue
    G = standardize(G[:, ok]); Gr = standardize(Gr[:, ok]); p = len(ok)
    j = rng.integers(p)
    c_true = rng.uniform(4, 12)   # target non-centrality (signal detected)
    b = c_true / np.sqrt(n)
    y = G[:, j] * b + rng.standard_normal(n) * np.sqrt(max(1 - b * b, 1e-3))
    y = (y - y.mean()) / y.std()
    z = G.T @ y / np.sqrt(n)
    if np.abs(z).max() < 5.2:          # only genome-wide significant signals
        continue
    R_in = G.T @ G / n; R_ref = Gr.T @ Gr / 500
    # SER with prior variance s0^2 on standardized effect (in z units: W = n s0^2)
    Wv = 0.04 * n * 0.01
    lbf = 0.5 * np.log(1 / (1 + Wv)) + 0.5 * z * z * Wv / (1 + Wv)
    # sequence channel: binormal scores, AUC drawn per locus around a model-level AUC
    auc = 0.75; mu = mu_from_auc(auc)
    s = rng.standard_normal(p); s[j] += mu
    lp_seq = mu * s
    def post(lp):
        a = lbf + lp; a = np.exp(a - a.max()); return a / a.sum()
    a0 = post(np.zeros(p)); a1 = post(lp_seq); a2 = post(3.0 * lp_seq)
    # LD of causal to its strongest competitor (in-sample)
    rj = np.abs(np.delete(R_in[j], j)); rmax = rj.max()
    k_tight = int((rj > 0.95).sum()) + 1
    # theory certificate (oracle c, in-sample R): equicorr approx using top competitor block
    blk = np.where(np.abs(R_in[j]) > 0.8)[0]
    rbar = np.abs(R_in[np.ix_(blk, blk)])[np.triu_indices(len(blk), 1)].mean() if len(blk) > 1 else 0
    pred_eq = p_correct_equicorr_abs(len(blk), c_true, min(rbar, 0.99999)) if len(blk) > 1 else 1.0
    from compass.theory import p_correct_equicorr
    pred_eq_comb = p_correct_equicorr(len(blk), np.sqrt(c_true**2 * (1 - min(rbar, 1)) + mu**2)) if len(blk) > 1 else 1.0
    # MC certificate on reference LD with plug-in c_hat = max|z| (shrunk 0.9)
    c_hat = 0.9 * np.abs(z).max()
    Lc = np.linalg.cholesky(R_ref + 1e-4 * np.eye(p))
    i_top = np.argmax(np.abs(z))
    zs = c_hat * R_ref[:, i_top][None, :] + (Lc @ rng.standard_normal((p, 400))).T
    cert_mc = np.mean(np.argmax(np.abs(zs), 1) == i_top)
    def cs95(a):
        o = np.argsort(-a); cm = np.cumsum(a[o]); m = np.searchsorted(cm, 0.95) + 1
        return set(o[:m]), m
    S0, m0 = cs95(a0); S1, m1 = cs95(a1); S2, m2 = cs95(a2)
    def aps(a):  # conformal score: mass of variants ranked at or above causal
        o = np.argsort(-a); return np.cumsum(a[o])[np.where(o == j)[0][0]]
    rows.append(dict(locus=L, n=n, p=p, c_true=c_true, sw=sw, rmax=rmax, k_tight=k_tight,
                     top0=int(np.argmax(a0) == j), top1=int(np.argmax(a1) == j),
                     topseq=int(np.argmax(s) == j), maxpip0=a0.max(), maxpip1=a1.max(),
                     pred_eq=pred_eq, pred_eq_comb=pred_eq_comb, k_blk=len(blk), rbar=rbar, top2=int(np.argmax(a2) == j),
                     cs2_cov=int(j in S2), cs2_size=m2, a2=a2.astype(np.float32).tobytes(), pip2_causal=a2[j], cert_mc=cert_mc, pip0_causal=a0[j], pip1_causal=a1[j],
                     cs0_cov=int(j in S0), cs0_size=m0, cs1_cov=int(j in S1), cs1_size=m1,
                     aps0=aps(a0), aps1=aps(a1),
                     lbf=lbf.astype(np.float32).tobytes(), sv=s.astype(np.float32).tobytes(), zv=z.astype(np.float32).tobytes(),
                     rrow=R_in[j].astype(np.float32).tobytes(), j=int(j), mu=mu,
                     a0=a0.astype(np.float32).tobytes(), a1=a1.astype(np.float32).tobytes()))
df = pd.DataFrame(rows)
df.to_pickle(args.out)
print(len(df), 'loci kept')
