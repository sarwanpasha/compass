"""Benchmarks for one or more sequence models.
Usage: python benchmark.py MODELS  (e.g. 'enformer' or 'enformer,borzoi')
(a) MPRA within-set (labeled CS members); (b) full-set truth benchmarks: replication truth & MAGE (multi-ancestry) truth,
incl. indels; SR-FM (label-free EB); LD strata + quadrature law; conformal functional credible sets;
(c) cross-model test of the correlated-channel (Mahalanobis) form of Theorem 3."""
import numpy as np, pandas as pd, glob, json, sys
from scipy.stats import norm
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from common import R, D
models = sys.argv[1].split(','); tag = '_'.join(models)
lab = pd.read_parquet(R + 'mpra_variants_hg38.parquet')[['variant', 'emvar']]
def features(m):
    metas, locs = [], []
    for f in [f for f in sorted(glob.glob(R + f'scores/{m}_*_meta.parquet')) if not f.endswith('_tss_meta.parquet')]:
        mt = pd.read_parquet(f); l = np.load(f.replace('_meta.parquet', '_loc.npy')).astype(np.float32)
        metas.append(mt[mt.ok]); locs.append(l[mt.ok.values])
    M = pd.concat(metas, ignore_index=True); X = np.vstack(locs)
    M['variant'] = 'chr' + M.chrom.astype(str).str.replace('chr', '') + '_' + M.pos.astype(str) + '_' + M.ref + '_' + M.alt
    tg = pd.read_csv(D + f'models/{m}_targets_human.txt', sep='\t')
    kind = tg.description.astype(str).str.split(':').str[0].str.upper().values[:X.shape[1]]
    Z = X / (X.std(0) + 1e-6)
    sel = np.isin(kind, ['DNASE', 'ATAC', 'CAGE'])
    M[f'{m}_unsup'] = np.log(np.sqrt((Z[:, sel] ** 2).mean(1)) + 1e-3)
    F = np.sign(X) * np.log1p(np.abs(Z)); rng = np.random.default_rng(0)
    P = PCA(64, random_state=0).fit(F[rng.choice(len(F), min(60000, len(F)), replace=False)]).transform(F); P = np.hstack([P, np.abs(P)])
    M = M.merge(lab, on='variant', how='left'); y = M.emvar.fillna(False).values.astype(int); has = M.emvar.notna().values
    M[f'{m}_sup'] = np.nan
    for c in M.chrom.astype(str).unique():
        te = (M.chrom.astype(str) == c).values; tr = has & ~te
        M.loc[te, f'{m}_sup'] = LogisticRegression(C=0.1, max_iter=3000).fit(P[tr], y[tr]).decision_function(P[te])
    return M[['variant', f'{m}_unsup', f'{m}_sup']]
F = None
for m in models:
    f = features(m); F = f if F is None else F.merge(f, on='variant', how='outer')
for k in ['unsup', 'sup']:
    cols = [f'{m}_{k}' for m in models]
    Zc = [(F[c] - F[c].mean()) / F[c].std() for c in cols]
    F[f'ens_{k}'] = pd.concat(Zc, axis=1).mean(1, skipna=False)
F.to_parquet(R + f'features_{tag}.parquet')
SC = [c for c in F.columns if c != 'variant']
OUT = {'models': models}

def ws_auc(G, col, ycol):
    num = den = 0
    for g in G:
        y = g[ycol].values.astype(bool); s = g[col].values; a, b = s[y], s[~y]
        num += (a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum(); den += len(a) * len(b)
    return num / max(den, 1)
def top1(G, col, ycol): return float(np.mean([g[ycol].values[np.argmax(g[col].values)] for g in G]))

# ---------- (a) MPRA within-set
T = pd.read_parquet(R + 'labeled_cs_members.parquet').merge(F, on='variant', how='inner')
g = T.groupby('cs_id').agg(n=('emvar', 'size'), e=('emvar', 'sum')); T = T[T.cs_id.isin(g[(g.n >= 2) & (g.e >= 1) & (g.e < g.n)].index)]
best = T.drop_duplicates('cs_id').sort_values('sample_size', ascending=False).drop_duplicates('gene_id').cs_id
P = T[T.cs_id.isin(set(best))].dropna(subset=SC)
GP = [x for _, x in P.groupby('cs_id')]
if len(GP) > 50:
    OUT['mpra'] = dict(n_sets=len(GP), auc={c: ws_auc(GP, c, 'emvar') for c in ['pip'] + SC})
    print('MPRA', json.dumps(OUT['mpra']), flush=True)

# ---------- (b) full-set truth benchmarks
def bench(truthfile, name):
    A = pd.read_parquet(R + truthfile).merge(F, on='variant', how='left')
    full = A.groupby('cs_id').apply(lambda d: bool(d[SC].notna().all().all()))
    A = A[A.cs_id.isin(full[full].index)].copy()
    A['lpip'] = np.log(np.clip(A.pip, 1e-6, 1))
    for c in SC: A[c + '_z'] = (A[c] - A[c].mean()) / A[c].std()
    G = [x for _, x in A.groupby('cs_id')]
    res = dict(n_sets=len(G), n_members=len(A), indel_frac=float(((A.ref.str.len() != 1) | (A.alt.str.len() != 1)).mean()),
               random_top1=float(np.mean([x.truth.mean() for x in G])))
    lams = np.linspace(0, 4, 81)
    for c in [s for s in SC if s.endswith('unsup')]:
        ll = [sum(logsumexp(x.lpip.values + l * x[c + '_z'].values - logsumexp(l * x[c + '_z'].values)) for x in G) for l in lams]
        lam = float(lams[int(np.argmax(ll))]); res['lambda_' + c] = lam
        A['srfm_' + c] = A.lpip + lam * A[c + '_z']
    G = [x for _, x in A.groupby('cs_id')]
    cols = ['pip'] + SC + [c for c in A.columns if c.startswith('srfm_')]
    res['metrics'] = {c: dict(top1=top1(G, c, 'truth'), auc=ws_auc(G, c, 'truth')) for c in cols}
    main = f'srfm_{"ens" if len(models) > 1 else models[0]}_unsup'; seqc = f'{"ens" if len(models) > 1 else models[0]}_unsup'
    rng = np.random.default_rng(0); pc = np.array([[x.truth.values[np.argmax(x[c].values)] for c in ['pip', seqc, main]] for x in G], float)
    bs = np.array([pc[rng.choice(len(G), len(G))].mean(0) for _ in range(2000)])
    res['ci_top1_srfm_minus_pip'] = np.percentile(bs[:, 2] - bs[:, 0], [2.5, 97.5]).tolist()
    res['ci_top1_seq_minus_pip'] = np.percentile(bs[:, 1] - bs[:, 0], [2.5, 97.5]).tolist()
    rows = []
    for lo, hi in [(-0.01, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.0)]:
        GG = [x for x in G if lo < x.cs_min_r2.iloc[0] <= hi]
        if len(GG) < 40: continue
        ap, asq, ac = ws_auc(GG, 'pip', 'truth'), ws_auc(GG, seqc, 'truth'), ws_auc(GG, main, 'truth')
        orc = max(ws_auc([x.assign(t=x.lpip + w * x[seqc + '_z']) for x in GG], 't', 'truth') for w in [0, .25, .5, 1, 2, 4])
        pred = float(norm.cdf(np.sqrt(max(norm.ppf(ap), 0) ** 2 + max(norm.ppf(asq), 0) ** 2)))
        rows.append(dict(ld=f'({lo},{hi}]', n=len(GG), auc_pip=ap, auc_seq=asq, auc_srfm=ac, auc_oracle=orc, pred=pred,
                         top1_pip=top1(GG, 'pip', 'truth'), top1_seq=top1(GG, seqc, 'truth'), top1_srfm=top1(GG, main, 'truth')))
    res['ld'] = rows
    chrom = np.array([x.variant.iloc[0].split('_')[0][3:] for x in G]); odd = np.array([(int(c) % 2 == 1) if c.isdigit() else False for c in chrom])
    for col, nm in [('lpip', 'susie'), (main, 'srfm')]:
        PP = [np.exp(x[col].values - logsumexp(x[col].values)) for x in G]
        sc = np.array([1 - p[x.truth.values].max() for p, x in zip(PP, G)])
        for alpha in [0.2, 0.1]:
            cov, size = [], []
            for calm in [odd, ~odd]:
                m = calm.sum(); q = np.quantile(sc[calm], min(1, np.ceil((m + 1) * (1 - alpha)) / m), method='higher')
                te = np.where(~calm)[0]; cov += list(sc[te] <= q + 1e-12); size += [max(1, int((PP[i] >= 1 - q - 1e-12).sum())) for i in te]
            res[f'conformal_{nm}_{alpha}'] = dict(cov=float(np.mean(cov)), size=float(np.mean(size)))
    res['cs_size_mean'] = float(np.mean([len(x) for x in G]))
    print(name, json.dumps({k: v for k, v in res.items() if k != 'ld'}, default=float)[:3000], flush=True)
    print(pd.DataFrame(rows).round(3).to_string(), flush=True)
    return res
OUT['repl'] = bench('truth_repl_fullsets.parquet', 'REPL')
OUT['mage'] = bench('truth_mage_fullsets.parquet', 'MAGE')

# ---------- (c) correlated-channel Theorem 3 across models (MPRA primary sets)
if len(models) == 2 and len(GP) > 50:
    a, b = [f'{m}_unsup' for m in models]
    Pz = P.copy()
    for c in [a, b]: Pz[c] = Pz.groupby('cs_id')[c].transform(lambda x: x - x.mean())   # within-set centering removes set effects
    nul = Pz[~Pz.emvar.astype(bool)]; rho = float(np.corrcoef(nul[a], nul[b])[0, 1])
    sa, sb = nul[a].std(), nul[b].std(); Pz[a] /= sa; Pz[b] /= sb
    GP2 = [x for _, x in Pz.groupby('cs_id')]
    Aa, Ab = ws_auc(GP2, a, 'emvar'), ws_auc(GP2, b, 'emvar')
    ma, mb = np.sqrt(2) * norm.ppf(Aa), np.sqrt(2) * norm.ppf(Ab)
    mu2 = (ma ** 2 + mb ** 2 - 2 * rho * ma * mb) / (1 - rho ** 2)
    w = np.linalg.solve(np.array([[1, rho], [rho, 1]]), [ma, mb])
    obs = ws_auc([x.assign(t=w[0] * x[a] + w[1] * x[b]) for x in GP2], 't', 'emvar')
    naive = ws_auc([x.assign(t=x[a] + x[b]) for x in GP2], 't', 'emvar')
    OUT['two_channel'] = dict(rho=rho, auc_a=Aa, auc_b=Ab, pred_opt=float(norm.cdf(np.sqrt(mu2) / np.sqrt(2))), obs_opt=obs, obs_naive=naive,
                              pred_indep=float(norm.cdf(np.sqrt(ma ** 2 + mb ** 2) / np.sqrt(2))))
    print('TWO-CHANNEL', OUT['two_channel'], flush=True)
json.dump(OUT, open(R + f'bench2_{tag}.json', 'w'), indent=1, default=float)
print('DONE')
