"""Reporter-assay labels, benchmark credible sets and the variant universe to score.
 - MPRA labels (Siraj 2026, hg19->hg38): emVar = active allele (-log10 padj>=2 & log2FC>=1) & Skew_logFDR_act>=1
 - U1: MPRA-tested members of labeled eQTL credible sets (r7 ge)
 - U2: FULL members of r7 CSs unresolved in their study (max PIP<0.5, size<=50) whose member is resolved (PIP>=0.9) in another study
 - U3: FULL members of r7 CSs unresolved in their study whose member is resolved (PIP>=0.9) in MAGE (r8-beta, multi-ancestry 1kGP LCLs)
Resolving and discovery studies must come from different donor cohorts (studies that share donors are grouped).
The strict replication truth requires resolution in at least two independent donor cohorts.
Each variant to score carries the genes it is tested against and their GENCODE v39 TSS positions.
"""
import pandas as pd, numpy as np, glob, re
from pyliftover import LiftOver
from common import R, D

# studies that share donors are treated as one cohort
COHORT = {s: 'HipSci' for s in ['HipSci', 'Jerber_2021', 'Schwartzentruber_2018', 'Alasoo_2018']}
COHORT.update({'GEUVADIS': '1kGP', 'MAGE': '1kGP', 'Walker_2019': 'UNC_fetal_brain', 'Aygun_2021': 'UNC_fetal_brain'})
cohort = lambda st: COHORT.get(st, st)
# ---- MPRA
out = []
for f in sorted(glob.glob(D + 'mpra/data/MPRA/processed_results/DE/*emVAR_glm_paired*.out.gz')):
    lib, cell = re.search(r'DE/(OL[0-9A-Z_]+?)_([A-Za-z0-9]+)_emVAR', f).groups()
    d = pd.read_csv(f, sep='\t', usecols=['SNP', 'chr', 'pos', 'ref_allele', 'alt_allele', 'window', 'A_logPadj_BH', 'B_logPadj_BH',
                                         'A_log2FC', 'B_log2FC', 'Log2Skew', 'Skew_logFDR_act'], low_memory=False)
    d = d[d.window == 'center']
    g = d.groupby('SNP').agg(chr=('chr', 'first'), pos=('pos', 'first'), ref=('ref_allele', 'first'), alt=('alt_allele', 'first'),
                             A=('A_logPadj_BH', 'max'), B=('B_logPadj_BH', 'max'), fcA=('A_log2FC', 'max'), fcB=('B_log2FC', 'max'),
                             skew=('Log2Skew', 'mean'), sfa=('Skew_logFDR_act', 'max')).reset_index()
    g['cell'] = cell; out.append(g)
L = pd.concat(out); L['chr'] = L.chr.astype(str).str.replace('chr', ''); L['pos'] = L.pos.astype(int)
L['emvar'] = (((L.A >= 2) & (L.fcA >= 1)) | ((L.B >= 2) & (L.fcB >= 1))) & (L.sfa >= 1)
L['skew_emvar'] = L['skew'].where(L['emvar'])   # allelic skew (alt vs ref) in cell lines where the variant is an emVar
V = L.groupby(['chr', 'pos', 'ref', 'alt']).agg(emvar=('emvar', 'max'), n_cells=('cell', 'nunique'), skew=('skew_emvar', 'mean')).reset_index()
lo = LiftOver(D + 'ref/hg19ToHg38.over.chain.gz'); p38 = []
for c, p in zip(V.chr, V.pos):
    r = lo.convert_coordinate('chr' + c, int(p) - 1); p38.append(r[0][1] + 1 if r and r[0][0] == 'chr' + c else -1)
V['pos38'] = p38; V = V[V.pos38 > 0]
V['variant'] = 'chr' + V.chr + '_' + V.pos38.astype(str) + '_' + V.ref + '_' + V.alt
V.to_parquet(R + 'mpra_variants_hg38.parquet'); print('MPRA', len(V), V.emvar.sum(), flush=True)
Lc = L.merge(V[['chr', 'pos', 'ref', 'alt', 'variant']], on=['chr', 'pos', 'ref', 'alt'])
Lc.groupby(['variant', 'cell']).agg(emvar=('emvar', 'max'), skew=('skew', 'mean')).reset_index().to_parquet(R + 'mpra_cell_level.parquet')
# ---- eQTL catalogue r7
C = pd.read_parquet(D + 'eqtlcat/cs_ge_r7.parquet', columns=['dataset_id', 'gene_id', 'cs_id', 'variant', 'chromosome', 'position', 'ref', 'alt', 'pip', 'z', 'cs_size', 'cs_min_r2'])
M = pd.read_csv(D + 'eqtlcat/dataset_metadata_r7.tsv', sep='\t')[['dataset_id', 'study_label', 'sample_size', 'tissue_label']]
C = C.merge(M, on='dataset_id')
C['cohort'] = C.study_label.map(cohort)
# U1
lab = V.set_index('variant')
T = C[C.cs_size >= 2].merge(lab[['emvar']], left_on='variant', right_index=True)
s = T.groupby('cs_id').agg(n_t=('emvar', 'size'), n_e=('emvar', 'sum'))
T = T[T.cs_id.isin(s[(s.n_t >= 2) & (s.n_e >= 1) & (s.n_e < s.n_t)].index)]
T.to_parquet(R + 'labeled_cs_members.parquet'); print('U1 sets', T.cs_id.nunique(), flush=True)
# all members of credible sets with at least one reporter-tested member (tested vs untested comparison)
Ct = C[C.cs_id.isin(set(C[C.variant.isin(lab.index)].cs_id))][['cs_id', 'dataset_id', 'gene_id', 'variant', 'pip', 'cs_size', 'cs_min_r2', 'tissue_label', 'study_label']]
Ct['tested'] = Ct.variant.isin(lab.index); Ct.to_parquet(R + 'cs_tested_status.parquet'); print('tested-status sets', Ct.cs_id.nunique(), flush=True)
# U2 (replication truth) and U3 (MAGE truth)
mx = C.groupby('cs_id').pip.transform('max')
A = C[(mx < 0.5) & (C.cs_size <= 50)].copy()
res = C[C.pip >= 0.9][['gene_id', 'variant', 'cohort']].drop_duplicates().rename(columns={'cohort': 'cB'})
A2 = A.merge(res, on=['gene_id', 'variant'], how='left')
A2['indep'] = A2.cB.notna() & (A2.cB != A2.cohort)
n_ind = A2[A2.indep].groupby(['cs_id', 'variant']).cB.nunique().rename('n_resolving_cohorts')
A2 = A2.sort_values('indep', ascending=False).drop_duplicates(['cs_id', 'variant']).drop(columns='cB')
A2 = A2.merge(n_ind, left_on=['cs_id', 'variant'], right_index=True, how='left'); A2['n_resolving_cohorts'] = A2.n_resolving_cohorts.fillna(0).astype(int)
A2['truth'] = A2.n_resolving_cohorts >= 1; A2['truth_strict'] = A2.n_resolving_cohorts >= 2; A2 = A2.drop(columns='indep')
t = A2.groupby('cs_id').truth.sum(); A2 = A2[A2.cs_id.isin(t[t >= 1].index)]
best = A2.drop_duplicates('cs_id').sort_values('sample_size', ascending=False).drop_duplicates('gene_id').cs_id
A2 = A2[A2.cs_id.isin(set(best))]; A2.to_parquet(R + 'truth_repl_fullsets.parquet'); print('U2 sets', A2.cs_id.nunique(), flush=True)
B = pd.read_parquet(D + 'eqtlcat/cs_ge_r8b.parquet')
B = B[B.dataset_id == 'QTD000764']; print('MAGE CS rows', len(B), 'cols', list(B.columns)[:12], flush=True)
mres = B[B.pip >= 0.9][['gene_id', 'variant']].drop_duplicates(); mres['mtruth'] = True
A3 = A[A.cohort != '1kGP'].merge(mres, on=['gene_id', 'variant'], how='left'); A3['truth'] = A3.mtruth.fillna(False).astype(bool); A3 = A3.drop(columns='mtruth')
t = A3.groupby('cs_id').truth.sum(); A3 = A3[A3.cs_id.isin(t[t >= 1].index)]
best = A3.drop_duplicates('cs_id').sort_values('sample_size', ascending=False).drop_duplicates('gene_id').cs_id
A3p = A3[A3.cs_id.isin(set(best))]; A3p.to_parquet(R + 'truth_mage_fullsets.parquet'); print('U3 sets', A3p.cs_id.nunique(), flush=True)
# EMS-style training variants: confidently fine-mapped (PIP >= 0.9) vs unlikely (PIP < 0.01) members of
# credible sets that are not benchmark sets (sampled, SNVs and indels)
bench_ids = set(A2.cs_id) | set(A3p.cs_id) | set(T.cs_id)
O = C[~C.cs_id.isin(bench_ids)]
pos_ = O[O.pip >= 0.9].drop_duplicates('variant')
neg_ = O[(O.pip < 0.01)].drop_duplicates('variant')
EMS = pd.concat([pos_.sample(min(20000, len(pos_)), random_state=0).assign(label=1),
                 neg_.sample(min(20000, len(neg_)), random_state=0).assign(label=0)])
EMS = EMS.drop_duplicates('variant', keep='first')
EMS[['variant', 'gene_id', 'chromosome', 'position', 'ref', 'alt', 'pip', 'label']].to_parquet(R + 'ems_training.parquet')
print('EMS-style training variants', len(EMS), EMS.label.sum(), flush=True)
# GENCODE v39 TSS (strand-aware gene start) for the eQTL genes
gtf = pd.read_csv(D + 'gencode/gencode.v39.basic.annotation.gtf.gz', sep='\t', comment='#', header=None, usecols=[0, 2, 3, 4, 6, 8])
gtf = gtf[gtf[2] == 'gene']
gtf['gene_id'] = gtf[8].str.extract(r'gene_id "(ENSG\d+)')[0]
gtf['tss'] = np.where(gtf[6] == '+', gtf[3], gtf[4])
TSS = gtf.drop_duplicates('gene_id').set_index('gene_id').tss
P = pd.concat([T, A2, A3p, EMS])[['chromosome', 'position', 'ref', 'alt', 'gene_id']].drop_duplicates()
P['tss'] = P.gene_id.map(TSS)
P = P.dropna(subset=['tss'])
U = P.groupby(['chromosome', 'position', 'ref', 'alt']).agg(genes=('gene_id', list), tss=('tss', lambda x: [int(v) for v in x])).reset_index()
U = pd.concat([U, pd.concat([T, A2, A3p, EMS])[['chromosome', 'position', 'ref', 'alt']].drop_duplicates()]).drop_duplicates(['chromosome', 'position', 'ref', 'alt'])
U['genes'] = U.genes.apply(lambda x: x if isinstance(x, list) else []); U['tss'] = U.tss.apply(lambda x: x if isinstance(x, list) else [])
U.columns = ['chrom', 'pos', 'ref', 'alt', 'genes', 'tss']
U = U.sample(frac=1, random_state=0).reset_index(drop=True)
U.to_parquet(R + 'to_score_all.parquet'); print('to score', len(U), 'indels', int(((U.ref.str.len() != 1) | (U.alt.str.len() != 1)).sum()))
