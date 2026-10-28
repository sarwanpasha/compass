"""AlphaGenome variant-effect scoring with the PyTorch port (PyPI alphagenome-pytorch==0.3.1, Python >= 3.12).
Window centered on the variant (131,072 bp by default; set AG_LEN to change it); 128-bp heads; local alt-ref
delta summed over 3 central bins. SNVs and indels. With list columns `genes` and `tss` in the input, also writes
CAGE and RNA-seq deltas summed over the three bins around each TSS inside the window (as in score_model.py).
Usage (AlphaGenome environment): python score_alphagenome.py in.parquet out_prefix shard nshards"""
import os, sys, numpy as np, pandas as pd, torch, pyfaidx, time
from common import R, D
DEV = os.environ.get("COMPASS_DEVICE", "cuda")
from alphagenome_pytorch import AlphaGenome
inp, outp, shard, nsh = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
V = pd.read_parquet(inp); V = V.iloc[shard::nsh].reset_index(drop=True)
fa = pyfaidx.Fasta(D + 'ref/hg38.fa')
model = AlphaGenome.from_pretrained(D + 'models/alphagenome/model_all_folds.safetensors').to(DEV).eval()
L = int(os.environ.get('AG_LEN', 131072)); B = int(os.environ.get('AG_BATCH', 4)); HEADS = ['atac', 'dnase', 'procap', 'cage', 'rna_seq', 'chip_tf', 'chip_histone']
KIND = {'atac': 'ATAC', 'dnase': 'DNASE', 'procap': 'PROCAP', 'cage': 'CAGE', 'rna_seq': 'RNA', 'chip_tf': 'CHIP', 'chip_histone': 'HIST'}
LUT = np.zeros(256, np.int64) + 4
for i, c in enumerate('ACGT'): LUT[ord(c)] = i; LUT[ord(c.lower())] = i
def onehot(s):
    idx = LUT[np.frombuffer(s.encode(), np.uint8)]
    o = np.zeros((len(s), 4), np.float32); m = idx < 4; o[np.arange(len(s))[m], idx[m]] = 1; return o
loc = None; ok = np.zeros(len(V), bool); kinds = None; t0 = time.time()
HAS_TSS = 'tss' in V.columns; tss_rows, tss_meta = [], []
for b0 in range(0, len(V), B):
    seqs, idxs = [], []
    for i in range(b0, min(len(V), b0 + B)):
        c, p, r, a = 'chr' + str(V.chrom[i]).replace('chr', ''), int(V.pos[i]), str(V.ref[i]), str(V.alt[i])
        if len(r) > 200 or len(a) > 200 or not set((r + a).upper()) <= set('ACGT'): continue
        st = p - 1 - L // 2
        if st < 0: continue
        s = str(fa[c][st:st + L])
        if len(s) != L or s[L // 2:L // 2 + len(r)].upper() != r.upper(): continue
        tail = str(fa[c][st + L:st + L + max(0, len(r) - len(a))])
        sa = (s[:L // 2] + a + s[L // 2 + len(r):] + tail)[:L]
        if len(sa) != L: continue
        seqs += [onehot(s), onehot(sa)]; idxs.append(i)
    if not idxs: continue
    x = torch.from_numpy(np.stack(seqs)).to(DEV)
    with torch.no_grad(), torch.autocast(DEV, dtype=torch.bfloat16):
        out = model.predict(x, organism_index=0, resolutions=(128,), heads=tuple(HEADS))
    parts, kk = [], []
    for h in HEADS:
        y = out[h][128].float()                      # (2n, 1024, T)
        y = y.view(len(idxs), 2, y.shape[1], y.shape[2]); c0 = y.shape[2] // 2
        parts.append((y[:, 1, c0 - 1:c0 + 2] - y[:, 0, c0 - 1:c0 + 2]).sum(1)); kk += [KIND[h]] * y.shape[3]
    d = torch.cat(parts, 1).cpu().numpy().astype(np.float16)
    if HAS_TSS:
        ys = {h: out[h][128].float() for h in ['cage', 'rna_seq']}
        nb = ys['cage'].shape[1]
        for t, i in enumerate(idxs):
            st = int(V.pos[i]) - 1 - L // 2
            for g, tp in zip(V.genes[i], V.tss[i]):
                b = (int(tp) - 1 - st) // 128
                if 1 <= b < nb - 1:
                    dd = [(ys[h][2 * t + 1, b - 1:b + 2] - ys[h][2 * t, b - 1:b + 2]).sum(0) for h in ['cage', 'rna_seq']]
                    tss_rows.append(torch.cat(dd).cpu().numpy().astype(np.float16)); tss_meta.append((i, g, int(b - nb // 2)))
    if loc is None: loc = np.zeros((len(V), d.shape[1]), np.float16); kinds = kk
    loc[idxs] = d; ok[idxs] = True
    if b0 % (500 * B) == 0: print(b0, len(V), '%.2f var/s' % ((b0 + B) / (time.time() - t0)), flush=True)
np.save(outp + '_loc.npy', loc); V['ok'] = ok; V.drop(columns=[c for c in ['genes', 'tss'] if c in V]).to_parquet(outp + '_meta.parquet')
if HAS_TSS:
    np.save(outp + '_tss.npy', np.stack(tss_rows) if tss_rows else np.zeros((0, 0), np.float16))
    pd.DataFrame(tss_meta, columns=['row', 'gene_id', 'bin_offset']).to_parquet(outp + '_tss_meta.parquet')
pd.DataFrame({'description': [k + ':' for k in kinds]}).to_csv(D + 'models/alphagenome_targets_human.txt', sep='\t', index=False)
print('DONE', ok.sum(), len(V))
