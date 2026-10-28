"""Variant-effect scoring with Enformer or Borzoi; SNVs and indels (allele-centered, right-filled to constant length).
Output per variant: local alt-ref delta summed over central bins for every track (float16).
If the input has list columns `genes` and `tss` (GRCh38 TSS positions of the genes the variant is tested
against), also writes the alt-ref delta of CAGE and RNA-seq tracks summed over the three bins around each
TSS that falls inside the output window (`_tss.npy`, rows described by `_tss_meta.parquet`).
Usage: python score_model.py {enformer|borzoi} in.parquet out_prefix shard nshards"""
import os, sys, numpy as np, pandas as pd, torch, pyfaidx, time
from common import R, D
DEV = os.environ.get("COMPASS_DEVICE", "cuda")
mname, inp, outp, shard, nsh = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
V = pd.read_parquet(inp); V = V.iloc[shard::nsh].reset_index(drop=True)
fa = pyfaidx.Fasta(D + 'ref/hg38.fa')
if mname == 'enformer':
    from enformer_pytorch import from_pretrained
    model = from_pretrained(D + 'models/EleutherAI__enformer-official-rough', target_length=896).to(DEV).eval()
    L, NT, B, cbins, BS = 196608, 5313, 4, slice(446, 449), 128
    def run(x):  # x: (b, L, 4)
        return model(x)['human'].float()             # (b, 896, NT)
else:
    from borzoi_pytorch import Borzoi
    model = Borzoi.from_pretrained(D + 'models/johahi__borzoi-replicate-0', low_cpu_mem_usage=False).to(DEV).eval()  # transformers<4.45
    L, NT, B, cbins, BS = 524288, 7611, 2, slice(3070, 3075), 32
    def run(x):  # borzoi expects (b, 4, L); returns (b, NT, bins)
        y = model(x.permute(0, 2, 1))
        if isinstance(y, dict): y = y['human'] if 'human' in y else list(y.values())[0]
        y = y.float()
        return y.permute(0, 2, 1) if y.shape[1] == NT else y
kinds = pd.read_csv(D + f'models/{mname}_targets_human.txt', sep='\t').description.astype(str).str.split(':').str[0].str.upper().values[:NT]
TSS_TRACKS = np.flatnonzero(np.isin(kinds, ['CAGE', 'RNA']))
HAS_TSS = 'tss' in V.columns
tss_rows, tss_meta = [], []
LUT = np.zeros(256, np.int64) + 4
for i, c in enumerate('ACGT'): LUT[ord(c)] = i; LUT[ord(c.lower())] = i
def onehot(s):
    idx = LUT[np.frombuffer(s.encode(), np.uint8)]
    o = np.zeros((len(s), 4), np.float32); m = idx < 4; o[np.arange(len(s))[m], idx[m]] = 1; return o
loc = np.zeros((len(V), NT), np.float16); ok = np.zeros(len(V), bool); t0 = time.time()
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
        y = run(x)
    nb = y.shape[1]; y = y.view(len(idxs), 2, nb, NT)
    loc[idxs] = (y[:, 1, cbins] - y[:, 0, cbins]).sum(1).cpu().numpy().astype(np.float16); ok[idxs] = True
    if HAS_TSS:
        off0 = (L - nb * BS) // 2                      # sequence offset of the first output bin
        for t, i in enumerate(idxs):
            st = int(V.pos[i]) - 1 - L // 2
            for g, tp in zip(V.genes[i], V.tss[i]):
                b = (int(tp) - 1 - st - off0) // BS
                if 1 <= b < nb - 1:
                    d = (y[t, 1, b - 1:b + 2, TSS_TRACKS] - y[t, 0, b - 1:b + 2, TSS_TRACKS]).sum(0)
                    tss_rows.append(d.cpu().numpy().astype(np.float16)); tss_meta.append((i, g, int(b - nb // 2)))
    if b0 % (500 * B) == 0: print(b0, len(V), '%.2f var/s' % ((b0 + B) / (time.time() - t0)), flush=True)
np.save(outp + '_loc.npy', loc); V['ok'] = ok; V.drop(columns=[c for c in ['genes', 'tss'] if c in V]).to_parquet(outp + '_meta.parquet')
if HAS_TSS:
    np.save(outp + '_tss.npy', np.stack(tss_rows) if tss_rows else np.zeros((0, len(TSS_TRACKS)), np.float16))
    pd.DataFrame(tss_meta, columns=['row', 'gene_id', 'bin_offset']).to_parquet(outp + '_tss_meta.parquet')
    np.save(outp + '_tss_tracks.npy', TSS_TRACKS)
print('DONE', ok.sum(), len(V), 'tss pairs', len(tss_meta))
