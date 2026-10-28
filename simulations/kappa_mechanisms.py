"""Which departures from the single-effect model explain the gap between theory and SuSiE PIPs?

Genotypes are real EUR individuals from HGDP+1kG (windows of 150 consecutive SNVs with MAF >= 5%; a random
half of the individuals is the study sample and the other half the LD reference). Expression is simulated
under one mechanism at a time and fine-mapped with susieR::susie on individual-level data (L = 10, estimated
residual and prior variances), as in the eQTL Catalogue. Mechanisms:
  single        one causal variant
  infinitesimal one causal variant plus small effects on every variant (h2_inf = 0.05)
  two_indep     two causal variants with |r| < 0.5
  two_linked    two causal variants with 0.8 <= |r| < 0.99, same direction
  ld_mismatch   one causal variant; prediction uses LD from the reference half
  geno_error    one causal variant; susie sees genotypes with 5% of entries resampled
For every 95% credible set the observed lead PIP is compared with the theory prediction (expected maximum
PIP of a single-effect regression on the LD among credible-set members) computed with
  (a) c = |z_lead| (plug-in), (b) the finite-sample value c^2 (n - 1) / (c^2 + n - 2) of the plug-in,
  (c) the true non-centrality of the causal variant, and (a) under the effective-resolution transform for
  kappa on a grid.

Usage: python simulations/kappa_mechanisms.py CHROM N_LOCI OUT.parquet   (needs bcftools, R with susieR)
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))
from compass.theory import expected_max_pip, kappa_transform  # noqa: E402
from common import bcf_path, sample_table  # noqa: E402

chrom, n_loci, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
KAPPAS = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]
P = 150
rng = np.random.default_rng(int(chrom) if chrom.isdigit() else 0)
tmp = tempfile.mkdtemp(prefix=f"kappa{chrom}_")

eur = sample_table().query("region == 'EUR'").s.tolist()
open(f"{tmp}/s.txt", "w").write("\n".join(eur) + "\n")
c0 = int(rng.integers(5, 40)) * 1_000_000
cmd = (f"bcftools view -r chr{chrom}:{c0}-{c0 + 20_000_000} -S {tmp}/s.txt --force-samples -m2 -M2 -v snps {bcf_path(chrom)} -Ou | "
       f"bcftools view -q 0.05:minor -Ou | bcftools query -f '[%GT\\t]\\n'")
txt = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True).stdout
H = np.array([np.array(l.rstrip("\t").replace("|", "\t").split("\t"), np.int8) for l in txt.splitlines()])
G_all = (H[:, 0::2] + H[:, 1::2]).T.astype(np.float32)
N = G_all.shape[0]
print("chr", chrom, "EUR individuals", N, "common SNVs", G_all.shape[1], flush=True)


def stdz(X):
    sd = X.std(0)
    sd[sd == 0] = 1
    return (X - X.mean(0)) / sd


mechs = ["single", "infinitesimal", "two_indep", "two_linked", "ld_mismatch", "geno_error"]
jobs = []
for locus in range(n_loci):
    st = rng.integers(0, G_all.shape[1] - P)
    Gw = G_all[:, st:st + P]
    perm = rng.permutation(N)
    study, ref = perm[: N // 2], perm[N // 2:]
    keep = (Gw[study].std(0) > 0) & (Gw[ref].std(0) > 0)
    Gs, Gr = Gw[study][:, keep], Gw[ref][:, keep]
    X = stdz(Gs.copy())
    n, p = X.shape
    R_in = X.T @ X / n
    for mech in mechs:
        b = np.zeros(p)
        j = int(rng.integers(p))
        b[j] = rng.uniform(0.25, 1.0) * rng.choice([-1, 1])
        if mech == "infinitesimal":
            b += rng.normal(0, np.sqrt(0.05 / p), p)
        if mech in ("two_indep", "two_linked"):
            r = np.abs(R_in[j])
            lo, hi = (0.0, 0.5) if mech == "two_indep" else (0.8, 0.99)
            cand = np.flatnonzero((r >= lo) & (r < hi))
            cand = cand[cand != j]
            if len(cand) == 0:
                continue
            k = int(rng.choice(cand))
            b[k] = rng.uniform(0.25, 1.0) * (np.sign(b[j] * R_in[j, k]) if mech == "two_linked" else rng.choice([-1, 1]))
        yraw = X @ b + rng.standard_normal(n)
        Gfit = Gs
        if mech == "geno_error":
            Gfit = Gs.copy()
            m = rng.random(Gfit.shape) < 0.05
            Gfit[m] = rng.choice([0, 1, 2], m.sum())
        Xf = stdz(Gfit.astype(np.float64))
        y = (yraw - yraw.mean()) / yraw.std()
        z = Xf.T @ y / np.sqrt(n)
        if np.abs(z).max() < 5:
            continue
        causal = np.flatnonzero(np.abs(b) >= 0.25)
        c_true = {int(i): float(abs(b[i]) * np.sqrt(n) / yraw.std()) for i in causal}
        tag = f"{len(jobs):05d}"
        np.savetxt(f"{tmp}/X{tag}.txt", Gfit, fmt="%d")
        np.savetxt(f"{tmp}/y{tag}.txt", y, fmt="%.6f")
        R_pred = (stdz(Gr).T @ stdz(Gr) / Gr.shape[0]) if mech == "ld_mismatch" else Xf.T @ Xf / n
        jobs.append(dict(tag=tag, locus=locus, mech=mech, z=z, R=R_pred, n=n, c_true=c_true))

rscript = f'''
library(susieR)
files <- list.files("{tmp}", pattern="^X[0-9]+\\\\.txt$")
for (f in files) {{
  tag <- sub("X", "", sub(".txt", "", f))
  X <- as.matrix(read.table(file.path("{tmp}", f))); storage.mode(X) <- "double"
  y <- scan(file.path("{tmp}", paste0("y", tag, ".txt")), quiet=TRUE)
  fit <- tryCatch(susie(X, y, L=10), error=function(e) NULL)
  if (is.null(fit)) next
  write(susie_get_pip(fit), file.path("{tmp}", paste0("pip", tag, ".txt")), ncolumns=1)
  cs <- fit$sets$cs
  con <- file(file.path("{tmp}", paste0("cs", tag, ".txt")), "w")
  if (!is.null(cs)) for (s in cs) writeLines(paste(s, collapse=","), con)
  close(con)
}}
'''
open(f"{tmp}/run.R", "w").write(rscript)
subprocess.run(["Rscript", f"{tmp}/run.R"], check=True, capture_output=True)

rows = []
for jb in jobs:
    pf, cf = f"{tmp}/pip{jb['tag']}.txt", f"{tmp}/cs{jb['tag']}.txt"
    if not os.path.exists(pf):
        continue
    pip = np.loadtxt(pf)
    lines = [l.strip() for l in open(cf) if l.strip()]
    z, R, n = jb["z"], jb["R"], jb["n"]
    for line in lines:
        mem = np.array([int(x) - 1 for x in line.split(",")])
        if len(mem) < 2:
            continue
        lead = mem[np.argmax(pip[mem])]
        order = [lead] + [x for x in mem if x != lead]
        Rcs = R[np.ix_(order, order)]
        zl = float(abs(z[lead]))
        zadj = zl * np.sqrt((n - 1) / (zl * zl + n - 2))
        cin = [jb["c_true"][i] for i in jb["c_true"] if i in set(mem)]
        rec = dict(chrom=chrom, locus=jb["locus"], mech=jb["mech"], k=len(mem), lead_pip=float(pip[lead]), lead_z=zl,
                   causal_in_cs=bool(cin), c_true=cin[0] if cin else np.nan)
        rec["pred_adj"] = expected_max_pip(zadj, Rcs, n_draws=300, rng=rng, prior_var=max(zadj ** 2, 1))[0]
        if cin:
            ct = cin[0]
            cta = ct * np.sqrt((n - 1) / (ct * ct + n - 2))
            rec["pred_true_c"] = expected_max_pip(ct, Rcs, n_draws=300, rng=rng, prior_var=max(ct ** 2, 1))[0]
            rec["pred_true_c_adj"] = expected_max_pip(cta, Rcs, n_draws=300, rng=rng, prior_var=max(cta ** 2, 1))[0]
        for kap in KAPPAS:
            Rk = Rcs if kap == 1 else kappa_transform(Rcs, kap)
            rec[f"pred_k{kap}"] = expected_max_pip(zl, Rk, n_draws=300, rng=rng)[0]
            rec[f"predadj_k{kap}"] = expected_max_pip(zadj, Rk, n_draws=300, rng=rng, prior_var=max(zadj ** 2, 1))[0]
        rows.append(rec)
shutil.rmtree(tmp, ignore_errors=True)
pd.DataFrame(rows).to_parquet(out)
print("DONE", len(rows))
