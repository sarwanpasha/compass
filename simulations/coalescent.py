"""Coalescent simulations of perfect-LD partners under an out-of-Africa model.

Simulates 1-Mb regions under OutOfAfrica_3G09 (stdpopsim, msprime >= 1.0) with 250
diploid individuals each from YRI, CEU and CHB, mutation rate 1.29e-8 per base per
generation and a range of recombination rates. For 300 common focal variants
(MAF >= 5%) per population and replicate, counts the other variants within 250 kb
with r^2 >= 0.99 in that population (the focal variant itself is not counted).

Usage: python simulations/coalescent.py [--reps 18] [--out results/coalescent.csv]
Requires: pip install msprime stdpopsim
"""
import argparse
import os

import msprime
import numpy as np
import pandas as pd
import stdpopsim

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=18)
ap.add_argument("--rates", type=float, nargs="+", default=[0.1, 0.3, 1.0, 3.0], help="cM/Mb")
ap.add_argument("--n", type=int, default=250, help="diploid individuals per population")
ap.add_argument("--focal", type=int, default=300)
ap.add_argument("--out", default="results/coalescent.csv")
args = ap.parse_args()
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

model = stdpopsim.get_species("HomSap").get_demographic_model("OutOfAfrica_3G09")
POPS = ["YRI", "CEU", "CHB"]
L, WIN, MU = 1_000_000, 250_000, 1.29e-8
rows = []
for rate in args.rates:
    for rep in range(args.reps):
        seed = int(rate * 1000) * 100 + rep + 1
        ts = msprime.sim_ancestry(samples={p: args.n for p in POPS}, demography=model.model,
                                  sequence_length=L, recombination_rate=rate * 1e-8, random_seed=seed)
        ts = msprime.sim_mutations(ts, rate=MU, random_seed=seed, discrete_genome=True)
        pos = np.array([s.position for s in ts.sites()])
        biallelic = np.array([len(s.mutations) == 1 for s in ts.sites()])
        G = ts.genotype_matrix()[biallelic].astype(np.int8)          # sites x haplotypes
        pos = pos[biallelic]
        pop_id = {ts.population(i).metadata["name"]: i for i in range(ts.num_populations)}
        rng = np.random.default_rng(seed)
        for p in POPS:
            hap = ts.samples(population=pop_id[p])
            X = G[:, hap].astype(np.float32)
            f = X.mean(1)
            maf = np.minimum(f, 1 - f)
            poly = np.flatnonzero(maf > 0)
            common = np.flatnonzero(maf >= 0.05)
            if len(common) == 0:
                continue
            focal = rng.choice(common, min(args.focal, len(common)), replace=False)
            Xp = X[poly]
            Z = (Xp - Xp.mean(1, keepdims=True)) / Xp.std(1, keepdims=True)
            pp = pos[poly]
            where = {v: i for i, v in enumerate(poly)}
            for v in focal:
                i = where[v]
                win = np.flatnonzero(np.abs(pp - pp[i]) <= WIN)
                r = Z[win] @ Z[i] / Z.shape[1]
                n_partners = int((r ** 2 >= 0.99).sum()) - 1                 # exclude the focal variant
                rows.append(dict(rate=rate, rep=rep, pop=p, n_partners=n_partners, irreducible=n_partners > 0))
        print(f"rate {rate} cM/Mb replicate {rep + 1}/{args.reps}", flush=True)
D = pd.DataFrame(rows)
D.to_csv(args.out, index=False)
print(D.groupby(["pop", "rate"]).agg(mean_partners=("n_partners", "mean"),
                                     frac_irreducible=("irreducible", "mean")).round(3).to_string())
