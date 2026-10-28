"""Variant list for Borzoi: all members of the two full-set benchmarks."""
import pandas as pd

from common import R

A = pd.concat([pd.read_parquet(R + "truth_repl_fullsets.parquet"), pd.read_parquet(R + "truth_mage_fullsets.parquet")])
keys = set(zip(A.chromosome.astype(str), A.position.astype(int), A.ref, A.alt))
U = pd.read_parquet(R + "to_score_all.parquet")
U = U[[k in keys for k in zip(U.chrom.astype(str), U.pos.astype(int), U.ref, U.alt)]]
U.sample(frac=1, random_state=1).to_parquet(R + "to_score_bench.parquet")
print("benchmark variants", len(U))
