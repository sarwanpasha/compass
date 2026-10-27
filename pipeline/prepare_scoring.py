"""Variant list for Borzoi: all members of the two full-set benchmarks."""
import pandas as pd

from common import R

A = pd.concat([pd.read_parquet(R + "truth_repl_fullsets.parquet"), pd.read_parquet(R + "truth_mage_fullsets.parquet")])
U = A[["chromosome", "position", "ref", "alt"]].drop_duplicates()
U.columns = ["chrom", "pos", "ref", "alt"]
U.sample(frac=1, random_state=1).to_parquet(R + "to_score_bench.parquet")
print("benchmark variants", len(U))
