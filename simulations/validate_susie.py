"""Compare compass.susie_rss with susieR::susie_rss on simulated loci built from real EUR haplotypes.

Usage: python simulations/validate_susie.py OUT.tsv   (needs R with susieR, bcftools and COMPASS_DIR data)
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))
from compass.susie import susie_rss  # noqa: E402
from common import bcf_path, sample_table  # noqa: E402

out = sys.argv[1]
rng = np.random.default_rng(11)
eur = sample_table().query("region == 'EUR'").s.tolist()
tmp = tempfile.mkdtemp()
open(f"{tmp}/s.txt", "w").write("\n".join(eur) + "\n")
cmd = (f"bcftools view -r chr22:20000000-40000000 -S {tmp}/s.txt --force-samples -m2 -M2 -v snps {bcf_path(22)} -Ou | "
       f"bcftools view -q 0.05:minor -Ou | bcftools query -f '[%GT\\t]\\n'")
txt = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True).stdout
H = np.array([np.array(l.rstrip("\t").replace("|", "\t").split("\t"), np.int8) for l in txt.splitlines()])
G = (H[:, 0::2] + H[:, 1::2]).T.astype(float)
rows = []
for locus in range(60):
    st = rng.integers(0, G.shape[1] - 150)
    X = G[:, st:st + 150]
    X = X[:, X.std(0) > 0]
    X = (X - X.mean(0)) / X.std(0)
    n, p = X.shape
    b = np.zeros(p)
    for j in rng.choice(p, rng.integers(1, 3), replace=False):
        b[j] = rng.uniform(0.25, 0.8) * rng.choice([-1, 1])
    y = X @ b + rng.standard_normal(n)
    y = (y - y.mean()) / y.std()
    z = X.T @ y / np.sqrt(n)
    R = X.T @ X / n
    np.savetxt(f"{tmp}/z.txt", z)
    np.savetxt(f"{tmp}/R.txt", R)
    rcode = (f'library(susieR); z <- scan("{tmp}/z.txt", quiet=TRUE); R <- as.matrix(read.table("{tmp}/R.txt")); '
             f'f <- susie_rss(z, R, n={n}, L=10, estimate_residual_variance=FALSE); write(susie_get_pip(f), "{tmp}/pipR.txt", ncolumns=1)')
    subprocess.run(["Rscript", "-e", rcode], check=True, capture_output=True)
    pr = np.loadtxt(f"{tmp}/pipR.txt")
    pp = susie_rss(z, R, n, L=10)["pip"]
    rows.append(dict(locus=locus, p=p, max_abs_diff=float(np.abs(pr - pp).max()),
                     corr=float(np.corrcoef(pr, pp)[0, 1]) if pr.std() > 0 and pp.std() > 0 else np.nan,
                     same_top=int(np.argmax(pr) == np.argmax(pp)), top_pip_R=float(pr.max()), top_pip_py=float(pp.max())))
D = pd.DataFrame(rows)
D.to_csv(out, sep="\t", index=False)
print(D.describe().round(4).to_string())
