# COMPASS

Resolution theory for statistical fine-mapping, sequence-resolved fine-mapping (SR-FM) and conformal functional credible sets.

The repository contains:

1. **`compass/`**: a small Python package.
   - Closed-form accuracy of single-effect fine-mapping as a function of effect size and linkage disequilibrium (LD).
   - The quadrature law for combining association data with a sequence-model channel.
   - Effective-sample-size and effective-resolution utilities.
   - SR-FM with an empirical Bayes temperature.
   - Split-conformal credible sets.
   - Within-set evaluation metrics.
2. **`examples/demo.py`**: a self-contained demonstration that runs on a laptop CPU in about 10 seconds.
3. **`simulations/`**: simulation studies (equicorrelated LD, individual-level genotypes, coalescent model).
4. **`pipeline/`**: the real-data analyses, covering eQTL Catalogue, UK Biobank, HGDP+1kG, the reporter-assay data, and Enformer, Borzoi and AlphaGenome scoring.

## Method in brief

For a credible set with LD matrix **R**, summary statistics follow z ~ N(c **R** e_j, **R**), where j is the causal variant and c = √n β.

**Association data alone.** The probability that the causal variant is ranked first among k candidates with pairwise correlation r is

P_k = ∫ φ(x − c√(1−r)) Φ(x)^{k−1} dx.

**Adding a sequence channel.** A sequence model with within-set AUC A_s contributes separation μ = √2 Φ⁻¹(A_s). The two channels combine in quadrature:

σ² = c²(1−r) + μ².

Sequence information therefore matters most where LD is high and association data cannot separate candidates.

**SR-FM.** SR-FM turns this into a procedure:

- **Sequence score.** A label-free score s_i is computed from predicted variant effects on chromatin-accessibility and transcription-initiation tracks.
- **Prior.** The score defines a within-set prior π_i ∝ exp(λ s_i).
- **Temperature.** λ is chosen by maximizing the marginal likelihood across credible sets, so no functional labels are needed.
- **Posterior.** The posterior is q_i ∝ p_i π_i.

**Conformal sets.** Split-conformal sets threshold q_i at a level calibrated on credible sets with a known causal variant. They contain a causal variant with probability at least 1 − α under exchangeability.

## System requirements

- **Package, demo and tests.**
  - Any operating system with Python ≥ 3.9, NumPy ≥ 1.22, SciPy ≥ 1.8 and pandas ≥ 1.4.
  - Tested on Ubuntu 24.04 with Python 3.13, NumPy 2.5, SciPy 1.18 and pandas 3.0.
  - No special hardware is needed.
- **Coalescent simulations.** msprime ≥ 1.0 and stdpopsim ≥ 0.2.
- **Real-data pipeline.**
  - Linux, bcftools ≥ 1.15, and at least 200 GB of free disk.
  - A CUDA GPU with at least 40 GB of memory for sequence-model scoring.
  - Python ≥ 3.12 for AlphaGenome.
  - A Slurm cluster is optional. The driver script submits array jobs, and each step can also be run by hand.

## Installation

```bash
git clone https://github.com/sarwanpasha/compass.git
cd compass
pip install -r requirements.txt
```

Installation takes under a minute on a standard desktop. The package needs no build step; scripts import it from the repository root. To install it as a package, run `pip install -e .`.

## Demo

```bash
python examples/demo.py
```

The demo runs in about 10 seconds on a single CPU core.

**Part 1** checks the closed-form results against 20,000 Monte Carlo replicates per setting.

**Part 2** simulates 3,000 credible sets with LD from r = 0.5 to 0.999 and a sequence model that outputs 20 tracks. It then runs the complete workflow:

1. Compute the label-free sequence score.
2. Estimate λ by empirical Bayes.
3. Compute SR-FM posteriors.
4. Report top-1 accuracy and within-set AUC.
5. Compare observed AUC with the quadrature-law prediction in each LD stratum.
6. Build cross-fitted conformal sets.

Tables are written to `examples/output/`. The full expected console output is in [`examples/expected_output.txt`](examples/expected_output.txt). An abridged version:

```
Part 1. Closed-form accuracy against Monte Carlo (20,000 replicates each)
 k    r  c  AUC_seq  sim_data  thm2  sim_abs  thm2_abs  sim_comb  thm3
 5 0.95  5     0.75     0.537 0.532    0.537     0.532     0.642 0.644
 5 1.00  5     0.70     0.200 0.200    0.200     0.200     0.407 0.410
maximum absolute error: 0.004

Part 2. SR-FM and conformal sets on 3,000 simulated credible sets
   score  top1  within_set_AUC
     PIP 0.577           0.840
Sequence 0.256           0.722
   SR-FM 0.631           0.887

 ld_r  n_sets  auc_pip  auc_seq  auc_srfm  predicted
0.990     518    0.689    0.719     0.782      0.777
0.999     504    0.570    0.709     0.713      0.718

 target score  coverage  mean_size
    0.8   PIP     0.802      3.132
    0.8 SR-FM     0.800      2.160
```

How to read these results:

- **Equicorrelated settings.** The simulated and predicted columns agree to within 0.004.
- **Accuracy.** SR-FM ranks the causal variant first more often than PIP or the sequence score alone.
- **Quadrature law.** The observed SR-FM AUC matches the prediction in every LD stratum.
- **Conformal sets.** Both PIP-based and SR-FM sets reach the target coverage, and SR-FM sets are smaller.

Unit tests:

```bash
pip install pytest
python -m pytest -q tests
```

## Using COMPASS on your own data

The input is a table with one row per credible-set member:

| column | meaning |
|---|---|
| `cs_id` | credible-set identifier |
| `pip` | posterior inclusion probability from SuSiE or another single-effect method |
| track columns | predicted alternative-minus-reference effects from a sequence model, one column per track |
| `truth` (optional) | known causal variant, used for evaluation and conformal calibration only |

```python
import pandas as pd
import compass as cp

D = pd.read_csv("my_credible_sets.csv")
tracks = [c for c in D.columns if c.startswith("track")]
kinds = ["DNASE" if "DNASE" in c else "CAGE" if "CAGE" in c else "OTHER" for c in tracks]

D["seq"] = cp.standardize(cp.sequence_score(D[tracks].values, kinds))   # label-free score
lam, _ = cp.fit_lambda(D.pip, D.seq, D.cs_id)                           # empirical Bayes temperature
D["srfm"] = cp.srfm_posterior(D.pip, D.seq, D.cs_id, lam)               # SR-FM posterior

# conformal sets, calibrated on credible sets with a known causal variant
calibration_ids = D.loc[D.truth, "cs_id"].unique()[::2]          # sets with a known causal variant
cal = D[D.cs_id.isin(calibration_ids)]
tau = cp.conformal_threshold(cal.srfm, cal.truth, cal.cs_id, alpha=0.1)
D["in_set"] = cp.conformal_sets(D.srfm, D.cs_id, tau)
```

**Theory functions:**

- `cp.p_correct_equicorr(k, snr)` and `cp.p_correct_equicorr_abs(k, c, r)`: expected top-1 accuracy.
- `cp.combined_snr(a, b, rho)`: combines two evidence channels, including correlated ones.
- `cp.neff_ratio(auc, c, r)`: the sample size a sequence model is worth.
- `cp.expected_max_pip(z_lead, R)` and `cp.kappa_transform(R, kappa)`: calibration against observed PIPs.

## Reproducing the analyses

### Simulations (CPU)

```bash
python simulations/equicorr.py --out results/equicorr.csv                    # 980 settings, about 3 min
python simulations/individual_level.py --loci 30000 --seed 7 --out results/il_a.pkl
python simulations/individual_level.py --loci 30000 --seed 8 --out results/il_b.pkl
python simulations/individual_level_summary.py --loci results/il_a.pkl
python simulations/pairs_em.py results/il_a.pkl results/il_b.pkl              # pairwise tests and lambda recovery
python simulations/coalescent.py --out results/coalescent.csv                 # needs msprime and stdpopsim
```

### Real data

All inputs are public. The scripts read and write under one working directory set by `COMPASS_DIR`.

| Input | Source |
|---|---|
| eQTL Catalogue release 7 SuSiE credible sets (gene expression) and metadata | Zenodo record 23350544; eQTL-Catalogue-resources |
| eQTL Catalogue release 8 beta credible sets (MAGE, QTD000764) | Zenodo record 22846842 |
| UK Biobank 94-trait SuSiE fine-mapping, release 1.1 | finucanelab.org/data |
| HGDP+1kG phased haplotypes (GRCh38) and sample metadata | gnomAD public bucket, `resources/hgdp_1kg/phased_haplotypes_v2` |
| Reporter-assay (MPRA) results | Zenodo record 15297965 |
| GRCh38 reference and hg19-to-hg38 chain | UCSC |
| Enformer and Borzoi weights | Hugging Face `EleutherAI/enformer-official-rough`, `johahi/borzoi-replicate-0` |
| AlphaGenome weights (PyTorch port) | distributed under the AlphaGenome terms of use; place at `data/models/alphagenome/model_all_folds.safetensors` |

```bash
export COMPASS_DIR=/path/to/work
bash pipeline/setup.sh       # Python environments
bash pipeline/download.sh    # public inputs
bash pipeline/run_slurm.sh   # submits all jobs (names compass_*)
```

| Script | Analysis | Main output in `results/` |
|---|---|---|
| `hgdp_samples.py` | unrelated HGDP+1kG samples by genetic region | `../data/hgdp1kg/samples_unrelated_pop.tsv` |
| `build_sets.py` | reporter-assay labels; replication and multi-ancestry truths; variants to score | `truth_*_fullsets.parquet`, `to_score_all.parquet` |
| `score_model.py`, `score_alphagenome.py` | variant-effect scoring (SNVs and indels) | `scores/*` |
| `benchmark.py MODELS` | reporter-assay and full-set benchmarks, SR-FM, LD strata, conformal sets, two-channel test | `bench2_*.json`, `features_*.parquet` |
| `semisynthetic.py` | real LD and real sequence channel with simulated z | `semisynthetic_enformer.json` |
| `atlas.py`, `atlas_summary.py`, `gwas_prep.py` | resolution atlas for eQTL and GWAS signals | `atlas_*_summary.json` |
| `calibration.py`, `calibration_summary.py` | theory against SuSiE with in-sample and reference LD, effective resolution κ | `calibration.json` |
| `reporter_direction.py` | direction of reporter effects and haplotype concordance | `reporter_direction.json` |

Sequence-model scoring is the most expensive step and runs as eight array tasks per model. All other steps run on CPUs; the atlas and calibration jobs run per chromosome.

## License

MIT. See [LICENSE](LICENSE).
