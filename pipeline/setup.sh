#!/bin/bash
# Create the Python environments used by the real-data pipeline.
#   env/      Python 3.10+  : Enformer, Borzoi, analysis
#   agenv/    Python 3.12+  : AlphaGenome (PyTorch port)
# bcftools (>= 1.15) must be on PATH, e.g. `sudo apt-get install bcftools` or `conda install -c bioconda bcftools`.
set -euo pipefail
W=${COMPASS_DIR:-$PWD/work}; mkdir -p "$W"/{data,results,logs}
cd "$W"
python3 -m venv env && . env/bin/activate && pip install -q -U pip
pip install -q torch --index-url https://download.pytorch.org/whl/cu124
pip install -q enformer-pytorch borzoi-pytorch "transformers<4.45" pyfaidx pandas pyarrow scipy scikit-learn \
    safetensors einops huggingface_hub tqdm statsmodels pyliftover msprime stdpopsim
deactivate
PY312=${PY312:-python3.12}
if command -v "$PY312" > /dev/null; then
  "$PY312" -m venv agenv && . agenv/bin/activate && pip install -q -U pip
  pip install -q torch --index-url https://download.pytorch.org/whl/cu124
  pip install -q alphagenome-pytorch==0.3.1 pyfaidx pandas pyarrow
  deactivate
else
  echo "Python 3.12 not found; skipping the AlphaGenome environment"
fi
command -v bcftools > /dev/null || echo "WARNING: bcftools not found on PATH"
echo "environments ready in $W"
