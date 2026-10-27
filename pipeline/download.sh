#!/bin/bash
# Download public inputs into $COMPASS_DIR/data (about 120 GB, mostly HGDP+1kG haplotypes).
set -uo pipefail
W=${COMPASS_DIR:-$PWD/work}/data; mkdir -p "$W"/{eqtlcat,ref,mpra,hgdp1kg,ukbb,models}; cd "$W"
G="wget -q -c --tries=10 --timeout=120"
( cd eqtlcat
  $G https://raw.githubusercontent.com/eQTL-Catalogue/eQTL-Catalogue-resources/master/data_tables/dataset_metadata_r7.tsv
  $G https://raw.githubusercontent.com/eQTL-Catalogue/eQTL-Catalogue-resources/master/data_tables/dataset_metadata_r8_beta.tsv
  $G -O cs_ge_r7.parquet "https://zenodo.org/records/23350544/files/eQTL_Catalogue_r7_cs_ge_071026.parquet?download=1"
  $G -O cs_ge_r8b.parquet "https://zenodo.org/records/22846842/files/eQTL_Catalogue_r8-beta_cs_ge_190926.parquet?download=1"
  echo "eQTL Catalogue done" ) &
( cd ref
  $G https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz && gunzip -f hg38.fa.gz
  $G https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz
  echo "reference done" ) &
( cd mpra
  $G -O data_MPRA.zip https://zenodo.org/api/records/15297965/files/data_MPRA.zip/content
  unzip -oq data_MPRA.zip 'data/MPRA/processed_results/DE/*'
  echo "MPRA done" ) &
( cd hgdp1kg
  B=https://storage.googleapis.com/gcp-public-data--gnomad/resources/hgdp_1kg/phased_haplotypes_v2
  $G https://storage.googleapis.com/gcp-public-data--gnomad/release/3.1.2/vcf/genomes/gnomad.genomes.v3.1.2.hgdp_1kg_subset_sample_meta.tsv.bgz
  seq 1 22 | xargs -P 8 -I{} sh -c "$G $B/hgdp1kgp_chr{}.filtered.SNV_INDEL.phased.shapeit5.bcf && $G $B/hgdp1kgp_chr{}.filtered.SNV_INDEL.phased.shapeit5.bcf.csi"
  echo "HGDP+1kG done" ) &
( cd ukbb
  [ -f UKBB_94traits_release1.bed.gz ] || \
    echo "Download UKBB_94traits_release1.bed.gz (fine-mapping release 1.1) from https://www.finucanelab.org/data into data/ukbb/"
  echo "UK Biobank checked" ) &
( cd models
  $G -O enformer_targets_human.txt https://raw.githubusercontent.com/calico/basenji/master/manuscripts/cross2020/targets_human.txt
  $G -O borzoi_targets_human.txt https://raw.githubusercontent.com/calico/borzoi/main/examples/targets_human.txt
  "${COMPASS_DIR:-$PWD/work}"/env/bin/python - <<'PY'
from huggingface_hub import snapshot_download
for r in ['EleutherAI/enformer-official-rough', 'johahi/borzoi-replicate-0']:
    print(snapshot_download(r, local_dir=r.replace('/', '__')))
PY
  echo "models done" ) &
wait
echo "AlphaGenome weights are distributed under their own terms of use; place the PyTorch weights at"
echo "  $W/models/alphagenome/model_all_folds.safetensors"
du -sh "$W"/*
