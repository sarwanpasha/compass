#!/bin/bash
# Run the full real-data pipeline as Slurm jobs (names compass_*).
# Prerequisites: setup.sh and download.sh completed; COMPASS_DIR points to the working directory.
# GPU jobs request one GPU each; adjust -c/--mem/--gres to your cluster.
set -euo pipefail
export COMPASS_DIR=${COMPASS_DIR:-$PWD/work}
P=$(cd "$(dirname "$0")" && pwd)
W=$COMPASS_DIR; mkdir -p "$W"/{logs,results/scores}
job() {  # job NAME SBATCH_ARGS... -- COMMAND
  local name=$1; shift; local args=(); while [ "$1" != "--" ]; do args+=("$1"); shift; done; shift
  sbatch --parsable -J "compass_$name" -o "$W/logs/$name%a.log" "${args[@]}" \
    --wrap "export COMPASS_DIR=$W OMP_NUM_THREADS=4; cd $P; $*"
}
ENV="source $W/env/bin/activate;"
AGENV="source $W/agenv/bin/activate;"
S=$W/results/scores
B=$(job build -c 8 --mem=200G -- "$ENV python hgdp_samples.py && python build_sets.py && python prepare_scoring.py")
E=$(job enformer -c 6 --mem=60G --gres=gpu:1 -a 0-7 --dependency=afterok:$B -- \
    "$ENV python score_model.py enformer $W/results/to_score_all.parquet $S/enformer_\$SLURM_ARRAY_TASK_ID \$SLURM_ARRAY_TASK_ID 8")
A=$(job alphagenome -c 6 --mem=60G --gres=gpu:1 -a 0-7 --dependency=afterok:$B -- \
    "$AGENV python score_alphagenome.py $W/results/to_score_all.parquet $S/alphagenome_\$SLURM_ARRAY_TASK_ID \$SLURM_ARRAY_TASK_ID 8")
Z=$(job borzoi -c 6 --mem=60G --gres=gpu:1 -a 0-7 --dependency=afterok:$B -- \
    "$ENV python score_model.py borzoi $W/results/to_score_bench.parquet $S/borzoi_\$SLURM_ARRAY_TASK_ID \$SLURM_ARRAY_TASK_ID 8")
for M in enformer alphagenome borzoi enformer,alphagenome alphagenome,borzoi enformer,borzoi enformer,alphagenome,borzoi; do
  ID=$(job "bench_${M//,/_}" -c 8 --mem=200G --dependency=afterok:$E:$A:$Z -- "$ENV python benchmark.py $M")
  [ "$M" = enformer ] && BE=$ID
done
job semisynthetic -c 8 --mem=100G --dependency=afterok:$BE -- "$ENV python semisynthetic.py enformer" > /dev/null
job reporter_direction -c 4 --mem=60G --dependency=afterok:$B -- "$ENV python reporter_direction.py" > /dev/null
C=$(job calib -c 4 --mem=40G -a 1-22 --dependency=afterok:$B -- "$ENV python calibration.py \$SLURM_ARRAY_TASK_ID")
job calib_summary -c 2 --mem=40G --dependency=afterany:$C -- "$ENV python calibration_summary.py" > /dev/null
AE=$(job atlas_eqtl -c 4 --mem=60G -a 1-22 --dependency=afterok:$B -- "$ENV python atlas.py \$SLURM_ARRAY_TASK_ID eqtl")
job atlas_eqtl_summary -c 2 --mem=20G --dependency=afterany:$AE -- "$ENV python atlas_summary.py eqtl" > /dev/null
GP=$(job gwas_prep -c 2 --mem=40G -- "$ENV python gwas_prep.py")
AG=$(job atlas_gwas -c 4 --mem=40G -a 1-22 --dependency=afterok:$GP:$B -- "$ENV python atlas.py \$SLURM_ARRAY_TASK_ID gwas")
job atlas_gwas_summary -c 2 --mem=20G --dependency=afterany:$AG -- "$ENV python atlas_summary.py gwas" > /dev/null
job coalescent -c 2 --mem=20G -- "$ENV python ../simulations/coalescent.py --out $W/results/coalescent.csv" > /dev/null
squeue --me
