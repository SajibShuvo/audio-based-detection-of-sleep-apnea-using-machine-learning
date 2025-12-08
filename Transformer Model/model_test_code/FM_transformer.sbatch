#!/bin/bash

#SBATCH -J FM_Transformer
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=256G
#SBATCH -t 1-00:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH -o ./Logs/slurm_fast.%A_%a.out
#SBATCH -e ./Logs/slurm_fast.%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user="%u@asu.edu"
#SBATCH --export=NONE

LOGDIR="./Logs"
mkdir -p "$LOGDIR"
echo "All logs will be stored in $LOGDIR"

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

module load mamba/latest
source activate pytorch-gpu-2.1.0-cuda-12.1

cd /scratch/sshuvo13/project_shared_folder_bspml_1/transformer_model

python -u transformer_female.py

echo "Job finished at $(date)"
