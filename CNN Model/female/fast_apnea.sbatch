#!/bin/bash

#SBATCH -J FAST_CNN
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=256G
#SBATCH -t 1-00:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH -o ./Logs_fast/slurm_fast.%A_%a.out
#SBATCH -e ./Logs_fast/slurm_fast.%A_%a.err
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

cd /scratch/sshuvo13/project_shared_folder_bspml_1/lite_CNN_model

python -u fast_train.py

echo "Job finished at $(date)"
