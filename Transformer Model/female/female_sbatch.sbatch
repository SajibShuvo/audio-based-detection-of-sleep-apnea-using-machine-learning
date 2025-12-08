#!/bin/bash

#SBATCH -J LGFull_FM                  # Job name
#SBATCH -N 1                        # Number of nodes
#SBATCH -c 32                       # Number of CPU cores
#SBATCH --mem=128G                  # Total memory
#SBATCH -G 1
#SBATCH -t 0-16:00:00               # Time (D-HH:MM:SS)
#SBATCH -p public                   # Partition (adjust if you have a GPU partition)
#SBATCH -o ./Logs_GPU/slurm.%A_%a.out   # STDOUT log
#SBATCH -e ./Logs_GPU/slurm.%A_%a.err   # STDERR log
#SBATCH --mail-type=ALL             # Email on begin, end, fail
#SBATCH --mail-user="%u@asu.edu"    # Your ASU email (cluster macro)
#SBATCH --export=NONE               # Purge submitting shell env

LOGDIR="./Logs"
mkdir -p "$LOGDIR"
echo "All logs will be stored in $LOGDIR"

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# Load Mamba (or Conda) module
module load mamba/latest

# Activate the PyTorch GPU environment
# Adjust this if your env is created differently (e.g. 'conda activate' instead of 'source activate')
source activate pytorch-gpu-2.1.0-cuda-12.1

# Go to a working directory (optional, but nice for relative paths)
cd /scratch/sshuvo13/project_shared_folder_bspml_1/new_model/full_female/new_label

# Run the training script
python -u full_female.py

echo "Job finished at $(date)"
