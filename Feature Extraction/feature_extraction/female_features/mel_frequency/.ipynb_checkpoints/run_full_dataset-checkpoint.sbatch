#!/bin/bash
#SBATCH -J mel_F
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 0-05:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH -o ./Logs/slurm.%A_%a.out
#SBATCH -e ./Logs/slurm.%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user="%u@asu.edu"
#SBATCH --export=NONE

# Create single log folder
LOGDIR="./Logs"
mkdir -p "$LOGDIR"
echo "All logs will be stored in $LOGDIR"

# loading module and activating env
module load mamba/latest
source activate tfenv1


# Call the job script
bash ./job_full_dataset.sh
