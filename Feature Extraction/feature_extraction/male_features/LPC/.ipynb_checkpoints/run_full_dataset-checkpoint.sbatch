#!/bin/bash
#SBATCH -J LPC_Male
#SBATCH -N 1
#SBATCH -c 24
#SBATCH --mem=64G
#SBATCH -t 0-01:10:00
#SBATCH -p public
#SBATCH -q class
#SBATCH -o ./Logs/slurm.%A_%a.out
#SBATCH -e ./Logs/slurm.%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user="%u@asu.edu"
#SBATCH --export=NONE
#SBATCH --array=0-72   # Must exactly match number of patients


# Create single log folder
LOGDIR="./Logs"
mkdir -p "$LOGDIR"
echo "All logs will be stored in $LOGDIR"

# loading module and activating env
module load mamba/latest
source activate tfenv1


# Call the job script
bash ./job_full_dataset.sh
