#!/bin/bash
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 0-01:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH -o ./Logs/slurm.%A_%a.out
#SBATCH -e ./Logs/slurm.%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user="%u@asu.edu"
#SBATCH --export=NONE
#SBATCH --array=0-70   # Must exactly match number of patients


# Create single log folder
LOGDIR="./Logs"
mkdir -p "$LOGDIR"
echo "All logs will be stored in $LOGDIR"

# loading module and activating env
module load mamba/latest
source activate tfenv1


# Call the job script
bash /scratch/sshuvo13/scratch_run_dir_sajib/LPC_sajib/job_full_dataset.sh
