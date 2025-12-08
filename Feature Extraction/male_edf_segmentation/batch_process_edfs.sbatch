#!/bin/bash

#SBATCH -J male_30s
#SBATCH -N 1           # number of nodes
#SBATCH -c 32          # number of cores
#SBATCH --mem=128G
#SBATCH -t 0-4:00:00  # time in d-hh:mm:ss
#SBATCH -p public     # partition
#SBATCH -q public      # QOS
#SBATCH -o ./Logs/slurm.%A_%a.out
#SBATCH -e ./Logs/slurm.%A_%a.err
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user="\%u@asu.edu"
#SBATCH --export=NONE   # Purge the job-submitting shell environment
#SBATCH --array=0-72
LOGDIR="./Logs"
mkdir -p "$LOGDIR"
echo "All logs will be stored in $LOGDIR"

#Load Mamba
module load mamba/latest

#Activate environment
source activate tfenv1

#Run the python script
python /home/sshuvo13/BSPML_project_sbs_files/segmentation_30s/male_edf_segmentation/segment_edfs.py