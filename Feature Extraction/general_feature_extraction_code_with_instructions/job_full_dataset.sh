#!/bin/bash

# module load mamba/latest

# *********** ACTIVATE YOUR ENVIRONMENT ****************************************
# source activate tfenv1
# ******************************************************************************

# ***************** SET YOUR DIRECTORIES ***************************************
DATA_DIR="/scratch/sshuvo13/project_shared_folder_bspml_1/segmented_edfs/female_segmented_edfs"

LABEL_DIR="/scratch/sshuvo13/project_shared_folder_bspml_1/rml_analysis/segment_csv_data/labels_of_each_segment"

OUTPUT_DIR="/scratch/sshuvo13/project_shared_folder_bspml_1/whole_dataset_features/female/LPC_sajib"

# FROM THE OPTIONS BELOW, UNCOMMENT THE LIST YOU WANT TO TEST (CHOOSE 1) 
PATIENT_LIST="/scratch/sshuvo13/scratch_run_dir_sajib/patient_list_wo_two/female_ids.txt"
# PATIENT_LIST="scratch/sshuvo13/scratch_run_dir_sajib/patient_list_wo_two/male_ids.txt"
# ******************************************************************************


# ==============================================================================
# DON'T EDIT BELOW THIS LINE
# ==============================================================================
PYTHON_SCRIPT="/scratch/sshuvo13/scratch_run_dir_sajib/LPC_sajib/LPC_sajib.py"

task_id=${SLURM_ARRAY_TASK_ID}
patient_id=$(sed -n "$((task_id + 1))p" "$PATIENT_LIST")
#echo "Processing patient $patient_id"

export PATIENT_ID="$patient_id"
export DATA_DIR="$DATA_DIR"
export LABEL_DIR="$LABEL_DIR"
export OUTPUT_DIR="$OUTPUT_DIR"

python "$PYTHON_SCRIPT"

