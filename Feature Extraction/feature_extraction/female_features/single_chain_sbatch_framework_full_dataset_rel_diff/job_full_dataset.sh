#!/bin/bash

# module load mamba/latest

# *********** ACTIVATE YOUR ENVIRONMENT ****************************************
# source activate tfenv1
# ******************************************************************************

# ***************** SET YOUR DIRECTORIES ***************************************
DATA_DIR="/scratch/sshuvo13/project_shared_folder_bspml_1/segments_30s/edfs/female"

LABEL_DIR="/scratch/sshuvo13/project_shared_folder_bspml_1/rml_analysis/fixed_rml_analysis/labels_again/fixed_30s_label_outputs"

OUTPUT_DIR="/scratch/sshuvo13/project_shared_folder_bspml_1/segments_30s/features/female/RDF"

# FROM THE OPTIONS BELOW, UNCOMMENT THE LIST YOU WANT TO TEST (CHOOSE 1) 
PATIENT_LIST="/home/sshuvo13/BSPML_project_sbs_files/segmentation_30s/patient_list_wo_two/female_ids.txt"
# PATIENT_LIST="/scratch/sshuvo13/project_shared_folder_bspml_1/sbatch_framework_full_dataset/male_patient_list_full_dataset.txt"
# ******************************************************************************


# ==============================================================================
# DON'T EDIT BELOW THIS LINE
# ==============================================================================
PYTHON_SCRIPT="./rel_diff_sbatch_framework_full_dataset.py"

task_id=${SLURM_ARRAY_TASK_ID}
patient_id=$(sed -n "$((task_id + 1))p" "$PATIENT_LIST")
#echo "Processing patient $patient_id"

export PATIENT_ID="$patient_id"
export DATA_DIR="$DATA_DIR"
export LABEL_DIR="$LABEL_DIR"
export OUTPUT_DIR="$OUTPUT_DIR"

python "$PYTHON_SCRIPT"

