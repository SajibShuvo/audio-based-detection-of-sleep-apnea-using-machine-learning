# Overview

This folder contains the code required to calculate and plot the Integrated Gradients for a selected subset of input Mel-spectrogram images. It also contains a slight rework of the main transformer model better leverage HPC resources on ASU Sol, which became necessary during the creation of the Integrated Gradients code. 
All file paths reference locations in the Scratch drive in ASU Sol.

## Folder Contents

```
.
├── transformer_model_jess_mods_for_speed.py  # Main transformer script, slightly modified for better HPC utilization. The script was also modified to save the best performing model as a .pt file and feature data as .npz file
├── integrated_gradients_stacked_plots.ipynb  # Jupyter notebook containing the code required to calculate Integrated Gradients for a set number of samples, and plot the 1D Integrated Gradients representation above the original Mel-spectrogram image
├── female_sbatch_jess_mods_v2.sbatch        # SLURM job submission script for submission on ASU Sol
└── README.md
```

## Outputs

transformer_model_jess_mods_for_speed.py generates:
- `best_female_transformer.pt` - Best model checkpoint (based on validation AUC)
- `dataset_metadata.npz` - Feature dimensions and dataset statistics

integrated_gradients_stacked_plots.ipynb generates:
- Integrated gradients plots (PNG files)

## HPC Job Configuration

The SLURM script requests:
- 1 GPU
- 32 CPU cores
- 128GB RAM
- 13-hour time limit 
- Email notifications on job completion


