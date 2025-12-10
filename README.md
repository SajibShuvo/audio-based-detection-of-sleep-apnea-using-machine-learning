# Sleep Apnea Audio Classification

This repository contains the full codebase, processing workflow, and model implementations used for audio-based sleep apnea classification. It includes pipelines for data preparation, feature extraction, and training of both traditional machine learning models and deep learning architectures.

---

## Repository Structure

### **1. Feature Extraction**
Code for generating 30-second mel-spectrogram segments and aggregated features from raw sleep audio and event annotations.  
Includes:
- Scripts and notebooks for preprocessing  
- Label generation and consistency checks  
- SLURM job files for running large-scale extraction  

---

### **2. Traditional Machine Learning Models**
Implements baseline classifiers using engineered features.  
Models include:
- Random Forest  
- Ensemble methods (stacked models)

This section contains:
- Training and evaluation scripts  
- Performance summaries for male and female datasets  
- Visualization utilities (confusion matrices, PR/ROC curves)

---

### **3. CNN Model**
A convolutional neural network trained directly on mel-spectrogram inputs.  
Contents include:
- Female and male CNN training scripts  
- Model inference utilities  
- Validation and test confusion matrices  
- ROC and PR curve images  
- HPC job scripts and logs  

---

### **4. Transformer Model**
A lightweight Transformer architecture designed for sequence-level modeling of mel-spectrogram time–frequency features.  
Includes:
- Training scripts for both male and female datasets  
- Epoch-wise confusion matrices  
- Final evaluation plots (ROC/PR)  
- Threshold-based prediction outputs  
- SLURM batch files and logs  

---

## Supporting Components
- Label distribution statistics  
- EDF segmentation helpers  
- Intermediate analysis notebooks  
- Experiment organization folders  

---

## Computational Environment
Experiments were executed on an HPC cluster using SLURM. All job scripts and logs are included to support reproducibility.

---

## Contributors
- **Sajib Biswas Shuvo**
- **Jessica Frantz**
- **Yash Bellary**
- **Daniel Graves**

---

## License
This repository is intended for academic and research use. Please cite appropriately if used in derivative work.

---
