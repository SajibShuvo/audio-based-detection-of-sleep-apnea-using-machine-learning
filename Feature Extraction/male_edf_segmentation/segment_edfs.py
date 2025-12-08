import os
import numpy as np
import pyedflib
from numpy.lib.format import open_memmap
import csv
import gc

# ================================
# CONFIGURATION
# ================================
base_dir = r"/scratch/sshuvo13/project_shared_folder_bspml_1/patient_data_raw_complete/male_patients"
output_dir = r"/scratch/sshuvo13/project_shared_folder_bspml_1/segments_30s/edfs/male"
target_channel = 18
sample_rate = 48000
chunk_seconds = 30
overlap_seconds = 20
chunk_samples = sample_rate * chunk_seconds
overlap_samples = sample_rate * overlap_seconds
step_samples = chunk_samples - overlap_samples

os.makedirs(output_dir, exist_ok=True)

metadata_path = os.path.join(output_dir, "male_subset_segment_metadata.csv")

# Create CSV if it does not exist
if not os.path.exists(metadata_path):
    with open(metadata_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "num_segments", "total_samples"])

# ================================
# Load patient list and pick one
# ================================
patients_file = "/home/sshuvo13/BSPML_project_sbs_files/segmentation_30s/patient_list_wo_two/male_ids.txt"
with open(patients_file) as f:
    patients = [line.strip() for line in f]

task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", -1))
if task_id < 0 or task_id >= len(patients):
    if task_id >= len(patients):
        print('task_id is greater than patients')
    if task_id < 0:
        print("task_id < 0")
    raise ValueError("Invalid or missing SLURM_ARRAY_TASK_ID")

patient_id = patients[task_id]
print(f"\nProcessing patient: {patient_id}")

patient_folder = os.path.join(base_dir, patient_id)
if not os.path.isdir(patient_folder):
    raise ValueError(f"Patient folder not found: {patient_folder}")

# ================================
# Find EDF files
# ================================
edf_files = sorted(
    [os.path.join(patient_folder, f) for f in os.listdir(patient_folder) if f.endswith(".edf")]
)
if not edf_files:
    raise ValueError("No EDF files found for patient: " + patient_id)

# ================================
# Step 1: Compute total samples
# ================================
total_samples = 0
for fpath in edf_files:
    f = pyedflib.EdfReader(fpath)
    total_samples += int(f.getNSamples()[target_channel])
    f.close()

# ================================
# Step 2: Create 1D memmap (unique filename per patient)
# ================================
temp_1d_file = os.path.join(output_dir, f"temp_patient_1d_{patient_id}.npy")
mm1d = open_memmap(temp_1d_file, dtype='float64', mode='w+', shape=(total_samples,))

pos = 0
chunk_size = sample_rate * chunk_seconds
for fpath in edf_files:
    print("Processing:", fpath)
    f = pyedflib.EdfReader(fpath)
    n_samples = int(f.getNSamples()[target_channel])

    start = 0
    while start < n_samples:
        stop = min(start + chunk_size, n_samples)
        seg = np.asarray(f.readSignal(target_channel, start=start, n=stop-start), dtype='float64')
        mm1d[pos:pos + len(seg)] = seg
        pos += len(seg)
        start = stop
    f.close()

del mm1d  # flush 1D memmap to disk

# ================================
# Step 3: Convert 1D memmap to overlapping 2D windows
# ================================
data = np.load(temp_1d_file, mmap_mode='r')
num_segments = int(np.ceil((total_samples - overlap_samples) / step_samples))

# ================================
# Step 4: Create 2D memmap and fill
# ================================
outfile_2d = os.path.join(output_dir, f"{patient_id}_segmented.npy")
if os.path.exists(outfile_2d):
    os.remove(outfile_2d)

X = open_memmap(outfile_2d, dtype='float64', mode='w+', shape=(num_segments, chunk_samples))

# Append metadata
with open(metadata_path, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([patient_id, num_segments, total_samples])

print("Filling 2D array")
for i in range(num_segments):
    start = i * step_samples
    end = start + chunk_samples
    if end <= total_samples:
        seg = data[start:end]
    else:
        seg = np.zeros(chunk_samples, dtype='float64')
        seg[:total_samples-start] = data[start:]
    X[i, :] = seg
print("Done")
del X
del data
gc.collect()

# ================================
# Remove temporary 1D memmap
# ================================
os.remove(temp_1d_file)
print(f"Saved 2D array for patient {patient_id} to {outfile_2d}")

