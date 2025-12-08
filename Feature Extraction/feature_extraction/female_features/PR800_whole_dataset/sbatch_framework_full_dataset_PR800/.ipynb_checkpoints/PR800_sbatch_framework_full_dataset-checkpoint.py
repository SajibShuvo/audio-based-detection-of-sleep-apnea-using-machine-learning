import numpy as np
from scipy.signal import welch
import os
import re
import pandas as pd
import librosa
import csv
from scipy.signal import resample_poly


def spectral_energy_ratio(signal, sr, cutoff=800):
    """
    Calculate the ratio of spectral energy above 'cutoff' Hz to that below 'cutoff' Hz.
    
    Parameters:
        signal (np.ndarray): 1D audio signal array.
        sr (int): Sampling rate in Hz.
        cutoff (float): Frequency threshold in Hz (default 800).
    
    Returns:
        float: Ratio of energy above cutoff to below cutoff.
    """
    # Compute the power spectral density (PSD)
    freqs, psd = welch(signal, fs=sr, nperseg=1024)
    
    # Separate energy bands
    low_band_energy = np.sum(psd[freqs < cutoff])
    high_band_energy = np.sum(psd[freqs >= cutoff])
    
    # Avoid division by zero
    if low_band_energy == 0:
        return np.inf if high_band_energy > 0 else 0.0
    
    return high_band_energy / low_band_energy

# ****************************************************************************

# def process_patient(patient_id, data, labels):
#     """
#     Compute features for a patient.
#     """
   
#     if data.ndim > 1:
#         signal = data[0]
#     else:
#         signal = data
#     # ********* Use your function here *************************
#     # feature = your_function_here(signal, fs)
#     f0 = fundamental_frequency_fft(signal, fs, fmin=1, fmax=5000)
#     # ***********************************************************
#     return {
#         "patient_id": patient_id,
#         # "feature": feature # <-- **** Output of your function *****
#         "fundamental_frequency": f0 # <-- **** Output of your function ***** 
#     }

def resample_to_16k(segment, orig_sr=48000, target_sr=16000):
    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    # ensure float32 for memory savings
    seg = np.asarray(segment, dtype=np.float32, order='C')
    # resample_poly returns float64 if input float64, so keep float32
    out = resample_poly(seg, up, down).astype(np.float32)
    return out


def main(patient_id, data_dir, label_dir, output_dir):
    # os.makedirs(output_dir, exist_ok = True)
    
    # # for filename in os.listdir(data_dir):
    # #     if filename.endswith('.npy'):
    
    #         # extracting patient id
    #         # patient_id_with_extra_string = os.path.splitext(filename)[0]
    #         # match = re.search(r'^\d{8}-\d{6}', patient_id_with_extra_string)
    #         # patient_id = match.group()
    
    
    # filename = f"{patient_id}_segmented.npy"
    # segments_file_path = os.path.join(data_dir, filename)

    # segments_file_path = os.path.join(data_dir, filename)

    # # Load the 2D numpy array (shape: [num_segments, segment_length])
    # segments = np.load(segments_file_path, mmap_mode = 'r')


    # # extracting the labels for the segments
    # label_file_name = f"{patient_id}_segments_labels.npy"
    # label_file_path = os.path.join(label_dir, label_file_name)
    # label_file = np.load(label_file_path)        
    # number_of_segments = segments.shape[0]
    # labels_subset_matching_segments_number = label_file[:number_of_segments]

    
    # # Sanity check
    # if segments.ndim != 2:
    #     raise ValueError(f"File {filename} does not contain a 2D array.")
    
    # PR800_of_the_patient = []
    
    # # Iterate through each audio segment
    # for i, segment in enumerate(segments):
    #     # ---- Your function goes here ----
    #     PR800_of_segment = spectral_energy_ratio(signal = segment, sr = 48000, cutoff=800)
    #     PR800_of_the_patient.append(PR800_of_segment)
    #     # e.g., process_segment(segment, patient_id, i)
    #     # yield patient_id, i, segment

    # data = {'PR800': PR800_of_the_patient, 'label':labels_subset_matching_segments_number}
    # df = pd.DataFrame(data = data)
    # output_file_name = f"{patient_id}_PR800.csv"
    # df.to_csv( os.path.join(output_dir, output_file_name) , index = False)

    os.makedirs(output_dir, exist_ok=True)

    filename = f"{patient_id}_segmented.npy"
    segments_file_path = os.path.join(data_dir, filename)
    print(f"Opening (memmap) {segments_file_path}")
    segments = np.load(segments_file_path, mmap_mode='r')  # <-- memory-mapped

    if segments.ndim != 2:
        print(f"File {filename} does not contain a 2D array; skipping.")
        return

    # load labels (still likely small)
    label_file_name = f"{patient_id}_segments_labels.npy"
    label_file_path = os.path.join(label_dir, label_file_name)
    label_file = np.load(label_file_path)  # if this is huge, memmap it too

    number_of_segments = segments.shape[0]
    labels_subset = label_file[:number_of_segments]

    # Prepare CSV output (write header)
    output_file = os.path.join(output_dir, f"{patient_id}_PR800_features.csv")
    header_written = False

    # iterate by index to avoid copying the whole 'segments' at once
    for i in range(number_of_segments):
        try:
            seg = segments[i]            # this is a memmap slice (view)
        except Exception as e:
            print(f"Error reading segment {i}: {e}")
            continue

        # resample to 16k
        try:
            signal_16khz = resample_to_16k(seg, orig_sr=48000, target_sr=16000)
        except Exception:
            # fallback to librosa if needed
            signal_16khz = librosa.resample(np.asarray(seg, dtype=np.float32), orig_sr=48000, target_sr=16000)


        #custom feature function goes here
        feat_raw = spectral_energy_ratio(signal = signal_16khz, sr = 16000, cutoff=800)
        feats = {'PR800': feat_raw} 
        feats['label'] = labels_subset[i] if i < len(labels_subset) else None
        feats['segment_index'] = i

        # write row immediately to CSV (append)
        if not header_written:
            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(feats.keys()))
                writer.writeheader()
                writer.writerow(feats)
            header_written = True
        else:
            with open(output_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(feats.keys()))
                writer.writerow(feats)

    print(f"Finished {filename}, wrote {output_file}")
    

if __name__ == "__main__":
    # Read environment variables set by job.sh
    patient_id = os.environ.get("PATIENT_ID")
    data_dir = os.environ.get("DATA_DIR")
    label_dir = os.environ.get("LABEL_DIR")
    output_dir = os.environ.get("OUTPUT_DIR")

    # Simple check
    if not all([patient_id, data_dir, label_dir, output_dir]):
        raise ValueError("Missing required environment variables: PATIENT_ID, DATA_DIR, LABEL_DIR, OUTPUT_DIR")

    # print(f"\nProcessing patient: {patient_id}")


    

    main(patient_id, data_dir, label_dir, output_dir)



