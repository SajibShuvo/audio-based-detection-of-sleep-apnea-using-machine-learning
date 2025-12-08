import numpy as np
from scipy.signal import welch
import os
import re
import pandas as pd
import librosa
import csv
from scipy.signal import resample_poly


def overall_avg_area_moment_cq_mfcc(y,
                                    sr=16000,
                                    n_mfcc=13,
                                    n_bins=84,            # CQT bins (e.g., 7 octaves * 12)
                                    bins_per_octave=12,
                                    hop_length=512,
                                    use_abs_weights=True):
    """
    Compute overall average of area-method (2nd central) moments of constant-Q based MFCCs.
    Returns a single float.
    """
    # 1) load audio (mono)
    # y, _ = librosa.load(path, sr=sr, mono=True, duration=None)

    # 2) compute CQT magnitude (power)
    C = librosa.cqt(y, sr=sr, hop_length=hop_length,
                    n_bins=n_bins, bins_per_octave=bins_per_octave)
    # C is complex; convert to power
    C_power = np.abs(C)**2

    # 3) convert to log-power (dB)
    C_db = librosa.power_to_db(C_power, ref=np.max)

    # 4) compute MFCCs from this log-CQT spectrogram
    # librosa.feature.mfcc expects either y or S (Mel spectrogram), but giving S works:
    # pass S as log-power (linear scale expected) — we supply C_db which is in dB, so convert back:
    # librosa.feature.mfcc expects "S" in power (not dB), so we'll convert dB back to amplitude-like:
    S_for_mfcc = librosa.db_to_power(C_db)
    mfccs = librosa.feature.mfcc(S=S_for_mfcc, n_mfcc=n_mfcc)

    # mfccs shape: (n_mfcc, n_frames)
    n_m, n_t = mfccs.shape
    moments = np.zeros(n_m)

    # frame indices (use numeric frame positions as 'distance' axis)
    t_idx = np.arange(n_t, dtype=np.float64)

    for m in range(n_m):
        coeff = mfccs[m, :]

        # weight = absolute value or squared depending on preference
        if use_abs_weights:
            w = np.abs(coeff)
        else:
            w = coeff**2

        # avoid zero total weight
        W = w.sum()
        if W <= 0:
            moments[m] = 0.0
            continue

        centroid = (t_idx * w).sum() / W
        second_central = (( (t_idx - centroid)**2 ) * w).sum() / W
        moments[m] = second_central

    # overall average of the per-band area moments
    overall_avg = float(moments.mean())
    return overall_avg

# Example usage:
# result = overall_avg_area_moment_cq_mfcc("path/to/your_9s_clip.wav")
# print("Overall average area moment (constant-Q MFCCs):", result)

def main(patient_id, data_dir, label_dir, output_dir):


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
    output_file = os.path.join(output_dir, f"{patient_id}_Q_MFCC_avg.csv")
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
        feats_raw = overall_avg_area_moment_cq_mfcc(signal_16khz)
        feats = {'Q_MFCC_avg': feats_raw}
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