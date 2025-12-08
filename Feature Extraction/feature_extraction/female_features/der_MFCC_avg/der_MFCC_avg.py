import numpy as np
from scipy.signal import welch
import os
import re
import pandas as pd
import librosa
import csv
from scipy.signal import resample_poly
from typing import Union, List, Dict, Tuple

def mfcc_delta_overall_average(
    audio: Union[str, np.ndarray],
    sr: int = 16000,
    target_coeffs: List[int] = [10, 12, 2, 3, 4, 9],
    n_mfcc: int = 13,
    win_length_ms: float = 25.0,
    hop_length_ms: float = 10.0,
    delta_order: int = 1
) -> Dict[str, float]:
    """
    Compute the overall average of the (first-order) derivative (delta) of MFCCs
    for requested MFCC coefficient indices.

    Parameters
    ----------
    audio : str or np.ndarray
        If str -> path to audio file readable by librosa.
        If np.ndarray -> waveform array. If 2D, it will be converted to mono by averaging channels.
    sr : int
        Sampling rate (Hz). If `audio` is a path, librosa will load at this sr.
    target_coeffs : list of int
        1-based MFCC coefficient indices to return (e.g. 10 means MFCC index 10).
    n_mfcc : int
        Number of MFCC coefficients to compute (should be >= max(target_coeffs)).
    win_length_ms : float
        Analysis window length in milliseconds (typical: 25 ms).
    hop_length_ms : float
        Hop length (frame step) in milliseconds (typical: 10 ms).
    delta_order : int
        Order of the derivative to compute. 1 -> first-order (delta). Use 0 to request no derivative.

    Returns
    -------
    features : dict
        Dictionary keyed by "Derivative_of_MFCC_OverallAverage{index}" with float values,
        where {index} is the integer index from target_coeffs.
    """

    # load audio if a path is provided
    if isinstance(audio, str):
        y, file_sr = librosa.load(audio, sr=sr, mono=False)
    elif isinstance(audio, np.ndarray):
        y = audio
        file_sr = sr
    else:
        raise ValueError("audio must be a filepath or numpy array")

    # ensure mono
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    # frame parameters in samples
    win_length = int(round(win_length_ms * file_sr / 1000.0))
    hop_length = int(round(hop_length_ms * file_sr / 1000.0))
    if win_length < 1: win_length = 1
    if hop_length < 1: hop_length = 1

    # compute MFCCs: shape (n_mfcc, n_frames)
    # librosa's mfcc uses a mel-spectrogram internally; n_mfcc commonly 13
    mfcc = librosa.feature.mfcc(y=y, sr=file_sr, n_mfcc=n_mfcc,
                                n_fft=2**(int(np.ceil(np.log2(win_length)))+1),
                                hop_length=hop_length, win_length=win_length)

    # compute derivative (delta) of MFCCs
    if delta_order == 0:
        mfcc_delta = mfcc.copy()
    else:
        mfcc_delta = librosa.feature.delta(mfcc, order=delta_order)

    # overall average across frames for each coefficient
    # result shape: (n_mfcc,)
    coeff_means = np.mean(mfcc_delta, axis=1)

    # prepare output for requested indices (user provided as 1-based)
    features = {}
    for idx in target_coeffs:
        if idx < 1 or idx > len(coeff_means):
            # If requested index exceeds computed n_mfcc, warn and set NaN
            features[f"Derivative_of_MFCC_OverallAverage{idx}"] = float("nan")
        else:
            # convert 1-based idx to 0-based numpy index
            val = float(coeff_means[idx - 1])
            features[f"Derivative_of_MFCC_OverallAverage{idx}"] = val

    return features


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
    output_file = os.path.join(output_dir, f"{patient_id}_der_MFCC_avg.csv")
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
        feats = mfcc_delta_overall_average(audio = signal_16khz)
        # feats = {'Q_MFCC_avg': feats_raw}
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