import numpy as np
from scipy.signal import welch
import os
import re
import pandas as pd
import librosa
import csv
from scipy.signal import resample_poly


import numpy as np

def fraction_low_energy_windows_metrics(
    audio: np.ndarray,
    sr: int = 48000,
    win_ms: float = 20.0,
    hop_ms: float = 10.0,
    threshold_multiplier: float = 0.5,
    segment_sec: float = 1.0,
    eps: float = 1e-12,
):
    """
    Compute metrics based on the Fraction Of Low Energy Windows (FOLEW)
    for a 1D numpy audio array.

    Parameters
    ----------
    audio : np.ndarray
        1D array with audio samples.
    sr : int
        Sample rate (Hz). Default 48000.
    win_ms : float
        Window length in milliseconds for short-time energy windows. Default 20 ms.
    hop_ms : float
        Hop (step) between windows in milliseconds. Default 10 ms (50% overlap).
    threshold_multiplier : float
        Low-energy window = energy < (median_energy * threshold_multiplier).
        Default 0.5.
    segment_sec : float
        Length (seconds) of larger analysis segments. Default 1.0.
    eps : float
        Small constant to avoid divide-by-zero or zero thresholds.

    Returns
    -------
    dict with keys:
        'fractions_per_segment' : np.ndarray
        'overall_fraction' : float
        'fraction_std' : float
        'derivative_per_second' : np.ndarray
        'derivative_mean' : float
        'derivative_std' : float
        'params' : dict
    """
    audio = np.asarray(audio).ravel()
    n = audio.shape[0]
    win_samples = max(1, int(round(sr * win_ms / 1000.0)))
    hop_samples = max(1, int(round(sr * hop_ms / 1000.0)))
    segment_frames = int(round(segment_sec * sr / hop_samples))

    # pad audio so last window fits
    if n < win_samples:
        pad = win_samples - n
        audio = np.concatenate([audio, np.zeros(pad, dtype=audio.dtype)])
        n = audio.shape[0]
    else:
        pad = (hop_samples - ((n - win_samples) % hop_samples)) % hop_samples
        if pad > 0:
            audio = np.concatenate([audio, np.zeros(pad, dtype=audio.dtype)])
            n = audio.shape[0]

    # frame the signal
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        frames = sliding_window_view(audio, window_shape=win_samples)[::hop_samples]
    except Exception:
        num_frames = 1 + (n - win_samples) // hop_samples
        frames = np.empty((num_frames, win_samples), dtype=audio.dtype)
        for i in range(num_frames):
            start = i * hop_samples
            frames[i] = audio[start:start + win_samples]

    # short-time energy
    energies = np.mean(frames.astype(np.float64) ** 2, axis=1)

    med = np.median(energies)
    if med <= eps:
        med = np.mean(energies) + eps
    threshold = med * threshold_multiplier + eps

    low_flags = energies < threshold
    overall_fraction = float(np.sum(low_flags) / len(low_flags)) if len(low_flags) > 0 else 0.0

    # compute fraction per segment
    num_frames = len(low_flags)
    if segment_frames < 1:
        segment_frames = 1
    num_segments = int(np.ceil(num_frames / segment_frames))
    fractions = np.zeros(num_segments, dtype=float)
    for s in range(num_segments):
        start = s * segment_frames
        end = min(start + segment_frames, num_frames)
        if end <= start:
            fractions[s] = 0.0
        else:
            fractions[s] = float(np.sum(low_flags[start:end]) / (end - start))

    fraction_std = float(np.std(fractions)) if fractions.size > 0 else 0.0

    # derivative of fractions (per second)
    if len(fractions) >= 2:
        derivatives = np.diff(fractions) / max(segment_sec, eps)
        derivative_mean = float(np.mean(derivatives))
        derivative_std = float(np.std(derivatives))
    else:
        derivatives = np.array([], dtype=float)
        derivative_mean = 0.0
        derivative_std = 0.0

    return {
        # 'fractions_per_segment': fractions,
        'overall_fraction': overall_fraction,
        'fraction_std': fraction_std,
        # 'derivative_per_second': derivatives,
        'derivative_mean': derivative_mean,
        'derivative_std': derivative_std,
        # 'params': {
        #     'sr': sr,
        #     'win_ms': win_ms,
        #     'hop_ms': hop_ms,
        #     'threshold_multiplier': threshold_multiplier,
        #     'segment_sec': segment_sec,
        #     'win_samples': win_samples,
        #     'hop_samples': hop_samples,
        #     'segment_frames': segment_frames,
        #     'num_frames': num_frames,
        #     'num_segments': num_segments,
        # }
    }


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
    output_file = os.path.join(output_dir, f"{patient_id}_frac_low_en_features.csv")
    header_written = False

    # iterate by index to avoid copying the whole 'segments' at once
    for i in range(number_of_segments):
        try:
            seg = segments[i]            # this is a memmap slice (view)
        except Exception as e:
            print(f"Error reading segment {i}: {e}")
            continue

        # # resample to 16k
        # try:
        #     signal_16khz = resample_to_16k(seg, orig_sr=48000, target_sr=16000)
        # except Exception:
        #     # fallback to librosa if needed
        #     signal_16khz = librosa.resample(np.asarray(seg, dtype=np.float32), orig_sr=48000, target_sr=16000)


        #custom feature function goes here
        feats = fraction_low_energy_windows_metrics(
                        audio = seg,
                        sr = 48000,
                    )
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



