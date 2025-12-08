import scipy.signal

import numpy as np
from scipy.signal import welch
import os
import re
import pandas as pd
import librosa
import csv
from scipy.signal import resample_poly
from typing import Union, List, Dict, Tuple, Any

def compute_lpc_apnea_features(
    audio: np.ndarray,
    sr: int = 16000,
    frame_length_ms: float = 25.0,
    hop_length_ms: float = 10.0,
    lpc_order: int = 12,
    n_fft: int = 512,
    delta_widths = (0, 3, 4, 5),
) -> Dict[str, Any]:
    """
    Compute LPC-based spectral-envelope features (and derivative/std summaries)
    intended for sleep-apnea-from-sound experiments.

    Parameters
    ----------
    audio : np.ndarray
        1D audio array (mono). Example: 9 seconds at 16 kHz.
    sr : int
        Sample rate (default 16000).
    frame_length_ms : float
        Frame length in milliseconds for short-time framing (default 25 ms).
    hop_length_ms : float
        Hop length in milliseconds (default 10 ms).
    lpc_order : int
        LPC order for each frame (default 12).
    n_fft : int
        Number of frequency bins to evaluate spectral envelope (default 512).
    delta_widths : iterable
        Iterable of integer delta widths to compute temporal deltas of the envelope.
        We use 0 to indicate "no delta" (original envelope), others are passed
        to librosa.feature.delta as `width=width*2+1` internally.

    Returns
    -------
    features : dict
        Dictionary with summary features. Keys include:
          - 'env_mean' : mean over all frames and freqs of spectral envelope
          - 'env_std'  : std over all frames and freqs of spectral envelope
          - 'freq_deriv_mean' : mean of frequency-derivative (env diff along freq)
          - 'freq_deriv_std'  : std of frequency-derivative
          - 'delta_{w}_std'   : overall std of temporal delta of envelope for width w
          - 'delta_{w}_mean'  : overall mean of temporal delta of envelope for width w
        and also per-frequency / per-frame arrays are returned under keys if needed.
    """
    # --- frame/hop sizes in samples
    frame_len = int(round(sr * (frame_length_ms / 1000.0)))
    hop_len = int(round(sr * (hop_length_ms / 1000.0)))
    if frame_len <= lpc_order:
        raise ValueError("frame length (samples) must be > lpc_order")

    # Short-time framing (librosa.util.frame helps but we'll use simple framing approach)
    # Use librosa.util.frame to create a 2D array (frames x frame_len)
    frames = librosa.util.frame(audio, frame_length=frame_len, hop_length=hop_len).T  # shape (n_frames, frame_len)

    n_frames = frames.shape[0]

    # Frequency grid for spectral envelope
    w, _ = scipy.signal.freqz([1.0], [1.0], worN=n_fft, fs=sr)  # returns freqs and response for placeholders
    freqs = w[:n_fft] if len(w) >= n_fft else w

    # Container for envelopes: shape (n_frames, n_fft)
    envelopes = np.zeros((n_frames, n_fft), dtype=float)

    for i in range(n_frames):
        frame = frames[i] * np.hamming(frame_len)  # window
        # compute LPC (librosa.lpc returns coefficients a: [1, -a1, -a2, ...] style)
        try:
            a = librosa.lpc(frame, order=lpc_order)
        except np.linalg.LinAlgError:
            # if LPC fails due to ill-conditioned autocorrelation, fallback to zeros
            a = np.zeros(lpc_order + 1, dtype=float)
            a[0] = 1.0
        # Compute frequency response of the all-pole filter 1/A(z) -> magnitude is spectral envelope
        # scipy.signal.freqz expects numerator (b) and denominator (a). For LPC, numerator is 1.
        freq_axis, h = scipy.signal.freqz([1.0], a, worN=n_fft, fs=sr)
        mag = np.abs(h)
        # Avoid zeros and stabilize with small epsilon
        envelopes[i, :] = mag + 1e-12

    # Global envelope statistics
    env_mean = float(np.mean(envelopes))
    env_std = float(np.std(envelopes))

    # Frequency derivative (first diff along frequency axis)
    freq_deriv = np.diff(envelopes, axis=1)  # shape (n_frames, n_fft-1)
    freq_deriv_mean = float(np.mean(freq_deriv))
    freq_deriv_std  = float(np.std(freq_deriv))

    # Temporal deltas of the envelope:
    # librosa.feature.delta uses width = 2 * N + 1 where N is the order.
    # We'll map delta_width w -> librosa width = 2*w + 1 (so w=3 -> width 7)
    delta_stats = {}
    for w in delta_widths:
        if w == 0:
            # interpret width 0 as the original envelope (no delta)
            delta_env = envelopes.copy()
        else:
            # compute delta along time axis for each frequency bin
            # librosa expects shape (n_freq, n_frames) -> so transpose
            delta_env = librosa.feature.delta(envelopes.T, width=(2*w + 1), order=1, axis=1).T
            # librosa.feature.delta returns same shape as input
        # overall summary
        key_mean = f"delta_{w}_mean"
        key_std  = f"delta_{w}_std"
        delta_stats[key_mean] = float(np.mean(delta_env))
        delta_stats[key_std]  = float(np.std(delta_env))

    # Pack features
    features = {
        "env_mean": env_mean,
        "env_std": env_std,
        "freq_deriv_mean": freq_deriv_mean,
        "freq_deriv_std": freq_deriv_std,
    }
    features.update(delta_stats)

    # # Optionally include some arrays for deeper inspection (but keep them optional)
    # features["_debug"] = {
    #     "n_frames": n_frames,
    #     "frame_len": frame_len,
    #     "hop_len": hop_len,
    #     "lpc_order": lpc_order,
    #     "n_fft": n_fft,
    #     # Note: don't include entire envelopes if very large; included here for completeness
    #     # Comment out if you want only summary stats:
    #     # "envelopes": envelopes,
    #     # "freq_deriv": freq_deriv,
    # }

    return features



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
    output_file = os.path.join(output_dir, f"{patient_id}_LPC.csv")
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
        feats = compute_lpc_apnea_features(audio = signal_16khz)
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