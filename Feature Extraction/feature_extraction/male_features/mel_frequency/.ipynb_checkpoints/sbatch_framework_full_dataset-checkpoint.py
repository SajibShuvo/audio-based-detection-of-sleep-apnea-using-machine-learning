import numpy as np
import librosa
from typing import Optional, Sequence, Union
import tqdm
import os
import re


# fs = 48000  # Hz


# def find_file_with_prefix(directory, prefix, extension):
#     """
#     Search 'directory' for a file starting with 'prefix' and ending with 'extension'.
#     Returns the full path if found; raises an error otherwise.
#     """
#     for name in os.listdir(directory):
#         if name.startswith(prefix) and name.endswith(extension):
#             return os.path.join(directory, name)
#     raise FileNotFoundError(f"No file found for ID {prefix} with {extension} in {directory}")

# ********************** Paste your whole function here **********************

target_sample_rate = 16000
original_sample_rate = 48000

def process_segment(
    x: np.ndarray,
    sr_target: int = 16000,
    orig_sr: Optional[int] = None,
    n_fft: int = 512,
    win_length: int = 400,
    hop_length: int = 160,
    n_mels: int = 64,
    add_deltas: bool = True,
    target_dur: float = 9.0
) -> np.ndarray:
    """
    Process a single 1D audio segment -> feature tensor (C, n_mels, frames).
    If orig_sr is None, assumes x is already at sr_target.
    """
    # ensure float32
    x = np.asarray(x, dtype=np.float32)

    # resample if user provided an orig_sr and it's different
    if orig_sr is not None and orig_sr != sr_target:
        x = librosa.resample(x, orig_sr=orig_sr, target_sr=sr_target)

    # trim/pad to target duration
    target_len = int(target_dur * sr_target)
    if x.shape[0] < target_len:
        x = np.pad(x, (0, target_len - x.shape[0]))
    else:
        x = x[:target_len]

    # mel spectrogram
    S = librosa.feature.melspectrogram(
        y=x,
        sr=sr_target,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        power=2.0
    )

    # convert to dB (log)
    logmel = librosa.power_to_db(S, ref=np.max)

    # per-band CMVN (per-segment)
    mean = logmel.mean(axis=1, keepdims=True)
    std = logmel.std(axis=1, keepdims=True) + 1e-9
    logmel_norm = (logmel - mean) / std

    if add_deltas:
        delta = librosa.feature.delta(logmel_norm)
        delta2 = librosa.feature.delta(logmel_norm, order=2)
        features = np.stack([logmel_norm, delta, delta2], axis=0)  # (3, n_mels, frames)
    else:
        features = logmel_norm[np.newaxis, :, :]  # (1, n_mels, frames)

    return features


def extract_batch_from_npy(
    npy_path: str,
    sr_target: int = 16000,
    orig_sr: Optional[Union[int, Sequence[int]]] = None,
    n_fft: int = 512,
    win_length: int = 400,
    hop_length: int = 160,
    n_mels: int = 64,
    add_deltas: bool = True,
    target_dur: float = 9.0,
    show_progress: bool = True
) -> np.ndarray:
    """
    Load a .npy file that contains a list/array of segments and extract features for all.
    Returns: np.ndarray of shape (N, C, n_mels, frames)

    Parameters:
    - orig_sr: either
        - None (assume all segments already at sr_target), or
        - single int (all segments share the same original sr), or
        - sequence of ints with length == N (original sr per segment).
    """
    container = np.load(npy_path, allow_pickle=True)
    # container may be an array of arrays or list
    segments = list(container)

    N = len(segments)
    # process a single dummy to determine frames and C
    example_features = process_segment(
        segments[0],
        sr_target=sr_target,
        orig_sr=(orig_sr[0] if (isinstance(orig_sr, (list, tuple, np.ndarray)) and len(orig_sr) > 0) else orig_sr),
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        add_deltas=add_deltas,
        target_dur=target_dur
    )
    C, H, W = example_features.shape

    # allocate output array
    out = np.zeros((N, C, H, W), dtype=np.float32)

    iterator = range(N)
    if show_progress:
        iterator = tqdm.tqdm(iterator, desc="Extracting features", unit="seg")

    for i in iterator:
        seg = segments[i]
        this_orig_sr = None
        if isinstance(orig_sr, (list, tuple, np.ndarray)):
            this_orig_sr = orig_sr[i]
        elif isinstance(orig_sr, int):
            this_orig_sr = orig_sr
        # process
        feats = process_segment(
            seg,
            sr_target=sr_target,
            orig_sr=this_orig_sr,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            add_deltas=add_deltas,
            target_dur=target_dur
        )
        # safety: if frames differ, trim/pad along time axis
        if feats.shape[2] != W:
            # if shorter -> pad, if longer -> trim
            if feats.shape[2] < W:
                pad_width = W - feats.shape[2]
                feats = np.pad(feats, ((0,0),(0,0),(0,pad_width)))
            else:
                feats = feats[:, :, :W]
        out[i] = feats

    return out


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


def main(patient_id, data_dir, label_dir, output_dir):
    segment_file_paths = []
    segment_file_names = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.npy'):
            segment_file_paths.append(os.path.join(data_dir, filename))
            segment_file_names.append(filename)


    os.makedirs(output_dir, exist_ok=True)
    for segment_file_path in segment_file_paths:
        
        features_batch = extract_batch_from_npy(
        segment_file_path,
        sr_target=16000,
        orig_sr=48000,         # assume segments already 16kHz; set int or list if different
        n_fft=512,
        win_length=400,
        hop_length=160,
        n_mels=64,
        add_deltas=True,
        target_dur=30.0,
        show_progress=False
        )
        match = re.search(r"/([0-9]+-[0-9]+)_segmented\.npy$", segment_file_path)
        file_name = match.group(1)
        np.save(os.path.join(output_dir, f"{file_name}.npy"), features_batch)
    

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



