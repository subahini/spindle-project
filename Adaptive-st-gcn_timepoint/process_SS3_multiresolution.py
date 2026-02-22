"""
process_SS3_multiresolution.py

Enhanced data processing for time-point prediction.
Extracts both:
1. DE features (for graph-based coarse detection)
2. Raw EEG windows (for fine-grained timing)

Output .npz contains:
- Fold_Data_DE: DE features (n_win, channels, bands)
- Fold_Data_Raw: Raw EEG (n_win, samples_per_window, channels)
- Fold_Label: Time-point labels (n_win, samples_per_window)
"""

import os
import re
import json
import numpy as np
import mne
from collections import defaultdict
from DE_PSD import DE_PSD


# ============================================================================
# File discovery (same as before)
# ============================================================================

def get_stems(raw_dir):
    stems = []
    for f in sorted(os.listdir(raw_dir)):
        if f.endswith('_raw.edf'):
            stems.append(f.replace('_raw.edf', ''))
    return stems


def group_by_subject(stems):
    groups = defaultdict(list)
    for stem in stems:
        m = re.match(r'(P\d+)', stem)
        if m is None:
            raise ValueError(f"Cannot extract subject ID from stem '{stem}'")
        groups[m.group(1)].append(stem)
    return dict(sorted(groups.items()))


# ============================================================================
# Read EDF and extract BOTH DE and raw EEG
# ============================================================================

EEG_KEEP = ["C3", "C4", "O1", "O2", "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
            "T3", "T4", "T5", "T6", "P3", "Pz", "P4", "Oz"]


def read_edf_dual(edf_path, window_sec=2.0, target_fs=200):
    """
    Read EDF and return BOTH:
    1. Windowed data for DE extraction
    2. Raw EEG windows
    
    Returns:
    --------
    windows_for_de : (n_win, n_ch, samples_per_win)
        For DE feature extraction
    raw_windows : (n_win, samples_per_win, n_ch)
        Raw EEG in time-major format for 1D CNN
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    
    # Force fixed channel order
    present = set(raw.ch_names)
    missing = [ch for ch in EEG_KEEP if ch not in present]
    if missing:
        raise RuntimeError(f"{os.path.basename(edf_path)} missing channels: {missing}")
    
    raw.pick(EEG_KEEP)
    raw.resample(target_fs, verbose=False)
    fs = int(raw.info['sfreq'])
    
    data = raw.get_data()  # (n_ch, total_samples)
    n_ch, total_samp = data.shape
    samples_per_win = int(window_sec * fs)
    n_windows = total_samp // samples_per_win
    
    # Format 1: Channel-major for DE (same as before)
    windows_for_de = np.zeros((n_windows, n_ch, samples_per_win), dtype=np.float64)
    
    # Format 2: Time-major for 1D CNN
    raw_windows = np.zeros((n_windows, samples_per_win, n_ch), dtype=np.float32)
    
    for i in range(n_windows):
        s = i * samples_per_win
        e = s + samples_per_win
        windows_for_de[i] = data[:, s:e]
        raw_windows[i] = data[:, s:e].T  # Transpose to (time, channels)
    
    return windows_for_de, raw_windows


def extract_DE_and_raw(edf_path, stft_para, cache_dir='./data/cache_multireso/'):
    """
    Extract both DE features and raw EEG, with caching.
    
    Returns:
    --------
    de_features : (n_win, n_ch, n_bands)
    raw_eeg : (n_win, samples_per_win, n_ch)
    """
    os.makedirs(cache_dir, exist_ok=True)
    stem = os.path.basename(edf_path).replace('_raw.edf', '')
    
    de_cache = os.path.join(cache_dir, f"{stem}_de.npy")
    raw_cache = os.path.join(cache_dir, f"{stem}_raw.npy")
    
    # Check cache
    if os.path.exists(de_cache) and os.path.exists(raw_cache):
        print(f"    [CACHE HIT] {stem}")
        return np.load(de_cache), np.load(raw_cache)
    
    # Read and process
    print(f"    [COMPUTING] {stem}")
    windows_for_de, raw_windows = read_edf_dual(
        edf_path,
        window_sec=stft_para['window'],
        target_fs=stft_para['fs']
    )
    
    # Compute DE features
    n_windows, n_ch, _ = windows_for_de.shape
    n_bands = len(stft_para['fStart'])
    de_features = np.zeros((n_windows, n_ch, n_bands), dtype=np.float32)
    
    for i in range(n_windows):
        _, de_features[i] = DE_PSD(windows_for_de[i], stft_para)
    
    # Save to cache
    np.save(de_cache, de_features)
    np.save(raw_cache, raw_windows)
    
    return de_features, raw_windows


# ============================================================================
# Labels (same as before)
# ============================================================================

def read_labels_timepoint(stem, labels_dir, n_windows, window_sec=2.0, fs=200):
    json_path = os.path.join(
        labels_dir,
        f"sleep_block_spindle_output_{stem}.json"
    )
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Label JSON not found: {json_path}")
    
    with open(json_path) as f:
        J = json.load(f)
    
    spindles = J.get("detected_spindles", [])
    samples_per_window = int(window_sec * fs)
    
    y = np.zeros((n_windows, samples_per_window), dtype=np.int32)
    
    if len(spindles) == 0:
        return y
    
    for spindle in spindles:
        start_sec = spindle["start"]
        end_sec = spindle["end"]
        
        start_sample = int(start_sec * fs)
        end_sample = int(end_sec * fs)
        
        for sample_idx in range(start_sample, end_sample):
            window_idx = sample_idx // samples_per_window
            within_window_idx = sample_idx % samples_per_window
            
            if window_idx < n_windows:
                y[window_idx, within_window_idx] = 1
    
    return y


# ============================================================================
# Process subject
# ============================================================================

def process_subject_multireso(stems, raw_dir, labels_dir, stft_para, cache_dir):
    """
    Process one subject, returning both DE and raw data.
    
    Returns:
    --------
    de_data : (total_windows, n_channels, n_bands)
    raw_data : (total_windows, samples_per_window, n_channels)
    labels : (total_windows, samples_per_window)
    """
    all_de = []
    all_raw = []
    all_label = []
    
    for stem in stems:
        edf_path = os.path.join(raw_dir, stem + '_raw.edf')
        print(f"    {stem} ...", end=' ', flush=True)
        
        de, raw = extract_DE_and_raw(edf_path, stft_para, cache_dir=cache_dir)
        n_win = de.shape[0]
        
        label = read_labels_timepoint(
            stem, labels_dir, n_win,
            window_sec=stft_para['window'],
            fs=stft_para['fs']
        )
        
        assert de.shape[0] == raw.shape[0] == label.shape[0], \
            f"{stem}: DE={de.shape[0]}, Raw={raw.shape[0]}, Label={label.shape[0]}"
        
        all_de.append(de)
        all_raw.append(raw)
        all_label.append(label)
        
        spindle_ratio = label.mean()
        print(f"windows={n_win}, spindle_ratio={spindle_ratio:.3f}")
    
    de_data = np.concatenate(all_de, axis=0)
    raw_data = np.concatenate(all_raw, axis=0)
    labels = np.concatenate(all_label, axis=0)
    
    return de_data, raw_data, labels


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    
    # Paths
    RAW_DIR = '../P002/edf/'
    LABELS_DIR = '../P002/labels/'
    SAVE_PATH = './data/SS3_multireso_19channels.npz'
    CACHE_DIR = './data/cache_multireso/'
    
    # DE parameters
    stft_para = {
        'stftn': 512,
        'fStart': [0.5, 2, 4, 6, 8, 11, 14, 22, 31],
        'fEnd': [4, 6, 8, 11, 14, 22, 31, 40, 50],
        'fs': 200,
        'window': 2,
    }
    
    # Discover files
    stems = get_stems(RAW_DIR)
    subjects = group_by_subject(stems)
    n_folds = len(subjects)
    
    print(f"Found {len(stems)} files across {n_folds} subjects")
    for subj, s_stems in subjects.items():
        print(f"  {subj}: {s_stems}")
    
    # Process each subject
    Fold_Data_DE = []
    Fold_Data_Raw = []
    Fold_Label = []
    Fold_Num = []
    
    for subj, s_stems in subjects.items():
        print(f"\n=== Subject {subj} ===")
        de, raw, labels = process_subject_multireso(
            s_stems, RAW_DIR, LABELS_DIR, stft_para, CACHE_DIR
        )
        
        Fold_Data_DE.append(de)
        Fold_Data_Raw.append(raw)
        Fold_Label.append(labels)
        Fold_Num.append(de.shape[0])
        
        print(f"  → windows: {de.shape[0]}, spindle ratio: {labels.mean():.3f}")
    
    Fold_Num = np.array(Fold_Num, dtype=int)
    
    # Normalize DE features (raw EEG stays unnormalized for 1D CNN)
    All_DE = np.concatenate(Fold_Data_DE, axis=0)
    mean_de = All_DE.mean(axis=0)
    std_de = All_DE.std(axis=0)
    std_de[std_de == 0] = 1.0
    
    Fold_Data_DE = [(d - mean_de) / std_de for d in Fold_Data_DE]
    
    # Normalize raw EEG per-channel (z-score)
    All_Raw = np.concatenate(Fold_Data_Raw, axis=0)  # (N, T, C)
    mean_raw = All_Raw.mean(axis=(0, 1), keepdims=True)  # (1, 1, C)
    std_raw = All_Raw.std(axis=(0, 1), keepdims=True)
    std_raw[std_raw == 0] = 1.0
    
    Fold_Data_Raw = [(r - mean_raw) / std_raw for r in Fold_Data_Raw]
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Total folds: {n_folds}")
    print(f"Total windows: {Fold_Num.sum()}")
    print(f"Fold_Num: {Fold_Num}")
    all_labels = np.concatenate(Fold_Label, axis=0)
    print(f"Overall spindle ratio: {all_labels.mean():.3f}")
    print(f"DE features shape: {Fold_Data_DE[0].shape}")
    print(f"Raw EEG shape: {Fold_Data_Raw[0].shape}")
    print(f"Labels shape: {Fold_Label[0].shape}")
    
    # Save
    np.savez(
        SAVE_PATH,
        Fold_Num=Fold_Num,
        Fold_Data_DE=np.array(Fold_Data_DE, dtype=object),
        Fold_Data_Raw=np.array(Fold_Data_Raw, dtype=object),
        Fold_Label=np.array(Fold_Label, dtype=object),
        # Save normalization params for later use
        de_mean=mean_de,
        de_std=std_de,
        raw_mean=mean_raw,
        raw_std=std_raw,
    )
    
    print(f"\nSaved → {SAVE_PATH}")
