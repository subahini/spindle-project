"""
process_SS3.py

Pipeline:  data/raw/*.edf  +  data/labels/*.json  →  .npz (k-fold by subject)

Layout expected:
    data/
    ├── raw/       P002_1_raw.edf, P002_2_raw.edf, P003_1_raw.edf, ...
    └── labels/    sleep_block_spindle_output_P002_1.json, ...

Each EDF stem like "P002_1" maps to:
    raw:    P002_1_raw.edf
    label:  sleep_block_spindle_output_P002_1.json

Subjects are grouped by the Pxxx prefix.  Each subject becomes one fold.
Output .npz has the same keys train.py expects:
    Fold_Num   – (n_folds,) int  — number of windows per fold
    Fold_Data  – object array of length n_folds, each element (n_win, channels, bands)
    Fold_Label – object array of length n_folds, each element (n_win, 1)
"""

import os
import re
import json
import numpy as np
import mne                       # pip install mne
from collections import defaultdict
from DE_PSD import DE_PSD


# ===========================================================================
# 1.  Discover files and group by subject
# ===========================================================================

def get_stems(raw_dir):
    """
    Scan raw_dir for *_raw.edf, return sorted list of stems.
    P002_1_raw.edf  →  "P002_1"
    """
    stems = []
    for f in sorted(os.listdir(raw_dir)):
        if f.endswith('_raw.edf'):
            stems.append(f.replace('_raw.edf', ''))
    return stems


def group_by_subject(stems):
    """
    Group stems by subject prefix (Pxxx).
    Returns OrderedDict:  { "P002": ["P002_1", "P002_2"], "P003": [...], ... }
    """
    groups = defaultdict(list)
    for stem in stems:
        m = re.match(r'(P\d+)', stem)
        if m is None:
            raise ValueError(f"Cannot extract subject ID from stem '{stem}'. "
                             f"Expected format like P002_1")
        groups[m.group(1)].append(stem)
    # sort subjects so fold order is deterministic
    return dict(sorted(groups.items()))


# ===========================================================================
# 2.  Read one EDF → segment into 2-second windows
#     Returns (n_windows, n_channels, samples_per_window)
# ===========================================================================

def read_edf_windows(edf_path, window_sec=2.0, target_fs=200):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.resample(target_fs, verbose=False)
    fs = int(raw.get_sfreq())

    data = raw.get_data()                          # (n_channels, total_samples)
    n_ch, total_samp = data.shape
    samples_per_win  = int(window_sec * fs)
    n_windows        = total_samp // samples_per_win   # drop leftover tail

    windows = np.zeros((n_windows, n_ch, samples_per_win), dtype=np.float64)
    for i in range(n_windows):
        s = i * samples_per_win
        windows[i] = data[:, s:s + samples_per_win]

    return windows


# ===========================================================================
# 3.  Extract DE features for one EDF file
#     Returns (n_windows, n_channels, n_bands)
# ===========================================================================

def extract_DE(edf_path, stft_para):
    windows = read_edf_windows(edf_path,
                               window_sec=stft_para['window'],
                               target_fs=stft_para['fs'])
    n_windows, n_ch, _ = windows.shape
    n_bands = len(stft_para['fStart'])

    de_all = np.zeros((n_windows, n_ch, n_bands), dtype=np.float64)
    for i in range(n_windows):
        # DE_PSD expects (n_channels, n_samples) for one window
        _, de_all[i] = DE_PSD(windows[i], stft_para)

    return de_all


# ===========================================================================
# 4.  Read spindle labels from JSON for one file
#     Returns (n_windows, 1) int32
# ===========================================================================

def read_labels(stem, labels_dir, n_windows, window_sec=2.0, overlap_thr_sec=0.5):
    """
    A window is labelled 1 if any detected spindle overlaps it by
    >= overlap_thr_sec seconds.
    """
    json_path = os.path.join(labels_dir,
                             f"sleep_block_spindle_output_{stem}.json")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Label JSON not found: {json_path}")

    with open(json_path) as f:
        J = json.load(f)

    spindles = J.get("detected_spindles", [])
    y = np.zeros((n_windows, 1), dtype=np.int32)

    if len(spindles) == 0:
        return y

    starts = np.array([s["start"] for s in spindles], dtype=float)
    ends   = np.array([s["end"]   for s in spindles], dtype=float)

    for i in range(n_windows):
        w0 = i * window_sec
        w1 = w0 + window_sec
        overlap = np.maximum(0.0, np.minimum(w1, ends) - np.maximum(w0, starts))
        if np.any(overlap >= overlap_thr_sec):
            y[i, 0] = 1

    return y


# ===========================================================================
# 5.  Process one subject (all its EDF blocks) → concatenated data + labels
# ===========================================================================

def process_subject(stems, raw_dir, labels_dir, stft_para):
    """
    For one subject, process all blocks, stack them vertically.
    Returns:
        data   (total_windows, n_channels, n_bands)
        labels (total_windows, 1)
    """
    all_data  = []
    all_label = []

    for stem in stems:
        edf_path = os.path.join(raw_dir, stem + '_raw.edf')
        print(f"    {stem} ...", end=' ', flush=True)

        de = extract_DE(edf_path, stft_para)
        n_win = de.shape[0]

        label = read_labels(stem, labels_dir, n_win,
                            window_sec=stft_para['window'],
                            overlap_thr_sec=0.5)

        assert de.shape[0] == label.shape[0], (
            f"{stem}: feature windows={de.shape[0]}, label windows={label.shape[0]}"
        )

        all_data.append(de)
        all_label.append(label)
        print(f"windows={n_win}  spindle_ratio={label.mean():.3f}")

    data   = np.concatenate(all_data,  axis=0)
    labels = np.concatenate(all_label, axis=0)
    return data, labels


# ===========================================================================
# 6.  Main
# ===========================================================================

if __name__ == "__main__":

    # --- paths -----------------------------------------------------------
    RAW_DIR    = './data/raw/'
    LABELS_DIR = './data/labels/'
    SAVE_PATH  = './data/SS3_DE_26channels.npz'

    # --- DE/PSD parameters -----------------------------------------------
    stft_para = {
        'stftn' : 7680,
        'fStart': [0.5, 2, 4,  6,  8, 11, 14, 22, 31],
        'fEnd'  : [4,   6, 8, 11, 14, 22, 31, 40, 50],
        'fs'    : 200,
        'window': 2,        # seconds — must match labelling window
    }

    # --- discover and group ----------------------------------------------
    stems   = get_stems(RAW_DIR)
    subjects = group_by_subject(stems)
    n_folds  = len(subjects)

    print(f"Found {len(stems)} files across {n_folds} subjects (= {n_folds} folds)")
    for subj, s_stems in subjects.items():
        print(f"  {subj}: {s_stems}")

    # --- process each subject --------------------------------------------
    Fold_Data  = []     # list of (n_win, ch, bands)  — one per fold
    Fold_Label = []     # list of (n_win, 1)          — one per fold
    Fold_Num   = []     # window count per fold

    for subj, s_stems in subjects.items():
        print(f"\n=== Subject {subj} ===")
        data, labels = process_subject(s_stems, RAW_DIR, LABELS_DIR, stft_para)
        Fold_Data.append(data)
        Fold_Label.append(labels)
        Fold_Num.append(data.shape[0])
        print(f"  → total windows: {data.shape[0]}, spindle ratio: {labels.mean():.3f}")

    Fold_Num = np.array(Fold_Num, dtype=int)

    # --- normalize (global mean/std across ALL folds, same as original) --
    All_Data = np.concatenate(Fold_Data, axis=0)
    mean = All_Data.mean(axis=0)
    std  = All_Data.std(axis=0)
    std[std == 0] = 1.0

    Fold_Data = [(d - mean) / std for d in Fold_Data]

    # --- summary ---------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Total folds : {n_folds}")
    print(f"Total windows: {Fold_Num.sum()}")
    print(f"Fold_Num     : {Fold_Num}")
    all_labels = np.concatenate(Fold_Label, axis=0)
    print(f"Overall spindle ratio: {all_labels.mean():.3f}")

    # --- save ------------------------------------------------------------
    np.savez(
        SAVE_PATH,
        Fold_Num   = Fold_Num,
        Fold_Data  = np.array(Fold_Data,  dtype=object),
        Fold_Label = np.array(Fold_Label, dtype=object),
    )
    print(f"\nSaved → {SAVE_PATH}")