"""
Data loading and preprocessing for EEG spindle detection
"""
import mne
import numpy as np
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from config import *


class EEGDataset(Dataset):
    """PyTorch Dataset for EEG data"""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_and_preprocess_data():
    """Load and preprocess EEG data"""
    config = Config()

    print("Loading EDF file...")
    if not config.EDF_PATH.exists():
        raise FileNotFoundError(f"EDF file not found: {config.EDF_PATH}")

    # Load EDF file
    raw = mne.io.read_raw_edf(config.EDF_PATH, preload=True, verbose=False)
    sfreq = raw.info["sfreq"]

    # Select available EEG channels
    available_channels = [ch for ch in config.EEG_CHANNELS if ch in raw.ch_names]
    if not available_channels:
        raise ValueError("No EEG channels found in the data")

    print(f"Using {len(available_channels)} EEG channels: {available_channels}")
    raw.pick_channels(available_channels)

    # Apply bandpass filter
    print(f"Applying bandpass filter: {config.FILTER_LOW}-{config.FILTER_HIGH} Hz")
    raw.filter(config.FILTER_LOW, config.FILTER_HIGH, fir_design='firwin', verbose=False)

    return raw, sfreq


def load_spindle_labels():
    """Load spindle annotations from JSON file"""
    config = Config()

    print("Loading spindle labels...")
    if not config.JSON_PATH.exists():
        raise FileNotFoundError(f"JSON file not found: {config.JSON_PATH}")

    with open(config.JSON_PATH) as f:
        spindle_data = json.load(f)

    spindles = [(s["start"], s["end"]) for s in spindle_data["detected_spindles"]]
    print(f"Loaded {len(spindles)} spindle annotations")
    return spindles


def is_spindle_window(start_time, end_time, spindles, overlap_threshold=0.7):
    """Check if window contains spindle with sufficient overlap"""
    window_duration = end_time - start_time

    for s_start, s_end in spindles:
        overlap = max(0, min(end_time, s_end) - max(start_time, s_start))
        if overlap / window_duration >= overlap_threshold:
            return True
    return False


def create_windows(raw, sfreq, spindles, start_sec, end_sec, prefix):
    """Create overlapping windows with labels"""
    config = Config()

    print(f"Creating windows for {prefix} split: {start_sec / 3600:.1f}h - {end_sec / 3600:.1f}h")

    X, y = [], []
    win_samples = int(config.WINDOW_SEC * sfreq)
    step_samples = int(config.STEP_SEC * sfreq)
    start_sample = int(start_sec * sfreq)
    end_sample = int(end_sec * sfreq)

    # Ensure we don't exceed data length
    max_start = min(end_sample - win_samples, raw.n_times - win_samples)

    for s in range(start_sample, max_start, step_samples):
        try:
            segment = raw.get_data(start=s, stop=s + win_samples)
            t0 = s / sfreq
            t1 = (s + win_samples) / sfreq

            label = 1 if is_spindle_window(t0, t1, spindles, config.OVERLAP_THRESHOLD) else 0
            X.append(segment)
            y.append(label)

        except Exception as e:
            print(f"Warning: Skipping window at {s}: {e}")
            continue

    if not X:
        raise ValueError(f"No valid windows created for {prefix}")

    X = np.array(X)[:, np.newaxis, :, :]  # Add channel dimension
    y = np.array(y)

    # Save windows
    np.save(config.SAVE_DIR / f"X_{prefix}.npy", X)
    np.save(config.SAVE_DIR / f"y_{prefix}.npy", y)

    pos_count = np.sum(y)
    neg_count = len(y) - pos_count
    print(f"Saved {prefix}: {X.shape}, Positive: {pos_count}, Negative: {neg_count}")

    return X, y


def get_data_loaders():
    """Create PyTorch data loaders"""
    config = Config()

    print("Loading saved windows...")
    X_train = np.load(config.SAVE_DIR / "X_train.npy")
    y_train = np.load(config.SAVE_DIR / "y_train.npy")
    X_val = np.load(config.SAVE_DIR / "X_val.npy")
    y_val = np.load(config.SAVE_DIR / "y_val.npy")
    X_test = np.load(config.SAVE_DIR / "X_test.npy")
    y_test = np.load(config.SAVE_DIR / "y_test.npy")

    # Create datasets
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)
    test_dataset = EEGDataset(X_test, y_test)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Training data: {X_train.shape}")
    print(f"Validation data: {X_val.shape}")
    print(f"Test data: {X_test.shape}")

    return train_loader, val_loader, test_loader