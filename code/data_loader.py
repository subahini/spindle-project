"""
Data loading and preprocessing for EEG spindle detection
"""

import mne
import numpy as np
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from config import Config


class EEGDataset(Dataset):
    """Basic PyTorch Dataset for in-memory EEG data"""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ChunkedEEGDataset(Dataset):
    """Memory-efficient PyTorch Dataset that loads EEG samples on-the-fly"""

    def __init__(self, split, model_type="UNet1D"):
        self.config = Config()
        self.model_type = model_type
        self.X_path = self.config.SAVE_DIR / f"X_{split}.npy"
        self.y_path = self.config.SAVE_DIR / f"y_{split}.npy"

       # self.X_memmap = np.load(self.X_path, mmap_mode="r")
        #self.y_memmap = np.load(self.y_path, mmap_mode="r")
            # downsampling...
        y_all = np.load(self.y_path)
        X_all = np.load(self.X_path, mmap_mode="r")

        if self.config.DOWNSAMPLE:
            pos_idx = np.where(y_all == 1)[0]
            neg_idx = np.where(y_all == 0)[0]
            np.random.seed(42)
            neg_sampled_idx = np.random.choice(neg_idx, size=len(pos_idx), replace=False)
            selected_idx = np.concatenate([pos_idx, neg_sampled_idx])
            np.random.shuffle(selected_idx)

            self.indices = selected_idx
        else:
            self.indices = np.arange(len(y_all))

        self.X_memmap = X_all
        self.y_memmap = y_all

    def __len__(self):
        return len(self.y_memmap)

    def __getitem__(self, idx):
        idx = self.indices[idx]

        x = self.X_memmap[idx]
        y = self.y_memmap[idx]

        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor([y], dtype=torch.float32)
        return x_tensor, y_tensor


def load_and_preprocess_data():
    """Load and preprocess EEG data"""
    config = Config()

    print("Loading EDF file...")
    if not config.EDF_PATH.exists():
        raise FileNotFoundError(f"EDF file not found: {config.EDF_PATH}")

    raw = mne.io.read_raw_edf(config.EDF_PATH, preload=True, verbose=False)
    sfreq = raw.info["sfreq"]

    available_channels = [ch for ch in config.EEG_CHANNELS if ch in raw.ch_names]
    if not available_channels:
        raise ValueError("No EEG channels found in the data")

    print(f"Using {len(available_channels)} EEG channels: {available_channels}")
    raw.pick_channels(available_channels)

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


def create_windows(raw, sfreq, spindles, start_sec, end_sec, prefix, model_type="UNet1D"):
    """Create overlapping windows and save them"""
    config = Config()
    print(f"Creating windows for {prefix} split: {start_sec / 3600:.1f}h - {end_sec / 3600:.1f}h")

    X, y = [], []
    win_samples = int(config.WINDOW_SEC * sfreq)
    step_samples = int(config.STEP_SEC * sfreq)
    start_sample = int(start_sec * sfreq)
    end_sample = int(end_sec * sfreq)
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
            print(f"Skipping window at {s}: {e}")
            continue

    if not X:
        raise ValueError(f"No valid windows created for {prefix}")

    X = np.array(X)

    if model_type == "SpindleCNN":
        X = X[:, np.newaxis, :, :]  # → [N, 1, 16, 400]
    elif model_type == "UNet1D":
        pass  # keep as [N, 16, 400]
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    y = np.array(y)

    np.save(config.SAVE_DIR / f"X_{prefix}.npy", X)
    np.save(config.SAVE_DIR / f"y_{prefix}.npy", y)

    print(f"Saved {prefix}: {X.shape}, Positive: {np.sum(y)}, Negative: {len(y) - np.sum(y)}")
    return X, y


def get_data_loaders(model_type="UNet1D"):
    """Return PyTorch DataLoaders"""
    config = Config()
    print("Creating chunked DataLoaders...")

    train_dataset = ChunkedEEGDataset("train", model_type=model_type)
    val_dataset = EEGDataset(
        np.load(config.SAVE_DIR / "X_val.npy"),
        np.load(config.SAVE_DIR / "y_val.npy")
    )
    test_dataset = EEGDataset(
        np.load(config.SAVE_DIR / "X_test.npy"),
        np.load(config.SAVE_DIR / "y_test.npy")
    )

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader
