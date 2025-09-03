"""
Enhanced data loading and preprocessing for time-based EEG spindle detection
"""

import mne
import numpy as np
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from config import Config
from typing import List, Tuple, Optional

class EEGDataset(Dataset):
    """Enhanced PyTorch Dataset for in-memory EEG data"""
    def __init__(self, X, y, transform=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        self.transform = transform
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

class ChunkedEEGDataset(Dataset):
    """Memory-efficient PyTorch Dataset with data augmentation support"""
    def __init__(self, split: str, model_type: str = "UNet1D", config=None, augment: bool = False):
        self.config = config or Config()
        self.model_type = model_type
        self.augment = augment
        self.split = split
        # Data paths
        self.X_path = self.config.SAVE_DIR / f"X_{split}.npy"
        self.y_path = self.config.SAVE_DIR / f"y_{split}.npy"
        if not self.X_path.exists() or not self.y_path.exists():
            raise FileNotFoundError(f"Data files not found for split '{split}'. Run data preparation first.")
        # Load labels and set up memory mapping
        y_all = np.load(self.y_path)
        X_all = np.load(self.X_path, mmap_mode="r")
        # Apply downsampling if configured
        if self.config.DOWNSAMPLE and split == 'train':
            self.indices = self._downsample_majority_class(y_all)
            print(f"Downsampling applied: {len(self.indices)} samples selected from {len(y_all)}")
        else:
            self.indices = np.arange(len(y_all))
        self.X_memmap = X_all
        self.y_memmap = y_all
        # Compute data statistics for normalization
        self._compute_statistics()
        print(f"Dataset {split}: {len(self.indices)} samples, Positive: {int(np.sum(self.y_memmap[self.indices]))}, Negative: {len(self.indices) - int(np.sum(self.y_memmap[self.indices]))}")

    def _downsample_majority_class(self, y_all):
        """Downsample majority class to balance dataset"""
        pos_idx = np.where(y_all == 1)[0]
        neg_idx = np.where(y_all == 0)[0]
        if len(neg_idx) < len(pos_idx):
            print(f"Warning: Only {len(neg_idx)} negative samples available, but need {len(pos_idx)}")
            neg_sampled_idx = neg_idx
        else:
            np.random.seed(self.config.get('seed', 42))
            neg_sampled_idx = np.random.choice(neg_idx, size=len(pos_idx), replace=False)
        selected_idx = np.concatenate([pos_idx, neg_sampled_idx])
        np.random.shuffle(selected_idx)
        return selected_idx

    def _compute_statistics(self):
        """Compute dataset statistics for normalization"""
        sample_size = min(1000, len(self.indices))
        sample_indices = np.random.choice(self.indices, sample_size, replace=False)
        sample_data = []
        for idx in sample_indices:
            sample_data.append(self.X_memmap[idx])
        sample_data = np.array(sample_data)
        self.data_mean = np.mean(sample_data, axis=(0, -1), keepdims=True)
        self.data_std = np.std(sample_data, axis=(0, -1), keepdims=True)
        self.data_std = np.maximum(self.data_std, 1e-8)  # avoid division by zero

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if idx >= len(self.indices):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.indices)}")
        actual_idx = self.indices[idx]
        x = self.X_memmap[actual_idx].copy()
        y = self.y_memmap[actual_idx]
        if self.augment and self.split == 'train':
            x = self._apply_augmentation(x)
        # Normalize
        x = (x - self.data_mean) / self.data_std
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor([y], dtype=torch.float32)
        return x_tensor, y_tensor

    def _apply_augmentation(self, x):
        """Apply data augmentation techniques"""
        # Gaussian noise
        noise_std = self.config.get('advanced.augmentation.noise_std', 0.01)
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, x.shape)
            x = x + noise
        # Amplitude scaling
        scale_range = self.config.get('advanced.augmentation.amplitude_scale_range', [0.9, 1.1])
        if scale_range[0] != 1.0 or scale_range[1] != 1.0:
            scale_factor = np.random.uniform(scale_range[0], scale_range[1])
            x = x * scale_factor
        # Time shifting (circular shift)
        max_shift = self.config.get('advanced.augmentation.time_shift_max', 0.1)
        if max_shift > 0:
            shift_samples = int(np.random.uniform(-max_shift, max_shift) * x.shape[-1])
            if shift_samples != 0:
                x = np.roll(x, shift_samples, axis=-1)
        return x

def load_and_preprocess_data(config):
    """Load and preprocess EEG data using the provided configuration"""
    print("Loading EDF file...")
    if not config.EDF_PATH.exists():
        raise FileNotFoundError(f"EDF file not found: {config.EDF_PATH}")
    try:
        raw = mne.io.read_raw_edf(config.EDF_PATH, preload=True, verbose=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load EDF file: {e}")
    sfreq = raw.info["sfreq"]
    print(f"Original sampling frequency: {sfreq} Hz")
    print(f"Recording duration: {raw.times[-1] / 3600:.2f} hours")
    # Channel selection with fallback options
    available_channels = [ch for ch in config.EEG_CHANNELS if ch in raw.ch_names]
    if not available_channels:
        print("Warning: No configured EEG channels found. Using all available channels.")
        available_channels = raw.ch_names
    print(f"Available EEG channels ({len(available_channels)}): {available_channels}")
    try:
        raw.pick_channels(available_channels)
    except ValueError as e:
        print(f"Warning: Some channels not found: {e}")
        valid_channels = [ch for ch in available_channels if ch in raw.ch_names]
        if not valid_channels:
            raise ValueError("No valid EEG channels found")
        raw.pick_channels(valid_channels)
        available_channels = valid_channels
    print(f"Using {len(available_channels)} EEG channels: {available_channels}")
    # Bandpass filtering
    filter_low = config.FILTER_LOW
    filter_high = config.FILTER_HIGH
    print(f"Applying bandpass filter: {filter_low}-{filter_high} Hz")
    try:
        raw.filter(filter_low, filter_high, fir_design='firwin', verbose=False)
    except Exception as e:
        print(f"Warning: Filtering failed with firwin, trying IIR: {e}")
        raw.filter(filter_low, filter_high, method='iir', verbose=False)
    # Check for data quality issues
    data = raw.get_data()
    if np.any(np.isnan(data)):
        print("Warning: NaN values detected in data. Interpolating...")
        mask = np.isnan(data)
        data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])
        raw._data = data
    data_range = np.percentile(data, [1, 99])
    if data_range[1] - data_range[0] > 1e6:
        print(f"Warning: Large data range detected ({data_range[1] - data_range[0]:.2e}). Consider checking units.")
    return raw, sfreq

def load_spindle_labels(config):
    """Load spindle annotations using the provided configuration"""
    print("Loading spindle labels...")
    if not config.JSON_PATH.exists():
        raise FileNotFoundError(f"JSON file not found: {config.JSON_PATH}")
    try:
        with open(config.JSON_PATH) as f:
            spindle_data = json.load(f)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON file: {e}")
    if "detected_spindles" not in spindle_data:
        raise ValueError("JSON file must contain 'detected_spindles' key")
    spindles = []
    invalid_count = 0
    for i, s in enumerate(spindle_data["detected_spindles"]):
        if not isinstance(s, dict) or "start" not in s or "end" not in s:
            invalid_count += 1
            continue
        start, end = s["start"], s["end"]
        if end <= start:
            print(f"Warning: Invalid spindle {i}: end ({end}) <= start ({start})")
            invalid_count += 1
            continue
        if end - start > 10:
            print(f"Warning: Very long spindle {i}: {end - start:.1f}s")
        spindles.append((start, end))
    if invalid_count > 0:
        print(f"Warning: Skipped {invalid_count} invalid spindle annotations")
    spindles.sort(key=lambda x: x[0])
    print(f"Loaded {len(spindles)} valid spindle annotations")
    if spindles:
        durations = [end - start for start, end in spindles]
        print("Spindle statistics:")
        print(f"  Duration: {np.mean(durations):.2f} ± {np.std(durations):.2f}s")
        print(f"  Range: {np.min(durations):.2f}s - {np.max(durations):.2f}s")
        print(f"  Time span: {spindles[0][0]:.1f}s - {spindles[-1][1]:.1f}s")
    return spindles

def is_spindle_window(start_time: float, end_time: float, spindles: List[Tuple[float, float]], overlap_threshold=0.7):
    """Check if a window contains a spindle with sufficient overlap"""
    window_duration = end_time - start_time
    for s_start, s_end in spindles:
        overlap_start = max(start_time, s_start)
        overlap_end = min(end_time, s_end)
        overlap = max(0, overlap_end - overlap_start)
        if overlap / window_duration >= overlap_threshold:
            return True
    return False

def create_windows(raw, sfreq, spindles, start_sec: float, end_sec: float, prefix: str, model_type: str, config):
    """Create overlapping windows with labels using the provided configuration"""
    print(f"Creating windows for {prefix} split: {start_sec / 3600:.1f}h - {end_sec / 3600:.1f}h")
    max_time = float(raw.times[-1])
    if end_sec > max_time:
        print(f"Warning: Requested end time ({end_sec:.2f}s) exceeds recording duration ({max_time:.2f}s); trimming.")
        end_sec = max_time
    if start_sec >= end_sec:
        raise ValueError(f"Invalid time range: start ({start_sec}) >= end ({end_sec})")

    # Window/step in samples
    win_samples = int(round(config.WINDOW_SEC * sfreq))
    step_samples = int(round(config.STEP_SEC * sfreq))
    if win_samples <= 0 or step_samples <= 0:
        raise ValueError(f"Non-positive window/step: window={win_samples}, step={step_samples}")

    start_sample = int(round(start_sec * sfreq))
    end_sample = int(round(end_sec * sfreq))
    max_start = min(end_sample - win_samples, raw.n_times - win_samples)
    if start_sample > max_start:
        raise ValueError(f"No valid windows can be created in time range {start_sec}-{end_sec}s with window={config.WINDOW_SEC}s and step={config.STEP_SEC}s")

    print(f"Window parameters: {config.WINDOW_SEC}s window, {config.STEP_SEC}s step")
    approx_windows = max(0, (max_start - start_sample) // step_samples + 1)
    print(f"Expected windows: ~{approx_windows}")

    X, y = [], []
    skipped = 0

    for s in range(start_sample, max_start + 1, step_samples):
        try:
            seg = raw.get_data(start=s, stop=s + win_samples)  # [C, T]
            if not np.isfinite(seg).all():
                skipped += 1
                continue
            t0 = s / sfreq
            t1 = (s + win_samples) / sfreq
            label = 1 if is_spindle_window(t0, t1, spindles, config.OVERLAP_THRESHOLD) else 0
            X.append(seg.astype(np.float32))
            y.append(label)
        except Exception as e:
            print(f"Warning: skipping window at sample {s}: {e}")
            skipped += 1

    if len(X) == 0:
        raise ValueError(f"No valid windows created for {prefix} in {start_sec}-{end_sec}s")

    if skipped:
        print(f"Skipped {skipped} windows due to data issues")

    X = np.stack(X, axis=0)  # [N, C, T]
    y = np.asarray(y, dtype=np.float32)  # [N]

    # Shape to model
    if model_type == "SpindleCNN":
        X = X[:, None, :, :]  # [N, 1, C, T]
    elif model_type == "UNet1D":
        pass  # keep [N, C, T]
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    pos = int(y.sum())
    neg = int(len(y) - pos)
    print(f"Final data shape: X={X.shape}, y={y.shape} | Pos={pos} ({pos/len(y)*100:.1f}%), Neg={neg} ({neg/len(y)*100:.1f}%)")

    save_dir = config.SAVE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / f"X_{prefix}.npy", X)
    np.save(save_dir / f"y_{prefix}.npy", y)
    print(f"Saved {prefix} split to {save_dir}")

    return X, y

def get_data_loaders(model_type: str = "UNet1D", augment_train: bool = False, config=None):
    """Create PyTorch DataLoaders for train, val, test splits"""
    if config is None:
        config = Config()
    print("Creating data loaders...")
    train_dataset = ChunkedEEGDataset("train", model_type=model_type, config=config, augment=augment_train)
    val_dataset = EEGDataset(np.load(config.SAVE_DIR / "X_val.npy"), np.load(config.SAVE_DIR / "y_val.npy"))
    test_dataset = EEGDataset(np.load(config.SAVE_DIR / "X_test.npy"), np.load(config.SAVE_DIR / "y_test.npy"))
    num_workers = config.get('hardware.num_workers', 4)
    pin_memory = config.get('hardware.pin_memory', True)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    print("Data loaders created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test:  {len(test_dataset)} samples, {len(test_loader)} batches")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Workers: {num_workers}")
    return train_loader, val_loader, test_loader

def analyze_dataset_statistics(data_loaders, spindle_annotations, config):
    """Analyze and report dataset statistics"""
    train_loader, val_loader, test_loader = data_loaders
    print("\n" + "=" * 50)
    print("DATASET STATISTICS")
    print("=" * 50)
    stats = {}
    for name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        pos_count = 0
        total_count = 0
        for _, labels in loader:
            pos_count += torch.sum(labels).item()
            total_count += labels.numel()
        pos_ratio = pos_count / total_count if total_count > 0 else 0
        stats[name] = {
            'total_samples': total_count,
            'positive_samples': pos_count,
            'negative_samples': total_count - pos_count,
            'positive_ratio': pos_ratio,
            'class_imbalance': (total_count - pos_count) / max(pos_count, 1)
        }
        print(f"{name.upper()} SET:")
        print(f"  Total samples: {total_count}")
        print(f"  Positive: {pos_count} ({pos_ratio * 100:.1f}%)")
        print(f"  Negative: {total_count - pos_count} ({(1 - pos_ratio) * 100:.1f}%)")
        print(f"  Class imbalance ratio: {stats[name]['class_imbalance']:.2f}:1")
    if spindle_annotations:
        durations = [end - start for start, end in spindle_annotations]
        print("\nSPINDLE ANNOTATIONS:")
        print(f"  Total spindles: {len(spindle_annotations)}")
        print(f"  Duration: {np.mean(durations):.2f} ± {np.std(durations):.2f}s")
        print(f"  Total spindle time: {np.sum(durations):.1f}s ({np.sum(durations) / 60:.1f}min)")
    return stats
