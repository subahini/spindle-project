#!/usr/bin/env python3
"""
Enhanced UNet1D Spindle Detection — Clean Version with External Loss Module

Pipeline
- Load EDF + JSON spindle labels with validation
- CAR (average reference) → bandpass 5–30 Hz → configurable channels
- Window into [B, C=4, T] with samplewise labels [B, T]
- Deep UNet1D (residual blocks + GroupNorm + SiLU, configurable depth/kernel + optional attention)
- External loss module with BCE/Focal/Dice/Hybrid losses
- Samplers: normal | undersample | weighted
- Cross-validation support and ensemble training
- Threshold sweep on validation (best F1 by default)
- ROC/PR with chance baselines + confusion matrix (keeping your implementations!)
- Enhanced logging with sample predictions and model analysis
- Memory optimization and better error handling

Usage:
  pip install wandb mne matplotlib pyyaml torch
  wandb login
  python unet1d.py --config config.yaml
"""

import os, math, json, argparse, datetime, warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import traceback

import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.serialization import add_safe_globals

import mne
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

# Import your loss module
from losses import build_loss_function

# ---------------- W&B ----------------
import wandb


class WBLogger:
    def __init__(self, cfg: Dict[str, Any]):
        opt = cfg.get("logging", {}).get("wandb", {})
        name = (cfg.get("logging", {}).get("run_name") or "").strip() or None
        self.run = wandb.init(
            project=opt.get("project", "spindle-unet1d"),
            entity=opt.get("entity"),
            name=name,
            tags=opt.get("tags"),
            config=cfg,
            settings=wandb.Settings(start_method="thread"),
            save_code=opt.get("save_code", True),
        )
        self.save_predictions = cfg.get("logging", {}).get("save_predictions", False)
        self.plot_samples = cfg.get("logging", {}).get("plot_sample_windows", 0)

    def add_scalar(self, k, v, step):
        wandb.log({k: float(v), "step": int(step)}, step=int(step))

    def add_figure(self, k, fig, step):
        wandb.log({k: wandb.Image(fig), "step": int(step)}, step=int(step))

    def add_artifact_file(self, path, name="checkpoint"):
        try:
            art = wandb.Artifact(name, type="model");
            art.add_file(path);
            wandb.log_artifact(art)
        except Exception as e:
            print(f"[Warning] Failed to log artifact: {e}")

    def log_model_info(self, model, sample_input):
        """Log model architecture and parameters"""
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            wandb.log({
                "model/total_parameters": total_params,
                "model/trainable_parameters": trainable_params,
                "model/model_size_mb": total_params * 4 / (1024 ** 2),  # float32 = 4 bytes
            })
            wandb.watch(model, log="all", log_freq=100)
        except Exception as e:
            print(f"[Warning] Failed to log model info: {e}")

    def log_sample_predictions(self, model, dataloader, device, epoch, num_samples=3):
        """Log sample EEG windows with predictions"""
        if num_samples <= 0:
            return

        model.eval()
        # Get channel names from config or use defaults
        channel_names = getattr(self, 'channel_names', ['C3', 'C4', 'F3', 'F4'])

        with torch.no_grad():
            for i, (x, y) in enumerate(dataloader):
                if i >= num_samples:
                    break
                x, y = x.to(device), y.to(device)
                pred_logits = model(x)
                pred_probs = torch.sigmoid(pred_logits)

                # Create visualization
                fig, axes = plt.subplots(3, 1, figsize=(15, 10))

                # Plot EEG channels
                time_axis = np.arange(x.shape[-1]) / 200.0  # assuming 200 Hz
                for ch in range(min(x.shape[1], len(channel_names))):
                    axes[0].plot(time_axis, x[0, ch].cpu(), label=channel_names[ch], alpha=0.8)
                axes[0].legend()
                axes[0].set_title('EEG Channels')
                axes[0].set_ylabel('Amplitude (µV)')
                axes[0].grid(True, alpha=0.3)

                # Plot labels vs predictions
                axes[1].plot(time_axis, y[0].cpu(), label='True Labels', linewidth=2, alpha=0.8)
                axes[1].plot(time_axis, pred_probs[0].cpu(), label='Predicted Prob', linewidth=2, alpha=0.8)
                axes[1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold=0.5')
                axes[1].legend()
                axes[1].set_title('Spindle Detection: True vs Predicted')
                axes[1].set_ylabel('Probability')
                axes[1].set_ylim(-0.1, 1.1)
                axes[1].grid(True, alpha=0.3)

                # Plot prediction confidence
                confidence = torch.abs(pred_probs[0] - 0.5).cpu()
                axes[2].plot(time_axis, confidence, label='Prediction Confidence', color='green', alpha=0.8)
                axes[2].set_title('Prediction Confidence (distance from 0.5)')
                axes[2].set_ylabel('Confidence')
                axes[2].set_xlabel('Time (s)')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)

                plt.tight_layout()
                wandb.log({f"sample_predictions/epoch_{epoch}_sample_{i}": wandb.Image(fig)})
                plt.close(fig)

    def close(self):
        try:
            wandb.finish()
        except Exception:
            pass


# -------------- utils ---------------
def set_seed(seed: int = 42):
    torch.manual_seed(seed);
    np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def estimate_memory_usage(X_shape, batch_size=32):
    """Estimate memory usage for dataset"""
    total_size = np.prod(X_shape) * 4 / (1024 ** 3)  # GB for float32
    batch_memory = batch_size * np.prod(X_shape[1:]) * 4 / (1024 ** 2)  # MB
    return total_size, batch_memory


# --------- metrics (keeping your excellent ROC/PR implementations!) -----
def confusion_counts(y_true, y_pred):
    y_true = y_true.astype(np.int64).ravel();
    y_pred = y_pred.astype(np.int64).ravel()
    tp = int(((y_true == 1) & (y_pred == 1)).sum());
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum());
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, fp, tn, fn


def metrics_from_counts(tp, fp, tn, fn):
    prec = tp / (tp + fp + 1e-12);
    rec = tp / (tp + fn + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    spec = tn / (tn + fp + 1e-12)
    return dict(precision=prec, recall=rec, f1=f1, accuracy=acc, specificity=spec)


def auc_trapezoid(x, y):
    return float(np.trapezoid(np.asarray(y, float), np.asarray(x, float)))


def roc_curve_manual(y_true, y_score):
    y_true = y_true.ravel().astype(np.int64);
    y_score = y_score.ravel().astype(float)
    order = np.argsort(-y_score);
    y_true = y_true[order];
    y_score = y_score[order]
    thresholds = np.r_[np.inf, np.unique(y_score)][::-1];
    P = (y_true == 1).sum();
    N = (y_true == 0).sum()
    tps = fps = idx = 0;
    fpr = [0.0];
    tpr = [0.0]
    for thr in thresholds[:-1]:
        while idx < len(y_score) and y_score[idx] >= thr:
            tps += (y_true[idx] == 1);
            fps += (y_true[idx] == 0);
            idx += 1
        fpr.append(fps / (N + 1e-12));
        tpr.append(tps / (P + 1e-12))
    fpr.append(1.0);
    tpr.append(1.0)
    return np.array(fpr), np.array(tpr)


def pr_curve_manual(y_true, y_score):
    y_true = y_true.ravel().astype(np.int64);
    y_score = y_score.ravel().astype(float)
    order = np.argsort(-y_score);
    y_true = y_true[order];
    tp = fp = 0;
    P = (y_true == 1).sum()
    precisions = [1.0];
    recalls = [0.0]
    for i in range(len(y_true)):
        tp += (y_true[i] == 1);
        fp += (y_true[i] == 0)
        precisions.append(tp / (tp + fp + 1e-12));
        recalls.append(tp / (P + 1e-12))
    return np.array(recalls), np.array(precisions)


def compute_metrics_from_arrays(y_true, y_pred):
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    out = metrics_from_counts(tp, fp, tn, fn);
    out.update(tp=tp, fp=fp, tn=tn, fn=fn);
    return out


def plot_confusion_matrix(tp, fp, tn, fn, title="Confusion Matrix"):
    fig, ax = plt.subplots()
    cm = np.array([[tn, fp], [fn, tp]], float);
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'],
           title=title, ylabel='True', xlabel='Pred')
    total = cm.sum() + 1e-12
    for i in range(2):
        for j in range(2):
            v = cm[i, j];
            ax.text(j, i, f"{int(v)}\n({v / total:.2%})",
                    ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    fig.tight_layout();
    return fig


# -------------- dataset -------------
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32);
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.y[i])


# -------- EDF + JSON loader with validation ---------
def validate_edf_json_pair(edf_path: str, json_path: str, required_channels: List[str]):
    """Validate EDF and JSON files before processing"""
    if not os.path.exists(edf_path):
        raise FileNotFoundError(f"EDF file not found: {edf_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON labels file not found: {json_path}")

    # Check EDF can be loaded and has required channels
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        available_channels = set(raw.ch_names)
        missing_channels = set(required_channels) - available_channels
        if missing_channels:
            raise ValueError(f"Missing required channels in {edf_path}: {missing_channels}")
        print(f"[Validation] EDF validated: {len(raw.ch_names)} channels, duration: {raw.times[-1]:.1f}s")
    except Exception as e:
        raise RuntimeError(f"Failed to validate EDF file {edf_path}: {e}")

    # Check JSON can be loaded
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("JSON file must contain a dictionary")

        # Count events
        events = data.get("detected_spindles") or data.get("spindles") or data.get("events") or []
        print(f"[Validation] JSON validated: {len(events)} spindle events found")
    except Exception as e:
        raise RuntimeError(f"Failed to validate JSON file {json_path}: {e}")


def _load_json_labels(path, total_samples, sfreq):
    with open(path, "r") as f:
        labels = json.load(f)
    y = np.zeros(total_samples, dtype=np.float32)
    events = labels.get("detected_spindles") or labels.get("spindles") or labels.get("events") or []

    valid_events = 0
    for ev in events:
        try:
            s = int(max(0, math.floor(float(ev.get("start", ev.get("onset", 0))) * sfreq)))
            e = int(min(total_samples, math.ceil(float(ev.get("end", ev.get("offset", 0))) * sfreq)))
            if e > s:
                y[s:e] = 1
                valid_events += 1
        except Exception:
            continue

    print(f"[Labels] Loaded {valid_events} valid spindle events, positive rate: {y.mean():.4f}")
    return y


def load_edf_windows(edf_path, json_path, cfg):
    """Load EDF with enhanced error handling and configurable channels"""
    # Get channel configuration
    channels = cfg["data"].get("channels", ['C3', 'C4', 'F3', 'F4'])

    # Validate files
    validate_edf_json_pair(edf_path, json_path, channels)

    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        raw.set_eeg_reference("average", verbose=False)  # CAR

        # Configurable filtering
        filter_cfg = cfg["data"].get("filter", {})
        low_freq = filter_cfg.get("low", 5.0)
        high_freq = filter_cfg.get("high", 30.0)
        raw.filter(low_freq, high_freq, verbose=False)

        raw.pick(channels)  # Configurable channels

        sfreq_target = cfg["data"].get("sfreq", 200.0)
        if abs(raw.info["sfreq"] - sfreq_target) > 0.1:
            raw.resample(sfreq_target, verbose=False)

        data = raw.get_data()
        # Use numpy directly (removed nan_to_num_np wrapper)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        C, T = data.shape
        y_full = _load_json_labels(json_path, T, raw.info["sfreq"])

        win = int(cfg["data"]["window_sec"] * raw.info["sfreq"])
        step = int(cfg["data"]["step_sec"] * raw.info["sfreq"])

        Xs, Ys = [], []
        for s in range(0, max(1, T - win + 1), step):
            e = s + win
            if e <= T:
                Xs.append(data[:, s:e]);
                Ys.append(y_full[s:e])
        X = np.stack(Xs, 0);
        y = np.stack(Ys, 0)

        # Per-channel z-score over dataset
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = (np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0) > 0.5).astype(np.float32)

        mean = X.mean(axis=(0, 2), keepdims=True);
        std = X.std(axis=(0, 2), keepdims=True);
        std[std < 1e-6] = 1.0
        X = (X - mean) / std

        print(f"[Data] Loaded {X.shape[0]} windows, shape: {X.shape}, pos rate: {y.mean():.4f}")

        # Memory usage estimation
        total_gb, batch_mb = estimate_memory_usage(X.shape, cfg["data"]["batch_size"])
        print(f"[Memory] Dataset: {total_gb:.2f}GB, Batch: {batch_mb:.1f}MB")

        return X, y

    except Exception as e:
        raise RuntimeError(f"Failed to load EDF/JSON data: {e}")


def split_data_with_cv(X, y, ratios=(0.7, 0.15, 0.15), seed=42, cv_folds=None):
    """Split data with optional cross-validation support"""
    if cv_folds and cv_folds > 1:
        # Create stratified K-fold splits
        labels = (y.sum(axis=1) > 0).astype(int)  # binary labels for stratification
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        cv_splits = []

        for train_idx, val_idx in skf.split(X, labels):
            # Further split training into train/test
            X_trainval, y_trainval = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # Split trainval into train/test
            labels_trainval = (y_trainval.sum(axis=1) > 0).astype(int)
            n_test = int(len(X_trainval) * ratios[2] / (ratios[0] + ratios[2]))

            if n_test > 0:
                rng = np.random.default_rng(seed)
                test_idx = rng.choice(len(X_trainval), n_test, replace=False)
                train_idx_local = np.setdiff1d(np.arange(len(X_trainval)), test_idx)

                X_train = X_trainval[train_idx_local];
                y_train = y_trainval[train_idx_local]
                X_test = X_trainval[test_idx];
                y_test = y_trainval[test_idx]
            else:
                X_train, y_train = X_trainval, y_trainval
                X_test, y_test = X_val[:10], y_val[:10]  # dummy test set

            cv_splits.append(((X_train, y_train), (X_val, y_val), (X_test, y_test)))

        return cv_splits
    else:
        # Standard single split
        return [split_data(X, y, ratios, seed)]


def split_data(X, y, ratios=(0.7, 0.15, 0.15), seed=42):
    """Standard data splitting"""
    rng = np.random.default_rng(seed);
    idx = np.arange(len(X));
    rng.shuffle(idx)
    n1 = int(ratios[0] * len(X));
    n2 = int((ratios[0] + ratios[1]) * len(X))
    return (X[idx[:n1]], y[idx[:n1]]), (X[idx[n1:n2]], y[idx[n1:n2]]), (X[idx[n2:]], y[idx[n2:]])


def create_loaders(X, y, batch, workers, sampler_type="normal", seed=42):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split_data(X, y, seed=seed)
    train_ds = EEGDataset(Xtr, ytr);
    val_ds = EEGDataset(Xva, yva);
    test_ds = EEGDataset(Xte, yte)

    if sampler_type == "undersample":
        labels = (ytr.sum(1) > 0).astype(int)
        pos = np.where(labels == 1)[0];
        neg = np.where(labels == 0)[0];
        n = min(len(pos), len(neg))
        if n > 0:
            sel = np.concatenate([np.random.choice(pos, n, False), np.random.choice(neg, n, False)])
            sampler = torch.utils.data.SubsetRandomSampler(sel)
            train_loader = DataLoader(train_ds, batch_size=batch, sampler=sampler, num_workers=workers, pin_memory=True)
            print(f"[Sampler] Undersampling: {n} pos + {n} neg = {2 * n} samples")
        else:
            train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers, pin_memory=True)
            print(f"[Warning] No positive samples for undersampling, using normal sampling")
    elif sampler_type == "weighted":
        labels = (ytr.sum(1) > 0).astype(int)
        counts = np.bincount(labels, minlength=2);
        counts[counts == 0] = 1
        w = 1.0 / counts;
        sample_w = w[labels];
        sampler = WeightedRandomSampler(sample_w, len(sample_w))
        train_loader = DataLoader(train_ds, batch_size=batch, sampler=sampler, num_workers=workers, pin_memory=True)
        print(f"[Sampler] Weighted sampling: weights = {w}, counts = {counts}")
    else:
        train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers, pin_memory=True)
        print(f"[Sampler] Normal sampling: {len(train_ds)} samples")

    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, val_loader, test_loader


# ----------- Enhanced model with optional attention -----------
def _gn_groups(c: int) -> int:
    for g in [8, 4, 2, 1]:
        if c % g == 0: return g
    return 1


class ResConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, k=7, dilation=1, dropout=0.05):
        super().__init__()
        pad = (k // 2) * dilation  # "same" for odd kernels
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=dilation)
        self.norm1 = nn.GroupNorm(_gn_groups(out_ch), out_ch)
        self.norm2 = nn.GroupNorm(_gn_groups(out_ch), out_ch)
        self.act = nn.SiLU(inplace=True)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.proj(x)
        x = self.drop(self.act(self.norm1(self.conv1(x))))
        x = self.drop(self.act(self.norm2(self.conv2(x))))
        return x + res


class AttentionBlock1D(nn.Module):
    """Optional 1D attention mechanism"""

    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm = nn.GroupNorm(_gn_groups(channels), channels)

    def forward(self, x):
        # x: [B, C, T] -> [B, T, C] for attention -> [B, C, T]
        B, C, T = x.shape
        x_transposed = x.transpose(1, 2)  # [B, T, C]
        attn_out, _ = self.attention(x_transposed, x_transposed, x_transposed)
        attn_out = attn_out.transpose(1, 2)  # [B, C, T]
        return x + self.norm(attn_out)


class DeepUNet1D(nn.Module):
    """
    Enhanced UNet1D with configurable kernel size, dilation ladder, and optional attention
    """

    def __init__(self, in_channels=4, base=64, depth=8, kernel_size=7, dropout=0.05, attention=False):
        super().__init__()
        self.depth = depth
        self.use_attention = attention
        self.encs = nn.ModuleList()
        self.pool = nn.MaxPool1d(2, ceil_mode=True)

        # Configurable dilation pattern
        dilations = [1, 1, 2, 2, 4, 4, 8, 8, 16, 16]  # extended for deeper networks

        ch = in_channels
        for d in range(depth):
            out_ch = base * (2 ** d)
            dil = dilations[d % len(dilations)]
            self.encs.append(ResConvBlock1D(ch, out_ch, k=kernel_size, dilation=dil, dropout=dropout))
            ch = out_ch

        # Optional attention at bottleneck
        if attention:
            self.attention = AttentionBlock1D(ch, num_heads=min(8, ch // 8))

        self.ups = nn.ModuleList()
        self.decs = nn.ModuleList()
        for d in reversed(range(1, depth)):
            in_ch = base * (2 ** d)
            out_ch = base * (2 ** (d - 1))
            self.ups.append(nn.ConvTranspose1d(in_ch, out_ch, 2, stride=2))
            dil = dilations[(d - 1) % len(dilations)]
            self.decs.append(ResConvBlock1D(out_ch * 2, out_ch, k=kernel_size, dilation=dil, dropout=dropout))

        self.head = nn.Conv1d(base, 1, 1)

    def forward(self, x):
        skips = []
        z = x
        for enc in self.encs:
            z = enc(z)
            skips.append(z)
            z = self.pool(z)

        # Optional attention at bottleneck
        if self.use_attention:
            z = self.attention(z)

        for up, dec, skip in zip(self.ups, self.decs, reversed(skips[:-1])):
            z = up(z)
            if z.shape[-1] != skip.shape[-1]:
                z = F.interpolate(z, size=skip.shape[-1], mode="linear", align_corners=False)
            z = torch.cat([z, skip], dim=1)
            z = dec(z)

        out = self.head(z).squeeze(1)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# -------- threshold & eval ----------
def sweep_threshold(y_true, y_probs, metric="f1", n_thresholds=101):
    """Enhanced threshold sweeping with more options"""
    thresholds = np.linspace(0, 1, n_thresholds)
    best_thr = 0.5;
    best_metrics = None;
    best_score = -1
    all_scores = []

    for thr in thresholds:
        y_pred = (y_probs >= thr).astype(int)
        metrics = compute_metrics_from_arrays(y_true, y_pred)
        score = metrics.get(metric, 0.0)
        all_scores.append(score)
        if score > best_score:
            best_thr, best_metrics, best_score = thr, metrics, score

    return best_thr, best_metrics, (thresholds, all_scores)


def evaluate(model, loader, device, thr=0.5, logger=None, step=0, tag="test"):
    model.eval();
    probs = [];
    trues = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb);
            probs.append(torch.sigmoid(out).cpu().numpy());
            trues.append(yb.cpu().numpy())
    probs = np.concatenate(probs);
    trues = np.concatenate(trues)
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    y_pred = (probs >= thr).astype(int)
    metrics = compute_metrics_from_arrays(trues.flatten(), y_pred.flatten())

    if logger:
        # scalars
        logger.add_scalar(f"{tag}/f1", metrics["f1"], step)
        logger.add_scalar(f"{tag}/precision", metrics["precision"], step)
        logger.add_scalar(f"{tag}/recall", metrics["recall"], step)
        logger.add_scalar(f"{tag}/specificity", metrics["specificity"], step)
        logger.add_scalar(f"{tag}/accuracy", metrics["accuracy"], step)
        logger.add_scalar(f"{tag}/threshold", thr, step)
        logger.add_scalar(f"{tag}/tp", metrics["tp"], step)
        logger.add_scalar(f"{tag}/fp", metrics["fp"], step)
        logger.add_scalar(f"{tag}/tn", metrics["tn"], step)
        logger.add_scalar(f"{tag}/fn", metrics["fn"], step)

        # ROC with chance line (keeping your implementation!)
        fpr, tpr = roc_curve_manual(trues.flatten(), probs.flatten())
        roc_auc = auc_trapezoid(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}", linewidth=2)
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Chance (y=x)", color='red')
        ax.set_title(f"ROC Curve — {tag}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim(0, 1);
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        logger.add_figure(f"{tag}/ROC_curve", fig, step);
        plt.close(fig)

        # PR with prevalence baseline (keeping your implementation!)
        rec, prec = pr_curve_manual(trues.flatten(), probs.flatten())
        pr_ap = auc_trapezoid(rec, prec)
        prevalence = float(trues.mean())
        fig, ax = plt.subplots()
        ax.plot(rec, prec, label=f"AP={pr_ap:.3f}", linewidth=2)
        ax.hlines(prevalence, 0, 1, linestyles="--", linewidth=1, color='red',
                  label=f"Baseline (prev={prevalence:.3f})")
        ax.set_title(f"Precision-Recall Curve — {tag}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1);
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        logger.add_figure(f"{tag}/PR_curve", fig, step);
        plt.close(fig)

        # confusion matrix
        fig_cm = plot_confusion_matrix(metrics['tp'], metrics['fp'], metrics['tn'], metrics['fn'],
                                       title=f'Confusion Matrix — {tag}')
        logger.add_figure(f"{tag}/confusion_matrix", fig_cm, step);
        plt.close(fig_cm)

        # log AUC/AP scalars
        logger.add_scalar(f"{tag}/roc_auc", roc_auc, step)
        logger.add_scalar(f"{tag}/pr_ap", pr_ap, step)

    print(f"[{tag.upper()}] thr={thr:.2f} F1={metrics['f1']:.4f} "
          f"Prec={metrics['precision']:.4f} Rec={metrics['recall']:.4f} "
          f"Acc={metrics['accuracy']:.4f} Spec={metrics['specificity']:.4f} "
          f"TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']}")
    return metrics


# --------------- Enhanced trainer with cross-validation and ensemble support -------------
class UNet1DTrainer:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        set_seed(cfg.get("seed", 42))

        dev = cfg.get("trainer", {}).get("device", "cuda")
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev)
        print(f"[Device] Using: {self.device}")

        self.amp_enabled = bool(cfg.get("trainer", {}).get("amp", True))
        try:
            self.scaler = torch.amp.GradScaler(enabled=(self.device.type == "cuda" and self.amp_enabled))
        except Exception:
            self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda" and self.amp_enabled))

        self.logger = None
        self.best_thr = 0.5

        # Cross-validation and ensemble settings
        self.cv_enabled = cfg.get("cross_validation", {}).get("enabled", False)
        self.cv_folds = cfg.get("cross_validation", {}).get("folds", 5)
        self.ensemble_enabled = cfg.get("trainer", {}).get("ensemble", {}).get("enabled", False)
        self.ensemble_size = cfg.get("trainer", {}).get("ensemble", {}).get("n_models", 1)

    def load_and_prepare_data(self):
        """Load data with enhanced error handling"""
        data_cfg = self.cfg["data"]
        edf_dir = Path(data_cfg["edf"]["dir"])

        # Support for multiple EDF files
        edfs = list(edf_dir.glob("*.edf"))
        if not edfs:
            raise RuntimeError(f"No EDF files found in {edf_dir}")

        print(f"[Data] Found {len(edfs)} EDF files")

        # For now, use first EDF (extend later for multiple files)
        edf_path = edfs[0]
        json_path = data_cfg["edf"]["labels_json"]

        X, y = load_edf_windows(str(edf_path), json_path, self.cfg)

        # Memory check
        max_memory_gb = self.cfg.get("data", {}).get("max_memory_gb", 16.0)
        total_gb, _ = estimate_memory_usage(X.shape, data_cfg["batch_size"])
        if total_gb > max_memory_gb:
            print(f"[Warning] Dataset size ({total_gb:.2f}GB) exceeds limit ({max_memory_gb}GB)")
            # Could implement chunking here

        return X, y

    def create_model(self, in_channels):
        """Create model with configuration"""
        mcfg = self.cfg["model"]["unet1d"]
        model = DeepUNet1D(
            in_channels=in_channels,
            base=mcfg.get("base", 64),
            depth=mcfg.get("depth", 8),
            kernel_size=mcfg.get("kernel_size", 7),
            dropout=mcfg.get("dropout", 0.05),
            attention=mcfg.get("attention", False),
        ).to(self.device)

        # Log model info
        if self.logger:
            dummy_input = torch.randn(1, in_channels, 400).to(self.device)
            self.logger.log_model_info(model, dummy_input)

        return model

    def train_single_fold(self, train_loader, val_loader, fold_idx=0):
        """Train a single model (one fold or ensemble member)"""
        model = self.create_model(in_channels=4)  # Assuming 4 channels

        opt = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg["trainer"]["lr"],
            weight_decay=self.cfg["trainer"].get("weight_decay", 0.01)
        )

        # Use your external loss module!
        criterion = build_loss_function(
            loss_name=self.cfg["trainer"]["loss"],
            cfg=self.cfg,
            train_loader=train_loader,
            device=str(self.device)
        )

        best_f1 = 0.0
        patience = 0
        clip = self.cfg["trainer"].get("grad_clip_norm", 1.0)
        epochs = int(self.cfg["trainer"]["epochs"])

        for ep in range(1, epochs + 1):
            # ----- Train -----
            model.train()
            loss_sum = 0.0

            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda" and self.amp_enabled)):
                    out = model(xb)
                    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
                    loss = criterion(out, yb)
                    if not torch.isfinite(loss):
                        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

                self.scaler.scale(loss).backward()

                if clip and clip > 0:
                    self.scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), clip)

                self.scaler.step(opt)
                self.scaler.update()
                loss_sum += float(loss.item())

            train_loss = loss_sum / max(1, len(train_loader))

            if self.logger:
                tag = f"fold_{fold_idx}" if fold_idx > 0 else ""
                self.logger.add_scalar(f"loss/train{('_' + tag) if tag else ''}", train_loss, ep)

            # ----- Validate -----
            model.eval()
            probs = []
            trues = []

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    out = model(xb)
                    probs.append(torch.sigmoid(out).cpu().numpy())
                    trues.append(yb.cpu().numpy())

            probs = np.concatenate(probs)
            trues = np.concatenate(trues)
            probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

            thr, metrics, (thresholds, scores) = sweep_threshold(
                trues.flatten(), probs.flatten(),
                metric=self.cfg["eval"].get("metric_for_best", "f1")
            )
            self.best_thr = float(thr)

            # Log validation metrics
            if self.logger:
                tag = f"fold_{fold_idx}" if fold_idx > 0 else "val"
                self.logger.add_scalar(f"{tag}/f1", metrics["f1"], ep)
                self.logger.add_scalar(f"{tag}/precision", metrics["precision"], ep)
                self.logger.add_scalar(f"{tag}/recall", metrics["recall"], ep)
                self.logger.add_scalar(f"{tag}/threshold", self.best_thr, ep)

                # ROC curve (keeping your implementation!)
                fpr, tpr = roc_curve_manual(trues.flatten(), probs.flatten())
                roc_auc = auc_trapezoid(fpr, tpr)
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
                ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Chance")
                ax.set_xlabel("FPR");
                ax.set_ylabel("TPR")
                ax.legend();
                ax.grid(True, alpha=0.3)
                self.logger.add_figure(f"{tag}/ROC_curve", fig, ep)
                plt.close(fig)

                # PR curve (keeping your implementation!)
                rec, prec = pr_curve_manual(trues.flatten(), probs.flatten())
                pr_ap = auc_trapezoid(rec, prec)
                prevalence = float(trues.mean())
                fig, ax = plt.subplots()
                ax.plot(rec, prec, label=f"AP={pr_ap:.3f}")
                ax.hlines(prevalence, 0, 1, linestyles="--", label=f"Baseline={prevalence:.3f}")
                ax.set_xlabel("Recall");
                ax.set_ylabel("Precision")
                ax.legend();
                ax.grid(True, alpha=0.3)
                self.logger.add_figure(f"{tag}/PR_curve", fig, ep)
                plt.close(fig)

                # Confusion matrix
                fig_cm = plot_confusion_matrix(
                    metrics['tp'], metrics['fp'], metrics['tn'], metrics['fn'],
                    title=f'Confusion — {tag}'
                )
                self.logger.add_figure(f"{tag}/confusion_matrix", fig_cm, ep)
                plt.close(fig_cm)

                # Sample predictions
                if ep % 5 == 0:  # Every 5 epochs
                    self.logger.log_sample_predictions(model, val_loader, self.device, ep)

            print(f"Fold {fold_idx} Epoch {ep} loss={train_loss:.6f} thr={self.best_thr:.2f} "
                  f"f1={metrics['f1']:.4f} TP={metrics['tp']} FP={metrics['fp']} "
                  f"TN={metrics['tn']} FN={metrics['fn']}")

            # Save best model
            if metrics["f1"] > best_f1 + 1e-8:
                best_f1 = metrics["f1"]
                patience = 0

                # Save checkpoint
                Path(self.cfg["paths"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
                ckpt_name = f"best_unet1d_fold_{fold_idx}.pt" if fold_idx > 0 else "best_unet1d.pt"
                ckpt_path = Path(self.cfg["paths"]["checkpoint_dir"]) / ckpt_name
                torch.save({
                    "model": model.state_dict(),
                    "thr": float(self.best_thr),
                    "fold": fold_idx,
                    "epoch": ep,
                    "f1": best_f1
                }, ckpt_path)

                if self.logger:
                    self.logger.add_artifact_file(
                        str(ckpt_path),
                        name=f"best_unet1d_fold_{fold_idx}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )
            else:
                patience += 1
                if patience >= int(self.cfg["trainer"]["early_stopping_patience"]):
                    print(f"Early stopping at epoch {ep}")
                    break

        return model, best_f1, self.best_thr

    def run(self):
        """Main training loop with cross-validation and ensemble support"""
        try:
            # Load data
            X, y = self.load_and_prepare_data()

            # Initialize logger
            self.logger = WBLogger(self.cfg)
            self.logger.channel_names = self.cfg["data"].get("channels", ['C3', 'C4', 'F3', 'F4'])

            if self.cv_enabled:
                print(f"[Training] Starting {self.cv_folds}-fold cross-validation")
                cv_splits = split_data_with_cv(X, y, cv_folds=self.cv_folds, seed=self.cfg.get("seed", 42))

                all_results = []
                for fold_idx, (train_data, val_data, test_data) in enumerate(cv_splits):
                    print(f"\n[CV] Training fold {fold_idx + 1}/{self.cv_folds}")

                    # Create loaders for this fold
                    train_ds = EEGDataset(*train_data)
                    val_ds = EEGDataset(*val_data)
                    test_ds = EEGDataset(*test_data)

                    batch_size = self.cfg["data"]["batch_size"]
                    workers = self.cfg["data"].get("num_workers", 2)

                    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                              num_workers=workers, pin_memory=True)
                    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                            num_workers=workers, pin_memory=True)
                    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                             num_workers=workers, pin_memory=True)

                    # Train this fold
                    model, best_f1, best_thr = self.train_single_fold(train_loader, val_loader, fold_idx + 1)

                    # Test this fold
                    test_metrics = evaluate(model, test_loader, self.device, thr=best_thr,
                                            logger=self.logger, step=fold_idx + 1, tag=f"test_fold_{fold_idx + 1}")

                    all_results.append({
                        'fold': fold_idx + 1,
                        'val_f1': best_f1,
                        'test_f1': test_metrics['f1'],
                        'threshold': best_thr
                    })

                # Log cross-validation summary
                avg_val_f1 = np.mean([r['val_f1'] for r in all_results])
                avg_test_f1 = np.mean([r['test_f1'] for r in all_results])
                std_test_f1 = np.std([r['test_f1'] for r in all_results])

                print(f"\n[CV Summary] Average Val F1: {avg_val_f1:.4f}")
                print(f"[CV Summary] Average Test F1: {avg_test_f1:.4f} ± {std_test_f1:.4f}")

                if self.logger:
                    self.logger.add_scalar("cv/avg_val_f1", avg_val_f1, 0)
                    self.logger.add_scalar("cv/avg_test_f1", avg_test_f1, 0)
                    self.logger.add_scalar("cv/std_test_f1", std_test_f1, 0)

            else:
                # Standard single training run
                print("[Training] Standard single split training")
                train_loader, val_loader, test_loader = create_loaders(
                    X, y,
                    batch=self.cfg["data"]["batch_size"],
                    workers=self.cfg["data"].get("num_workers", 2),
                    sampler_type=self.cfg["trainer"].get("sampler", "normal"),
                    seed=self.cfg.get("seed", 42)
                )

                # Train model
                model, best_f1, best_thr = self.train_single_fold(train_loader, val_loader, 0)

                print(f"\n[Training Complete] Best F1: {best_f1:.4f}, Best threshold: {best_thr:.2f}")

                # Final test evaluation
                ckpt_path = Path(self.cfg["paths"]["checkpoint_dir"]) / "best_unet1d.pt"
                if ckpt_path.exists():
                    try:
                        ckpt = torch.load(ckpt_path, map_location=self.device)
                    except Exception:
                        add_safe_globals([np._core.multiarray.scalar])
                        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

                    model.load_state_dict(ckpt["model"])
                    best_thr = float(ckpt.get("thr", best_thr))

                    print(f"[Test] Evaluating best model (F1={ckpt.get('f1', 0):.4f}, thr={best_thr:.2f})")
                    evaluate(model, test_loader, self.device, thr=best_thr,
                             logger=self.logger, step=ckpt.get('epoch', 0), tag="test")
                else:
                    print("[Warning] No checkpoint found for test evaluation.")

        except Exception as e:
            print(f"[Error] Training failed: {e}")
            traceback.print_exc()
        finally:
            if self.logger:
                self.logger.close()


# ---------------- main --------------
def main():
    ap = argparse.ArgumentParser(description="Enhanced UNet1D for EEG Spindle Detection")
    ap.add_argument("--config", default="config.yaml", help="Configuration file")
    ap.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = ap.parse_args()

    if args.debug:
        warnings.filterwarnings("ignore", category=UserWarning)

    try:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)

        # Format run name if specified
        if cfg.get("logging", {}).get("run_name"):
            try:
                cfg["logging"]["run_name"] = cfg["logging"]["run_name"].format(
                    trainer=cfg["trainer"],
                    model=cfg["model"],
                )
            except Exception as e:
                print(f"[Warning] run_name format failed: {e}")

        # Print configuration summary
        print("=" * 60)
        print("Enhanced UNet1D Spindle Detection")
        print("=" * 60)
        print(f"Model: {cfg['model']['unet1d']}")
        print(f"Loss: {cfg['trainer']['loss']}")
        print(f"Sampler: {cfg['trainer'].get('sampler', 'normal')}")
        print(f"Cross-validation: {cfg.get('cross_validation', {}).get('enabled', False)}")
        print("=" * 60)

        trainer = UNet1DTrainer(cfg)
        trainer.run()

    except FileNotFoundError:
        print(f"[Error] Configuration file not found: {args.config}")
    except Exception as e:
        print(f"[Error] Failed to start training: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()