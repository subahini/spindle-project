#!/usr/bin/env python3
"""
U-Net Model Evaluation with:
- Single subject: 70-15-15 time-based split
- GroupKFold: Subject-wise cross-validation
- Full W&B logging with ROC/PR curves and confusion matrices
"""
import os
import sys
import json
import yaml
import math
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from tabulate import tabulate
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score,
                             average_precision_score, roc_curve,
                             precision_recall_curve, auc)
import matplotlib.pyplot as plt
import wandb
import mne

# Import your model and loss
from unet1d import UNet1D
from losses import build_loss_function


# ----------------------------
# Helper Functions
# ----------------------------
def subject_id_from_path(edf_path: str) -> str:
    """Extract subject ID from EDF filename (P002_1_raw -> P002)"""
    return Path(edf_path).stem.split("_")[0]


def get_all_subject_pairs(cfg: Dict) -> Dict[str, List[Tuple[Path, Path]]]:
    """
    Get all EDF/JSON pairs grouped by subject
    """
    data_root = Path(cfg['data']['edf']['dir']).parent
    edf_dir = Path(cfg['data']['edf']['dir'])
    label_dir = Path(cfg['data']['edf']['labels_json']).parent

    # Find all EDF files
    all_edf_files = sorted(edf_dir.glob("*.edf"))

    # Group by subject
    subject_pairs = {}

    for edf_path in all_edf_files:
        subject = subject_id_from_path(str(edf_path))

        # Find matching JSON (assuming same naming convention)
        base = edf_path.stem.replace("_raw", "").replace("_filtered", "")
        json_path = label_dir / f"{base}.json"

        if json_path.exists():
            if subject not in subject_pairs:
                subject_pairs[subject] = []
            subject_pairs[subject].append((edf_path, json_path))

    print(f"\nFound {len(subject_pairs)} subjects:")
    for subject, pairs in subject_pairs.items():
        print(f"  {subject}: {len(pairs)} files")

    return subject_pairs


def load_subject_windows(edf_paths: List[Path], json_paths: List[Path], cfg: Dict):
    """
    Load and concatenate all windows for a subject from multiple files
    """
    all_X = []
    all_y = []
    total_duration = 0

    for edf_path, json_path in zip(edf_paths, json_paths):
        # Load windows from this file
        X, y = load_edf_windows(str(edf_path), str(json_path), cfg)
        all_X.append(X)
        all_y.append(y)

        # Calculate duration (each window is window_sec seconds)
        duration = len(X) * cfg['data']['window_sec']
        total_duration += duration

    # Concatenate all windows
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    print(f"    Total windows: {len(X)} ({total_duration / 3600:.2f} hours)")
    print(f"    X shape: {X.shape}, y shape: {y.shape}")

    return X, y


def load_edf_windows(edf_path: str, json_path: str, cfg: Dict):
    """Load EDF file and create windows (copied from unet1d.py)"""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.set_eeg_reference("average", verbose=False)
    raw.filter(cfg["data"]["filter"]["low"], cfg["data"]["filter"]["high"], verbose=False)
    raw.pick(cfg["data"]["channels"])

    sf_target = cfg["data"]["sfreq"]
    if abs(raw.info["sfreq"] - sf_target) > 0.1:
        raw.resample(sfreq=sf_target, verbose=False)

    data = raw.get_data()  # [C,T]
    C, T = data.shape

    # Load labels
    with open(json_path, "r") as f:
        labels = json.load(f)

    y_full = np.zeros(T, dtype=np.float32)
    events = labels.get("detected_spindles") or labels.get("spindles") or []
    for ev in events:
        s = int(max(0, math.floor(float(ev["start"]) * raw.info["sfreq"])))
        e = int(min(T, math.ceil(float(ev["end"]) * raw.info["sfreq"])))
        if e > s:
            y_full[s:e] = 1

    # Create windows
    win = int(cfg["data"]["window_sec"] * raw.info["sfreq"])
    step = int(cfg["data"]["step_sec"] * raw.info["sfreq"])

    Xs, Ys = [], []
    for s in range(0, T - win + 1, step):
        e = s + win
        Xs.append(data[:, s:e])
        Ys.append(y_full[s:e])

    X = np.stack(Xs, 0)  # [N, C, Tw]
    y = np.stack(Ys, 0)  # [N, Tw]

    return X, y


def split_time_70_15_15(X: np.ndarray, y: np.ndarray, gap: int = 5):
    """
    Split windows into 70% train, 15% validation, 15% test with gap
    """
    n = len(X)
    n1 = int(0.7 * n)
    n2 = int(0.85 * n)

    # Add gap between splits to avoid leakage
    tr_end = max(0, n1 - gap)
    va_start = min(n, n1 + gap)
    va_end = max(va_start, n2 - gap)
    te_start = min(n, n2 + gap)

    # Split data
    X_train = X[:tr_end]
    y_train = y[:tr_end]

    X_val = X[va_start:va_end]
    y_val = y[va_start:va_end]

    X_test = X[te_start:]
    y_test = y[te_start:]

    print(f"\n  Split sizes:")
    print(f"    Train: {len(X_train)} windows ({len(X_train) / n * 100:.1f}%)")
    print(f"    Val:   {len(X_val)} windows ({len(X_val) / n * 100:.1f}%)")
    print(f"    Test:  {len(X_test)} windows ({len(X_test) / n * 100:.1f}%)")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def create_groupkfold_splits(subject_pairs: Dict[str, List[Tuple[Path, Path]]],
                             n_folds: int,
                             fold_idx: int,
                             random_state: int = 42):
    """
    Create train/val/test splits using GroupKFold at subject level
    """
    subjects = list(subject_pairs.keys())

    if len(subjects) < n_folds:
        raise ValueError(f"Need >= {n_folds} subjects, got {len(subjects)}")

    # Create group labels
    groups = subjects
    indices = np.arange(len(subjects))

    # Create GroupKFold splits
    gkf = GroupKFold(n_splits=n_folds)
    splits = list(gkf.split(indices, groups=groups))
    train_idx, test_idx = splits[fold_idx - 1]

    # Get train and test subjects
    train_subjects = [subjects[i] for i in train_idx]
    test_subjects = [subjects[i] for i in test_idx]

    # Pick ONE validation subject from train subjects
    rng = np.random.RandomState(random_state + fold_idx)
    val_subject = rng.choice(train_subjects)

    # Remove validation subject from train
    train_subjects = [s for s in train_subjects if s != val_subject]

    print(f"\n  Fold {fold_idx}/{n_folds}:")
    print(f"    Train subjects ({len(train_subjects)}): {train_subjects[:3]}...")
    print(f"    Val subject: {val_subject}")
    print(f"    Test subjects ({len(test_subjects)}): {test_subjects[:3]}...")

    return train_subjects, val_subject, test_subjects


class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.y[i])


def create_dataloader(X, y, batch_size, num_workers, shuffle=True, sampler_type="normal"):
    """Create dataloader with optional sampling"""
    dataset = EEGDataset(X, y)

    if sampler_type == "undersample" and shuffle:
        labels = (y.sum(1) > 0).astype(int)
        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]
        n = min(len(pos_idx), len(neg_idx))
        if n > 0:
            sel_idx = np.concatenate([
                np.random.choice(pos_idx, n, False),
                np.random.choice(neg_idx, n, False)
            ])
            sampler = torch.utils.data.SubsetRandomSampler(sel_idx)
            return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers)

    elif sampler_type == "weighted" and shuffle:
        labels = (y.sum(1) > 0).astype(int)
        class_count = np.bincount(labels)
        weight = 1.0 / class_count
        sample_weights = weight[labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers)

    # Default: regular shuffling
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers)


def calculate_metrics(y_true, y_pred, y_score=None):
    """
    Calculate all metrics
    """
    y_true = y_true.astype(np.int32).ravel()
    y_pred = y_pred.astype(np.int32).ravel()

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Basic metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    metrics = {
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy),
    }

    # ROC/PR AUC if scores are available
    if y_score is not None and len(np.unique(y_true)) > 1:
        y_score = y_score.ravel()
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_score))
        metrics['pr_auc'] = float(average_precision_score(y_true, y_score))
    else:
        metrics['roc_auc'] = 0.0
        metrics['pr_auc'] = 0.0

    return metrics


def find_best_threshold(y_true, y_score, n_thresholds=101):
    """Find threshold that maximizes F1 score"""
    if len(np.unique(y_true)) < 2:
        return 0.5

    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    best_f1 = -1.0
    best_thr = 0.5

    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    return best_thr


def plot_curves_and_log(y_true, y_pred, y_score, prefix="", step=0):
    """
    Plot ROC and PR curves and log to W&B
    """
    y_true_flat = np.asarray(y_true).ravel()
    y_score_flat = np.asarray(y_score).ravel()

    if len(np.unique(y_true_flat)) < 2:
        return

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true_flat, y_score_flat)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{prefix} ROC Curve')
    plt.grid(alpha=0.3)
    plt.legend()
    wandb.log({f"{prefix}/roc_curve": wandb.Image(plt)}, step=step)
    plt.close()

    # PR Curve
    precision, recall, _ = precision_recall_curve(y_true_flat, y_score_flat)
    pr_auc = average_precision_score(y_true_flat, y_score_flat)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, 'b-', linewidth=2, label=f'PR (AUC = {pr_auc:.4f})')
    baseline = y_true_flat.mean()
    plt.axhline(y=baseline, color='r', linestyle='--', label=f'Baseline = {baseline:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{prefix} PR Curve')
    plt.grid(alpha=0.3)
    plt.legend()
    wandb.log({f"{prefix}/pr_curve": wandb.Image(plt)}, step=step)
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_true_flat, y_pred.ravel())

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f'{prefix} Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks([0, 1], ['No Spindle', 'Spindle'])
    plt.yticks([0, 1], ['No Spindle', 'Spindle'])

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    wandb.log({f"{prefix}/confusion_matrix": wandb.Image(plt)}, step=step)
    plt.close()


def train_epoch(model, loader, optimizer, criterion, device, scaler, cfg):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda") and cfg['trainer']['amp']):
            logits = model(xb)
            loss = criterion(logits, yb)

        scaler.scale(loss).backward()

        if cfg['trainer']['grad_clip_norm']:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['trainer']['grad_clip_norm'])

        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item()) * len(xb)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device, cfg):
    """Evaluate model and return metrics"""
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb)
            loss = criterion(logits, yb)

            probs = torch.sigmoid(logits).cpu().numpy()

            total_loss += float(loss.item()) * len(xb)
            all_probs.append(probs)
            all_targets.append(yb.cpu().numpy())

    # Concatenate all predictions and targets
    probs = np.concatenate(all_probs, axis=0).reshape(-1)
    targets = np.concatenate(all_targets, axis=0).reshape(-1)

    # Find best threshold on validation set
    best_thr = find_best_threshold(targets, probs)
    preds = (probs >= best_thr).astype(int)

    # Calculate metrics
    metrics = calculate_metrics(targets, preds, probs)
    metrics['loss'] = total_loss / len(loader.dataset)
    metrics['threshold'] = best_thr

    return metrics, probs, preds, targets


# ----------------------------
# Main Function
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--mode", choices=['single', 'groupkfold'], required=True,
                        help="'single' for one subject, 'groupkfold' for cross-validation")
    parser.add_argument("--subject", type=str, default=None,
                        help="Subject ID for single mode (e.g., P002)")
    parser.add_argument("--fold", type=int, default=1,
                        help="Fold number for GroupKFold (1-based)")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of folds for GroupKFold")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Custom run name for W&B")

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Device setup
    device = cfg["trainer"]["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"\nUsing device: {device}")

    # Create W&B run name
    if args.run_name:
        run_name = args.run_name
    else:
        if args.mode == 'single':
            run_name = f"unet_subject_{args.subject}"
        else:
            run_name = f"unet_groupkfold_fold{args.fold}_of_{args.n_folds}"

    # Initialize W&B
    wandb.init(
        project=cfg['logging']['wandb']['project'],
        entity=cfg['logging']['wandb']['entity'],
        name=run_name,
        config=cfg,
        reinit=True
    )

    print("\n" + "=" * 70)
    print(f" W&B Run: {run_name}")
    print(f" Mode: {args.mode.upper()}")
    print(f"Model: UNet1D with {cfg['model']['unet1d']['base_channels']} base channels")
    print("=" * 70)

    # Get all subject pairs
    subject_pairs = get_all_subject_pairs(cfg)

    # Store results per subject/fold
    all_results = {}

    if args.mode == 'single':
        # ----------------------------
        # SINGLE SUBJECT MODE
        # ----------------------------
        if args.subject not in subject_pairs:
            raise ValueError(f"Subject {args.subject} not found. Available: {list(subject_pairs.keys())}")

        subject_files = subject_pairs[args.subject]
        edf_paths = [p[0] for p in subject_files]
        json_paths = [p[1] for p in subject_files]

        print(f"\nProcessing subject: {args.subject} ({len(subject_files)} files)")

        # Load all windows for this subject
        X, y = load_subject_windows(edf_paths, json_paths, cfg)

        # Split into train/val/test (70-15-15)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_time_70_15_15(X, y)

        # Create dataloaders
        train_loader = create_dataloader(X_train, y_train,
                                         batch_size=cfg['data']['batch_size'],
                                         num_workers=cfg['data']['num_workers'],
                                         sampler_type=cfg['trainer'].get('sampler', 'normal'))
        val_loader = create_dataloader(X_val, y_val,
                                       batch_size=cfg['data']['batch_size'],
                                       num_workers=cfg['data']['num_workers'],
                                       shuffle=False)
        test_loader = create_dataloader(X_test, y_test,
                                        batch_size=cfg['data']['batch_size'],
                                        num_workers=cfg['data']['num_workers'],
                                        shuffle=False)

        # Initialize model
        mcfg = cfg['model']['unet1d']
        model = UNet1D(
            in_channels=len(cfg['data']['channels']),
            base_channels=int(mcfg['base_channels']),
            depth=int(mcfg['depth']),
            kernel_size=int(mcfg['kernel_size']),
            dropout=float(mcfg['dropout']),
            use_attention=bool(mcfg['use_attention']),
            final_activation=str(mcfg['final_activation'])
        ).to(device)

        # Optimizer and loss
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg['trainer']['lr']),
            weight_decay=float(cfg['trainer']['weight_decay'])
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg['trainer']['epochs']
        )
        criterion = build_loss_function(cfg['trainer']['loss'], cfg['trainer'], train_loader)
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda") and cfg['trainer']['amp'])

        # Training loop
        best_val_f1 = -1.0
        best_epoch = 0

        for epoch in range(1, cfg['trainer']['epochs'] + 1):
            # Train
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler, cfg)

            # Validate
            val_metrics, val_probs, val_preds, val_targets = evaluate(model, val_loader, criterion, device, cfg)

            # Log to W&B
            wandb.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'val/loss': val_metrics['loss'],
                'val/f1': val_metrics['f1'],
                'val/precision': val_metrics['precision'],
                'val/recall': val_metrics['recall'],
                'val/roc_auc': val_metrics['roc_auc'],
                'val/pr_auc': val_metrics['pr_auc'],
                'val/threshold': val_metrics['threshold'],
                'lr': scheduler.get_last_lr()[0]
            }, step=epoch)

            # Plot curves every 10 epochs
            if epoch % 10 == 0:
                plot_curves_and_log(val_targets, val_preds, val_probs, prefix="val", step=epoch)

            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f} | Val ROC-AUC: {val_metrics['roc_auc']:.4f}")

            # Save best model
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                best_epoch = epoch
                torch.save(model.state_dict(), f"best_model_subject_{args.subject}.pt")

            scheduler.step()

        # Final test evaluation
        print(f"\nBest model from epoch {best_epoch} (Val F1={best_val_f1:.4f})")
        model.load_state_dict(torch.load(f"best_model_subject_{args.subject}.pt"))
        test_metrics, test_probs, test_preds, test_targets = evaluate(model, test_loader, criterion, device, cfg)

        print(f"\n📊 Test Results:")
        print(f"  F1: {test_metrics['f1']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        print(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")
        print(f"  PR-AUC: {test_metrics['pr_auc']:.4f}")

        # Log test results
        wandb.log({
            'test/f1': test_metrics['f1'],
            'test/precision': test_metrics['precision'],
            'test/recall': test_metrics['recall'],
            'test/roc_auc': test_metrics['roc_auc'],
            'test/pr_auc': test_metrics['pr_auc'],
        })
        plot_curves_and_log(test_targets, test_preds, test_probs, prefix="test")

        all_results = {
            'subject': args.subject,
            'test_metrics': test_metrics
        }

    else:
        # ----------------------------
        # GROUPKFOLD MODE
        # ----------------------------
        # Create subject-wise splits
        train_subjects, val_subject, test_subjects = create_groupkfold_splits(
            subject_pairs, args.n_folds, args.fold
        )

        # Load training data
        train_X_list = []
        train_y_list = []
        for subject in train_subjects:
            files = subject_pairs[subject]
            X, y = load_subject_windows([f[0] for f in files], [f[1] for f in files], cfg)
            train_X_list.append(X)
            train_y_list.append(y)

        X_train = np.concatenate(train_X_list, axis=0)
        y_train = np.concatenate(train_y_list, axis=0)

        # Load validation data
        val_files = subject_pairs[val_subject]
        X_val, y_val = load_subject_windows([val_files[0][0]], [val_files[0][1]], cfg)

        # Load test data
        test_X_list = []
        test_y_list = []
        for subject in test_subjects:
            files = subject_pairs[subject]
            X, y = load_subject_windows([f[0] for f in files], [f[1] for f in files], cfg)
            test_X_list.append(X)
            test_y_list.append(y)

        X_test = np.concatenate(test_X_list, axis=0)
        y_test = np.concatenate(test_y_list, axis=0)

        print(f"\nData sizes:")
        print(f"  Train: {len(X_train)} windows")
        print(f"  Val:   {len(X_val)} windows")
        print(f"  Test:  {len(X_test)} windows")

        # Create dataloaders
        train_loader = create_dataloader(X_train, y_train,
                                         batch_size=cfg['data']['batch_size'],
                                         num_workers=cfg['data']['num_workers'],
                                         sampler_type=cfg['trainer'].get('sampler', 'normal'))
        val_loader = create_dataloader(X_val, y_val,
                                       batch_size=cfg['data']['batch_size'],
                                       num_workers=cfg['data']['num_workers'],
                                       shuffle=False)
        test_loader = create_dataloader(X_test, y_test,
                                        batch_size=cfg['data']['batch_size'],
                                        num_workers=cfg['data']['num_workers'],
                                        shuffle=False)

        # Initialize model (same as before)
        mcfg = cfg['model']['unet1d']
        model = UNet1D(
            in_channels=len(cfg['data']['channels']),
            base_channels=int(mcfg['base_channels']),
            depth=int(mcfg['depth']),
            kernel_size=int(mcfg['kernel_size']),
            dropout=float(mcfg['dropout']),
            use_attention=bool(mcfg['use_attention']),
            final_activation=str(mcfg['final_activation'])
        ).to(device)

        # Optimizer and loss
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg['trainer']['lr']),
            weight_decay=float(cfg['trainer']['weight_decay'])
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg['trainer']['epochs']
        )
        criterion = build_loss_function(cfg['trainer']['loss'], cfg['trainer'], train_loader)
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda") and cfg['trainer']['amp'])

        # Training loop (same as single mode)
        best_val_f1 = -1.0
        best_epoch = 0

        for epoch in range(1, cfg['trainer']['epochs'] + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler, cfg)
            val_metrics, val_probs, val_preds, val_targets = evaluate(model, val_loader, criterion, device, cfg)

            wandb.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'val/loss': val_metrics['loss'],
                'val/f1': val_metrics['f1'],
                'val/precision': val_metrics['precision'],
                'val/recall': val_metrics['recall'],
                'val/roc_auc': val_metrics['roc_auc'],
                'val/pr_auc': val_metrics['pr_auc'],
                'val/threshold': val_metrics['threshold'],
                'lr': scheduler.get_last_lr()[0]
            }, step=epoch)

            if epoch % 10 == 0:
                plot_curves_and_log(val_targets, val_preds, val_probs, prefix="val", step=epoch)

            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f} | Val ROC-AUC: {val_metrics['roc_auc']:.4f}")

            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                best_epoch = epoch
                torch.save(model.state_dict(), f"best_model_fold{args.fold}.pt")

            scheduler.step()

        # Final test evaluation
        print(f"\nBest model from epoch {best_epoch} (Val F1={best_val_f1:.4f})")
        model.load_state_dict(torch.load(f"best_model_fold{args.fold}.pt"))
        test_metrics, test_probs, test_preds, test_targets = evaluate(model, test_loader, criterion, device, cfg)

        print(f"\n📊 Test Results (Fold {args.fold}):")
        print(f"  F1: {test_metrics['f1']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        print(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")
        print(f"  PR-AUC: {test_metrics['pr_auc']:.4f}")

        # Log test results
        wandb.log({
            'test/f1': test_metrics['f1'],
            'test/precision': test_metrics['precision'],
            'test/recall': test_metrics['recall'],
            'test/roc_auc': test_metrics['roc_auc'],
            'test/pr_auc': test_metrics['pr_auc'],
        })
        plot_curves_and_log(test_targets, test_preds, test_probs, prefix="test")

        all_results = {
            'fold': args.fold,
            'n_folds': args.n_folds,
            'val_subject': val_subject,
            'test_metrics': test_metrics
        }

    # Save results
    out_dir = Path("unet_results")
    out_dir.mkdir(exist_ok=True)

    if args.mode == 'single':
        out_file = out_dir / f"subject_{args.subject}_results.json"
    else:
        out_file = out_dir / f"fold_{args.fold}_of_{args.n_folds}_results.json"

    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n💾 Results saved to {out_file}")
    print(f"\n✅ W&B Run Complete: {wandb.run.url}")

    wandb.finish()


if __name__ == "__main__":
    main()