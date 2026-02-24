#!/usr/bin/env python3
"""
U-Net Model GroupKFold Evaluation
Subject-wise cross-validation with same structure as baseline and CNN
"""
import os
import sys
import json
import yaml
import math
import argparse
import re
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
# Helper Functions (same as baseline)
# ----------------------------
def subject_id_from_path(edf_path: str) -> str:
    """Extract subject ID from EDF filename (P002_3_raw -> P002)"""
    return Path(edf_path).stem.split("_")[0]


def build_pairs_from_dirs(edf_dir: str, json_dir: str):
    """Build EDF/JSON pairs from directories (same as baseline)"""
    edf_dir = Path(edf_dir)
    json_dir = Path(json_dir)

    edfs = sorted(edf_dir.glob("*.edf"))
    jsons = sorted(json_dir.glob("*.json"))

    json_map = {}
    for jp in jsons:
        m = re.search(r"(P\d+_\d+)", jp.stem)
        if m:
            json_map[m.group(1)] = jp

    pairs = []
    for ep in edfs:
        m = re.search(r"(P\d+_\d+)", ep.stem)
        if not m:
            continue
        key = m.group(1)
        if key in json_map:
            pairs.append((str(ep), str(json_map[key])))

    if not pairs:
        raise ValueError("No EDF/JSON pairs found in folders.")
    return pairs


def get_pairs_from_config(cfg):
    """Get pairs from config (same as baseline)"""
    if "edf_dir" in cfg["data"]["edf"] and "labels_json" in cfg["data"]["edf"]:
        # Single file mode
        return [(cfg["data"]["edf"]["dir"], cfg["data"]["edf"]["labels_json"])]
    else:
        # Directory mode
        return build_pairs_from_dirs(cfg["data"]["edf"]["dir"],
                                     cfg["data"]["edf"]["labels_dir"])


def load_subject_data(edf_paths: List[str], json_paths: List[str], cfg: Dict):
    """
    Load and concatenate all data for a subject from multiple files
    """
    all_X = []
    all_y = []
    total_windows = 0

    for edf_path, json_path in zip(edf_paths, json_paths):
        # Load windows from this file
        X, y = load_edf_windows(edf_path, json_path, cfg)
        all_X.append(X)
        all_y.append(y)
        total_windows += len(X)

    # Concatenate all windows
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    print(f"    Total windows: {total_windows}")
    print(f"    X shape: {X.shape}, y shape: {y.shape}")

    return X, y


def load_edf_windows(edf_path: str, json_path: str, cfg: Dict):
    """Load EDF file and create windows"""
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

    # Handle both list and dict formats
    if isinstance(events, dict):
        events = events.values()

    for ev in events:
        if isinstance(ev, dict) and "start" in ev and "end" in ev:
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


def plot_curves_and_log(y_true, y_pred, y_score, fold, prefix=""):
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
    plt.title(f'Fold {fold} {prefix} ROC Curve')
    plt.grid(alpha=0.3)
    plt.legend()
    wandb.log({f"fold_{fold}/{prefix}/roc_curve": wandb.Image(plt)})
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
    plt.title(f'Fold {fold} {prefix} PR Curve')
    plt.grid(alpha=0.3)
    plt.legend()
    wandb.log({f"fold_{fold}/{prefix}/pr_curve": wandb.Image(plt)})
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_true_flat, y_pred.ravel())

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f'Fold {fold} {prefix} Confusion Matrix')
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

    wandb.log({f"fold_{fold}/{prefix}/confusion_matrix": wandb.Image(plt)})
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


def evaluate(model, loader, criterion, device):
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
# Main GroupKFold Function
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--fold", type=int, default=1, help="Fold number (1-based)")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of folds")
    parser.add_argument("--run_name", type=str, default=None, help="Custom run name for W&B")

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
    run_name = args.run_name if args.run_name else f"unet_groupkfold_fold{args.fold}_of_{args.n_folds}"

    # Initialize W&B
    wandb.init(
        project=cfg['logging']['wandb']['project'],
        entity=cfg['logging']['wandb']['entity'],
        name=run_name,
        config=cfg,
        reinit=True
    )

    print("\n" + "=" * 70)
    print(f"🔬 W&B Run: {run_name}")
    print(f"📊 Mode: GroupKFold {args.fold}/{args.n_folds}")
    print(f"📈 Model: UNet1D")
    print("=" * 70)

    # Get all pairs from config
    pairs = get_pairs_from_config(cfg)
    print(f"\nFound {len(pairs)} total files")

    # Extract subject IDs
    subjects = [subject_id_from_path(edf) for edf, _ in pairs]
    unique_subjects = sorted(set(subjects))
    print(f"Found {len(unique_subjects)} unique subjects")

    if len(unique_subjects) < args.n_folds:
        raise ValueError(f"Need at least {args.n_folds} subjects, got {len(unique_subjects)}")

    # Create GroupKFold splits
    gkf = GroupKFold(n_splits=args.n_folds)
    splits = list(gkf.split(pairs, groups=subjects))
    train_idx, test_idx = splits[args.fold - 1]

    # Get train and test pairs
    train_pairs = [pairs[i] for i in train_idx]
    test_pairs = [pairs[i] for i in test_idx]

    # Pick ONE validation subject from train subjects
    rng = np.random.RandomState(42 + args.fold)
    train_subjs = sorted(set(subject_id_from_path(edf) for edf, _ in train_pairs))
    val_subject = rng.choice(train_subjs)

    val_pairs = [p for p in train_pairs if subject_id_from_path(p[0]) == val_subject]
    train_pairs = [p for p in train_pairs if subject_id_from_path(p[0]) != val_subject]

    print(f"\nFold {args.fold}/{args.n_folds}:")
    print(f"  Train subjects: {len(set(subject_id_from_path(p[0]) for p in train_pairs))}")
    print(f"  Val subject: {val_subject}")
    print(f"  Test subjects: {len(set(subject_id_from_path(p[0]) for p in test_pairs))}")
    print(f"  Train files: {len(train_pairs)}")
    print(f"  Val files: {len(val_pairs)}")
    print(f"  Test files: {len(test_pairs)}")

    # Load training data
    print("\n📂 Loading training data...")
    train_edfs = [p[0] for p in train_pairs]
    train_jsons = [p[1] for p in train_pairs]
    X_train, y_train = load_subject_data(train_edfs, train_jsons, cfg)

    # Load validation data
    print("\n📂 Loading validation data...")
    val_edfs = [p[0] for p in val_pairs]
    val_jsons = [p[1] for p in val_pairs]
    X_val, y_val = load_subject_data(val_edfs, val_jsons, cfg)

    # Load test data
    print("\n📂 Loading test data...")
    test_edfs = [p[0] for p in test_pairs]
    test_jsons = [p[1] for p in test_pairs]
    X_test, y_test = load_subject_data(test_edfs, test_jsons, cfg)

    print(f"\nData sizes:")
    print(f"  Train: {len(X_train)} windows")
    print(f"  Val:   {len(X_val)} windows")
    print(f"  Test:  {len(X_test)} windows")

    # Create dataloaders
    train_loader = create_dataloader(
        X_train, y_train,
        batch_size=cfg['data']['batch_size'],
        num_workers=cfg['data']['num_workers'],
        sampler_type=cfg['trainer'].get('sampler', 'normal')
    )
    val_loader = create_dataloader(
        X_val, y_val,
        batch_size=cfg['data']['batch_size'],
        num_workers=cfg['data']['num_workers'],
        shuffle=False
    )
    test_loader = create_dataloader(
        X_test, y_test,
        batch_size=cfg['data']['batch_size'],
        num_workers=cfg['data']['num_workers'],
        shuffle=False
    )

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

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,} total, {trainable_params:,} trainable")

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
    patience = 0
    patience_limit = cfg['trainer'].get('early_stopping_patience', 10)

    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)

    for epoch in range(1, cfg['trainer']['epochs'] + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler, cfg)

        # Validate
        val_metrics, val_probs, val_preds, val_targets = evaluate(model, val_loader, criterion, device)

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
            plot_curves_and_log(val_targets, val_preds, val_probs, args.fold, prefix="val")

        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f} | Val ROC-AUC: {val_metrics['roc_auc']:.4f}")

        # Save best model
        if val_metrics['f1'] > best_val_f1 + 1e-4:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch
            patience = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_f1': val_metrics['f1'],
                'config': cfg
            }, f"best_model_fold{args.fold}.pt")
            print(f"  → New best model! (F1={val_metrics['f1']:.4f})")
        else:
            patience += 1
            if patience_limit > 0 and patience >= patience_limit:
                print(f"Early stopping at epoch {epoch}")
                break

        scheduler.step()

    # Load best model and evaluate on test set
    print(f"\n📊 Best model from epoch {best_epoch} (Val F1={best_val_f1:.4f})")
    checkpoint = torch.load(f"best_model_fold{args.fold}.pt")
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics, test_probs, test_preds, test_targets = evaluate(model, test_loader, criterion, device)

    print(f"\n{'=' * 70}")
    print(f"TEST RESULTS - Fold {args.fold}/{args.n_folds}")
    print(f"{'=' * 70}")
    print(f"  F1:         {test_metrics['f1']:.4f}")
    print(f"  Precision:  {test_metrics['precision']:.4f}")
    print(f"  Recall:     {test_metrics['recall']:.4f}")
    print(f"  ROC-AUC:    {test_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:     {test_metrics['pr_auc']:.4f}")
    print(f"  Threshold:  {test_metrics['threshold']:.4f}")
    print(f"  TP: {test_metrics['tp']}, FP: {test_metrics['fp']}, "
          f"FN: {test_metrics['fn']}, TN: {test_metrics['tn']}")

    # Log test results
    wandb.log({
        'test/f1': test_metrics['f1'],
        'test/precision': test_metrics['precision'],
        'test/recall': test_metrics['recall'],
        'test/roc_auc': test_metrics['roc_auc'],
        'test/pr_auc': test_metrics['pr_auc'],
        'test/threshold': test_metrics['threshold'],
        'test/tp': test_metrics['tp'],
        'test/fp': test_metrics['fp'],
        'test/fn': test_metrics['fn'],
        'test/tn': test_metrics['tn'],
    })

    # Plot test curves
    plot_curves_and_log(test_targets, test_preds, test_probs, args.fold, prefix="test")

    # Save results
    out_dir = Path("unet_groupkfold_results")
    out_dir.mkdir(exist_ok=True)

    results = {
        'fold': args.fold,
        'n_folds': args.n_folds,
        'val_subject': val_subject,
        'n_train_subjects': len(set(subject_id_from_path(p[0]) for p in train_pairs)),
        'n_test_subjects': len(set(subject_id_from_path(p[0]) for p in test_pairs)),
        'n_train_files': len(train_pairs),
        'n_val_files': len(val_pairs),
        'n_test_files': len(test_pairs),
        'best_epoch': best_epoch,
        'best_val_f1': best_val_f1,
        'test_metrics': test_metrics
    }

    out_file = out_dir / f"fold_{args.fold}_of_{args.n_folds}_results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to {out_file}")
    print(f"\n✅ W&B Run Complete: {wandb.run.url}")

    wandb.finish()


if __name__ == "__main__":
    main()