#!/usr/bin/env python
"""
Run original Schimicek on each channel separately with:
- Single subject: Concatenate all files for subject, then 70-15-15 split
- GroupKFold: Subject-wise splits (all files from subject stay together)


"""
import json
import yaml
import mne
import numpy as np
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
from tabulate import tabulate
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score,
                             average_precision_score, roc_curve,
                             precision_recall_curve, auc)
import matplotlib.pyplot as plt
import wandb
import re

from schimicek_spindle import (
    SchimicekParams,
    detect_spindles_schimicek
)


# ----------------------------
# Helper Functions
# ----------------------------
def subject_id_from_path(edf_path: str) -> str:
    """Extract subject ID from EDF filename (P002_1_raw -> P002)"""
    return Path(edf_path).stem.split("_")[0]


def get_all_subject_pairs(cfg: Dict) -> Dict[str, List[Tuple[Path, Path]]]:
    """
    Get all EDF/JSON pairs grouped by subject
    eg:
        'P001': [(edf_path1, json_path1), (edf_path2, json_path2), ...],
        'P002': [...],
        ...
    }
    """
    data_root = Path(cfg['data_root'])
    edf_dir = data_root / cfg['edf_subdir']
    label_dir = data_root / cfg['label_subdir']

    # Find all EDF files
    all_edf_files = sorted(edf_dir.glob("*.edf"))

    # Group by subject
    subject_pairs = {}

    for edf_path in all_edf_files:
        subject = subject_id_from_path(str(edf_path))

        # Find matching JSON
        base = edf_path.stem.replace("_raw", "").replace("_filtered", "")
        json_files = list(label_dir.glob(f"*{base}*.json"))

        if json_files:
            if subject not in subject_pairs:
                subject_pairs[subject] = []
            subject_pairs[subject].append((edf_path, json_files[0]))

    print(f"\nFound {len(subject_pairs)} subjects:")
    for subject, pairs in subject_pairs.items():
        print(f"  {subject}: {len(pairs)} files")

    return subject_pairs


def load_subject_data(subject_pairs: List[Tuple[Path, Path]], channel: str):
    """
    Load and concatenate all data for a subject from multiple files
    """
    all_eeg = []
    all_labels = []
    total_duration = 0
    sfreq_ref = None

    for edf_path, json_path in subject_pairs:
        # Read EDF
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        sfreq = float(raw.info["sfreq"])

        if sfreq_ref is None:
            sfreq_ref = sfreq
        elif abs(sfreq - sfreq_ref) > 1e-6:
            raise ValueError(f"Sampling rate mismatch in {edf_path}")

        # Check if channel exists
        if channel not in raw.ch_names:
            raise ValueError(f"Channel {channel} not found in {edf_path}")

        # Get data for this channel (convert to microvolts)
        eeg_data = raw.get_data(picks=[channel])[0] * 1e6
        all_eeg.append(eeg_data)

        # Load ground truth labels
        with open(json_path, 'r') as f:
            label_data = json.load(f)

        # Create mask for this file
        labels = np.zeros(len(eeg_data), dtype=np.float32)

        spindles = label_data.get("detected_spindles") or label_data.get("spindles") or []

        if isinstance(spindles, dict):
            spindles = spindles.values()

        for ev in spindles:
            if isinstance(ev, dict) and "start" in ev and "end" in ev:
                start = float(ev["start"])
                end = float(ev["end"])

                s = int(max(0, np.floor(start * sfreq)))
                e = int(min(labels.shape[0], np.ceil(end * sfreq)))

                if e > s:
                    labels[s:e] = 1.0

        all_labels.append(labels)
        total_duration += len(eeg_data) / sfreq

    # Concatenate all files
    concatenated_eeg = np.concatenate(all_eeg)
    concatenated_labels = np.concatenate(all_labels)

    print(f"    Total duration: {total_duration:.1f}s ({total_duration / 3600:.2f} hours)")
    print(f"    Total samples: {len(concatenated_eeg)}")

    return concatenated_eeg, concatenated_labels, sfreq_ref


def split_time_70_15_15(data: np.ndarray, labels: np.ndarray, sfreq: float):
    """
    Split single subject's concatenated data into 70% train, 15% validation, 15% test
    """
    n_samples = len(data)
    train_end = int(0.7 * n_samples)
    val_end = train_end + int(0.15 * n_samples)

    # Split data
    train_data = data[:train_end]
    train_labels = labels[:train_end]

    val_data = data[train_end:val_end]
    val_labels = labels[train_end:val_end]

    test_data = data[val_end:]
    test_labels = labels[val_end:]

    # Print split info
    print(f"\n  Split sizes:")
    print(f"    Train: {len(train_data) / n_samples * 100:.1f}% ({len(train_data) / sfreq / 60:.1f} min)")
    print(f"    Val:   {len(val_data) / n_samples * 100:.1f}% ({len(val_data) / sfreq / 60:.1f} min)")
    print(f"    Test:  {len(test_data) / n_samples * 100:.1f}% ({len(test_data) / sfreq / 60:.1f} min)")

    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)


def create_groupkfold_splits(subject_pairs: Dict[str, List[Tuple[Path, Path]]],
                             n_folds: int,
                             fold_idx: int,
                             random_state: int = 42):
    """
    Create train/val/test splits using GroupKFold at subject level
    Returns: (train_subjects, val_subject, test_subjects)
    """
    subjects = list(subject_pairs.keys())

    if len(subjects) < n_folds:
        raise ValueError(f"Need >= {n_folds} subjects, got {len(subjects)}")

    # Create group labels (each subject is a group)
    groups = subjects  # Each subject is its own group

    # Create dummy indices for splitting
    indices = np.arange(len(subjects))

    # Create GroupKFold splits
    gkf = GroupKFold(n_splits=n_folds)
    splits = list(gkf.split(indices, groups=groups))
    train_idx, test_idx = splits[fold_idx - 1]  # fold_idx is 1-based

    # Get train and test subjects
    train_subjects = [subjects[i] for i in train_idx]
    test_subjects = [subjects[i] for i in test_idx]

    # Pick ONE validation subject from train subjects
    rng = np.random.RandomState(random_state + fold_idx)
    val_subject = rng.choice(train_subjects)

    # Remove validation subject from train
    train_subjects = [s for s in train_subjects if s != val_subject]

    print(f"\n  Fold {fold_idx}/{n_folds}:")
    print(f"    Train subjects ({len(train_subjects)}): {train_subjects}")
    print(f"    Val subject: {val_subject}")
    print(f"    Test subjects ({len(test_subjects)}): {test_subjects}")

    return train_subjects, val_subject, test_subjects


def find_best_threshold(y_true, y_score):
    """
    Find threshold that maximizes F1 score on validation set
    """
    from sklearn.metrics import precision_recall_curve

    if len(np.unique(y_true)) < 2:
        return 0.5

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    best_idx = np.argmax(f1_scores)

    return thresholds[best_idx]

def calculate_metrics(y_true, y_pred, y_score=None):
    """
    Calculate all metrics for a single channel
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


def plot_curves_and_log(y_true, y_pred, y_score, channel_name, prefix=""):
    """
    Plot ROC and PR curves and log to W&B
    """
    y_true_flat = np.asarray(y_true).ravel()
    y_score_flat = np.asarray(y_score).ravel()
    y_pred_flat = np.asarray(y_pred).ravel()

    if len(np.unique(y_true_flat)) < 2:
        print(f"  Not enough classes for ROC/PR curves for {channel_name}")
        return

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true_flat, y_score_flat)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{prefix} {channel_name} ROC Curve')
    plt.grid(alpha=0.3)
    plt.legend()
    wandb.log({f"{prefix}/{channel_name}/roc_curve": wandb.Image(plt)})
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
    plt.title(f'{prefix} {channel_name} PR Curve')
    plt.grid(alpha=0.3)
    plt.legend()
    wandb.log({f"{prefix}/{channel_name}/pr_curve": wandb.Image(plt)})
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_true_flat, y_pred_flat)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f'{prefix} {channel_name} Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks([0, 1], ['No Spindle', 'Spindle'])
    plt.yticks([0, 1], ['No Spindle', 'Spindle'])

    # Add numbers to cells
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    wandb.log({f"{prefix}/{channel_name}/confusion_matrix": wandb.Image(plt)})
    plt.close()


def process_channel(channel: str,
                    train_data: np.ndarray, train_labels: np.ndarray,
                    val_data: np.ndarray, val_labels: np.ndarray,
                    test_data: np.ndarray, test_labels: np.ndarray,
                    params: SchimicekParams):
    """
    Process a single channel with train/val/test splits
    """
    print(f"\n    Processing channel {channel}...")

    # Run on TRAIN
    train_result = detect_spindles_schimicek(train_data, params)
    train_metrics = calculate_metrics(train_labels, train_result['spindle_mask'])

    # Run on VAL (for threshold optimization)
    val_result = detect_spindles_schimicek(val_data, params)
    val_confidence = val_result['peak_to_peak'] / (params.amplitude_threshold_uv + 1e-8)
    val_confidence = np.clip(val_confidence, 0, 2)

    # Find best threshold on validation set
    best_thr = find_best_threshold(val_labels, val_confidence)

    # Apply threshold to get validation predictions
    val_pred = (val_confidence >= best_thr).astype(int)

    # Calculate ALL validation metrics with scores (for ROC/PR AUC)
    val_metrics = calculate_metrics(val_labels, val_pred, val_confidence)

    # Run on TEST with optimal threshold
    test_result = detect_spindles_schimicek(test_data, params)
    test_confidence = test_result['peak_to_peak'] / (params.amplitude_threshold_uv + 1e-8)
    test_confidence = np.clip(test_confidence, 0, 2)
    test_pred = (test_confidence >= best_thr).astype(int)
    test_metrics = calculate_metrics(test_labels, test_pred, test_confidence)

    print(f"      Train F1: {train_metrics['f1']:.4f}")
    print(f"      Val F1:   {val_metrics['f1']:.4f} (thr={best_thr:.3f})")
    print(f"      Val ROC-AUC: {val_metrics.get('roc_auc', 0):.4f}")
    print(f"      Val PR-AUC: {val_metrics.get('pr_auc', 0):.4f}")
    print(f"      Test F1:  {test_metrics['f1']:.4f}")

    # Log curves to W&B (like CNN script)
    if wandb.run is not None:
        # Validation curves
        plot_curves_and_log(
            val_labels, val_pred, val_confidence,
            channel, prefix="val"
        )
        # Test curves
        plot_curves_and_log(
            test_labels, test_pred, test_confidence,
            channel, prefix="test"
        )

    return {
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics,
        'best_threshold': float(best_thr),
        'val_labels': val_labels.tolist(),
        'val_pred': val_pred.tolist(),
        'val_scores': val_confidence.tolist(),
        'test_labels': test_labels.tolist(),
        'test_pred': test_pred.tolist(),
        'test_scores': test_confidence.tolist()
    }
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

    # Get all channels
    all_channels = cfg["channels"]["names"]

    # Initialize Schimicek parameters
    params = SchimicekParams(
        fs=cfg.get('sampling_rate', 200.0),
        amplitude_threshold_uv=cfg['schimicek']['amplitude_threshold_uv'],
        min_duration_s=cfg['schimicek']['min_duration_s'],
        alpha_ratio_threshold=cfg['schimicek']['alpha_ratio_threshold'],
        muscle_rms_threshold_uv=cfg['schimicek']['muscle_rms_threshold_uv']
    )

    # Create W&B run name
    if args.run_name:
        run_name = args.run_name
    else:
        if args.mode == 'single':
            run_name = f"schimicek_subject_{args.subject}"
        else:
            run_name = f"schimicek_groupkfold_fold{args.fold}_of_{args.n_folds}"

    # Initialize W&B
    wandb.init(
        project=cfg['wandb']['project'],
        entity=cfg['wandb']['entity'],
        name=run_name,
        config={
            'mode': args.mode,
            'subject': args.subject,
            'fold': args.fold if args.mode == 'groupkfold' else None,
            'n_folds': args.n_folds if args.mode == 'groupkfold' else None,
            'channels': all_channels,
            'params': {
                'amplitude_threshold_uv': params.amplitude_threshold_uv,
                'min_duration_s': params.min_duration_s,
                'alpha_ratio_threshold': params.alpha_ratio_threshold,
                'muscle_rms_threshold_uv': params.muscle_rms_threshold_uv,
                'fs': params.fs
            }
        },
        reinit=True
    )

    print("\n" + "=" * 70)
    print(f"W&B Run: {run_name}")
    print(f"Mode: {args.mode.upper()}")
    print(f"Channels: {all_channels}")
    print("=" * 70)

    # Log parameters to W&B
    wandb.log({
        'params/amplitude_threshold_uv': params.amplitude_threshold_uv,
        'params/min_duration_s': params.min_duration_s,
        'params/alpha_ratio_threshold': params.alpha_ratio_threshold,
        'params/muscle_rms_threshold_uv': params.muscle_rms_threshold_uv,
        'params/fs': params.fs,
    })

    # Get all subject pairs
    subject_pairs = get_all_subject_pairs(cfg)

    if args.mode == 'single':
        # ----------------------------
        # SINGLE SUBJECT MODE
        # ----------------------------
        if args.subject not in subject_pairs:
            raise ValueError(f"Subject {args.subject} not found. Available: {list(subject_pairs.keys())}")

        subject_data = subject_pairs[args.subject]
        print(f"\n📁 Processing subject: {args.subject} ({len(subject_data)} files)")

        # Store results per channel
        channel_results = {}

        # Process each channel
        for channel in all_channels:
            print(f"\n  {'=' * 50}")
            print(f"  Channel: {channel}")
            print(f"  {'=' * 50}")

            try:
                # Load and concatenate all data for this subject
                eeg_data, labels, sfreq = load_subject_data(subject_data, channel)

                # Split into train/val/test (70-15-15)
                (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = \
                    split_time_70_15_15(eeg_data, labels, sfreq)

                # Process channel
                results = process_channel(
                    channel,
                    train_data, train_labels,
                    val_data, val_labels,
                    test_data, test_labels,
                    params
                )

                channel_results[channel] = results

                # Log metrics to W&B
                # After process_channel, log ALL metrics
                wandb.log({
                    f"{channel}/train_f1": results['train']['f1'],
                    f"{channel}/train_precision": results['train']['precision'],
                    f"{channel}/train_recall": results['train']['recall'],
                    f"{channel}/train_accuracy": results['train']['accuracy'],

                    f"{channel}/val_f1": results['val']['f1'],
                    f"{channel}/val_precision": results['val']['precision'],
                    f"{channel}/val_recall": results['val']['recall'],
                    f"{channel}/val_accuracy": results['val']['accuracy'],
                    f"{channel}/val_roc_auc": results['val'].get('roc_auc', 0),
                    f"{channel}/val_pr_auc": results['val'].get('pr_auc', 0),

                    f"{channel}/test_f1": results['test']['f1'],
                    f"{channel}/test_precision": results['test']['precision'],
                    f"{channel}/test_recall": results['test']['recall'],
                    f"{channel}/test_accuracy": results['test']['accuracy'],
                    f"{channel}/test_roc_auc": results['test'].get('roc_auc', 0),
                    f"{channel}/test_pr_auc": results['test'].get('pr_auc', 0),

                    f"{channel}/best_threshold": results['best_threshold'],
                    f"{channel}/test_tp": results['test']['tp'],
                    f"{channel}/test_fp": results['test']['fp'],
                    f"{channel}/test_fn": results['test']['fn'],
                    f"{channel}/test_tn": results['test']['tn'],
                })

            except Exception as e:
                print(f"  ❌ Error: {e}")
                import traceback
                traceback.print_exc()

    else:
        # ----------------------------
        # GROUPKFOLD MODE
        # ----------------------------
        # Create subject-wise splits
        train_subjects, val_subject, test_subjects = create_groupkfold_splits(
            subject_pairs, args.n_folds, args.fold, random_state=42
        )

        # Store results per channel
        channel_results = {}

        # Process each channel
        for channel in all_channels:
            print(f"\n  {'=' * 50}")
            print(f"  Channel: {channel}")
            print(f"  {'=' * 50}")

            try:
                # Load and concatenate TRAIN subjects
                train_data_list = []
                train_labels_list = []
                for subject in train_subjects:
                    data, labels, _ = load_subject_data(subject_pairs[subject], channel)
                    train_data_list.append(data)
                    train_labels_list.append(labels)

                train_data = np.concatenate(train_data_list)
                train_labels = np.concatenate(train_labels_list)

                # Load VAL subject
                val_data, val_labels, _ = load_subject_data(subject_pairs[val_subject], channel)

                # Load TEST subjects
                test_data_list = []
                test_labels_list = []
                for subject in test_subjects:
                    data, labels, _ = load_subject_data(subject_pairs[subject], channel)
                    test_data_list.append(data)
                    test_labels_list.append(labels)

                test_data = np.concatenate(test_data_list)
                test_labels = np.concatenate(test_labels_list)

                print(f"\n    Train: {len(train_data) / params.fs / 60:.1f} min")
                print(f"    Val:   {len(val_data) / params.fs / 60:.1f} min")
                print(f"    Test:  {len(test_data) / params.fs / 60:.1f} min")

                # Process channel
                results = process_channel(
                    channel,
                    train_data, train_labels,
                    val_data, val_labels,
                    test_data, test_labels,
                    params
                )

                channel_results[channel] = results

                # Log metrics to W&B
                wandb.log({
                    f"{channel}/train_f1": results['train']['f1'],
                    f"{channel}/val_f1": results['val']['f1'],
                    f"{channel}/test_f1": results['test']['f1'],
                    f"{channel}/test_precision": results['test']['precision'],
                    f"{channel}/test_recall": results['test']['recall'],
                    f"{channel}/test_roc_auc": results['test'].get('roc_auc', 0),
                    f"{channel}/test_pr_auc": results['test'].get('pr_auc', 0),
                    f"{channel}/best_threshold": results['best_threshold'],
                    f"{channel}/test_tp": results['test']['tp'],
                    f"{channel}/test_fp": results['test']['fp'],
                    f"{channel}/test_fn": results['test']['fn'],
                    f"{channel}/test_tn": results['test']['tn'],
                })

            except Exception as e:
                print(f"  ❌ Error: {e}")
                import traceback
                traceback.print_exc()

    # Print summary table for TEST set
    print("\n" + "=" * 70)
    if args.mode == 'single':
        print(f"📊 TEST SET RESULTS PER CHANNEL (Subject: {args.subject})")
    else:
        print(f"📊 TEST SET RESULTS PER CHANNEL (Fold {args.fold}/{args.n_folds})")
    print("=" * 70)

    test_summary = []
    for channel in all_channels:
        if channel in channel_results:
            m = channel_results[channel]['test']
            test_summary.append([
                channel,
                f"{m['precision']:.4f}",
                f"{m['recall']:.4f}",
                f"{m['f1']:.4f}",
                f"{m.get('roc_auc', 0):.4f}",
                f"{m.get('pr_auc', 0):.4f}",
                m['tp'],
                m['fp'],
                m['fn'],
                channel_results[channel]['best_threshold']
            ])

    headers = ['Channel', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC', 'TP', 'FP', 'FN', 'Opt Thr']
    print(tabulate(test_summary, headers=headers, tablefmt='grid'))

    # Calculate across-channel statistics
    f1_scores = [float(row[3]) for row in test_summary]
    precision_scores = [float(row[1]) for row in test_summary]
    recall_scores = [float(row[2]) for row in test_summary]
    roc_auc_scores = [float(row[4]) for row in test_summary]
    pr_auc_scores = [float(row[5]) for row in test_summary]

    print("\n" + "=" * 70)
    print("📈 ACROSS-CHANNEL STATISTICS (TEST SET)")
    print("=" * 70)
    print(f"F1 Score     - Mean: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"               Min: {np.min(f1_scores):.4f}, Max: {np.max(f1_scores):.4f}")
    print(f"Precision    - Mean: {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
    print(f"Recall       - Mean: {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
    print(f"ROC-AUC      - Mean: {np.mean(roc_auc_scores):.4f} ± {np.std(roc_auc_scores):.4f}")
    print(f"PR-AUC       - Mean: {np.mean(pr_auc_scores):.4f} ± {np.std(pr_auc_scores):.4f}")

    # Best and worst channels
    best_channel = all_channels[np.argmax(f1_scores)]
    worst_channel = all_channels[np.argmin(f1_scores)]
    print(f"\nBest channel:  {best_channel} (F1={np.max(f1_scores):.4f})")
    print(f"Worst channel: {worst_channel} (F1={np.min(f1_scores):.4f})")

    # Log across-channel statistics to W&B
    wandb.log({
        'across_channels/mean_f1': np.mean(f1_scores),
        'across_channels/std_f1': np.std(f1_scores),
        'across_channels/min_f1': np.min(f1_scores),
        'across_channels/max_f1': np.max(f1_scores),
        'across_channels/mean_precision': np.mean(precision_scores),
        'across_channels/mean_recall': np.mean(recall_scores),
        'across_channels/mean_roc_auc': np.mean(roc_auc_scores),
        'across_channels/mean_pr_auc': np.mean(pr_auc_scores),
        'across_channels/best_channel': best_channel,
        'across_channels/best_f1': np.max(f1_scores),
        'across_channels/worst_channel': worst_channel,
        'across_channels/worst_f1': np.min(f1_scores),
        'across_channels/n_channels': len(all_channels),
    })

    # Log summary table to W&B
    summary_table = wandb.Table(
        columns=['Channel', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC', 'TP', 'FP', 'FN', 'Opt Thr']
    )
    for row in test_summary:
        summary_table.add_data(*row)
    wandb.log({'test_summary_table': summary_table})

    # Save results
    out_dir = Path("schimicek_results")
    out_dir.mkdir(exist_ok=True)

    if args.mode == 'single':
        output_file = out_dir / f"subject_{args.subject}_results.json"
    else:
        output_file = out_dir / f"fold_{args.fold}_of_{args.n_folds}_results.json"

    with open(output_file, 'w') as f:
        json.dump({
            'mode': args.mode,
            'subject': args.subject if args.mode == 'single' else None,
            'fold': args.fold if args.mode == 'groupkfold' else None,
            'n_folds': args.n_folds if args.mode == 'groupkfold' else None,
            'per_channel': {
                ch: {
                    'test': channel_results[ch]['test'],
                    'best_threshold': channel_results[ch]['best_threshold']
                } for ch in all_channels if ch in channel_results
            },
            'across_channels': {
                'mean_f1': float(np.mean(f1_scores)),
                'std_f1': float(np.std(f1_scores)),
                'best_channel': best_channel,
                'best_f1': float(np.max(f1_scores)),
                'mean_roc_auc': float(np.mean(roc_auc_scores)),
                'mean_pr_auc': float(np.mean(pr_auc_scores)),
            }
        }, f, indent=2)

    print(f"\n💾 Results saved to {output_file}")
    print(f"\n✅ W&B Run Complete: {wandb.run.url}")

    wandb.finish()


if __name__ == "__main__":
    main()