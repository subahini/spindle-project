#!/usr/bin/env python
"""
Run original Schimicek on each channel separately and log ALL metrics to W&B
"""
import json
import yaml
import mne
import numpy as np
import argparse
from pathlib import Path
from typing import List, Dict
import pandas as pd
from tabulate import tabulate
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score,
                             average_precision_score)
import wandb

from schimicek_spindle import (
    SchimicekParams,
    detect_spindles_schimicek
)


def load_single_channel_data(edf_path: str, json_path: str, channel: str):
    """
    Load data for a single channel (original method)
    """
    # Read EDF
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    sfreq = float(raw.info["sfreq"])

    # Check if channel exists
    if channel not in raw.ch_names:
        raise ValueError(f"Channel {channel} not found in {edf_path}")

    # Get data for this channel (convert to microvolts)
    eeg_data = raw.get_data(picks=[channel])[0] * 1e6  # Convert to µV
    n_samples = raw.n_times

    # Load ground truth labels
    with open(json_path, 'r') as f:
        label_data = json.load(f)

    # Create ground truth mask (spindle if marked on ANY channel)
    labels = np.zeros(n_samples, dtype=np.float32)

    spindles = label_data.get("detected_spindles") or label_data.get("spindles") or []

    # Handle both list and dict formats
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

    return eeg_data, labels, sfreq


def calculate_metrics(y_true, y_pred):
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

    return {
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--fold", type=int, default=None,
                        help="Fold number for GroupKFold (if using cross-validation)")
    parser.add_argument("--file", type=str, default=None,
                        help="Single file to process (if not using folds)")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Custom run name for W&B")

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Get all channels (same as DL model)
    all_channels = cfg["channels"]["names"]  # ["C3", "C4", "F3", "F4"]

    # Initialize Schimicek parameters from config
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
        run_name = f"schimicek_all_channels"
        if args.fold:
            run_name += f"_fold{args.fold}"
        if args.file:
            run_name += f"_{Path(args.file).stem}"

    # Initialize W&B (ALWAYS, even if not specified)
    wandb.init(
        project=cfg['wandb']['project'],
        entity=cfg['wandb']['entity'],
        name=run_name,
        config={
            'fold': args.fold,
            'file': args.file,
            'channels': all_channels,
            'params': {
                'amplitude_threshold_uv': params.amplitude_threshold_uv,
                'min_duration_s': params.min_duration_s,
                'alpha_ratio_threshold': params.alpha_ratio_threshold,
                'muscle_rms_threshold_uv': params.muscle_rms_threshold_uv,
                'fs': params.fs
            },
            'config_file': cfg
        },
        reinit=True  # Allow multiple runs
    )

    print("\n" + "=" * 70)
    print(f" W&B Run: {run_name}")
    print(f"Running Schimicek on {len(all_channels)} channels: {all_channels}")
    print(f"Parameters used: amp_thr={params.amplitude_threshold_uv}, min_dur={params.min_duration_s}")
    print("=" * 70)

    # Log parameters to W&B
    wandb.log({
        'params/amplitude_threshold_uv': params.amplitude_threshold_uv,
        'params/min_duration_s': params.min_duration_s,
        'params/alpha_ratio_threshold': params.alpha_ratio_threshold,
        'params/muscle_rms_threshold_uv': params.muscle_rms_threshold_uv,
        'params/fs': params.fs,
    })

    # Get list of files to process
    data_root = Path(cfg['data_root'])
    edf_dir = data_root / cfg['edf_subdir']
    label_dir = data_root / cfg['label_subdir']

    if args.file:
        # Process single file
        edf_files = [Path(args.file)]
    else:
        # Process all files
        all_edf_files = sorted(edf_dir.glob("*.edf"))
        edf_files = all_edf_files

    print(f"\nFound {len(edf_files)} EDF files to process")

    # Store results for all channels
    all_results = {ch: {
        'metrics': [],
        'total_tp': 0,
        'total_fp': 0,
        'total_fn': 0,
        'total_tn': 0,
        'n_spindles_detected': 0,
        'n_spindles_true': 0
    } for ch in all_channels}

    # Also store per-file results
    per_file_results = []

    # Process each EDF file
    for file_idx, edf_path in enumerate(edf_files):
        # Find matching JSON
        base = edf_path.stem.replace("_raw", "").replace("_filtered", "")
        json_files = list(label_dir.glob(f"*{base}*.json"))

        if not json_files:
            print(f"Warning: No JSON for {edf_path.name}")
            continue

        json_path = json_files[0]
        print(f"\n Processing ({file_idx + 1}/{len(edf_files)}): {edf_path.name}")

        file_results = {'file': edf_path.name}

        # Run Schimicek on each channel separately
        for channel in all_channels:
            try:
                # Load data for this channel
                eeg_data, true_mask, sfreq = load_single_channel_data(
                    str(edf_path), str(json_path), channel
                )

                # Run original Schimicek detector (NO fusion, just single channel)
                result = detect_spindles_schimicek(eeg_data, params)

                # Calculate metrics
                metrics = calculate_metrics(true_mask, result['spindle_mask'])

                # Store in file results
                file_results[f'{channel}_f1'] = metrics['f1']
                file_results[f'{channel}_precision'] = metrics['precision']
                file_results[f'{channel}_recall'] = metrics['recall']
                file_results[f'{channel}_accuracy'] = metrics['accuracy']
                file_results[f'{channel}_spindles'] = result['n_spindles']
                file_results[f'{channel}_tp'] = metrics['tp']
                file_results[f'{channel}_fp'] = metrics['fp']
                file_results[f'{channel}_fn'] = metrics['fn']
                file_results[f'{channel}_tn'] = metrics['tn']

                # Aggregate for channel results
                all_results[channel]['metrics'].append(metrics)
                all_results[channel]['total_tp'] += metrics['tp']
                all_results[channel]['total_fp'] += metrics['fp']
                all_results[channel]['total_fn'] += metrics['fn']
                all_results[channel]['total_tn'] += metrics['tn']
                all_results[channel]['n_spindles_detected'] += result['n_spindles']
                all_results[channel]['n_spindles_true'] += int(true_mask.sum())

                print(f"  Channel {channel}: F1={metrics['f1']:.3f}, "
                      f"P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, "
                      f"Spindles={result['n_spindles']}")

                # LOG PER-CHANNEL PER-FILE METRICS TO W&B
                wandb.log({
                    f"{edf_path.stem}/{channel}/f1": metrics['f1'],
                    f"{edf_path.stem}/{channel}/precision": metrics['precision'],
                    f"{edf_path.stem}/{channel}/recall": metrics['recall'],
                    f"{edf_path.stem}/{channel}/accuracy": metrics['accuracy'],
                    f"{edf_path.stem}/{channel}/spindles_detected": result['n_spindles'],
                    f"{edf_path.stem}/{channel}/tp": metrics['tp'],
                    f"{edf_path.stem}/{channel}/fp": metrics['fp'],
                    f"{edf_path.stem}/{channel}/fn": metrics['fn'],
                    f"{edf_path.stem}/{channel}/tn": metrics['tn'],
                })

            except Exception as e:
                print(f"  ❌ Error on channel {channel}: {e}")
                file_results[f'{channel}_f1'] = None

        per_file_results.append(file_results)

        # LOG SUMMARY FOR THIS FILE
        wandb.log({
            f"{edf_path.stem}/n_files_processed": file_idx + 1,
        })

    if not all_results[all_channels[0]]['metrics']:
        print("No files processed successfully!")
        wandb.finish()
        return

    # Calculate aggregate metrics for each channel
    print("\n" + "=" * 70)
    title = "📊 AGGREGATE RESULTS PER CHANNEL"
    if args.fold:
        title += f" (Fold {args.fold})"
    print(title)
    print("=" * 70)

    # Create summary table
    summary_data = []
    channel_summaries = {}

    for channel in all_channels:
        res = all_results[channel]
        total_tp_fp = res['total_tp'] + res['total_fp']
        total_tp_fn = res['total_tp'] + res['total_fn']

        agg_precision = res['total_tp'] / total_tp_fp if total_tp_fp > 0 else 0
        agg_recall = res['total_tp'] / total_tp_fn if total_tp_fn > 0 else 0
        agg_f1 = 2 * agg_precision * agg_recall / (agg_precision + agg_recall) if (
                                                                                              agg_precision + agg_recall) > 0 else 0
        agg_accuracy = (res['total_tp'] + res['total_tn']) / (
                    res['total_tp'] + res['total_tn'] + res['total_fp'] + res['total_fn'])

        summary_data.append([
            channel,
            f"{agg_precision:.4f}",
            f"{agg_recall:.4f}",
            f"{agg_f1:.4f}",
            f"{agg_accuracy:.4f}",
            res['total_tp'],
            res['total_fp'],
            res['total_fn'],
            res['total_tn'],
            res['n_spindles_detected'],
            res['n_spindles_true']
        ])

        # Store for logging
        channel_summaries[channel] = {
            'precision': agg_precision,
            'recall': agg_recall,
            'f1': agg_f1,
            'accuracy': agg_accuracy,
            'tp': res['total_tp'],
            'fp': res['total_fp'],
            'fn': res['total_fn'],
            'tn': res['total_tn'],
            'spindles_detected': res['n_spindles_detected'],
            'spindle_samples_true': res['n_spindles_true']
        }

        # LOG CHANNEL SUMMARY TO W&B
        wandb.log({
            f"summary/{channel}/precision": agg_precision,
            f"summary/{channel}/recall": agg_recall,
            f"summary/{channel}/f1": agg_f1,
            f"summary/{channel}/accuracy": agg_accuracy,
            f"summary/{channel}/tp": res['total_tp'],
            f"summary/{channel}/fp": res['total_fp'],
            f"summary/{channel}/fn": res['total_fn'],
            f"summary/{channel}/tn": res['total_tn'],
            f"summary/{channel}/spindles_detected": res['n_spindles_detected'],
            f"summary/{channel}/spindle_samples_true": res['n_spindles_true'],
        })

    # Print table
    headers = ['Channel', 'Precision', 'Recall', 'F1', 'Accuracy',
               'TP', 'FP', 'FN', 'TN', 'Detected', 'True Samples']
    print(tabulate(summary_data, headers=headers, tablefmt='grid'))

    # Calculate across-channel statistics
    f1_scores = [float(row[3]) for row in summary_data]
    precision_scores = [float(row[1]) for row in summary_data]
    recall_scores = [float(row[2]) for row in summary_data]
    accuracy_scores = [float(row[4]) for row in summary_data]

    print("\n" + "=" * 70)
    print("📈 ACROSS-CHANNEL STATISTICS")
    print("=" * 70)
    print(f"F1 Score - Mean: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"          Min: {np.min(f1_scores):.4f}, Max: {np.max(f1_scores):.4f}")
    print(f"Precision - Mean: {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
    print(f"Recall    - Mean: {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
    print(f"Accuracy  - Mean: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")

    # Best and worst channels
    best_channel = all_channels[np.argmax(f1_scores)]
    worst_channel = all_channels[np.argmin(f1_scores)]
    print(f"\nBest channel:  {best_channel} (F1={np.max(f1_scores):.4f})")
    print(f"Worst channel: {worst_channel} (F1={np.min(f1_scores):.4f})")

    # LOG ACROSS-CHANNEL STATISTICS TO W&B
    wandb.log({
        'across_channels/mean_f1': np.mean(f1_scores),
        'across_channels/std_f1': np.std(f1_scores),
        'across_channels/min_f1': np.min(f1_scores),
        'across_channels/max_f1': np.max(f1_scores),
        'across_channels/mean_precision': np.mean(precision_scores),
        'across_channels/mean_recall': np.mean(recall_scores),
        'across_channels/mean_accuracy': np.mean(accuracy_scores),
        'across_channels/best_channel': best_channel,
        'across_channels/best_f1': np.max(f1_scores),
        'across_channels/worst_channel': worst_channel,
        'across_channels/worst_f1': np.min(f1_scores),
        'across_channels/n_channels': len(all_channels),
        'across_channels/n_files': len(per_file_results),
    })

    # LOG SUMMARY TABLE AS W&B TABLE
    summary_table = wandb.Table(
        columns=['Channel', 'Precision', 'Recall', 'F1', 'Accuracy',
                 'TP', 'FP', 'FN', 'TN', 'Spindles Detected', 'True Samples']
    )
    for row in summary_data:
        summary_table.add_data(*row)
    wandb.log({'summary_table': summary_table})

    # LOG PER-FILE RESULTS AS W&B TABLE
    if per_file_results:
        file_columns = list(per_file_results[0].keys())
        file_table = wandb.Table(columns=file_columns)
        for file_res in per_file_results:
            file_table.add_data(*[file_res.get(col, None) for col in file_columns])
        wandb.log({'per_file_results': file_table})

    # Save results to file
    out_dir = Path("schimicek_per_channel_results")
    out_dir.mkdir(exist_ok=True)

    suffix = ""
    if args.fold:
        suffix = f"_fold{args.fold}"
    if args.file:
        suffix = f"_{Path(args.file).stem}"

    # Save summary
    with open(out_dir / f"results{suffix}.json", 'w') as f:
        json.dump({
            'fold': args.fold,
            'file': str(args.file) if args.file else None,
            'per_channel': channel_summaries,
            'across_channels': {
                'mean_f1': float(np.mean(f1_scores)),
                'std_f1': float(np.std(f1_scores)),
                'best_channel': best_channel,
                'best_f1': float(np.max(f1_scores)),
                'worst_channel': worst_channel,
                'worst_f1': float(np.min(f1_scores)),
            },
            'n_files': len(per_file_results)
        }, f, indent=2)

    # Save per-file results
    df = pd.DataFrame(per_file_results)
    df.to_csv(out_dir / f"per_file{suffix}.csv", index=False)

    # LOG FILE SAVED
    wandb.log({
        'results_saved': str(out_dir / f"results{suffix}.json"),
        'per_file_saved': str(out_dir / f"per_file{suffix}.csv")
    })

    print(f"\n💾 Results saved to {out_dir}/")
    print(f"\n✅ W&B Run Complete: {wandb.run.url}")

    # Finish W&B run
    wandb.finish()


if __name__ == "__main__":
    main()