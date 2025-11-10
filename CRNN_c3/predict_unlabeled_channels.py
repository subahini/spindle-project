#!/usr/bin/env python3
"""
Apply trained single-channel model to predict spindles on unlabeled channels
"""
import os
import argparse
import json
from typing import List, Dict, Tuple
import numpy as np
import torch
import yaml
from tqdm import tqdm

import Crnn


# Removed: from cross_channel_train import SingleChannelDataset, get_channel_index


def get_channel_index(channel_name: str, all_channels: List[str]) -> int:
    """Get index of channel in the channel list"""
    try:
        return all_channels.index(channel_name)
    except ValueError:
        raise ValueError(f"Channel {channel_name} not found in {all_channels}")


def load_trained_model(checkpoint_path: str, device):
    """Load a trained single-channel model"""
    print(f"Loading model from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    cfg = ckpt["cfg"]
    mcfg = cfg["model"]
    scfg = cfg["spectrogram"]

    # Create model with single channel
    model = Crnn.CRNN2D_BiGRU(
        c_in=1,  # Single channel
        base_ch=mcfg["base_ch"],
        fpn_ch=mcfg["fpn_ch"],
        rnn_hidden=mcfg["rnn_hidden"],
        rnn_layers=mcfg["rnn_layers"],
        bidirectional=mcfg["bidirectional"],
        bias_init_prior=mcfg.get("bias_init_prior", None),
        use_se=mcfg["use_se"],
        sfreq=cfg["data"]["sfreq"],
        n_fft=scfg["n_fft"],
        hop_length=scfg["hop_length"],
        win_length=scfg["win_length"],
        center=scfg["center"],
        power=scfg["power"],
        upsample_mode=mcfg.get("upsample_mode", "linear")
    )

    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    train_channel = ckpt.get("train_channel", "C3")
    print(f"Model was trained on channel: {train_channel}")
    print(f"Best validation F1: {ckpt.get('best_f1', 'N/A')}")

    return model, cfg, train_channel


def predict_channel(model, X_full: np.ndarray, channel_idx: int,
                    device, batch_size: int = 16, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict spindles for a specific channel

    Returns:
        probs: [N, T] probability predictions
        binary_preds: [N, T] binary predictions at threshold
    """
    # Create dataset for this channel
    dummy_labels = np.zeros((len(X_full), X_full.shape[-1]), dtype=np.float32)
    ds = SingleChannelDataset(X_full, dummy_labels, channel_idx, normalize="zscore")
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_probs = []

    with torch.no_grad():
        for x, _ in tqdm(loader, desc=f"Predicting", leave=False):
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    binary_preds = (probs >= threshold).astype(np.int32)

    return probs, binary_preds


def extract_events(predictions: np.ndarray, sfreq: float, min_duration: float = 0.3) -> List[Dict]:
    """
    Convert binary predictions to spindle events

    Args:
        predictions: [N, T] binary predictions (flattened across all windows)
        sfreq: sampling frequency
        min_duration: minimum spindle duration in seconds

    Returns:
        List of spindle events with start/end times
    """
    # Flatten predictions
    preds_flat = predictions.reshape(-1)

    events = []
    in_spindle = False
    start_idx = 0

    min_samples = int(min_duration * sfreq)

    for i, val in enumerate(preds_flat):
        if val == 1 and not in_spindle:
            # Start of spindle
            in_spindle = True
            start_idx = i
        elif val == 0 and in_spindle:
            # End of spindle
            duration_samples = i - start_idx
            if duration_samples >= min_samples:
                events.append({
                    "start": float(start_idx / sfreq),
                    "end": float(i / sfreq),
                    "duration": float(duration_samples / sfreq)
                })
            in_spindle = False

    # Handle case where spindle extends to end
    if in_spindle:
        duration_samples = len(preds_flat) - start_idx
        if duration_samples >= min_samples:
            events.append({
                "start": float(start_idx / sfreq),
                "end": float(len(preds_flat) / sfreq),
                "duration": float(duration_samples / sfreq)
            })

    return events


def predict_all_unlabeled_channels(
        checkpoint_path: str,
        data_path: str,
        unlabeled_channels: List[str],
        output_dir: str,
        threshold: float = 0.5,
        batch_size: int = 16,
        min_duration: float = 0.3
):
    """
    Apply trained model to predict on multiple unlabeled channels
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model, cfg, train_channel = load_trained_model(checkpoint_path, device)

    # Load data
    print(f"\nLoading data from: {data_path}")
    if data_path.endswith('.npy'):
        X_full = np.load(data_path, mmap_mode='r')
    else:
        raise ValueError("Currently only supports .npy files")

    print(f"Data shape: {X_full.shape}")

    # Get all available channels
    all_channels = cfg["data"].get("channels", Crnn.ALL_EEG_19)
    print(f"Available channels: {', '.join(all_channels)}")

    # Verify unlabeled channels exist
    for ch in unlabeled_channels:
        if ch not in all_channels:
            raise ValueError(f"Channel {ch} not in available channels")

    os.makedirs(output_dir, exist_ok=True)

    # Predict for each unlabeled channel
    all_results = {}

    print(f"\n{'=' * 60}")
    print(f"Predicting spindles on {len(unlabeled_channels)} channels")
    print(f"Threshold: {threshold}")
    print(f"Min duration: {min_duration}s")
    print(f"{'=' * 60}\n")

    for channel_name in unlabeled_channels:
        print(f"Processing channel: {channel_name}")

        channel_idx = get_channel_index(channel_name, all_channels)

        # Get predictions
        probs, binary_preds = predict_channel(
            model, X_full, channel_idx, device, batch_size, threshold
        )

        # Extract events (need window parameters from config)
        events = extract_events(
            binary_preds,
            cfg["data"]["sfreq"],
            cfg["data"]["window_sec"],
            cfg["data"]["step_sec"],
            min_duration
        )

        # Calculate statistics
        total_duration = X_full.shape[0] * X_full.shape[-1] / cfg["data"]["sfreq"]
        spindle_duration = sum(e["duration"] for e in events)
        spindle_rate = len(events) / (total_duration / 60.0)  # per minute
        coverage = spindle_duration / total_duration * 100

        print(f"  Detected: {len(events)} spindles")
        print(f"  Rate: {spindle_rate:.2f} spindles/min")
        print(f"  Coverage: {coverage:.2f}%")
        print(f"  Total duration: {spindle_duration:.1f}s")

        # Save results
        channel_results = {
            "channel": channel_name,
            "train_channel": train_channel,
            "threshold": threshold,
            "min_duration": min_duration,
            "n_spindles": len(events),
            "spindle_rate_per_min": float(spindle_rate),
            "coverage_percent": float(coverage),
            "total_spindle_duration_sec": float(spindle_duration),
            "total_recording_duration_sec": float(total_duration),
            "events": events
        }

        all_results[channel_name] = channel_results

        # Save individual channel results
        channel_file = os.path.join(output_dir, f"predictions_{channel_name}.json")
        with open(channel_file, 'w') as f:
            json.dump(channel_results, f, indent=2)
        print(f"  Saved to: {channel_file}")

        # Save probabilities as NPY
        probs_file = os.path.join(output_dir, f"probabilities_{channel_name}.npy")
        np.save(probs_file, probs)

        # Save binary predictions as NPY
        preds_file = os.path.join(output_dir, f"predictions_{channel_name}.npy")
        np.save(preds_file, binary_preds)

        print()

    # Save combined summary
    summary_file = os.path.join(output_dir, "prediction_summary.json")
    summary = {
        "model_checkpoint": checkpoint_path,
        "train_channel": train_channel,
        "threshold": threshold,
        "min_duration": min_duration,
        "channels": all_results
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"{'=' * 60}")
    print(f"Summary saved to: {summary_file}")
    print(f"{'=' * 60}\n")

    # Print summary table
    print("PREDICTION SUMMARY:")
    print(f"{'Channel':<10} {'N Spindles':<12} {'Rate/min':<10} {'Coverage %':<12}")
    print("-" * 50)
    for ch_name, results in all_results.items():
        print(f"{ch_name:<10} {results['n_spindles']:<12} "
              f"{results['spindle_rate_per_min']:<10.2f} "
              f"{results['coverage_percent']:<12.2f}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict spindles on unlabeled channels using trained model"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint (.pt file)")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data file (.npy)")
    parser.add_argument("--channels", type=str, nargs="+", required=True,
                        help="List of unlabeled channels to predict on")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for predictions")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Prediction threshold (default: 0.5)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for prediction (default: 16)")
    parser.add_argument("--min-duration", type=float, default=0.3,
                        help="Minimum spindle duration in seconds (default: 0.3)")

    args = parser.parse_args()

    predict_all_unlabeled_channels(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        unlabeled_channels=args.channels,
        output_dir=args.output_dir,
        threshold=args.threshold,
        batch_size=args.batch_size,
        min_duration=args.min_duration
    )