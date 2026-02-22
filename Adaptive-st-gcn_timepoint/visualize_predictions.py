"""
visualize_predictions.py

Visualize time-point predictions vs ground truth.
Helps debug and understand what the model is learning.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


def plot_sample_predictions(
    model,
    x_de,
    x_raw,
    y_true,
    sample_idx=0,
    channel_to_plot=0,
    save_path=None
):
    """
    Visualize predictions for a single sample.
    
    Creates 3-panel plot:
    1. Raw EEG with ground truth overlay
    2. Coarse prediction (window-level)
    3. Fine prediction (sample-level)
    
    Parameters:
    -----------
    model : keras.Model
        Trained multi-resolution model
    x_de : np.ndarray
        DE features (N, context, channels, bands)
    x_raw : np.ndarray
        Raw EEG (N, samples_per_window, channels)
    y_true : np.ndarray
        Ground truth labels (N, samples_per_window)
    sample_idx : int
        Which sample to visualize
    channel_to_plot : int
        Which EEG channel to show
    save_path : str, optional
        Where to save figure
    """
    
    # Get predictions
    preds = model.predict({
        'DE_Input': x_de[sample_idx:sample_idx+1],
        'Raw_EEG_Input': x_raw[sample_idx:sample_idx+1]
    }, verbose=0)
    
    coarse_pred = preds['coarse'][0]  # (context, 1)
    fine_pred = preds['fine'][0]      # (samples_per_window, 1)
    
    raw_signal = x_raw[sample_idx, :, channel_to_plot]
    true_label = y_true[sample_idx]
    
    samples_per_window = len(raw_signal)
    time_axis = np.arange(samples_per_window) / 200  # Assume 200 Hz
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # Panel 1: Raw EEG with ground truth
    ax = axes[0]
    ax.plot(time_axis, raw_signal, 'k-', linewidth=0.5, label='Raw EEG')
    ax.fill_between(
        time_axis,
        raw_signal.min(),
        raw_signal.max(),
        where=true_label.astype(bool),
        alpha=0.3,
        color='green',
        label='True Spindle'
    )
    ax.set_ylabel('Amplitude (μV)')
    ax.set_title(f'Sample {sample_idx} - Channel {channel_to_plot}')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Coarse predictions
    ax = axes[1]
    context = len(coarse_pred)
    # Each context window corresponds to 2 seconds
    window_times = np.linspace(0, 2, context)
    window_widths = 2.0 / context
    
    for i, (t, p) in enumerate(zip(window_times, coarse_pred[:, 0])):
        color = plt.cm.RdYlGn(p)  # Red=0, Yellow=0.5, Green=1
        ax.barh(
            y=0,
            width=window_widths,
            left=t,
            height=1,
            color=color,
            edgecolor='black',
            linewidth=0.5
        )
    
    # Highlight center window (the one being refined)
    center_idx = context // 2
    center_time = window_times[center_idx]
    ax.axvline(center_time, color='blue', linestyle='--', linewidth=2, label='Center')
    ax.axvline(center_time + window_widths, color='blue', linestyle='--', linewidth=2)
    
    ax.set_ylabel('Coarse Prob')
    ax.set_ylim(-0.1, 1.1)
    ax.set_title('Window-Level Predictions (from DE features)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Fine predictions
    ax = axes[2]
    ax.plot(time_axis, fine_pred[:, 0], 'b-', linewidth=1, label='Fine Prediction')
    ax.fill_between(
        time_axis,
        0,
        1,
        where=true_label.astype(bool),
        alpha=0.3,
        color='green',
        label='True Spindle'
    )
    ax.axhline(0.5, color='gray', linestyle=':', label='Threshold 0.5')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Fine Prob')
    ax.set_ylim(-0.1, 1.1)
    ax.set_title('Sample-Level Predictions (from Raw EEG)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    
    return fig


def plot_comparison_grid(
    model,
    x_de,
    x_raw,
    y_true,
    n_samples=6,
    channel=0,
    save_path=None
):
    """
    Create grid comparing multiple samples.
    Shows raw EEG, ground truth, and predictions side-by-side.
    """
    
    n_cols = 3
    n_rows = n_samples
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 2.5*n_rows))
    
    for i in range(n_samples):
        # Get predictions
        preds = model.predict({
            'DE_Input': x_de[i:i+1],
            'Raw_EEG_Input': x_raw[i:i+1]
        }, verbose=0)
        
        fine_pred = preds['fine'][0, :, 0]
        
        raw = x_raw[i, :, channel]
        true = y_true[i]
        time = np.arange(len(raw)) / 200
        
        # Column 1: Raw EEG
        ax = axes[i, 0]
        ax.plot(time, raw, 'k-', linewidth=0.5)
        ax.set_ylabel(f'Sample {i}')
        if i == 0:
            ax.set_title('Raw EEG')
        if i == n_samples - 1:
            ax.set_xlabel('Time (s)')
        
        # Column 2: Ground Truth
        ax = axes[i, 1]
        ax.fill_between(time, 0, 1, where=true.astype(bool), color='green', alpha=0.6)
        ax.set_ylim(-0.1, 1.1)
        if i == 0:
            ax.set_title('Ground Truth')
        if i == n_samples - 1:
            ax.set_xlabel('Time (s)')
        
        # Column 3: Prediction
        ax = axes[i, 2]
        ax.plot(time, fine_pred, 'b-', linewidth=1)
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        ax.set_ylim(-0.1, 1.1)
        if i == 0:
            ax.set_title('Model Prediction')
        if i == n_samples - 1:
            ax.set_xlabel('Time (s)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    
    return fig


def analyze_errors(
    model,
    x_de,
    x_raw,
    y_true,
    threshold=0.5,
    n_worst=5
):
    """
    Find and visualize worst predictions.
    Helps identify failure modes.
    """
    
    # Get all predictions
    preds = model.predict({
        'DE_Input': x_de,
        'Raw_EEG_Input': x_raw
    }, verbose=0)
    
    fine_pred = preds['fine'].squeeze()  # (N, samples)
    fine_bin = (fine_pred >= threshold).astype(int)
    
    # Compute per-sample F1
    from sklearn.metrics import f1_score
    
    f1_scores = []
    for i in range(len(y_true)):
        f1 = f1_score(
            y_true[i].reshape(-1),
            fine_bin[i].reshape(-1),
            zero_division=0
        )
        f1_scores.append(f1)
    
    f1_scores = np.array(f1_scores)
    
    # Find worst samples
    worst_indices = np.argsort(f1_scores)[:n_worst]
    
    print(f"\n{'='*60}")
    print(f"WORST {n_worst} PREDICTIONS")
    print('='*60)
    
    for rank, idx in enumerate(worst_indices):
        print(f"\n{rank+1}. Sample {idx}: F1 = {f1_scores[idx]:.3f}")
        
        # Compute confusion for this sample
        y_t = y_true[idx].reshape(-1)
        y_p = fine_bin[idx].reshape(-1)
        
        tp = ((y_t == 1) & (y_p == 1)).sum()
        fp = ((y_t == 0) & (y_p == 1)).sum()
        fn = ((y_t == 1) & (y_p == 0)).sum()
        tn = ((y_t == 0) & (y_p == 0)).sum()
        
        print(f"   TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        
        # Common failure modes
        if fp > fn:
            print("   → Over-predicting (false alarms)")
        elif fn > fp:
            print("   → Under-predicting (missed spindles)")
        
        if tp == 0 and fn > 0:
            print("   → Completely missed spindle")
        
        if tp == 0 and fp > 0:
            print("   → False detection (no spindle present)")
    
    return worst_indices, f1_scores


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    
    # Load data
    print("Loading data...")
    npz = np.load('./data/SS3_multireso_19channels.npz', allow_pickle=True)
    
    x_de = npz['Fold_Data_DE'][0]    # First fold
    x_raw = npz['Fold_Data_Raw'][0]
    y_true = npz['Fold_Label'][0]
    
    print(f"Data shapes:")
    print(f"  DE:    {x_de.shape}")
    print(f"  Raw:   {x_raw.shape}")
    print(f"  Label: {y_true.shape}")
    
    # Add context (if not already done)
    # ... (use functions from train_timepoint.py)
    
    # Load trained model
    print("\nLoading model...")
    # This is a placeholder - you need to load your actual trained model
    # model = keras.models.load_model('./result/best_fold0.h5')
    
    # For demo, we'll create a dummy model
    # Replace this with your actual model loading code
    
    print("\nTo use this script:")
    print("1. Train a model using train_timepoint.py")
    print("2. Load the trained weights:")
    print("   model = build_GraphSleepNet_TimePoint(...)")
    print("   model.load_weights('./result/best_fold0.weights.h5')")
    print("3. Run visualizations:")
    print("   plot_sample_predictions(model, x_de, x_raw, y_true, sample_idx=10)")
    print("   plot_comparison_grid(model, x_de, x_raw, y_true, n_samples=6)")
    print("   worst_idx, f1s = analyze_errors(model, x_de, x_raw, y_true)")
