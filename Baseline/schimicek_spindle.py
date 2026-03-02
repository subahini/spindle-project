from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
from scipy.signal import butter, filtfilt


@dataclass
class SchimicekParams:
    """Parameters for Schimicek spindle detection"""
    # Sampling rate (Hz)
    fs: float = 200.0

    # Frequency bands (Hz)
    spindle_band: Tuple[float, float] = (11.5, 16.0)
    alpha_band: Tuple[float, float] = (5.0, 12.0)
    muscle_band: Tuple[float, float] = (30.0, 40.0)

    # Detection criteria (EEG in microvolts)
    amplitude_threshold_uv: float = 25.0  # min peak-to-peak amplitude (µV)
    min_duration_s: float = 0.5  # min duration (s)

    # Epoch length for artifact detection (seconds)
    epoch_length_s: float = 5.0

    # Artifact thresholds
    alpha_ratio_threshold: float = 1.2  # RMS_alpha / RMS_spindle > 1.2 → alpha artifact
    muscle_rms_threshold_uv: float = 5.0  # RMS_muscle > 5 µV → muscle artifact

    # Filter order
    filter_order: int = 2


# =========================
# Butterworth Bandpass Filter
# =========================

def _butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 2):

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a


def _apply_bandpass(signal: np.ndarray, lowcut: float, highcut: float,
                    fs: float, order: int = 2) -> np.ndarray:

    b, a = _butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, signal)


# =========================
# RMS Calculation
# =========================

def _calculate_rms(signal: np.ndarray) -> float:

    if signal.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(signal ** 2)))


# =========================
# Peak-to-Peak Amplitude
# =========================

def _calculate_peak_to_peak_amplitude(signal: np.ndarray, window_samples: int) -> np.ndarray:
    """

    Peak-to-peak = max - min within window

    """
    n = len(signal)
    p2p = np.zeros(n)

    half_window = window_samples // 2

    for i in range(n):
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        window = signal[start:end]
        p2p[i] = np.max(window) - np.min(window)

    return p2p #This creates a line of True/False


# =========================
# STEP 1-3: Candidate Spindle Detection
# =========================

def _detect_candidate_spindles(eeg_uv: np.ndarray, params: SchimicekParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect candidate spindles and return both mask and peak-to-peak amplitude

    this will Returns:
        candidate_mask: Boolean mask of candidate spindles
        peak_to_peak: Array of peak-to-peak amplitudes
    """
    # STEP 1: Bandpass filter (11.5-16 Hz)
    spindle_filtered = _apply_bandpass(
        eeg_uv,
        params.spindle_band[0],
        params.spindle_band[1],
        params.fs,
        params.filter_order
    )

    # STEP 2: Calculate peak-to-peak amplitude
    # Use a window of ~0.5 seconds for peak-to-peak calculation
    window_samples = int(0.5 * params.fs)
    peak_to_peak = _calculate_peak_to_peak_amplitude(spindle_filtered, window_samples)

    # Check amplitude threshold
    above_threshold = peak_to_peak >= params.amplitude_threshold_uv

    # STEP 3: Check duration criterion
    min_samples = int(params.min_duration_s * params.fs)
    candidate_mask = np.zeros_like(above_threshold, dtype=bool)

    # Find contiguous regions above threshold
    in_spindle = False
    start_idx = 0

    for i in range(len(above_threshold)):
        if above_threshold[i] and not in_spindle:
            # Start of potential spindle
            start_idx = i
            in_spindle = True
        elif not above_threshold[i] and in_spindle:
            # End of potential spindle
            duration = i - start_idx
            if duration >= min_samples:
                candidate_mask[start_idx:i] = True
            in_spindle = False

    # Handle case where spindle extends to end of signal
    if in_spindle:
        duration = len(above_threshold) - start_idx
        if duration >= min_samples:
            candidate_mask[start_idx:] = True

    return candidate_mask, peak_to_peak


# =========================
# Artifact Detection per 5-second Epoch (with boundary handling)
# =========================

def _detect_artifacts_per_epoch(eeg_uv: np.ndarray,
                                candidate_mask: np.ndarray,
                                params: SchimicekParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    this should Returns:
        final_mask: Boolean array with artifacts removed
        epoch_labels: Array of labels for each epoch
    """
    n_samples = len(eeg_uv)
    epoch_samples = int(params.epoch_length_s * params.fs)
    n_epochs = n_samples // epoch_samples

    # Apply three bandpass filters once
    spindle_filtered = _apply_bandpass(
        eeg_uv,
        params.spindle_band[0],
        params.spindle_band[1],
        params.fs,
        params.filter_order
    )

    alpha_filtered = _apply_bandpass(
        eeg_uv,
        params.alpha_band[0],
        params.alpha_band[1],
        params.fs,
        params.filter_order
    )

    muscle_filtered = _apply_bandpass(
        eeg_uv,
        params.muscle_band[0],
        params.muscle_band[1],
        params.fs,
        params.filter_order
    )

    # Initialize outputs
    artifact_flags = np.zeros(n_samples, dtype=bool)  # True = artifact
    epoch_labels = np.empty(n_epochs, dtype=object)
    epoch_labels[:] = 'clean_or_no_spindle'

    # Process each 5-second epoch
    for epoch_idx in range(n_epochs):
        start = epoch_idx * epoch_samples
        end = start + epoch_samples

        # Extract epoch data
        spindle_epoch = spindle_filtered[start:end]
        alpha_epoch = alpha_filtered[start:end]
        muscle_epoch = muscle_filtered[start:end]
        candidate_epoch = candidate_mask[start:end]

        # Calculate RMS for each band
        rms_spindle = _calculate_rms(spindle_epoch)
        rms_alpha = _calculate_rms(alpha_epoch)
        rms_muscle = _calculate_rms(muscle_epoch)

        artifact_detected = False

        # Check for alpha artifact
        if rms_spindle > 0:
            alpha_ratio = rms_alpha / rms_spindle
            if alpha_ratio > params.alpha_ratio_threshold:
                artifact_detected = True
                epoch_labels[epoch_idx] = 'alpha_artifact'

        # Check for muscle artifact
        if rms_muscle > params.muscle_rms_threshold_uv:
            artifact_detected = True
            epoch_labels[epoch_idx] = 'muscle_artifact'

        # Mark samples in this epoch as artifact
        if artifact_detected:
            artifact_flags[start:end] = True
        elif candidate_epoch.any():
            epoch_labels[epoch_idx] = 'spindle'

    # Create final mask: candidate spindles with artifacts removed
    final_mask = candidate_mask & ~artifact_flags

    return final_mask, epoch_labels


# =========================
# Convert Mask to Events (with statistics)
# =========================

def _mask_to_events(mask: np.ndarray, fs: float) -> List[Tuple[float, float, float]]:
    """
    this will Convert boolean mask to list of (start_time, end_time, duration_in_seconds)

    Returns:
        events: List of (start_time_sec, end_time_sec, duration_sec) tuples
    """
    events = []

    in_event = False
    start_idx = 0

    for i in range(len(mask)):
        if mask[i] and not in_event:
            start_idx = i
            in_event = True
        elif not mask[i] and in_event:
            start_sec = start_idx / fs
            end_sec = i / fs
            duration = end_sec - start_sec
            events.append((start_sec, end_sec, duration))
            in_event = False

    # Handle case where event extends to end
    if in_event:
        start_sec = start_idx / fs
        end_sec = len(mask) / fs
        duration = end_sec - start_sec
        events.append((start_sec, end_sec, duration))

    return events


# =========================
# Main Detection Function
# =========================

def detect_spindles_schimicek(eeg_uv: np.ndarray,
                              params: SchimicekParams = None) -> Dict:
    """

    Exact implementation following the paper's methodology:
    1. Bandpass filter (11.5-16 Hz)
    2. Peak-to-peak amplitude check (≥ 25 µV)
    3. Duration check (≥ 0.5 s)
    4. Artifact rejection per 5-second epoch:
       - Alpha artifact: RMS_alpha / RMS_spindle > 1.2
       - Muscle artifact: RMS_muscle > 5 µV  """



    if params is None:
        params = SchimicekParams()

    eeg_uv = np.asarray(eeg_uv, dtype=float)

    # STEP 1-3: Detect candidate spindles (now returns p2p too)
    candidate_mask, peak_to_peak = _detect_candidate_spindles(eeg_uv, params) # candidate mask and peak to peak

    # STEP 4: Artifact detection per 5-second epoch
    final_mask, epoch_labels = _detect_artifacts_per_epoch(
        eeg_uv, candidate_mask, params
    )  # this is for 5 sec epoch .....final mask

    # Convert mask to events
    events = _mask_to_events(final_mask, params.fs)

    # Extract event-level statistics
    spindle_durations = np.array([ev[2] for ev in events]) if events else np.array([])

    spindle_amplitudes = []
    for start_sec, end_sec, _ in events:
        start_idx = int(round(start_sec * params.fs))
        end_idx = int(round(end_sec * params.fs))
        mean_amp = np.mean(peak_to_peak[start_idx:end_idx])
        spindle_amplitudes.append(mean_amp)
    spindle_amplitudes = np.array(spindle_amplitudes) if spindle_amplitudes else np.array([])

    return {
        'spindle_mask': final_mask,
        'candidate_mask': candidate_mask,
        'peak_to_peak': peak_to_peak,
        'events': events,
        'epoch_labels': epoch_labels,
        'n_spindles': len(events),
        'spindle_durations': spindle_durations,
        'spindle_amplitudes': spindle_amplitudes,
    }

