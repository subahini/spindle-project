"""
Configuration settings for EEG spindle detection
"""
from pathlib import Path
import torch


class Config:
    """Configuration class for all settings"""

    # Data paths
    EDF_PATH = Path("data/raw/P002_1_raw.edf")
    JSON_PATH = Path("data/labels/sleep_block_spindle_output_P002_1.json")
    SAVE_DIR = Path("data/windows/")
    MODEL_DIR = Path("models/")
    RESULTS_DIR = Path("results/")

    # Create directories
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # EEG channels to use
    EEG_CHANNELS = ['C3', 'C4', 'O1', 'O2', 'F3', 'F4', 'P3', 'P4',
                    'Fp1', 'Fp2', 'T3', 'T4', 'T5', 'T6', 'F7', 'F8']

    # Signal processing parameters
    FILTER_LOW = 5.0  # High-pass filter to remove DC drift    #  based on the observation this 5 to 30 looks good
    FILTER_HIGH = 30.0  # Low-pass filter
    WINDOW_SEC = 2.0  # Window length in seconds
    STEP_SEC = 1.0  # Step size for overlapping windows
    OVERLAP_THRESHOLD = 0.7  # Minimum overlap to consider window as containing spindle

    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 5e-5  # Very low learning rate for stability
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10
    DOWNSAMPLE = True  # enable downsampling

    # Data splits (in seconds)
    TRAIN_START = 0
    TRAIN_END = 3 * 3600  # 0-3 hours
    TEST_START = 3 * 3600
    TEST_END = 6 * 3600  # 3-6 hours
    VAL_START = 6 * 3600
    VAL_END = 8 * 3600  # 6-8 hours

    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")