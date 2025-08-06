"""
Enhanced EEG Spindle Detection Pipeline using 2D CNN
Fixes: Missing imports, better architecture, proper evaluation, validation tracking
"""

import mne
import numpy as np
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import time


# === ENHANCED CONFIG ===
class Config:
    # Data paths
    EDF_PATH = Path("data/raw/P002_1_raw.edf")
    JSON_PATH = Path("data/labels/sleep_block_spindle_output_P002_1.json")
    SAVE_DIR = Path("data/windows/")
    MODEL_DIR = Path("models/")

    # Create directories
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # EEG channels
    EEG_CHANNELS = ['C3', 'C4', 'O1', 'O2', 'F3', 'F4', 'P3', 'P4',
                    'Fp1', 'Fp2', 'T3', 'T4', 'T5', 'T6', 'F7', 'F8']

    # Signal processing
    FILTER_LOW = 0.5  # High-pass to remove DC drift
    FILTER_HIGH = 30.0  # Low-pass to include broader frequency content
    WINDOW_SEC = 2.0
    STEP_SEC = 1.0
    OVERLAP_THRESHOLD = 0.5

    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    PATIENCE = 10  # Early stopping

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config = Config()


# === EVALUATION METRICS CLASS ===
class EvaluationMetrics:
    """Comprehensive evaluation metrics for spindle detection"""

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate comprehensive metrics"""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'confusion_matrix': cm,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }

    @staticmethod
    def print_detailed_report(metrics: Dict, dataset_name: str = "Dataset"):
        """Print detailed evaluation report"""
        print(f"\n=== {dataset_name} Evaluation Report ===")
        print(f"Accuracy:    {metrics['accuracy']:.4f}")
        print(f"Precision:   {metrics['precision']:.4f}")
        print(f"Recall:      {metrics['recall']:.4f}")
        print(f"F1-Score:    {metrics['f1_score']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")

        print("\nConfusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"[[{cm[0, 0]:4d}, {cm[0, 1]:4d}]")
        print(f" [{cm[1, 0]:4d}, {cm[1, 1]:4d}]]")
        print("  TN   FP")
        print("  FN   TP")

    @staticmethod
    def plot_confusion_matrix(metrics: Dict, title: str = "Confusion Matrix"):
        """Plot confusion matrix"""
        cm = metrics['confusion_matrix']
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Spindle', 'Spindle'],
                    yticklabels=['No Spindle', 'Spindle'])
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()


# === EARLY STOPPING CLASS ===
class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience: int = 15, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score: float) -> bool:
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0

        return self.early_stop


# === ENHANCED MODEL ARCHITECTURE ===
class EnhancedSpindle2DCNN(nn.Module):
    """Improved 2D CNN for EEG spindle detection"""

    def __init__(self, dropout_rate: float = 0.3):
        super().__init__()

        # Feature extraction blocks
        self.conv_block1 = self._make_conv_block(1, 16, (3, 7))
        self.conv_block2 = self._make_conv_block(16, 32, (3, 5))
        self.conv_block3 = self._make_conv_block(32, 64, (3, 3))

        # Global average pooling instead of multiple adaptive pools
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )

        # Initialize weights
        self._initialize_weights()

    def _make_conv_block(self, in_channels: int, out_channels: int, kernel_size: tuple) -> nn.Sequential:
        """Create a convolutional block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)),
            nn.Dropout2d(0.2)
        )

    def _initialize_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        # Global pooling
        x = self.global_pool(x)

        # Classification
        x = self.classifier(x)
        return x


# === ENHANCED EVALUATION FUNCTION ===
def evaluate_model(model: nn.Module, data_loader, device: torch.device, threshold: float = 0.3) -> Dict:
    """Comprehensive model evaluation"""
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_targets = []
    total_loss = 0.0

    # Use the same criterion as training for loss calculation
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)

            outputs = model(data)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Convert to probabilities and predictions
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > threshold).float()

            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Calculate metrics
    metrics = EvaluationMetrics.calculate_metrics(
        np.array(all_targets).flatten(),
        np.array(all_predictions).flatten()
    )
    metrics['avg_loss'] = total_loss / len(data_loader)

    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(
        np.array(all_targets).flatten(),
        np.array(all_probabilities).flatten()
    )
    metrics['optimal_threshold'] = optimal_threshold

    return metrics


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Find optimal threshold that maximizes F1 score"""
    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_proba > threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold


# === DATA LOADING AND PREPROCESSING ===
def load_and_preprocess_data():
    """Load and preprocess EEG data"""
    print("Loading EDF...")
    if not config.EDF_PATH.exists():
        raise FileNotFoundError(f"EDF file not found: {config.EDF_PATH}")

    raw = mne.io.read_raw_edf(config.EDF_PATH, preload=True, verbose=False)
    sfreq = raw.info["sfreq"]

    # Select available EEG channels
    available_channels = [ch for ch in config.EEG_CHANNELS if ch in raw.ch_names]
    if not available_channels:
        raise ValueError("No EEG channels found in the data")

    print(f"Using {len(available_channels)} EEG channels: {available_channels}")
    raw.pick_channels(available_channels)

    # Apply bandpass filter
    print(f"Applying bandpass filter: {config.FILTER_LOW}-{config.FILTER_HIGH} Hz")
    raw.filter(config.FILTER_LOW, config.FILTER_HIGH, fir_design='firwin', verbose=False)

    return raw, sfreq


def load_spindle_labels():
    """Load spindle annotations"""
    print("Loading spindle labels...")
    if not config.JSON_PATH.exists():
        raise FileNotFoundError(f"JSON file not found: {config.JSON_PATH}")

    with open(config.JSON_PATH) as f:
        spindle_data = json.load(f)

    spindles = [(s["start"], s["end"]) for s in spindle_data["detected_spindles"]]
    print(f"Loaded {len(spindles)} spindle annotations")
    return spindles


def is_spindle_window(start: float, end: float, spindles: List[Tuple],
                      overlap_threshold: float = 0.5) -> bool:
    """Check if window contains spindle with sufficient overlap"""
    window_duration = end - start
    for s_start, s_end in spindles:
        overlap = max(0, min(end, s_end) - max(start, s_start))
        if overlap / window_duration >= overlap_threshold:
            return True
    return False


def create_windows(raw, sfreq: float, spindles: List[Tuple],
                   start_sec: float, end_sec: float, prefix: str):
    """Create overlapping windows with labels"""
    print(f"Creating windows for {prefix} split: {start_sec / 3600:.1f}h - {end_sec / 3600:.1f}h")

    X, y = [], []
    win_samples = int(config.WINDOW_SEC * sfreq)
    step_samples = int(config.STEP_SEC * sfreq)
    start_sample = int(start_sec * sfreq)
    end_sample = int(end_sec * sfreq)

    # Ensure we don't exceed data length
    max_start = min(end_sample - win_samples, raw.n_times - win_samples)

    for s in range(start_sample, max_start, step_samples):
        try:
            segment = raw.get_data(start=s, stop=s + win_samples)
            t0 = s / sfreq
            t1 = (s + win_samples) / sfreq

            label = 1 if is_spindle_window(t0, t1, spindles, config.OVERLAP_THRESHOLD) else 0
            X.append(segment)
            y.append(label)

        except Exception as e:
            print(f"Warning: Skipping window at {s}: {e}")
            continue

    if not X:
        raise ValueError(f"No valid windows created for {prefix}")

    X = np.array(X)[:, np.newaxis, :, :]  # Add channel dimension for CNN
    y = np.array(y)

    # Save windows
    np.save(config.SAVE_DIR / f"X_{prefix}.npy", X)
    np.save(config.SAVE_DIR / f"y_{prefix}.npy", y)

    pos_count = np.sum(y)
    neg_count = len(y) - pos_count
    print(f"Saved {prefix}: {X.shape}, Positive: {pos_count}, Negative: {neg_count}")

    return X, y


# === TRAINING FUNCTION ===
def train_model(model: nn.Module, train_loader, val_loader, device: torch.device):
    """Enhanced training loop with validation tracking"""
    print("\nStarting training...")

    # Calculate class weights for imbalanced data
    train_labels = []
    for _, labels in train_loader:
        train_labels.extend(labels.numpy().flatten())

    train_labels = np.array(train_labels)
    n_pos = int(np.sum(train_labels))
    n_neg = int(len(train_labels) - n_pos)

    # Reduce the class weight to prevent over-correction
    pos_weight = torch.tensor([min(n_neg / max(n_pos, 1), 5.0)], dtype=torch.float32).to(device)

    print(f"Class distribution - Positive: {n_pos}, Negative: {n_neg}")
    print(f"Positive class weight: {pos_weight.item():.3f}")

    # Setup training components
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=15)

    # Training history
    train_losses = []
    val_losses = []
    val_metrics_history = []
    best_val_f1 = 0.0
    best_model_state = None

    start_time = time.time()

    for epoch in range(config.EPOCHS):
        epoch_start = time.time()

        # Training phase
        model.train()
        total_train_loss = 0.0
        num_batches = 0

        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_train_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_train_loss / num_batches
        train_losses.append(avg_train_loss)

        # Validation phase
        val_metrics = evaluate_model(model, val_loader, device, threshold=0.3)
        val_losses.append(val_metrics['avg_loss'])
        val_metrics_history.append(val_metrics)

        # Learning rate scheduling
        scheduler.step(val_metrics['f1_score'])

        # Save best model
        if val_metrics['f1_score'] > best_val_f1:
            best_val_f1 = val_metrics['f1_score']
            best_model_state = model.state_dict().copy()

        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch + 1}/{config.EPOCHS} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {val_metrics['avg_loss']:.4f}")
        print(
            f"  Val F1: {val_metrics['f1_score']:.4f}, Val Precision: {val_metrics['precision']:.4f}, Val Recall: {val_metrics['recall']:.4f}")
        print(
            f"  Val Specificity: {val_metrics['specificity']:.4f}, Optimal Threshold: {val_metrics['optimal_threshold']:.3f}")

        # Early stopping
        if early_stopping(val_metrics['f1_score']):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best model with validation F1: {best_val_f1:.4f}")

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f} seconds")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_metrics': val_metrics_history,
        'best_val_f1': best_val_f1,
        'total_time': total_time
    }


# === MAIN EXECUTION ===
def main():
    """Main pipeline execution"""
    try:
        print(f"Using device: {config.DEVICE}")

        # Load and preprocess data
        raw, sfreq = load_and_preprocess_data()
        spindles = load_spindle_labels()

        # Create data splits
        print("\nCreating data splits...")
        create_windows(raw, sfreq, spindles, 0, 3 * 3600, "train")
        create_windows(raw, sfreq, spindles, 3 * 3600, 6 * 3600, "test")
        create_windows(raw, sfreq, spindles, 6 * 3600, 8 * 3600, "val")

        # Load training data
        print("\nLoading training data...")
        X_train = np.load(config.SAVE_DIR / "X_train.npy")
        y_train = np.load(config.SAVE_DIR / "y_train.npy")
        X_val = np.load(config.SAVE_DIR / "X_val.npy")
        y_val = np.load(config.SAVE_DIR / "y_val.npy")

        # Convert to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                                                   shuffle=True, num_workers=2)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                                                 shuffle=False, num_workers=2)

        print(f"Training data: {X_train.shape}")
        print(f"Validation data: {X_val.shape}")

        # Create and train model
        model = EnhancedSpindle2DCNN().to(config.DEVICE)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Train model
        training_results = train_model(model, train_loader, val_loader, config.DEVICE)

        # Save model
        model_path = config.MODEL_DIR / "enhanced_spindle_cnn2d.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config.__dict__,
            'training_results': training_results
        }, model_path)
        print(f"\nModel saved to {model_path}")

        # Final test evaluation
        print("\nFinal test evaluation...")
        X_test = np.load(config.SAVE_DIR / "X_test.npy")
        y_test = np.load(config.SAVE_DIR / "y_test.npy")
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

        test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                                                  shuffle=False, num_workers=2)

        # Evaluate with multiple thresholds
        print("\n=== Test Evaluation with Default Threshold (0.3) ===")
        test_metrics = evaluate_model(model, test_loader, config.DEVICE, threshold=0.3)
        EvaluationMetrics.print_detailed_report(test_metrics, "Test Set")

        print(f"\n=== Test Evaluation with Optimal Threshold ({test_metrics['optimal_threshold']:.3f}) ===")
        test_metrics_optimal = evaluate_model(model, test_loader, config.DEVICE,
                                              threshold=test_metrics['optimal_threshold'])
        EvaluationMetrics.print_detailed_report(test_metrics_optimal, "Test Set (Optimal Threshold)")

        EvaluationMetrics.plot_confusion_matrix(test_metrics, "Test Set Confusion Matrix")

        print("\nPipeline completed successfully!")

    except Exception as e:
        print(f"Error in pipeline: {e}")
        raise


if __name__ == "__main__":
    main()