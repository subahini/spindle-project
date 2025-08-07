import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

from sympy.abc import alpha

from metrics import CustomMetrics
from models import FocalLoss
from config import Config
import torch

class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
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


def calculate_class_weights(train_loader, device):
    """Calculate class weights for imbalanced data"""
    train_labels = []
    for _, labels in train_loader:
        train_labels.extend(labels.numpy().flatten())

    train_labels = np.array(train_labels)
    n_pos = int(np.sum(train_labels))
    n_neg = int(len(train_labels) - n_pos)

    # Cap weight to prevent over-correction
    pos_weight_value = min(n_neg / max(n_pos, 1), 1)

    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)

    print(f"Class distribution - Positive: {n_pos}, Negative: {n_neg}")
    print(f"Positive class weight: {pos_weight_value:.3f}")

    return pos_weight


def train_model(model, train_loader, val_loader, device, model_name="CNN"):
    """Train CNN model with comprehensive evaluation"""
    config = Config()
    print(f"\nStarting {model_name} training...")

    # Calculate class weights
    pos_weight = calculate_class_weights(train_loader, device)

    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    print(f"loss focal loss with alpha{alpha}")
   # criterion = nn.CrossEntropyLoss()
    #criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    #print(f" BCE loss entropy loss {pos_weight}")

    # Optimizer with very low learning rate
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=config.PATIENCE)

    # Initialize metrics calculator
    metrics_calc = CustomMetrics()

    # Training history
    train_losses = []
    val_metrics_history = []
    best_val_f1 = 0.0
    best_model_state = None

    start_time = time.time()

    for epoch in range(config.EPOCHS):
        epoch_start = time.time()

        # === TRAINING PHASE ===
        model.train()
        total_train_loss = 0.0
        num_batches = 0

        for data, targets in train_loader:
            data = data.to(device)
            
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_train_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_train_loss / num_batches
        train_losses.append(avg_train_loss)

        # === VALIDATION PHASE ===
        val_metrics = metrics_calc.evaluate_model_with_threshold(model, val_loader, device, threshold=0.2)
        val_metrics_history.append(val_metrics)

        # Learning rate scheduling
        if val_metrics is not None and 'f1_score' in val_metrics:
            scheduler.step(val_metrics['f1_score'])
        else:
            print("Warning: Validation metrics missing or incomplete; skipping scheduler step.")

        # Save best model
        if val_metrics and val_metrics.get('f1_score', 0) > best_val_f1:
            best_val_f1 = val_metrics['f1_score']
            best_model_state = model.state_dict().copy()

        epoch_time = time.time() - epoch_start

        # Print epoch results
        print(f"Epoch {epoch + 1}/{config.EPOCHS} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {avg_train_loss:.4f}")

        if val_metrics:
            print(f"  Val F1: {val_metrics.get('f1_score', 0):.4f}, Precision: {val_metrics.get('precision', 0):.4f}, Recall: {val_metrics.get('recall', 0):.4f}")
            print(f"  Val Specificity: {val_metrics.get('specificity', 0):.4f}, Accuracy: {val_metrics.get('accuracy', 0):.4f}")

            if 'auc_roc' in val_metrics:
                print(f"  Val AUC-ROC: {val_metrics['auc_roc']:.4f}, AUC-PR: {val_metrics['auc_pr']:.4f}")

        # Early stopping
        if val_metrics and early_stopping(val_metrics.get('f1_score', 0)):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
    return {
        'train_losses': train_losses,
        'val_metrics_history': val_metrics_history,
        'best_val_f1': best_val_f1,
        'best_model_state': best_model_state,
        'total_time': time.time() - start_time    #time error
    }
