
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
    average_precision_score,
    f1_score,
    balanced_accuracy_score
)

try:
    import wandb  # type: ignore
except Exception:
    wandb = None


@torch.no_grad()
def collect_predictions(model, loader, device, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
     this will Collect all predictions and targets from a dataloader.

    """
    model.eval()
    all_probs = []
    all_targets = []

    for batch_list in loader:
        # Stack batch (same pattern as in trainer)
        batch = {
            'x': torch.stack([s['x'] for s in batch_list]),
            'y_t': torch.stack([s['y_t'] for s in batch_list]),
        }

        x = batch['x'].to(device)
        y_t = batch['y_t'].to(device)

        out = model(x)
        probs = torch.sigmoid(out['logits_global'])  # (B, T)

        all_probs.append(probs.cpu().numpy())
        all_targets.append(y_t.cpu().numpy())

    # Flatten all batches
    y_probs = np.concatenate([p.flatten() for p in all_probs])
    y_true = np.concatenate([t.flatten() for t in all_targets])
    y_pred = (y_probs >= threshold).astype(np.int32)

    return y_true, y_probs, y_pred


def compute_metrics(y_true: np.ndarray, y_probs: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
     this Compute classification metrics.

    A
    """
    # Handle edge case: all one class
    if len(np.unique(y_true)) < 2:
        print("[WARN] Only one class present in y_true. Some metrics will be invalid.")
        return {
            'accuracy': float(np.mean(y_true == y_pred)),
            'roc_auc': np.nan,
            'pr_auc': np.nan,
            'f1': np.nan,
            'balanced_acc': np.nan,
        }

    metrics = {}

    # Basic metrics
    metrics['accuracy'] = float(np.mean(y_true == y_pred))
    metrics['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics['balanced_acc'] = float(balanced_accuracy_score(y_true, y_pred))

    # ROC-AUC
    try:
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_probs))
    except Exception as e:
        print(f"[WARN] Could not compute ROC-AUC: {e}")
        metrics['roc_auc'] = np.nan

    # PR-AUC (Average Precision)
    try:
        metrics['pr_auc'] = float(average_precision_score(y_true, y_probs))
    except Exception as e:
        print(f"[WARN] Could not compute PR-AUC: {e}")
        metrics['pr_auc'] = np.nan

    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True) -> plt.Figure:
    """
    Create confusion matrix plot.

    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    # Labels
    classes = ['Negative', 'Positive']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data and create text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    title = 'Normalized Confusion Matrix' if normalize else 'Confusion Matrix'
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_roc_curve(y_true: np.ndarray, y_probs: np.ndarray, roc_auc: float) -> plt.Figure:
    """
    Create ROC curve plot.


    """
    fpr, tpr, _ = roc_curve(y_true, y_probs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC)')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_pr_curve(y_true: np.ndarray, y_probs: np.ndarray, pr_auc: float) -> plt.Figure:
    """
    Create Precision-Recall curve plot.

    A
    """
    precision, recall, _ = precision_recall_curve(y_true, y_probs)

    # Baseline (random classifier)
    baseline = float(y_true.sum()) / len(y_true)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AP = {pr_auc:.3f})')
    ax.axhline(y=baseline, color='navy', lw=2, linestyle='--', label=f'Baseline ({baseline:.3f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def log_metrics_to_wandb(
        y_true: np.ndarray,
        y_probs: np.ndarray,
        y_pred: np.ndarray,
        step: Optional[int] = None,
        prefix: str = "val"
):
    """
    Compute and log all metrics to W&B.


    """
    if wandb is None or not wandb.run:
        print("[WARN] W&B not initialized. Skipping metric logging.")
        return

    # Compute metrics
    metrics = compute_metrics(y_true, y_probs, y_pred)

    # Log scalar metrics
    wandb_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

    # Generate and log plots (only if we have both classes)
    if len(np.unique(y_true)) >= 2:
        try:
            # Confusion Matrix
            cm_fig = plot_confusion_matrix(y_true, y_pred, normalize=True)
            wandb_metrics[f"{prefix}/confusion_matrix"] = wandb.Image(cm_fig)
            plt.close(cm_fig)

            # ROC Curve
            if not np.isnan(metrics['roc_auc']):
                roc_fig = plot_roc_curve(y_true, y_probs, metrics['roc_auc'])
                wandb_metrics[f"{prefix}/roc_curve"] = wandb.Image(roc_fig)
                plt.close(roc_fig)

            # PR Curve
            if not np.isnan(metrics['pr_auc']):
                pr_fig = plot_pr_curve(y_true, y_probs, metrics['pr_auc'])
                wandb_metrics[f"{prefix}/pr_curve"] = wandb.Image(pr_fig)
                plt.close(pr_fig)

        except Exception as e:
            print(f"[WARN] Error creating plots: {e}")

    wandb.log(wandb_metrics, step=step)

    # Print summary
    print(f"\n{prefix.upper()} Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


def evaluate_and_log(
        model,
        loader,
        device,
        epoch: Optional[int] = None,
        prefix: str = "val",
        threshold: float = 0.5
):
    """
    Complete evaluation: collect predictions, compute metrics, and log to W&B.


    """
    print(f"\nEvaluating on {prefix} set...")

    # Collect predictions
    y_true, y_probs, y_pred = collect_predictions(model, loader, device, threshold)

    # Log to W&B
    log_metrics_to_wandb(y_true, y_probs, y_pred, step=epoch, prefix=prefix)

    return y_true, y_probs, y_pred
def evaluate_and_log(model, loader, device, threshold: float = 0.5, prefix: str = "val", step: int | None = None):
    """
    Run eval, compute metrics, and log to W&B.
    - Logs scalar metrics
    - Logs BOTH confusion matrices: counts and normalized
    - Logs ROC and PR curves (if both classes present)
    - Logs raw confusion matrix as a W&B table
    """
    # 1) Collect predictions
    y_true, y_probs, y_pred = collect_predictions(model, loader, device, threshold)

    # 2) Compute scalar metrics
    metrics = compute_metrics(y_true, y_probs, y_pred)

    # 3) Build logging payload
    log_payload = {"epoch": int(step) if step is not None else None}
    log_payload.update({f"{prefix}/{k}": v for k, v in metrics.items()})

    # 4) Confusion matrices (counts and normalized)
    from sklearn.metrics import confusion_matrix
    cm_counts = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm_counts.astype("float") / (cm_counts.sum(axis=1, keepdims=True) + 1e-8)

    # Figures
    fig_cm_counts = plot_confusion_matrix(y_true, y_pred, normalize=False)
    fig_cm_norm   = plot_confusion_matrix(y_true, y_pred, normalize=True)

    log_payload[f"{prefix}/cm_counts"] = wandb.Image(fig_cm_counts)
    log_payload[f"{prefix}/cm_normalized"] = wandb.Image(fig_cm_norm)

    # Also log raw counts as table (TP/FP/FN/TN clarity)
    try:
        cm_table = wandb.Table(columns=["", "Pred 0", "Pred 1"])
        cm_table.add_data("True 0", int(cm_counts[0, 0]), int(cm_counts[0, 1]))
        cm_table.add_data("True 1", int(cm_counts[1, 0]), int(cm_counts[1, 1]))
        log_payload[f"{prefix}/cm_table"] = cm_table
    except Exception:
        pass

    # 5) ROC / PR only if both classes are present
    if len(np.unique(y_true)) >= 2:
        roc_fig = plot_roc_curve(y_true, y_probs, metrics["roc_auc"])
        pr_fig  = plot_pr_curve(y_true, y_probs, metrics["pr_auc"])
        log_payload[f"{prefix}/roc"] = wandb.Image(roc_fig)
        log_payload[f"{prefix}/pr"]  = wandb.Image(pr_fig)
        plt.close(roc_fig); plt.close(pr_fig)

    # 6) Push to W&B (epoch-based; do NOT pass step=...)
    if wandb is not None and wandb.run:
        wandb.log(log_payload)

    # Close CM figs
    plt.close(fig_cm_counts); plt.close(fig_cm_norm)

    return metrics
