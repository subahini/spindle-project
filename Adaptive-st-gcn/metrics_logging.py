import numpy as np, matplotlib.pyplot as plt, torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
try:
    import wandb
except Exception:
    wandb = None


@torch.no_grad()
def collect_predictions(model, loader, device):
    y_true, y_prob = [], []
    model.eval()
    for batch in loader:
        x = torch.stack([b["x"] for b in batch]).to(device)
        y = torch.stack([b["y_t"] for b in batch]).cpu().numpy()
        p = torch.sigmoid(model(x)["logits_global"]).cpu().numpy()
        y_true.append(y.reshape(-1)); y_prob.append(p.reshape(-1))
    return np.concatenate(y_true), np.concatenate(y_prob)


def compute_metrics(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "pr_auc": average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
    }
    return out, y_pred


# --- plotting ---
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="black")
    plt.tight_layout(); return fig


def plot_roc_curve(y_true, y_prob, auc):
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title(f"ROC Curve (AUC = {auc:.3f})"); plt.tight_layout(); return fig


def plot_pr_curve(y_true, y_prob, auc):
    fig, ax = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title(f"PR Curve (AUC = {auc:.3f})"); plt.tight_layout(); return fig


# --- main evaluation+logging ---
@torch.no_grad()
def evaluate_and_log(model, loader, device, prefix="val", step=None, threshold=0.5):
    y_true, y_prob = collect_predictions(model, loader, device)
    metrics, y_pred = compute_metrics(y_true, y_prob, threshold)
    print(f"{prefix.upper()} | Acc {metrics['accuracy']:.3f} F1 {metrics['f1']:.3f} ROC {metrics['roc_auc']:.3f} PR {metrics['pr_auc']:.3f}")

    if wandb and wandb.run:
        payload = {f"{prefix}/{k}": v for k, v in metrics.items()}
        payload["epoch"] = step
        if len(np.unique(y_true)) > 1:
            payload[f"{prefix}/confusion_matrix"] = wandb.Image(plot_confusion_matrix(y_true, y_pred))
            payload[f"{prefix}/roc_curve"] = wandb.Image(plot_roc_curve(y_true, y_prob, metrics["roc_auc"]))
            payload[f"{prefix}/pr_curve"] = wandb.Image(plot_pr_curve(y_true, y_prob, metrics["pr_auc"]))
            plt.close("all")
        wandb.log(payload)
    return metrics
