import json
import yaml
import wandb
import mne
import numpy as np
import torch

from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)

from models import SpindleCNN
from losses import build_loss_function


# ============================================================
# JSON LOADER (CRNN-COMPATIBLE: detected_spindles dict OR list)
# ============================================================

def load_spindles_from_json(json_path: str):
    """
    Supports both dataset variants:

    Variant A:
    {"detected_spindles": {"0": {"start":..,"end":..}, ...}}

    Variant B:
    {"detected_spindles": [{"start":..,"end":..}, ...]}

    Returns: list[(start_sec, end_sec)]
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    spindles = data.get("detected_spindles") or data.get("spindles") or []

    if isinstance(spindles, dict):
        iterable = spindles.values()
    elif isinstance(spindles, list):
        iterable = spindles
    else:
        iterable = []

    pairs = []
    for ev in iterable:
        if isinstance(ev, dict) and ("start" in ev) and ("end" in ev):
            try:
                pairs.append((float(ev["start"]), float(ev["end"])))
            except Exception:
                pass

    pairs.sort(key=lambda x: x[0])
    return pairs


# ============================================================
# EDF -> preprocessing -> sample-labels -> windows -> window-labels
# ============================================================

def preprocess_raw(raw: mne.io.BaseRaw, channels, hp, lp):
    raw.pick(channels)
    raw.set_eeg_reference("average", verbose=False)   # average reference
    raw.filter(hp, lp, verbose=False)                 # bandpass
    return raw.get_data()                             # (C, T)


def make_windows(data, labels, sfreq, win_s, step_s, overlap_thr):
    """
    data:   (C, T)
    labels: (T,) 0/1 sample-level
    Return:
      X: (N, 1, C, win)
      y: (N, 1) window label
    """
    win = int(win_s * sfreq)
    step = int(step_s * sfreq)

    X_list, y_list = [], []
    T = data.shape[1]

    for start in range(0, T - win + 1, step):
        end = start + win
        seg = data[:, start:end]
        frac = labels[start:end].mean()
        y_win = 1.0 if frac >= overlap_thr else 0.0

        X_list.append(seg[None, :, :])   # (1, C, win)
        y_list.append([y_win])           # (1,)

    X = np.stack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.float32)

    # z-score per-window per-channel
    X = (X - X.mean(axis=-1, keepdims=True)) / (X.std(axis=-1, keepdims=True) + 1e-6)
    return X, y


def load_single_file(edf_path: str, json_path: str, cfg):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    sfreq = float(raw.info["sfreq"])

    data = preprocess_raw(
        raw,
        cfg["channels"]["names"],
        cfg["filter"]["hp_freq"],
        cfg["filter"]["lp_freq"],
    )

    # sample-level labels
    labels = np.zeros(data.shape[1], dtype=np.float32)
    for s_sec, e_sec in load_spindles_from_json(json_path):
        s = int(max(0, np.floor(s_sec * sfreq)))
        e = int(min(labels.shape[0], np.ceil(e_sec * sfreq)))
        if e > s:
            labels[s:e] = 1.0

    # windows + window labels
    return make_windows(
        data,
        labels,
        sfreq,
        cfg["windowing"]["window_sec"],
        cfg["windowing"]["step_sec"],
        cfg["windowing"]["overlap_threshold"],
    )


# ============================================================
# FILE-LEVEL SPLIT (NO LEAKAGE)
# ============================================================

def build_data_splits_fixed(cfg):
    pairs = [(p["edf"], p["json"]) for p in cfg["subject_files"]]
    rng = np.random.RandomState(cfg["splits"]["random_state"])
    rng.shuffle(pairs)

    n = len(pairs)
    if n < 3:
        raise ValueError(f"Need at least 3 EDF/JSON pairs for train/val/test, got {n}")

    train_pairs = pairs[:-2]
    val_pairs = pairs[-2:-1]
    test_pairs = pairs[-1:]

    def load_set(pairs_list, name):
        Xs, ys = [], []
        for edf, js in pairs_list:
            print(f"  {name}: processing {Path(edf).name}")
            Xi, yi = load_single_file(edf, js, cfg)
            Xs.append(Xi)
            ys.append(yi)
        X = np.concatenate(Xs, axis=0)
        y = np.concatenate(ys, axis=0)
        print(f"  {name}: {len(X)} windows, {float(y.mean())*100:.2f}% positive")
        return X, y

    print("\n============================================================")
    print("FILE SPLIT (prevents data leakage):")
    print(f"  Train: {len(train_pairs)} files")
    print(f"  Val:   {len(val_pairs)} files")
    print(f"  Test:  {len(test_pairs)} files")
    print("============================================================\n")

    return load_set(train_pairs, "TRAIN"), load_set(val_pairs, "VAL"), load_set(test_pairs, "TEST")


# ============================================================
# W&B INTERACTIVE PLOTS (NOT IMAGES) — CRNN-STYLE
# ============================================================
def wandb_log_interactive_curves(prefix, y_true, y_score, step):
    """
    W&B interactive PR/ROC curves using y_true + probs_2d (works with your wandb version)
    """
    if wandb.run is None:
        return

    y_true = np.asarray(y_true).reshape(-1).astype(np.int64)
    y_score = np.asarray(y_score).reshape(-1).astype(np.float32)

    # IMPORTANT: probs for BOTH classes -> shape [N,2]
    probs_2d = np.vstack([1.0 - y_score, y_score]).T

    wandb.log({
        f"{prefix}/roc_curve": wandb.plot.roc_curve(
            y_true,
            probs_2d,
            labels=["non-spindle", "spindle"],
        ),
        f"{prefix}/pr_curve": wandb.plot.pr_curve(
            y_true,
            probs_2d,
            labels=["non-spindle", "spindle"],
        ),
    }, step=step)
def wandb_log_interactive_confusion(prefix, y_true, y_pred, step):
    if wandb.run is None:
        return

    y_true = np.asarray(y_true).reshape(-1).astype(np.int64)
    y_pred = np.asarray(y_pred).reshape(-1).astype(np.int64)

    wandb.log({
        f"{prefix}/confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_true,
            preds=y_pred,
            class_names=["non-spindle", "spindle"],
        )
    }, step=step)


# ============================================================
# METRICS
# ============================================================

def find_best_threshold(y_true_int, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true_int, y_score)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    return float(thresholds[int(np.argmax(f1))])


# ============================================================
# EVALUATE
# ============================================================

def evaluate(model, loader, criterion, device):
    model.eval()
    ys, ps = [], []
    total_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)                      # (B,1)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)

            prob = torch.sigmoid(logits)
            ys.append(y.detach().cpu().numpy())
            ps.append(prob.detach().cpu().numpy())

    y_true = np.concatenate(ys, axis=0).ravel()
    y_score = np.concatenate(ps, axis=0).ravel()

    y_true_i = (y_true > 0.5).astype(np.int32)

    thr = find_best_threshold(y_true_i, y_score)
    y_pred_i = (y_score >= thr).astype(np.int32)

    tp = int(np.sum((y_true_i == 1) & (y_pred_i == 1)))
    fp = int(np.sum((y_true_i == 0) & (y_pred_i == 1)))
    tn = int(np.sum((y_true_i == 0) & (y_pred_i == 0)))
    fn = int(np.sum((y_true_i == 1) & (y_pred_i == 0)))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    roc_auc = float(roc_auc_score(y_true_i, y_score)) if len(np.unique(y_true_i)) > 1 else 0.0
    pr_auc = float(average_precision_score(y_true_i, y_score)) if len(np.unique(y_true_i)) > 1 else 0.0

    return {
        "loss": float(total_loss / len(loader.dataset)),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": float(thr),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "y_true": y_true_i,
        "y_score": y_score,
        "y_pred": y_pred_i,
    }


# ============================================================
# TRAIN
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return float(total_loss / len(loader.dataset))


# ============================================================
# MAIN
# ============================================================

def main():
    cfg = yaml.safe_load(open("config.yaml", "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg.get("enabled", True):
        wandb.init(
            project=wandb_cfg.get("project", "spindle_cnn2d"),
            entity=wandb_cfg.get("entity", None),
            name=wandb_cfg.get("run_name", None),
            config=cfg,
        )
    else:
        wandb.init(mode="disabled")

    (Xt, yt), (Xv, yv), (Xte, yte) = build_data_splits_fixed(cfg)

    def make_loader(X, y, shuffle):
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return DataLoader(
            ds,
            batch_size=int(cfg["training"]["batch_size"]),
            shuffle=shuffle,
            num_workers=int(cfg["training"]["num_workers"]),
            pin_memory=True,
        )

    train_loader = make_loader(Xt, yt, True)
    val_loader = make_loader(Xv, yv, False)
    test_loader = make_loader(Xte, yte, False)

    n_channels = len(cfg["channels"]["names"])
    model = SpindleCNN(n_channels=n_channels).to(device)

    criterion = build_loss_function(cfg["loss"]["name"], cfg["loss"], train_loader)

    lr = float(cfg["training"]["learning_rate"])
    wd = float(cfg["training"].get("weight_decay", 0.0))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    num_epochs = int(cfg["training"]["num_epochs"])
    log_val_plots_every = int(wandb_cfg.get("log_val_plots_every", 1))

    print("\n============================================================")
    print("CNN TRAINING (WINDOW-LEVEL)")
    print(f"  Device: {device}")
    print(f"  Epochs: {num_epochs}")
    print(f"  LR:     {lr}")
    print(f"  Loss:   {cfg['loss']['name']}")
    print("============================================================\n")

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        val = evaluate(model, val_loader, criterion, device)

        # Interactive plots into W&B Charts (NOT images)
        if log_val_plots_every <= 1 or (epoch % log_val_plots_every == 0):
            
            wandb_log_interactive_curves("val", val["y_true"], val["y_score"], step=epoch)
            wandb_log_interactive_confusion("val", val["y_true"], val["y_pred"], step=epoch)

        # Scalars (same step=epoch)
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val["loss"],
            "val/f1": val["f1"],
            "val/roc_auc": val["roc_auc"],
            "val/pr_auc": val["pr_auc"],
            "val/threshold": val["threshold"],
            "lr": optimizer.param_groups[0]["lr"],
        }, step=epoch)

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_f1={val['f1']:.4f} | val_pr_auc={val['pr_auc']:.4f}"
        )

    # FINAL TEST (new step so it never goes backwards)
    test = evaluate(model, test_loader, criterion, device)
    final_step = num_epochs + 1

    wandb_log_interactive_curves("test", test["y_true"], test["y_score"], step=final_step)
    wandb_log_interactive_confusion("test", test["y_true"], test["y_pred"], step=final_step)

    wandb.log({
        "test/loss": test["loss"],
        "test/f1": test["f1"],
        "test/roc_auc": test["roc_auc"],
        "test/pr_auc": test["pr_auc"],
        "test/threshold": test["threshold"],
    }, step=final_step)

    print("\n================ FINAL TEST ================")
    print(
        f"F1={test['f1']:.4f} | ROC-AUC={test['roc_auc']:.4f} | "
        f"PR-AUC={test['pr_auc']:.4f} | thr={test['threshold']:.3f}"
    )
    print("===========================================\n")

    wandb.finish()


if __name__ == "__main__":
    main()
