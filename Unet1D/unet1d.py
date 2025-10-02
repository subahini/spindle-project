#!/usr/bin/env python3

import os, math, json, datetime, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import yaml
from losses import build_loss_function

# Load CONFIG from YAML file
with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

# Optional W&B
try:
    import wandb

    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False

try:
    import mne

    _HAS_MNE = True
except Exception:
    _HAS_MNE = False

import matplotlib.pyplot as plt


# ------------------------
# Model:
# ------------------------
class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, dropout=0.1, activation="relu"):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True) if activation.lower() == "relu" else nn.SiLU(inplace=True)
        self.do = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.do(self.act(self.bn1(self.conv1(x))))
        x = self.act(self.bn2(self.conv2(x)))
        return x


class UNet1D(nn.Module):
    def __init__(self, in_channels=4, base_channels=64, depth=6, kernel_size=7, dropout=0.05,
                 use_attention=False, final_activation="none"):
        super().__init__()
        self.depth = depth
        self.final_activation = str(final_activation).lower()
        # Encoder
        self.enc = nn.ModuleList()
        self.pool = nn.ModuleList()
        ch = in_channels
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            self.enc.append(ConvBlock1D(ch, out_ch, kernel_size, dropout))
            if i < depth - 1:
                self.pool.append(nn.MaxPool1d(2, ceil_mode=True))
            ch = out_ch
        # Decoder
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            in_ch = base_channels * (2 ** i)
            out_ch = base_channels * (2 ** (i - 1))
            self.up.append(nn.ConvTranspose1d(in_ch, out_ch, 2, stride=2))
            self.dec.append(ConvBlock1D(out_ch * 2, out_ch, kernel_size, dropout))
        # Head
        self.head = nn.Conv1d(base_channels, 1, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight);
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.head.bias, -2.0)

    def forward(self, x):  # x:[B,C,T]
        skips = []
        h = x
        for i, blk in enumerate(self.enc):
            h = blk(h);
            skips.append(h)
            if i < len(self.pool): h = self.pool[i](h)
        for i, (up, dec) in enumerate(zip(self.up, self.dec)):
            h = up(h)
            skip = skips[-(i + 2)]
            if h.size(-1) != skip.size(-1):
                h = F.interpolate(h, size=skip.size(-1), mode='linear', align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = dec(h)
        logits = self.head(h)  # [B,1,T]
        logits = torch.clamp(logits, -8.0, 8.0).squeeze(1)  # [B,T]    # nol need to clamp
        if self.final_activation == "sigmoid":  # no need
            return torch.sigmoid(logits)
        return logits


# ------------------------
# Data
# ------------------------
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.y[i])


def _load_json_labels(path, total_samples, sfreq):
    with open(path, "r") as f:
        labels = json.load(f)
    y = np.zeros(total_samples, dtype=np.float32)
    events = labels.get("detected_spindles") or labels.get("spindles") or []
    for ev in events:
        s = int(max(0, math.floor(float(ev["start"]) * sfreq)))
        e = int(min(total_samples, math.ceil(float(ev["end"]) * sfreq)))
        if e > s: y[s:e] = 1
    return y


def load_edf_windows(edf_path, json_path, cfg):
    if not _HAS_MNE:
        raise RuntimeError("mne is not installed; either install it or use NPY inputs.")
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.set_eeg_reference("average", verbose=False)  # CAR
    raw.filter(cfg["data"]["filter"]["low"], cfg["data"]["filter"]["high"], verbose=False)
    raw.pick(cfg["data"]["channels"])
    sf_target = cfg["data"]["sfreq"]
    if abs(raw.info["sfreq"] - sf_target) > 0.1:
        raw.resample(sfreq=sf_target, verbose=False)
    data = raw.get_data()  # [C,T]
    C, T = data.shape
    y_full = _load_json_labels(json_path, T, raw.info["sfreq"])
    win = int(cfg["data"]["window_sec"] * raw.info["sfreq"])
    step = int(cfg["data"]["step_sec"] * raw.info["sfreq"])
    Xs, Ys = [], []
    for s in range(0, T - win + 1, step):
        e = s + win
        Xs.append(data[:, s:e])
        Ys.append(y_full[s:e])
    X = np.stack(Xs, 0)  # [N,C,Tw]
    y = np.stack(Ys, 0)  # [N,Tw]
    return X, y


def split_data(X, y, ratios=(0.7, 0.15, 0.15), seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X));
    rng.shuffle(idx)
    n1 = int(ratios[0] * len(X));
    n2 = int((ratios[0] + ratios[1]) * len(X))
    return (X[idx[:n1]], y[idx[:n1]]), (X[idx[n1:n2]], y[idx[n1:n2]]), (X[idx[n2:]], y[idx[n2:]])


def create_loaders(X, y, batch, workers, sampler_type="normal", seed=42):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split_data(X, y, seed=seed)
    train_ds = EEGDataset(Xtr, ytr);
    val_ds = EEGDataset(Xva, yva);
    test_ds = EEGDataset(Xte, yte)

    if sampler_type == "undersample":
        labels = (ytr.sum(1) > 0).astype(int)
        pos_idx = np.where(labels == 1)[0];
        neg_idx = np.where(labels == 0)[0]
        n = min(len(pos_idx), len(neg_idx))
        sel_idx = np.concatenate([np.random.choice(pos_idx, n, False),
                                  np.random.choice(neg_idx, n, False)])
        sampler = torch.utils.data.SubsetRandomSampler(sel_idx)
        train_loader = DataLoader(train_ds, batch_size=batch, sampler=sampler, num_workers=workers)
    elif sampler_type == "weighted":
        labels = (ytr.sum(1) > 0).astype(int)
        class_count = np.bincount(labels);
        weight = 1.0 / class_count
        sample_weights = weight[labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        train_loader = DataLoader(train_ds, batch_size=batch, sampler=sampler, num_workers=workers)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers)

    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=workers)
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=workers)
    return train_loader, val_loader, test_loader


# ------------------------
# Manual metrics & ROC/AUC
# ------------------------
def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray):
    """Return TP, TN, FP, FN as ints (arrays are flattened 0/1)."""
    y_true = y_true.astype(np.int64).reshape(-1)
    y_pred = y_pred.astype(np.int64).reshape(-1)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def basic_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    return {"prec": p, "rec": r, "f1": f1, "acc": acc, "TP": tp, "TN": tn, "FP": fp, "FN": fn}


def roc_curve_manual(y_true: np.ndarray, y_scores: np.ndarray, n_thresholds: int = 101):
    """Compute ROC points by thresholding scores; returns fpr, tpr, thresholds."""
    y_true = y_true.astype(np.int64).reshape(-1)
    y_scores = y_scores.reshape(-1)
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    tpr = np.zeros_like(thresholds)
    fpr = np.zeros_like(thresholds)
    # Pre-compute counts for denominator
    P = max(1, int(np.sum(y_true == 1)))
    N = max(1, int(np.sum(y_true == 0)))
    for i, thr in enumerate(thresholds):
        y_pred = (y_scores >= thr).astype(np.int64)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        tpr[i] = tp / P
        fpr[i] = fp / N
    # Ensure increasing FPR for trapezoid
    order = np.argsort(fpr)
    fpr = fpr[order];
    tpr = tpr[order];
    thresholds = thresholds[order]
    return fpr, tpr, thresholds


def auc_trapezoid(x: np.ndarray, y: np.ndarray):
    """Numerical integration via the trapezoidal rule; assumes x sorted ascending."""
    x = x.astype(np.float64);
    y = y.astype(np.float64)
    area = 0.0
    for i in range(1, len(x)):
        dx = x[i] - x[i - 1]
        area += dx * (y[i] + y[i - 1]) / 2.0
    return float(area)


def precision_recall_curve_manual(y_true: np.ndarray, y_scores: np.ndarray, n_thresholds: int = 101):
    """Compute PR curve points by thresholding scores; returns precision, recall, thresholds."""
    y_true = y_true.astype(np.int64).reshape(-1)
    y_scores = y_scores.reshape(-1)
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    precision = np.zeros_like(thresholds)
    recall = np.zeros_like(thresholds)

    # Pre-compute positive count
    P = max(1, int(np.sum(y_true == 1)))

    for i, thr in enumerate(thresholds):
        y_pred = (y_scores >= thr).astype(np.int64)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))

        # Precision: TP / (TP + FP)
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        # Recall: TP / P
        recall[i] = tp / P

    # Sort by recall (ascending) for proper AUC calculation
    order = np.argsort(recall)
    recall = recall[order]
    precision = precision[order]
    thresholds = thresholds[order]

    return precision, recall, thresholds
# ------------------------
# Trainer
# ------------------------
def main():
    cfg = CONFIG  # loaded from YAML

    # Device & AMP
    device = cfg["trainer"]["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # torch.autocast compat wrapper
    def autocast_ctx():
        try:
            return torch.autocast(device_type=device.type, dtype=torch.float16,
                                  enabled=(device.type == "cuda") and bool(cfg["trainer"]["amp"]))
        except TypeError:
            return torch.cuda.amp.autocast(enabled=(device.type == "cuda") and bool(cfg["trainer"]["amp"]))

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda") and bool(cfg["trainer"]["amp"]))

    # Data: prefer NPY if provided, else EDF+JSON
    x_npy, y_npy = cfg["data"].get("x_npy"), cfg["data"].get("y_npy")
    if x_npy and y_npy and Path(x_npy).exists() and Path(y_npy).exists():
        X = np.load(x_npy, mmap_mode="r");
        y = np.load(y_npy, mmap_mode="r")
    else:
        edf_dir = Path(cfg["data"]["edf"]["dir"])
        edfs = sorted(edf_dir.glob("*.edf"))
        if not edfs:
            raise RuntimeError(f"No EDF files found in {edf_dir}. Provide NPY paths in config or add EDFs.")
        X, y = load_edf_windows(str(edfs[0]), cfg["data"]["edf"]["labels_json"], cfg)

    # Build loaders
    train_loader, val_loader, test_loader = create_loaders(
        X, y,
        batch=int(cfg["data"]["batch_size"]),
        workers=int(cfg["data"]["num_workers"]),
        sampler_type=cfg["trainer"].get("sampler", "normal"),
        seed=int(cfg["seed"])
    )

    # Model
    mcfg = cfg["model"]["unet1d"]
    in_channels = X.shape[1] if int(mcfg["in_channels"]) <= 0 else int(mcfg["in_channels"])
    if in_channels != X.shape[1]:
        in_channels = X.shape[1]
    model = UNet1D(
        in_channels=in_channels,
        base_channels=int(mcfg["base_channels"]),
        depth=int(mcfg["depth"]),
        kernel_size=int(mcfg["kernel_size"]),
        dropout=float(mcfg["dropout"]),
        use_attention=bool(mcfg["use_attention"]),
        final_activation=str(mcfg["final_activation"])
    ).to(device)

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["trainer"]["lr"]),
                            weight_decay=float(cfg["trainer"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["trainer"]["epochs"])
    # Loss
    criterion = build_loss_function(cfg["trainer"]["loss"], cfg["trainer"], train_loader)

    # W&B init
    wb = cfg["logging"].get("wandb", {})
    use_wandb = bool(wb.get("enabled", False)) and _HAS_WANDB
    if use_wandb:
        # Set API key if not already logged in
        api_key = wb.get("api_key", None)  # You can add this to your config
        if api_key:
            os.environ["WANDB_API_KEY"] = api_key

        # Auto-generate run name from key config parameters
        if wb.get("name"):
            run_name = wb.get("name")
        else:
            loss_name = cfg["trainer"]["loss"]
            lr = cfg["trainer"]["lr"]
            batch = cfg["data"]["batch_size"]
            depth = cfg["model"]["unet1d"]["depth"]
            base_ch = cfg["model"]["unet1d"]["base_channels"]
            sampler = cfg["trainer"].get("sampler", "normal")

            run_name = f"D{depth}_C{base_ch}_{loss_name}_lr{lr}_b{batch}_{sampler}"

        wandb.init(project=wb.get("project", "spindle-Deep_unet"),
                   entity="subahininadarajh-basel-university",
                   name=run_name,
                   tags=wb.get("tags", None),
                   config=cfg)

    # Train
    best_f1 = -1.0
    best_thr = 0.5
    best_path = Path(cfg["paths"]["checkpoint_dir"]) / "unet1d_best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    patience = 0
    patience_limit = int(cfg["trainer"]["early_stopping_patience"])

    def run_epoch(loader, train=True):
        model.train(train)
        total = 0.0
        grad_norms = []

        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast_ctx():
                logits = model(xb)
                loss = criterion(logits, yb)
            if train:
                scaler.scale(loss).backward()

                # Calculate gradient norm BEFORE clipping
                if cfg["trainer"]["grad_clip_norm"]:
                    scaler.unscale_(opt)
                    total_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    grad_norms.append(total_norm)

                    nn.utils.clip_grad_norm_(model.parameters(), float(cfg["trainer"]["grad_clip_norm"]))

                scaler.step(opt)
                scaler.update()
            total += float(loss.item())

        avg_loss = total / max(1, len(loader))
        avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
        return avg_loss, avg_grad_norm

    for epoch in range(1, int(cfg["trainer"]["epochs"]) + 1):
       # tr_loss = run_epoch(train_loader, train=True)
        tr_loss, grad_norm = run_epoch(train_loader, train=True)
        scheduler.step()
        # Validation
        model.eval()
        probs, trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                p = torch.sigmoid(logits).detach().cpu().numpy()
                probs.append(p);
                trues.append(yb.detach().cpu().numpy())
        probs = np.concatenate(probs, 0).reshape(-1)
        trues = np.concatenate(trues, 0).reshape(-1)

        # Threshold choice
        ev = cfg.get("eval", {})
        if ev.get("threshold_mode", "sweep") == "fixed":
            thr = float(ev.get("fixed_threshold", 0.5))
            preds = (probs >= thr).astype(np.int64)
            m = basic_metrics(trues, preds)
        else:
            # sweep for best metric
            n_th = int(ev.get("n_thresholds", 101))
            metric = ev.get("metric_for_best", "f1")
            best_thr, best_m, best_score = 0.5, None, -1.0
            for thr in np.linspace(0.0, 1.0, n_th):
                preds = (probs >= thr).astype(np.int64)
                mm = basic_metrics(trues, preds)
                score = mm.get(metric, mm["f1"])
                if score > best_score:
                    best_thr, best_m, best_score = float(thr), mm, float(score)
            thr, m = best_thr, best_m

        # Manual ROC & AUC
        fpr, tpr, _thr_vec = roc_curve_manual(trues, probs, n_thresholds=int(ev.get("n_thresholds", 101)))
        roc_auc = auc_trapezoid(fpr, tpr)
        # Manual PR curve & AUC
        precision, recall, _pr_thr = precision_recall_curve_manual(trues, probs,
                                                                   n_thresholds=int(ev.get("n_thresholds", 101)))
        pr_auc = auc_trapezoid(recall, precision)
        # Logging
        log_payload = {
            "epoch": epoch,
            "loss/train": tr_loss,
            "train/grad_norm": grad_norm,     # logging gradding to observ it
            "val/threshold": thr,
            "val/precision": m["prec"],
            "val/recall": m["rec"],
            "val/f1": m["f1"],
            "val/accuracy": m["acc"],
            "val/TP": m["TP"], "val/TN": m["TN"], "val/FP": m["FP"], "val/FN": m["FN"],
            "val/roc_auc": roc_auc,
            "val/pr_auc": pr_auc,
        }

        if use_wandb:
            # ROC figure (manual)
            fig_roc = plt.figure(figsize=(6, 5))
            ax = fig_roc.add_subplot(111)
            ax.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC={roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color='gray')
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curve - Epoch {epoch}")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # PR figure (manual)
            fig_pr = plt.figure(figsize=(6, 5))
            ax_pr = fig_pr.add_subplot(111)
            ax_pr.plot(recall, precision, linewidth=2, label=f'PR (AUC={pr_auc:.3f})')
            ax_pr.axhline(y=np.sum(trues) / len(trues), linestyle="--", linewidth=1, color='gray', label='Baseline')
            ax_pr.set_xlabel("Recall")
            ax_pr.set_ylabel("Precision")
            ax_pr.set_title(f"Precision-Recall Curve - Epoch {epoch}")
            ax_pr.legend()
            ax_pr.grid(True, alpha=0.3)

            wandb.log({**log_payload, "val/roc_curve": wandb.Image(fig_roc), "val/pr_curve": wandb.Image(fig_pr)},
                      step=epoch)
            plt.close(fig_roc)
            plt.close(fig_pr)

        else:
            # Clean formatted output instead of dictionary dump
            print(f"Epoch {epoch} Metrics:")
            for k, v in log_payload.items():
                if k == "epoch":
                    continue
                formatted_v = int(v) if isinstance(v, (bool, int)) else f"{float(v):.4f}"
                print(f"  {k}: {formatted_v}")
            print()  # Empty line for readability

        print(f"Epoch {epoch}: loss={tr_loss:.4f} | thr={thr:.2f} | "
              f"F1={m['f1']:.4f} P={m['prec']:.3f} R={m['rec']:.3f} Acc={m['acc']:.3f} "
              f"| TP={m['TP']} FP={m['FP']} TN={m['TN']} FN={m['FN']} | ROC-AUC={roc_auc:.3f} PR-AUC={pr_auc:.3f}")

        # Early stopping on F1
        if m["f1"] > best_f1 + 1e-4:
            best_f1, best_thr = m["f1"], thr
            patience = 0
            torch.save(
                {"model": model.state_dict(),
                 "best_thr": float(best_thr),
                 "config": CONFIG},
                str(best_path)
            )
        """else:
            patience += 1
            if patience >= patience_limit:
                print("Early stopping.")
                break"""     # trying to observe the run

    print(f"Done. Best F1={best_f1:.4f} @ thr={best_thr:.2f}. Saved: {best_path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="Path to YAML config to merge")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--loss", type=str, default=None, choices=["bce", "weighted_bce", "focal", "dice"])
    ap.add_argument("--fixed_threshold", type=float, default=None)
    args = ap.parse_args()


    # --- merge YAML config into CONFIG if provided ---
    def _deep_update(dst, src):
        for k, v in (src or {}).items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                _deep_update(dst[k], v)
            else:
                dst[k] = v


    if args.config:
        with open(args.config, "r") as f:
            _deep_update(CONFIG, yaml.safe_load(f) or {})

    # --- keep CLI overrides (highest priority) ---
    if args.epochs is not None:
        CONFIG["trainer"]["epochs"] = int(args.epochs)
    if args.loss is not None:
        CONFIG["trainer"]["loss"] = args.loss
    if args.fixed_threshold is not None:
        CONFIG.setdefault("eval", {})
        CONFIG["eval"]["threshold_mode"] = "fixed"
        CONFIG["eval"]["fixed_threshold"] = float(args.fixed_threshold)

    main()