# Full CRNN pipeline comparable to UNet: sample-level predictions
import os, math, json, argparse, time, random, csv
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import yaml
import wandb

# Optional sklearn for AUCs
try:

    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

# mne only needed in RAW mode
try:
    import mne
except Exception:
    mne = None

# plotting libs
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
try:
    import seaborn as sns
except Exception:
    sns = None

from losses import build_loss_function

ALL_EEG_19 = ["C3", "C4", "O1", "O2", "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "T4", "T5", "T6", "P3", "Pz",
              "P4", "Oz"]


# ----------------- utils -----------------
def set_seed(seed: int):
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed);
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def safe_cfg(cfg):
    cfg = cfg or {}
    cfg.setdefault("project", {"name": "crnn_spindles", "device": "cuda", "seed": 42})
    for k in ["trainer", "signal", "spectrogram", "model", "paths", "loss", "eval", "log"]:
        cfg.setdefault(k, {})
    if not cfg["signal"].get("channels"): cfg["signal"]["channels"] = ALL_EEG_19[:]
    if "in_channels" not in cfg["model"]:
        cfg["model"]["in_channels"] = len(cfg["signal"]["channels"])
    return cfg


def apply_overrides(cfg, args):
    if args.loss_name:
        cfg.setdefault("loss", {})["name"] = args.loss_name
    if args.lr is not None:
        cfg.setdefault("trainer", {})["lr"] = float(args.lr)
    if args.wandb_entity:
        cfg.setdefault("project", {})["entity"] = args.wandb_entity
    return cfg


# ----------------- spectrogram & model -----------------
class Spectrogram(nn.Module):
    def __init__(self, sfreq: int, n_fft: int, hop_length: int, win_length: int, center: bool = True,
                 power: float = 2.0):
        super().__init__()
        self.sf = sfreq;
        self.n_fft = n_fft;
        self.hop = hop_length;
        self.win = win_length
        self.center = center;
        self.power = power
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        X = torch.stft(x.reshape(B * C, T), n_fft=self.n_fft, hop_length=self.hop, win_length=self.win,
                       center=self.center, window=self.window, return_complex=True).abs()
        if self.power != 1.0: X = X.pow(self.power)
        return X.reshape(B, C, X.shape[-2], X.shape[-1])


class SE2d(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, max(1, ch // r), 1);
        self.fc2 = nn.Conv2d(max(1, ch // r), ch, 1)

    def forward(self, x):
        s = x.mean(dim=(2, 3), keepdim=True)
        s = F.relu(self.fc1(s), inplace=True);
        s = torch.sigmoid(self.fc2(s))
        return x * s


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, se=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False);
        self.bn = nn.BatchNorm2d(out_ch)
        self.se = SE2d(out_ch) if se else nn.Identity()

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)), inplace=True);
        return self.se(x)


class MultiScaleStem(nn.Module):
    def __init__(self, in_ch, out_ch, se=True):
        super().__init__()
        mid = max(1, out_ch // 3)
        self.b1 = ConvBNReLU(in_ch, mid, k=3, p=1, se=se)
        self.b2 = ConvBNReLU(in_ch, mid, k=5, p=2, se=se)
        self.b3 = ConvBNReLU(in_ch, mid, k=7, p=3, se=se)
        self.fuse = ConvBNReLU(mid * 3, out_ch, k=1, p=0, se=False)

    def forward(self, x):
        return self.fuse(torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1))


class FPNLite(nn.Module):
    def __init__(self, c1, c2, c3, out_ch):
        super().__init__()
        self.l1 = nn.Conv2d(c1, out_ch, 1);
        self.l2 = nn.Conv2d(c2, out_ch, 1);
        self.l3 = nn.Conv2d(c3, out_ch, 1)

    def up2xF(self, x): return F.interpolate(x, scale_factor=(2, 1), mode="bilinear", align_corners=False)

    def forward(self, f1, f2, f3):
        p3 = self.l3(f3);
        p2 = self.l2(f2) + self.up2xF(p3);
        p1 = self.l1(f1) + self.up2xF(p2);
        return p1


class CRNN2D_BiGRU(nn.Module):
    def __init__(self, c_in: int, base_ch: int = 32, fpn_ch: int = 128, rnn_hidden: int = 128, rnn_layers: int = 2,
                 bidirectional: bool = True, bias_init_prior: float = None, use_se: bool = True,
                 sfreq: int = 200, n_fft: int = 128, hop_length: int = 20, win_length: int = 128, center: bool = True,
                 power: float = 2.0, upsample_mode: str = 'linear'):
        super().__init__()
        self.spec = Spectrogram(sfreq, n_fft, hop_length, win_length, center, power)
        self.stem = MultiScaleStem(c_in, base_ch, se=use_se)
        self.b1 = nn.Sequential(ConvBNReLU(base_ch, base_ch, se=use_se), ConvBNReLU(base_ch, base_ch, se=use_se),
                                nn.AvgPool2d((2, 1)))
        self.b2 = nn.Sequential(ConvBNReLU(base_ch, base_ch * 2, se=use_se), nn.AvgPool2d((2, 1)))
        self.b3 = nn.Sequential(ConvBNReLU(base_ch * 2, base_ch * 4, se=use_se), nn.AvgPool2d((2, 1)))
        self.fpn = FPNLite(base_ch, base_ch * 2, base_ch * 4, fpn_ch)
        self.post_fpn = nn.Conv2d(fpn_ch, fpn_ch, 1)
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.rnn = nn.GRU(fpn_ch, rnn_hidden, num_layers=rnn_layers, batch_first=False, bidirectional=bidirectional)
        rnn_out = rnn_hidden * (2 if bidirectional else 1)
        self.head = nn.Conv1d(rnn_out, 1, 1)
        self.upsample_mode = upsample_mode

        if bias_init_prior is not None and 0 < bias_init_prior < 1:
            with torch.no_grad():
                self.head.bias.data.fill_(math.log(bias_init_prior / (1 - bias_init_prior)))

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        B, C, Traw = x_raw.shape
        S = self.spec(x_raw)  # [B,C,F,Ts]
        f1 = self.b1(self.stem(S));
        f2 = self.b2(f1);
        f3 = self.b3(f2)
        p = self.fpn(f1, f2, f3);
        p = F.relu(self.post_fpn(p), inplace=True)
        p = self.freq_pool(p).squeeze(2)  # [B,fpn_ch,Ts]
        seq = p.permute(2, 0, 1)  # [Ts,B,fpn_ch]
        rnn_out, _ = self.rnn(seq)  # [Ts,B,rout]
        rnn_out = rnn_out.permute(1, 2, 0)  # [B,rout,Ts]
        logits = self.head(rnn_out)  # [B,1,Ts]

        # Upsample to sample-level resolution (matching UNet)
        logits = F.interpolate(logits, size=Traw, mode=self.upsample_mode,
                               align_corners=False if self.upsample_mode != 'nearest' else None)
        return logits.squeeze(1)  # [B, Traw]


# ----------------- Dataset (returns TUPLES) -----------------
class EEGDataset(Dataset):
    """Dataset that returns (x, y) tuples for compatibility with losses.py"""

    def __init__(self, X, y, normalize="zscore", reference="car"):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.normalize = normalize
        self.reference = reference

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i]
        if self.reference == "car":
            x = x - x.mean(axis=0, keepdims=True)
        if self.normalize == "zscore":
            mu = x.mean(axis=-1, keepdims=True)
            sd = x.std(axis=-1, keepdims=True) + 1e-6
            x = (x - mu) / sd
        return torch.from_numpy(x), torch.from_numpy(self.y[i])


# ----------------- RAW EDF loading (UNet-style) -----------------
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


def load_edf_windows(edf_path, json_path, cfg_data):
    assert mne is not None, "Install mne to read RAW EDF (pip install mne)"
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.set_eeg_reference("average", verbose=False)  # CAR
    raw.filter(cfg_data["filter"]["low"], cfg_data["filter"]["high"], verbose=False)
    raw.pick(cfg_data["channels"])
    sf_target = cfg_data["sfreq"]
    if abs(raw.info["sfreq"] - sf_target) > 0.1:
        raw.resample(sfreq=sf_target, verbose=False)
    data = raw.get_data()  # [C, T]
    C, T = data.shape
    y_full = _load_json_labels(json_path, T, raw.info["sfreq"])
    win = int(cfg_data["window_sec"] * raw.info["sfreq"])
    step = int(cfg_data["step_sec"] * raw.info["sfreq"])
    Xs, Ys = [], []
    for s in range(0, T - win + 1, step):
        e = s + win
        Xs.append(data[:, s:e])
        Ys.append(y_full[s:e])
    return np.stack(Xs, 0), np.stack(Ys, 0)  # [N,C,Tw], [N,Tw]


def split_data(X, y, ratios=(0.7, 0.15, 0.15), seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X));
    rng.shuffle(idx)
    n1 = int(ratios[0] * len(X));
    n2 = int((ratios[0] + ratios[1]) * len(X))
    return (X[idx[:n1]], y[idx[:n1]]), (X[idx[n1:n2]], y[idx[n1:n2]]), (X[idx[n2:]], y[idx[n2:]])


# ----------------- metrics (simplified, matching UNet) -----------------
def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray):
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
    return {"precision": p, "recall": r, "f1": f1, "accuracy": acc, "TP": tp, "TN": tn, "FP": fp, "FN": fn}


def roc_curve_manual(y_true: np.ndarray, y_scores: np.ndarray, n_thresholds: int = 101):
    y_true = y_true.astype(np.int64).reshape(-1)
    y_scores = y_scores.reshape(-1)
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    tpr = np.zeros_like(thresholds, dtype=np.float64)
    fpr = np.zeros_like(thresholds, dtype=np.float64)
    P = max(1, int(np.sum(y_true == 1)))
    N = max(1, int(np.sum(y_true == 0)))
    for i, thr in enumerate(thresholds):
        y_pred = (y_scores >= thr).astype(np.int64)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        tpr[i] = tp / P
        fpr[i] = fp / N
    order = np.argsort(fpr)
    return fpr[order], tpr[order], thresholds[order]


def auc_trapezoid(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64);
    y = y.astype(np.float64)
    area = 0.0
    for i in range(1, len(x)):
        dx = x[i] - x[i - 1]
        area += dx * (y[i] + y[i - 1]) / 2.0
    return float(area)


# ----------------- dataloaders & samplers -----------------
def build_dataloaders(cfg):
    """Build dataloaders matching UNet's structure"""
    d = cfg["data"]
    sig = cfg.get("signal", {})
    bs = int(d.get("batch_size", cfg["trainer"]["batch_size"]))
    nw = int(d.get("num_workers", cfg["trainer"]["num_workers"]))
    seed = int(cfg.get("seed", 42))

    # 1) Prefer NPY (exactly like UNet)
    x_npy, y_npy = d.get("x_npy"), d.get("y_npy")
    if x_npy and y_npy and os.path.exists(x_npy) and os.path.exists(y_npy):
        X = np.load(x_npy, mmap_mode="r")
        y = np.load(y_npy, mmap_mode="r")
    else:
        # 2) RAW fallback: use FIRST EDF in dir
        edf_dir = d["edf"]["dir"]
        edfs = sorted([f for f in os.listdir(edf_dir) if f.lower().endswith(".edf")])
        if not edfs:
            raise RuntimeError(f"No EDF files found in {edf_dir}. Provide NPY or add EDFs.")
        X, y = load_edf_windows(os.path.join(edf_dir, edfs[0]), d["edf"]["labels_json"], d)

    # split in memory
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = split_data(X, y, seed=seed)
    train_ds = EEGDataset(Xtr, Ytr, normalize=sig.get("normalize", "zscore"), reference=sig.get("reference", "car"))
    val_ds = EEGDataset(Xva, Yva, normalize=sig.get("normalize", "zscore"), reference=sig.get("reference", "car"))
    test_ds = EEGDataset(Xte, Yte, normalize=sig.get("normalize", "zscore"), reference=sig.get("reference", "car"))

    # sampler options
    sampler_mode = cfg["trainer"].get("sampler", "normal").lower()
    if sampler_mode == "undersample":
        labels = (Ytr.sum(1) > 0).astype(int)
        pos_idx = np.where(labels == 1)[0];
        neg_idx = np.where(labels == 0)[0]
        n = min(len(pos_idx), len(neg_idx))
        sel = np.concatenate([np.random.choice(pos_idx, n, False), np.random.choice(neg_idx, n, False)])
        sampler = torch.utils.data.SubsetRandomSampler(sel)
        train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler, num_workers=nw, pin_memory=True)
    elif sampler_mode == "weighted":
        labels = (Ytr.sum(1) > 0).astype(int)
        class_count = np.bincount(labels);
        w = 1.0 / class_count
        sample_w = w[labels];
        sampler = WeightedRandomSampler(sample_w, len(sample_w))
        train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler, num_workers=nw, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


# ----------------- trainer & eval -----------------
@dataclass
class TrainState:
    step: int = 0
    best: float = -1.0
    epoch: int = 0


def run_eval(dloader, cfg, model, device, split_name="val"):
    model.eval()
    all_probs = [];
    all_y = []
    with torch.no_grad():
        for x, y in dloader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)  # [B, T] - sample-level
            probs = torch.sigmoid(logits)
            all_y.append(y.detach().cpu())
            all_probs.append(probs.detach().cpu())

    probs = torch.cat(all_probs, dim=0).numpy()  # [N, T]
    y_true = torch.cat(all_y, dim=0).numpy()

    metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

    y_true_flat = y_true.reshape(-1)
    y_score_flat = probs.reshape(-1)

    # Threshold sweep (matching UNet)
    evcfg = cfg.get("eval", {})
    n_th = int(evcfg.get("n_thresholds", 101))
    metric = evcfg.get("metric_for_best", "f1")
    best_thr, best_m, best_score = 0.5, None, -1.0

    for thr in np.linspace(0.0, 1.0, n_th):
        preds = (probs >= thr).astype(np.int64)
        mm = basic_metrics(y_true, preds)
        score = mm.get(metric, mm["f1"])
        if score > best_score:
            best_thr, best_m, best_score = float(thr), mm, float(score)

    metrics = best_m
    metrics["threshold"] = best_thr

    # ROC/AUC
    fpr, tpr, _ = roc_curve_manual(y_true_flat, y_score_flat, n_thresholds=n_th)
    roc_auc = auc_trapezoid(fpr, tpr)
    metrics["roc_auc"] = roc_auc

    # --- PR-AUC (Average Precision) ---
    try:
        pr_auc = float(average_precision_score(y_true_flat, y_score_flat))
        metrics["pr_auc"] = pr_auc
    except Exception:
        pr_auc = None

    # Optional PR curve plot
    if plt is not None and wandb.run is not None:
        try:
            if 'precision_recall_curve' in globals():
                prec, rec, _ = precision_recall_curve(y_true_flat, y_score_flat)
            else:
                # Fallback: quick manual sweep (coarse)
                ths = np.linspace(0.0, 1.0, n_th)
                prec, rec = [], []
                for thr in ths:
                    preds = (y_score_flat >= thr).astype(np.int64)
                    m = basic_metrics(y_true_flat, preds)
                    prec.append(m["precision"]);
                    rec.append(m["recall"])
                prec, rec = np.array(prec), np.array(rec)

            fig = plt.figure()
            plt.plot(rec, prec, label=f"PR (AP={pr_auc:.3f})" if pr_auc is not None else "PR")
            # no-skill baseline = prevalence
            base = float(y_true_flat.mean()) if y_true_flat.size else 0.0
            plt.hlines(base, 0, 1, linestyles="dashed", label=f"Baseline={base:.3f}")
            plt.xlabel("Recall");
            plt.ylabel("Precision");
            plt.legend(loc="lower left")
            plt.title(f"Precision–Recall [{split_name}]")
            wandb.log({f"{split_name}/pr_curve": wandb.Image(fig)}, commit=False)
            plt.close(fig)
        except Exception:
            pass

    # Plot ROC
    if plt is not None and wandb.run is not None:
        try:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(fpr, tpr)
            ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC (AUC={roc_auc:.3f}) [{split_name}]")
            wandb.log({f"{split_name}/roc_curve": wandb.Image(fig)}, commit=False)
            plt.close(fig)
        except Exception:
            pass

    # Confusion matrix (separate figure)
    if plt is not None and wandb.run is not None:
        try:
            tp, tn, fp, fn = metrics["TP"], metrics["TN"], metrics["FP"], metrics["FN"]
            cm = np.array([[tn, fp], [fn, tp]], dtype=np.int64)

            fig_cm = plt.figure(figsize=(6, 5))
            ax = fig_cm.add_subplot(111)

            if sns is not None:
                sns.heatmap(cm, annot=True, fmt="d", cbar=False, ax=ax, cmap="Blues",
                            xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
            else:
                im = ax.imshow(cm, cmap="Blues", aspect='auto')
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.set_xticklabels(["Pred 0", "Pred 1"])
                ax.set_yticklabels(["True 0", "True 1"])
                # Add text annotations
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                                color='white' if cm[i, j] > cm.max() / 2 else 'black',
                                fontsize=16, fontweight='bold')

            ax.set_xlabel("Predicted Label", fontsize=12)
            ax.set_ylabel("True Label", fontsize=12)
            ax.set_title(f"Confusion Matrix @thr={best_thr:.2f} [{split_name}]", fontsize=14)

            plt.tight_layout()
            wandb.log({f"{split_name}/confusion_matrix": wandb.Image(fig_cm)}, commit=False)
            plt.close(fig_cm)
        except Exception as e:
            print(f"Warning: Could not plot confusion matrix: {e}")

    return probs, y_true, metrics


def train_and_eval(cfg):
    device = torch.device(cfg["project"].get("device", "cuda") if torch.cuda.is_available() else "cpu")
    set_seed(cfg["project"].get("seed", 42))
    ensure_dir(cfg["paths"]["out_dir"])

    # W&B
    wb = cfg.get("logging", {}).get("wandb", {})
    if wb.get("enabled", True):
        wandb.init(project=wb.get("project", cfg["project"]["name"]),
                   entity=wb.get("entity"),
                   name=wb.get("name"),
                   tags=wb.get("tags"),
                   config=cfg)

    # Data
    ds_tr, ds_va, ds_te, dl_tr, dl_va, dl_te = build_dataloaders(cfg)

    # Model
    mcfg, scfg, sgn = cfg["model"], cfg["spectrogram"], cfg.get("signal", {})
    model = CRNN2D_BiGRU(
        c_in=mcfg["in_channels"], base_ch=mcfg["base_ch"], fpn_ch=mcfg["fpn_ch"],
        rnn_hidden=mcfg["rnn_hidden"], rnn_layers=mcfg["rnn_layers"], bidirectional=mcfg["bidirectional"],
        bias_init_prior=mcfg.get("bias_init_prior", None), use_se=mcfg["use_se"],
        sfreq=cfg["data"]["sfreq"], n_fft=scfg["n_fft"], hop_length=scfg["hop_length"],
        win_length=scfg["win_length"], center=scfg["center"], power=scfg["power"],
        upsample_mode=mcfg.get("upsample_mode", "linear")
    ).to(device)

    # Optim & AMP
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["trainer"]["lr"], weight_decay=cfg["trainer"]["weight_decay"])
  #  scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda") and bool(cfg["trainer"]["amp"]))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda") and bool(cfg["trainer"]["amp"]))

    # Loss from losses.py (now works with tuple dataloaders)
    criterion = build_loss_function(cfg["loss"].get("name", "weighted_bce"), cfg["loss"], dl_tr)
    print(f"[loss] Using {cfg['loss'].get('name', 'weighted_bce')} from losses.py")

    state = TrainState()
    ckpt_path = os.path.join(cfg["paths"]["out_dir"], "best.pt")

    for epoch in range(1, cfg["trainer"]["epochs"] + 1):
        model.train()
        t0 = time.time()
        state.epoch = epoch
        for x, y in dl_tr:
            x = x.to(device)
            y = y.to(device)  # [B, T]

            logits = model(x)  # [B, T] - sample-level!

            opt.zero_grad(set_to_none=True)
            #with torch.amp.autocast(enabled=scaler.is_enabled()):
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):

                loss = criterion(logits.unsqueeze(1), y.unsqueeze(1))

            scaler.scale(loss).backward()
            if cfg["trainer"]["grad_clip"]:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["trainer"]["grad_clip"])
            scaler.step(opt)
            scaler.update()

            state.step += 1
            if state.step % cfg["log"]["every_steps"] == 0:
                wandb.log({"train/loss": loss.item(), "epoch": epoch, "step": state.step})

        # VAL
        _, _, m = run_eval(dl_va, cfg, model, device, split_name="val")
        dt = time.time() - t0

        log_payload = {
            "epoch": epoch,
            "val/threshold": m.get("threshold"),
            "val/precision": m.get("precision"),
            "val/recall": m.get("recall"),
            "val/f1": m.get("f1"),
            "val/accuracy": m.get("accuracy"),
            "val/TP": m.get("TP"),
            "val/TN": m.get("TN"),
            "val/FP": m.get("FP"),
            "val/FN": m.get("FN"),
            "val/roc_auc": m.get("roc_auc"),
            "val/pr_auc": m.get("pr_auc"),

        }
        wandb.log({k: v for k, v in log_payload.items() if v is not None})
        print(
            f"[val] epoch {epoch} {dt:.1f}s | thr={m.get('threshold', 0.5):.2f} F1={m.get('f1', 0):.3f} P={m.get('precision', 0):.3f} R={m.get('recall', 0):.3f} Acc={m.get('accuracy', 0):.3f}")

        current_key = m.get('f1', 0.0)
        if current_key > state.best:
            state.best = current_key
            torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt_path)
            print(f"[ckpt] saved best to {ckpt_path} (F1={current_key:.3f})")
            if wandb.run is not None:
                best_art = wandb.Artifact(name=f"{cfg['project']['name']}-best", type="model",
                                          metadata={"f1": float(current_key), "epoch": int(state.epoch)})
                best_art.add_file(ckpt_path)
                wandb.log_artifact(best_art)

    # TEST
    if dl_te is not None:
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"])
        _, _, m = run_eval(dl_te, cfg, model, device, split_name="test")
        test_payload = {
            "test/threshold": m.get("threshold"),
            "test/precision": m.get("precision"),
            "test/recall": m.get("recall"),
            "test/f1": m.get("f1"),
            "test/accuracy": m.get("accuracy"),
            "test/TP": m.get("TP"),
            "test/TN": m.get("TN"),
            "test/FP": m.get("FP"),
            "test/FN": m.get("FN"),
            "test/roc_auc": m.get("roc_auc"),
            "test/pr_auc": m.get("pr_auc"),

        }
        wandb.log({k: v for k, v in test_payload.items() if v is not None})
        print(
            f"[test] thr={m.get('threshold', 0.5):.2f} F1={m.get('f1', 0):.3f} P={m.get('precision', 0):.3f} R={m.get('recall', 0):.3f}")


# ----------------- CLI -----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--loss-name", type=str, help="Override loss (bce|weighted_bce|focal|dice)")
    ap.add_argument("--lr", type=float, help="Override learning rate")
    ap.add_argument("--wandb-entity", type=str, help="W&B entity/team")
    args = ap.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f) or {}
    cfg = safe_cfg(cfg)
    cfg = apply_overrides(cfg, args)

    train_and_eval(cfg)