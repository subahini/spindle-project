import os, math, json, argparse, time, random, csv
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Sampler
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


# ----------------- SAMPLERS (FIXED) -----------------
class UndersamplingSampler(Sampler):
    """Undersample majority class to balance dataset."""

    def __init__(self, labels, indices, target_ratio=1.0, seed=42):
        """
        Args:
            labels: Full label array (memmap) - shape (N, T)
            indices: Indices for this split (train/val/test)
            target_ratio: pos/neg ratio to achieve (1.0 = balanced)
            seed: Random seed
        """
        self.indices = np.asarray(indices)
        self.seed = seed

        # Calculate positive samples per window
        pos_counts = np.array([labels[i].sum() for i in self.indices])
        self.pos_mask = pos_counts > 0
        self.neg_mask = ~self.pos_mask

        n_pos = self.pos_mask.sum()
        n_neg = self.neg_mask.sum()

        print(f"[UndersamplingSampler] Pos windows: {n_pos}, Neg windows: {n_neg}")

        # Keep all positive samples
        self.pos_indices = self.indices[self.pos_mask]
        self.neg_indices = self.indices[self.neg_mask]

        # Undersample negatives
        self.n_neg_keep = int(n_pos / target_ratio)
        if self.n_neg_keep > n_neg:
            self.n_neg_keep = n_neg
            print(f"[UndersamplingSampler] Warning: Not enough negative samples. Using all {n_neg}")

        print(f"[UndersamplingSampler] Target: {len(self.pos_indices)} pos, {self.n_neg_keep} neg")

    def __iter__(self):
        # Shuffle and sample negatives each epoch
        rng = np.random.default_rng(self.seed + int(time.time()))

        # Sample negatives
        if self.n_neg_keep < len(self.neg_indices):
            neg_sampled = rng.choice(self.neg_indices, size=self.n_neg_keep, replace=False)
        else:
            neg_sampled = self.neg_indices

        # Combine and shuffle
        combined = np.concatenate([self.pos_indices, neg_sampled])
        rng.shuffle(combined)

        return iter(combined.tolist())

    def __len__(self):
        return len(self.pos_indices) + self.n_neg_keep


def build_weighted_sampler(labels, indices, seed=42):
    """
    Build WeightedRandomSampler based on positive sample frequency.
    Windows with more positive samples get higher weight.
    """
    indices = np.asarray(indices)

    # Calculate positive ratio per window
    pos_ratios = np.array([labels[i].mean() for i in indices])

    # Compute weights: inverse of class frequency
    # Windows with spindles (pos_ratio > 0) get higher weight
    weights = np.ones(len(indices), dtype=np.float32)

    pos_mask = pos_ratios > 0
    neg_mask = ~pos_mask

    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()

    if n_pos > 0 and n_neg > 0:
        # Weight inversely proportional to class frequency
        weights[pos_mask] = n_neg / n_pos
        weights[neg_mask] = 1.0

        # Normalize
        weights = weights / weights.sum() * len(weights)

    print(f"[WeightedSampler] Pos windows: {n_pos}, Neg windows: {n_neg}")
    if n_pos > 0:
        print(
            f"[WeightedSampler] Avg weight - pos: {weights[pos_mask].mean():.2f}, neg: {weights[neg_mask].mean():.2f}")

    return WeightedRandomSampler(
        weights=weights.tolist(),
        num_samples=len(indices),
        replacement=True
    )


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
        self.rnn = nn.GRU(fpn_ch, rnn_hidden, num_layers=rnn_layers, batch_first=False,
                          bidirectional=bidirectional)
        rnn_out = rnn_hidden * (2 if bidirectional else 1)
        self.head = nn.Conv1d(rnn_out, 1, 1)

        self.upsample_mode = upsample_mode

        if bias_init_prior is not None and 0 < bias_init_prior < 1:
            with torch.no_grad():
                self.head.bias.data.fill_(math.log(bias_init_prior / (1 - bias_init_prior)))

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        B, C, Traw = x_raw.shape
        S = self.spec(x_raw)
        f1 = self.b1(self.stem(S));
        f2 = self.b2(f1);
        f3 = self.b3(f2);
        p = self.fpn(f1, f2, f3);

        p = F.relu(self.post_fpn(p), inplace=True)
        p = self.freq_pool(p).squeeze(2)

        seq = p.permute(2, 0, 1)
        rnn_out, _ = self.rnn(seq)
        rnn_out = rnn_out.permute(1, 2, 0)
        logits = self.head(rnn_out)

        logits = F.interpolate(logits, size=Traw, mode=self.upsample_mode,
                               align_corners=False if self.upsample_mode != 'nearest' else None)
        return logits.squeeze(1)


# ----------------- Dataset -----------------
class EEGDataset(Dataset):

    def __init__(self, X_memmap, y_memmap, indices,
                 normalize="zscore", reference="car", channel_indices=None):
        self.X = X_memmap
        self.y = y_memmap
        self.indices = np.asarray(indices)
        self.normalize = normalize
        self.reference = reference
        self.channel_indices = channel_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])

        x = self.X[idx]
        y = self.y[idx]
        if self.channel_indices is not None:
            x = x[self.channel_indices, :]
        # referencing
        if self.reference == "car":
            x = x - x.mean(axis=0, keepdims=True)

        # normalization
        if self.normalize == "zscore":
            mu = x.mean(axis=-1, keepdims=True)
            sd = x.std(axis=-1, keepdims=True) + 1e-6
            x = (x - mu) / sd

        return torch.from_numpy(x.astype(np.float32)), \
            torch.from_numpy(y.astype(np.float32))


# ----------------- RAW EDF loading -----------------
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
    raw.set_eeg_reference("average", verbose=False)
    raw.filter(cfg_data["filter"]["low"], cfg_data["filter"]["high"], verbose=False)
    raw.pick(cfg_data["channels"])
    sf_target = cfg_data["sfreq"]
    if abs(raw.info["sfreq"] - sf_target) > 0.1:
        raw.resample(sfreq=sf_target, verbose=False)
    data = raw.get_data()
    C, T = data.shape
    y_full = _load_json_labels(json_path, T, raw.info["sfreq"])
    win = int(cfg_data["window_sec"] * raw.info["sfreq"])
    step = int(cfg_data["step_sec"] * raw.info["sfreq"])
    Xs, Ys = [], []
    for s in range(0, T - win + 1, step):
        e = s + win
        Xs.append(data[:, s:e])
        Ys.append(y_full[s:e])
    return np.stack(Xs, 0), np.stack(Ys, 0)


def split_data(n_samples, ratios=(0.7, 0.15, 0.15), seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n_samples)
    rng.shuffle(idx)

    n1 = int(ratios[0] * n_samples)
    n2 = int((ratios[0] + ratios[1]) * n_samples)

    idx_tr = idx[:n1]
    idx_va = idx[n1:n2]
    idx_te = idx[n2:]

    return idx_tr, idx_va, idx_te


# Metrics
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


def build_dataloaders(cfg):
    """Build dataloaders with FIXED samplers."""
    d = cfg["data"]
    sig = cfg.get("signal", {})
    bs = int(d.get("batch_size", cfg["trainer"]["batch_size"]))
    nw = int(d.get("num_workers", cfg["trainer"]["num_workers"]))
    seed = int(cfg.get("project", {}).get("seed", 42))

    # 1) Prefer NPY
    x_npy, y_npy = d.get("x_npy"), d.get("y_npy")
    if x_npy and y_npy and os.path.exists(x_npy) and os.path.exists(y_npy):
        X = np.load(x_npy, mmap_mode="r")
        y = np.load(y_npy, mmap_mode="r")
        channel_indices = [7, 8, 27, 28, 17, 18, 15, 13, 19, 14, 16,
                           44, 45, 46, 47, 30, 35, 31, 29]
    else:
        # 2) RAW fallback
        edf_dir = d["edf"]["dir"]
        edfs = sorted([f for f in os.listdir(edf_dir)
                       if f.lower().endswith(".edf")])
        if not edfs:
            raise RuntimeError(f"No EDF files found in {edf_dir}. Provide NPY or add EDFs.")
        X, y = load_edf_windows(os.path.join(edf_dir, edfs[0]),
                                d["edf"]["labels_json"], d)
        channel_indices = None

    n_samples = X.shape[0]

    # Split
    idx_tr, idx_va, idx_te = split_data(n_samples, seed=seed)

    # Build datasets
    ds_tr = EEGDataset(X, y, idx_tr,
                       normalize=sig.get("normalize", "zscore"),
                       reference=sig.get("reference", "car"),
                       channel_indices=channel_indices)
    ds_va = EEGDataset(X, y, idx_va,
                       normalize=sig.get("normalize", "zscore"),
                       reference=sig.get("reference", "car"),
                       channel_indices=channel_indices)
    ds_te = EEGDataset(X, y, idx_te,
                       normalize=sig.get("normalize", "zscore"),
                       reference=sig.get("reference", "car"),
                       channel_indices=channel_indices)

    # FIXED SAMPLER LOGIC
    sampler_mode = cfg["trainer"].get("sampler", "normal").lower()

    if sampler_mode == "undersample":
        print(f"[DataLoader] Using UndersamplingSampler")
        sampler = UndersamplingSampler(y, idx_tr, target_ratio=1.0, seed=seed)
        train_loader = DataLoader(ds_tr, batch_size=bs, sampler=sampler,
                                  num_workers=nw, pin_memory=True)
    elif sampler_mode == "weighted":
        print(f"[DataLoader] Using WeightedRandomSampler")
        sampler = build_weighted_sampler(y, idx_tr, seed=seed)
        train_loader = DataLoader(ds_tr, batch_size=bs, sampler=sampler,
                                  num_workers=nw, pin_memory=True)
    else:  # normal
        print(f"[DataLoader] Using normal shuffle")
        train_loader = DataLoader(ds_tr, batch_size=bs, shuffle=True,
                                  num_workers=nw, pin_memory=True)

    val_loader = DataLoader(ds_va, batch_size=bs, shuffle=False,
                            num_workers=nw, pin_memory=True)
    test_loader = DataLoader(ds_te, batch_size=bs, shuffle=False,
                             num_workers=nw, pin_memory=True)

    return ds_tr, ds_va, ds_te, train_loader, val_loader, test_loader


# ----------------- trainer & eval -----------------
@dataclass
class TrainState:
    step: int = 0
    best: float = -1.0
    epoch: int = 0


def run_eval(dloader, cfg, model, device, split_name="val", fixed_thr=None):
    """evaluating only using sklearn"""
    model.eval()
    all_probs = []
    all_y = []

    with torch.no_grad():
        for x, y in dloader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            all_y.append(y.detach().cpu())
            all_probs.append(probs.detach().cpu())

    probs = torch.cat(all_probs, dim=0).numpy()
    y_true = torch.cat(all_y, dim=0).numpy()

    y_true_flat = y_true.reshape(-1)
    y_score_flat = probs.reshape(-1)

    # Optional probability smoothing
    evcfg = cfg.get("eval", {})
    metric = evcfg.get("metric_for_best", "f1")

    smooth_sec = float(evcfg.get("smooth_sec", 0.0) or 0.0)
    if smooth_sec > 0:
        k = max(1, int(round(smooth_sec * cfg["data"].get("sfreq", 200))))
        if k > 1:
            kernel = np.ones(k, dtype=np.float32) / float(k)
            probs_sm = np.empty_like(probs, dtype=np.float32)
            for i in range(probs.shape[0]):
                probs_sm[i] = np.convolve(probs[i], kernel, mode="same")
            probs = probs_sm
        y_score_flat = probs.reshape(-1)

    if not _HAS_SKLEARN:
        print("Warning: sklearn not available, using default threshold 0.5")
        best_thr = 0.5
    else:
        if fixed_thr is not None:
            best_thr = float(fixed_thr)
        else:
            precision, recall, thresholds = precision_recall_curve(y_true_flat, y_score_flat)
            f1_scores = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
            best_idx = int(np.argmax(f1_scores)) if len(f1_scores) else 0
            best_thr = float(thresholds[best_idx]) if len(thresholds) else 0.5

    preds = (probs >= best_thr).astype(np.int64)
    best_m = basic_metrics(y_true, preds)
    best_m["threshold"] = best_thr

    if _HAS_SKLEARN:
        try:
            best_m["roc_auc"] = float(roc_auc_score(y_true_flat, y_score_flat))
            best_m["pr_auc"] = float(average_precision_score(y_true_flat, y_score_flat))
        except Exception as e:
            print(f"Warning: Could not compute AUC metrics: {e}")
            best_m["roc_auc"] = 0.0
            best_m["pr_auc"] = 0.0
    else:
        best_m["roc_auc"] = 0.0
        best_m["pr_auc"] = 0.0

    # Plot PR curve
    if plt is not None and wandb.run is not None and _HAS_SKLEARN:
        try:
            prec, rec, _ = precision_recall_curve(y_true_flat, y_score_flat)
            fig = plt.figure()
            plt.plot(rec, prec, linewidth=2, label=f"PR (AP={best_m['pr_auc']:.3f})")
            base = float(y_true_flat.mean())
            plt.hlines(base, 0, 1, linestyles="dashed", colors='gray', label=f"Baseline={base:.3f}")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.legend(loc="lower left")
            plt.title(f"Precision-Recall [{split_name}]")
            plt.grid(alpha=0.3)
            wandb.log({f"{split_name}/pr_curve": wandb.Image(fig)}, commit=False)
            plt.close(fig)
        except Exception as e:
            print(f"Warning: Could not plot PR curve: {e}")

    # Plot ROC curve
    if plt is not None and wandb.run is not None and _HAS_SKLEARN:
        try:
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_true_flat, y_score_flat)
            fig = plt.figure()
            plt.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC={best_m['roc_auc']:.3f})")
            plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color='gray', label="Random")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.legend(loc="lower right")
            plt.title(f"ROC Curve [{split_name}]")
            plt.grid(alpha=0.3)
            wandb.log({f"{split_name}/roc_curve": wandb.Image(fig)}, commit=False)
            plt.close(fig)
        except Exception as e:
            print(f"Warning: Could not plot ROC curve: {e}")

    # Plot confusion matrix
    if plt is not None and wandb.run is not None and _HAS_SKLEARN:
        try:
            from sklearn.metrics import confusion_matrix
            y_pred_flat = (y_score_flat >= best_thr).astype(np.int64)
            cm = confusion_matrix(y_true_flat, y_pred_flat)

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

    # Log metrics to wandb
    if wandb.run is not None:
        prevalence = float(y_true_flat.mean())
        wandb.log({
            f"{split_name}/prevalence": prevalence,
            f"{split_name}/threshold": best_m["threshold"],
            f"{split_name}/precision": best_m["precision"],
            f"{split_name}/recall": best_m["recall"],
            f"{split_name}/f1": best_m["f1"],
            f"{split_name}/accuracy": best_m["accuracy"],
            f"{split_name}/roc_auc": best_m["roc_auc"],
            f"{split_name}/pr_auc": best_m["pr_auc"],
            f"{split_name}/TP": best_m["TP"],
            f"{split_name}/TN": best_m["TN"],
            f"{split_name}/FP": best_m["FP"],
            f"{split_name}/FN": best_m["FN"],
        }, commit=False)

    return probs, y_true, best_m


def train_and_eval(cfg):
    device = torch.device(cfg["project"].get("device", "cuda") if torch.cuda.is_available() else "cpu")
    set_seed(cfg["project"].get("seed", 42))
    ensure_dir(cfg["paths"]["out_dir"])

    # W&B
    wb = cfg.get("logging", {}).get("wandb", {})
    if wb.get("enabled", True):
        wandb.init(
            project=wb.get("project", cfg["project"].get("name")),
            entity=wb.get("entity", cfg["project"].get("entity")),
            name=wb.get("name", cfg["project"].get("name")),
            tags=wb.get("tags", []),
            config=cfg
        )

        # Log sampler type at start
        wandb.config.update({
            "sampler": cfg["trainer"].get("sampler", "normal"),
            "loss_name": cfg["loss"].get("name"),
            "lr": cfg["trainer"].get("lr"),
            "batch_size": cfg["trainer"].get("batch_size"),
        }, allow_val_change=True)

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
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda") and bool(cfg["trainer"]["amp"]))

    # Loss
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
            y = y.to(device)

            logits = model(x)

            opt.zero_grad(set_to_none=True)
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
            torch.save({"model": model.state_dict(), "cfg": cfg, "best_val_thr": float(m.get("threshold", 0.5)),
                        "best_val_epoch": int(epoch)}, ckpt_path)
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
            best_val_thr = float(ckpt.get("best_val_thr", 0.5))
        else:
            best_val_thr = 0.5
        _, _, m = run_eval(dl_te, cfg, model, device, split_name="test", fixed_thr=best_val_thr)
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