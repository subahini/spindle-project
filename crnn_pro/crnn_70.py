#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
crnn_70.py (FULL, COMPLETE, UPDATED — logs like Crnn.py)

✅ raw_cache input: root/<recording>/{data.dat, labels.dat, meta.json}
✅ TIME split inside each recording (no leakage with overlap):
    train = first 70% time
    val   = next 15%
    test  = last 15%
✅ EEG channel picking at runtime from meta.json channels + config signal.channels
✅ Runtime preprocessing (NOT during cache creation):
    signal.reference: car|none
    signal.normalize: zscore|none
✅ Logs EVERYTHING to W&B for train/val/test:
    - train/loss + lr + step + epoch
    - val/* metrics each epoch (precision/recall/f1/accuracy/TP/TN/FP/FN/prevalence/pr_auc/roc_auc/threshold)
    - val PR curve, ROC curve, confusion matrix (subsampled)
    - saves BEST checkpoint by VAL F1 + stores best_val_thr + best epoch
    - test/* metrics + curves using best_val_thr (fixed threshold)
✅ Atomic checkpoint save (prevents partial-write corruption)
✅ Works with sweepTrainer70 overrides

Important:
- Curve plotting requires sklearn; if missing, we still log metrics and use thr=0.5.
- PR/ROC plots can be heavy; we subsample max_plot_points for plots.

Exports:
- safe_cfg(cfg)
- train_and_eval(cfg)

Run:
  python crnn_70.py --config config70.yaml
"""

import os
import json
import time
import math
import argparse
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import wandb

# must exist in your project
from losses import build_loss_function

# sklearn for threshold search + AUCs + curves
try:
    from sklearn.metrics import (
        precision_recall_curve,
        average_precision_score,
        roc_auc_score,
    )
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


# -----------------------------
# Defaults / Utils
# -----------------------------
EEG_19_DEFAULT = [
    "Fp1","Fp2","F3","F4","F7","F8","Fz",
    "C3","C4",
    "P3","P4","Pz",
    "O1","O2","Oz",
    "T3","T4","T5","T6"
]

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def atomic_torch_save(obj: Dict[str, Any], path: str):
    """Write temp then rename (atomic) to avoid partial/corrupt checkpoints on network FS."""
    ensure_dir(os.path.dirname(path) or ".")
    tmp = path + ".tmp"
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        raise

def safe_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = cfg or {}

    cfg.setdefault("project", {})
    cfg["project"].setdefault("name", "crnn_pro")
    cfg["project"].setdefault("seed", 42)
    cfg["project"].setdefault("device", "cuda")
    cfg["project"].setdefault("entity", None)

    cfg.setdefault("paths", {})
    cfg["paths"].setdefault("out_dir", "./checkpoints")
    cfg["paths"].setdefault("save_ckpt", True)
    cfg["paths"].setdefault("log_best_artifact", False)  # set True if you want artifact uploads

    cfg.setdefault("data", {})
    cfg["data"].setdefault("use_raw_cache", True)
    cfg["data"].setdefault("raw_cache", {})
    cfg["data"]["raw_cache"].setdefault("root", None)
    cfg["data"]["raw_cache"].setdefault("split_mode", "time")  # time expected
    cfg["data"]["raw_cache"].setdefault("split_ratios", [0.70, 0.15, 0.15])
    cfg["data"]["raw_cache"].setdefault("split_seed", cfg["project"]["seed"])
    cfg["data"].setdefault("sfreq", 200.0)
    cfg["data"].setdefault("window_sec", 5.0)
    cfg["data"].setdefault("step_sec", 1.0)
    cfg["data"].setdefault("batch_size", 32)
    cfg["data"].setdefault("num_workers", 4)

    cfg.setdefault("signal", {})
    cfg["signal"].setdefault("channels", EEG_19_DEFAULT[:])  # EEG picking
    cfg["signal"].setdefault("reference", "none")           # car|none
    cfg["signal"].setdefault("normalize", "none")           # zscore|none

    cfg.setdefault("spectrogram", {})
    cfg["spectrogram"].setdefault("n_fft", 128)
    cfg["spectrogram"].setdefault("hop_length", 20)
    cfg["spectrogram"].setdefault("win_length", 128)
    cfg["spectrogram"].setdefault("center", True)
    cfg["spectrogram"].setdefault("power", 2.0)

    cfg.setdefault("model", {})
    cfg["model"].setdefault("in_channels", len(cfg["signal"]["channels"]))
    cfg["model"].setdefault("base_ch", 32)
    cfg["model"].setdefault("fpn_ch", 64)
    cfg["model"].setdefault("rnn_hidden", 128)
    cfg["model"].setdefault("rnn_layers", 2)
    cfg["model"].setdefault("bidirectional", True)
    cfg["model"].setdefault("use_se", True)
    cfg["model"].setdefault("bias_init_prior", 0.02)
    cfg["model"].setdefault("upsample_mode", "linear")

    cfg.setdefault("trainer", {})
    cfg["trainer"].setdefault("epochs", 70)
    cfg["trainer"].setdefault("batch_size", cfg["data"].get("batch_size", 32))
    cfg["trainer"].setdefault("lr", 2.1e-4)
    cfg["trainer"].setdefault("weight_decay", 0.01)
    cfg["trainer"].setdefault("amp", True)
    cfg["trainer"].setdefault("grad_clip", 1.0)
    cfg["trainer"].setdefault("num_workers", cfg["data"].get("num_workers", 4))
    cfg["trainer"].setdefault("sampler", "normal")

    cfg.setdefault("loss", {})
    cfg["loss"].setdefault("name", "focal")

    cfg.setdefault("eval", {})
    cfg["eval"].setdefault("smooth_sec", 0.0)           # optional probability smoothing (seconds)
    cfg["eval"].setdefault("fixed_threshold", None)     # if set, skips threshold search
    cfg["eval"].setdefault("max_plot_points", 300000)   # subsample for W&B curve plots

    cfg.setdefault("logging", {})
    cfg["logging"].setdefault("wandb", {})
    cfg["logging"]["wandb"].setdefault("enabled", True)
    cfg["logging"]["wandb"].setdefault("project", "spindle-project-crnn_pro")
    cfg["logging"]["wandb"].setdefault("entity", cfg["project"].get("entity"))

    cfg.setdefault("log", {})
    cfg["log"].setdefault("every_steps", 50)

    # keep consistent: model.in_channels must match EEG list length
    cfg["model"]["in_channels"] = len(cfg["signal"]["channels"])

    return cfg


# -----------------------------
# raw_cache helpers
# -----------------------------
def _list_recording_dirs(raw_root: str) -> List[str]:
    if not raw_root or not os.path.isdir(raw_root):
        raise RuntimeError(f"[raw_cache] root not found: {raw_root}")

    recs = []
    for name in sorted(os.listdir(raw_root)):
        p = os.path.join(raw_root, name)
        if not os.path.isdir(p):
            continue
        if (os.path.exists(os.path.join(p, "data.dat")) and
            os.path.exists(os.path.join(p, "labels.dat")) and
            os.path.exists(os.path.join(p, "meta.json"))):
            recs.append(p)

    if not recs:
        raise RuntimeError(
            f"[raw_cache] No valid recordings in: {raw_root} "
            f"(need data.dat + labels.dat + meta.json)"
        )
    return recs


# -----------------------------
# Dataset: time segment + EEG pick + runtime CAR/zscore
# -----------------------------
class RawWindowDataset(Dataset):
    """
    Windowing on-the-fly from raw_cache.

    Time split:
      Uses only time fraction [t_start_frac, t_end_frac] within each recording.

    Channel pick:
      Uses meta.json["channels"] or meta.json["ch_names"].

    Runtime preprocess:
      reference=car and/or normalize=zscore applied here.
    """
    def __init__(self,
                 recording_dirs: List[str],
                 window_sec: float,
                 step_sec: float,
                 sfreq: float,
                 normalize: str = "none",
                 reference: str = "none",
                 desired_channels: Optional[List[str]] = None,
                 t_start_frac: float = 0.0,
                 t_end_frac: float = 1.0):

        self.recording_dirs = list(recording_dirs)
        if len(self.recording_dirs) == 0:
            raise RuntimeError("[RawWindowDataset] Got 0 recordings for this split.")

        self.window_sec = float(window_sec)
        self.step_sec = float(step_sec)
        self.sfreq = float(sfreq)
        self.normalize = str(normalize).lower()
        self.reference = str(reference).lower()

        self.desired_channels = desired_channels
        self.pick_idx: Optional[List[int]] = None
        self.pick_names: Optional[List[str]] = None

        self.t_start_frac = float(t_start_frac)
        self.t_end_frac = float(t_end_frac)
        if not (0.0 <= self.t_start_frac < self.t_end_frac <= 1.0):
            raise ValueError(f"[RawWindowDataset] invalid time frac: {self.t_start_frac}..{self.t_end_frac}")

        self.win = int(round(self.window_sec * self.sfreq))
        self.step = int(round(self.step_sec * self.sfreq))
        if self.win <= 0 or self.step <= 0:
            raise ValueError(f"[RawWindowDataset] invalid win/step: win={self.win}, step={self.step}")

        self.recs: List[Dict[str, Any]] = []
        self._mm_cache: Dict[str, Any] = {}

        for p in self.recording_dirs:
            meta_path = os.path.join(p, "meta.json")
            with open(meta_path, "r") as f:
                meta = json.load(f)

            if "shapes" not in meta or "data" not in meta["shapes"]:
                raise RuntimeError(f"[RawWindowDataset] meta.json in {p} missing shapes.data")

            C_full, T = int(meta["shapes"]["data"][0]), int(meta["shapes"]["data"][1])

            ch_names = meta.get("channels") or meta.get("ch_names")
            if ch_names is None:
                raise RuntimeError(f"[RawWindowDataset] meta.json in {p} must contain 'channels' or 'ch_names'")
            if len(ch_names) != C_full:
                raise RuntimeError(
                    f"[RawWindowDataset] channel list length mismatch in {p}: "
                    f"len(ch_names)={len(ch_names)} vs C_full={C_full}"
                )

            # Channel picking indices (enforce consistent order across recordings)
            if self.desired_channels is not None:
                ch_map = {c.strip(): i for i, c in enumerate(ch_names)}
                missing = [c for c in self.desired_channels if c not in ch_map]
                if missing:
                    raise RuntimeError(f"[RawWindowDataset] Missing channels in {p}: {missing}\nAvailable: {ch_names}")

                pick_idx = [ch_map[c] for c in self.desired_channels]
                pick_names = [ch_names[i] for i in pick_idx]

                if self.pick_names is None:
                    self.pick_idx = pick_idx
                    self.pick_names = pick_names
                else:
                    if pick_names != self.pick_names:
                        raise RuntimeError(
                            "[RawWindowDataset] Channel order differs across recordings.\n"
                            f"First: {self.pick_names}\nThis:  {pick_names}\nRec: {p}"
                        )
                _ = len(self.pick_idx)

            # Time segment bounds
            seg_start = int(round(self.t_start_frac * T))
            seg_end = int(round(self.t_end_frac * T))
            seg_start = max(0, min(seg_start, T))
            seg_end = max(0, min(seg_end, T))

            # Windows fully inside segment
            nwin = 0
            if (seg_end - seg_start) >= self.win:
                nwin = 1 + (seg_end - seg_start - self.win) // self.step

            self.recs.append({
                "dir": p,
                "name": os.path.basename(p),
                "C_full": C_full,
                "T": T,
                "seg_start": seg_start,
                "seg_end": seg_end,
                "nwin": int(nwin),
            })

        if len(self.recs) == 0:
            raise RuntimeError("[RawWindowDataset] No valid recordings after meta parsing.")

        # Verify consistent C_full
        C0_full = self.recs[0]["C_full"]
        bad = [r for r in self.recs if r["C_full"] != C0_full]
        if bad:
            raise RuntimeError(
                f"[RawWindowDataset] C_full mismatch across recordings. "
                f"Expected {C0_full}, got {[(r['name'], r['C_full']) for r in bad]}"
            )

        self.cum = np.cumsum([r["nwin"] for r in self.recs], dtype=np.int64)
        self.total = int(self.cum[-1]) if len(self.cum) else 0
        if self.total <= 0:
            raise RuntimeError(
                f"[RawWindowDataset] No windows possible in this time segment "
                f"(win={self.win}, step={self.step})."
            )

    def __len__(self):
        return self.total

    def _open_recording(self, rec_dir: str, C_full: int, T: int):
        if rec_dir in self._mm_cache:
            return self._mm_cache[rec_dir]["X"], self._mm_cache[rec_dir]["y"]
        X = np.memmap(os.path.join(rec_dir, "data.dat"), mode="r", dtype=np.float32, shape=(C_full, T))
        y = np.memmap(os.path.join(rec_dir, "labels.dat"), mode="r", dtype=np.uint8, shape=(T,))
        self._mm_cache[rec_dir] = {"X": X, "y": y}
        return X, y

    def __getitem__(self, idx: int):
        idx = int(idx)
        ridx = int(np.searchsorted(self.cum, idx, side="right"))
        prev = int(self.cum[ridx - 1]) if ridx > 0 else 0
        local = idx - prev

        rec = self.recs[ridx]
        if local < 0 or local >= rec["nwin"]:
            raise IndexError("RawWindowDataset index out of range")

        start = int(rec["seg_start"] + local * self.step)
        end = start + self.win
        if end > rec["seg_end"]:
            raise IndexError("Window exceeds segment end (should not happen)")

        X_mm, y_mm = self._open_recording(rec["dir"], rec["C_full"], rec["T"])

        x = X_mm[:, start:end]  # (C_full, win)
        if self.pick_idx is not None:
            x = x[self.pick_idx, :]  # (C_pick, win)

        y = y_mm[start:end]  # (win,)

        # Runtime preprocessing
        if self.reference == "car":
            x = x - x.mean(axis=0, keepdims=True)

        if self.normalize == "zscore":
            mu = x.mean(axis=-1, keepdims=True)
            sd = x.std(axis=-1, keepdims=True) + 1e-6
            x = (x - mu) / sd

        return (
            torch.from_numpy(np.asarray(x, dtype=np.float32)),
            torch.from_numpy(np.asarray(y, dtype=np.float32)),
        )


# -----------------------------
# Spectrogram + Model
# -----------------------------
class Spectrogram(nn.Module):
    def __init__(self, n_fft: int, hop_length: int, win_length: int, center: bool = True, power: float = 2.0):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop = int(hop_length)
        self.win = int(win_length)
        self.center = bool(center)
        self.power = float(power)
        self.register_buffer("window", torch.hann_window(self.win), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,T)
        B, C, T = x.shape
        X = torch.stft(
            x.reshape(B * C, T),
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win,
            center=self.center,
            window=self.window,
            return_complex=True,
        ).abs()
        if self.power != 1.0:
            X = X.pow(self.power)
        return X.reshape(B, C, X.shape[-2], X.shape[-1])  # (B,C,F,TT)

class SE2d(nn.Module):
    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        mid = max(1, ch // r)
        self.fc1 = nn.Conv2d(ch, mid, 1)
        self.fc2 = nn.Conv2d(mid, ch, 1)

    def forward(self, x):
        s = x.mean(dim=(2, 3), keepdim=True)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1, se: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.se = SE2d(out_ch) if se else nn.Identity()

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        return self.se(x)

class MultiScaleStem(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, se: bool = True):
        super().__init__()
        mid = max(1, out_ch // 3)
        self.b1 = ConvBNReLU(in_ch, mid, k=3, p=1, se=se)
        self.b2 = ConvBNReLU(in_ch, mid, k=5, p=2, se=se)
        self.b3 = ConvBNReLU(in_ch, mid, k=7, p=3, se=se)
        self.fuse = ConvBNReLU(mid * 3, out_ch, k=1, p=0, se=False)

    def forward(self, x):
        return self.fuse(torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1))

class FPNLite(nn.Module):
    def __init__(self, c1: int, c2: int, c3: int, out_ch: int):
        super().__init__()
        self.l1 = nn.Conv2d(c1, out_ch, 1)
        self.l2 = nn.Conv2d(c2, out_ch, 1)
        self.l3 = nn.Conv2d(c3, out_ch, 1)

    @staticmethod
    def up2xF(x):
        return F.interpolate(x, scale_factor=(2, 1), mode="bilinear", align_corners=False)

    def forward(self, f1, f2, f3):
        p3 = self.l3(f3)
        p2 = self.l2(f2) + self.up2xF(p3)
        p1 = self.l1(f1) + self.up2xF(p2)
        return p1

class CRNN2D_BiGRU(nn.Module):
    def __init__(self,
                 c_in: int,
                 base_ch: int,
                 fpn_ch: int,
                 rnn_hidden: int,
                 rnn_layers: int,
                 bidirectional: bool,
                 use_se: bool,
                 spec_cfg: Dict[str, Any],
                 bias_init_prior: Optional[float] = None,
                 upsample_mode: str = "linear"):
        super().__init__()
        self.spec = Spectrogram(
            n_fft=spec_cfg["n_fft"],
            hop_length=spec_cfg["hop_length"],
            win_length=spec_cfg["win_length"],
            center=spec_cfg["center"],
            power=spec_cfg["power"],
        )
        self.stem = MultiScaleStem(c_in, base_ch, se=use_se)

        self.b1 = nn.Sequential(
            ConvBNReLU(base_ch, base_ch, se=use_se),
            ConvBNReLU(base_ch, base_ch, se=use_se),
            nn.AvgPool2d((2, 1)),
        )
        self.b2 = nn.Sequential(
            ConvBNReLU(base_ch, base_ch * 2, se=use_se),
            nn.AvgPool2d((2, 1)),
        )
        self.b3 = nn.Sequential(
            ConvBNReLU(base_ch * 2, base_ch * 4, se=use_se),
            nn.AvgPool2d((2, 1)),
        )

        self.fpn = FPNLite(base_ch, base_ch * 2, base_ch * 4, fpn_ch)
        self.post = nn.Conv2d(fpn_ch, fpn_ch, 1)
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

        self.rnn = nn.GRU(
            input_size=fpn_ch,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=False,
            bidirectional=bidirectional,
        )
        rnn_out = rnn_hidden * (2 if bidirectional else 1)
        self.head = nn.Conv1d(rnn_out, 1, 1)
        self.upsample_mode = upsample_mode

        if bias_init_prior is not None and 0 < bias_init_prior < 1:
            with torch.no_grad():
                self.head.bias.data.fill_(math.log(bias_init_prior / (1 - bias_init_prior)))

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        # x_raw: (B,C,T)
        B, C, T = x_raw.shape
        S = self.spec(x_raw)               # (B,C,F,TT)

        f1 = self.b1(self.stem(S))
        f2 = self.b2(f1)
        f3 = self.b3(f2)

        p = self.fpn(f1, f2, f3)           # (B,fpn_ch,*,TT)
        p = F.relu(self.post(p), inplace=True)
        p = self.freq_pool(p).squeeze(2)   # (B,fpn_ch,TT)

        seq = p.permute(2, 0, 1)           # (TT,B,fpn_ch)
        rnn_out, _ = self.rnn(seq)
        rnn_out = rnn_out.permute(1, 2, 0) # (B,rnn_out,TT)

        logits = self.head(rnn_out)        # (B,1,TT)
        logits = F.interpolate(
            logits, size=T, mode=self.upsample_mode,
            align_corners=False if self.upsample_mode != "nearest" else None
        )
        return logits.squeeze(1)           # (B,T)


# -----------------------------
# Eval helpers (F1 + best threshold + W&B PR/ROC/CM)
# -----------------------------
def basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(np.uint8).reshape(-1)
    y_pred = y_pred.astype(np.uint8).reshape(-1)
    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())
    p = TP / (TP + FP + 1e-8)
    r = TP / (TP + FN + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    acc = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "accuracy": float(acc),
        "TP": TP, "TN": TN, "FP": FP, "FN": FN
    }

def _smooth_probs_time(probs_2d: np.ndarray, smooth_samples: int) -> np.ndarray:
    if smooth_samples <= 1:
        return probs_2d
    kernel = np.ones(smooth_samples, dtype=np.float32) / float(smooth_samples)
    out = np.empty_like(probs_2d, dtype=np.float32)
    for i in range(probs_2d.shape[0]):
        out[i] = np.convolve(probs_2d[i], kernel, mode="same")
    return out

def _maybe_subsample_for_plots(cfg: Dict[str, Any], y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray):
    maxn = int(cfg.get("eval", {}).get("max_plot_points", 300000))
    n = y_true.size
    if n <= maxn:
        return y_true, y_score, y_pred
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=maxn, replace=False)
    return y_true[idx], y_score[idx], y_pred[idx]

def wandb_log_curves(split_name: str, y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray):
    """Logs confusion matrix + PR/ROC curves to W&B (best-effort)."""
    if wandb.run is None:
        return

    # confusion matrix widget
    try:
        wandb.log({
            f"{split_name}/confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true,
                preds=y_pred,
                class_names=["non_spindle", "spindle"]
            )
        }, commit=False)
    except Exception:
        pass

    # PR/ROC curve widgets (best-effort; depends on wandb version)
    try:
        wandb.log({
            f"{split_name}/pr_curve": wandb.plot.pr_curve(
                y_true=y_true,
                y_probas=y_score,
                labels=["non_spindle", "spindle"]
            )
        }, commit=False)
    except Exception:
        pass

    try:
        wandb.log({
            f"{split_name}/roc_curve": wandb.plot.roc_curve(
                y_true=y_true,
                y_probas=y_score,
                labels=["non_spindle", "spindle"]
            )
        }, commit=False)
    except Exception:
        pass


@torch.no_grad()
def run_eval(dl: DataLoader,
             cfg: Dict[str, Any],
             model: nn.Module,
             device: torch.device,
             split_name: str,
             fixed_thr: Optional[float] = None,
             epoch: Optional[int] = None) -> Dict[str, Any]:
    """
    Sample-level eval across all windows (flattened).
    - if fixed_thr is None: choose threshold maximizing F1 on PR curve (if sklearn available).
    - logs W&B curves + all metrics under {split_name}/*
    """
    model.eval()

    all_probs = []
    all_y = []

    for x, y in dl:
        x = x.to(device, non_blocking=True)      # (B,C,T)
        y = y.to(device, non_blocking=True)      # (B,T)
        logits = model(x)                        # (B,T)
        probs = torch.sigmoid(logits).float()    # (B,T)
        all_probs.append(probs.detach().cpu().numpy().astype(np.float32))
        all_y.append(y.detach().cpu().numpy().astype(np.float32))

    probs_2d = np.concatenate(all_probs, axis=0)  # (N,T)
    y_2d = np.concatenate(all_y, axis=0)          # (N,T)

    # Optional smoothing over time
    smooth_sec = float(cfg.get("eval", {}).get("smooth_sec", 0.0))
    if smooth_sec > 0:
        sfreq = float(cfg["data"]["sfreq"])
        k = max(1, int(round(smooth_sec * sfreq)))
        probs_2d = _smooth_probs_time(probs_2d, k)

    y_true = y_2d.reshape(-1).astype(np.uint8)
    y_score = probs_2d.reshape(-1).astype(np.float32)

    # Choose threshold
    if fixed_thr is None:
        fixed_thr = cfg.get("eval", {}).get("fixed_threshold", None)

    if fixed_thr is not None:
        thr = float(fixed_thr)
    else:
        thr = 0.5
        if _HAS_SKLEARN:
            try:
                prec, rec, thrs = precision_recall_curve(y_true, y_score)
                f1s = 2 * prec * rec / (prec + rec + 1e-8)
                best_i = int(np.nanargmax(f1s))
                if len(thrs) > 0:
                    if best_i >= len(thrs):
                        best_i = len(thrs) - 1
                    thr = float(thrs[max(best_i, 0)])
            except Exception:
                thr = 0.5

    y_pred = (y_score >= thr).astype(np.uint8)

    m = basic_metrics(y_true, y_pred)
    m["threshold"] = float(thr)
    m["prevalence"] = float(y_true.mean())

    # AUCs
    if _HAS_SKLEARN:
        try:
            m["pr_auc"] = float(average_precision_score(y_true, y_score))
        except Exception:
            m["pr_auc"] = float("nan")
        try:
            m["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            m["roc_auc"] = float("nan")
    else:
        m["pr_auc"] = float("nan")
        m["roc_auc"] = float("nan")

    # W&B plots + full metric logging (like Crnn.py)
    if wandb.run is not None:
        # subsample for plots
        y_t_p, y_s_p, y_p_p = _maybe_subsample_for_plots(cfg, y_true, y_score, y_pred)
        wandb_log_curves(split_name, y_t_p, y_s_p, y_p_p)

        payload = {
            f"{split_name}/prevalence": m["prevalence"],
            f"{split_name}/threshold": m["threshold"],
            f"{split_name}/precision": m["precision"],
            f"{split_name}/recall": m["recall"],
            f"{split_name}/f1": m["f1"],
            f"{split_name}/accuracy": m["accuracy"],
            f"{split_name}/pr_auc": m.get("pr_auc", float("nan")),
            f"{split_name}/roc_auc": m.get("roc_auc", float("nan")),
            f"{split_name}/TP": m["TP"],
            f"{split_name}/TN": m["TN"],
            f"{split_name}/FP": m["FP"],
            f"{split_name}/FN": m["FN"],
        }
        if epoch is not None:
            payload["epoch"] = int(epoch)
        wandb.log(payload, commit=False)

    return {"metrics": m, "threshold": thr}


# -----------------------------
# Training
# -----------------------------
@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0
    best_f1: float = -1.0
    best_thr: float = 0.5
    best_epoch: int = -1

def build_dataloaders(cfg: Dict[str, Any]):
    d = cfg["data"]
    rc = d["raw_cache"]
    sig = cfg["signal"]

    raw_root = rc.get("root")
    rec_dirs = _list_recording_dirs(raw_root)

    ratios = rc.get("split_ratios", [0.70, 0.15, 0.15])
    r0, r1, r2 = float(ratios[0]), float(ratios[1]), float(ratios[2])
    s = r0 + r1 + r2
    r0, r1, r2 = r0 / s, r1 / s, r2 / s

    mode = str(rc.get("split_mode", "time")).lower()
    if mode != "time":
        raise RuntimeError("This crnn_70.py is set up for split_mode: time (as you requested).")

    t0, t1, t2, t3 = 0.0, r0, r0 + r1, 1.0
    print(f"[time split] train={t0:.2f}-{t1:.2f} val={t1:.2f}-{t2:.2f} test={t2:.2f}-{t3:.2f}")
    print(f"[raw_cache] recordings={len(rec_dirs)} -> using ALL recordings in each split, different time segments")

    desired = sig.get("channels", None)

    ds_tr = RawWindowDataset(
        rec_dirs, d["window_sec"], d["step_sec"], d["sfreq"],
        normalize=sig.get("normalize", "none"),
        reference=sig.get("reference", "none"),
        desired_channels=desired,
        t_start_frac=t0, t_end_frac=t1
    )
    ds_va = RawWindowDataset(
        rec_dirs, d["window_sec"], d["step_sec"], d["sfreq"],
        normalize=sig.get("normalize", "none"),
        reference=sig.get("reference", "none"),
        desired_channels=desired,
        t_start_frac=t1, t_end_frac=t2
    )
    ds_te = RawWindowDataset(
        rec_dirs, d["window_sec"], d["step_sec"], d["sfreq"],
        normalize=sig.get("normalize", "none"),
        reference=sig.get("reference", "none"),
        desired_channels=desired,
        t_start_frac=t2, t_end_frac=t3
    )

    if ds_tr.pick_names is not None:
        print("[channels picked]", ds_tr.pick_names)

    bs = int(cfg.get("trainer", {}).get("batch_size", cfg.get("data", {}).get("batch_size", 32)))
    nw = int(cfg.get("trainer", {}).get("num_workers", cfg.get("data", {}).get("num_workers", 4)))

    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    print(f"[windows] train={len(ds_tr)} val={len(ds_va)} test={len(ds_te)}")
    return ds_tr, ds_va, ds_te, dl_tr, dl_va, dl_te


def run_one_epoch(model, dl, device, opt, scaler, criterion, cfg, state: TrainState):
    model.train()
    log_every = int(cfg["log"]["every_steps"])

    for x, y in dl:
        x = x.to(device, non_blocking=True)  # (B,C,T)
        y = y.to(device, non_blocking=True)  # (B,T)

        opt.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            logits = model(x)  # (B,T)
            loss = criterion(logits.unsqueeze(1), y.unsqueeze(1))

        scaler.scale(loss).backward()
        if float(cfg["trainer"].get("grad_clip", 0.0)) > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["trainer"]["grad_clip"]))
        scaler.step(opt)
        scaler.update()

        state.step += 1
        if wandb.run is not None and (state.step % log_every == 0):
            lr = float(opt.param_groups[0]["lr"])
            wandb.log({
                "train/loss": float(loss.item()),
                "train/lr": lr,
                "step": state.step,
                "epoch": state.epoch,
            }, commit=False)


def train_and_eval(cfg: Dict[str, Any]):
    cfg = safe_cfg(cfg)
    set_seed(int(cfg["project"]["seed"]))

    device = torch.device(cfg["project"]["device"] if torch.cuda.is_available() else "cpu")

    # W&B init (only if not already initialized by sweepTrainer)
    wb = cfg.get("logging", {}).get("wandb", {})
    if wb.get("enabled", True) and wandb.run is None:
        wandb.init(
            project=wb.get("project", cfg["project"]["name"]),
            entity=wb.get("entity", cfg["project"].get("entity")),
            name=os.getenv("WANDB_NAME", None),
            config=cfg
        )

    ds_tr, ds_va, ds_te, dl_tr, dl_va, dl_te = build_dataloaders(cfg)

    mcfg = cfg["model"]
    scfg = cfg["spectrogram"]

    model = CRNN2D_BiGRU(
        c_in=int(mcfg["in_channels"]),
        base_ch=int(mcfg["base_ch"]),
        fpn_ch=int(mcfg["fpn_ch"]),
        rnn_hidden=int(mcfg["rnn_hidden"]),
        rnn_layers=int(mcfg["rnn_layers"]),
        bidirectional=bool(mcfg["bidirectional"]),
        use_se=bool(mcfg["use_se"]),
        spec_cfg=scfg,
        bias_init_prior=mcfg.get("bias_init_prior", None),
        upsample_mode=str(mcfg.get("upsample_mode", "linear")),
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["trainer"]["lr"]),
        weight_decay=float(cfg["trainer"]["weight_decay"])
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda") and bool(cfg["trainer"]["amp"])
    )

    criterion = build_loss_function(cfg["loss"].get("name", "weighted_bce"), cfg["loss"], dl_tr)

    state = TrainState()

    out_dir = cfg["paths"]["out_dir"]
    ensure_dir(out_dir)
    ckpt_path = os.path.join(out_dir, "best.pt")

    for epoch in range(1, int(cfg["trainer"]["epochs"]) + 1):
        state.epoch = epoch
        t0 = time.time()

        run_one_epoch(model, dl_tr, device, opt, scaler, criterion, cfg, state)

        # VAL eval: threshold chosen to maximize F1 (unless fixed_threshold set)
        ev = run_eval(dl_va, cfg, model, device, split_name="val", fixed_thr=None, epoch=epoch)
        m = ev["metrics"]
        dt = time.time() - t0

        print(
            f"[epoch {epoch}] time={dt:.1f}s  "
            f"val_f1={m['f1']:.4f} thr={m['threshold']:.3f} "
            f"pr_auc={m.get('pr_auc', float('nan')):.4f} roc_auc={m.get('roc_auc', float('nan')):.4f}"
        )

        # Save best by VAL F1
        if float(m["f1"]) > float(state.best_f1):
            state.best_f1 = float(m["f1"])
            state.best_thr = float(m["threshold"])
            state.best_epoch = int(epoch)

            if bool(cfg["paths"].get("save_ckpt", True)):
                payload = {
                    "model": model.state_dict(),
                    "cfg": cfg,
                    "best_val_f1": state.best_f1,
                    "best_val_thr": state.best_thr,
                    "best_val_epoch": state.best_epoch,
                }
                atomic_torch_save(payload, ckpt_path)
                print(f"[ckpt] saved best -> {ckpt_path} (F1={state.best_f1:.4f}, thr={state.best_thr:.3f})")

                # Optional artifact upload
                if wandb.run is not None and bool(cfg["paths"].get("log_best_artifact", False)):
                    try:
                        art = wandb.Artifact(
                            name=f"{cfg['project']['name']}-best",
                            type="model",
                            metadata={"f1": state.best_f1, "epoch": state.best_epoch, "thr": state.best_thr}
                        )
                        art.add_file(ckpt_path)
                        wandb.log_artifact(art)
                    except Exception:
                        pass

            # Log best summary like Crnn.py
            if wandb.run is not None:
                wandb.log({
                    "best/val_f1": state.best_f1,
                    "best/val_thr": state.best_thr,
                    "best/val_epoch": state.best_epoch,
                    "epoch": epoch,
                }, commit=False)

    # Load best checkpoint
    best_thr = state.best_thr
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        best_thr = float(ckpt.get("best_val_thr", best_thr))
        state.best_f1 = float(ckpt.get("best_val_f1", state.best_f1))
        state.best_epoch = int(ckpt.get("best_val_epoch", state.best_epoch))

    # TEST eval using best_val_thr (fixed)
    tev = run_eval(dl_te, cfg, model, device, split_name="test", fixed_thr=best_thr, epoch=state.best_epoch)
    tm = tev["metrics"]

    print(
        f"[test] f1={tm['f1']:.4f} thr={tm['threshold']:.3f} "
        f"pr_auc={tm.get('pr_auc', float('nan')):.4f} roc_auc={tm.get('roc_auc', float('nan')):.4f}"
    )

    # Final summary log
    if wandb.run is not None:
        wandb.log({
            "best/val_f1": state.best_f1,
            "best/val_thr": best_thr,
            "best/val_epoch": state.best_epoch,
            "test/f1_final": tm["f1"],
            "test/precision_final": tm["precision"],
            "test/recall_final": tm["recall"],
            "test/accuracy_final": tm["accuracy"],
            "test/pr_auc_final": tm.get("pr_auc", float("nan")),
            "test/roc_auc_final": tm.get("roc_auc", float("nan")),
        }, commit=True)


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}

    train_and_eval(cfg)

if __name__ == "__main__":
    main()
