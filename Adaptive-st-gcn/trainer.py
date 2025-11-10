#!/usr/bin/env python3
"""
Trainer for Adaptive ST-GCN Spindle Detection
---------------------------------------------
• Computes sklearn metrics and logs to W&B
• Saves best checkpoint based on validation F1
• No data augmentation, no Tversky loss
"""

from __future__ import annotations
import argparse, os, time
from datetime import datetime
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from typing import Any, Dict, Optional, List

from stgcn_spindle import STGCNSpindle
from graph_prior import build_distance_prior
from wandb_graph_logging import log_graphs_to_wandb
from metrics_logging import evaluate_and_log
from losses import FocalLoss, DiceLoss, BCELoss, WeightedBCELoss

try:
    import wandb
except Exception:
    wandb = None


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------
class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_config(path: str) -> DotDict:
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(path)
    data = OmegaConf.to_container(cfg, resolve=True)

    # --- recursive conversion ---
    def to_dotdict(obj):
        if isinstance(obj, dict):
            return DotDict({k: to_dotdict(v) for k, v in obj.items()})
        elif isinstance(obj, list):
            return [to_dotdict(v) for v in obj]
        else:
            return obj

    return to_dotdict(data)

def _resolve_windows_path(p: str) -> str:
    return os.path.join(p, "windows.npy") if os.path.isdir(p) else p


# ---------------------------------------------------------------------
# Dataset (no augmentation)
# ---------------------------------------------------------------------
class NumpyWindowsDataset(Dataset):
    def __init__(self, x_path: str, y_t_path: str, y_ct_path: Optional[str] = None,
                 idxs: Optional[np.ndarray] = None):
        x_path = _resolve_windows_path(x_path)
        self.X = np.load(x_path, mmap_mode="r")
        self.y_t = np.load(y_t_path, mmap_mode="r")
        self.y_ct = None
        if y_ct_path and os.path.exists(y_ct_path):
            self.y_ct = np.load(y_ct_path, mmap_mode="r")
        self._all_indices = np.arange(len(self.X)) if idxs is None else np.asarray(idxs)

    def __len__(self): return len(self._all_indices)

    def __getitem__(self, i: int):
        idx = int(self._all_indices[i])
        x = torch.from_numpy(self.X[idx].astype(np.float32))
        y_t = torch.from_numpy(self.y_t[idx].astype(np.float32))
        sample = {"x": x, "y_t": y_t}
        if self.y_ct is not None:
            sample["y_ct"] = torch.from_numpy(self.y_ct[idx].astype(np.float32))
        return sample


# ---------------------------------------------------------------------
# Balanced Sampler
# ---------------------------------------------------------------------
class BalancedBatchSampler(torch.utils.data.Sampler[List[int]]):
    def __init__(self, ds: NumpyWindowsDataset, batch_size: int, pos_frac: float):
        self.ds, self.batch_size = ds, int(batch_size)
        self.pos_frac = float(np.clip(pos_frac, 0, 1))
        flags = [self.ds.y_t[i].max() > 0 for i in range(len(self.ds))]
        self.pos = np.where(flags)[0]
        self.neg = np.where(~np.array(flags))[0]
        self.steps = int(np.ceil(len(ds) / self.batch_size))

    def __len__(self): return self.steps

    def __iter__(self):
        rng = np.random.default_rng()
        k_pos = int(round(self.batch_size * self.pos_frac))
        k_neg = self.batch_size - k_pos
        for _ in range(self.steps):
            pos = rng.choice(self.pos, k_pos, replace=True)
            neg = rng.choice(self.neg, k_neg, replace=True)
            b = np.concatenate([pos, neg]); rng.shuffle(b)
            yield b.tolist()


# ---------------------------------------------------------------------
# Loss builder (no Tversky)
# ---------------------------------------------------------------------
def build_losses(cfg: DotDict, device: torch.device):
    L = {}
    if cfg.loss.get("main_loss", "weighted_bce") == "focal":
        L["main"] = FocalLoss(alpha=0.25, gamma=2.0).to(device)
    elif cfg.loss.get("main_loss", "weighted_bce") == "bce":
        L["main"] = BCELoss(label_smoothing=0.03).to(device)
    else:
        pos_w = float(cfg.loss.bce.get("pos_weight", 5.0))
        L["main"] = WeightedBCELoss(pos_weight=pos_w).to(device)
    L["dice"] = DiceLoss().to(device) if cfg.loss.get("dice", {}).get("enabled", False) else None
    L["tversky"] = None
    return L


# ---------------------------------------------------------------------
# Epoch loops
# ---------------------------------------------------------------------
def _stack(batch_list):
    out = {"x": torch.stack([b["x"] for b in batch_list]),
           "y_t": torch.stack([b["y_t"] for b in batch_list])}
    if "y_ct" in batch_list[0]:
        out["y_ct"] = torch.stack([b["y_ct"] for b in batch_list])
    return out


def train_one_epoch(model, loader, optim, device, L, cfg, epoch):
    model.train(); tot = 0.0
    for step, b in enumerate(loader):
        batch = _stack(b)
        x, y = batch["x"].to(device), batch["y_t"].to(device)
        out = model(x); lg = out["logits_global"]
        loss = cfg.loss.det_weight * L["main"](lg, y)
        if L["dice"]: loss += cfg.loss.dice.get("weight", 0.5) * L["dice"](lg, y)
        optim.zero_grad(set_to_none=True); loss.backward()
        if cfg.train.get("grad_clip", 0): torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optim.step(); tot += float(loss.item())
    return tot / max(1, (step + 1))


@torch.no_grad()
def evaluate(model, loader, device, L, cfg):
    model.eval(); tot = 0.0
    for step, b in enumerate(loader):
        batch = _stack(b)
        x, y = batch["x"].to(device), batch["y_t"].to(device)
        out = model(x)
        loss = cfg.loss.det_weight * L["main"](out["logits_global"], y)
        if L["dice"]: loss += cfg.loss.dice.get("weight", 0.5) * L["dice"](out["logits_global"], y)
        tot += float(loss.item())
    return tot / max(1, (step + 1))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); args = ap.parse_args()
    cfg = load_config(args.config)
    cfg = DotDict(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Graph prior ---
    A = build_distance_prior(cfg.graph.coords_csv, cfg.graph.channel_names,
                             sigma=cfg.graph.sigma, normalize_coords=cfg.graph.normalize).float()

    # --- Model ---
    model = STGCNSpindle(
        C=int(cfg.model.C), T=int(cfg.model.T), F0=int(cfg.model.F0),
        block_channels=tuple(cfg.model.block_channels),
        dilations=tuple(cfg.model.dilations),
        A_prior=A, emb_dim=8,
        lambda_init=cfg.graph.lambda_init,
        use_dynamic=cfg.graph.use_dynamic,
        beta_init=cfg.graph.beta_init,
        dropout=cfg.model.dropout, se_ratio=cfg.model.se_ratio,
    ).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

    # --- Data ---
    X = np.load(_resolve_windows_path(cfg.data.windows_npy), mmap_mode="r")
    N = len(X); val_frac = float(cfg.data.get("val_fraction", 0.2))
    idxs = np.arange(N); np.random.default_rng(42).shuffle(idxs)
    n_val = int(N * val_frac); val_idx, tr_idx = idxs[:n_val], idxs[n_val:]
    print(f"Split: {len(tr_idx)} train, {len(val_idx)} val")

    ds_tr = NumpyWindowsDataset(cfg.data.windows_npy, cfg.data.labels_framewise, cfg.data.labels_per_channel, tr_idx)
    ds_va = NumpyWindowsDataset(cfg.data.windows_npy, cfg.data.labels_framewise, cfg.data.labels_per_channel, val_idx)
    collate = lambda x: x
    bs = int(cfg.train.batch_size)
    sampler = BalancedBatchSampler(ds_tr, bs, cfg.train.sampler.get("positive_fraction", 0.5))
    tr_loader = DataLoader(ds_tr, batch_sampler=sampler, num_workers=2, collate_fn=collate)
    va_loader = DataLoader(ds_va, batch_size=bs, shuffle=False, num_workers=2, collate_fn=collate)

    # --- Optimizer + scheduler ---
    lr, wd = float(cfg.train.lr), float(cfg.train.weight_decay)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=lr,
                                                epochs=int(cfg.train.epochs),
                                                steps_per_epoch=len(tr_loader)) \
             if str(cfg.train.scheduler).lower() == "onecycle" else None

    # --- Losses ---
    L = build_losses(cfg, device)

    # --- W&B ---
    if wandb and cfg.logging.wandb.get("enabled", False):
        wandb.init(project=cfg.logging.wandb.get("project", "spindle_stgcn"),
                   entity=cfg.logging.wandb.get("entity", None),
                   name=datetime.now().strftime("%Y%m%d-%H%M%S"),
                   config=dict(cfg))
        wandb.define_metric("epoch")
        wandb.define_metric("val/f1", summary="max")

    os.makedirs("checkpoints", exist_ok=True)
    best_val = 0.0

    print("\n===== START TRAINING =====")
    for epoch in range(int(cfg.train.epochs)):
        t0 = time.time()
        tr_loss = train_one_epoch(model, tr_loader, optim, device, L, cfg, epoch)
        va_loss = evaluate(model, va_loader, device, L, cfg)
        if sched: sched.step()
        print(f"Epoch {epoch+1}/{cfg.train.epochs} | train {tr_loss:.4f} | val {va_loss:.4f} | {time.time()-t0:.1f}s")

        metrics = evaluate_and_log(model, va_loader, device, prefix="val", step=epoch + 1,
                                   threshold=cfg.logging.get("detection_threshold", 0.5))
        f1 = metrics.get("f1", 0.0)
        if f1 > best_val:
            best_val = f1
            torch.save({"model": model.state_dict(), "cfg": dict(cfg),
                        "epoch": epoch + 1, "best_val_f1": best_val},
                       "checkpoints/stgcn_best.pt")
            print(f"✓ New best val F1: {best_val:.4f}")

        ckpt = f"checkpoints/stgcn_{datetime.now().strftime('%Y%m%d-%H%M%S')}_e{epoch+1}.pt"
        torch.save({"model": model.state_dict(), "cfg": dict(cfg), "epoch": epoch + 1}, ckpt)

    print(f"\nTRAINING COMPLETE — Best F1 = {best_val:.4f}")
    if wandb and cfg.logging.wandb.get("enabled", False): wandb.finish()


if __name__ == "__main__":
    main()
