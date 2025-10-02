#!/usr/bin/env python3
"""
Minimal trainer for Adaptive ST-GCN (EEG spindles)
-------------------------------------------------
Usage:
  python train_stgcn.py --config config.yaml

Assumptions:
- Windows stored as a NumPy array at cfg.data.windows_npy with shape (N, C, T)
- Framewise labels stored at cfg.data.labels_framewise with shape (N, T)
- Optional per-channel labels at cfg.data.labels_per_channel with shape (N, C, T)

This script:
- Loads config (OmegaConf if available, else PyYAML)
- Builds prior adjacency from 10–20 coords if cfg.graph.prior == 'distance'
- Instantiates STGCNSpindle
- Uses a balanced batch sampler with positive_fraction from the config
- Trains and validates; logs to Weights & Biases if enabled

Note: You can swap the loss functions with your existing losses.py without changing
this file (see build_losses()).
"""
from __future__ import annotations
import argparse
import os
import time
from datetime import datetime
from types import SimpleNamespace
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Local imports
from stgcn_spindle import STGCNSpindle
from graph_prior import build_distance_prior
from wandb_graph_logging import log_graphs_to_wandb  # optional

try:
    import wandb  # type: ignore
except Exception:
    wandb = None

# -------------------------
# Config loading utilities
# -------------------------

def load_config(path: str) -> SimpleNamespace:
    # Try OmegaConf first (if installed)
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(path)
        return SimpleNamespace(**OmegaConf.to_container(cfg, resolve=True))
    except Exception:
        pass
    # Fallback to PyYAML
    import yaml
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return SimpleNamespace(**data)


# -------------------------
# Dataset & Samplers
# -------------------------

class NumpyWindowsDataset(Dataset):
    def __init__(self, x_path: str, y_t_path: str, y_ct_path: Optional[str] = None, idxs: Optional[np.ndarray] = None):
        self.X = np.load(x_path, mmap_mode='r')  # (N,C,T)
        self.y_t = np.load(y_t_path, mmap_mode='r')  # (N,T)
        assert len(self.X) == len(self.y_t), "X and y_t must have same length"
        self.y_ct = None
        if y_ct_path and os.path.exists(y_ct_path) and y_ct_path.lower() != 'null':
            self.y_ct = np.load(y_ct_path, mmap_mode='r')  # (N,C,T)
            assert len(self.X) == len(self.y_ct), "X and y_ct must have same length"
        self._all_indices = np.arange(len(self.X)) if idxs is None else idxs

    def __len__(self):
        return len(self._all_indices)

    def __getitem__(self, i: int):
        idx = int(self._all_indices[i])
        x = torch.from_numpy(self.X[idx]).float()    # (C,T)
        y_t = torch.from_numpy(self.y_t[idx]).float()  # (T)
        sample = {'x': x, 'y_t': y_t}
        if self.y_ct is not None:
            y_ct = torch.from_numpy(self.y_ct[idx]).float()  # (C,T)
            sample['y_ct'] = y_ct
        return sample

    def has_positive(self, i: int, thr: float = 0.5) -> bool:
        idx = int(self._all_indices[i])
        # y_t is (T,) with 0/1 or probs
        return bool(np.any(self.y_t[idx] >= thr))


class BalancedBatchSampler(torch.utils.data.Sampler[List[int]]):
    """Yields indices in balanced batches with a target positive fraction.
    Positives determined from dataset.has_positive(i).
    """
    def __init__(self, dataset: NumpyWindowsDataset, batch_size: int, pos_frac: float, steps_per_epoch: Optional[int] = None):
        self.ds = dataset
        self.batch_size = batch_size
        self.pos_frac = float(np.clip(pos_frac, 0.0, 1.0))
        # Pre-scan which items are positive/negative
        flags = [self.ds.has_positive(i) for i in range(len(self.ds))]
        self.pos_indices = np.where(np.array(flags, dtype=bool))[0]
        self.neg_indices = np.where(~np.array(flags, dtype=bool))[0]
        if len(self.pos_indices) == 0:
            # fallback: treat top 10% as pos by max label value (very rare)
            y_max = [float(self.ds.y_t[int(self.ds._all_indices[i])].max()) for i in range(len(self.ds))]
            thresh = np.percentile(y_max, 90)
            self.pos_indices = np.where(np.array(y_max) >= thresh)[0]
            self.neg_indices = np.where(np.array(y_max) < thresh)[0]
        # steps per epoch default
        if steps_per_epoch is None:
            steps_per_epoch = int(np.ceil(len(self.ds) / self.batch_size))
        self.steps_per_epoch = steps_per_epoch

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        bs = self.batch_size
        k_pos = int(round(bs * self.pos_frac))
        k_neg = bs - k_pos
        rng = np.random.default_rng()
        for _ in range(self.steps_per_epoch):
            pos = rng.choice(self.pos_indices, size=k_pos, replace=(k_pos > len(self.pos_indices))) if k_pos > 0 else []
            neg = rng.choice(self.neg_indices, size=k_neg, replace=(k_neg > len(self.neg_indices))) if k_neg > 0 else []
            batch = np.concatenate([pos, neg]).tolist()
            rng.shuffle(batch)
            yield batch


# -------------------------
# Loss wiring
# -------------------------

def build_losses(cfg) -> Dict[str, Any]:
    # Try to import your project losses if available
    bce_fn = None
    dice_fn = None
    tversky_fn = None
    try:
        from losses import bce_with_logits as proj_bce  # type: ignore
        bce_fn = proj_bce
    except Exception:
        pass
    try:
        from losses import dice_loss as proj_dice  # type: ignore
        dice_fn = proj_dice if cfg.loss.dice.get('enabled', False) else None
    except Exception:
        dice_fn = None
    try:
        from losses import tversky_loss as proj_tversky  # type: ignore
        tversky_fn = proj_tversky if cfg.loss.tversky.get('enabled', False) else None
    except Exception:
        tversky_fn = None

    # Fallback BCE using torch if project loss not found
    if bce_fn is None:
        def bce_fn(pred, target, pos_weight=None):
            return F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight)

    return dict(bce=bce_fn, dice=dice_fn, tversky=tversky_fn)


# -------------------------
# Training / Eval
# -------------------------

def train_one_epoch(model, loader, optim, device, losses, cfg, epoch, pos_weight_tensor):
    model.train()
    total = 0.0
    for step, batch_idx in enumerate(loader):
        batch = loader.dataset[batch_idx] if isinstance(batch_idx, list) else batch_idx
        # If using custom sampler, batch is list of indices -> collate manually
        if isinstance(batch, list):
            samples = [loader.dataset[i] for i in batch]
        else:
            samples = batch
        x = torch.stack([s['x'] for s in samples]).to(device)
        y_t = torch.stack([s['y_t'] for s in samples]).to(device)
        y_ct = torch.stack([s['y_ct'] for s in samples]).to(device) if ('y_ct' in samples[0]) else None

        out = model(x)
        lg = out['logits_global']  # (B,T)
        loss = cfg.loss.det_weight * losses['bce'](lg, y_t, pos_weight=pos_weight_tensor)

        if cfg.loss.dice.get('enabled', False) and losses['dice'] is not None:
            loss = loss + cfg.loss.dice.get('weight', 0.5) * losses['dice'](torch.sigmoid(lg), y_t)
        if cfg.loss.tversky.get('enabled', False) and losses['tversky'] is not None:
            alpha = cfg.loss.tversky.get('alpha', 0.7)
            beta = cfg.loss.tversky.get('beta', 0.3)
            loss = loss + cfg.loss.tversky.get('weight', 0.0) * losses['tversky'](torch.sigmoid(lg), y_t, alpha=alpha, beta=beta)
        if cfg.loss.per_channel.get('enabled', False) and y_ct is not None:
            lch = out['logits_per_ch']
            loss = loss + cfg.loss.per_channel.get('weight', 0.3) * losses['bce'](lch, y_ct, pos_weight=pos_weight_tensor)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.train.get('grad_clip', 0.0):
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optim.step()

        total += loss.item()

        if wandb and cfg.logging.wandb.get('enabled', False) and (step % 10 == 0):
            wandb.log({'train/loss': loss.item()}, step=epoch * 10000 + step)

    return total / max(1, (step + 1))


def evaluate(model, loader, device, losses, cfg, pos_weight_tensor):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for step, batch in enumerate(loader):
            x = torch.stack([s['x'] for s in batch]).to(device)
            y_t = torch.stack([s['y_t'] for s in batch]).to(device)
            y_ct = torch.stack([s['y_ct'] for s in batch]).to(device) if ('y_ct' in batch[0]) else None
            out = model(x)
            lg = out['logits_global']
            loss = cfg.loss.det_weight * losses['bce'](lg, y_t, pos_weight=pos_weight_tensor)
            if cfg.loss.dice.get('enabled', False) and losses['dice'] is not None:
                loss = loss + cfg.loss.dice.get('weight', 0.5) * losses['dice'](torch.sigmoid(lg), y_t)
            if cfg.loss.per_channel.get('enabled', False) and y_ct is not None:
                lch = out['logits_per_ch']
                loss = loss + cfg.loss.per_channel.get('weight', 0.3) * losses['bce'](lch, y_ct, pos_weight=pos_weight_tensor)
            total += loss.item()
    return total / max(1, (step + 1))


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build prior adjacency
    if cfg.graph.prior == 'distance':
        A_prior = build_distance_prior(cfg.graph.coords_csv, cfg.graph.channel_names, sigma=cfg.graph.sigma, normalize_coords=cfg.graph.normalize)
        A_prior = A_prior.float()
    elif cfg.graph.prior == 'identity':
        A_prior = torch.eye(cfg.model.C)
    else:
        raise ValueError(f"Unknown graph.prior: {cfg.graph.prior}")

    # Model
    model = STGCNSpindle(
        C=cfg.model.C,
        T=cfg.model.T,
        F0=cfg.model.F0,
        block_channels=tuple(cfg.model.block_channels),
        dilations=tuple(cfg.model.dilations),
        A_prior=A_prior,
        emb_dim=8,
        lambda_init=cfg.graph.lambda_init,
        use_dynamic=cfg.graph.use_dynamic,
        beta_init=cfg.graph.beta_init,
        dropout=cfg.model.dropout,
        se_ratio=cfg.model.se_ratio,
    ).to(device)

    # Data split
    X = np.load(cfg.data.windows_npy, mmap_mode='r')
    N = len(X)
    val_frac = getattr(cfg.data, 'val_fraction', 0.2)
    idxs = np.arange(N)
    np.random.default_rng(42).shuffle(idxs)
    n_val = int(N * val_frac)
    val_idxs = idxs[:n_val]
    tr_idxs = idxs[n_val:]

    ds_tr = NumpyWindowsDataset(cfg.data.windows_npy, cfg.data.labels_framewise, cfg.data.labels_per_channel, idxs=tr_idxs)
    ds_va = NumpyWindowsDataset(cfg.data.windows_npy, cfg.data.labels_framewise, cfg.data.labels_per_channel, idxs=val_idxs)

    # Samplers & loaders
    bs = cfg.train.batch_size
    pos_frac = cfg.train.sampler.get('positive_fraction', 0.5)
    steps_per_epoch = None
    train_sampler = BalancedBatchSampler(ds_tr, batch_size=bs, pos_frac=pos_frac, steps_per_epoch=steps_per_epoch)
    train_loader = DataLoader(ds_tr, batch_sampler=train_sampler, num_workers=2)

    def collate_list(samples):
        return samples  # we collate manually in evaluate/train when needed

    val_loader = DataLoader(ds_va, batch_size=bs, shuffle=False, num_workers=2, collate_fn=collate_list)

    # Optimizer & scheduler
    if cfg.train.optimizer.lower() == 'adamw':
        optim = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    else:
        optim = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    sched = None
    if cfg.train.scheduler.lower() == 'onecycle':
        steps_per_epoch = len(train_sampler)
        sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=cfg.train.lr, epochs=cfg.train.epochs, steps_per_epoch=steps_per_epoch)

    # Losses
    L = build_losses(cfg)
    pos_weight = torch.tensor(float(cfg.loss.bce.get('pos_weight', 1.0)), device=device)

    # W&B
    if wandb and cfg.logging.wandb.get('enabled', False):
        run_name = cfg.logging.wandb.get('run_name', datetime.now().strftime('%Y%m%d-%H%M%S'))
        wandb.init(project=cfg.logging.wandb.get('project', 'spindle_stgcn'), name=str(run_name), config=dict(cfg.__dict__))

    # Output dirs
    os.makedirs('checkpoints', exist_ok=True)

    # Train loop
    best_val = float('inf')
    for epoch in range(cfg.train.epochs):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optim, device, L, cfg, epoch, pos_weight)
        va_loss = evaluate(model, val_loader, device, L, cfg, pos_weight)
        if sched is not None:
            sched.step()

        dt = time.time() - t0
        print(f"Epoch {epoch+1}/{cfg.train.epochs} | train {tr_loss:.4f} | val {va_loss:.4f} | {dt:.1f}s")

        if wandb and cfg.logging.wandb.get('enabled', False):
            wandb.log({'epoch': epoch+1, 'val/loss': va_loss, 'lr': optim.param_groups[0]['lr']})
            # Also log adjacencies once in a while
            if epoch % max(1, cfg.logging.get('save_every_epochs', 1)) == 0:
                with torch.no_grad():
                    dummy = torch.randn(2, cfg.model.C, cfg.model.T, device=device)
                    outs = model(dummy)
                    log_graphs_to_wandb(outs, step=(epoch+1))

        # Checkpoint
        if (epoch + 1) % max(1, cfg.logging.get('save_every_epochs', 1)) == 0:
            ckpt_path = f"checkpoints/stgcn_{datetime.now().strftime('%Y%m%d-%H%M%S')}_e{epoch+1}.pt"
            torch.save({'model': model.state_dict(), 'cfg': cfg.__dict__}, ckpt_path)

        if va_loss < best_val:
            best_val = va_loss
            torch.save({'model': model.state_dict(), 'cfg': cfg.__dict__}, 'checkpoints/stgcn_best.pt')

    print('Training done. Best val loss:', best_val)


if __name__ == '__main__':
    main()
