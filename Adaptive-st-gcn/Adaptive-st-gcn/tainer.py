#!/usr/bin/env python3
"""
Enhanced trainer for Adaptive ST-GCN with comprehensive metrics logging
"""

from __future__ import annotations
import argparse
import os
import time
from datetime import datetime
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Local modules
from stgcn_spindle import STGCNSpindle
from graph_prior import build_distance_prior
from wandb_graph_logging import log_graphs_to_wandb
from metrics_logging import evaluate_and_log  # wrapper must exist

try:
    import wandb  # type: ignore
except Exception:
    wandb = None

# ---------------------------------------------------------------------
# Small dict utility
# ---------------------------------------------------------------------

class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, *args, **kwargs):
        super().__init__()
        src = dict(*args, **kwargs)
        for k, v in src.items():
            super().__setitem__(k, self._wrap(v))

    @classmethod
    def _wrap(cls, v):
        if isinstance(v, dict):
            return DotDict(v)
        if isinstance(v, list):
            return [cls._wrap(x) for x in v]
        if isinstance(v, tuple):
            return tuple(cls._wrap(x) for x in v)
        return v

    def to_dict(self) -> dict:
        return to_plain(self)


def to_plain(x: Any) -> Any:
    if isinstance(x, DotDict):
        return {k: to_plain(v) for k, v in x.items()}
    if hasattr(x, "__dict__") and not isinstance(x, (torch.nn.Module,)):
        try:
            return {k: to_plain(v) for k, v in x.__dict__.items()}
        except Exception:
            pass
    if isinstance(x, dict):
        return {k: to_plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = [to_plain(v) for v in x]
        return t if isinstance(x, list) else tuple(t)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _resolve_run_name(raw: Optional[str]) -> str:
    if not raw or not isinstance(raw, str):
        return datetime.now().strftime("%Y%m%d-%H%M%S")
    if raw.startswith("${now:") and raw.endswith("}"):
        fmt = raw[len("${now:"):-1]
        return datetime.now().strftime(fmt)
    return raw


def load_config(path: str) -> DotDict:
    try:
        from omegaconf import OmegaConf  # type: ignore
        cfg = OmegaConf.load(path)
        data = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        import yaml
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    return DotDict(data)

# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

def _resolve_windows_path(p: str) -> str:
    if os.path.isdir(p):
        return os.path.join(p, "windows.npy")
    return p


class NumpyWindowsDataset(Dataset):
    def __init__(self, x_path: str, y_t_path: str, y_ct_path: Optional[str] = None, idxs: Optional[np.ndarray] = None):
        x_path = _resolve_windows_path(x_path)
        self.X = np.load(x_path, mmap_mode='r')
        self.y_t = np.load(y_t_path, mmap_mode='r')
        assert len(self.X) == len(self.y_t), "X and y_t must have same length"

        self.y_ct = None
        if y_ct_path and os.path.exists(y_ct_path) and y_ct_path.lower() != 'null':
            self.y_ct = np.load(y_ct_path, mmap_mode='r')
            assert len(self.X) == len(self.y_ct), "X and y_ct must have same length"

        self._all_indices = np.arange(len(self.X)) if idxs is None else np.asarray(idxs)

    def __len__(self):
        return int(self._all_indices.shape[0])

    def __getitem__(self, i: int):
        idx = int(self._all_indices[i])
        # copy() to avoid "non-writable numpy" warnings
        x = torch.from_numpy(self.X[idx].copy()).float()
        y_t = torch.from_numpy(self.y_t[idx].copy()).float()
        sample = {'x': x, 'y_t': y_t}
        if self.y_ct is not None:
            y_ct = torch.from_numpy(self.y_ct[idx].copy()).float()  # correct source
            sample['y_ct'] = y_ct
        return sample

    def has_positive(self, i: int, thr: float = 0.5) -> bool:
        idx = int(self._all_indices[i])
        return bool(np.any(self.y_t[idx] >= thr))


class BalancedBatchSampler(torch.utils.data.Sampler[List[int]]):
    """Yield index lists with a target positive fraction each batch."""
    def __init__(self, dataset: NumpyWindowsDataset, batch_size: int, pos_frac: float, steps_per_epoch: Optional[int] = None):
        self.ds = dataset
        self.batch_size = int(batch_size)
        self.pos_frac = float(np.clip(pos_frac, 0.0, 1.0))

        flags = [self.ds.has_positive(i) for i in range(len(self.ds))]
        flags = np.array(flags, dtype=bool)
        self.pos_indices = np.where(flags)[0]
        self.neg_indices = np.where(~flags)[0]

        if self.pos_indices.size == 0:
            y_max = np.array([float(self.ds.y_t[int(self.ds._all_indices[i])].max()) for i in range(len(self.ds))])
            thr = np.percentile(y_max, 90)
            self.pos_indices = np.where(y_max >= thr)[0]
            self.neg_indices = np.where(y_max < thr)[0]

        if steps_per_epoch is None:
            steps_per_epoch = int(np.ceil(len(self.ds) / max(1, self.batch_size)))
        self._steps = int(steps_per_epoch)

    def __len__(self):
        return self._steps

    def __iter__(self):
        rng = np.random.default_rng()
        k_pos = int(round(self.batch_size * self.pos_frac))
        k_neg = self.batch_size - k_pos
        for _ in range(self._steps):
            pos = rng.choice(self.pos_indices, size=k_pos, replace=(k_pos > self.pos_indices.size)) if k_pos > 0 else []
            neg = rng.choice(self.neg_indices, size=k_neg, replace=(k_neg > self.neg_indices.size)) if k_neg > 0 else []
            batch = np.concatenate([pos, neg]).tolist()
            rng.shuffle(batch)
            yield batch

# ---------------------------------------------------------------------
# Loss wiring
# ---------------------------------------------------------------------

def build_losses(cfg: DotDict) -> Dict[str, Any]:
    bce_fn = None
    dice_fn = None
    tversky_fn = None
    try:
        from losses import bce_with_logits as proj_bce  # optional project loss
        bce_fn = proj_bce
    except Exception:
        pass
    try:
        from losses import dice_loss as proj_dice
        dice_fn = proj_dice if cfg.loss.dice.get('enabled', False) else None
    except Exception:
        dice_fn = None
    try:
        from losses import tversky_loss as proj_tversky
        tversky_fn = proj_tversky if cfg.loss.tversky.get('enabled', False) else None
    except Exception:
        tversky_fn = None

    if bce_fn is None:
        def bce_fn(pred, target, pos_weight=None):
            return F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight)

    return dict(bce=bce_fn, dice=dice_fn, tversky=tversky_fn)

# ---------------------------------------------------------------------
# Train / Eval
# ---------------------------------------------------------------------

def _stack_from_list(batch_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out = {
        'x': torch.stack([s['x'] for s in batch_list]),
        'y_t': torch.stack([s['y_t'] for s in batch_list]),
    }
    if 'y_ct' in batch_list[0]:
        out['y_ct'] = torch.stack([s['y_ct'] for s in batch_list])
    return out


def train_one_epoch(model, loader, optim, device, losses, cfg: DotDict, epoch: int, pos_weight_tensor):
    model.train()
    total = 0.0
    for step, batch_list in enumerate(loader):
        batch = _stack_from_list(batch_list)
        x = batch['x'].to(device)
        y_t = batch['y_t'].to(device)
        y_ct = batch.get('y_ct')
        if y_ct is not None:
            y_ct = y_ct.to(device)

        out = model(x)
        lg = out['logits_global']
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip))
        optim.step()

        total += float(loss.item())

        # Optional per-batch logging (use a separate axis and metric name)
        if wandb and cfg.logging.wandb.get('enabled', False) and (step % 10 == 0):
            global_step = epoch * 10000 + step
            wandb.log({'global_step': global_step, 'train/batch_loss': float(loss.item())}, step=global_step)

    return total / max(1, (step + 1))


@torch.no_grad()
def evaluate(model, loader, device, losses, cfg: DotDict, pos_weight_tensor):
    model.eval()
    total = 0.0
    for step, batch_list in enumerate(loader):
        batch = _stack_from_list(batch_list)
        x = batch['x'].to(device)
        y_t = batch['y_t'].to(device)
        y_ct = batch.get('y_ct')
        if y_ct is not None:
            y_ct = y_ct.to(device)

        out = model(x)
        lg = out['logits_global']
        loss = cfg.loss.det_weight * losses['bce'](lg, y_t, pos_weight=pos_weight_tensor)
        if cfg.loss.dice.get('enabled', False) and losses['dice'] is not None:
            loss = loss + cfg.loss.dice.get('weight', 0.5) * losses['dice'](torch.sigmoid(lg), y_t)
        if cfg.loss.per_channel.get('enabled', False) and y_ct is not None:
            lch = out['logits_per_ch']
            loss = loss + cfg.loss.per_channel.get('weight', 0.3) * losses['bce'](lch, y_ct, pos_weight=pos_weight_tensor)
        total += float(loss.item())
    return total / max(1, (step + 1))

# ---------------------------------------------------------------------
# Spindle-travel logging
# ---------------------------------------------------------------------

import matplotlib.pyplot as plt
from trace_utils import extract_onsets_from_logits_per_channel, propagation_order_from_onsets

@torch.no_grad()
def log_spindle_travel(model, loader, device, channel_names, step: int = None, prefix: str = "val"):
    if wandb is None or not wandb.run:
        return
    model.eval()

    all_onsets = []
    all_orders = []

    for batch_list in loader:
        batch = {
            'x': torch.stack([s['x'] for s in batch_list]).to(device),
            'y_t': torch.stack([s['y_t'] for s in batch_list]).to(device),
        }
        out = model(batch['x'])
        if 'logits_per_ch' not in out:
            print("[WARN] Model outputs no 'logits_per_ch'; cannot compute spindle travel.")
            return

        lch = out['logits_per_ch']  # (B, C, T)
        onsets = extract_onsets_from_logits_per_channel(lch, thr=0.5, min_consec=5)  # (B, C)
        order = propagation_order_from_onsets(onsets)  # (B, C)

        all_onsets.append(onsets.detach().cpu())
        all_orders.append(order.detach().cpu())

    all_onsets = torch.cat(all_onsets, dim=0).numpy()  # (N, C)
    all_orders = torch.cat(all_orders, dim=0).numpy()  # (N, C)

    C = all_onsets.shape[1]
    ch_names = list(channel_names) if channel_names is not None else [f"Ch{i}" for i in range(C)]

    # Figure 1: mean onset per channel (ignore -1)
    mean_onset = []
    for c in range(C):
        vals = all_onsets[:, c]
        vals = vals[vals >= 0]
        mean_onset.append(float(vals.mean()) if vals.size > 0 else np.nan)

    fig1 = plt.figure(figsize=(10, 4))
    xs = np.arange(C)
    plt.bar(xs, mean_onset)
    plt.xticks(xs, ch_names, rotation=45, ha='right')
    plt.ylabel("Mean onset (frame)")
    plt.title("Spindle travel: mean onset per channel")
    plt.tight_layout()

    # Figure 2: average propagation rank per channel
    ranks = {ch: [] for ch in range(C)}
    for sample_order, sample_onsets in zip(all_orders, all_onsets):
        if not (sample_onsets >= 0).any():
            continue
        r = np.full(C, -1, dtype=np.int32)
        for k, ch in enumerate(sample_order):
            r[ch] = k
        for ch in range(C):
            if sample_onsets[ch] >= 0 and r[ch] >= 0:
                ranks[ch].append(r[ch])

    avg_rank = [np.mean(ranks[ch]) if len(ranks[ch]) > 0 else np.nan for ch in range(C)]
    fig2 = plt.figure(figsize=(10, 4))
    plt.bar(xs, avg_rank)
    plt.xticks(xs, ch_names, rotation=45, ha='right')
    plt.ylabel("Average rank (lower = earlier)")
    plt.title("Spindle travel: average propagation rank per channel")
    plt.tight_layout()

    # Figure 3: example propagation rank heatmap (first K samples)
    K = min(32, all_orders.shape[0])
    ex = all_orders[:K]   # (K, C)
    ex_rank = np.full_like(ex, -1)
    for i in range(K):
        for k, ch in enumerate(ex[i]):
            ex_rank[i, ch] = k

    fig3 = plt.figure(figsize=(6, 6))
    plt.imshow(ex_rank, aspect='auto', interpolation='nearest')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xlabel("Channel"); plt.ylabel("Sample")
    plt.title("Propagation rank per channel (lower=earlier)")
    plt.tight_layout()

    wandb.log({
        "epoch": step,
        f"{prefix}/travel_mean_onset": wandb.Image(fig1),
        f"{prefix}/travel_avg_rank": wandb.Image(fig2),
        f"{prefix}/travel_example_rank_heatmap": wandb.Image(fig3),
    })

    plt.close(fig1); plt.close(fig2); plt.close(fig3)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='config.yaml')
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Graph prior
    if cfg.graph.prior == 'distance':
        A_prior = build_distance_prior(
            cfg.graph.coords_csv,
            cfg.graph.channel_names,
            sigma=float(cfg.graph.sigma),
            normalize_coords=bool(cfg.graph.normalize),
        ).float()
    elif cfg.graph.prior == 'identity':
        A_prior = torch.eye(int(cfg.model.C))
    else:
        raise ValueError(f"Unknown graph.prior: {cfg.graph.prior}")

    # Model
    model = STGCNSpindle(
        C=int(cfg.model.C),
        T=int(cfg.model.T),
        F0=int(cfg.model.F0),
        block_channels=tuple(int(x) for x in cfg.model.block_channels),
        dilations=tuple(int(x) for x in cfg.model.dilations),
        A_prior=A_prior,
        emb_dim=8,
        lambda_init=float(cfg.graph.lambda_init),
        use_dynamic=bool(cfg.graph.use_dynamic),
        beta_init=float(cfg.graph.beta_init),
        dropout=float(cfg.model.dropout),
        se_ratio=int(cfg.model.se_ratio),
    ).to(device)

    # Split
    X_path = _resolve_windows_path(cfg.data.windows_npy)
    X = np.load(X_path, mmap_mode='r')
    N = len(X)
    val_frac = float(cfg.data.get('val_fraction', 0.2))
    rng = np.random.default_rng(42)
    idxs = np.arange(N)
    rng.shuffle(idxs)
    n_val = int(N * val_frac)
    val_idxs = idxs[:n_val]
    tr_idxs = idxs[n_val:]

    ds_tr = NumpyWindowsDataset(cfg.data.windows_npy, cfg.data.labels_framewise, cfg.data.labels_per_channel, idxs=tr_idxs)
    ds_va = NumpyWindowsDataset(cfg.data.windows_npy, cfg.data.labels_framewise, cfg.data.labels_per_channel, idxs=val_idxs)

    # Loaders
    def collate_list(samples):
        return samples

    bs = int(cfg.train.batch_size)
    pos_frac = float(cfg.train.sampler.get('positive_fraction', 0.5))
    sampler = BalancedBatchSampler(ds_tr, batch_size=bs, pos_frac=pos_frac, steps_per_epoch=None)
    train_loader = DataLoader(ds_tr, batch_sampler=sampler, num_workers=2, collate_fn=collate_list)
    val_loader = DataLoader(ds_va, batch_size=bs, shuffle=False, num_workers=2, collate_fn=collate_list)

    # Optimizer & scheduler
    if str(cfg.train.optimizer).lower() == 'adamw':
        optim = torch.optim.AdamW(model.parameters(), lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay))
    else:
        optim = torch.optim.Adam(model.parameters(), lr=float(cfg.train.lr))

    sched = None
    if str(cfg.train.scheduler).lower() == 'onecycle':
        steps_per_epoch = len(sampler)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optim, max_lr=float(cfg.train.lr), epochs=int(cfg.train.epochs), steps_per_epoch=steps_per_epoch
        )

    # Losses
    L = build_losses(cfg)
    pos_weight = torch.tensor(float(cfg.loss.bce.get('pos_weight', 1.0)), device=device)

    # W&B
    if wandb and cfg.logging.wandb.get('enabled', False):
        run_name = _resolve_run_name(cfg.logging.wandb.get('run_name'))
        wandb.init(
            project=str(cfg.logging.wandb.get('project', 'spindle_stgcn')),
            name=str(run_name),
            config=to_plain(cfg),
        )
        # define axes / step mapping
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")         # epoch-based train metrics
        wandb.define_metric("val/*", step_metric="epoch")           # epoch-based validation metrics
        wandb.define_metric("final_val/*", step_metric="epoch")     # final evaluation
        wandb.define_metric("travel/*", step_metric="epoch")        # spindle-travel plots
        wandb.define_metric("global_step")                          # for per-batch visuals
        wandb.define_metric("train/batch_*", step_metric="global_step")

    # Output dir
    os.makedirs('checkpoints', exist_ok=True)

    # Eval cadence
    eval_metrics_every = int(cfg.logging.get('eval_metrics_every_epochs', 5))
    detection_threshold = float(cfg.logging.get('detection_threshold', 0.5))

    # Train loop
    best_val = float('inf')

    for epoch in range(int(cfg.train.epochs)):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optim, device, L, cfg, epoch, pos_weight)
        va_loss = evaluate(model, val_loader, device, L, cfg, pos_weight)
        if sched is not None:
            sched.step()

        dt = time.time() - t0
        print(f"Epoch {epoch + 1}/{cfg.train.epochs} | train {tr_loss:.4f} | val {va_loss:.4f} | {dt:.1f}s")

        # Epoch-level W&B logging (epoch axis)
        if wandb and cfg.logging.wandb.get('enabled', False):
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': tr_loss,
                'val/loss': va_loss,
                'lr': optim.param_groups[0]['lr']
            })

        # Comprehensive metrics (CM/ROC/PR + travel) on schedule
        should_eval_metrics = ((epoch + 1) % eval_metrics_every == 0) or ((epoch + 1) == int(cfg.train.epochs))
        if should_eval_metrics:
            print("\n" + "=" * 60)
            print(f"Running comprehensive evaluation at epoch {epoch + 1}")
            print("=" * 60)

            # full metrics on val (epoch axis, pass step=epoch+1)
            evaluate_and_log(
                model=model,
                loader=val_loader,
                device=device,
                step=epoch + 1,
                prefix="val",
                threshold=detection_threshold
            )

            # Spindle travel (uses per-channel logits)
            if cfg.logging.wandb.get('enabled', False):
                log_spindle_travel(
                    model,
                    val_loader,
                    device,
                    channel_names=cfg.graph.channel_names,
                    step=epoch + 1,
                    prefix="val"
                )

            # Log adjacency heatmaps (quick pass on dummy input)
            if cfg.logging.get('log_graphs', True):
                with torch.no_grad():
                    dummy = torch.randn(2, int(cfg.model.C), int(cfg.model.T), device=device)
                    outs = model(dummy)
                    log_graphs_to_wandb(outs, step=epoch + 1)

        # Checkpoints
        if (epoch + 1) % max(1, int(cfg.logging.get('save_every_epochs', 1))) == 0:
            ckpt_path = f"checkpoints/stgcn_{datetime.now().strftime('%Y%m%d-%H%M%S')}_e{epoch + 1}.pt"
            torch.save({'model': model.state_dict(), 'cfg': to_plain(cfg), 'epoch': epoch + 1}, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save({'model': model.state_dict(), 'cfg': to_plain(cfg), 'epoch': epoch + 1}, 'checkpoints/stgcn_best.pt')
            print(f"New best validation loss: {best_val:.4f}")

    print('\n' + '=' * 60)
    print('Training complete!')
    print(f'Best validation loss: {best_val:.4f}')
    print('=' * 60)

    # Final comprehensive evaluation on full validation set
    print("\nRunning final evaluation on full validation set...")
    evaluate_and_log(
        model=model,
        loader=val_loader,
        device=device,
        step=int(cfg.train.epochs),   # plotted at final epoch on epoch axis
        prefix="final_val",
        threshold=detection_threshold
    )


if __name__ == '__main__':
    main()
