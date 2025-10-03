#!/usr/bin/env python3
"""
Standalone evaluation script for trained ST-GCN models

Usage:
  python evaluate_model.py --checkpoint checkpoints/stgcn_best.pt --threshold 0.5
"""

import argparse
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from stgcn_spindle import STGCNSpindle
from metrics_logging import (
    collect_predictions,
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve
)

try:
    import wandb
except Exception:
    wandb = None


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


def _resolve_windows_path(p: str) -> str:
    if os.path.isdir(p):
        return os.path.join(p, "windows.npy")
    return p


class NumpyWindowsDataset(torch.utils.data.Dataset):
    def __init__(self, x_path: str, y_t_path: str, y_ct_path=None, idxs=None):
        x_path = _resolve_windows_path(x_path)
        self.X = np.load(x_path, mmap_mode='r')
        self.y_t = np.load(y_t_path, mmap_mode='r')
        assert len(self.X) == len(self.y_t)

        self.y_ct = None
        if y_ct_path and os.path.exists(y_ct_path) and y_ct_path.lower() != 'null':
            self.y_ct = np.load(y_ct_path, mmap_mode='r')

        self._all_indices = np.arange(len(self.X)) if idxs is None else np.asarray(idxs)

    def __len__(self):
        return int(self._all_indices.shape[0])

    def __getitem__(self, i: int):
        idx = int(self._all_indices[i])
        x = torch.from_numpy(self.X[idx]).float()
        y_t = torch.from_numpy(self.y_t[idx]).float()

        sample = {'x': x, 'y_t': y_t}
        if self.y_ct is not None:
            y_ct = torch.from_numpy(self.y_ct[idx]).float()
            sample['y_ct'] = y_ct
        return sample


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = DotDict(ckpt['cfg'])

    # Rebuild model
    A_prior = None
    if cfg.graph.prior == 'distance':
        from graph_prior import build_distance_prior
        A_prior = build_distance_prior(
            cfg.graph.coords_csv,
            cfg.graph.channel_names,
            sigma=float(cfg.graph.sigma),
            normalize_coords=bool(cfg.graph.normalize),
        ).float()
    elif cfg.graph.prior == 'identity':
        A_prior = torch.eye(int(cfg.model.C))

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

    model.load_state_dict(ckpt['model'])
    model.eval()

    return model, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--output_dir', type=str, default='./eval_results', help='Output directory for plots')
    parser.add_argument('--val_only', action='store_true', help='Only evaluate validation set')
    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.checkpoint}...")
    model, cfg = load_model(args.checkpoint, device)
    print(f"Model loaded. Config: C={cfg.model.C}, T={cfg.model.T}")

    # Load data
    print(f"\nLoading data from {args.data_dir}...")
    X_path = os.path.join(args.data_dir, 'windows.npy')
    y_t_path = os.path.join(args.data_dir, 'labels_framewise.npy')
    y_ct_path = os.path.join(args.data_dir, 'labels_per_channel.npy')

    X = np.load(X_path, mmap_mode='r')
    N = len(X)

    # Split (same logic as trainer)
    val_frac = float(cfg.data.get('val_fraction', 0.2))
    rng = np.random.default_rng(42)
    idxs = np.arange(N)
    rng.shuffle(idxs)
    n_val = int(N * val_frac)
    val_idxs = idxs[:n_val]
    tr_idxs = idxs[n_val:]

    # Datasets
    ds_val = NumpyWindowsDataset(X_path, y_t_path, y_ct_path, idxs=val_idxs)

    def collate_list(samples):
        return samples

    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, collate_fn=collate_list)

    # Evaluate validation set
    print(f"\n{'=' * 60}")
    print("VALIDATION SET EVALUATION")
    print(f"{'=' * 60}")

    y_true_val, y_probs_val, y_pred_val = collect_predictions(
        model, val_loader, device, threshold=args.threshold
    )

    metrics_val = compute_metrics(y_true_val, y_probs_val, y_pred_val)

    print("\nValidation Metrics:")
    print(f"  Accuracy:        {metrics_val['accuracy']:.4f}")
    print(f"  Balanced Acc:    {metrics_val['balanced_acc']:.4f}")
    print(f"  F1 Score:        {metrics_val['f1']:.4f}")
    print(f"  ROC-AUC:         {metrics_val['roc_auc']:.4f}")
    print(f"  PR-AUC (AP):     {metrics_val['pr_auc']:.4f}")

    # Save plots
    if len(np.unique(y_true_val)) >= 2:
        print("\nGenerating plots...")

        cm_fig = plot_confusion_matrix(y_true_val, y_pred_val, normalize=True)
        cm_fig.savefig(output_dir / 'confusion_matrix_val.png', dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_dir / 'confusion_matrix_val.png'}")

        roc_fig = plot_roc_curve(y_true_val, y_probs_val, metrics_val['roc_auc'])
        roc_fig.savefig(output_dir / 'roc_curve_val.png', dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_dir / 'roc_curve_val.png'}")

        pr_fig = plot_pr_curve(y_true_val, y_probs_val, metrics_val['pr_auc'])
        pr_fig.savefig(output_dir / 'pr_curve_val.png', dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_dir / 'pr_curve_val.png'}")

    # Optionally evaluate train set
    if not args.val_only:
        print(f"\n{'=' * 60}")
        print("TRAINING SET EVALUATION")
        print(f"{'=' * 60}")

        ds_train = NumpyWindowsDataset(X_path, y_t_path, y_ct_path, idxs=tr_idxs)
        train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=False,
                                  num_workers=2, collate_fn=collate_list)

        y_true_train, y_probs_train, y_pred_train = collect_predictions(
            model, train_loader, device, threshold=args.threshold
        )

        metrics_train = compute_metrics(y_true_train, y_probs_train, y_pred_train)

        print("\nTraining Metrics:")
        print(f"  Accuracy:        {metrics_train['accuracy']:.4f}")
        print(f"  Balanced Acc:    {metrics_train['balanced_acc']:.4f}")
        print(f"  F1 Score:        {metrics_train['f1']:.4f}")
        print(f"  ROC-AUC:         {metrics_train['roc_auc']:.4f}")
        print(f"  PR-AUC (AP):     {metrics_train['pr_auc']:.4f}")

        if len(np.unique(y_true_train)) >= 2:
            cm_fig = plot_confusion_matrix(y_true_train, y_pred_train, normalize=True)
            cm_fig.savefig(output_dir / 'confusion_matrix_train.png', dpi=150, bbox_inches='tight')

            roc_fig = plot_roc_curve(y_true_train, y_probs_train, metrics_train['roc_auc'])
            roc_fig.savefig(output_dir / 'roc_curve_train.png', dpi=150, bbox_inches='tight')

            pr_fig = plot_pr_curve(y_true_train, y_probs_train, metrics_train['pr_auc'])
            pr_fig.savefig(output_dir / 'pr_curve_train.png', dpi=150, bbox_inches='tight')

    print(f"\n{'=' * 60}")
    print("EVALUATION COMPLETE")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()