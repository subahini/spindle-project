# data_loader.py
"""
this will build pytorch data loader  (sample-level shapes):
  - x: [C, T]
  - y: [T]
"""

from typing import Tuple
from torch.utils.data import DataLoader ,WeightedRandomSampler
from dataset import build_splits

from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
import numpy as np

def make_loaders(cfg) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_set, val_set, test_set = build_splits(cfg)

    kind = getattr(cfg.data.sampler, "kind", "")

    if kind == "weighted":
        # ---- weighted sampler ----
        y = train_set.y if hasattr(train_set, "y") else train_set.labels
        has_pos = (y.sum(axis=1) > 0)
        pos_w = float(getattr(cfg.data.sampler, "pos_weight", 10.0))
        w = np.where(has_pos, pos_w, 1.0).astype(np.float64)

        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(w),
            num_samples=len(train_set),  # keep epoch size stable
            replacement=True
        )
        train_loader = DataLoader(
            train_set, batch_size=cfg.trainer.batch_size, sampler=sampler,
            num_workers=cfg.data.num_workers, pin_memory=True, drop_last=False
        )

    elif kind == "undersample":
        # ---- true undersampling ----
        y = train_set.y if hasattr(train_set, "y") else train_set.labels
        has_pos = (y.sum(axis=1) > 0)

        pos_idx = np.where(has_pos)[0]
        neg_idx = np.where(~has_pos)[0]

        ratio = float(getattr(cfg.data.sampler, "neg_pos_ratio", 2.0))
        k = int(len(pos_idx) * ratio)
        sel_neg = np.random.choice(neg_idx, size=min(k, len(neg_idx)), replace=False)
        sel = np.concatenate([pos_idx, sel_neg])

        train_set = Subset(train_set, sel)

        train_loader = DataLoader(
            train_set, batch_size=cfg.trainer.batch_size, shuffle=True,
            num_workers=cfg.data.num_workers, pin_memory=True, drop_last=False
        )

    else:
        # ---- plain shuffle ----
        train_loader = DataLoader(
            train_set, batch_size=cfg.trainer.batch_size, shuffle=True,
            num_workers=cfg.data.num_workers, pin_memory=True, drop_last=False
        )

    # validation / test loaders are always simple shuffle=False
    val_loader = DataLoader(
        val_set, batch_size=cfg.trainer.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=True, drop_last=False
    )
    test_loader = DataLoader(
        test_set, batch_size=cfg.trainer.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=True, drop_last=False
    )

    return train_loader, val_loader, test_loader
