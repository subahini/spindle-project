#!/usr/bin/env python3
"""
Cross-Channel Generalization Training for CRNN Spindle Detection
Train on one channel (c3 c4 f3 f4)
"""

import os
import json
import argparse
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import wandb
import yaml

from losses import build_loss_function
import Crnn  # Import for model and utils


# ----------------------------------------------------------
# Dataset + helper functions
# ----------------------------------------------------------

class SingleChannelDataset(Dataset):
    """Dataset for a single EEG channel"""
    def __init__(self, X, y, channel_idx: int, normalize="zscore"):
        self.X = X[:, channel_idx:channel_idx + 1, :].astype(np.float32)
        self.y = y.astype(np.float32)
        self.normalize = normalize

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i]
        if self.normalize == "zscore":
            mu = x.mean(axis=-1, keepdims=True)
            sd = x.std(axis=-1, keepdims=True) + 1e-6
            x = (x - mu) / sd
        return torch.from_numpy(x), torch.from_numpy(self.y[i])


def get_channel_index(ch, all_channels):
    try:
        return all_channels.index(ch)
    except ValueError:
        raise ValueError(f"Channel {ch} not found in {all_channels}")


def build_single_channel_dataloaders(cfg, channel_name, all_channels):
    """Load data and create train/val/test dataloaders for one channel"""
    d = cfg["data"]
    bs = int(cfg["trainer"]["batch_size"])
    nw = int(cfg["trainer"]["num_workers"])
    seed = int(cfg.get("seed", 42))

    ch_idx = get_channel_index(channel_name, all_channels)
    print(f"[Channel] Using {channel_name} (index {ch_idx})")

    # Load NPY or EDF
    x_npy, y_npy = d.get("x_npy"), d.get("y_npy")
    if x_npy and y_npy and os.path.exists(x_npy) and os.path.exists(y_npy):
        X = np.load(x_npy, mmap_mode="r")
        y = np.load(y_npy, mmap_mode="r")
    else:
        edf_dir = d["edf"]["dir"]
        edfs = sorted([f for f in os.listdir(edf_dir) if f.lower().endswith(".edf")])
        if not edfs:
            raise RuntimeError(f"No EDF files found in {edf_dir}")
        X, y = Crnn.load_edf_windows(
            os.path.join(edf_dir, edfs[0]),
            d["edf"]["labels_json"],
            d
        )

    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = Crnn.split_data(X, y, seed=seed)

    train_ds = SingleChannelDataset(Xtr, Ytr, ch_idx)
    val_ds = SingleChannelDataset(Xva, Yva, ch_idx)
    test_ds = SingleChannelDataset(Xte, Yte, ch_idx)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    return train_loader, val_loader, test_loader


def wandb_safe_log(data: dict):
    """Avoid crash if wandb is not active"""
    if wandb.run is not None:
        wandb.log(data)


# ----------------------------------------------------------
# Training and Evaluation
# ----------------------------------------------------------

def train_cross_channel(cfg, train_channel="C3", test_channels=["C4", "F3", "F4"]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Crnn.set_seed(cfg["project"].get("seed", 42))
    Crnn.ensure_dir(cfg["paths"]["out_dir"])

    all_channels = cfg["data"].get("channels", Crnn.ALL_EEG_19)
    for ch in [train_channel] + test_channels:
        if ch not in all_channels:
            raise ValueError(f"{ch} not found in available channels.")

    # W&B setup
    wb = cfg.get("logging", {}).get("wandb", {})
    if wb.get("enabled", True):
        run_name = f"cross_channel_train_{train_channel}"
        wandb.init(
            project=wb.get("project", "CRNN-cross-channel"),
            entity=wb.get("entity"),
            name=run_name,
            tags=wb.get("tags", []) + [f"train:{train_channel}"],
            config={"train_channel": train_channel, "test_channels": test_channels}
        )
    else:
        print("[INFO] WandB logging disabled — running locally.")

    # ---------------------------------------
    # Build dataloaders for training channel
    # ---------------------------------------
    train_loader, val_loader, test_loader = build_single_channel_dataloaders(cfg, train_channel, all_channels)

    # Model
    mcfg = cfg["model"]
    scfg = cfg["spectrogram"]

    model = Crnn.CRNN2D_BiGRU(
        c_in=1, base_ch=mcfg["base_ch"], fpn_ch=mcfg["fpn_ch"],
        rnn_hidden=mcfg["rnn_hidden"], rnn_layers=mcfg["rnn_layers"],
        bidirectional=mcfg["bidirectional"], bias_init_prior=mcfg.get("bias_init_prior", None),
        use_se=mcfg["use_se"], sfreq=cfg["data"]["sfreq"],
        n_fft=scfg["n_fft"], hop_length=scfg["hop_length"], win_length=scfg["win_length"],
        center=scfg["center"], power=scfg["power"], upsample_mode=mcfg.get("upsample_mode", "linear")
    ).to(device)

    # Optimizer & loss
    opt = torch.optim.AdamW(model.parameters(),
                            lr=cfg["trainer"]["lr"],
                            weight_decay=cfg["trainer"]["weight_decay"])
    criterion = build_loss_function(cfg["loss"]["name"], cfg["loss"], train_loader)

    print(f"[Model] Single-channel CRNN with {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    print(f"[Loss] {cfg['loss']['name']}")

    best_f1 = -1
    ckpt_path = os.path.join(cfg["paths"]["out_dir"], f"best_{train_channel}.pt")

    # ------------------ Training loop ------------------
    for epoch in range(1, cfg["trainer"]["epochs"] + 1):
        model.train()
        total_loss, t0 = 0, time.time()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits.unsqueeze(1), yb.unsqueeze(1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["trainer"]["grad_clip"])
            opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation metrics using run_eval (returns 3 outputs)
        _, _, val_metrics = Crnn.run_eval(val_loader, cfg, model, device, split_name=f"val_{train_channel}")

        wandb_safe_log({
            "epoch": epoch,
            "train/loss": avg_loss,
            f"val_{train_channel}/f1": val_metrics["f1"]
        })

        print(f"[Epoch {epoch:2d}] {time.time()-t0:.1f}s | Loss={avg_loss:.4f} | "
              f"Val F1={val_metrics['f1']:.3f}")

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save({
                "model": model.state_dict(),
                "cfg": cfg,
                "train_channel": train_channel,
                "test_channels": test_channels,
                "best_f1": best_f1
            }, ckpt_path)
            print(f"✓ Saved best model (F1={best_f1:.3f})")

    # ------------------ Cross-channel evaluation ------------------
    print("\n" + "="*60)
    print("Loading best model for cross-channel evaluation...")
    print("="*60 + "\n")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    d = cfg["data"]
    if d.get("x_npy") and d.get("y_npy") and os.path.exists(d["x_npy"]) and os.path.exists(d["y_npy"]):
        X_full = np.load(d["x_npy"], mmap_mode="r")
        y_full = np.load(d["y_npy"], mmap_mode="r")
    else:
        edf_dir = d["edf"]["dir"]
        edfs = sorted([f for f in os.listdir(edf_dir) if f.lower().endswith(".edf")])
        X_full, y_full = Crnn.load_edf_windows(
            os.path.join(edf_dir, edfs[0]), d["edf"]["labels_json"], d
        )

    _, _, (X_test, y_test) = Crnn.split_data(X_full, y_full, seed=cfg["project"].get("seed", 42))

    results = {}
    print(f"\nEvaluating on TRAINING channel: {train_channel}")
    tr_idx = get_channel_index(train_channel, all_channels)
    ds_train = SingleChannelDataset(X_test, y_test, tr_idx, normalize="zscore")
    loader_train = DataLoader(ds_train, batch_size=cfg["trainer"]["batch_size"], shuffle=False, num_workers=0)
    _, _, metrics = Crnn.run_eval(loader_train, cfg, model, device, split_name=f"cross_{train_channel}")
    results[train_channel] = metrics

    print(f"  {train_channel}: F1={metrics['f1']:.3f} "
          f"P={metrics['precision']:.3f} R={metrics['recall']:.3f}")

    print(f"\nEvaluating on TEST channels (cross-channel generalization):")
    for ch in test_channels:
        ch_idx = get_channel_index(ch, all_channels)
        ds = SingleChannelDataset(X_test, y_test, ch_idx, normalize="zscore")
        loader = DataLoader(ds, batch_size=cfg["trainer"]["batch_size"], shuffle=False, num_workers=0)

        _, _, m = Crnn.run_eval(loader, cfg, model, device, split_name=f"cross_{ch}")
        results[ch] = m
        print(f"  {ch}: F1={m['f1']:.3f} P={m['precision']:.3f} R={m['recall']:.3f}")

    # ------------------ Summary ------------------
    print("\n" + "="*60)
    print("CROSS-CHANNEL GENERALIZATION SUMMARY")
    print("="*60)
    print(f"Training channel: {train_channel}")
    print(f"  F1: {results[train_channel]['f1']:.3f}")
    test_f1s = [results[ch]["f1"] for ch in test_channels]
    print(f"\nGeneralization (average over test channels):")
    print(f"  Mean F1: {np.mean(test_f1s):.3f} ± {np.std(test_f1s):.3f}")
    print("="*60)

    if wandb.run:
        wandb.log({
            "final_train_f1": results[train_channel]["f1"],
            "mean_test_f1": np.mean(test_f1s),
            "std_test_f1": np.std(test_f1s)
        })
        wandb.finish()

    # Save JSON results
    out_path = os.path.join(cfg["paths"]["out_dir"], f"cross_channel_results_{train_channel}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {out_path}")

    return results


# ----------------------------------------------------------
# Entry Point
# ----------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--train-channel", type=str, default="C3")
    parser.add_argument("--test-channels", nargs="+", default=["C4", "F3", "F4"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_cross_channel(cfg, args.train_channel, args.test_channels)
