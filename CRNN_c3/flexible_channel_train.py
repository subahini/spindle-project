import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import wandb
import yaml
from losses import build_loss_function
from Crnn import (
    CRNN2D_BiGRU,
    load_edf_windows,
    EEGDataset,
    split_data,
    run_eval,
    set_seed,
    ensure_dir,
)


def smart_weight_transfer(train_model, test_model, train_channels, test_channels, verbose=True):
    """
    Intelligently transfer weights between models with different input channels.
    Handles the first layer (stem) specially.
    """
    train_state = train_model.state_dict()
    test_state = test_model.state_dict()

    transferred = 0
    skipped = 0
    adapted = 0

    for name, train_param in train_state.items():
        if name not in test_state:
            skipped += 1
            continue

        test_param = test_state[name]

        # Check if this is the first convolutional layer in stem
        if "stem" in name and "conv.weight" in name:
            # This is the critical first layer - needs special handling
            if train_param.shape != test_param.shape:
                if verbose:
                    print(f"[Adapt] {name}: {train_param.shape} -> {test_param.shape}")

                # Weight shape: [out_channels, in_channels, kernel_h, kernel_w]
                train_in_ch = train_param.shape[1]
                test_in_ch = test_param.shape[1]

                if train_in_ch > test_in_ch:
                    # More training channels than test channels
                    # Take the first N channels (or average)
                    adapted_weight = train_param[:, :test_in_ch, :, :].clone()
                    test_state[name] = adapted_weight
                    adapted += 1
                    if verbose:
                        print(f"  → Took first {test_in_ch} channels from {train_in_ch}")

                elif test_in_ch > train_in_ch:
                    # Fewer training channels than test channels
                    # Replicate training channels
                    repeats = test_in_ch // train_in_ch
                    remainder = test_in_ch % train_in_ch

                    adapted_weight = train_param.repeat(1, repeats, 1, 1)
                    if remainder > 0:
                        adapted_weight = torch.cat([
                            adapted_weight,
                            train_param[:, :remainder, :, :]
                        ], dim=1)

                    test_state[name] = adapted_weight
                    adapted += 1
                    if verbose:
                        print(f"  → Replicated {train_in_ch} channels to {test_in_ch}")

                continue

        # For all other layers
        if train_param.shape == test_param.shape:
            test_state[name] = train_param
            transferred += 1
        else:
            skipped += 1
            if verbose and "weight" in name:
                print(f"[Skip] {name}: shape mismatch {train_param.shape} vs {test_param.shape}")

    # Load the adapted state
    test_model.load_state_dict(test_state, strict=False)

    if verbose:
        print(f"\n[Transfer Summary]")
        print(f"  Transferred: {transferred} layers")
        print(f"  Adapted: {adapted} layers")
        print(f"  Skipped: {skipped} layers")

    return test_model


def train_flexible_channels(cfg, train_channels, test_channels):
    """
    Train on train_channels and test on test_channels.
    Supports cross-channel generalization testing with smart weight transfer.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["project"].get("seed", 42))
    ensure_dir(cfg["paths"]["out_dir"])

    print(f"\n{'=' * 60}")
    print(f"[TRAIN] Training on: {train_channels}")
    print(f"[TEST] Testing on: {test_channels}")
    print(f"{'=' * 60}\n")

    # ========== 1. Load Full Dataset ==========
    dcfg = cfg["data"]
    sig = cfg.get("signal", {})

    # Check for NPY files first
    x_npy, y_npy = dcfg.get("x_npy"), dcfg.get("y_npy")
    if x_npy and y_npy and os.path.exists(x_npy) and os.path.exists(y_npy):
        print("[Data] Loading from NPY files...")
        X_full = np.load(x_npy, mmap_mode="r")
        y_full = np.load(y_npy, mmap_mode="r")
    else:
        print("[Data] Loading from EDF + JSON...")
        edf_dir = dcfg["edf"]["dir"]
        edfs = sorted([f for f in os.listdir(edf_dir) if f.lower().endswith(".edf")])
        if not edfs:
            raise RuntimeError(f"No EDF files found in {edf_dir}")
        X_full, y_full = load_edf_windows(
            os.path.join(edf_dir, edfs[0]),
            dcfg["edf"]["labels_json"],
            dcfg
        )

    all_channels = dcfg["channels"]
    print(f"[Data] Loaded shape: {X_full.shape}, Labels: {y_full.shape}")

    # ========== 2. Channel Selection ==========
    train_idx = [all_channels.index(ch) for ch in train_channels if ch in all_channels]
    test_idx = [all_channels.index(ch) for ch in test_channels if ch in all_channels]

    if not train_idx:
        raise ValueError(f"No valid train channels found in {all_channels}")
    if not test_idx:
        raise ValueError(f"No valid test channels found in {all_channels}")

    X_train_channels = X_full[:, train_idx, :]
    X_test_channels = X_full[:, test_idx, :]

    print(f"[Channels] Train: {len(train_idx)} channels {train_channels}")
    print(f"[Channels] Test: {len(test_idx)} channels {test_channels}")

    # ========== 3. Data Split ==========
    # Use the same seed for both to ensure aligned splits
    (Xtr, Ytr), (Xva, Yva), (Xte_unused, Yte_unused) = split_data(
        X_train_channels, y_full,
        ratios=(0.7, 0.15, 0.15),
        seed=cfg["project"].get("seed", 42)
    )

    _, _, (Xte, Yte) = split_data(
        X_test_channels, y_full,
        ratios=(0.7, 0.15, 0.15),
        seed=cfg["project"].get("seed", 42)
    )

    print(f"[Split] Train={len(Xtr)} | Val={len(Xva)} | Test={len(Xte)}")

    # ========== 4. Build Datasets ==========
    tr_ds = EEGDataset(Xtr, Ytr, normalize=sig.get("normalize", "zscore"),
                       reference=sig.get("reference", "car"))
    va_ds = EEGDataset(Xva, Yva, normalize=sig.get("normalize", "zscore"),
                       reference=sig.get("reference", "car"))
    te_ds = EEGDataset(Xte, Yte, normalize=sig.get("normalize", "zscore"),
                       reference=sig.get("reference", "car"))

    # ========== 5. Build Dataloaders with Sampling Strategy ==========
    bs = int(cfg["trainer"]["batch_size"])
    nw = int(cfg["trainer"]["num_workers"])

    sampler_mode = cfg["trainer"].get("sampler", "normal").lower()
    if sampler_mode == "undersample":
        labels = (Ytr.sum(1) > 0).astype(int)
        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]
        n = min(len(pos_idx), len(neg_idx))
        sel = np.concatenate([
            np.random.choice(pos_idx, n, False),
            np.random.choice(neg_idx, n, False)
        ])
        sampler = torch.utils.data.SubsetRandomSampler(sel)
        tr_loader = DataLoader(tr_ds, batch_size=bs, sampler=sampler,
                               num_workers=nw, pin_memory=True)
        print(f"[Sampler] Undersample: {len(sel)} samples ({n} pos, {n} neg)")
    elif sampler_mode == "weighted":
        labels = (Ytr.sum(1) > 0).astype(int)
        class_count = np.bincount(labels)
        w = 1.0 / class_count
        sample_w = w[labels]
        sampler = WeightedRandomSampler(sample_w, len(sample_w))
        tr_loader = DataLoader(tr_ds, batch_size=bs, sampler=sampler,
                               num_workers=nw, pin_memory=True)
        print(f"[Sampler] Weighted sampling")
    else:
        tr_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True,
                               num_workers=nw, pin_memory=True)
        print(f"[Sampler] Normal (random)")

    va_loader = DataLoader(va_ds, batch_size=bs, shuffle=False,
                           num_workers=nw, pin_memory=True)
    te_loader = DataLoader(te_ds, batch_size=bs, shuffle=False,
                           num_workers=nw, pin_memory=True)

    # ========== 6. Build Models ==========
    mcfg = cfg["model"]
    scfg = cfg["spectrogram"]

    # Training model
    train_model = CRNN2D_BiGRU(
        c_in=len(train_idx),
        base_ch=mcfg["base_ch"],
        fpn_ch=mcfg["fpn_ch"],
        rnn_hidden=mcfg["rnn_hidden"],
        rnn_layers=mcfg["rnn_layers"],
        bidirectional=mcfg["bidirectional"],
        bias_init_prior=mcfg.get("bias_init_prior"),
        use_se=mcfg["use_se"],
        sfreq=cfg["data"]["sfreq"],
        n_fft=scfg["n_fft"],
        hop_length=scfg["hop_length"],
        win_length=scfg["win_length"],
        center=scfg["center"],
        power=scfg["power"],
        upsample_mode=mcfg.get("upsample_mode", "linear")
    ).to(device)

    # Test model
    test_model = CRNN2D_BiGRU(
        c_in=len(test_idx),
        base_ch=mcfg["base_ch"],
        fpn_ch=mcfg["fpn_ch"],
        rnn_hidden=mcfg["rnn_hidden"],
        rnn_layers=mcfg["rnn_layers"],
        bidirectional=mcfg["bidirectional"],
        bias_init_prior=mcfg.get("bias_init_prior"),
        use_se=mcfg["use_se"],
        sfreq=cfg["data"]["sfreq"],
        n_fft=scfg["n_fft"],
        hop_length=scfg["hop_length"],
        win_length=scfg["win_length"],
        center=scfg["center"],
        power=scfg["power"],
        upsample_mode=mcfg.get("upsample_mode", "linear")
    ).to(device)

    train_params = sum(p.numel() for p in train_model.parameters()) / 1e6
    test_params = sum(p.numel() for p in test_model.parameters()) / 1e6
    print(f"[Model] Train: {train_params:.2f}M params ({len(train_idx)} channels)")
    print(f"[Model] Test: {test_params:.2f}M params ({len(test_idx)} channels)")

    # ========== 7. Loss Function & Optimizer ==========
    criterion = build_loss_function(
        cfg["loss"].get("name", "weighted_bce"),
        cfg["loss"],
        tr_loader
    )
    print(f"[Loss] Using {cfg['loss'].get('name', 'weighted_bce')}")

    optimizer = torch.optim.AdamW(
        train_model.parameters(),
        lr=cfg["trainer"]["lr"],
        weight_decay=cfg["trainer"]["weight_decay"]
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda") and bool(cfg["trainer"]["amp"])
    )

    # ========== 8. W&B Init ==========
    wb = cfg.get("logging", {}).get("wandb", {})
    run_name = f"train_{'_'.join(train_channels)}_test_{'_'.join(test_channels)}"

    if wb.get("enabled", True):
        wandb.init(
            project=wb.get("project", cfg["project"].get("name")),
            entity=wb.get("entity", cfg["project"].get("entity")),
            name=run_name,
            tags=wb.get("tags", []) + ["flexible_channels"],
            config={
                **cfg,
                "train_channels": train_channels,
                "test_channels": test_channels,
                "n_train_channels": len(train_idx),
                "n_test_channels": len(test_idx),
            }
        )

    # ========== 9. Training Loop ==========
    best_f1 = -1.0
    epochs = cfg["trainer"]["epochs"]
    ckpt_path = os.path.join(cfg["paths"]["out_dir"], "flex_best.pt")
    step = 0

    print(f"\n[Training] Starting {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        train_model.train()
        epoch_loss = 0.0
        n_batches = 0

        for xb, yb in tr_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                logits = train_model(xb)
                loss = criterion(logits.unsqueeze(1), yb.unsqueeze(1))

            scaler.scale(loss).backward()
            if cfg["trainer"]["grad_clip"]:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    train_model.parameters(),
                    cfg["trainer"]["grad_clip"]
                )
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1
            step += 1

            if step % cfg["log"]["every_steps"] == 0:
                if wandb.run:
                    wandb.log({
                        "train/loss": loss.item(),
                        "epoch": epoch,
                        "step": step
                    })

        avg_loss = epoch_loss / n_batches
        # ===== Evaluate on training set =====
        train_model.eval()
        with torch.no_grad():
            _, _, train_metrics = run_eval(tr_loader, cfg, train_model, device, split_name="train")

        # Display training metrics
        print(f"[Epoch {epoch}/{epochs}] "
              f"Train: F1={train_metrics['f1']:.3f} "
              f"P={train_metrics['precision']:.3f} "
              f"R={train_metrics['recall']:.3f} "
              f"ROC={train_metrics['roc_auc']:.3f} "
              f"PR={train_metrics['pr_auc']:.3f}")

        # ========== 10. Validation ==========
        _, _, val_metrics = run_eval(va_loader, cfg, train_model, device, split_name="val")

        print(f"[Epoch {epoch}/{epochs}] "
              f"Loss={avg_loss:.4f} | "
              f"Val: thr={val_metrics['threshold']:.2f} "
              f"F1={val_metrics['f1']:.3f} "
              f"P={val_metrics['precision']:.3f} "
              f"R={val_metrics['recall']:.3f} "
              f"Acc={val_metrics['accuracy']:.3f}")

        if wandb.run:
            log_payload = {
                "epoch": epoch,
                "train/loss_epoch": avg_loss,
                "train/f1": train_metrics.get("f1"),
                "train/precision": train_metrics.get("precision"),
                "train/recall": train_metrics.get("recall"),
                "train/roc_auc": train_metrics.get("roc_auc"),
                "train/pr_auc": train_metrics.get("pr_auc"),
                "val/threshold": val_metrics.get("threshold"),
                "val/f1": val_metrics.get("f1"),
                "val/precision": val_metrics.get("precision"),
                "val/recall": val_metrics.get("recall"),
                "val/accuracy": val_metrics.get("accuracy"),
                "val/roc_auc": val_metrics.get("roc_auc"),
                "val/pr_auc": val_metrics.get("pr_auc"),
            }
            wandb.log({k: v for k, v in log_payload.items() if v is not None})

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save({
                "model": train_model.state_dict(),
                "cfg": cfg,
                "train_channels": train_channels,
                "test_channels": test_channels,
                "epoch": epoch,
                "f1": best_f1
            }, ckpt_path)
            print(f"[Checkpoint] Saved best model (F1={best_f1:.3f})")

    # ========== 11. Smart Weight Transfer & Testing ==========
    print(f"\n{'=' * 60}")
    print("[Transfer] Loading best model and transferring to test model...")
    print(f"{'=' * 60}\n")

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        train_model.load_state_dict(ckpt["model"])

    # Use smart weight transfer
    test_model = smart_weight_transfer(
        train_model, test_model,
        train_channels, test_channels,
        verbose=True
    )

    # Test evaluation
    _, _, test_metrics = run_eval(te_loader, cfg, test_model, device, split_name="test")

    print(f"\n{'=' * 60}")
    print(f"[RESULTS] Cross-Channel Testing")
    print(f"{'=' * 60}")
    print(f"  Train Channels: {train_channels}")
    print(f"  Test Channels: {test_channels}")
    print(f"  Threshold: {test_metrics['threshold']:.2f}")
    print(f"  F1: {test_metrics['f1']:.3f}")
    print(f"  Precision: {test_metrics['precision']:.3f}")
    print(f"  Recall: {test_metrics['recall']:.3f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.3f}")
    print(f"  ROC-AUC: {test_metrics['roc_auc']:.3f}")
    print(f"  PR-AUC: {test_metrics['pr_auc']:.3f}")
    print(f"{'=' * 60}\n")

    if wandb.run:
        test_payload = {
            "test/threshold": test_metrics.get("threshold"),
            "test/precision": test_metrics.get("precision"),
            "test/recall": test_metrics.get("recall"),
            "test/f1": test_metrics.get("f1"),
            "test/accuracy": test_metrics.get("accuracy"),
            "test/roc_auc": test_metrics.get("roc_auc"),
            "test/pr_auc": test_metrics.get("pr_auc"),
        }
        wandb.log({k: v for k, v in test_payload.items() if v is not None})

        wandb.run.summary.update({
            "final_val_f1": best_f1,
            "final_test_f1": test_metrics["f1"],
            "train_channels": train_channels,
            "test_channels": test_channels,
        })

        wandb.finish()

    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CRNN on specific channels and test on different channels"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--train-channels", nargs="+", required=True,
                        help="Channels to use for training")
    parser.add_argument("--test-channels", nargs="+", required=True,
                        help="Channels to use for testing")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    train_flexible_channels(cfg, args.train_channels, args.test_channels)