# cnn_train.py
"""
2D CNN Spindle Detection Training Script

- Mode "subject_edf": use multiple EDF+JSON files from ONE subject (EDF_JSON_LIST)
- Mode "npy": use precomputed X_all.npy / y_all.npy (full dataset)

Depends on:
- config.Config
- models.SpindleCNN  (you already have this)
- losses.build_loss_function
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List

import mne
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from torch import optim, nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

import wandb

from config import Config
from models import SpindleCNN
from losses import build_loss_function


# ----------------- METRICS -----------------


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    avg_loss: float,
) -> Dict[str, float]:
    """
    y_true, y_pred: binary arrays (0/1)
    """
    y_true = y_true.astype(int).ravel()
    y_pred = y_pred.astype(int).ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp + 1e-8)

    return {
        "avg_loss": float(avg_loss),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "specificity": float(specificity),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


# ----------------- EDF + JSON → WINDOWS -----------------


def _load_spindles_from_json(
    json_path: Path,
    key_start: str = "start",
    key_end: str = "end",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Very generic JSON reader.

    Expects JSON to be either:
    - {"spindles": [{"start": <sec>, "end": <sec>}, ...]
    - or a plain list: [{"start": <sec>, "end": <sec>}, ...]

    Returns two arrays (starts_sec, ends_sec).
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "spindles" in data:
        events = data["spindles"]
    else:
        events = data

    starts = []
    ends = []
    for ev in events:
        if key_start in ev and key_end in ev:
            starts.append(float(ev[key_start]))
            ends.append(float(ev[key_end]))

    return np.asarray(starts, dtype=float), np.asarray(ends, dtype=float)


def _windows_from_one_edf_json(
    edf_path: Path,
    json_path: Path,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build X, y from ONE EDF + JSON file.

    X shape: (N_windows, 1, n_channels, window_samples)
    y shape: (N_windows, 1)
    """
    if not edf_path.exists():
        raise FileNotFoundError(f"EDF not found: {edf_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    print(f"Processing EDF: {edf_path.name}, JSON: {json_path.name}")

    # --- load raw ---
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")

    if cfg.CHANNEL_NAMES:
        ch_to_pick = [ch for ch in cfg.CHANNEL_NAMES if ch in raw.ch_names]
        if len(ch_to_pick) == 0:
            raise ValueError(
                f"None of the requested channels {cfg.CHANNEL_NAMES} are in EDF {edf_path}."
            )
        raw.pick_channels(ch_to_pick)
    else:
        n = min(cfg.N_CHANNELS, len(raw.ch_names))
        raw.pick(raw.ch_names[:n])

    sfreq = raw.info["sfreq"]
    # band-pass filter
    raw.filter(cfg.HP_FREQ, cfg.LP_FREQ, fir_design="firwin", verbose="ERROR")

    data = raw.get_data()  # (n_channels, n_samples)
    n_channels, n_samples = data.shape

    # --- label vector per sample ---
    starts_sec, ends_sec = _load_spindles_from_json(json_path)
    labels_per_sample = np.zeros(n_samples, dtype=np.float32)

    for s_sec, e_sec in zip(starts_sec, ends_sec):
        s_idx = int(max(0, np.floor(s_sec * sfreq)))
        e_idx = int(min(n_samples, np.ceil(e_sec * sfreq)))
        if e_idx > s_idx:
            labels_per_sample[s_idx:e_idx] = 1.0

    # --- windowing ---
    win_size = int(cfg.WINDOW_SEC * sfreq)
    step_size = int(cfg.STEP_SEC * sfreq)
    if win_size <= 0 or step_size <= 0:
        raise ValueError("WINDOW_SEC and STEP_SEC must be > 0")

    windows: List[np.ndarray] = []
    win_labels: List[List[float]] = []

    for start in range(0, n_samples - win_size + 1, step_size):
        end = start + win_size
        segment = data[:, start:end]  # (n_channels, win_size)
        seg_labels = labels_per_sample[start:end]
        frac_spindle = seg_labels.mean()

        label = 1.0 if frac_spindle >= cfg.OVERLAP_THRESHOLD else 0.0
        # add "image channel" dim = 1, final shape (1, n_channels, win_size)
        windows.append(segment[None, :, :])
        win_labels.append([label])

    if not windows:
        raise RuntimeError(
            f"No windows created from EDF {edf_path}. Check window/step sizes."
        )

    X = np.stack(windows, axis=0)      # (N, 1, n_channels, win_size)
    y = np.asarray(win_labels, dtype=np.float32)  # (N, 1)

    X = X.astype(np.float32)
    return X, y


def build_subject_windows_from_edf_json(cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loops over all (EDF, JSON) pairs in cfg.EDF_JSON_LIST for ONE subject
    and concatenates all windows.
    """
    all_X = []
    all_y = []

    for edf_str, json_str in cfg.EDF_JSON_LIST:
        edf_path = Path(edf_str)
        json_path = Path(json_str)
        X_i, y_i = _windows_from_one_edf_json(edf_path, json_path, cfg)
        all_X.append(X_i)
        all_y.append(y_i)

    if not all_X:
        raise RuntimeError("EDF_JSON_LIST is empty. Set it in config.Config.")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    return X, y


# ----------------- NPY MODE LOADING -----------------


def load_from_npy(cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    data_dir = Path(cfg.DATA_DIR)
    x_path = data_dir / cfg.X_FILE
    y_path = data_dir / cfg.Y_FILE

    if not x_path.exists():
        raise FileNotFoundError(f"X data not found: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"y data not found: {y_path}")

    X = np.load(x_path)  # expected (N, C, T) or (N, 1, H, W)
    y = np.load(y_path)

    X = X.astype(np.float32)
    y = y.astype(np.float32).reshape(-1, 1)

    # Convert to (N, 1, H, W) for CNN.
    if X.ndim == 3:
        # assume (N, C, T) from 1D pipelines -> treat C as "height", T as "width"
        X = X[:, None, :, :]
    elif X.ndim == 4:
        # either already (N, 1, H, W) or (N, H, W, C)
        if X.shape[1] not in (1, 3) and X.shape[-1] in (1, 3):
            # likely (N, H, W, C) -> transpose
            X = np.transpose(X, (0, 3, 1, 2))
    else:
        raise ValueError(f"Unsupported X shape: {X.shape}")

    return X, y


def create_splits(
    X: np.ndarray,
    y: np.ndarray,
    cfg: Config,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Returns (X_train, y_train), (X_val, y_val), (X_test, y_test).
    Stratified on y if possible.
    """
    try:
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE, stratify=y
        )
        val_fraction = cfg.VAL_SIZE / (1.0 - cfg.TEST_SIZE)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_fraction,
            random_state=cfg.RANDOM_STATE, stratify=y_trainval
        )
    except ValueError:
        # if stratify fails (e.g. all labels same), do non-stratified
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE
        )
        val_fraction = cfg.VAL_SIZE / (1.0 - cfg.TEST_SIZE)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_fraction,
            random_state=cfg.RANDOM_STATE,
        )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def create_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    num_workers: int,
    sampling: str = "none",
) -> DataLoader:
    """
    sampling: "none" or "weighted_sampler"
    """
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)

    dataset = TensorDataset(X_tensor, y_tensor)

    if sampling == "weighted_sampler":
        labels = y_tensor.view(-1).long()
        num_pos = labels.sum().item()
        num_total = labels.numel()
        num_neg = num_total - num_pos

        if num_pos > 0:
            w_pos = num_neg / max(num_pos, 1)
            w_neg = 1.0
            weights = torch.where(labels == 1, w_pos, w_neg).float()
            sampler = WeightedRandomSampler(weights, num_samples=num_total, replacement=True)

            return DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
            )

    # default: plain shuffled loader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )


# ----------------- TRAIN / EVAL LOOPS -----------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    threshold: float,
) -> Dict[str, Any]:
    model.train()
    total_loss = 0.0
    all_targets = []
    all_preds = []

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(inputs)  # (B, 1)
        logits = logits.view_as(targets)

        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

        all_targets.append(targets.detach().cpu().numpy())
        all_preds.append(preds.detach().cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    avg_loss = total_loss / len(loader.dataset)

    metrics = compute_metrics(y_true, y_pred, avg_loss)
    return metrics


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(inputs)
            logits = logits.view_as(targets)

            loss = criterion(logits, targets)
            total_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    avg_loss = total_loss / len(loader.dataset)

    metrics = compute_metrics(y_true, y_pred, avg_loss)
    return metrics


# ----------------- ARGPARSE + MAIN -----------------


def parse_args():
    parser = argparse.ArgumentParser(description="2D CNN Spindle Detection Training")

    # sweep-controllable hyperparameters
    parser.add_argument(
        "--loss_name",
        type=str,
        default="weighted_bce",
        choices=["weighted_bce", "focal", "dice", "bce"],
    )
    parser.add_argument(
        "--sampling",
        type=str,
        default="none",
        choices=["none", "weighted_sampler"],
    )
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)

    # wandb
    parser.add_argument("--project", type=str, default="spindle_cnn2d")
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)

    return parser.parse_args()


def main():
    cfg = Config()
    args = parse_args()

    # override config from CLI if provided (for sweeps)
    if args.learning_rate is not None:
        cfg.LEARNING_RATE = args.learning_rate
    if args.weight_decay is not None:
        cfg.WEIGHT_DECAY = args.weight_decay
    if args.batch_size is not None:
        cfg.BATCH_SIZE = args.batch_size

    device = torch.device(cfg.DEVICE)
    print(f"Using device: {device}")

    # init wandb
    wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        config={
            "model": "SpindleCNN",
            "data_mode": cfg.DATA_MODE,
            "loss_name": args.loss_name,
            "sampling": args.sampling,
            "learning_rate": cfg.LEARNING_RATE,
            "weight_decay": cfg.WEIGHT_DECAY,
            "batch_size": cfg.BATCH_SIZE,
            "num_epochs": cfg.NUM_EPOCHS,
            "threshold": cfg.THRESHOLD,
            "window_sec": cfg.WINDOW_SEC,
            "step_sec": cfg.STEP_SEC,
            "hp_freq": cfg.HP_FREQ,
            "lp_freq": cfg.LP_FREQ,
            "n_edf_files": len(cfg.EDF_JSON_LIST),
        },
    )

    # ----- data -----
    if cfg.DATA_MODE == "subject_edf":
        print("Building windows from EDF+JSON files for ONE subject...")
        X, y = build_subject_windows_from_edf_json(cfg)
    elif cfg.DATA_MODE == "npy":
        print(f"Loading data from NPY files in {cfg.DATA_DIR}")
        X, y = load_from_npy(cfg)
    else:
        raise ValueError(f"Unknown DATA_MODE: {cfg.DATA_MODE}")

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = create_splits(X, y, cfg)

    train_loader = create_loader(
        X_train, y_train,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        sampling=args.sampling,
    )
    val_loader = create_loader(
        X_val, y_val,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        sampling="none",
    )
    test_loader = create_loader(
        X_test, y_test,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        sampling="none",
    )

    # ----- model & loss -----
    model = SpindleCNN().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer_cfg = {
        "focal": {
            "alpha": 0.15,
            "gamma": 3.0,
            "label_smoothing": 0.05,
        },
        "weighted_bce": {
            "pos_weight": "auto",  # estimate from training loader
            "label_smoothing": 0.03,
            "adaptive": False,
            "target_pos_rate": 0.1,
        },
        "dice": {
            "smooth": 1.0,
        },
    }

    criterion = build_loss_function(
        name=args.loss_name,
        trainer_cfg=trainer_cfg,
        train_loader=train_loader,
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        verbose=True,
    )

    # ----- training loop with early stopping -----
    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(cfg.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{cfg.NUM_EPOCHS}")

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, cfg.THRESHOLD
        )
        val_metrics = evaluate(
            model, val_loader, criterion, device, cfg.THRESHOLD
        )

        # step scheduler on validation F1
        scheduler.step(val_metrics["f1"])

        print(f"Train: loss={train_metrics['avg_loss']:.4f} "
              f"F1={train_metrics['f1']:.4f} "
              f"Prec={train_metrics['precision']:.4f} "
              f"Rec={train_metrics['recall']:.4f}")
        print(f"Val  : loss={val_metrics['avg_loss']:.4f} "
              f"F1={val_metrics['f1']:.4f} "
              f"Prec={val_metrics['precision']:.4f} "
              f"Rec={val_metrics['recall']:.4f}")

        # wandb logging
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_metrics["avg_loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "val_loss": val_metrics["avg_loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_specificity": val_metrics["specificity"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        })

        # early stopping
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    wandb.summary["best_val_f1"] = best_val_f1

    # ----- evaluate best model on test set -----
    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(
        model, test_loader, criterion, device, cfg.THRESHOLD
    )

    print("\n=== FINAL TEST METRICS ===")
    for k, v in test_metrics.items():
        if k in ("tp", "fp", "tn", "fn"):
            print(f"{k}: {v:.0f}")
        else:
            print(f"{k}: {v:.4f}")

    wandb.log({f"test_{k}": v for k, v in test_metrics.items()})


if __name__ == "__main__":
    main()
