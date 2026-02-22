import json
import yaml
import wandb
import mne
import numpy as np
import torch
import re
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold

from models import SpindleCNN
from losses import build_loss_function

import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
)
# ----------------------------
# Helpers: pairing + subject id
# ----------------------------
def subject_id_from_path(edf_path: str) -> str:
    return Path(edf_path).stem.split("_")[0]  # P002_3_raw -> P002


def build_pairs_from_dirs(edf_dir: str, json_dir: str):
    edf_dir = Path(edf_dir)
    json_dir = Path(json_dir)

    edfs = sorted(edf_dir.glob("*.edf"))
    jsons = sorted(json_dir.glob("*.json"))

    json_map = {}
    for jp in jsons:
        m = re.search(r"(P\d+_\d+)", jp.stem)
        if m:
            json_map[m.group(1)] = jp

    pairs = []
    for ep in edfs:
        m = re.search(r"(P\d+_\d+)", ep.stem)
        if not m:
            continue
        key = m.group(1)
        if key in json_map:
            pairs.append((str(ep), str(json_map[key])))

    if not pairs:
        raise ValueError("No EDF/JSON pairs found in folders.")
    return pairs


def get_pairs_from_config(cfg):
    sf = cfg["subject_files"]
    if len(sf) == 1 and "edf_dir" in sf[0] and "json_dir" in sf[0]:
        return build_pairs_from_dirs(sf[0]["edf_dir"], sf[0]["json_dir"])
    return [(p["edf"], p["json"]) for p in sf]


# ----------------------------
# JSON spindle loader
# ----------------------------
def load_spindles_from_json(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)

    spindles = data.get("detected_spindles") or data.get("spindles") or []
    iterable = spindles.values() if isinstance(spindles, dict) else spindles if isinstance(spindles, list) else []

    pairs = []
    for ev in iterable:
        if isinstance(ev, dict) and "start" in ev and "end" in ev:
            try:
                pairs.append((float(ev["start"]), float(ev["end"])))
            except Exception:
                pass
    pairs.sort(key=lambda x: x[0])
    return pairs


# ----------------------------
# EDF -> data + sample labels (NO windowing here)
# ----------------------------
def preprocess_raw(raw: mne.io.BaseRaw, channels, hp, lp):
    raw.pick(channels)
    raw.set_eeg_reference("average", verbose=False)
    raw.filter(hp, lp, verbose=False)
    return raw.get_data()  # (C, T)


def edf_to_data_and_labels(edf_path: str, json_path: str, cfg):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    sfreq = float(raw.info["sfreq"])

    data = preprocess_raw(
        raw,
        cfg["channels"]["names"],
        cfg["filter"]["hp_freq"],
        cfg["filter"]["lp_freq"],
    )

    labels = np.zeros(data.shape[1], dtype=np.float32)
    for s_sec, e_sec in load_spindles_from_json(json_path):
        s = int(max(0, np.floor(s_sec * sfreq)))
        e = int(min(labels.shape[0], np.ceil(e_sec * sfreq)))
        if e > s:
            labels[s:e] = 1.0

    return data.astype(np.float32), labels, sfreq


# ----------------------------
# Split FIRST, then window
# ----------------------------
def split_by_time(data, labels, sfreq, train_frac=0.70, val_frac=0.15, test_frac=0.15):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    T = data.shape[1]
    t_train = int(T * train_frac)
    t_val   = int(T * (train_frac + val_frac))

    d_tr, y_tr = data[:, :t_train], labels[:t_train]
    d_va, y_va = data[:, t_train:t_val], labels[t_train:t_val]
    d_te, y_te = data[:, t_val:], labels[t_val:]

    return (d_tr, y_tr), (d_va, y_va), (d_te, y_te)


def make_windows(data, labels, sfreq, win_s, step_s, overlap_thr):
    win = int(win_s * sfreq)
    step = int(step_s * sfreq)

    X_list, y_list = [], []
    T = data.shape[1]

    for start in range(0, T - win + 1, step):
        end = start + win
        seg = data[:, start:end]
        frac = labels[start:end].mean()
        y_win = 1.0 if frac >= overlap_thr else 0.0

        X_list.append(seg[None, :, :])  # (1, C, win)
        y_list.append([y_win])

    X = np.stack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.float32)

    # z-score per-window per-channel
    X = (X - X.mean(axis=-1, keepdims=True)) / (X.std(axis=-1, keepdims=True) + 1e-6)
    return X, y


def windowize_split(split_tuple, sfreq, cfg):
    data, labels = split_tuple
    return make_windows(
        data, labels, sfreq,
        cfg["windowing"]["window_sec"],
        cfg["windowing"]["step_sec"],
        cfg["windowing"]["overlap_threshold"],
    )


# ----------------------------
# Case A: ONE SUBJECT (even if multiple EDF segments)
# Split-by-time after concatenating raw signals (still split before windowing)
# ----------------------------
def load_one_subject_concat_raw(pairs, cfg):
    # pairs are all EDF segments for this subject
    datas, labels_list = [], []
    sfreq_ref = None

    for edf, js in pairs:
        d, y, sf = edf_to_data_and_labels(edf, js, cfg)
        if sfreq_ref is None:
            sfreq_ref = sf
        elif abs(sf - sfreq_ref) > 1e-6:
            raise ValueError("Mismatched sampling frequencies across EDF segments.")
        datas.append(d)
        labels_list.append(y)

    data = np.concatenate(datas, axis=1)       # concat in time
    labels = np.concatenate(labels_list, axis=0)
    return data, labels, sfreq_ref


def build_splits_single_subject(pairs, cfg):
    # split raw time first, then window
    data, labels, sfreq = load_one_subject_concat_raw(pairs, cfg)

    train_frac = float(cfg["splits"].get("train_size", 0.70))
    val_frac   = float(cfg["splits"].get("val_size", 0.15))
    test_frac  = float(cfg["splits"].get("test_size", 0.15))

    tr, va, te = split_by_time(data, labels, sfreq, train_frac, val_frac, test_frac)

    Xt, yt = windowize_split(tr, sfreq, cfg)
    Xv, yv = windowize_split(va, sfreq, cfg)
    Xte, yte = windowize_split(te, sfreq, cfg)
    return (Xt, yt), (Xv, yv), (Xte, yte)


# ----------------------------
# Case B: MULTI SUBJECT 5-FOLD (GroupKFold) — split by subject FIRST, then window
# ----------------------------
def build_splits_subject_kfold(pairs, cfg, fold_idx, n_folds=5):
    groups = [subject_id_from_path(edf) for edf, _ in pairs]
    unique_subjects = sorted(set(groups))
    if len(unique_subjects) < n_folds:
        raise ValueError(f"Need >= {n_folds} subjects for {n_folds}-fold, got {len(unique_subjects)}")

    gkf = GroupKFold(n_splits=n_folds)
    splits = list(gkf.split(pairs, groups=groups))
    train_idx, test_idx = splits[fold_idx]

    train_pairs = [pairs[i] for i in train_idx]
    test_pairs  = [pairs[i] for i in test_idx]

    # pick ONE validation subject from train subjects
    rng = np.random.RandomState(int(cfg["splits"].get("random_state", 42)) + fold_idx)
    train_subjs = sorted(set(subject_id_from_path(edf) for edf, _ in train_pairs))
    val_subject = rng.choice(train_subjs)

    val_pairs = [p for p in train_pairs if subject_id_from_path(p[0]) == val_subject]
    train_pairs = [p for p in train_pairs if subject_id_from_path(p[0]) != val_subject]

    def load_and_window_pairs(pairs_list):
        Xs, ys = [], []
        for edf, js in pairs_list:
            d, lab, sfreq = edf_to_data_and_labels(edf, js, cfg)
            Xi, yi = make_windows(
                d, lab, sfreq,
                cfg["windowing"]["window_sec"],
                cfg["windowing"]["step_sec"],
                cfg["windowing"]["overlap_threshold"],
            )
            Xs.append(Xi); ys.append(yi)
        return np.concatenate(Xs, 0), np.concatenate(ys, 0)

    Xt, yt = load_and_window_pairs(train_pairs)
    Xv, yv = load_and_window_pairs(val_pairs)
    Xte, yte = load_and_window_pairs(test_pairs)

    print(f"[KFold] fold {fold_idx+1}/{n_folds} | val_subject={val_subject} | "
          f"train_subj={len(set(subject_id_from_path(p[0]) for p in train_pairs))} | "
          f"test_subj={len(set(subject_id_from_path(p[0]) for p in test_pairs))}")

    return (Xt, yt), (Xv, yv), (Xte, yte)


# ----------------------------
# Dataloaders
# ----------------------------
def make_loader(X, y, batch_size, num_workers, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)


# ============================================================
# KEEP your evaluate(), train_one_epoch(), plotting functions here
# (copy from your current script)
# ============================================================


def print_groupkfold_splits(pairs, n_folds=5, random_state=42):
    """
    Print which subjects are used for train / val / test in each fold.
    """

   # subjects = [get_subject_id(edf) for edf, _ in pairs]
    subjects = [subject_id_from_path(edf) for edf, _ in pairs]
    subjects = np.array(subjects)

    unique_subjects = np.unique(subjects)
    print(f"\nTotal subjects: {len(unique_subjects)}")
    print("Subjects:", ", ".join(sorted(unique_subjects)))

    gkf = GroupKFold(n_splits=n_folds)

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(pairs, groups=subjects), start=1
    ):
        train_subjects = sorted(set(subjects[train_idx]))
        test_subjects = sorted(set(subjects[test_idx]))

        # ---- choose ONE validation subject from train subjects ----
        rng = np.random.default_rng(random_state + fold)
        val_subject = rng.choice(train_subjects)

        final_train_subjects = sorted(
            s for s in train_subjects if s != val_subject
        )

        print("\n" + "=" * 60)
        print(f"FOLD {fold}/{n_folds}")
        print("=" * 60)
        print(f"TEST subjects ({len(test_subjects)}): {test_subjects}")
        print(f"VAL  subject (1): [{val_subject}]")
        print(f"TRAIN subjects ({len(final_train_subjects)}): {final_train_subjects}")
# ============================================================
# THRESHOLD (best F1)
# ============================================================

def find_best_threshold(y_true_int, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true_int, y_score)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    return float(thresholds[int(np.argmax(f1))])


# ============================================================
# LOG PLOTS AS IMAGES INTO W&B RUN (Media -> Images)
# ============================================================

def log_pr_roc_cm_images(prefix, y_true_int, y_score, y_pred_int, pr_auc, roc_auc, thr, step):
    if wandb.run is None:
        return

    y_true_int = np.asarray(y_true_int).reshape(-1).astype(np.int32)
    y_score = np.asarray(y_score).reshape(-1).astype(np.float32)
    y_pred_int = np.asarray(y_pred_int).reshape(-1).astype(np.int32)

    # --- PR curve image ---
    try:
        prec, rec, _ = precision_recall_curve(y_true_int, y_score)
        fig = plt.figure()
        plt.plot(rec, prec, linewidth=2, label=f"AP={pr_auc:.4f}")
        base = float(y_true_int.mean())
        plt.hlines(base, 0, 1, linestyles="dashed", colors="gray", label=f"Baseline={base:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("PR Curve (sklearn)")
        plt.grid(alpha=0.3)
        plt.legend(loc="lower left")
        wandb.log({f"{prefix}/pr_curve": wandb.Image(fig)}, step=step)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] PR image log failed: {e}")

    # --- ROC curve image ---
    try:
        fpr, tpr, _ = roc_curve(y_true_int, y_score)
        fig = plt.figure()
        plt.plot(fpr, tpr, linewidth=2, label=f"AUC={roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (sklearn)")
        plt.grid(alpha=0.3)
        plt.legend(loc="upper left")
        wandb.log({f"{prefix}/roc_curve": wandb.Image(fig)}, step=step)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] ROC image log failed: {e}")

    # --- Confusion matrix image ---
    try:
        cm = confusion_matrix(y_true_int, y_pred_int)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix @thr={thr:.2f}")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12)
        plt.tight_layout()
        wandb.log({f"{prefix}/confusion_matrix": wandb.Image(fig)}, step=step)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] CM image log failed: {e}")


# ============================================================
# EVALUATE
# ============================================================

def evaluate(model, loader, criterion, device):
    model.eval()
    ys, ps = [], []
    total_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)  # (B,1)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)

            prob = torch.sigmoid(logits)
            ys.append(y.detach().cpu().numpy())
            ps.append(prob.detach().cpu().numpy())

    y_true = np.concatenate(ys, axis=0).ravel()
    y_score = np.concatenate(ps, axis=0).ravel()

    y_true_i = (y_true > 0.5).astype(np.int32)

    thr = find_best_threshold(y_true_i, y_score)
    y_pred_i = (y_score >= thr).astype(np.int32)

    tp = int(np.sum((y_true_i == 1) & (y_pred_i == 1)))
    fp = int(np.sum((y_true_i == 0) & (y_pred_i == 1)))
    tn = int(np.sum((y_true_i == 0) & (y_pred_i == 0)))
    fn = int(np.sum((y_true_i == 1) & (y_pred_i == 0)))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    roc_auc = float(roc_auc_score(y_true_i, y_score)) if len(np.unique(y_true_i)) > 1 else 0.0
    pr_auc = float(average_precision_score(y_true_i, y_score)) if len(np.unique(y_true_i)) > 1 else 0.0

    return {
        "loss": float(total_loss / len(loader.dataset)),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": float(thr),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "y_true": y_true_i,
        "y_score": y_score,
        "y_pred": y_pred_i,
    }


# ============================================================
# TRAIN
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return float(total_loss / len(loader.dataset))


# ============================================================
# MAIN
# ============================================================

def main():
    #cfg = yaml.safe_load(open("config.yaml", "r"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")

    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--loss_name", type=str, default=None)
    parser.add_argument("--sampling", type=str, default=None)

    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    #cfg = apply_cli_overrides(cfg, args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg.get("enabled", True):
        wandb.init(
            project=wandb_cfg.get("project", "spindle_cnn2d"),
            entity=wandb_cfg.get("entity", None),
            name=wandb_cfg.get("run_name", None),
            config=cfg,
        )
    else:
        wandb.init(mode="disabled")

  #  (Xt, yt), (Xv, yv), (Xte, yte) = build_data_splits_fixed(cfg)
    fold = int(cfg["splits"].get("fold", 1))  # 1..5
    n_folds = int(cfg["splits"].get("n_folds", 5))

    pairs = get_pairs_from_config(cfg)
    subjects = {subject_id_from_path(edf) for edf, _ in pairs}
    if cfg["splits"]["n_folds"] > 1:
        print_groupkfold_splits(
            pairs,
            n_folds=cfg["splits"]["n_folds"],
            random_state=cfg["splits"].get("random_state", 42),
        )    # subject print ------ decode k fold
    # ---- SINGLE SUBJECT or NO CV ----
    if n_folds <= 1 or len(subjects) == 1:
        print("Running SINGLE-SUBJECT / NO-CV mode")
    # single subject -> concatenate raw time, split time, then window
    data_list, lab_list = [], []
    sfreq_ref = None
    for edf, js in pairs:
        d, lab, sf = edf_to_data_and_labels(edf, js, cfg)
        if sfreq_ref is None:
            sfreq_ref = sf
        data_list.append(d)
        lab_list.append(lab)

    data = np.concatenate(data_list, axis=1)
    labels = np.concatenate(lab_list, axis=0)

    train_frac = float(cfg["splits"].get("train_size", 0.70))
    val_frac = float(cfg["splits"].get("val_size", 0.15))
    test_frac = float(cfg["splits"].get("test_size", 0.15))

    (tr_d, tr_y), (va_d, va_y), (te_d, te_y) = split_by_time(data, labels, sfreq_ref, train_frac, val_frac, test_frac)

    Xt, yt = make_windows(tr_d, tr_y, sfreq_ref, cfg["windowing"]["window_sec"], cfg["windowing"]["step_sec"],
                          cfg["windowing"]["overlap_threshold"])
    Xv, yv = make_windows(va_d, va_y, sfreq_ref, cfg["windowing"]["window_sec"], cfg["windowing"]["step_sec"],
                          cfg["windowing"]["overlap_threshold"])
    Xte, yte = make_windows(te_d, te_y, sfreq_ref, cfg["windowing"]["window_sec"], cfg["windowing"]["step_sec"],
                            cfg["windowing"]["overlap_threshold"])
    # ---- SINGLE SUBJECT or NO CV ----
    if n_folds <= 1 or len(subjects) == 1:
        print("Running SINGLE-SUBJECT / NO-CV mode")

        data_list, lab_list = [], []
        sfreq_ref = None
        for edf, js in pairs:
            d, lab, sf = edf_to_data_and_labels(edf, js, cfg)
            if sfreq_ref is None:
                sfreq_ref = sf
            data_list.append(d)
            lab_list.append(lab)

        data = np.concatenate(data_list, axis=1)
        labels = np.concatenate(lab_list, axis=0)

        train_frac = float(cfg["splits"].get("train_size", 0.70))
        val_frac = float(cfg["splits"].get("val_size", 0.15))
        test_frac = float(cfg["splits"].get("test_size", 0.15))

        (tr_d, tr_y), (va_d, va_y), (te_d, te_y) = split_by_time(
            data, labels, sfreq_ref, train_frac, val_frac, test_frac
        )

        Xt, yt = make_windows(tr_d, tr_y, sfreq_ref,
                              cfg["windowing"]["window_sec"],
                              cfg["windowing"]["step_sec"],
                              cfg["windowing"]["overlap_threshold"])
        Xv, yv = make_windows(va_d, va_y, sfreq_ref,
                              cfg["windowing"]["window_sec"],
                              cfg["windowing"]["step_sec"],
                              cfg["windowing"]["overlap_threshold"])
        Xte, yte = make_windows(te_d, te_y, sfreq_ref,
                                cfg["windowing"]["window_sec"],
                                cfg["windowing"]["step_sec"],
                                cfg["windowing"]["overlap_threshold"])
    # ---- MULTI-SUBJECT CV ----
    else:
        print(f"Running {n_folds}-FOLD SUBJECT-WISE CV (fold {fold})")
        (Xt, yt), (Xv, yv), (Xte, yte) = build_splits_subject_kfold(
            pairs, cfg, fold_idx=fold - 1, n_folds=n_folds
        )

    def make_loader(X, y, shuffle):
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return DataLoader(
            ds,
            batch_size=int(cfg["training"]["batch_size"]),
            shuffle=shuffle,
            num_workers=int(cfg["training"]["num_workers"]),
            pin_memory=True,
        )

    train_loader = make_loader(Xt, yt, True)
    val_loader = make_loader(Xv, yv, False)
    test_loader = make_loader(Xte, yte, False)

    n_channels = len(cfg["channels"]["names"])
    model = SpindleCNN(n_channels=n_channels).to(device)

    criterion = build_loss_function(cfg["loss"]["name"], cfg["loss"], train_loader)

    lr = float(cfg["training"]["learning_rate"])
    wd = float(cfg["training"].get("weight_decay", 0.0))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    num_epochs = int(cfg["training"]["num_epochs"])
    log_plots_every = int(wandb_cfg.get("log_val_plots_every", 1))

    print("\n============================================================")
    print("CNN TRAINING (WINDOW-LEVEL)")
    print(f"  Device: {device}")
    print(f"  Epochs: {num_epochs}")
    print(f"  LR:     {lr}")
    print(f"  Loss:   {cfg['loss']['name']}")
    print("============================================================\n")

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val = evaluate(model, val_loader, criterion, device)

        # ---- log PR/ROC/CM IMAGES into THIS run ----
        if log_plots_every <= 1 or (epoch % log_plots_every == 0):
            log_pr_roc_cm_images(
                "val",
                val["y_true"], val["y_score"], val["y_pred"],
                pr_auc=val["pr_auc"], roc_auc=val["roc_auc"], thr=val["threshold"],
                step=epoch
            )

        # scalars
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val["loss"],
            "val/f1": val["f1"],
            "val/roc_auc": val["roc_auc"],
            "val/pr_auc": val["pr_auc"],
            "val/threshold": val["threshold"],
            "lr": optimizer.param_groups[0]["lr"],
        }, step=epoch)

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_f1={val['f1']:.4f} | val_pr_auc={val['pr_auc']:.4f}"
        )

    # FINAL TEST
    test = evaluate(model, test_loader, criterion, device)
    final_step = num_epochs + 1

    log_pr_roc_cm_images(
        "test",
        test["y_true"], test["y_score"], test["y_pred"],
        pr_auc=test["pr_auc"], roc_auc=test["roc_auc"], thr=test["threshold"],
        step=final_step
    )

    wandb.log({
        "test/loss": test["loss"],
        "test/f1": test["f1"],
        "test/roc_auc": test["roc_auc"],
        "test/pr_auc": test["pr_auc"],
        "test/threshold": test["threshold"],
    }, step=final_step)

    print("\n================ FINAL TEST ================")
    print(
        f"F1={test['f1']:.4f} | ROC-AUC={test['roc_auc']:.4f} | "
        f"PR-AUC={test['pr_auc']:.4f} | thr={test['threshold']:.3f}"
    )
    print("===========================================\n")

    wandb.finish()

    out_dir = Path("CNN_kGroup_results")
    out_dir.mkdir(exist_ok=True)

    result = {
        "fold": fold,
        "f1": test["f1"],
        "roc_auc": test["roc_auc"],
        "pr_auc": test["pr_auc"],
    }

    with open(out_dir / f"fold_{fold}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(
        "RUN CONFIG → "
        f"LR={cfg['training']['learning_rate']} | "
        f"BS={cfg['training']['batch_size']} | "
        f"LOSS={cfg['loss']['name']}"
    )    #sweep check

if __name__ == "__main__":
    main()
