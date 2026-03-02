import os
import gc
import argparse
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
import tensorflow as tf
from tensorflow import keras
import wandb
import json

from GraphSleepNet import build_GraphSleepNet
from Utils import ReadConfig, AddContext, scaled_Laplacian, cheb_polynomial, AddContextLabelSeq


# ----------------------------
# Helper Functions (metrics, plotting)
# ----------------------------
def _safe_div(a, b):
    return float(a) / float(b) if b != 0 else 0.0


def compute_confusion(y_true, y_pred_bin):
    y_true = y_true.astype(int).reshape(-1)
    y_pred_bin = y_pred_bin.astype(int).reshape(-1)
    tp = int(np.sum((y_true == 1) & (y_pred_bin == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred_bin == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred_bin == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred_bin == 0)))
    return tp, fp, fn, tn


def precision_recall_f1(tp, fp, fn):
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * prec * rec, prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def pick_best_threshold(y_true, y_prob, n_grid=201):
    y_true = y_true.reshape(-1)
    y_prob = y_prob.reshape(-1)
    best_thr, best_f1 = 0.5, -1.0
    best_stats = (0, 0, 0, 0)
    for thr in np.linspace(0.0, 1.0, n_grid):
        y_bin = (y_prob >= thr).astype(int)
        tp, fp, fn, tn = compute_confusion(y_true, y_bin)
        _, _, f1 = precision_recall_f1(tp, fp, fn)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
            best_stats = (tp, fp, fn, tn)
    return best_thr, best_f1, best_stats


def plot_confusion(tp, fp, fn, tn, title="Confusion matrix"):
    cm = np.array([[tn, fp], [fn, tp]], dtype=int)
    fig = plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation="nearest", cmap='Blues')
    plt.title(title)
    plt.colorbar()
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    return fig


def plot_roc_pr(y_true, y_prob):
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    y_true = y_true.reshape(-1)
    y_prob = y_prob.reshape(-1)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = float(auc(fpr, tpr))

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(auc(rec, prec))

    roc_fig = plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, 'b-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC (AUC={roc_auc:.3f})")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    pr_fig = plt.figure(figsize=(5, 4))
    plt.plot(rec, prec, 'b-', linewidth=2)
    baseline = y_true.mean()
    plt.axhline(y=baseline, color='r', linestyle='--', label=f'Baseline={baseline:.3f}')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR (AUC={pr_auc:.3f})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    return roc_fig, pr_fig, roc_auc, pr_auc


class BestF1Checkpoint(keras.callbacks.Callback):
    """Save model when best F1 improves on validation set"""

    def __init__(self, fold_name, x_val, y_val, best_path, meta_path, batch_size=128, n_grid=201):
        super().__init__()
        self.fold_name = fold_name
        self.x_val = x_val
        self.y_val = y_val.reshape(-1)
        self.best_path = best_path
        self.meta_path = meta_path
        self.batch_size = batch_size
        self.n_grid = n_grid
        self.best_f1 = -1.0
        self.best_thr = 0.5
        self.best_epoch = -1

    def on_epoch_end(self, epoch, logs=None):
        y_prob = self.model.predict(self.x_val, batch_size=self.batch_size, verbose=0).reshape(-1)
        thr, f1, _ = pick_best_threshold(self.y_val, y_prob, n_grid=self.n_grid)

        if f1 > self.best_f1:
            self.best_f1 = float(f1)
            self.best_thr = float(thr)
            self.best_epoch = int(epoch + 1)
            self.model.save_weights(self.best_path)

            with open(self.meta_path, "w") as f:
                f.write(f"best_val_f1={self.best_f1}\n")
                f.write(f"best_val_thr={self.best_thr}\n")
                f.write(f"best_epoch={self.best_epoch}\n")

        wandb.log({
            "epoch": epoch + 1,
            f"{self.fold_name}/val_best_f1": float(f1),
            f"{self.fold_name}/val_best_thr": float(thr),
        })


class WandbEpochLogger(keras.callbacks.Callback):
    """Log per-epoch metrics and plots"""

    def __init__(self, fold_name, x_val, y_val, batch_size=128):
        super().__init__()
        self.fold_name = fold_name
        self.x_val = x_val
        self.y_val = y_val.reshape(-1)
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        to_log = {f"{self.fold_name}/" + k: float(v) for k, v in logs.items() if v is not None}

        y_prob = self.model.predict(self.x_val, batch_size=self.batch_size, verbose=0).reshape(-1)
        thr = 0.5
        y_bin = (y_prob >= thr).astype(int)

        tp, fp, fn, tn = compute_confusion(self.y_val, y_bin)
        prec, rec, f1 = precision_recall_f1(tp, fp, fn)

        roc_fig, pr_fig, roc_auc, pr_auc = plot_roc_pr(self.y_val, y_prob)
        cm_fig = plot_confusion(tp, fp, fn, tn, title=f"{self.fold_name} VAL CM (thr={thr:.2f})")

        to_log.update({
            f"{self.fold_name}/val_thr_fixed": thr,
            f"{self.fold_name}/val_tp": tp,
            f"{self.fold_name}/val_fp": fp,
            f"{self.fold_name}/val_fn": fn,
            f"{self.fold_name}/val_tn": tn,
            f"{self.fold_name}/val_precision": prec,
            f"{self.fold_name}/val_recall": rec,
            f"{self.fold_name}/val_f1": f1,
            f"{self.fold_name}/val_roc_auc": roc_auc,
            f"{self.fold_name}/val_pr_auc": pr_auc,
            f"{self.fold_name}/val_confusion": wandb.Image(cm_fig),
            f"{self.fold_name}/val_roc_curve": wandb.Image(roc_fig),
            f"{self.fold_name}/val_pr_curve": wandb.Image(pr_fig),
        })

        plt.close(cm_fig)
        plt.close(roc_fig)
        plt.close(pr_fig)

        to_log["epoch"] = epoch + 1
        wandb.log(to_log)


# ----------------------------
# Split Functions
# ----------------------------
def split_time_70_15_15(data, labels):
    """70-15-15 time-based split for single subject"""
    n = len(data)
    n1 = int(0.7 * n)
    n2 = int(0.85 * n)

    train_data = data[:n1]
    train_labels = labels[:n1]
    val_data = data[n1:n2]
    val_labels = labels[n1:n2]
    test_data = data[n2:]
    test_labels = labels[n2:]

    print(f"\n  Split sizes (70-15-15):")
    print(f"    Train: {len(train_data)} windows ({len(train_data) / n * 100:.1f}%)")
    print(f"    Val:   {len(val_data)} windows ({len(val_data) / n * 100:.1f}%)")
    print(f"    Test:  {len(test_data)} windows ({len(test_data) / n * 100:.1f}%)")

    return train_data, train_labels, val_data, val_labels, test_data, test_labels


def split_train_val(x, y, val_frac=0.15, seed=42):
    """Split train into train/validation with stratification"""
    from sklearn.model_selection import train_test_split

    cut = y.shape[1] // 2
    y_center = y[:, cut, 0].astype(int)

    x_tr, x_va, y_tr, y_va = train_test_split(
        x, y, test_size=val_frac, random_state=seed, stratify=y_center
    )
    return x_tr, y_tr, x_va, y_va


# ----------------------------
# Main Function
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, help="configuration file", required=True)
    parser.add_argument("-g", type=str, help="GPU number to use", required=True)
    parser.add_argument("--fold", type=int, default=1,
                        help="Fold number for GroupKFold (1-5)")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of folds for GroupKFold")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Custom run name for W&B")
    args = parser.parse_args()

    # Read config
    PathCfg, cfgTrain, cfgModel = ReadConfig(args.c)

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.g
    if args.g != "-1":
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Use GPU #{args.g}")
    else:
        tf.config.set_visible_devices([], "GPU")
        print("Use CPU only")

    # Hyperparameters
    channels = int(cfgTrain["channels"])
    context = int(cfgTrain["context"])
    num_epochs = int(cfgTrain["epoch"])
    batch_size = int(cfgTrain["batch_size"])
    optimizer_name = cfgTrain["optimizer"]
    learn_rate = float(cfgTrain["learn_rate"])
    val_frac = float(cfgTrain.get("val_frac", 0.15))

    # Model parameters
    dense_size = np.array(str.split(cfgModel["Globaldense"], ","), dtype=int)
    conf_adj = cfgModel["adj_matrix"]
    GLalpha = float(cfgModel["GLalpha"])
    num_of_chev_filters = int(cfgModel["cheb_filters"])
    num_of_time_filters = int(cfgModel["time_filters"])
    time_conv_strides = int(cfgModel["time_conv_strides"])
    time_conv_kernel = int(cfgModel["time_conv_kernel"])
    num_block = int(cfgModel["num_block"])
    cheb_k = int(cfgModel["cheb_k"])
    l1 = float(cfgModel["l1"])
    l2 = float(cfgModel["l2"])
    dropout = float(cfgModel["dropout"])

    # Regularizer
    if l1 != 0 and l2 != 0:
        regularizer = keras.regularizers.l1_l2(l1=l1, l2=l2)
    elif l1 != 0:
        regularizer = keras.regularizers.l1(l1)
    elif l2 != 0:
        regularizer = keras.regularizers.l2(l2)
    else:
        regularizer = None

    # Create save directory
    save_dir = PathCfg.get("save", "./result/")
    os.makedirs(save_dir, exist_ok=True)
    shutil.copyfile(args.c, os.path.join(save_dir, "last.config"))

    # ========== LOAD DATA ==========
    print(f"\n{'=' * 60}")
    print(f"Loading data from: {PathCfg['data']}")
    print(f"{'=' * 60}")

    ReadList = np.load(PathCfg["data"], allow_pickle=True)
    Fold_Data = list(ReadList["Fold_Data"])  # List of subjects
    Fold_Label = list(ReadList["Fold_Label"])  # List of subjects
    Fold_Num = ReadList["Fold_Num"]  # Windows per subject
    # ✅ unwrap object containers into real numeric arrays (the real bug fix)
    Fold_Data = [np.asarray(x, dtype=np.float32) for x in Fold_Data]
    Fold_Label = [np.asarray(y, dtype=np.int32) for y in Fold_Label]
    n_subjects = len(Fold_Data)
    print(f"\nLoaded {n_subjects} subjects")
    print(f"Windows per subject: {Fold_Num}")

    # Add context to each subject's data
    Fold_Data = AddContext(Fold_Data, context)

    Fold_Label = AddContext(Fold_Label, context, label=True, dtype=np.int32)
    # ========== DETERMINE MODE ==========
    if n_subjects == 1:
        # ===== SINGLE SUBJECT MODE (70-15-15) =====
        mode = "single"
        print(f"\n{'=' * 60}")
        print(f"MODE: SINGLE SUBJECT - 70/15/15 TIME-BASED SPLIT")
        print(f"{'=' * 60}")

        # Get the single subject's data
        X_all = Fold_Data[0]
        y_all = Fold_Label[0]

        # Split 70-15-15
        X_train, y_train, X_val, y_val, X_test, y_test = split_time_70_15_15(X_all, y_all)

        fold_name = "single"

    else:
        # ===== MULTI-SUBJECT MODE (5-Fold GroupKFold) =====
        mode = "groupkfold"
        n_folds = args.n_folds
        fold_idx = args.fold - 1  # Convert to 0-based

        print(f"\n{'=' * 60}")
        print(f"MODE: {n_folds}-FOLD GROUPKFOLD (running fold {args.fold}/{n_folds})")
        print(f"{'=' * 60}")

        # Create subject IDs for each window
        subject_ids = []
        for i, (data, label) in enumerate(zip(Fold_Data, Fold_Label)):
            assert len(data) == len(label), f"Subject {i}: data and label length mismatch"
            subject_ids.extend([i] * len(data))

        # Concatenate all data
        X_all = np.concatenate(Fold_Data, axis=0)
        y_all = np.concatenate(Fold_Label, axis=0)

        print(f"\nTotal windows: {len(X_all)}")
        print(f"Total subjects: {n_subjects}")

        # Create GroupKFold splits
        gkf = GroupKFold(n_splits=n_folds)
        splits = list(gkf.split(X_all, y_all, groups=subject_ids))
        train_idx, test_idx = splits[fold_idx]

        # Split into train and test
        X_train_all = X_all[train_idx]
        y_train_all = y_all[train_idx]
        X_test = X_all[test_idx]
        y_test = y_all[test_idx]

        # Get subject IDs for train set to pick validation subject
        train_subject_ids = [subject_ids[i] for i in train_idx]
        unique_train_subjects = sorted(set(train_subject_ids))

        # Randomly pick ONE subject for validation (just like CNN)
        np.random.seed(42 + fold_idx)
        val_subject_idx = np.random.choice(unique_train_subjects)

        # Split train into train and validation by subject
        val_mask = [sid == val_subject_idx for sid in train_subject_ids]
        train_mask = [not v for v in val_mask]

        X_train = X_train_all[train_mask]
        y_train = y_train_all[train_mask]
        X_val = X_train_all[val_mask]
        y_val = y_train_all[val_mask]

        # Print fold info
        train_subjects = set([subject_ids[i] for i in train_idx if subject_ids[i] != val_subject_idx])
        test_subjects = set([subject_ids[i] for i in test_idx])

        print(f"\nFold {args.fold} composition:")
        print(f"  Train subjects: {len(train_subjects)}")
        print(f"  Val subject: {val_subject_idx}")
        print(f"  Test subjects: {len(test_subjects)}")
        print(f"\n  Train windows: {len(X_train)}")
        print(f"  Val windows:   {len(X_val)}")
        print(f"  Test windows:  {len(X_test)}")

        fold_name = f"fold{args.fold}"

    # ========== BUILD GRAPH ==========
    if conf_adj != "GL":
        import scipy.io as sio
        if conf_adj.startswith("DD_"):
            mat_path = os.path.join("graphs", f"adj_{conf_adj}.mat")
            if not os.path.isfile(mat_path):
                raise FileNotFoundError(f"Graph file not found: {mat_path}")
            adj = sio.loadmat(mat_path)["adj"]
        elif conf_adj in ("topk", "PLV", "DD"):
            adj = sio.loadmat(PathCfg["cheb"])["adj"]
        elif conf_adj == "identity":
            adj = np.eye(channels)
        elif conf_adj == "random":
            adj = np.random.rand(channels, channels)
        else:
            raise ValueError(f"Unknown adj_matrix: {conf_adj}")

        L = scaled_Laplacian(adj)
        cheb_polynomials = cheb_polynomial(L, cheb_k)
    else:
        cheb_polynomials = None

    # ========== BUILD MODEL ==========
    sample_shape = (context, X_train.shape[2], X_train.shape[3])

    keras.backend.clear_session()
    gc.collect()

    if optimizer_name == "adam":
        opt = keras.optimizers.Adam(learning_rate=learn_rate)
    elif optimizer_name == "RMSprop":
        opt = keras.optimizers.RMSprop(learning_rate=learn_rate)
    elif optimizer_name == "SGD":
        opt = keras.optimizers.SGD(learning_rate=learn_rate)
    else:
        raise ValueError("Config: check optimizer")

    model = build_GraphSleepNet(
        cheb_k,
        num_of_chev_filters,
        num_of_time_filters,
        time_conv_strides,
        cheb_polynomials,
        time_conv_kernel,
        sample_shape,
        num_block,
        dense_size,
        opt,
        conf_adj == "GL",
        GLalpha,
        regularizer,
        dropout,
    )

    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(curve="ROC", name="auc_roc"),
            keras.metrics.AUC(curve="PR", name="auc_pr"),
        ],
    )

    # ========== WANDB INIT ==========
    run_name = args.run_name if args.run_name else f"graphsleepnet_{mode}_{fold_name}"

    wandb.init(
        project=cfgTrain.get("project", "spindle-graphsleepsnet"),
        name=run_name,
        config={
            "mode": mode,
            "fold": args.fold if mode == "groupkfold" else None,
            "n_folds": args.n_folds if mode == "groupkfold" else None,
            "lr": learn_rate,
            "epochs": num_epochs,
            "batch": batch_size,
            "context": context,
            "adj_matrix": conf_adj,
        }
    )

    if mode == "single" or (mode == "groupkfold" and fold_idx == 0):
        model.summary()

    # ========== SETUP CALLBACKS ==========
    best_path = os.path.join(save_dir, f"Best_model_{fold_name}.weights.h5")
    meta_path = os.path.join(save_dir, f"Best_model_{fold_name}.meta.txt")

    best_f1_ckpt = BestF1Checkpoint(
        fold_name=fold_name,
        x_val=X_val,
        y_val=y_val,
        best_path=best_path,
        meta_path=meta_path,
        batch_size=batch_size,
    )

    epoch_logger = WandbEpochLogger(
        fold_name=fold_name,
        x_val=X_val,
        y_val=y_val,
        batch_size=batch_size,
    )

    # ========== TRAIN ==========
    model.fit(
        x=X_train,
        y=y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(X_val, y_val),
        callbacks=[best_f1_ckpt, epoch_logger],
        verbose=1,
    )

    # ========== EVALUATE ON TEST ==========
    # Load best model
    if os.path.exists(best_path):
        model.load_weights(best_path)
        # Load best threshold
        best_thr = 0.5
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                for line in f:
                    if line.startswith('best_val_thr'):
                        best_thr = float(line.split('=')[1].strip())
    else:
        best_thr = 0.5

    # Predict on test
    test_prob = model.predict(X_test, batch_size=batch_size, verbose=0).reshape(-1)
    test_true = y_test.reshape(-1)
    test_bin = (test_prob >= best_thr).astype(int)

    # Calculate metrics
    tp, fp, fn, tn = compute_confusion(test_true, test_bin)
    prec, rec, f1 = precision_recall_f1(tp, fp, fn)
    roc_fig, pr_fig, roc_auc, pr_auc = plot_roc_pr(test_true, test_prob)
    cm_fig = plot_confusion(tp, fp, fn, tn, title=f"{fold_name} TEST CM")

    print(f"\n{'=' * 60}")
    print(f"TEST RESULTS - {fold_name}")
    print(f"{'=' * 60}")
    print(f"  F1:         {f1:.4f}")
    print(f"  Precision:  {prec:.4f}")
    print(f"  Recall:     {rec:.4f}")
    print(f"  ROC-AUC:    {roc_auc:.4f}")
    print(f"  PR-AUC:     {pr_auc:.4f}")
    print(f"  Threshold:  {best_thr:.4f}")
    print(f"  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

    # Log test results
    wandb.log({
        f"{fold_name}/test_best_thr": best_thr,
        f"{fold_name}/test_tp": tp,
        f"{fold_name}/test_fp": fp,
        f"{fold_name}/test_fn": fn,
        f"{fold_name}/test_tn": tn,
        f"{fold_name}/test_precision": prec,
        f"{fold_name}/test_recall": rec,
        f"{fold_name}/test_f1": f1,
        f"{fold_name}/test_roc_auc": roc_auc,
        f"{fold_name}/test_pr_auc": pr_auc,
        f"{fold_name}/test_confusion": wandb.Image(cm_fig),
        f"{fold_name}/test_roc_curve": wandb.Image(roc_fig),
        f"{fold_name}/test_pr_curve": wandb.Image(pr_fig),
    })

    plt.close(cm_fig)
    plt.close(roc_fig)
    plt.close(pr_fig)

    # ========== SAVE RESULTS ==========
    results = {
        'mode': mode,
        'fold': args.fold if mode == "groupkfold" else None,
        'n_folds': args.n_folds if mode == "groupkfold" else None,
        'test_metrics': {
            'f1': float(f1),
            'precision': float(prec),
            'recall': float(rec),
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'threshold': float(best_thr),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
        }
    }

    out_dir = os.path.join(save_dir, "graph_results")
    os.makedirs(out_dir, exist_ok=True)

    if mode == "single":
        out_file = os.path.join(out_dir, "single_subject_results.json")
    else:
        out_file = os.path.join(out_dir, f"fold_{args.fold}_of_{args.n_folds}_results.json")

    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_file}")
    wandb.finish()


if __name__ == "__main__":
    main()