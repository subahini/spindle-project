"""
train_timepoint.py

Training script for time-point spindle detection using multi-resolution model.

Key differences from window-level training:
1. Loads both DE features and raw EEG
2. Labels are (n_win, samples_per_window) instead of (n_win, 1)
3. Model outputs per-sample predictions
4. Evaluation at sample-level (not window-level)
"""

import os
import gc
import argparse
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

import wandb

from GraphSleepNet_TimePoint import (
    build_GraphSleepNet_TimePoint,
    build_GraphSleepNet_WindowRefiner
)
from Utils import ReadConfig, scaled_Laplacian, cheb_polynomial
from DataGenerator import kFoldGenerator


# ============================================================================
# Custom data generator for dual inputs
# ============================================================================

class DualInputFoldGenerator:
    """
    K-fold generator that handles both DE and raw EEG inputs.
    """
    def __init__(self, k, x_de, x_raw, y):
        assert len(x_de) == k and len(x_raw) == k and len(y) == k
        self.k = k
        self.x_de_list = x_de
        self.x_raw_list = x_raw
        self.y_list = y
    
    def getFold(self, i):
        """Return train and test sets for fold i."""
        train_de, train_raw, train_y = [], [], []
        
        for p in range(self.k):
            if p != i:
                train_de.append(self.x_de_list[p])
                train_raw.append(self.x_raw_list[p])
                train_y.append(self.y_list[p])
            else:
                test_de = self.x_de_list[p]
                test_raw = self.x_raw_list[p]
                test_y = self.y_list[p]
        
        train_de = np.concatenate(train_de, axis=0)
        train_raw = np.concatenate(train_raw, axis=0)
        train_y = np.concatenate(train_y, axis=0)
        
        return (train_de, train_raw), train_y, (test_de, test_raw), test_y


def AddContext_DE(x_list, context):
    """Add context to DE features (same as before)."""
    assert context % 2 == 1
    cut = context // 2
    ret = []
    
    for x in x_list:
        n = x.shape[0]
        out = np.zeros((n - 2*cut, context, x.shape[1], x.shape[2]), dtype=x.dtype)
        for i in range(cut, n - cut):
            out[i - cut] = x[i - cut:i + cut + 1]
        ret.append(out)
    
    return ret


def AddContext_Raw_and_Label(x_raw_list, y_list, context):
    """
    For raw EEG and labels, extract only the CENTER window.
    
    Input:
        x_raw_list: list of (n_win, samples_per_win, n_ch)
        y_list: list of (n_win, samples_per_win)
        context: int (e.g., 5)
    
    Output:
        x_raw_out: list of (n_win - context + 1, samples_per_win, n_ch)
        y_out: list of (n_win - context + 1, samples_per_win)
        
    We only need the center window's raw data and labels.
    """
    assert context % 2 == 1
    cut = context // 2
    
    x_out = []
    y_out = []
    
    for x_raw, y in zip(x_raw_list, y_list):
        n = x_raw.shape[0]
        # Extract center windows
        x_center = x_raw[cut:n-cut]  # (n - 2*cut, samples, ch)
        y_center = y[cut:n-cut]      # (n - 2*cut, samples)
        
        x_out.append(x_center)
        y_out.append(y_center)
    
    return x_out, y_out

def make_coarse_labels(y_fine, context):
    y_win = (y_fine.sum(axis=1) > 0).astype(np.float32)  # (B,)
    y_coarse = np.repeat(y_win[:, None, None], context, axis=1)  # (B, context, 1)
    return y_coarse

# ============================================================================
# Metrics and evaluation
# ============================================================================

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
    f1 = _safe_div(2 * prec * rec, prec + rec)
    return prec, rec, f1


def pick_best_threshold(y_true, y_prob, n_grid=201):
    y_true = y_true.reshape(-1)
    y_prob = y_prob.reshape(-1)
    best_thr, best_f1 = 0.5, -1.0
    
    for thr in np.linspace(0.0, 1.0, n_grid):
        y_bin = (y_prob >= thr).astype(int)
        tp, fp, fn, tn = compute_confusion(y_true, y_bin)
        _, _, f1 = precision_recall_f1(tp, fp, fn)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    
    return best_thr, best_f1


def plot_roc_pr(y_true, y_prob):
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    y_true = y_true.reshape(-1)
    y_prob = y_prob.reshape(-1)
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = float(auc(fpr, tpr))
    
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(auc(rec, prec))
    
    roc_fig = plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC (AUC={roc_auc:.3f})")
    plt.tight_layout()
    
    pr_fig = plt.figure(figsize=(5, 4))
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR (AUC={pr_auc:.3f})")
    plt.tight_layout()
    
    return roc_fig, pr_fig, roc_auc, pr_auc


# ============================================================================
# Callbacks
# ============================================================================

class TimepointEpochLogger(keras.callbacks.Callback):
    """Log per-epoch metrics for time-point predictions."""
    
    def __init__(self, fold_idx, x_val, y_val, batch_size=128, use_raw=True):
        super().__init__()
        self.fold_idx = fold_idx
        self.x_val = x_val  # Tuple (de, raw) or just de
        self.y_val = y_val.reshape(-1)
        self.batch_size = batch_size
        self.use_raw = use_raw
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Predict
        if self.use_raw:
            preds = self.model.predict(self.x_val, batch_size=self.batch_size, verbose=0)
            y_prob = preds['fine'].reshape(-1)  # Use fine predictions
        else:
            preds = self.model.predict(self.x_val, batch_size=self.batch_size, verbose=0)
            y_prob = preds['fine'].reshape(-1)
        
        # Metrics at threshold 0.5
        y_bin = (y_prob >= 0.5).astype(int)
        tp, fp, fn, tn = compute_confusion(self.y_val, y_bin)
        prec, rec, f1 = precision_recall_f1(tp, fp, fn)
        
        roc_fig, pr_fig, roc_auc, pr_auc = plot_roc_pr(self.y_val, y_prob)
        
        to_log = {
            f"fold{self.fold_idx}/epoch": epoch + 1,
            f"fold{self.fold_idx}/val_tp": tp,
            f"fold{self.fold_idx}/val_fp": fp,
            f"fold{self.fold_idx}/val_fn": fn,
            f"fold{self.fold_idx}/val_tn": tn,
            f"fold{self.fold_idx}/val_precision": prec,
            f"fold{self.fold_idx}/val_recall": rec,
            f"fold{self.fold_idx}/val_f1": f1,
            f"fold{self.fold_idx}/val_roc_auc": roc_auc,
            f"fold{self.fold_idx}/val_pr_auc": pr_auc,
            f"fold{self.fold_idx}/val_roc": wandb.Image(roc_fig),
            f"fold{self.fold_idx}/val_pr": wandb.Image(pr_fig),
        }
        
        # Add training metrics from logs
        for k, v in logs.items():
            if v is not None:
                to_log[f"fold{self.fold_idx}/{k}"] = float(v)
        
        plt.close(roc_fig)
        plt.close(pr_fig)
        
        wandb.log(to_log)

class BestF1CheckpointTimepoint(keras.callbacks.Callback):
    """
    Save model weights when VAL best-F1 improves (threshold sweep on VAL probs).
    Also stores/logs best threshold.
    """
    def __init__(self, fold_idx, x_val_dict, y_val, best_path, meta_path,
                 batch_size=128, n_grid=201, pred_key="fine"):
        super().__init__()
        self.fold_idx = fold_idx
        self.x_val_dict = x_val_dict                 # dict inputs for model.predict
        self.y_val = y_val.reshape(-1).astype(int)   # (N*samples,)
        self.best_path = best_path
        self.meta_path = meta_path
        self.batch_size = batch_size
        self.n_grid = n_grid
        self.pred_key = pred_key

        self.best_f1 = -1.0
        self.best_thr = 0.5
        self.best_epoch = -1

    def on_epoch_end(self, epoch, logs=None):
        # 1) predict probs on val (fine output)
        preds = self.model.predict(self.x_val_dict, batch_size=self.batch_size, verbose=0)
        y_prob = preds[self.pred_key].reshape(-1)

        # 2) pick best threshold on val
        thr, f1 = pick_best_threshold(self.y_val, y_prob, n_grid=self.n_grid)

        # 3) if improved → save weights + meta
        if f1 > self.best_f1:
            self.best_f1 = float(f1)
            self.best_thr = float(thr)
            self.best_epoch = int(epoch + 1)

            self.model.save_weights(self.best_path)
            with open(self.meta_path, "w") as f:
                f.write(f"best_val_f1={self.best_f1}\n")
                f.write(f"best_val_thr={self.best_thr}\n")
                f.write(f"best_epoch={self.best_epoch}\n")

        # 4) log per epoch
        wandb.log({
            f"fold{self.fold_idx}/epoch": epoch + 1,
            f"fold{self.fold_idx}/val_best_f1_sweep": float(f1),
            f"fold{self.fold_idx}/val_best_thr_sweep": float(thr),
            f"fold{self.fold_idx}/best_f1_so_far": float(self.best_f1),
            f"fold{self.fold_idx}/best_thr_so_far": float(self.best_thr),
        })

def split_train_val(x_de, x_raw, y, val_frac=0.15, seed=42):
    """
    Split with stratification based on window-level spindle presence.
    """
    from sklearn.model_selection import train_test_split
    
    # Use window-level labels for stratification
    y_window = (y.sum(axis=1) > 0).astype(int)
    
    x_de_tr, x_de_va, x_raw_tr, x_raw_va, y_tr, y_va = train_test_split(
        x_de, x_raw, y,
        test_size=val_frac,
        random_state=seed,
        stratify=y_window
    )
    
    return (x_de_tr, x_raw_tr), y_tr, (x_de_va, x_raw_va), y_va


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to config.yaml")
    parser.add_argument("-g", "--gpu", default="0", help="GPU ID")
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Disable XLA
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
    tf.config.optimizer.set_jit(False)
    
    # Load config
    PathCfg, cfgTrain, cfgModel = ReadConfig(args.config)
    
    # Params
    channels = int(cfgTrain["channels"])
    context = int(cfgTrain["context"])
    num_epochs = int(cfgTrain["epoch"])
    batch_size = int(cfgTrain["batch_size"])
    learn_rate = float(cfgTrain["learn_rate"])
    optimizer_name = cfgTrain["optimizer"]
    
    cheb_k = int(cfgModel["cheb_k"])
    num_of_chev_filters = int(cfgModel["cheb_filters"])
    num_of_time_filters = int(cfgModel["time_filters"])
    time_conv_strides = int(cfgModel["time_conv_strides"])
    time_conv_kernel = int(cfgModel["time_conv_kernel"])
    num_block = int(cfgModel["num_block"])
    dense_size = cfgModel["Globaldense"].split(",")
    dropout = float(cfgModel["dropout"])
    GLalpha = float(cfgModel["GLalpha"])
    conf_adj = cfgModel["adj_matrix"]
    
    l1_reg = float(cfgModel.get("l1", 0))
    l2_reg = float(cfgModel.get("l2", 0))
    
    if l1_reg > 0 and l2_reg > 0:
        regularizer = keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)
    elif l1_reg > 0:
        regularizer = keras.regularizers.L1(l1_reg)
    elif l2_reg > 0:
        regularizer = keras.regularizers.L2(l2_reg)
    else:
        regularizer = None
    
    # W&B
    project_name = cfgTrain.get("project", "spindle-timepoint")

    if wandb.run is None:
        wandb.init(project=project_name, config={**cfgTrain, **cfgModel}, settings=wandb.Settings(init_timeout=300))
    else:
        wandb.run.config.update({**cfgTrain, **cfgModel}, allow_val_change=True)



    os.makedirs(PathCfg["save"], exist_ok=True)
    
    # Load multi-resolution data
    print(f"Loading data from {PathCfg['data']}")
    npz = np.load(PathCfg["data"], allow_pickle=True)
    
    Fold_Num = npz["Fold_Num"]
    Fold_Data_DE = [np.asarray(x, dtype=np.float32) for x in npz["Fold_Data_DE"]]
    Fold_Data_Raw = [np.asarray(x, dtype=np.float32) for x in npz["Fold_Data_Raw"]]
    Fold_Label = [np.asarray(y, dtype=np.int32) for y in npz["Fold_Label"]]

    fold = len(Fold_Data_DE)
    samples_per_window = Fold_Data_Raw[0].shape[1]  # Should be 400
    
    print(f"Loaded {fold} folds")
    print(f"Samples per window: {samples_per_window}")
    
    # Graph construction
    if conf_adj != "GL":
        import scipy.io as scio
        
        if conf_adj == "1":
            adj = np.ones((channels, channels))
        elif conf_adj.startswith("DD_"):
            mat_path = os.path.join("graphs", f"adj_{conf_adj}.mat")
            adj = scio.loadmat(mat_path)["adj"]
        else:
            adj = scio.loadmat(PathCfg["cheb"])["adj"]
        
        L = scaled_Laplacian(adj)
        cheb_polynomials = cheb_polynomial(L, cheb_k)
    else:
        cheb_polynomials = None
    
    # Add context
    print(f"Adding context (window={context})...")
    Fold_Data_DE = AddContext_DE(Fold_Data_DE, context)
    Fold_Data_Raw, Fold_Label = AddContext_Raw_and_Label(Fold_Data_Raw, Fold_Label, context)
    
    print(f"After context: DE shape = {Fold_Data_DE[0].shape}")
    print(f"After context: Raw shape = {Fold_Data_Raw[0].shape}")
    print(f"After context: Label shape = {Fold_Label[0].shape}")
    
    # K-fold or single split
    val_frac = float(cfgTrain.get("val_frac", 0.15))
    
    if fold == 1:
        print("[INFO] Single subject - using 70/15/15 split")
        x_de_all = Fold_Data_DE[0]
        x_raw_all = Fold_Data_Raw[0]
        y_all = Fold_Label[0]
        x_de_all = x_de_all.astype(np.float32)
        x_raw_all = x_raw_all.astype(np.float32)
        y_all = y_all.astype(np.int32)

        # Split into train+val and test
        from sklearn.model_selection import train_test_split
        y_window = (y_all.sum(axis=1) > 0).astype(int)
        
        x_de_tmp, x_de_test, x_raw_tmp, x_raw_test, y_tmp, y_test = train_test_split(
            x_de_all, x_raw_all, y_all,
            test_size=0.15,
            random_state=42,
            stratify=y_window
        )
        
        y_window_tmp = (y_tmp.sum(axis=1) > 0).astype(int)
        x_de_train, x_de_val, x_raw_train, x_raw_val, y_train, y_val = train_test_split(
            x_de_tmp, x_raw_tmp, y_tmp,
            test_size=0.15/0.85,
            random_state=43,
            stratify=y_window_tmp
        )
        
        fold_indices = [0]
        
    else:
        DataGen = DualInputFoldGenerator(fold, Fold_Data_DE, Fold_Data_Raw, Fold_Label)
        fold_indices = list(range(fold))
    
    # Training loop
    per_fold_results = []
    
    for i in fold_indices:
        print(f"\n{'='*60}")
        print(f"FOLD {i}")
        print('='*60)
        
        if fold != 1:
            (train_de, train_raw), train_y, (test_de, test_raw), test_y = DataGen.getFold(i)
            (train_de, train_raw), train_y, (val_de, val_raw), val_y = split_train_val(
                train_de, train_raw, train_y,
                val_frac=val_frac,
                seed=42 + i
            )
        else:
            train_de, train_raw, train_y = x_de_train, x_raw_train, y_train
            val_de, val_raw, val_y = x_de_val, x_raw_val, y_val
            test_de, test_raw, test_y = x_de_test, x_raw_test, y_test
        
        print(f"Train: {train_de.shape[0]} samples")
        print(f"Val:   {val_de.shape[0]} samples")
        print(f"Test:  {test_de.shape[0]} samples")
        train_coarse = make_coarse_labels(train_y, context)
        val_coarse = make_coarse_labels(val_y, context)
        test_coarse = make_coarse_labels(test_y, context)  # optional, only if you want test coarse metrics

        # Build model
        keras.backend.clear_session()
        gc.collect()
        
        if optimizer_name == "adam":
            opt = keras.optimizers.Adam(learning_rate=learn_rate)
        elif optimizer_name == "RMSprop":
            opt = keras.optimizers.RMSprop(learning_rate=learn_rate)
        else:
            opt = keras.optimizers.SGD(learning_rate=learn_rate)
        
        sample_shape = (context, train_de.shape[2], train_de.shape[3])
        
        model = build_GraphSleepNet_TimePoint(
            k=cheb_k,
            num_of_chev_filters=num_of_chev_filters,
            num_of_time_filters=num_of_time_filters,
            time_conv_strides=time_conv_strides,
            cheb_polynomials=cheb_polynomials,
            time_conv_kernel=time_conv_kernel,
            sample_shape=sample_shape,
            num_block=num_block,
            dense_size=dense_size,
            opt=opt,
            useGL=(conf_adj == "GL"),
            GLalpha=GLalpha,
            regularizer=regularizer,
            dropout=dropout,
            samples_per_window=samples_per_window,
            use_raw_eeg=True,
            raw_eeg_channels=channels,
        )
        
        if i == 0:
            model.summary()
        
        # Callbacks
        epoch_logger = TimepointEpochLogger(
            fold_idx=i,
            x_val={'DE_Input': val_de, 'Raw_EEG_Input': val_raw},
            y_val=val_y,
            batch_size=batch_size,
            use_raw=True
        )

        best_path = os.path.join(PathCfg["save"], f"best_fold{i}.weights.h5")


        meta_path = os.path.join(PathCfg["save"], f"best_fold{i}.meta.txt")

        best_f1_ckpt = BestF1CheckpointTimepoint(
            fold_idx=i,
            x_val_dict={'DE_Input': val_de, 'Raw_EEG_Input': val_raw},
            y_val=val_y,
            best_path=best_path,
            meta_path=meta_path,
            batch_size=batch_size,
            n_grid=201,
            pred_key="fine"
        )
        callbacks = [best_f1_ckpt, epoch_logger]
        print("DTYPES:",
              train_de.dtype, train_raw.dtype, train_y.dtype,
              val_de.dtype, val_raw.dtype, val_y.dtype)

        # Train
        """history = model.fit(
            x={'DE_Input': train_de, 'Raw_EEG_Input': train_raw},
            y={'coarse': train_y, 'fine': train_y},
            epochs=num_epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(
                {'DE_Input': val_de, 'Raw_EEG_Input': val_raw},
                {'coarse': val_y, 'fine': val_y}
            ),"""

        model.compile(
            optimizer=opt,
            loss={'coarse': 'binary_crossentropy', 'fine': 'binary_crossentropy'},
            loss_weights={'coarse': 0.0, 'fine': 1.0}
        )

        history = model.fit(
            x={'DE_Input': train_de, 'Raw_EEG_Input': train_raw},
            y={'coarse': train_coarse, 'fine': train_y},
            validation_data=(
                {'DE_Input': val_de, 'Raw_EEG_Input': val_raw},
                {'coarse': val_coarse, 'fine': val_y}
            ),
            epochs=num_epochs,
            batch_size=batch_size,
            shuffle=True,
            callbacks=callbacks,
            verbose=1
        )

        # Load best weights (best VAL F1 saved by callback)
        if os.path.exists(best_path):
            model.load_weights(best_path)
        else:
            print(f"[WARN] BestF1 checkpoint not found for fold {i}, using last epoch weights.")

        # Use the threshold chosen during training (VAL sweep)
        best_thr = float(best_f1_ckpt.best_thr)
        best_f1_val = float(best_f1_ckpt.best_f1)

        wandb.log({
            f"fold{i}/best_thr_val": best_thr,
            f"fold{i}/best_f1_val": best_f1_val,
            f"fold{i}/best_epoch_val": int(best_f1_ckpt.best_epoch),
        })

        # Evaluate on TEST
        print("\nEvaluating on TEST set...")
        preds = model.predict(
            {'DE_Input': test_de, 'Raw_EEG_Input': test_raw},
            batch_size=batch_size,
            verbose=0
        )

        test_prob = preds['fine'].reshape(-1)
        test_true = test_y.reshape(-1)

        # Apply best_thr to TEST
        test_bin = (test_prob >= best_thr).astype(int)
        tp, fp, fn, tn = compute_confusion(test_true, test_bin)
        prec, rec, f1 = precision_recall_f1(tp, fp, fn)

        roc_fig, pr_fig, roc_auc, pr_auc = plot_roc_pr(test_true, test_prob)

        wandb.log({
            f"fold{i}/test_thr": best_thr,
            f"fold{i}/test_tp": tp,
            f"fold{i}/test_fp": fp,
            f"fold{i}/test_fn": fn,
            f"fold{i}/test_tn": tn,
            f"fold{i}/test_precision": prec,
            f"fold{i}/test_recall": rec,
            f"fold{i}/test_f1": f1,
            f"fold{i}/test_roc_auc": roc_auc,
            f"fold{i}/test_pr_auc": pr_auc,
            f"fold{i}/test_roc": wandb.Image(roc_fig),
            f"fold{i}/test_pr": wandb.Image(pr_fig),
        })

        plt.close(roc_fig)
        plt.close(pr_fig)

        # Apply to test
        test_bin = (test_prob >= best_thr).astype(int)
        tp, fp, fn, tn = compute_confusion(test_true, test_bin)
        prec, rec, f1 = precision_recall_f1(tp, fp, fn)
        
        roc_fig, pr_fig, roc_auc, pr_auc = plot_roc_pr(test_true, test_prob)
        
        print(f"\nTEST Results (threshold={best_thr:.3f}):")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1:        {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        print(f"  PR-AUC:    {pr_auc:.4f}")
        
        wandb.log({
            f"fold{i}/test_thr": best_thr,
            f"fold{i}/test_precision": prec,
            f"fold{i}/test_recall": rec,
            f"fold{i}/test_f1": f1,
            f"fold{i}/test_roc_auc": roc_auc,
            f"fold{i}/test_pr_auc": pr_auc,
            f"fold{i}/test_roc": wandb.Image(roc_fig),
            f"fold{i}/test_pr": wandb.Image(pr_fig),
        })
        
        plt.close(roc_fig)
        plt.close(pr_fig)
        
        per_fold_results.append((prec, rec, f1, roc_auc, pr_auc))
    
    # Summary
    results = np.array(per_fold_results)
    mean = results.mean(axis=0)
    std = results.std(axis=0)
    
    summary = {
        "mean_test_precision": mean[0], "std_test_precision": std[0],
        "mean_test_recall": mean[1], "std_test_recall": std[1],
        "mean_test_f1": mean[2], "std_test_f1": std[2],
        "mean_test_roc_auc": mean[3], "std_test_roc_auc": std[3],
        "mean_test_pr_auc": mean[4], "std_test_pr_auc": std[4],
    }
    
    print("\n" + "="*60)
    print("FINAL RESULTS (mean ± std across folds)")
    print("="*60)
    for k, v in summary.items():
        print(f"{k}: {v:.4f}")
    
    wandb.summary.update(summary)
    wandb.finish()


if __name__ == "__main__":
    main()
