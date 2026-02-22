import os
import gc
import argparse
import shutil
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

import wandb

from GraphSleepNet import build_GraphSleepNet
from Utils import ReadConfig, AddContext, scaled_Laplacian, cheb_polynomial ,AddContextLabelSeq
from DataGenerator import kFoldGenerator


# ----------------------------
# Helpers: metrics + plotting
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
    # maximize F1 on given set
    y_true = y_true.reshape(-1)
    y_prob = y_prob.reshape(-1)
    best_thr, best_f1 = 0.5, -1.0
    best_stats = (0, 0, 0, 0)  # tp, fp, fn, tn
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
    cm = np.array([[tn, fp],
                   [fn, tp]], dtype=int)
    fig = plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["0", "1"])
    plt.yticks(tick_marks, ["0", "1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    return fig


def plot_roc_pr(y_true, y_prob):
    """
    Returns (roc_fig, pr_fig, roc_auc, pr_auc).
    """
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    y_true = y_true.reshape(-1)
    y_prob = y_prob.reshape(-1)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = float(auc(fpr, tpr))

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(auc(rec, prec))

    roc_fig = plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])
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


class WandbEpochEvalLogger(keras.callbacks.Callback):
    """
    Logs per-epoch metrics + plots on validation set.
    """
    def __init__(self, fold_idx, x_val, y_val, batch_size=128):
        super().__init__()
        self.fold_idx = fold_idx
        self.x_val = x_val
        self.y_val = y_val.reshape(-1)
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        to_log = {f"fold{self.fold_idx}/" + k: float(v) for k, v in logs.items() if v is not None}
        to_log["fold"] = self.fold_idx
        y_prob = self.model.predict(self.x_val, batch_size=self.batch_size, verbose=0).reshape(-1)
        thr = 0.5
        y_bin = (y_prob >= thr).astype(int)

        tp, fp, fn, tn = compute_confusion(self.y_val, y_bin)
        prec, rec, f1 = precision_recall_f1(tp, fp, fn)

        roc_fig, pr_fig, roc_auc, pr_auc = plot_roc_pr(self.y_val, y_prob)
        cm_fig = plot_confusion(tp, fp, fn, tn, title=f"Fold {self.fold_idx} VAL CM (thr={thr:.2f})")

        to_log.update({
            f"fold{self.fold_idx}/val_thr_fixed": thr,
            f"fold{self.fold_idx}/val_tp": tp,
            f"fold{self.fold_idx}/val_fp": fp,
            f"fold{self.fold_idx}/val_fn": fn,
            f"fold{self.fold_idx}/val_tn": tn,
            f"fold{self.fold_idx}/val_precision@0.5": prec,
            f"fold{self.fold_idx}/val_recall@0.5": rec,
            f"fold{self.fold_idx}/val_f1@0.5": f1,
            f"fold{self.fold_idx}/val_roc_auc": roc_auc,
            f"fold{self.fold_idx}/val_pr_auc": pr_auc,
            f"fold{self.fold_idx}/val_confusion": wandb.Image(cm_fig),
            f"fold{self.fold_idx}/val_roc_curve": wandb.Image(roc_fig),
            f"fold{self.fold_idx}/val_pr_curve": wandb.Image(pr_fig),
        })

        plt.close(cm_fig)
        plt.close(roc_fig)
        plt.close(pr_fig)
        to_log["fold"] = self.fold_idx
        to_log["epoch"] = epoch + 1
        wandb.log(to_log)


def split_train_val(x, y, val_frac=0.1, seed=42):
    """
    Split train into train/val with stratification.
    """



    from sklearn.model_selection import train_test_split
    #y = y.reshape(-1)
    cut = y.shape[1] // 2
    y_center = y[:, cut, 0].astype(int)
    x_tr, x_va, y_tr, y_va = train_test_split(
        x, y, test_size=val_frac, random_state=seed, stratify=y_center



    )
    return x_tr, y_tr, x_va, y_va
# this function i shelper function to kust use 1 subject
def split_train_val_test(x, y, val_frac=0.15, test_frac=0.15, seed=42):
    """
    Split x,y into train/val/test with stratification based on the CENTER label.
    x: (N, context, V, F)
    y: (N, context, 1)
    """
    from sklearn.model_selection import train_test_split

    assert 0.0 < val_frac < 1.0
    assert 0.0 < test_frac < 1.0
    assert (val_frac + test_frac) < 1.0

    cut = y.shape[1] // 2
    y_center = y[:, cut, 0].astype(int)

    # 1) split out TEST
    x_tmp, x_te, y_tmp, y_te, yc_tmp, yc_te = train_test_split(
        x, y, y_center,
        test_size=test_frac,
        random_state=seed,
        stratify=y_center
    )

    # 2) split remaining into TRAIN + VAL
    val_frac2 = val_frac / (1.0 - test_frac)  # re-normalize
    x_tr, x_va, y_tr, y_va = train_test_split(
        x_tmp, y_tmp,
        test_size=val_frac2,
        random_state=seed + 1,
        stratify=yc_tmp
    )

    return x_tr, y_tr, x_va, y_va, x_te, y_te

def disable_xla():
    # Avoid libdevice / XLA JIT issues on clusters
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
    os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir="
    tf.config.optimizer.set_jit(False)

class BestF1Checkpoint(keras.callbacks.Callback):
    """
    Save model weights when VAL best-F1 improves.
    Best-F1 is computed by sweeping thresholds on VAL probs.
    Also saves the best threshold.
    """
    def __init__(self, fold_idx, x_val, y_val, best_path, meta_path, batch_size=128, n_grid=201):
        super().__init__()
        self.fold_idx = fold_idx
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
        # 1) predict probs on val
        y_prob = self.model.predict(self.x_val, batch_size=self.batch_size, verbose=0).reshape(-1)

        # 2) pick best threshold on val (maximize F1)
        thr, f1, _ = pick_best_threshold(self.y_val, y_prob, n_grid=self.n_grid)

        # 3) if improved, save weights + threshold
        if f1 > self.best_f1:
            self.best_f1 = float(f1)
            self.best_thr = float(thr)
            self.best_epoch = int(epoch + 1)

            self.model.save_weights(self.best_path)

            # store threshold + f1 so we can freeze it later
            with open(self.meta_path, "w") as f:
                f.write(f"best_val_f1={self.best_f1}\n")
                f.write(f"best_val_thr={self.best_thr}\n")
                f.write(f"best_epoch={self.best_epoch}\n")

        # (optional) log these per epoch
        wandb.log({
            "epoch": epoch + 1,

            f"fold{self.fold_idx}/val_best_f1": float(f1),
            f"fold{self.fold_idx}/val_best_thr": float(thr),
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, help="configuration file", required=True)
    parser.add_argument("-g", type=str, help="GPU number to use, set '-1' to use CPU", required=True)
    args = parser.parse_args()

    PathCfg, cfgTrain, cfgModel = ReadConfig(args.c)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.g
    if args.g != "-1":
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        print("Use GPU #" + args.g)
    else:
        tf.config.set_visible_devices([], "GPU")
        print("Use CPU only")

    disable_xla()

    channels = int(cfgTrain["channels"])
    context = int(cfgTrain["context"])
    num_epochs = int(cfgTrain["epoch"])
    batch_size = int(cfgTrain["batch_size"])
    optimizer_name = cfgTrain["optimizer"]
    learn_rate = float(cfgTrain["learn_rate"])
    val_frac = float(cfgTrain.get("val_frac", 0.1))

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

    if l1 != 0 and l2 != 0:
        regularizer = keras.regularizers.l1_l2(l1=l1, l2=l2)
    elif l1 != 0:
        regularizer = keras.regularizers.l1(l1)
    elif l2 != 0:
        regularizer = keras.regularizers.l2(l2)
    else:
        regularizer = None

    save_dir = PathCfg.get("save", "./result/")
    PathCfg["save"] = save_dir  # keep rest of code working

    os.makedirs(save_dir, exist_ok=True)

    shutil.copyfile(args.c, os.path.join(PathCfg["save"], "last.config"))

    if wandb.run is None:
        wandb.init(
            project=cfgTrain.get("project", "spindle-graphsleepsnet"),
            name=cfgTrain.get("run_name", None),
            config={"config_file": args.c, "lr": learn_rate, "epochs": num_epochs, "batch": batch_size},
            settings=wandb.Settings(init_timeout=300),
        )
    else:
        # We are inside a sweep; just update config + extend timeout safety
        wandb.run.config.update(
            {"config_file": args.c, "lr": learn_rate, "epochs": num_epochs, "batch": batch_size},
            allow_val_change=True
        )

    # ===== Define epoch as global step =====
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")
    wandb.define_metric("fold")

    ReadList = np.load(PathCfg["data"], allow_pickle=True)
    Fold_Num = ReadList["Fold_Num"]
    Fold_Data = list(ReadList["Fold_Data"])
    Fold_Label = list(ReadList["Fold_Label"])
    fold = len(Fold_Data)

    print("Read data successfully")
    print(f"Number of folds: {fold}")
    print(f"Windows per fold: {Fold_Num}")

    if conf_adj != "GL":
        import scipy.io as scio

        if conf_adj == "1":
            adj = np.ones((channels, channels))

        elif conf_adj == "random":
            adj = np.random.rand(channels, channels)

        # NEW: allow presets like "DD_dense", "DD_knn_k4", ...
        elif conf_adj.startswith("DD_"):
            mat_path = os.path.join("graphs", f"adj_{conf_adj}.mat")
            if not os.path.isfile(mat_path):
                raise FileNotFoundError(f"Graph file not found: {mat_path}")
            adj = scio.loadmat(mat_path)["adj"]

        # keep old behavior too (if you ever use it)
        elif conf_adj in ("topk", "PLV", "DD"):
            adj = scio.loadmat(PathCfg["cheb"])["adj"]

        else:
            raise ValueError(f"Config: check ADJ (got {conf_adj})")

        L = scaled_Laplacian(adj)
        cheb_polynomials = cheb_polynomial(L, cheb_k)
    else:
        cheb_polynomials = None



    Fold_Data = AddContext(Fold_Data, context)
  #  Fold_Label = AddContext(Fold_Label, context, label=True)
    Fold_Label = AddContextLabelSeq(Fold_Label, context)
    Fold_Num_c = Fold_Num + 1 - context

    print("Context added successfully.")
    print("Number of samples:", np.sum(Fold_Num_c))

    val_frac = float(cfgTrain.get("val_frac", 0.15))
    test_frac = float(cfgTrain.get("test_frac", 0.15))

    per_fold = []
    all_fold_pr_aucs = []

    if fold == 1:
        print("[INFO] Only 1 fold found -> using train/val/test split (70/15/15).")

        x_all = Fold_Data[0]
        y_all = Fold_Label[0]

        train_x, train_y, val_x, val_y, test_x, test_y = split_train_val_test(
            x_all, y_all,
            val_frac=val_frac,
            test_frac=test_frac,
            seed=42
        )

        fold_indices = [0]
    else:
        DataGen = kFoldGenerator(fold, Fold_Data, Fold_Label)
        fold_indices = list(range(fold))

    for i in fold_indices:
        print("Fold #", i)

        if fold != 1:
            # IMPORTANT: fold i as TEST
            train_all_x, train_all_y, test_x, test_y = DataGen.getFold(i)

            train_x, train_y, val_x, val_y = split_train_val(
                train_all_x, train_all_y,
                val_frac=val_frac,
                seed=42 + i
            )


        sample_shape = (context, train_x.shape[2], train_x.shape[3])

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

        if i == 0:
            model.summary()

        best_path = os.path.join(PathCfg["save"], f"Best_model_{i}.weights.h5")
        meta_path = os.path.join(PathCfg["save"], f"Best_model_{i}.meta.txt")
        final_path = os.path.join(PathCfg["save"], f"Final_model_{i}.weights.h5")

        """ckpt = keras.callbacks.ModelCheckpoint(
            filepath=best_path,
            monitor="val_auc_pr",
            save_best_only=True,
            save_weights_only=True,
            mode="max",
            verbose=0,
        )"""
        best_f1_ckpt = BestF1Checkpoint(
            fold_idx=i,
            x_val=val_x,
            y_val=val_y,
            best_path=best_path,
            meta_path=meta_path,
            batch_size=batch_size,
            n_grid=201
        )



        epoch_logger = WandbEpochEvalLogger(
            fold_idx=i,
            x_val=val_x,
            y_val=val_y,
            batch_size=batch_size,
        )
        callbacks = [best_f1_ckpt, epoch_logger]

        model.fit(
            x=train_x,
            y=train_y,
            epochs=num_epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(val_x, val_y),
           # callbacks=([ckpt, epoch_logger],
            callbacks = callbacks,
            verbose=1,
        )

        model.save_weights(final_path)

        if os.path.exists(best_path):
            model.load_weights(best_path)
        else:
            print(f"[WARN] Best checkpoint not found for fold {i}, using final weights.")

        # ---- Threshold tuning on VAL
        val_prob = model.predict(val_x, batch_size=batch_size, verbose=0).reshape(-1)
        val_true = val_y.reshape(-1)
        best_thr, best_f1, _ = pick_best_threshold(val_true, val_prob, n_grid=201)

        # ---- Evaluate on TEST using best_thr
        test_prob = model.predict(test_x, batch_size=batch_size, verbose=0).reshape(-1)
        test_true = test_y.reshape(-1)
        test_bin = (test_prob >= best_thr).astype(int)

        tp, fp, fn, tn = compute_confusion(test_true, test_bin)
        prec, rec, f1 = precision_recall_f1(tp, fp, fn)
        roc_fig, pr_fig, roc_auc, pr_auc = plot_roc_pr(test_y, test_prob)
        cm_fig = plot_confusion(tp, fp, fn, tn, title=f"Fold {i} TEST CM (thr={best_thr:.2f})")
        all_fold_pr_aucs.append(pr_auc)

        wandb.log({
            "epoch": num_epochs,

            f"fold{i}/best_thr_val": best_thr,
            f"fold{i}/best_f1_val": best_f1,
            f"fold{i}/test_tp": tp,
            f"fold{i}/test_fp": fp,
            f"fold{i}/test_fn": fn,
            f"fold{i}/test_tn": tn,
            f"fold{i}/test_precision": prec,
            f"fold{i}/test_recall": rec,
            f"fold{i}/test_f1": f1,
            f"fold{i}/test_roc_auc": roc_auc,
            f"fold{i}/test_pr_auc": pr_auc,
            f"fold{i}/test_confusion": wandb.Image(cm_fig),
            f"fold{i}/test_roc_curve": wandb.Image(roc_fig),
            f"fold{i}/test_pr_curve": wandb.Image(pr_fig),
        })

        plt.close(cm_fig)
        plt.close(roc_fig)
        plt.close(pr_fig)

        per_fold.append((prec, rec, f1, roc_auc, pr_auc))
    wandb.log({
        "mean_val_pr_auc": float(np.mean(all_fold_pr_aucs)),
        "std_val_pr_auc": float(np.std(all_fold_pr_aucs)),
    })

    # Summary mean±std
    per_fold = np.array(per_fold, dtype=float)
    mean = np.nanmean(per_fold, axis=0)
    std = np.nanstd(per_fold, axis=0)

    summary = {
        "mean_test_precision": mean[0], "std_test_precision": std[0],
        "mean_test_recall": mean[1], "std_test_recall": std[1],
        "mean_test_f1": mean[2], "std_test_f1": std[2],
        "mean_test_roc_auc": mean[3], "std_test_roc_auc": std[3],
        "mean_test_pr_auc": mean[4], "std_test_pr_auc": std[4],
    }

    print("\n===  Final Result (TEST, best thr from VAL) ===")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}")

    wandb.summary.update(summary)
    wandb.finish()


if __name__ == "__main__":
    main()
