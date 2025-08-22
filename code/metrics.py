# this is the custom metrics tp fp fn tn precission amd recal
from typing import Tuple, Dict, Optional, Any
import io, os, json, datetime as _dt
import numpy as np
import torch
from torch import nn

# optional libs
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    _HAS_PLT = True
except Exception:
    plt = None; mpimg = None; _HAS_PLT = False

try:
    from sklearn.metrics import roc_curve, precision_recall_curve, auc
    _HAS_SK = True
except Exception:
    _HAS_SK = False

try:
    import wandb  # type: ignore
    _HAS_WANDB = True
except Exception:
    wandb = None; _HAS_WANDB = False


# --------------------------
#     helper utils
# --------------------------
def _expects_2d(model: nn.Module) -> bool:
    if hasattr(model, "_expected_input"):
        return getattr(model, "_expected_input") == "B1CT"
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            return True
    return False

def _standardize_inputs(x: torch.Tensor, expects_2d: bool) -> torch.Tensor:
    if expects_2d and x.ndim == 3:     # [B,C,T] -> [B,1,C,T]
        return x.unsqueeze(1)
    if (not expects_2d) and x.ndim == 4 and x.shape[1] == 1:  # [B,1,C,T] -> [B,C,T]
        return x.squeeze(1)
    return x

def _squeeze_bt(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 3 and t.shape[1] == 1: t = t.squeeze(1)
    if t.ndim == 3 and t.shape[2] == 1: t = t.squeeze(2)
    return t

def _cfg_get(cfg: Optional[Any], dotted: str, default=None):
    if cfg is None: return default
    cur = cfg
    for k in dotted.split("."):
        try:
            cur = cur[k] if isinstance(cur, dict) else getattr(cur, k)
        except Exception:
            return default
        if cur is None:
            return default
    return cur


# --------------------------
#     core metrics
# --------------------------
def _confusion(y_true: np.ndarray, probs: np.ndarray, thr: float) -> Dict[str, float]:
    y = y_true.reshape(-1)
    p = probs.reshape(-1)
    pred = (p >= float(thr)).astype(np.int32)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": prec, "recall": rec, "f1": f1}


def _rocpr(labels: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    """
    Returns {'roc_auc': ..., 'pr_auc': ...} where present.
    If labels contain a single class only, returns NaN for both and skips sklearn calls.
    """
    if not _HAS_SK:
        return {}
    y = labels.reshape(-1)
    p = probs.reshape(-1)
    classes = np.unique(y)
    """if classes.size < 2:
        # single-class edge case (avoid sklearn warnings)
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    fpr, tpr, _ = roc_curve(y, p)
    pr, rc, _ = precision_recall_curve(y, p)
    return {"roc_auc": float(auc(fpr, tpr)), "pr_auc": float(auc(rc, pr))             roc debug"""

    # sklearn raises if y has only one class
    if np.unique(y).size < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    fpr, tpr, _ = roc_curve(y, p)
    pr, rc, _ = precision_recall_curve(y, p)
    return {"roc_auc": float(auc(fpr, tpr)), "pr_auc": float(auc(rc, pr))}


def _fig_to_wandb_image(buf: io.BytesIO):
    if not (_HAS_PLT and _HAS_WANDB and mpimg is not None):
        return None
    try:
        buf.seek(0)
        arr = mpimg.imread(buf, format="png")
        return wandb.Image(arr)
    except Exception:
        return None


def _plots(labels: np.ndarray, probs: np.ndarray, thr: float):
    if not _HAS_PLT:
        raise RuntimeError("matplotlib required to plot")

    y = labels.reshape(-1)
    p = probs.reshape(-1)

    # Confusion
    pred = (p >= thr).astype(int)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    cm = np.array([[tn, fp], [fn, tp]])

    fig_cm = plt.figure(figsize=(6, 5)); ax = plt.gca()
    im = ax.imshow(cm, cmap="Blues"); ax.set_title(f"confusion @ {thr:.2f}")
    ax.set_xlabel("pred"); ax.set_ylabel("true")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["neg", "pos"]); ax.set_yticklabels(["neg", "pos"])
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]:,}", ha="center", va="center",
                color=("white" if cm[i, j] > cm.max() / 2 else "black"),
            )
    plt.colorbar(im, fraction=0.046, pad=0.04)
    buf_cm = io.BytesIO(); fig_cm.savefig(buf_cm, format="png", bbox_inches="tight"); plt.close(fig_cm)

    # ROC / PR (skip if single-class)
    buf_roc = buf_pr = None
    roc_auc = pr_auc = None
    if _HAS_SK and (y.min() == 0) and (y.max() == 1):
        fpr, tpr, _ = roc_curve(y, p); roc_auc = float(auc(fpr, tpr))
        fig1 = plt.figure(); plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "--"); plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.title("ROC"); plt.legend()
        buf_roc = io.BytesIO(); fig1.savefig(buf_roc, format="png", bbox_inches="tight"); plt.close(fig1)

        prec, rec, _ = precision_recall_curve(y, p); pr_auc = float(auc(rec, prec))
        fig2 = plt.figure(); plt.plot(rec, prec, label=f"AUC={pr_auc:.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR"); plt.legend()
        buf_pr = io.BytesIO(); fig2.savefig(buf_pr, format="png", bbox_inches="tight"); plt.close(fig2)

    return {"cm": buf_cm, "roc": buf_roc, "pr": buf_pr, "roc_auc": roc_auc, "pr_auc": pr_auc}


# --------------------------
#     high-level API
# --------------------------
class SampleMetrics:
    def __init__(self, sfreq: float = 200.0, window_sec: float = 2.0, step_sec: float = 1.0,
                 use_wandb: bool = False, out_dir: Optional[str] = None):
        self.sfreq = float(sfreq)
        self.window_sec = float(window_sec)
        self.step_sec = float(step_sec)
        self.use_wandb = bool(use_wandb)
        self.out_dir = out_dir or "./_metrics"
        os.makedirs(self.out_dir, exist_ok=True)
        self._wandb = wandb if (_HAS_WANDB and self.use_wandb) else None

    def stitch(self, model: nn.Module, loader, device="cpu") -> Tuple[np.ndarray, np.ndarray]:
        model.eval()
        probs, labels = [], []
        exp2d = _expects_2d(model)
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                xb = _standardize_inputs(xb, exp2d)
                logits = model(xb); logits = _squeeze_bt(logits)
                p = torch.sigmoid(logits)
                probs.append(p.detach().cpu().numpy())
                labels.append(yb.detach().cpu().numpy())
        return np.concatenate(probs, 0), np.concatenate(labels, 0)

    def best_threshold_from_arrays(self, labels: np.ndarray, probs: np.ndarray, num: int = 101):
        y = labels.reshape(-1)
        p = probs.reshape(-1)
        best_thr = 0.5; best = None
        for thr in np.linspace(0, 1, num=num):
            c = _confusion(y, p, float(thr))
            if best is None or c["f1"] > best["f1"]:
                best = c; best_thr = float(thr)
        return best_thr, best

    def evaluate(self, model: nn.Module, loader, device="cpu", threshold=0.5,
                 sweep_threshold=False, log_curves=False):
        probs, labels = self.stitch(model, loader, device=device)
        out = {"threshold": float(threshold), "confusion": _confusion(labels, probs, threshold)}
        out.update({k: v for k, v in _rocpr(labels, probs).items() if k in ("roc_auc", "pr_auc")})

        if log_curves and self._wandb is not None and _HAS_PLT:
            ps = _plots(labels, probs, float(threshold))
            logs = {}
            img = _fig_to_wandb_image(ps["cm"])
            if img is not None:
                logs["eval/confusion_matrix"] = img
            if ps["roc"] is not None:
                img = _fig_to_wandb_image(ps["roc"])
                if img is not None:
                    logs["eval/roc_curve"] = img; logs["eval/roc_auc"] = ps["roc_auc"]
            if ps["pr"] is not None:
                img = _fig_to_wandb_image(ps["pr"])
                if img is not None:
                    logs["eval/pr_curve"] = img; logs["eval/pr_auc"] = ps["pr_auc"]
            if logs:
                self._wandb.log(logs)

        if sweep_threshold:
            bt, bc = self.best_threshold_from_arrays(labels, probs, num=101)
            out["best_threshold"] = float(bt); out["best_confusion"] = bc

        return out

    def export_artifacts(self, model: nn.Module, loader, cfg: Optional[Any] = None, device: str = "cpu",
                         prefix: str = "test", out_dir: Optional[str] = None, save_dir: Optional[str] = None,
                         threshold: Optional[float] = None, sweep_threshold: Optional[bool] = None,
                         save_arrays: bool = True, save_json: bool = True, save_plots: bool = True) -> Dict[str, str]:
        if out_dir is None and save_dir is not None:
            out_dir = save_dir
        if threshold is None:
            threshold = float(_cfg_get(cfg, "eval.threshold", 0.5))
        if sweep_threshold is None:
            sweep_threshold = bool(_cfg_get(cfg, "eval.sweep_threshold", False))
        use_wb = bool(_cfg_get(cfg, "logging.use_wandb", self.use_wandb))

        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = prefix
        out_dir = out_dir or os.path.join("./_artifacts", f"{base}_{stamp}")
        os.makedirs(out_dir, exist_ok=True)

        probs, labels = self.stitch(model, loader, device=device)
        out = {"threshold": float(threshold), "confusion": _confusion(labels, probs, threshold)}
        out.update({k: v for k, v in _rocpr(labels, probs).items() if k in ("roc_auc", "pr_auc")})
        if sweep_threshold:
            bt, bc = self.best_threshold_from_arrays(labels, probs, num=101)
            out["best_threshold"] = float(bt); out["best_confusion"] = bc

        paths: Dict[str, str] = {}

        if save_arrays:
            p_path = os.path.join(out_dir, f"{prefix}_probs.npy")
            y_path = os.path.join(out_dir, f"{prefix}_labels.npy")
            np.save(p_path, probs); np.save(y_path, labels)
            paths["probs"] = p_path; paths["labels"] = y_path

        if save_json:
            j_path = os.path.join(out_dir, f"{prefix}_metrics.json")
            with open(j_path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
            paths["json"] = j_path

        if save_plots and _HAS_PLT:
            ps = _plots(labels, probs, float(threshold))
            cm_path = os.path.join(out_dir, f"{prefix}_confusion.png"); ps["cm"].seek(0); open(cm_path, "wb").write(ps["cm"].getvalue())
            paths["confusion_png"] = cm_path
            if ps["roc"] is not None:
                roc_path = os.path.join(out_dir, f"{prefix}_roc.png"); ps["roc"].seek(0); open(roc_path, "wb").write(ps["roc"].getvalue())
                paths["roc_png"] = roc_path
            if ps["pr"] is not None:
                pr_path = os.path.join(out_dir, f"{prefix}_pr.png"); ps["pr"].seek(0); open(pr_path, "wb").write(ps["pr"].getvalue())
                paths["pr_png"] = pr_path

            if use_wb and self._wandb is not None:
                logs = {}
                img = _fig_to_wandb_image(ps["cm"])
                if img is not None: logs[f"{prefix}/confusion_matrix"] = img
                if ps["roc"] is not None:
                    img = _fig_to_wandb_image(ps["roc"])
                    if img is not None: logs[f"{prefix}/roc_curve"] = img; logs[f"{prefix}/roc_auc"] = ps["roc_auc"]
                if ps["pr"] is not None:
                    img = _fig_to_wandb_image(ps["pr"])
                    if img is not None: logs[f"{prefix}/pr_curve"] = img; logs[f"{prefix}/pr_auc"] = ps["pr_auc"]
                if logs: self._wandb.log(logs)

        return paths


# module-level convenience
def evaluate(model: nn.Module, loader, device="cpu", sfreq=200.0, window_sec=2.0, step_sec=1.0,
             threshold=0.5, sweep_threshold=False, log_curves=False, use_wandb=False, out_dir=None, **_):
    sm = SampleMetrics(sfreq, window_sec, step_sec, use_wandb, out_dir)
    return sm.evaluate(model, loader, device, threshold, sweep_threshold, log_curves)

def export_artifacts(model: nn.Module, loader, cfg=None, device="cpu", prefix="test",
                     out_dir=None, save_dir=None, threshold=None, sweep_threshold=None,
                     save_arrays=True, save_json=True, save_plots=True):
    sm = SampleMetrics(
        sfreq=float(_cfg_get(cfg, "data.sfreq", 200.0)),
        window_sec=float(_cfg_get(cfg, "data.window_sec", 2.0)),
        step_sec=float(_cfg_get(cfg, "data.step_sec", 1.0)),
        use_wandb=bool(_cfg_get(cfg, "logging.use_wandb", False)),
        out_dir=out_dir or save_dir
    )
    return sm.export_artifacts(model, loader, cfg, device, prefix, out_dir, save_dir,
                               threshold, sweep_threshold, save_arrays, save_json, save_plots)
