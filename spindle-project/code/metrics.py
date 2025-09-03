# metrics.py
import numpy as np
from typing import Dict, Tuple, List, Any
from dataclasses import dataclass

# Optional plotting (safe if matplotlib is present)
try:
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except Exception:
    _HAS_PLT = False

# ---------- Basic helpers (prob + thresholded metrics) ----------

def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    y_true = y_true.astype(int).ravel()
    y_pred = y_pred.astype(int).ravel()
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tn, fp, fn, tp

def compute_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = _confusion(y_true, y_pred)
    total = tn + fp + fn + tp
    acc = (tp + tn) / total if total else 0.0
    prec_d = tp + fp
    rec_d = tp + fn
    prec = tp / prec_d if prec_d else 0.0
    rec = tp / rec_d if rec_d else 0.0
    f1_d = prec + rec
    f1 = 2 * prec * rec / f1_d if f1_d else 0.0
    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }

def compute_prob_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    # Simple trapezoidal AUCs without sklearn (keeps deps minimal)
    # ROC
    y_true = y_true.astype(int).ravel()
    y_prob = y_prob.astype(float).ravel()
    order = np.argsort(-y_prob)
    y_true_sorted = y_true[order]
    y_prob_sorted = y_prob[order]
    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum(y_true_sorted == 0)
    P = (y_true == 1).sum()
    N = (y_true == 0).sum()
    tpr = tps / P if P else np.zeros_like(tps, dtype=float)
    fpr = fps / N if N else np.zeros_like(fps, dtype=float)
    auc_roc = np.trapz(tpr, fpr) if P and N else float("nan")
    # PR
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = tps / (tps + fps)
        recall = tps / P if P else np.zeros_like(tps, dtype=float)
    precision[np.isnan(precision)] = 0.0
    # Sort recall ascending for trapezoid
    pr_order = np.argsort(recall)
    auc_pr = np.trapz(precision[pr_order], recall[pr_order]) if P else float("nan")
    return {"auc_roc": float(auc_roc), "auc_pr": float(auc_pr)}

def threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray, thresholds: List[float]) -> List[Dict[str, float]]:
    out = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        m = compute_basic_metrics(y_true, y_pred)
        m["threshold"] = float(t)
        out.append(m)
    return out

# ---------- Segment/event utilities (merge consecutive positives) ----------

def _windows_to_segments(start_times: np.ndarray, window_sec: float, is_pos: np.ndarray) -> List[Tuple[float, float]]:
    """Merge consecutive positive windows into [start,end] segments."""
    segments = []
    on = False
    seg_start = None
    for i, pos in enumerate(is_pos.astype(bool)):
        if pos and not on:
            on = True
            seg_start = start_times[i]
        elif not pos and on:
            on = False
            segments.append((seg_start, start_times[i] + window_sec))
    if on:
        segments.append((seg_start, start_times[len(is_pos) - 1] + window_sec))
    return segments

def segment_metrics(y_true_events: List[Tuple[float, float]],
                    y_pred_events: List[Tuple[float, float]],
                    tolerance: float = 0.5) -> Dict[str, float]:
    """Event-level precision/recall/F1 with simple overlap/tolerance matching."""
    matched_true = set()
    matched_pred = set()
    for i, (ts, te) in enumerate(y_true_events):
        for j, (ps, pe) in enumerate(y_pred_events):
            if j in matched_pred:
                continue
            # Consider a match if segments overlap OR ends are within tolerance
            overlap = not (pe <= ts or ps >= te)
            close = (abs(ts - ps) <= tolerance) or (abs(te - pe) <= tolerance)
            if overlap or close:
                matched_true.add(i)
                matched_pred.add(j)
                break
    tp = len(matched_true)
    fp = len(y_pred_events) - len(matched_pred)
    fn = len(y_true_events) - len(matched_true)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"segment_precision": prec, "segment_recall": rec, "segment_f1": f1,
            "segment_tp": tp, "segment_fp": fp, "segment_fn": fn}

# ---------- Compatibility class expected by trainer.py ----------

@dataclass
class _CfgView:
    WINDOW_SEC: float
    STEP_SEC: float
    def get(self, key: str, default=None):
        # Small helper so we can call cfg.get('evaluation.segment_tolerance', ...)
        return default

class TimeBasedMetrics:
    """
    Compatibility shim with the methods trainer.py expects.
    Works with window-level labels: reconstructs timeline using
    start_time + k*STEP_SEC and WINDOW_SEC, merges consecutive positive windows
    into predicted segments, and computes both timepoint and segment metrics.
    """
    def __init__(self, config):
        self.cfg = config

    def evaluate_model_timeline(self,
                                model,
                                data_loader,
                                device,
                                spindle_annotations: List[Tuple[float, float]],
                                start_time: float,
                                end_time: float,
                                threshold: float = 0.5) -> Dict[str, Any]:
        model.eval()
        probs_list, y_list = [], []
        with torch.no_grad():
            import torch
            for xb, yb in data_loader:
                xb = xb.to(device)
                logits = model(xb)
                # Accept logits shape [B,1] or [B] or [B,1,T] (take mean over T)
                if logits.dim() == 3:
                    logits = logits.mean(dim=2)
                if logits.size(-1) == 1:
                    logits = logits.squeeze(-1)
                pb = torch.sigmoid(logits).detach().cpu().numpy().ravel()
                yb = yb.detach().cpu().numpy().ravel()
                probs_list.append(pb)
                y_list.append(yb)
        if len(probs_list) == 0:
            return {"time_metrics": {}, "segment_metrics": {}}

        y_prob = np.concatenate(probs_list, axis=0)
        y_true = np.concatenate(y_list, axis=0).astype(int)
        y_pred = (y_prob >= float(threshold)).astype(int)

        # Reconstruct window start times (assuming shuffle=False)
        n = len(y_true)
        starts = start_time + np.arange(n) * float(self.cfg.STEP_SEC)

        # Time metrics (window-level)
        time_m = compute_basic_metrics(y_true, y_pred)
        time_m.update(compute_prob_metrics(y_true, y_prob))

        # Segment metrics
        pred_segments = _windows_to_segments(starts, float(self.cfg.WINDOW_SEC), y_pred)
        true_segments = spindle_annotations  # already absolute times (seconds)
        seg_tol = self.cfg.get('evaluation.segment_tolerance', 0.5)
        seg_m = segment_metrics(true_segments, pred_segments, tolerance=seg_tol)

        results = {
            "time_grid": starts,
            "probabilities": y_prob,
            "predictions": y_pred,
            "targets": y_true,
            "predicted_segments": pred_segments,
            "target_segments": true_segments,
            "time_metrics": time_m,
            "segment_metrics": seg_m,
        }
        return results

    def find_optimal_threshold_timeline(self,
                                        model,
                                        data_loader,
                                        device,
                                        spindle_annotations: List[Tuple[float, float]],
                                        start_time: float,
                                        end_time: float,
                                        test_thresholds: np.ndarray) -> Tuple[float, Dict[float, Dict[str, float]]]:
        # Run once to get probabilities and window starts
        model.eval()
        probs_list, y_list = [], []
        with torch.no_grad():
            import torch
            for xb, yb in data_loader:
                xb = xb.to(device)
                logits = model(xb)
                if logits.dim() == 3:
                    logits = logits.mean(dim=2)
                if logits.size(-1) == 1:
                    logits = logits.squeeze(-1)
                pb = torch.sigmoid(logits).detach().cpu().numpy().ravel()
                yb = yb.detach().cpu().numpy().ravel()
                probs_list.append(pb)
                y_list.append(yb)
        y_prob = np.concatenate(probs_list, axis=0)
        y_true = np.concatenate(y_list, axis=0).astype(int)
        n = len(y_true)
        starts = start_time + np.arange(n) * float(self.cfg.STEP_SEC)

        results: Dict[float, Dict[str, float]] = {}
        best_t = float(test_thresholds[0])
        best_score = -1.0

        for t in test_thresholds:
            t = float(t)
            y_pred = (y_prob >= t).astype(int)
            time_m = compute_basic_metrics(y_true, y_pred)
            pred_segments = _windows_to_segments(starts, float(self.cfg.WINDOW_SEC), y_pred)
            seg_tol = self.cfg.get('evaluation.segment_tolerance', 0.5)
            seg_m = segment_metrics(spindle_annotations, pred_segments, tolerance=seg_tol)
            # Combined score (avg of time F1 and segment F1)
            comb = 0.5 * (time_m["f1_score"] + seg_m["segment_f1"])
            results[t] = {
                "time_f1": time_m["f1_score"],
                "segment_f1": seg_m["segment_f1"],
                "combined_f1": comb,
                "time_metrics": time_m,
                "segment_metrics": seg_m,
            }
            if comb > best_score:
                best_score = comb
                best_t = t

        return best_t, results

    # --------- Reporting & plotting (minimal, safe to no-op) ---------

    def print_detailed_report(self, results: Dict[str, Any], title: str = "Evaluation"):
        tm = results.get("time_metrics", {})
        sm = results.get("segment_metrics", {})
        print(f"\n=== {title} ===")
        if tm:
            print(f"Time-based  -> F1: {tm.get('f1_score', 0):.4f}  "
                  f"P: {tm.get('precision', 0):.4f}  R: {tm.get('recall', 0):.4f}  "
                  f"Acc: {tm.get('accuracy', 0):.4f}")
            if "auc_roc" in tm:
                print(f"AUC-ROC: {tm['auc_roc']:.4f}  AUC-PR: {tm.get('auc_pr', 0):.4f}")
        if sm:
            print(f"Segment-based -> F1: {sm.get('segment_f1', 0):.4f}  "
                  f"P: {sm.get('segment_precision', 0):.4f}  R: {sm.get('segment_recall', 0):.4f}  "
                  f"TP/FP/FN: {sm.get('segment_tp', 0)}/{sm.get('segment_fp', 0)}/{sm.get('segment_fn', 0)}")

    def plot_timeline(self, results: Dict[str, Any], save_path: str = None):
        if not _HAS_PLT:
            return
        time_grid = results.get("time_grid")
        probs = results.get("probabilities")
        preds = results.get("predictions")
        if time_grid is None or probs is None:
            return
        plt.figure(figsize=(12, 3))
        plt.plot(time_grid, probs, label="prob")
        plt.plot(time_grid, preds * 1.02, ".", ms=2, label="pred")
        plt.ylim([-0.05, 1.1])
        plt.xlabel("Time (s)"); plt.ylabel("Prob / Pred")
        plt.legend(loc="upper right"); plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()
        plt.close()

    def plot_threshold_analysis(self, thresh_results: Dict[float, Dict[str, float]], save_path: str = None):
        if not _HAS_PLT or not thresh_results:
            return
        ths = sorted(thresh_results.keys())
        t_f1 = [thresh_results[t]["time_f1"] for t in ths]
        s_f1 = [thresh_results[t]["segment_f1"] for t in ths]
        c_f1 = [thresh_results[t]["combined_f1"] for t in ths]
        plt.figure(figsize=(8, 4))
        plt.plot(ths, t_f1, label="time F1")
        plt.plot(ths, s_f1, label="segment F1")
        plt.plot(ths, c_f1, label="combined F1")
        plt.xlabel("Threshold"); plt.ylabel("F1"); plt.grid(alpha=0.3); plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()
        plt.close()
