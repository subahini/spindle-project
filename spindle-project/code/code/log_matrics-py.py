# logging_extra.py
"""
Extra, optional logging for sample-level spindle detection.
- Stitches overlapping windows into a single timeline (like metrics.py)
- Computes TP/FP/TN/FN and logs counts
- Saves a CSV with per-sample: time_sec, y_true, prob, y_pred, hit_type
- If W&B is active, uploads the CSV as an artifact and logs counts

Usage (in main.py after evaluation):
    from logging_extra import export_timeline_and_confusion
    export_timeline_and_confusion(model, test_loader, cfg)
"""

from typing import Optional, Tuple, List, Dict
import os
import numpy as np
import torch

# optional W&B
try:
    import wandb
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False


def _model_expects_2d(model: torch.nn.Module) -> bool:
    if hasattr(model, "_expected_input"):
        return getattr(model, "_expected_input") == "B1CT"
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            return True
    return False


def _stitch_timeline(
    model: torch.nn.Module,
    loader,
    sfreq: float,
    window_sec: float,
    step_sec: float,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        probs_glob: [T_total] float32
        y_true_glob: [T_total] int64 (0/1)
        y_pred_glob: [T_total] int64 (0/1) at default threshold 0.5
    """
    model = model.to(device).eval()
    exp2d = _model_expects_2d(model)
    win_T  = int(round(window_sec * sfreq))
    step_T = int(round(step_sec  * sfreq))

    probs_sum = probs_cnt = y_true_sum = y_true_cnt = None

    with torch.no_grad():
        cursor = 0
        for xb, yb in loader:
            xb = xb.to(device)   # [B,C,T]
            yb = yb.to(device)   # [B,T]
            if exp2d and xb.dim() == 3:
                xb = xb.unsqueeze(1)  # [B,1,C,T]
            if (not exp2d) and xb.dim() == 4 and xb.size(1) == 1:
                xb = xb.squeeze(1)    # [B,C,T]

            logits = model(xb)        # [B,T] or [B,1,T]
            if logits.dim() == 3 and logits.size(1) == 1:
                logits = logits.squeeze(1)
            B, T = logits.shape
            assert T == win_T, f"Model output T={T} must equal window length {win_T}"

            probs = torch.sigmoid(logits).cpu().numpy()
            y_np  = yb.cpu().numpy()

            target_len = cursor + B * step_T + (win_T - step_T)
            if probs_sum is None:
                probs_sum = np.zeros(target_len, dtype=np.float64)
                probs_cnt = np.zeros(target_len, dtype=np.int32)
                y_true_sum = np.zeros(target_len, dtype=np.float64)
                y_true_cnt = np.zeros(target_len, dtype=np.int32)
            elif target_len > probs_sum.shape[0]:
                grow = target_len - probs_sum.shape[0]
                probs_sum = np.concatenate([probs_sum, np.zeros(grow)])
                probs_cnt = np.concatenate([probs_cnt, np.zeros(grow, dtype=np.int32)])
                y_true_sum = np.concatenate([y_true_sum, np.zeros(grow)])
                y_true_cnt = np.concatenate([y_true_cnt, np.zeros(grow, dtype=np.int32)])

            for i in range(B):
                start = cursor + i * step_T
                end   = start + win_T
                probs_sum[start:end] += probs[i]
                probs_cnt[start:end] += 1
                y_true_sum[start:end] += y_np[i]
                y_true_cnt[start:end] += 1

            cursor += B * step_T

    valid = probs_cnt > 0
    probs_glob = np.zeros_like(probs_sum, dtype=np.float32)
    y_true_glob = np.zeros_like(y_true_sum, dtype=np.float32)
    probs_glob[valid] = (probs_sum[valid] / np.maximum(1, probs_cnt[valid])).astype(np.float32)
    y_true_glob[valid] = (y_true_sum[valid] / np.maximum(1, y_true_cnt[valid])).astype(np.float32)
    y_true_glob = (y_true_glob >= 0.5).astype(np.int64)

    y_pred_glob = (probs_glob >= 0.5).astype(np.int64)  # fixed 0.5 here; you can change if needed

    return probs_glob, y_true_glob, y_pred_glob


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def export_timeline_and_confusion(
    model: torch.nn.Module,
    loader,
    cfg,
    save_dir: Optional[str] = None,
    device: Optional[str] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Creates artifacts:
      - CSV: timeline_{run_name or model}.csv  with per-sample info
      - Logs TP/FP/TN/FN counts to console and W&B (if enabled)
    Returns a small dict of summary stats for convenience.
    """
    device = device or cfg.trainer.device
    save_dir = save_dir or "./artifacts"
    os.makedirs(save_dir, exist_ok=True)

    probs, y_true, y_pred = _stitch_timeline(
        model, loader,
        sfreq=cfg.data.sfreq,
        window_sec=cfg.data.window_sec,
        step_sec=cfg.data.step_sec,
        device=device,
    )

    # Confusion counts
    counts = _confusion_counts(y_true, y_pred)
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
    total = int(len(y_true))

    # Build CSV with per-sample info
    t = np.arange(len(y_true), dtype=np.float32) / float(cfg.data.sfreq)
    hit_type = np.full(len(y_true), "TN", dtype=object)
    hit_type[(y_true == 1) & (y_pred == 1)] = "TP"
    hit_type[(y_true == 0) & (y_pred == 1)] = "FP"
    hit_type[(y_true == 1) & (y_pred == 0)] = "FN"

    run_tag = cfg.logging.run_name or cfg.model.name
    out_csv = os.path.join(save_dir, f"timeline_{run_tag}.csv")
    # Save CSV (manual writer to avoid pandas dependency)
    with open(out_csv, "w") as f:
        f.write("time_sec,y_true,prob,y_pred,hit_type\n")
        for i in range(len(y_true)):
            f.write(f"{t[i]:.4f},{int(y_true[i])},{float(probs[i]):.6f},{int(y_pred[i])},{hit_type[i]}\n")

    print(f"[extra] Saved timeline CSV → {out_csv}")
    print(f"[extra] Counts: TP={tp} FP={fp} TN={tn} FN={fn} total_samples={total}")

    # Log to W&B if available & enabled
    if _HAS_WANDB and ("WANDB_DISABLED" not in os.environ):
        # Scalar summary
        try:
            wandb.log({
                "extra/tp": tp, "extra/fp": fp, "extra/tn": tn, "extra/fn": fn,
                "extra/threshold_used": float(threshold),
                "extra/total_samples": total
            })
            # Upload CSV as artifact (nice to download later)
            art = wandb.Artifact(f"timeline_{run_tag}", type="evaluation")
            art.add_file(out_csv, name=os.path.basename(out_csv))
            wandb.log_artifact(art)
            print("[extra] Uploaded confusion counts and CSV to W&B.")
        except Exception as e:
            print(f"[extra] W&B logging failed (non-fatal): {e}")

    # Simple precision/recall/F1 for quick glance
    def safe_div(a, b): return (a / b) if b > 0 else 0.0
    prec = safe_div(tp, tp + fp)
    rec  = safe_div(tp, tp + fn)
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": float(prec), "recall": float(rec), "f1": float(f1),
        "threshold_used": float(threshold),
        "csv_path": out_csv,
    }
