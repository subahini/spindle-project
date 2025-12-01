import os
import glob
import json
import argparse

import numpy as np
import mne
import yaml
import wandb
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, precision_recall_curve, auc

from schimicek_spindle import detect_spindles_schimicek, SchimicekParams


# =========================
# Sample-Level Metrics
# =========================
def evaluate_mask(pred_mask: np.ndarray, true_mask: np.ndarray):
    """Compute sample-level confusion matrix metrics."""
    pred = np.asarray(pred_mask, dtype=bool).ravel()
    true = np.asarray(true_mask, dtype=bool).ravel()
    assert pred.shape == true.shape

    tp = np.logical_and(pred, true).sum()
    tn = np.logical_and(~pred, ~true).sum()
    fp = np.logical_and(pred, ~true).sum()
    fn = np.logical_and(~pred, true).sum()

    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
    }


# =========================
# Event-Level Metrics
# =========================
def mask_to_events(mask: np.ndarray, fs: float) -> list:
    """Convert boolean mask to list of (start_idx, end_idx) tuples."""
    events = []
    in_event = False
    start_idx = 0

    for i in range(len(mask)):
        if mask[i] and not in_event:
            start_idx = i
            in_event = True
        elif not mask[i] and in_event:
            events.append((start_idx, i))
            in_event = False

    if in_event:
        events.append((start_idx, len(mask)))

    return events


def evaluate_events(pred_mask: np.ndarray, true_mask: np.ndarray,
                    fs: float, tolerance_samples: int = None) -> dict:
    """
    Compute event-level metrics.

    An event is considered a true positive if it overlaps with a ground truth event
    within tolerance_samples (default: 0.5 seconds worth of samples).
    """
    if tolerance_samples is None:
        tolerance_samples = int(0.5 * fs)

    pred_events = mask_to_events(pred_mask, fs)
    true_events = mask_to_events(true_mask, fs)

    matched_true = set()
    tp_events = 0

    for pred_start, pred_end in pred_events:
        best_overlap = 0
        best_true_idx = -1

        for true_idx, (true_start, true_end) in enumerate(true_events):
            if true_idx in matched_true:
                continue

            overlap_start = max(pred_start, true_start)
            overlap_end = min(pred_end, true_end)
            overlap = max(0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_true_idx = true_idx

        if best_true_idx >= 0 and best_overlap > tolerance_samples:
            tp_events += 1
            matched_true.add(best_true_idx)

    fp_events = len(pred_events) - tp_events
    fn_events = len(true_events) - len(matched_true)

    eps = 1e-8
    event_precision = tp_events / (tp_events + fp_events + eps)
    event_recall = tp_events / (tp_events + fn_events + eps)
    event_f1 = 2 * event_precision * event_recall / (event_precision + event_recall + eps)

    return {
        "event_tp": float(tp_events),
        "event_fp": float(fp_events),
        "event_fn": float(fn_events),
        "event_precision": float(event_precision),
        "event_recall": float(event_recall),
        "event_f1": float(event_f1),
        "n_pred_events": float(len(pred_events)),
        "n_true_events": float(len(true_events)),
    }


# =========================
# Data Loading
# =========================
def load_mask(json_path, sfreq, n_samples, channel):
    with open(json_path, "r") as f:
        data = json.load(f)

    events = data.get("detected_spindles", [])
    chan_key = channel.lower() + "-ref"

    mask = np.zeros(n_samples, dtype=bool)

    for ev in events:
        if chan_key not in [c.lower() for c in ev["channel_names"]]:
            continue

        start = float(ev["start"])
        end = float(ev["end"])

        s = int(start * sfreq)
        e = int(end * sfreq)

        s = max(0, min(n_samples, s))
        e = max(s, min(n_samples, e))

        mask[s:e] = True

    return mask


# =========================
# Main Evaluation
# =========================
def main(args):
    # ------------- load config -------------
    cfg_path = args.config
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_root = cfg["data_root"]
    edf_dir = os.path.join(data_root, cfg.get("edf_subdir", "edf"))
    label_dir = os.path.join(data_root, cfg.get("label_subdir", "labels"))

    # Channel: CLI (sweep) overrides config
    channel = args.channel if args.channel is not None else cfg.get("channel", None)
    if channel is None:
        raise ValueError("No channel specified (set in config under 'channel' or use --channel).")

    # Schimicek config block from YAML
    sch_cfg = cfg["schimicek"]

    # ---- Overrides from CLI / Sweep (optional) ----
    if getattr(args, "schimicek_amplitude_threshold_uv", None) is not None:
        sch_cfg["amplitude_threshold_uv"] = args.schimicek_amplitude_threshold_uv

    if getattr(args, "schimicek_min_duration_s", None) is not None:
        sch_cfg["min_duration_s"] = args.schimicek_min_duration_s

    if getattr(args, "schimicek_alpha_ratio_threshold", None) is not None:
        sch_cfg["alpha_ratio_threshold"] = args.schimicek_alpha_ratio_threshold

    if getattr(args, "schimicek_muscle_rms_threshold_uv", None) is not None:
        sch_cfg["muscle_rms_threshold_uv"] = args.schimicek_muscle_rms_threshold_uv

    # W&B block: make sweep-safe
    wandb_cfg = cfg.get("wandb") or {}
    project = wandb_cfg.get("project", "debug_baseline")
    entity = wandb_cfg.get("entity", None)
    run_name_prefix = wandb_cfg.get("run_name_prefix", "schimicek")
    run_name = f"{run_name_prefix}_{channel}"

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=cfg,
        reinit=True,
    )

    # ===== CRITICAL FIX: Override cfg with wandb.config sweep parameters =====
    print("\n" + "=" * 70)
    print("[SWEEP PARAMETER CHECK]")
    print("=" * 70)
    print(f"Before override: amplitude_threshold_uv = {cfg['schimicek']['amplitude_threshold_uv']}")

    # Check if wandb has sweep parameters and override cfg
    if "schimicek.amplitude_threshold_uv" in wandb.config:
        cfg["schimicek"]["amplitude_threshold_uv"] = wandb.config["schimicek.amplitude_threshold_uv"]
    if "schimicek.min_duration_s" in wandb.config:
        cfg["schimicek"]["min_duration_s"] = wandb.config["schimicek.min_duration_s"]
    if "schimicek.alpha_ratio_threshold" in wandb.config:
        cfg["schimicek"]["alpha_ratio_threshold"] = wandb.config["schimicek.alpha_ratio_threshold"]
    if "schimicek.muscle_rms_threshold_uv" in wandb.config:
        cfg["schimicek"]["muscle_rms_threshold_uv"] = wandb.config["schimicek.muscle_rms_threshold_uv"]

    print(f"After override: amplitude_threshold_uv = {cfg['schimicek']['amplitude_threshold_uv']}")
    print(f"\nFinal parameters being used:")
    print(f"  amplitude_threshold_uv: {cfg['schimicek']['amplitude_threshold_uv']}")
    print(f"  min_duration_s: {cfg['schimicek']['min_duration_s']}")
    print(f"  alpha_ratio_threshold: {cfg['schimicek']['alpha_ratio_threshold']}")
    print(f"  muscle_rms_threshold_uv: {cfg['schimicek']['muscle_rms_threshold_uv']}")
    print("=" * 70 + "\n")

    # (rest of your code: EDF loop, metrics, logging)

    edf_files = sorted(glob.glob(os.path.join(edf_dir, "*.edf")))
    print(f"Channel: {channel}")
    print(f"Found {len(edf_files)} EDF files in {edf_dir}")

    all_true = []
    all_pred = []
    all_scores = []
    all_sample_metrics = []
    all_event_metrics = []

    for edf_path in edf_files:
        base = os.path.splitext(os.path.basename(edf_path))[0]
        clean_base = base.replace("_raw", "").replace("_filtered", "")

        json_candidates = glob.glob(os.path.join(label_dir, f"*{clean_base}*.json"))
        if len(json_candidates) == 0:
            print(f"[WARN] No JSON for {base} → looking for '*{clean_base}*.json' in {label_dir}")
            continue

        json_path = json_candidates[0]
        print(f"  Matched JSON: {os.path.basename(json_path)}")

        print(f"\n[PROCESS] {base} ({channel})")

        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        sfreq = float(raw.info["sfreq"])
        n_samples = raw.n_times

        if channel not in raw.ch_names:
            print(f"[WARN] Channel {channel} not in {edf_path}. Available: {raw.ch_names}")
            continue

        x = raw.get_data(picks=[channel])[0]
        eeg_uv = x * 1e6

        true_mask = load_mask(json_path, sfreq, n_samples, channel)

        sch_cfg = cfg["schimicek"]
        params = SchimicekParams(
            fs=sfreq if sch_cfg["fs"] is None else sch_cfg["fs"],
            spindle_band=tuple(sch_cfg["spindle_band"]),
            alpha_band=tuple(sch_cfg["alpha_band"]),
            muscle_band=tuple(sch_cfg["muscle_band"]),
            amplitude_threshold_uv=sch_cfg["amplitude_threshold_uv"],
            min_duration_s=sch_cfg["min_duration_s"],
            epoch_length_s=sch_cfg["epoch_length_s"],
            alpha_ratio_threshold=sch_cfg["alpha_ratio_threshold"],
            muscle_rms_threshold_uv=sch_cfg["muscle_rms_threshold_uv"],
            filter_order=sch_cfg["filter_order"],
        )

        results = detect_spindles_schimicek(eeg_uv, params)

        rec_sample_metrics = evaluate_mask(results["spindle_mask"], true_mask)
        print(
            f"  Sample-level - precision: {rec_sample_metrics['precision']:.3f}, "
            f"recall: {rec_sample_metrics['recall']:.3f}, "
            f"f1: {rec_sample_metrics['f1']:.3f}"
        )

        rec_event_metrics = evaluate_events(
            results["spindle_mask"], true_mask, sfreq,
            tolerance_samples=int(0.5 * sfreq)
        )
        print(
            f"  Event-level - precision: {rec_event_metrics['event_precision']:.3f}, "
            f"recall: {rec_event_metrics['event_recall']:.3f}, "
            f"f1: {rec_event_metrics['event_f1']:.3f}"
        )
        print(
            f"    Predicted {rec_event_metrics['n_pred_events']:.0f} events, "
            f"ground truth has {rec_event_metrics['n_true_events']:.0f} events"
        )

        wandb.log({f"{base}/sample_{k}": v for k, v in rec_sample_metrics.items()})
        wandb.log({f"{base}/event_{k}": v for k, v in rec_event_metrics.items()})

        scores = results["peak_to_peak"].astype(float)
        p2p_min, p2p_max = scores.min(), scores.max()
        if p2p_max > p2p_min:
            scores = (scores - p2p_min) / (p2p_max - p2p_min)
        else:
            scores = np.zeros_like(scores)

        all_true.append(true_mask.astype(int))
        all_pred.append(results["spindle_mask"].astype(int))
        all_scores.append(scores)
        all_sample_metrics.append(rec_sample_metrics)
        all_event_metrics.append(rec_event_metrics)

    if not all_true:
        print("No recordings with labels processed. Exiting.")
        run.finish()
        return

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    scores = np.concatenate(all_scores)

    scores_2d = np.stack([1.0 - scores, scores], axis=1)
    class_names = ["no_spindle", "spindle"]

    global_sample_metrics = evaluate_mask(y_pred.astype(bool), y_true.astype(bool))

    print("\n" + "=" * 70)
    print(f"=== GLOBAL SAMPLE-LEVEL METRICS (all EDFs, channel: {channel}) ===")
    print("=" * 70)
    for k, v in global_sample_metrics.items():
        print(f"  {k}: {v:.4f}")

    wandb.log({f"global/sample_{k}": v for k, v in global_sample_metrics.items()})

    agg_event_tp = sum(m["event_tp"] for m in all_event_metrics)
    agg_event_fp = sum(m["event_fp"] for m in all_event_metrics)
    agg_event_fn = sum(m["event_fn"] for m in all_event_metrics)
    agg_n_pred = sum(m["n_pred_events"] for m in all_event_metrics)
    agg_n_true = sum(m["n_true_events"] for m in all_event_metrics)

    eps = 1e-8
    global_event_precision = agg_event_tp / (agg_event_tp + agg_event_fp + eps)
    global_event_recall = agg_event_tp / (agg_event_tp + agg_event_fn + eps)
    global_event_f1 = 2 * global_event_precision * global_event_recall / (
        global_event_precision + global_event_recall + eps
    )

    print("\n" + "=" * 70)
    print(f"=== GLOBAL EVENT-LEVEL METRICS (all EDFs, channel: {channel}) ===")
    print("=" * 70)
    print(f"  event_tp: {agg_event_tp:.0f}")
    print(f"  event_fp: {agg_event_fp:.0f}")
    print(f"  event_fn: {agg_event_fn:.0f}")
    print(f"  event_precision: {global_event_precision:.4f}")
    print(f"  event_recall: {global_event_recall:.4f}")
    print(f"  event_f1: {global_event_f1:.4f}")
    print(f"  Total predicted events: {agg_n_pred:.0f}")
    print(f"  Total ground truth events: {agg_n_true:.0f}")

    wandb.log({
        "global/event_tp": agg_event_tp,
        "global/event_fp": agg_event_fp,
        "global/event_fn": agg_event_fn,
        "global/event_precision": global_event_precision,
        "global/event_recall": global_event_recall,
        "global/event_f1": global_event_f1,
        "global/n_pred_events": agg_n_pred,
        "global/n_true_events": agg_n_true,
    })

    cm = wandb.plot.confusion_matrix(
        y_true=y_true,
        preds=y_pred,
        class_names=class_names,
    )
    wandb.log({"global/confusion_matrix": cm})

    y_true_flat = np.asarray(y_true).ravel()
    scores_flat = np.asarray(scores).ravel()

    roc_auc = float("nan")
    pr_auc = float("nan")

    if y_true_flat.size > 1 and scores_flat.size > 1 and np.unique(y_true_flat).size > 1:
        fpr, tpr, _ = roc_curve(y_true_flat, scores_flat)
        roc_auc = auc(fpr, tpr)

        precision_vals, recall_vals, _ = precision_recall_curve(y_true_flat, scores_flat)
        pr_auc = auc(recall_vals, precision_vals)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (sklearn)")
        plt.legend()
        wandb.log({"sklearn/roc_image": wandb.Image(plt)})
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.plot(recall_vals, precision_vals, label=f"AUC={pr_auc:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("PR Curve (sklearn)")
        plt.legend()
        wandb.log({"sklearn/pr_image": wandb.Image(plt)})
        plt.close()

        wandb.log({
            "sklearn/roc_auc": roc_auc,
            "sklearn/pr_auc": pr_auc,
        })
    else:
        print("Not enough positive/negative samples to compute sklearn ROC/PR.")

    print("\n" + "=" * 70)
    print(f"Sklearn ROC AUC: {roc_auc:.4f}")
    print(f"Sklearn PR AUC:  {pr_auc:.4f}")
    print("Metrics logged to wandb. Run complete.")
    print("=" * 70)

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to YAML config."
    )
    parser.add_argument(
        "--channel",
        type=str,
        default=None,
        help="Override channel from config (e.g. C3, C4, F3, F4)."
    )

    # Schimicek sweep parameters (optional overrides)
    parser.add_argument(
        "--schimicek.amplitude_threshold_uv",
        type=float,
        dest="schimicek_amplitude_threshold_uv",
        default=None,
        help="Override Schimicek amplitude_threshold_uv (µV)."
    )
    parser.add_argument(
        "--schimicek.min_duration_s",
        type=float,
        dest="schimicek_min_duration_s",
        default=None,
        help="Override Schimicek min_duration_s (s)."
    )
    parser.add_argument(
        "--schimicek.alpha_ratio_threshold",
        type=float,
        dest="schimicek_alpha_ratio_threshold",
        default=None,
        help="Override Schimicek alpha_ratio_threshold."
    )
    parser.add_argument(
        "--schimicek.muscle_rms_threshold_uv",
        type=float,
        dest="schimicek_muscle_rms_threshold_uv",
        default=None,
        help="Override Schimicek muscle_rms_threshold_uv (µV)."
    )

    args = parser.parse_args()
    main(args)
