import os
import json
import argparse
from pathlib import Path

import yaml
import numpy as np
import torch
import mne
import matplotlib.pyplot as plt
import wandb

from Crnn import CRNN2D_BiGRU


# -----------------------------
# JSON utilities
# -----------------------------
def load_spindle_events(json_path: str, channel: str | None = None):
    """
    Returns list of (start_sec, end_sec) from JSON.
    If channel is provided, keep only events that include that channel.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    events = data.get("detected_spindles") or data.get("spindles") or []
    out = []

    chan = channel.lower() if channel else None

    for ev in events:
        if not isinstance(ev, dict):
            continue
        if "start" not in ev or "end" not in ev:
            continue

        if chan is not None:
            chs = ev.get("channel_names") or ev.get("channels") or []
            chs = [c.lower() for c in chs] if isinstance(chs, list) else []
            if chs and chan not in chs:
                continue

        try:
            s = float(ev["start"])
            e = float(ev["end"])
            if e > s:
                out.append((s, e))
        except Exception:
            pass

    out.sort(key=lambda x: x[0])
    return out


def mask_from_events(events, sfreq, n_samples, t0_sec=0.0):
    """
    Build boolean mask for a segment of length n_samples,
    where events are given in absolute seconds.
    Segment starts at absolute time t0_sec.
    """
    mask = np.zeros(n_samples, dtype=bool)
    for s_abs, e_abs in events:
        s = int(np.floor((s_abs - t0_sec) * sfreq))
        e = int(np.ceil((e_abs - t0_sec) * sfreq))
        s = max(0, s)
        e = min(n_samples, e)
        if e > s:
            mask[s:e] = True
    return mask


# -----------------------------
# Signal preprocessing
# -----------------------------
def preprocess_raw_segment(raw: mne.io.BaseRaw, channels, hp, lp, ref="car"):
    raw = raw.copy()
    raw.pick(channels)
    if ref == "car":
        raw.set_eeg_reference("average", verbose=False)
    raw.filter(hp, lp, verbose=False)
    data = raw.get_data()  # (C, T) in Volts
    return data


def zscore_per_channel(x):
    # x: (C, T)
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True) + 1e-6
    return (x - mu) / sd


# -----------------------------
# Model loading
# -----------------------------
def checkpoint_uses_se(state_dict: dict) -> bool:
    for k in state_dict.keys():
        if ".se." in k:
            return True
    return False


def find_checkpoint(artifact_dir: Path) -> Path:
    for name in ["best.pt", "model.pt", "checkpoint.pt"]:
        hits = list(artifact_dir.rglob(name))
        if hits:
            return hits[0]
    hits = list(artifact_dir.rglob("*.pt"))
    if not hits:
        raise FileNotFoundError(f"No .pt checkpoint found under {artifact_dir}")
    return hits[0]


# -----------------------------
# Clean window extraction
# -----------------------------
@torch.no_grad()
def find_clean_spindle_window(events, model, raw, cfg, device, window_sec,
                              max_search=50, min_confidence=0.6):
    
    dcfg = cfg["data"]
    hp = float(dcfg["filter"]["low"])
    lp = float(dcfg["filter"]["high"])
    channels = dcfg["channels"]
    ref = cfg.get("signal", {}).get("reference", "car")
    sfreq = float(dcfg["sfreq"])

    best = None
    best_score = -1.0
    best_smoothness = -1.0

    search_events = events[:max_search] if len(events) > max_search else events

    for (s_abs, e_abs) in search_events:
        duration = e_abs - s_abs

        # Skip very short or very long spindles
        if duration < 0.3 or duration > 3.0:
            continue

        # Center window on spindle
        mid = (s_abs + e_abs) / 2.0
        t0 = max(0.0, mid - window_sec / 2.0)
        t1 = t0 + window_sec

        try:
            seg_raw = raw.copy().crop(tmin=t0, tmax=t1, include_tmax=False)
        except:
            continue

        x_seg = preprocess_raw_segment(seg_raw, channels, hp, lp, ref=ref)

        # Single window prediction (no sliding)
        x_norm = zscore_per_channel(x_seg).astype(np.float32)
        x_t = torch.from_numpy(x_norm).unsqueeze(0).to(device)

        logits = model(x_t)
        if logits.ndim == 3:
            logits = logits[:, 0, :]
        probs = torch.sigmoid(logits)[0].detach().cpu().numpy().astype(np.float32)

        # Get probability within spindle region
        s_local = int(round((s_abs - t0) * sfreq))
        e_local = int(round((e_abs - t0) * sfreq))
        s_local = max(0, s_local)
        e_local = min(len(probs), e_local)

        if e_local <= s_local:
            continue

        spindle_probs = probs[s_local:e_local]

        # Quality metrics
        mean_prob = float(spindle_probs.mean())
        min_prob = float(spindle_probs.min())

        # Smoothness: lower variance = smoother
        prob_std = float(spindle_probs.std())
        smoothness = 1.0 / (1.0 + prob_std)  # Higher is smoother

        # Combined score: high mean, high min, smooth
        score = mean_prob * 0.5 + min_prob * 0.3 + smoothness * 0.2

        # Require minimum confidence
        if mean_prob < min_confidence:
            continue

        if score > best_score:
            best_score = score
            best_smoothness = smoothness
            best = {
                'start_abs': s_abs,
                'end_abs': e_abs,
                'duration': duration,
                't0': t0,
                'x_seg': x_seg,
                'probs': probs,
                'mean_prob': mean_prob,
                'min_prob': min_prob,
                'smoothness': smoothness,
                'score': score
            }

    if best is None:
        raise RuntimeError(
            f"No clean spindle window found. Searched {len(search_events)} events. "
            f"Try lowering --min_confidence or using different data."
        )

    return best


# -----------------------------
# Plotting
# -----------------------------
def plot_clean_spindle(eeg_1ch, gt_mask, probs, sfreq, thr, metadata, save_path=None):
    """
    Plot a clean spindle example with detailed metadata
    """
    t = np.arange(len(eeg_1ch)) / sfreq

    fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    # (A) Ground truth
    ax[0].plot(t, eeg_1ch, linewidth=1, color='C0')
    ax[0].fill_between(t, eeg_1ch.min(), eeg_1ch.max(), where=gt_mask,
                       alpha=0.3, color='C0', label='Annotated spindle')
    ax[0].set_title(
        f"(A) Ground-truth spindle annotation — ch={metadata['channel']} "
        f"(duration={metadata['duration']:.2f}s)"
    )
    ax[0].set_ylabel("EEG (filtered)")
    ax[0].legend(loc='upper right')
    ax[0].grid(True, alpha=0.3)

    # (B) Prediction
    ax[1].plot(t, eeg_1ch, linewidth=1, color='C0', alpha=0.7, label='EEG')
    axp = ax[1].twinx()
    axp.plot(t, probs, linewidth=2.5, color='C1', label='P(spindle)')
    axp.fill_between(t, 0, probs, alpha=0.2, color='C1')
    axp.axhline(thr, linestyle='--', linewidth=1.5, color='C3', label=f'threshold={thr}')

    # Add spindle region markers
    spindle_region = np.where(gt_mask)[0]
    if len(spindle_region) > 0:
        t_start = t[spindle_region[0]]
        t_end = t[spindle_region[-1]]
        axp.axvspan(t_start, t_end, alpha=0.1, color='green', label='GT spindle region')

    ax[1].set_title(
        f"(B) CRNN prediction — mean P={metadata['mean_prob']:.3f}, "
        f"min P={metadata['min_prob']:.3f}, smoothness={metadata['smoothness']:.3f}"
    )
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("EEG (filtered)")
    axp.set_ylabel("P(spindle)", color='C1')
    axp.tick_params(axis='y', labelcolor='C1')
    axp.set_ylim([-0.05, 1.05])

    # Combine legends
    lines1, labels1 = ax[1].get_legend_handles_labels()
    lines2, labels2 = axp.get_legend_handles_labels()
    axp.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# -----------------------------
# MAIN
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Generate clean spindle detection example (ideal case for thesis)"
    )
    ap.add_argument("--config", default="config70.yaml")
    ap.add_argument("--artifact_dir", required=True)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--channel_plot", default="C3")
    ap.add_argument("--window_sec", type=float, default=4.0,
                    help="Window size in seconds (centered on spindle)")
    ap.add_argument("--min_confidence", type=float, default=0.6,
                    help="Minimum mean probability to consider")
    ap.add_argument("--max_search", type=int, default=100,
                    help="Maximum events to search")
    ap.add_argument("--save_png", default="clean_spindle_example.png")
    ap.add_argument("--wandb_project", default=None)
    ap.add_argument("--edf_file", default=None)
    args = ap.parse_args()

    # Load config
    cfg = yaml.safe_load(open(args.config, "r"))
    dcfg = cfg["data"]
    scfg = cfg["spectrogram"]
    mcfg = cfg["model"]

    # W&B settings
    wandb_cfg = cfg.get("logging", {}).get("wandb", {})
    entity = wandb_cfg.get("entity", cfg.get("project", {}).get("entity", None))
    project = args.wandb_project or wandb_cfg.get("project", "CRNN-sweep")

    # Paths
    if args.edf_file:
        edf_path = args.edf_file
    else:
        edfs = sorted(Path(dcfg["edf"]["dir"]).glob("*.edf"))
        if not edfs:
            raise FileNotFoundError(f"No EDF found in {dcfg['edf']['dir']}")
        edf_path = str(edfs[0])

    if not os.path.exists(edf_path):
        raise FileNotFoundError(f"EDF file not found: {edf_path}")

    json_path = dcfg["edf"]["labels_json"]
    sfreq = float(dcfg["sfreq"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] EDF:  {edf_path}")
    print(f"[INFO] JSON: {json_path}")

    # Load raw
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # Load events
    events = load_spindle_events(json_path, channel=args.channel_plot)
    if not events:
        events = load_spindle_events(json_path, channel=None)

    if not events:
        raise RuntimeError("No spindle events found in JSON.")

    print(f"[INFO] Found {len(events)} spindle events")

    # Load checkpoint
    artifact_dir = Path(args.artifact_dir)
    ckpt_path = find_checkpoint(artifact_dir)
    state = torch.load(ckpt_path, map_location=device)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state

    use_se = checkpoint_uses_se(sd)
    print(f"[INFO] Using checkpoint: {ckpt_path}")
    print(f"[INFO] Auto-detected use_se={use_se}")

    # Build model
    model = CRNN2D_BiGRU(
        c_in=mcfg["in_channels"],
        base_ch=mcfg["base_ch"],
        fpn_ch=mcfg["fpn_ch"],
        rnn_hidden=mcfg["rnn_hidden"],
        rnn_layers=mcfg["rnn_layers"],
        bidirectional=mcfg["bidirectional"],
        bias_init_prior=mcfg.get("bias_init_prior", None),
        use_se=use_se,
        sfreq=int(sfreq),
        n_fft=scfg["n_fft"],
        hop_length=scfg["hop_length"],
        win_length=scfg["win_length"],
        center=scfg["center"],
        power=scfg["power"],
        upsample_mode=mcfg.get("upsample_mode", "linear"),
    ).to(device).eval()

    model.load_state_dict(sd, strict=True)

    # Channel setup
    chs = dcfg["channels"]
    if args.channel_plot in chs:
        ch_idx = chs.index(args.channel_plot)
        ch_name = args.channel_plot
    else:
        ch_idx = 0
        ch_name = chs[0]
        print(f"[WARN] channel_plot {args.channel_plot} not found; using {ch_name}")

    # Find clean spindle
    print(f"[INFO] Searching for clean spindle (window={args.window_sec}s, "
          f"min_confidence={args.min_confidence})...")

    best = find_clean_spindle_window(
        events, model, raw, cfg, device,
        window_sec=args.window_sec,
        max_search=args.max_search,
        min_confidence=args.min_confidence
    )

    print(f"[INFO] Found clean spindle:")
    print(f"       Time: {best['start_abs']:.2f}s - {best['end_abs']:.2f}s")
    print(f"       Duration: {best['duration']:.2f}s")
    print(f"       Mean probability: {best['mean_prob']:.3f}")
    print(f"       Min probability: {best['min_prob']:.3f}")
    print(f"       Smoothness: {best['smoothness']:.3f}")
    print(f"       Overall score: {best['score']:.3f}")

    # Prepare plot data
    gt_mask = mask_from_events(
        [(best['start_abs'], best['end_abs'])],
        sfreq,
        best['x_seg'].shape[1],
        t0_sec=best['t0']
    )
    eeg_plot = best['x_seg'][ch_idx]

    metadata = {
        'channel': ch_name,
        'duration': best['duration'],
        'mean_prob': best['mean_prob'],
        'min_prob': best['min_prob'],
        'smoothness': best['smoothness']
    }

    # Create plot
    out_png = Path(args.save_png)
    fig = plot_clean_spindle(
        eeg_plot, gt_mask, best['probs'], sfreq, args.thr, metadata, save_path=out_png
    )

    print(f"[INFO] Saved: {out_png}")

    # Log to W&B
    wandb.init(
        project=project,
        entity=entity,
        job_type="analysis/clean_example",
        name=f"CLEAN_SPINDLE_{ch_name}",
        config={
            "edf": edf_path,
            "json": json_path,
            "artifact_dir": str(artifact_dir),
            "checkpoint": str(ckpt_path),
            "use_se": use_se,
            "thr": args.thr,
            "window_sec": args.window_sec,
            "min_confidence": args.min_confidence,
            "spindle_time": f"{best['start_abs']:.2f}-{best['end_abs']:.2f}",
            "mean_prob": best['mean_prob'],
            "min_prob": best['min_prob'],
            "smoothness": best['smoothness'],
        },
        reinit=True,
    )

    wandb.log({
        "clean_spindle_example": wandb.Image(fig),
        "png_path": str(out_png)
    })
    wandb.finish()
    plt.close(fig)

    print("[DONE] Clean spindle example generated and logged to W&B")


if __name__ == "__main__":
    main()