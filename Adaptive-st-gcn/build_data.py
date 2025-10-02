#!/usr/bin/env python3
"""
Build ST-GCN training arrays from RAW EEG + JSON labels
------------------------------------------------------
This script reads raw EEG files (EDF/BDF/FIF) and JSON spindle labels, then creates:
  - windows.npy           -> float32 (N, C, T)           EEG windows
  - labels_framewise.npy  -> float32 (N, T)               0/1 per time sample
  - labels_per_channel.npy-> float32 (N, C, T) (optional) 0/1 per channel & time

JSON formats supported (auto-detected):
  1) Global intervals:   {"spindles": [{"start": 12.3, "end": 13.8}, ...]}
  2) Per-channel dict:   {"per_channel": {"C3": [{"start": ...,"end": ...}], "C4": [...]}}
  3) Flat list:          [{"start": s, "end": e}]  (treated as global)

Usage examples
--------------
# Build from a folder of EDFs where each EDF has a matching JSON next to it
python build_data.py \
  --raw_dir ./raw \
  --labels ./labels \
  --channels F3,F4,C3,C4,O1,O2,F7,F8,T3,T4,P3,P4,Fz,Cz,Pz,Oz \
  --sfreq 200 --win 2.0 --stride 1.0 --band 10 16 \
  --out_dir ./data

# If you have one EDF + one JSON
python build_data.py --raw ./raw/sub01.edf --labels ./labels/sub01.json --channels C3,C4,Cz --out_dir ./data

Notes
-----
- Requires: mne (preferred) or pyedflib (fallback for EDF). Install with: pip install mne pyedflib
- You can skip bandpass by omitting --band.
- If per-channel labels are absent, only framewise labels will be saved.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

# Optional deps
try:
    import mne  # type: ignore
except Exception:
    mne = None
try:
    import pyedflib  # type: ignore
except Exception:
    pyedflib = None


# -----------------------------
# JSON parsing
# -----------------------------

def _load_json(path: Path) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)


def parse_label_json(j: Dict) -> Tuple[List[Tuple[float, float]], Dict[str, List[Tuple[float, float]]]]:
    """Return (global_intervals, per_channel_intervals).
    Each interval is (start_sec, end_sec).
    """
    global_iv = []  # type: List[Tuple[float,float]]
    per_ch = {}     # type: Dict[str, List[Tuple[float,float]]]

    if isinstance(j, list):
        # flat list of intervals
        for it in j:
            s = float(it.get('start', it.get('onset', 0.0)))
            e = float(it.get('end', it.get('offset', s)))
            global_iv.append((s, e))
        return global_iv, per_ch

    if 'spindles' in j and isinstance(j['spindles'], list):
        for it in j['spindles']:
            s = float(it.get('start', it.get('onset', 0.0)))
            e = float(it.get('end', it.get('offset', s)))
            global_iv.append((s, e))

    if 'per_channel' in j and isinstance(j['per_channel'], dict):
        for ch, lst in j['per_channel'].items():
            per_ch[ch] = []
            for it in lst:
                s = float(it.get('start', it.get('onset', 0.0)))
                e = float(it.get('end', it.get('offset', s)))
                per_ch[ch].append((s, e))

    # Allow alternative keying like {"C3": [{...}], "C4": [...]}
    if not per_ch:
        # Heuristic: if top-level contains channel-like keys
        for k, v in j.items():
            if isinstance(v, list) and all(isinstance(it, dict) and ('start' in it or 'onset' in it) for it in v):
                per_ch[k] = [(float(it.get('start', it.get('onset', 0.0))), float(it.get('end', it.get('offset', 0.0)))) for it in v]

    return global_iv, per_ch


# -----------------------------
# RAW loading
# -----------------------------

def load_raw_signals(path: Path, pick_channels: List[str], target_sfreq: float) -> Tuple[np.ndarray, List[str], float]:
    """Load (C,T) data with channel order = pick_channels. Returns (data, picked_names, sfreq)."""
    if mne is not None and path.suffix.lower() in ('.edf', '.bdf', '.fif'):
        raw = mne.io.read_raw(path.as_posix(), preload=True, verbose='ERROR')
        # Rename channels to be safe (strip spaces)
        raw.rename_channels({n: n.strip() for n in raw.ch_names})
        picks = []
        for ch in pick_channels:
            if ch in raw.ch_names:
                picks.append(ch)
            else:
                # try case-insensitive match
                match = [n for n in raw.ch_names if n.lower() == ch.lower()]
                if match:
                    picks.append(match[0])
                else:
                    raise ValueError(f"Channel '{ch}' not found in {path.name}. Available: {raw.ch_names}")
        raw.pick(picks)
        if abs(raw.info['sfreq'] - target_sfreq) > 1e-3:
            raw.resample(target_sfreq)
        data = raw.get_data()  # (C,T)
        return data.astype(np.float32), [ch.strip() for ch in picks], float(raw.info['sfreq'])

    # Fallback: pyedflib for EDF
    if pyedflib is not None and path.suffix.lower() == '.edf':
        f = pyedflib.EdfReader(path.as_posix())
        names = [f.getLabel(i).strip() for i in range(f.signals_in_file)]
        # map picks
        idxs = []
        for ch in pick_channels:
            if ch in names:
                idxs.append(names.index(ch))
            elif ch.upper() in names:
                idxs.append(names.index(ch.upper()))
            else:
                raise ValueError(f"Channel '{ch}' not found in {path.name}. Available: {names}")
        # Assume same sample rate for selected channels
        sfreq = f.getSampleFrequency(idxs[0])
        data = np.vstack([f.readSignal(i) for i in idxs]).astype(np.float32)
        f._close(); del f
        # Resample if needed
        if abs(sfreq - target_sfreq) > 1e-3:
            import scipy.signal as sps
            g = np.gcd(int(sfreq), int(target_sfreq))
            up = int(target_sfreq // g)
            down = int(sfreq // g)
            data = sps.resample_poly(data, up, down, axis=1)
            sfreq = target_sfreq
        return data, [pick_channels[i] for i in range(len(pick_channels))], float(sfreq)

    raise RuntimeError("No suitable reader for this file. Install mne or pyedflib, or use EDF/BDF/FIF.")


# -----------------------------
# Filtering & windowing
# -----------------------------

def bandpass(data: np.ndarray, sfreq: float, lo: Optional[float], hi: Optional[float]) -> np.ndarray:
    if lo is None or hi is None:
        return data
    import scipy.signal as sps
    ny = 0.5 * sfreq
    lo_n = max(1e-3, lo / ny)
    hi_n = min(0.999, hi / ny)
    b, a = sps.butter(4, [lo_n, hi_n], btype='band')
    return sps.filtfilt(b, a, data, axis=1).astype(np.float32)


def intervals_to_mask(T: int, sfreq: float, intervals: List[Tuple[float, float]]) -> np.ndarray:
    mask = np.zeros(T, dtype=np.float32)
    for s, e in intervals:
        i0 = max(0, int(round(s * sfreq)))
        i1 = min(T, int(round(e * sfreq)))
        if i1 > i0:
            mask[i0:i1] = 1.0
    return mask


def make_windows(X: np.ndarray, mask_global: np.ndarray, masks_per_ch: Optional[np.ndarray], sfreq: float, win_s: float, stride_s: float) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    C, T = X.shape
    W = int(round(win_s * sfreq))
    S = int(round(stride_s * sfreq))
    if W <= 0 or S <= 0:
        raise ValueError("Window and stride must be > 0")
    starts = list(range(0, max(1, T - W + 1), S))
    Xw = np.zeros((len(starts), C, W), dtype=np.float32)
    Yt = np.zeros((len(starts), W), dtype=np.float32)
    Yct = np.zeros((len(starts), C, W), dtype=np.float32) if masks_per_ch is not None else None
    for k, s0 in enumerate(starts):
        s1 = s0 + W
        Xw[k] = X[:, s0:s1]
        Yt[k] = mask_global[s0:s1]
        if Yct is not None:
            Yct[k] = masks_per_ch[:, s0:s1]
    return Xw, Yt, Yct


# -----------------------------
# Main build
# -----------------------------

def build_from_inputs(raw_paths: List[Path], labels_paths: Dict[str, Path], channels: List[str], sfreq: float, win_s: float, stride_s: float, band: Optional[Tuple[float, float]], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    X_all = []
    Yt_all = []
    Yct_all = []

    for rp in raw_paths:
        print(f"Processing {rp.name} ...")
        # match a label file: exact stem match if available, else use single provided path
        if rp.stem in labels_paths:
            lp = labels_paths[rp.stem]
        elif len(labels_paths) == 1:
            lp = list(labels_paths.values())[0]
        else:
            raise ValueError(f"No matching JSON for {rp.name}. Provide a single JSON or a json with same stem.")

        labels_json = _load_json(lp)
        glob_iv, per_ch = parse_label_json(labels_json)

        # load raw
        X, picked, actual_sfreq = load_raw_signals(rp, channels, target_sfreq=sfreq)
        if band is not None:
            X = bandpass(X, sfreq=sfreq, lo=band[0], hi=band[1])

        # masks
        mask_global = intervals_to_mask(X.shape[1], sfreq, glob_iv)
        masks_per_ch = None
        if per_ch:
            masks_per_ch = np.zeros((len(channels), X.shape[1]), dtype=np.float32)
            name_to_idx = {ch: i for i, ch in enumerate(channels)}
            for ch_name, ivs in per_ch.items():
                if ch_name not in name_to_idx:
                    # allow case-insensitive match
                    matches = [k for k in name_to_idx.keys() if k.lower() == ch_name.lower()]
                    if not matches:
                        print(f"[WARN] Channel '{ch_name}' in labels not in pick list; skipping its intervals")
                        continue
                    ch_idx = name_to_idx[matches[0]]
                else:
                    ch_idx = name_to_idx[ch_name]
                masks_per_ch[ch_idx] = intervals_to_mask(X.shape[1], sfreq, ivs)

        # windows
        Xw, Yt, Yct = make_windows(X, mask_global, masks_per_ch, sfreq, win_s, stride_s)
        X_all.append(Xw)
        Yt_all.append(Yt)
        if Yct is not None:
            Yct_all.append(Yct)

    X_out = np.concatenate(X_all, axis=0)
    Yt_out = np.concatenate(Yt_all, axis=0)
    np.save(out_dir / 'windows.npy', X_out)
    np.save(out_dir / 'labels_framewise.npy', Yt_out)
    print(f"Saved: {out_dir/'windows.npy'} shape={X_out.shape}")
    print(f"Saved: {out_dir/'labels_framewise.npy'} shape={Yt_out.shape}")

    if Yct_all:
        Yct_out = np.concatenate(Yct_all, axis=0)
        np.save(out_dir / 'labels_per_channel.npy', Yct_out)
        print(f"Saved: {out_dir/'labels_per_channel.npy'} shape={Yct_out.shape}")
    else:
        print("Per-channel labels not detected in JSON -> skipped labels_per_channel.npy")


# -----------------------------
# CLI
# -----------------------------

def discover_raw_and_labels(raw: Optional[str], raw_dir: Optional[str], labels: Optional[str]) -> Tuple[List[Path], Dict[str, Path]]:
    # raw files
    raw_paths: List[Path] = []
    if raw:
        p = Path(raw)
        if not p.exists():
            raise FileNotFoundError(p)
        raw_paths = [p]
    else:
        rd = Path(raw_dir or '.')
        if not rd.exists():
            raise FileNotFoundError(rd)
        for ext in ('*.edf', '*.bdf', '*.fif'):
            raw_paths += list(rd.glob(ext))
        if not raw_paths:
            raise RuntimeError("No raw files found in raw_dir (looking for .edf/.bdf/.fif)")

    # label files: dict by stem
    labels_map: Dict[str, Path] = {}
    if labels:
        lp = Path(labels)
        if lp.is_file():
            labels_map[lp.stem] = lp
        elif lp.is_dir():
            for j in lp.glob('*.json'):
                labels_map[j.stem] = j
        else:
            raise FileNotFoundError(lp)
    else:
        # try next to raw file with same stem
        for rp in raw_paths:
            j = rp.with_suffix('.json')
            if j.exists():
                labels_map[rp.stem] = j
        if not labels_map:
            raise RuntimeError("No labels provided and no neighboring .json found.")
    return raw_paths, labels_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', type=str, default=None, help='Single raw file (edf/bdf/fif)')
    ap.add_argument('--raw_dir', type=str, default=None, help='Folder containing raw files')
    ap.add_argument('--labels', type=str, default=None, help='JSON file or folder of JSONs (matching stems)')
    ap.add_argument('--channels', type=str, required=True, help='Comma-separated channel list in desired order')
    ap.add_argument('--sfreq', type=float, default=200, help='Target sampling frequency')
    ap.add_argument('--win', type=float, default=2.0, help='Window length (seconds)')
    ap.add_argument('--stride', type=float, default=1.0, help='Stride (seconds)')
    ap.add_argument('--band', nargs=2, type=float, default=None, metavar=('LO','HI'), help='Bandpass (Hz), e.g., --band 10 16')
    ap.add_argument('--out_dir', type=str, default='./data', help='Output folder')
    args = ap.parse_args()

    channels = [s.strip() for s in args.channels.split(',') if s.strip()]
    raw_paths, labels_map = discover_raw_and_labels(args.raw, args.raw_dir, args.labels)

    build_from_inputs(
        raw_paths=raw_paths,
        labels_paths=labels_map,
        channels=channels,
        sfreq=float(args.sfreq),
        win_s=float(args.win),
        stride_s=float(args.stride),
        band=(args.band[0], args.band[1]) if args.band is not None else None,
        out_dir=Path(args.out_dir),
    )


if __name__ == '__main__':
    main()
