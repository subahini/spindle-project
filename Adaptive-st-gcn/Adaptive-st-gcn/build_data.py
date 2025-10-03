#!/usr/bin/env python3
"""
Builder for ST-GCN data (TAILORED to your JSON screenshots)
-----------------------------------------------------------
Reads RAW EEG (EDF/BDF/FIF) + your JSON format and writes:
  - data/windows.npy              (N, C, T)  float32
  - data/labels_framewise.npy    (N, T)     float32 (0/1)
  - data/labels_per_channel.npy  (N, C, T)  float32 (0/1)  [if channels present]

Your JSON format (supported):
  {
    "channel_names": ["F3","F4","C3","C4",...],   # optional at top level
    "detected_spindles": [
      { "start": 1669.572, "end": 1670.604, "channels": ["C4-REF"] },
      { "start": 1676.488, "end": 1677.004, "channels": [3] },        # 3 -> channel_names[3]
      { "start": 1691.312, "end": 1691.960 }                           # no per-channel info
    ],
    ... other fields ignored ...
  }

The 'channels' field per spindle may also be a dict, e.g.:
  { "channels": { "channel_names": ["C3","C4"], "0": 1, "1": 0 } }  # truthy keys -> included
or { "channels": { "C4-REF": 1 } }

Usage example
-------------
python builder.py \
  --raw_dir ./raw \
  --labels ./labels \
  --channels F3,F4,C3,C4,O1,O2,F7,F8,T3,T4,P3,P4,Fz,Cz,Pz,Oz \
  --sfreq 200 --win 2.0 --stride 1.0 --band 10 16 \
  --out_dir ./data

Dependencies
------------
  pip install mne pyedflib scipy numpy

"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

# Optional: prefer MNE, fallback to pyedflib for EDF
try:
    import mne  # type: ignore
except Exception:
    mne = None
try:
    import pyedflib  # type: ignore
except Exception:
    pyedflib = None

# -----------------------------
# JSON parsing for your schema
# -----------------------------

def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)


def _canon(name: str) -> str:
    n = name.strip().lower()
    # strip common refs/suffixes: -ref, _ref, -avg, etc.
    for sep in ('-', '_'):
        if sep in n:
            n = n.split(sep)[0]
    return n.upper()


def parse_label_json_viewer(j: Dict[str, Any]) -> Tuple[List[Tuple[float,float]], Dict[str, List[Tuple[float,float]]]]:
    """Parse your viewer-like JSON with detected_spindles.
    Returns:
      global_iv: list of (start,end) seconds
      per_ch: dict: CANONICAL_CHANNEL_NAME -> list of (start,end)
    """
    global_iv: List[Tuple[float,float]] = []
    per_ch: Dict[str, List[Tuple[float,float]]] = {}

    # optional top-level channel names for index mapping
    top_names = None
    if isinstance(j.get('channel_names'), list):
        top_names = [str(x) for x in j['channel_names']]

    ds = j.get('detected_spindles')
    if not isinstance(ds, list):
        raise ValueError("JSON missing 'detected_spindles' list")

    def add_per_ch(ch_any, s: float, e: float):
        if ch_any is None:
            return
        if isinstance(ch_any, (list, tuple)):
            for c in ch_any:
                add_per_ch(c, s, e)
            return
        # dict case: keys can be indexes or names; values truthy -> included
        if isinstance(ch_any, dict):
            local_names = ch_any.get('channel_names') if isinstance(ch_any.get('channel_names'), list) else None
            for k, v in ch_any.items():
                if k == 'channel_names':
                    continue
                if not v:
                    continue
                try:
                    idx = int(k)
                    name = None
                    if local_names and 0 <= idx < len(local_names):
                        name = local_names[idx]
                    elif top_names and 0 <= idx < len(top_names):
                        name = top_names[idx]
                    key = _canon(name if name is not None else str(idx))
                except Exception:
                    key = _canon(str(k))
                per_ch.setdefault(key, []).append((float(s), float(e)))
            return
        # scalar name or index
        if isinstance(ch_any, (int, np.integer)):
            if top_names and 0 <= int(ch_any) < len(top_names):
                name = top_names[int(ch_any)]
                key = _canon(name)
            else:
                key = _canon(str(ch_any))
        else:
            key = _canon(str(ch_any))
        per_ch.setdefault(key, []).append((float(s), float(e)))

    for it in ds:
        if not isinstance(it, dict):
            continue
        s = float(it.get('start', it.get('onset', 0.0)))
        e = float(it.get('end', it.get('offset', s)))
        global_iv.append((s, e))
        ch_field = it.get('channels', None)
        if ch_field is not None:
            add_per_ch(ch_field, s, e)

    return global_iv, per_ch


# -----------------------------
# RAW loading
# -----------------------------

def load_raw_signals(path: Path, pick_channels: List[str], target_sfreq: float) -> Tuple[np.ndarray, List[str], float]:
    """Load (C,T) with channel order = pick_channels. Returns data, picked_names, sfreq."""
    if mne is not None and path.suffix.lower() in ('.edf', '.bdf', '.fif'):
        raw = mne.io.read_raw(path.as_posix(), preload=True, verbose='ERROR')
        raw.rename_channels({n: n.strip() for n in raw.ch_names})
        picks = []
        for ch in pick_channels:
            if ch in raw.ch_names:
                picks.append(ch)
            else:
                match = [n for n in raw.ch_names if n.lower() == ch.lower()]
                if match:
                    picks.append(match[0])
                else:
                    raise ValueError(f"Channel '{ch}' not found in {path.name}. Available: {raw.ch_names}")
        raw.pick(picks)
        if abs(raw.info['sfreq'] - target_sfreq) > 1e-3:
            raw.resample(target_sfreq)
        data = raw.get_data()
        return data.astype(np.float32), [ch.strip() for ch in picks], float(raw.info['sfreq'])

    if pyedflib is not None and path.suffix.lower() == '.edf':
        f = pyedflib.EdfReader(path.as_posix())
        names = [f.getLabel(i).strip() for i in range(f.signals_in_file)]
        idxs = []
        for ch in pick_channels:
            if ch in names:
                idxs.append(names.index(ch))
            elif ch.upper() in names:
                idxs.append(names.index(ch.upper()))
            else:
                raise ValueError(f"Channel '{ch}' not found in {path.name}. Available: {names}")
        sfreq = f.getSampleFrequency(idxs[0])
        data = np.vstack([f.readSignal(i) for i in idxs]).astype(np.float32)
        f._close(); del f
        if abs(sfreq - target_sfreq) > 1e-3:
            try:
                import scipy.signal as sps
            except Exception:
                raise RuntimeError("scipy is required for resampling when using pyedflib. pip install scipy")
            # rational resample
            g = np.gcd(int(sfreq), int(target_sfreq))
            up = int(target_sfreq // g)
            down = int(sfreq // g)
            data = sps.resample_poly(data, up, down, axis=1)
            sfreq = target_sfreq
        return data, pick_channels, float(sfreq)

    raise RuntimeError("No suitable reader for this file. Install mne or pyedflib, or use EDF/BDF/FIF.")


# -----------------------------
# Filtering & windowing
# -----------------------------

def bandpass(data: np.ndarray, sfreq: float, lo: Optional[float], hi: Optional[float]) -> np.ndarray:
    if lo is None or hi is None:
        return data
    try:
        import scipy.signal as sps
    except Exception:
        raise RuntimeError("scipy is required for bandpass filtering. pip install scipy")
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
# Build per recording
# -----------------------------

def build_from_inputs(raw_paths: List[Path], labels_paths: Dict[str, Path], channels: List[str], sfreq: float, win_s: float, stride_s: float, band: Optional[Tuple[float, float]], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    X_all = []
    Yt_all = []
    Yct_all = []

    for rp in raw_paths:
        print(f"Processing {rp.name} ...")
        # Match JSON by stem if a folder of JSONs is provided; else use the single JSON file
        if rp.stem in labels_paths:
            lp = labels_paths[rp.stem]
        elif len(labels_paths) == 1:
            lp = list(labels_paths.values())[0]
        else:
            raise ValueError(f"No matching JSON for {rp.name}. Provide a single JSON or matching stems.")

        j = _load_json(lp)
        glob_iv, per_ch = parse_label_json_viewer(j)

        # load raw
        X, picked, _ = load_raw_signals(rp, channels, target_sfreq=sfreq)
        if band is not None:
            X = bandpass(X, sfreq=sfreq, lo=band[0], hi=band[1])

        # global mask
        mask_global = intervals_to_mask(X.shape[1], sfreq, glob_iv)

        # per-channel masks
        masks_per_ch = None
        if per_ch:
            def canon(n: str) -> str:
                n = n.strip().lower()
                for sep in ('-', '_'):
                    if sep in n:
                        n = n.split(sep)[0]
                return n.upper()
            name_to_idx = {canon(ch): i for i, ch in enumerate(channels)}
            masks_per_ch = np.zeros((len(channels), X.shape[1]), dtype=np.float32)
            for ch_name, ivs in per_ch.items():
                key = canon(ch_name)
                if key not in name_to_idx:
                    print(f"[WARN] Label channel '{ch_name}' not in --channels list; skipping")
                    continue
                ch_idx = name_to_idx[key]
                masks_per_ch[ch_idx] = intervals_to_mask(X.shape[1], sfreq, ivs)

        # windowing
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
        print("Per-channel labels not present -> skipped labels_per_channel.npy")


# -----------------------------
# CLI helpers
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

    # labels (file or folder). If folder, map by stem
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
        # try next to each raw file: same stem .json
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
    ap.add_argument('--channels', type=str, required=True, help='Comma-separated channel list in desired order (e.g., C3,C4,Cz,...)')
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
