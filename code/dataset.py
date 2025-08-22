# dataset.py
"""
Datasets & split builder for sample-level training.
- Each item: x:[C,T], y:[T] (binary vector per timepoint)
- Sources:
    edf        -> EDF + JSON labels -> windows with CAR; real spindle labels
    npy        -> load X:[N,C,T], y:[N,T] (pre-processed data)
"""

from typing import Tuple, List, Optional, Dict, Any, Iterable
import os
import json
import re
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import mne
    _HAS_MNE = True
except Exception:
    _HAS_MNE = False


# --------------------------
#     PyTorch Dataset
# --------------------------
class WindowSampleDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 3, f"X must be [N,C,T], got {X.shape}"
        assert y.ndim == 2 and y.shape[0] == X.shape[0] and y.shape[1] == X.shape[2], f"y must be [N,T], got {y.shape}"
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])


# --------------------------
#     JSON label helpers
# --------------------------
def _load_json_labels(json_path: str) -> Any:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Label file not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _top_keys(x: Any, maxn: int = 40) -> List[str]:
    return list(x.keys())[:maxn] if isinstance(x, dict) else []

def _iter_event_lists(x: Any) -> Iterable[List[Dict[str, Any]]]:
    """
    Recursively yield lists of dict-like 'events'.
    Matches common container keys and ANY key that contains 'spind'
    (e.g., 'detected_spindles', 'spindleEvents', ...).
    Handles when the value is a list-of-dicts OR a dict with a list nested inside.
    """
    if isinstance(x, list) and x and isinstance(x[0], dict):
        yield x
    if isinstance(x, dict):
        for k, v in x.items():
            low = k.lower()
            looks_like_spindle_list = ("spind" in low)
            is_common_container = low in (
                "spindles", "annotations", "events", "detections", "labels", "items", "data", "results", "payload"
            )
            if isinstance(v, list) and v and isinstance(v[0], dict) and (looks_like_spindle_list or is_common_container):
                yield v
            elif isinstance(v, dict):
                # if the spindle key maps to a dict, dive into it to find lists
                yield from _iter_event_lists(v)
            elif isinstance(v, list):
                yield from _iter_event_lists(v)

def _try_per_sample_mask(labels_data: Any, total_samples: int, sfreq: float) -> Optional[np.ndarray]:
    """
    Try to read a direct per-sample (or per-second) 0/1 mask from the JSON.
    """
    candidate_keys = ["labels", "y", "mask", "spindle_mask", "binary_labels", "per_sample", "per_timepoint"]
    nodes = [labels_data]
    if isinstance(labels_data, dict):
        for cont_key in ["data", "payload", "result", "meta", "results"]:
            if cont_key in labels_data and isinstance(labels_data[cont_key], dict):
                nodes.append(labels_data[cont_key])

    for node in nodes:
        if not isinstance(node, dict):
            continue
        for k in candidate_keys:
            if k in node and isinstance(node[k], list):
                arr = np.asarray(node[k], dtype=np.float32).reshape(-1)
                if arr.size == total_samples:
                    print(f"[labels] Using per-sample mask from '{k}' (matched {arr.size} samples).")
                    return arr
                sec_len = int(round(total_samples / float(sfreq)))
                if sec_len > 0 and arr.size == sec_len:
                    print(f"[labels] Using per-second mask from '{k}' and upsampling x{int(round(sfreq))}.")
                    return np.repeat(arr, int(round(sfreq)))[:total_samples]
    return None

def _to_seconds(val: float, T_all_sec: float, sfreq: float) -> float:
    """
    Normalize to seconds using heuristics:
      - Large but <= T*sfreq -> samples -> /sfreq
      - Very large (>100k)   -> ms -> /1000
      - else assume seconds
    """
    v = float(val)
    if v > T_all_sec * 1.5 and v <= T_all_sec * sfreq * 1.5:
        return v / sfreq
    if v > 100_000.0:
        return v / 1000.0
    return v

def _extract_onset_offset(d: Dict[str, Any], T_all_sec: float, sfreq: float) -> Optional[Tuple[float, float]]:
    """
    Extract onset/offset seconds from a single event dict using common field names.
    Supports (onset, offset), (start_time, end_time), (start, end), indices, or center+duration.
    """
    # direct pairs
    for a, b in [
        ("onset", "offset"),
        ("start_time", "end_time"),
        ("start", "end"),
        ("begin", "end"),
        ("t_start", "t_end"),
        ("start_sec", "end_sec"),
        ("start_idx", "end_idx"),
        ("startSample", "endSample"),
        ("start_ms", "end_ms"),
    ]:
        if a in d and b in d:
            s = _to_seconds(d[a], T_all_sec, sfreq)
            e = _to_seconds(d[b], T_all_sec, sfreq)
            return (s, e) if e > s else None

    # center + duration/half_duration
    center_keys = ["center", "mid", "peak_time"]
    dur_keys = ["duration", "dur", "length", "half_duration", "duration_ms"]
    for ck in center_keys:
        if ck in d:
            c = _to_seconds(d[ck], T_all_sec, sfreq)
            for dk in dur_keys:
                if dk in d:
                    dur = float(d[dk])
                    if "ms" in dk.lower():
                        dur = dur / 1000.0
                    if "half" in dk.lower():
                        s, e = c - dur, c + dur
                    else:
                        s, e = c - dur / 2.0, c + dur / 2.0
                    return (s, e) if e > s else None
    return None

# ---- channel name matching (robust: case/format-insensitive) ----
def _norm_ch(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())

def _any_channel_matches(ev_ch_names: List[str], raw_ch_names: List[str]) -> bool:
    if not ev_ch_names:
        return True
    raw_norm = [_norm_ch(rn) for rn in raw_ch_names]
    for ec in ev_ch_names:
        n = _norm_ch(ec)
        # match by equality or substring either way to survive "C4" vs "C4-REF" vs "EEG C4-Ref"
        for rn in raw_norm:
            if n == rn or n in rn or rn in n:
                return True
    return False

def _create_labels_from_json_any_schema(
    labels_data: Any,
    total_samples: int,
    sfreq: float,
    channel_names: List[str],
) -> np.ndarray:
    """
    Build binary label vector [total_samples] from various JSON schemas.
    - First tries a per-sample/per-second mask.
    - Otherwise searches for event lists under many keys (including any that contain 'spind').
    - Accepts events whose 'type'/'label'/'name' contains 'spind' OR events with 'start'/'end' only.
    - Converts time units (sec/ms/samples) to seconds.
    - Keeps only events for kept channels if JSON supplies 'channel' or 'channel_names'.
    """
    T_all_sec = float(total_samples) / float(sfreq)
    print(f"[labels] Top-level keys: {_top_keys(labels_data)}")

    # 1) direct per-sample/per-second mask
    mask = _try_per_sample_mask(labels_data, total_samples, sfreq)
    if mask is not None:
        return mask.astype(np.float32)

    # 2) candidate event lists
    candidate_lists: List[List[Dict[str, Any]]] = list(_iter_event_lists(labels_data))
    if not candidate_lists:
        print("[labels] No event lists discovered in JSON.")
        return np.zeros(total_samples, dtype=np.float32)

    def parse_list(ev_list: List[Dict[str, Any]]) -> List[Tuple[float, float, Optional[List[str]]]]:
        out = []
        # show a couple of examples
        for i, ev in enumerate(ev_list[:3]):
            if isinstance(ev, dict):
                print(f"[labels] example event keys {i}: {list(ev.keys())[:10]}")
        for ev in ev_list:
            if not isinstance(ev, dict):
                continue
            # accept if start/end are present OR if the type mentions 'spind'
            ev_type = str(ev.get("type", ev.get("label", ev.get("name", "")))).lower()
            has_start_end = ("start" in ev and "end" in ev) or ("onset" in ev and "offset" in ev)
            if not has_start_end and ev_type and ("spind" not in ev_type):
                continue

            oo = _extract_onset_offset(ev, T_all_sec, sfreq)
            if oo is None:
                continue
            s_sec, e_sec = oo

            # channels can be a single string, list of strings, or indices. Prefer string names.
            ch_field = ev.get("channel", ev.get("chan", ev.get("ch", None)))
            ch_names = ev.get("channel_names", None)

            ev_ch_names: List[str] = []
            if isinstance(ch_names, list) and ch_names and isinstance(ch_names[0], str):
                ev_ch_names = ch_names
            elif isinstance(ch_field, str):
                ev_ch_names = [ch_field]
            elif isinstance(ch_field, list) and ch_field and isinstance(ch_field[0], str):
                ev_ch_names = ch_field
            # if only numeric indices were provided, we skip mapping (can't trust index order); keep empty => no filter

            out.append((s_sec, e_sec, ev_ch_names if ev_ch_names else None))
        return out

    best: List[Tuple[float, float, Optional[List[str]]]] = []
    for L in candidate_lists:
        parsed = parse_list(L)
        if len(parsed) > len(best):
            best = parsed

    print(f"[labels] Candidate event lists: {len(candidate_lists)}  |  Best parsed events: {len(best)}")
    y = np.zeros(total_samples, dtype=np.float32)
    if not best:
        return y

    placed = 0
    for (s_sec, e_sec, ev_ch_names) in best:
        # channel filtering: if event lists channel names, require at least one to match kept channels
        if ev_ch_names and not _any_channel_matches(ev_ch_names, channel_names):
            continue
        start_idx = int(max(0, np.floor(s_sec * sfreq)))
        end_idx   = int(min(total_samples, np.ceil(e_sec * sfreq)))
        if end_idx > start_idx:
            y[start_idx:end_idx] = 1.0
            placed += 1

    print(f"[labels] Placed {placed}/{len(best)} events after channel filtering "
          f"(kept-channels: {channel_names[:6]}{'...' if len(channel_names)>6 else ''})")
    return y


# --------------------------
#     EDF -> windows
# --------------------------
def _edf_to_windows_with_json_labels(
    edf_path: str,
    json_path: str,
    sfreq: float,
    window_sec: float,
    step_sec: float,
    reference: str = "car",
    montage: str = "standard_1020",
    bandpass: Optional[Tuple[float, float]] = (0.3, 35.0),
    notch_hz: Optional[List[float]] = None,
    # channel selection
    n_channels: Optional[int] = None,
    keep_channels: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if not _HAS_MNE:
        raise RuntimeError("Install mne to use source='edf'")

    print(f"Loading EDF: {edf_path}")
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    raw.set_montage(montage, on_missing="ignore")

    # EEG only
    picks = mne.pick_types(raw.info, eeg=True, eog=False, emg=False, ecg=False)
    if len(picks) == 0:
        raise RuntimeError(f"No EEG channels found in {edf_path}")
    raw.pick(picks)
    print(f"Initial EEG picks ({len(raw.ch_names)}): {raw.ch_names}")

    # Reference & filters
    if str(reference).lower() == "car":
        raw.set_eeg_reference("average")
    if notch_hz:
        raw.notch_filter(notch_hz)
    if bandpass:
        raw.filter(bandpass[0], bandpass[1])

    # Optional resample to cfg sfreq
    try:
        if sfreq and float(raw.info["sfreq"]) != float(sfreq):
            print(f"Resampling from {raw.info['sfreq']} Hz -> {sfreq} Hz")
            raw.resample(float(sfreq))
    except Exception:
        pass

    # Channel selection (apply BEFORE get_data)
    if keep_channels and len(keep_channels) > 0:
        sel = mne.pick_channels(raw.ch_names, include=keep_channels, ordered=True)
        missing = sorted(set(keep_channels) - set([raw.ch_names[i] for i in sel if i < len(raw.ch_names)]))
        if missing:
            print(f"[warn] Missing channels in EDF (ignored): {missing}")
        if len(sel) == 0:
            raise RuntimeError("None of the requested keep_channels found in EDF.")
        raw.pick(sel)
        print(f"Using keep_channels (ordered): {raw.ch_names}")
    elif n_channels is not None and len(raw.ch_names) > int(n_channels):
        raw.pick(raw.ch_names[: int(n_channels)])
        print(f"Using first {n_channels} EEG channels: {raw.ch_names}")

    # Raw to numpy
    data = raw.get_data()  # [C, T_all]
    C, T_all = data.shape

    # Labels
    print(f"Loading labels: {json_path}")
    labels_data = _load_json_labels(json_path)
    continuous_labels = _create_labels_from_json_any_schema(
        labels_data=labels_data,
        total_samples=T_all,
        sfreq=float(raw.info["sfreq"]),
        channel_names=raw.ch_names,
    )

    print(f"Label statistics: {continuous_labels.sum():.0f} positive samples "
          f"({continuous_labels.mean() * 100:.2f}%)")

    # Windowing
    win_T = int(round(window_sec * float(raw.info["sfreq"])))
    step_T = int(round(step_sec * float(raw.info["sfreq"])))
    X_list, Y_list = [], []

    for start in range(0, max(0, T_all - win_T + 1), step_T):
        end = start + win_T
        seg = data[:, start:end]            # [C, T]
        yseg = continuous_labels[start:end] # [T]
        if seg.shape[1] == win_T:
            X_list.append(seg.astype(np.float32))
            Y_list.append(yseg.astype(np.float32))

    if not X_list:
        raise RuntimeError(f"No valid windows created from {edf_path}")

    X = np.stack(X_list, axis=0)  # [N, C, T]
    y = np.stack(Y_list, axis=0)  # [N, T]

    print(f"Created {len(X_list)} windows of shape {X.shape[1:]} from {edf_path}")
    pos_wins = (y.sum(axis=1) > 0).sum()
    print(f"Windows with spindles: {pos_wins}/{len(y)} ({(pos_wins / len(y)) * 100:.1f}%)")

    return X, y


# --------------------------
#     Build splits
# --------------------------
def build_splits(cfg) -> Tuple[Dataset, Dataset, Dataset]:
    rng = np.random.default_rng(cfg.data.split_seed)

    if cfg.data.source == "npy":
        print("Loading pre-processed NPY data...")
        X = np.load(cfg.data.npy.x_path)  # [N,C,T]
        y = np.load(cfg.data.npy.y_path)  # [N,T]
        print(f"Loaded NPY data: X={X.shape}, y={y.shape}")

    elif cfg.data.source == "edf":
        edf_dir = cfg.data.edf.dir
        json_path = cfg.data.edf.labels_json

        if not os.path.exists(edf_dir):
            raise RuntimeError(f"EDF directory not found: {edf_dir}")
        if not os.path.exists(json_path):
            raise RuntimeError(f"Labels JSON not found: {json_path}")

        edf_files = [f for f in os.listdir(edf_dir) if f.lower().endswith(".edf")]
        if not edf_files:
            raise RuntimeError(f"No EDF files found in {edf_dir}")
        print(f"Found {len(edf_files)} EDF file(s)")

        X_list, y_list = [], []
        for edf_file in edf_files[:1]:  # process first file for now (extend if needed)
            edf_path = os.path.join(edf_dir, edf_file)
            X_file, y_file = _edf_to_windows_with_json_labels(
                edf_path=edf_path,
                json_path=json_path,
                sfreq=float(cfg.data.sfreq),
                window_sec=float(cfg.data.window_sec),
                step_sec=float(cfg.data.step_sec),
                reference=getattr(cfg.data.edf, "reference", "car"),
                montage=getattr(cfg.data.edf, "montage", "standard_1020"),
                bandpass=tuple(cfg.data.edf.bandpass) if hasattr(cfg.data.edf, "bandpass") and cfg.data.edf.bandpass else (0.3, 35.0),
                notch_hz=list(cfg.data.edf.notch_hz) if hasattr(cfg.data.edf, "notch_hz") and cfg.data.edf.notch_hz else [50, 60],
                n_channels=int(getattr(cfg.data, "n_channels", 16)),
                keep_channels=list(getattr(cfg.data.edf, "keep_channels", [])) if hasattr(cfg.data, "edf") and hasattr(cfg.data.edf, "keep_channels") else None,
            )
            X_list.append(X_file)
            y_list.append(y_file)

        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)

    else:
        raise ValueError(f"Unsupported data source: {cfg.data.source}. Use 'edf' or 'npy'.")

    print(f"Final dataset: X={X.shape}, y={y.shape}")
    print(f"Total positive samples: {y.sum():.0f} ({y.mean() * 100:.2f}%)")

    # ---------- Stratified split by window-level positivity ----------
    has_pos = (y.sum(axis=1) > 0)
    pos_idx = np.where(has_pos)[0]
    neg_idx = np.where(~has_pos)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    def split_idxs(idxs: np.ndarray, fracs=(0.70, 0.15, 0.15)):
        n = len(idxs)
        n_tr = int(fracs[0] * n)
        n_va = int(fracs[1] * n)
        return idxs[:n_tr], idxs[n_tr:n_tr + n_va], idxs[n_tr + n_va:]

    p_tr, p_va, p_te = split_idxs(pos_idx)
    n_tr, n_va, n_te = split_idxs(neg_idx)

    tr_idx = np.concatenate([p_tr, n_tr]); rng.shuffle(tr_idx)
    va_idx = np.concatenate([p_va, n_va]); rng.shuffle(va_idx)
    te_idx = np.concatenate([p_te, n_te]); rng.shuffle(te_idx)

    # Fallback: if a split has 0 positives but positives exist overall, move one positive
    def ensure_pos(split_idx: np.ndarray):
        if (y[split_idx].sum() == 0) and (len(pos_idx) > 0):
            split_idx[0] = pos_idx[0]
        return split_idx

    va_idx = ensure_pos(va_idx)
    te_idx = ensure_pos(te_idx)

    train = WindowSampleDataset(X[tr_idx], y[tr_idx])
    val   = WindowSampleDataset(X[va_idx], y[va_idx])
    test  = WindowSampleDataset(X[te_idx], y[te_idx])

    print(f"Splits: train={len(train)}, val={len(val)}, test={len(test)}")
    print(f"Pos windows -> train: {(train.y.sum(axis=1)>0).sum()}, "
          f"val: {(val.y.sum(axis=1)>0).sum()}, "
          f"test: {(test.y.sum(axis=1)>0).sum()}")

    return train, val, test
