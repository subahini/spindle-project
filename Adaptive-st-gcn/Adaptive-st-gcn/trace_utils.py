from typing import Optional

import torch
import torch.nn.functional as F

"""
Trace helpers for per-channel onset extraction and propagation order.
"""

@torch.no_grad()
def extract_onsets_from_logits_per_channel(
    logits_per_ch: torch.Tensor,
    thr: float = 0.5,
    min_consec: int = 5,
) -> torch.Tensor:
    """Return onset indices per channel for each batch.
    Args:
      logits_per_ch: (B, C, T)
      thr: probability threshold on sigmoid(logits)
      min_consec: require this many consecutive frames above thr to accept onset
    Returns:
      onsets: LongTensor (B, C) with onset index in [0, T-1] or -1 if none.
    """
    probs = torch.sigmoid(logits_per_ch)  # (B, C, T)
    B, C, T = probs.shape
    onsets = torch.full((B, C), -1, dtype=torch.long, device=probs.device)
    above = probs >= thr
    starts = (above[..., 1:] & ~above[..., :-1])
    starts = F.pad(starts.float(), (1, 0)) > 0
    for b in range(B):
        for c in range(C):
            idx = -1
            t = 0
            while t < T:
                while t < T and not starts[b, c, t]:
                    t += 1
                if t >= T:
                    break
                k = t
                while k < T and above[b, c, k]:
                    k += 1
                if k - t >= min_consec:
                    idx = t
                    break
                t = k + 1
            onsets[b, c] = idx
    return onsets


@torch.no_grad()
def propagation_order_from_onsets(onsets: torch.Tensor) -> torch.Tensor:
    """Sort channels by onset; ties keep original order; -1 go to end.
    Args:
      onsets: (B, C) LongTensor of indices or -1
    Returns:
      order: (B, C) LongTensor of channel indices sorted by onset
    """
    B, C = onsets.shape
    large = onsets[onsets >= 0].max().item() + 10_000 if (onsets >= 0).any() else 10_000
    keyed = onsets.clone()
    keyed[onsets < 0] = large
    order = torch.argsort(keyed, dim=1, stable=True)
    return order
