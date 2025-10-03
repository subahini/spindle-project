"""
losses.py
Imbalance-aware loss functions for sample-level spindle detection.
All losses expect LOGITS as input (no sigmoid before the loss).
"""

from typing import Dict, Any
import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.15, gamma: float = 3.0,
                 label_smoothing: float = 0.05, logit_clamp: float = 8.0):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.logit_clamp = float(logit_clamp)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = torch.clamp(logits, -self.logit_clamp, self.logit_clamp)
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        p = torch.sigmoid(logits).clamp(1e-8, 1 - 1e-8)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        bce = -targets * torch.log(p) - (1 - targets) * torch.log(1 - p)
        focal_weight = alpha_t * (1 - p_t).pow(self.gamma)
        return (focal_weight * bce).mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, logit_clamp: float = 8.0):
        super().__init__()
        self.smooth = float(smooth)
        self.logit_clamp = float(logit_clamp)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = torch.clamp(logits, -self.logit_clamp, self.logit_clamp)
        probs = torch.sigmoid(logits).clamp(1e-8, 1 - 1e-8)
        p = probs.reshape(-1)
        y = targets.reshape(-1)
        inter = (p * y).sum()
        dice = (2.0 * inter + self.smooth) / (p.sum() + y.sum() + self.smooth)
        return 1.0 - dice


class BCELoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.03, entropy_weight: float = 0.01,
                 logit_clamp: float = 8.0):
        super().__init__()
        self.label_smoothing = float(label_smoothing)
        self.entropy_weight = float(entropy_weight)
        self.logit_clamp = float(logit_clamp)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = torch.clamp(logits, -self.logit_clamp, self.logit_clamp)
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
        if self.entropy_weight > 0:
            probs = torch.sigmoid(logits).clamp(1e-8, 1 - 1e-8)
            entropy = -probs * torch.log(probs) - (1 - probs) * torch.log(1 - probs)
            entropy_reg = -entropy.mean() * self.entropy_weight  # encourage entropy (reduce overconfidence)
        else:
            entropy_reg = 0.0
        return bce_loss + entropy_reg


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight: float = 3.0, adaptive: bool = True,
                 target_pos_rate: float = 0.1, label_smoothing: float = 0.03,
                 logit_clamp: float = 8.0):
        super().__init__()
        self.base_pos_weight = float(pos_weight)
        self.adaptive = bool(adaptive)
        self.target_pos_rate = float(target_pos_rate)
        self.label_smoothing = float(label_smoothing)
        self.logit_clamp = float(logit_clamp)
        self._ema = None  # smooth positive-rate estimate

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = torch.clamp(logits, -self.logit_clamp, self.logit_clamp)
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        pos_weight = self.base_pos_weight
        if self.adaptive:
            with torch.no_grad():
                cur_prob = torch.sigmoid(logits).mean().item()
                self._ema = 0.9 * (self._ema if self._ema is not None else self.target_pos_rate) + 0.1 * cur_prob
                if self._ema > self.target_pos_rate * 3:
                    pos_weight = self.base_pos_weight * 0.5
                elif self._ema < self.target_pos_rate * 0.3:
                    pos_weight = self.base_pos_weight * 1.5
                pos_weight = max(0.1, min(10.0, pos_weight))

        pw = torch.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)


# ---------- helpers ----------

def estimate_pos_weight(train_loader, max_batches: int = 10) -> float:
    """Estimate neg/pos ratio from a few batches (targets shape [B,T])."""
    pos_count = 0.0
    total_count = 0.0
    with torch.no_grad():
        for i, (_, targets) in enumerate(train_loader):
            pos_count += float(targets.sum().item())
            total_count += float(targets.numel())
            if i + 1 >= max_batches:
                break
    if pos_count <= 0:
        return 3.0
    neg_count = total_count - pos_count
    return max(1.0, min(20.0, float(neg_count / pos_count)))


def build_loss_function(loss_name: str, trainer_cfg: Dict[str, Any], train_loader=None) -> nn.Module:
    """Map config -> loss instance."""
    name = str(loss_name).lower()

    if name == "focal":
        f = trainer_cfg.get("focal", {})
        return FocalLoss(
            alpha=float(f.get("alpha", 0.25)),
            gamma=float(f.get("gamma", 2.0)),
            label_smoothing=float(f.get("label_smoothing", 0.05)),
        )

    if name == "dice":
        d = trainer_cfg.get("dice", {})
        return DiceLoss(smooth=float(d.get("smooth", 1.0)))

    if name == "bce":
        b = trainer_cfg.get("bce", {})
        return BCELoss(
            label_smoothing=float(b.get("label_smoothing", 0.03)),
            entropy_weight=float(b.get("entropy_weight", 0.01)),
        )

    if name == "weighted_bce":
        w = trainer_cfg.get("weighted_bce", {})
        posw = w.get("pos_weight", "auto")
        if posw == "auto" and train_loader is not None:
            posw = estimate_pos_weight(train_loader)
        else:
            posw = float(posw) if isinstance(posw, (int, float, str)) else 3.0
        return WeightedBCELoss(
            pos_weight=float(posw),
            adaptive=bool(w.get("adaptive", True)),
            target_pos_rate=float(w.get("target_pos_rate", 0.1)),
            label_smoothing=float(w.get("label_smoothing", 0.03)),
        )

    # default
    return FocalLoss()
