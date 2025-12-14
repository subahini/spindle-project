# losses.py
from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


class FocalLoss(nn.Module):
    """
    Standard binary focal loss for logits.
    """
    def __init__(
        self,
        alpha: float = 0.15,
        gamma: float = 3.0,
        label_smoothing: float = 0.05,
        logit_clamp: float = 8.0,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.logit_clamp = float(logit_clamp)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = torch.clamp(logits, -self.logit_clamp, self.logit_clamp)
        targets = targets.float()

        if self.label_smoothing > 0.0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        prob = torch.sigmoid(logits)
        prob = prob.view_as(targets)

        # BCE per element
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        pt = torch.where(targets == 1, prob, 1 - prob)
        alpha_factor = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_factor * (1 - pt).pow(self.gamma)

        loss = focal_weight * bce
        return loss.mean()


class DiceLoss(nn.Module):
    """
    Dice loss on logits for binary masks.
    """
    def __init__(self, smooth: float = 1.0, logit_clamp: float = 8.0):
        super().__init__()
        self.smooth = float(smooth)
        self.logit_clamp = float(logit_clamp)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = torch.clamp(logits, -self.logit_clamp, self.logit_clamp)
        probs = torch.sigmoid(logits)
        targets = targets.float()

        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class WeightedBCELoss(nn.Module):
    """
    BCEWithLogits with configurable pos_weight and optional label smoothing.
    Supports an 'adaptive' mode.
    """
    def __init__(
        self,
        pos_weight: float = 3.0,
        adaptive: bool = False,
        target_pos_rate: float = 0.1,
        label_smoothing: float = 0.03,
        logit_clamp: float = 8.0,
    ):
        super().__init__()
        self.base_pos_weight = float(pos_weight)
        self.adaptive = bool(adaptive)
        self.target_pos_rate = float(target_pos_rate)
        self.label_smoothing = float(label_smoothing)
        self.logit_clamp = float(logit_clamp)
        self._ema = None  # for adaptive pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = torch.clamp(logits, -self.logit_clamp, self.logit_clamp)
        targets = targets.float()

        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        pos_weight = self.base_pos_weight

        if self.adaptive:
            with torch.no_grad():
                cur_prob = torch.sigmoid(logits).mean().item()
                if self._ema is None:
                    self._ema = self.target_pos_rate
                self._ema = 0.9 * self._ema + 0.1 * cur_prob

                if self._ema > self.target_pos_rate * 3:
                    pos_weight = self.base_pos_weight * 0.5
                elif self._ema < self.target_pos_rate * 0.3:
                    pos_weight = self.base_pos_weight * 1.5

        pos_w = torch.tensor(pos_weight, device=logits.device)
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_w
        )


def estimate_pos_weight(train_loader: DataLoader) -> float:
    """
    Estimate pos_weight = (#neg / #pos) from the training loader.
    """
    total_pos = 0.0
    total = 0.0
    with torch.no_grad():
        for _, targets in train_loader:
            t = targets.view(-1).float()
            total_pos += t.sum().item()
            total += t.numel()

    if total_pos == 0 or total == 0:
        return 1.0

    total_neg = total - total_pos
    if total_pos == 0:
        return 1.0

    return float(total_neg / total_pos)


def build_loss_function(
    name: str,
    trainer_cfg: Optional[Dict[str, Any]] = None,
    train_loader: Optional[DataLoader] = None,
) -> nn.Module:
    """
    Factory for loss functions.

    name: "focal", "dice", "weighted_bce", "bce"
    trainer_cfg: optional dict with hyperparams per loss type, e.g.

        trainer_cfg = {
          "focal": {"alpha": 0.2, "gamma": 2.5},
          "weighted_bce": {"pos_weight": "auto", "label_smoothing": 0.03},
          "dice": {"smooth": 1.0},
        }
    """
    trainer_cfg = trainer_cfg or {}
    name = name.lower()

    if name == "focal":
        cfg = trainer_cfg.get("focal", {})
        return FocalLoss(
            alpha=float(cfg.get("alpha", 0.15)),
            gamma=float(cfg.get("gamma", 3.0)),
            label_smoothing=float(cfg.get("label_smoothing", 0.05)),
            logit_clamp=float(cfg.get("logit_clamp", 8.0)),
        )

    if name == "dice":
        cfg = trainer_cfg.get("dice", {})
        return DiceLoss(
            smooth=float(cfg.get("smooth", 1.0)),
            logit_clamp=float(cfg.get("logit_clamp", 8.0)),
        )

    if name in ("weighted_bce", "bce"):
        cfg = trainer_cfg.get("weighted_bce", {})
        posw = cfg.get("pos_weight", "auto")

        if posw == "auto" and train_loader is not None:
            posw = estimate_pos_weight(train_loader)
        elif posw == "auto":
            posw = 3.0

        return WeightedBCELoss(
            pos_weight=float(posw),
            adaptive=bool(cfg.get("adaptive", False)),
            target_pos_rate=float(cfg.get("target_pos_rate", 0.1)),
            label_smoothing=float(cfg.get("label_smoothing", 0.03)),
            logit_clamp=float(cfg.get("logit_clamp", 8.0)),
        )

    # default fallback
    return FocalLoss()
