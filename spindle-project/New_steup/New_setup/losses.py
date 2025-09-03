"""
Loss Functions for EEG Spindle Detection
========================================

This module contains all loss functions used in the project:
- Binary Cross Entropy (BCE)
- Focal Loss for addressing class imbalance
- Soft Dice Loss for segmentation-style training
- Hybrid Loss combining multiple loss terms
- Utilities for automatic positive weight estimation
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any


class BinaryFocalLoss(nn.Module):
    """
    Focal Loss for binary classification.

    Addresses class imbalance by down-weighting easy examples and focusing
    learning on hard examples. Particularly useful for EEG spindle detection
    where spindles are rare events.

    Args:
        alpha: Weighting factor for rare class [0,1]. 0.25 means 25% weight for positives
        gamma: Focusing parameter. Higher gamma puts more focus on hard examples
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute BCE loss
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Compute probabilities and probability of correct class
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)  # p if y=1, 1-p if y=0

        # Compute alpha weighting (favor minority class)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Apply focal term: (1-p_t)^gamma reduces loss for well-classified examples
        focal = alpha_t * (1 - p_t).pow(self.gamma) * bce
        return focal.mean()


class SoftDiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.

    Dice coefficient measures overlap between prediction and ground truth.
    Returns 1 - Dice so that minimizing the loss maximizes Dice coefficient.
    Particularly effective for segmentation tasks with spatial/temporal continuity.

    Args:
        smooth: Smoothing factor to prevent division by zero
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Flatten for computation
        p = probs.reshape(-1)
        y = targets.reshape(-1)

        # Dice coefficient: 2*|intersection| / (|A| + |B|)
        intersection = (p * y).sum()
        denominator = p.sum() + y.sum()
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)

        # Return 1 - dice (loss decreases as dice increases)
        return 1.0 - dice


class HybridLoss(nn.Module):
    """
    Hybrid Loss combining multiple loss terms.

    Allows flexible combination of BCE, Focal, and Dice losses with different weights.
    Useful for leveraging strengths of different loss functions:
    - BCE: stable gradient flow
    - Focal: handles class imbalance
    - Dice: spatial/temporal coherence

    Args:
        w_bce: Weight for BCE loss term
        w_focal: Weight for Focal loss term
        w_dice: Weight for Dice loss term
        pos_weight: Positive class weight for BCE term
        focal_alpha, focal_gamma: Parameters for Focal loss
        dice_smooth: Smoothing parameter for Dice loss
    """

    def __init__(
            self,
            w_bce: float = 1.0,
            w_focal: float = 0.0,
            w_dice: float = 0.0,
            pos_weight: Optional[torch.Tensor] = None,
            focal_alpha: float = 0.25,
            focal_gamma: float = 2.0,
            dice_smooth: float = 1.0,
    ):
        super().__init__()
        self.w_bce = float(w_bce)
        self.w_focal = float(w_focal)
        self.w_dice = float(w_dice)

        # Initialize component losses
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.focal = BinaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice = SoftDiceLoss(smooth=dice_smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total = 0.0

        if self.w_bce > 0:
            total += self.w_bce * self.bce(logits, targets)
        if self.w_focal > 0:
            total += self.w_focal * self.focal(logits, targets)
        if self.w_dice > 0:
            total += self.w_dice * self.dice(logits, targets)

        return total


def estimate_pos_weight(train_loader: DataLoader, device: torch.device, max_batches: int = 10) -> torch.Tensor:
    """
    Estimate positive class weight from training data.

    Computes the ratio of negative to positive samples to use as pos_weight
    in BCEWithLogitsLoss. This helps balance training when classes are imbalanced.

    Args:
        train_loader: DataLoader for training data
        device: Device to place the result tensor on
        max_batches: Maximum number of batches to sample for estimation

    Returns:
        Tensor with estimated positive weight [neg_samples / pos_samples]
    """
    pos_count = 0.0
    neg_count = 0.0

    with torch.no_grad():
        for i, (_, yb) in enumerate(train_loader):
            y = yb.to(device)
            pos_count += float(y.sum().item())
            neg_count += float(y.numel() - y.sum().item())

            if i + 1 >= max_batches:
                break

    if pos_count <= 0.0:
        return torch.tensor([1.0], device=device)

    pos_weight = neg_count / max(pos_count, 1.0)
    print(f"[Loss] Estimated pos_weight: {pos_weight:.2f} (neg: {neg_count:.0f}, pos: {pos_count:.0f})")

    return torch.tensor([pos_weight], device=device)


def build_loss_function(
        loss_name: str,
        cfg: Dict[str, Any],
        train_loader: Optional[DataLoader] = None,
        device: str = "cpu"
) -> nn.Module:
    """
    Factory function to build loss functions from configuration.

    Args:
        loss_name: Name of loss function (bce, bce_pos, focal, dice, hybrid)
        cfg: Configuration dictionary containing loss parameters
        train_loader: Training DataLoader (needed for auto pos_weight estimation)
        device: Device string for tensors

    Returns:
        Configured loss function module
    """
    device_obj = torch.device(device)
    name = (loss_name or "bce").lower()

    def get_nested(path: str, default=None):
        """Get nested config value using dot notation."""
        keys = path.split(".")
        current = cfg
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    if name == "bce":
        return nn.BCEWithLogitsLoss()

    elif name == "bce_pos":
        pos_cfg = get_nested("trainer.bce_pos.pos_weight", "auto")
        if isinstance(pos_cfg, (int, float)):
            pos_weight = torch.tensor([float(pos_cfg)], device=device_obj)
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif isinstance(pos_cfg, str) and pos_cfg == "auto" and train_loader is not None:
            pos_weight = estimate_pos_weight(train_loader, device_obj)
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            return nn.BCEWithLogitsLoss()

    elif name == "focal":
        alpha = float(get_nested("trainer.focal.alpha", 0.25) or 0.25)
        gamma = float(get_nested("trainer.focal.gamma", 2.0) or 2.0)
        return BinaryFocalLoss(alpha=alpha, gamma=gamma)

    elif name == "dice":
        smooth = float(get_nested("trainer.dice.smooth", 1.0) or 1.0)
        return SoftDiceLoss(smooth=smooth)

    elif name == "hybrid":
        w_bce = float(get_nested("trainer.hybrid.w_bce", 1.0) or 1.0)
        w_focal = float(get_nested("trainer.hybrid.w_focal", 0.0) or 0.0)
        w_dice = float(get_nested("trainer.hybrid.w_dice", 0.0) or 0.0)

        # Handle pos_weight for BCE component
        pos_cfg = get_nested("trainer.hybrid.pos_weight", None)
        pos_weight = None
        if isinstance(pos_cfg, (int, float)):
            pos_weight = torch.tensor([float(pos_cfg)], device=device_obj)
        elif isinstance(pos_cfg, str) and pos_cfg == "auto" and train_loader is not None:
            pos_weight = estimate_pos_weight(train_loader, device_obj)

        # Focal loss parameters
        f_alpha = float(get_nested("trainer.hybrid.focal.alpha",
                                   get_nested("trainer.focal.alpha", 0.25)) or 0.25)
        f_gamma = float(get_nested("trainer.hybrid.focal.gamma",
                                   get_nested("trainer.focal.gamma", 2.0)) or 2.0)

        # Dice loss parameters
        d_smooth = float(get_nested("trainer.hybrid.dice.smooth",
                                    get_nested("trainer.dice.smooth", 1.0)) or 1.0)

        return HybridLoss(
            w_bce=w_bce, w_focal=w_focal, w_dice=w_dice,
            pos_weight=pos_weight, focal_alpha=f_alpha, focal_gamma=f_gamma,
            dice_smooth=d_smooth
        )

    else:
        print(f"[Warning] Unknown loss '{name}', falling back to BCEWithLogitsLoss")
        return nn.BCEWithLogitsLoss()


# Convenience functions for common loss configurations
def get_balanced_bce_loss(train_loader: DataLoader, device: str = "cpu") -> nn.Module:
    """Get BCE loss with automatically estimated positive weight."""
    pos_weight = estimate_pos_weight(train_loader, torch.device(device))
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def get_spindle_focal_loss(alpha: float = 0.25, gamma: float = 2.0) -> nn.Module:
    """Get Focal loss with parameters optimized for spindle detection."""
    return BinaryFocalLoss(alpha=alpha, gamma=gamma)


def get_segmentation_loss(bce_weight: float = 0.7, dice_weight: float = 0.3,
                          smooth: float = 1.0) -> nn.Module:
    """Get hybrid loss combining BCE and Dice for segmentation tasks."""
    return HybridLoss(w_bce=bce_weight, w_focal=0.0, w_dice=dice_weight,
                      dice_smooth=smooth)