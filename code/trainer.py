# trainer.py
"""
Trainer for sample-level models (1D: [B,C,T], 2D: [B,1,C,T]) with inline losses.

Features
- choose optimizer: sgd | adam | adamw
- choose loss: bce | bce_pos | focal | dice | hybrid
- focal/dice/hybrid params can come from Hydra's composed cfg (.hydra/config.yaml) or CLI
- early stopping on val loss, grad clipping, optional W&B logging
- quick per-epoch validation confusion (threshold from cfg.eval.threshold or 0.5)

CLI examples (Hydra):
  python main.py trainer.loss=bce
  python main.py trainer.loss=bce_pos trainer.bce_pos.pos_weight=auto
  python main.py trainer.loss=focal trainer.focal.alpha=0.25 trainer.focal.gamma=2
  python main.py trainer.loss=dice trainer.dice.smooth=1.0
  python main.py trainer.loss=hybrid trainer.hybrid.w_bce=1.0 trainer.hybrid.w_dice=0.5
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

# optional deps (safe import)
try:
    import wandb
    _HAS_WANDB = True
except Exception:
    wandb = None
    _HAS_WANDB = False

try:
    from omegaconf import OmegaConf
    _HAS_OMEGACONF = True
except Exception:
    OmegaConf = None
    _HAS_OMEGACONF = False


# =========================
#         LOSSES
# =========================
class BinaryFocalLoss(nn.Module):
    """Focal loss for binary logits [B,T] (or broadcastable)."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)                 # if y=1 -> p, else 1-p
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = alpha_t * (1 - p_t).pow(self.gamma) * bce
        return focal.mean()


class SoftDiceLoss(nn.Module):
    """Soft Dice on probabilities; returns 1 - Dice."""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        p = probs.reshape(-1)
        y = targets.reshape(-1)
        inter = (p * y).sum()
        denom = p.sum() + y.sum()
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice


class HybridLoss(nn.Module):
    """
    Weighted sum of BCEWithLogits, Focal, and Dice.
      total = w_bce * BCE + w_focal * Focal + w_dice * Dice
    Any weight can be zero.
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
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.focal = BinaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice = SoftDiceLoss(smooth=dice_smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total = 0.0
        if self.w_bce:
            total += self.w_bce * self.bce(logits, targets)
        if self.w_focal:
            total += self.w_focal * self.focal(logits, targets)
        if self.w_dice:
            total += self.w_dice * self.dice(logits, targets)
        return total


def estimate_pos_weight(train_loader: DataLoader, device: torch.device, max_batches: int = 10) -> torch.Tensor:
    """
    Approximate pos_weight as negatives/positives over a few train batches.
    Returns a 1-element tensor on the given device.
    """
    pos = 0.0
    neg = 0.0
    with torch.no_grad():
        for i, (_, yb) in enumerate(train_loader):
            y = yb.to(device)
            pos += float(y.sum().item())
            neg += float(y.numel() - y.sum().item())
            if i + 1 >= max_batches:
                break
    if pos <= 0.0:
        return torch.tensor([1.0], device=device)
    return torch.tensor([neg / max(pos, 1.0)], device=device)


# =========================
#     TRAINER CONFIG
# =========================
@dataclass
class TrainConfig:
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS: int = 50
    LR: float = 1e-3
    BATCH_SIZE: int = 16
    WEIGHT_DECAY: float = 0.0
    OPTIMIZER: str = "adam"           # sgd | adam | adamw
    LOSS: str = "bce"                 # bce | bce_pos | focal | dice | hybrid
    EARLY_STOPPING_PATIENCE: int = 10
    GRAD_CLIP_NORM: float = 0.0
    USE_WANDB: bool = False
    WANDB_PROJECT: str = "eeg-spindle-detection"
    MODEL_NAME: str = "unet1d"
    RUN_NAME: Optional[str] = None


# =========================
#         TRAINER
# =========================
class SpindleTrainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device(config.DEVICE)
        self.criterion: Optional[nn.Module] = None
        self.hydra_cfg = self._load_hydra_cfg()  # composed Hydra cfg dict if present

    def fit(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            extra_cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        model = model.to(self.device)

        optimizer = self._make_optimizer(model)
        self.criterion = self._build_criterion(train_loader)
        run = self._wandb_init(model, extra_cfg)

        expects_2d = self._model_expects_2d(model)

        best_score = -float("inf")  # <-- PR-AUC (higher is better)
        best_state = None
        patience = 0

        # read threshold for per-epoch confusion (display only)
        threshold = float(self._get_cfg("eval.threshold", 0.5))

        # lazy import to avoid hard dependency
        try:
            from metrics import SampleMetrics  # your metrics.py
            _HAS_SM = True
        except Exception:
            _HAS_SM = False

        for epoch in range(1, self.config.EPOCHS + 1):
            # ----- train -----
            model.train()
            train_losses = []
            for xb, yb in train_loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                xb = self._standardize_inputs(xb, expects_2d)
                optimizer.zero_grad(set_to_none=True)

                logits = model(xb)  # [B,T] or [B,1,T]
                logits, yb = self._standardize_outputs_targets(logits, yb)  # -> [B,T], [B,T]

                loss = self.criterion(logits, yb)
                loss.backward()
                if self.config.GRAD_CLIP_NORM and self.config.GRAD_CLIP_NORM > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), self.config.GRAD_CLIP_NORM)
                optimizer.step()
                train_losses.append(loss.item())

            train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

            # ----- validate -----
            if val_loader is not None:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(self.device, non_blocking=True)
                        yb = yb.to(self.device, non_blocking=True)
                        xb = self._standardize_inputs(xb, expects_2d)
                        logits = model(xb)
                        logits, yb = self._standardize_outputs_targets(logits, yb)
                        vloss = self.criterion(logits, yb).item()
                        val_losses.append(vloss)
                val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

                # compute PR-AUC/ROC-AUC for *selection/logging*
                pr_auc = float("nan");
                roc_auc = float("nan")
                conf = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
                if _HAS_SM:
                    try:
                        sm = SampleMetrics()
                        # PR/ROC
                        aucs = sm.epoch_roc_pr(model, val_loader, device=str(self.device))
                        pr_auc = float(aucs.get("pr_auc", float("nan")))
                        roc_auc = float(aucs.get("roc_auc", float("nan")))
                        # quick confusion at display threshold
                        conf = sm.epoch_confusion(model, val_loader, device=str(self.device), threshold=threshold)
                    except Exception:
                        pass

                # model selection by PR-AUC
                improved = (pr_auc > best_score + 1e-6)
                if improved:
                    best_score = pr_auc
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1

                # log/print
                if run is not None:
                    wandb.log({
                        "loss/train": train_loss,
                        "loss/val": val_loss,
                        "val/pr_auc": pr_auc,
                        "val/roc_auc": roc_auc,
                        "val/tp": conf["tp"], "val/fp": conf["fp"],
                        "val/tn": conf["tn"], "val/fn": conf["fn"],
                        "val/precision": conf["precision"],
                        "val/recall": conf["recall"],
                        "val/f1": conf["f1"],
                        "epoch": epoch,
                    }, step=epoch)

                print(f"epoch {epoch}/{self.config.EPOCHS} "
                      f"train={train_loss:.6f} val_loss={val_loss:.6f} "
                      f"val_pr_auc={pr_auc if not np.isnan(pr_auc) else -1:.4f} "
                      f"P={conf['precision']:.4f} R={conf['recall']:.4f} F1={conf['f1']:.4f}")

                if self.config.EARLY_STOPPING_PATIENCE and patience >= self.config.EARLY_STOPPING_PATIENCE:
                    print(
                        f"early stopping at epoch {epoch} (no PR-AUC improvement for {self.config.EARLY_STOPPING_PATIENCE})")
                    break
            else:
                # no validation
                if run is not None:
                    wandb.log({"loss/train": train_loss, "epoch": epoch}, step=epoch)
                print(f"epoch {epoch}/{self.config.EPOCHS} train={train_loss:.6f}")

        # restore best PR-AUC weights
        if best_state is not None:
            model.load_state_dict(best_state)

        return {"best_val_pr_auc": best_score}

    # ---------- internals ----------
    def _make_optimizer(self, model: nn.Module):
        opt = (self.config.OPTIMIZER or "adam").lower()
        if opt == "sgd":
            return torch.optim.SGD(model.parameters(), lr=self.config.LR, momentum=0.9,
                                   weight_decay=self.config.WEIGHT_DECAY)
        if opt == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=self.config.LR,
                                     weight_decay=self.config.WEIGHT_DECAY)
        # default: adam
        return torch.optim.Adam(model.parameters(), lr=self.config.LR,
                                weight_decay=self.config.WEIGHT_DECAY)

    def _build_criterion(self, train_loader: Optional[DataLoader]) -> nn.Module:
        """Build loss; read nested params from composed Hydra cfg if present."""
        name = (self.config.LOSS or "bce").lower()

        def get(path: str, default=None):
            return self._get_cfg(path, default)

        if name == "bce":
            return nn.BCEWithLogitsLoss()

        elif name == "bce_pos":
            pos_cfg = get("trainer.bce_pos.pos_weight", "auto")
            if isinstance(pos_cfg, (int, float)):
                pw = torch.tensor([float(pos_cfg)], device=self.device)
                return nn.BCEWithLogitsLoss(pos_weight=pw)
            if isinstance(pos_cfg, str) and pos_cfg == "auto":
                pw = estimate_pos_weight(train_loader, self.device)
                return nn.BCEWithLogitsLoss(pos_weight=pw)
            return nn.BCEWithLogitsLoss()

        elif name == "focal":
            alpha = float(get("trainer.focal.alpha", 0.25) or 0.25)
            gamma = float(get("trainer.focal.gamma", 2.0) or 2.0)
            return BinaryFocalLoss(alpha=alpha, gamma=gamma)

        elif name == "dice":
            smooth = float(get("trainer.dice.smooth", 1.0) or 1.0)
            return SoftDiceLoss(smooth=smooth)

        elif name == "hybrid":
            w_bce = float(get("trainer.hybrid.w_bce", 1.0) or 1.0)
            w_focal = float(get("trainer.hybrid.w_focal", 0.0) or 0.0)
            w_dice = float(get("trainer.hybrid.w_dice", 0.0) or 0.0)

            # optional pos_weight for BCE term
            pos_cfg = get("trainer.hybrid.pos_weight", None)
            if isinstance(pos_cfg, (int, float)):
                pos_weight = torch.tensor([float(pos_cfg)], device=self.device)
            elif isinstance(pos_cfg, str) and pos_cfg == "auto":
                pos_weight = estimate_pos_weight(train_loader, self.device)
            else:
                pos_weight = None

            f_alpha = float(get("trainer.hybrid.focal.alpha", get("trainer.focal.alpha", 0.25)) or 0.25)
            f_gamma = float(get("trainer.hybrid.focal.gamma", get("trainer.focal.gamma", 2.0)) or 2.0)
            d_smooth = float(get("trainer.hybrid.dice.smooth", get("trainer.dice.smooth", 1.0)) or 1.0)

            return HybridLoss(
                w_bce=w_bce, w_focal=w_focal, w_dice=w_dice,
                pos_weight=pos_weight, focal_alpha=f_alpha, focal_gamma=f_gamma,
                dice_smooth=d_smooth
            )

        else:
            print(f"[warn] unknown loss '{name}', falling back to BCEWithLogitsLoss")
            return nn.BCEWithLogitsLoss()

    def _model_expects_2d(self, model: nn.Module) -> bool:
        if hasattr(model, "_expected_input"):
            return getattr(model, "_expected_input") == "B1CT"
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                return True
        return False

    def _standardize_inputs(self, x: torch.Tensor, expects_2d: bool) -> torch.Tensor:
        # Our datasets yield [B,C,T]. If the model is 2D, add a singleton channel.
        if expects_2d and x.ndim == 3:
            return x.unsqueeze(1)  # [B,1,C,T]
        if (not expects_2d) and x.ndim == 4 and x.shape[1] == 1:
            return x.squeeze(1)    # [B,C,T]
        return x

    def _standardize_outputs_targets(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # logits -> [B,T]
        if logits.ndim == 3 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        if logits.ndim == 3 and logits.shape[2] == 1:
            logits = logits.squeeze(2)
        # targets -> [B,T]
        if targets.ndim == 3 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        if targets.ndim == 3 and targets.shape[2] == 1:
            targets = targets.squeeze(2)
        return logits, targets

    def _wandb_init(self, model: nn.Module, extra_cfg: Optional[Dict[str, Any]]):
        """Version-tolerant wandb.init (works with old/new wandb)."""
        if (not self.config.USE_WANDB) or (wandb is None) or (os.environ.get("WANDB_DISABLED") == "true"):
            return None

        run_name = self.config.RUN_NAME or f"{self.config.MODEL_NAME}_lr{self.config.LR}"
        kwargs = {"project": self.config.WANDB_PROJECT, "name": run_name}

        run = None
        try:
            # newer wandb
            run = wandb.init(**kwargs, return_previous=False, finish_previous=True)
        except TypeError:
            try:
                # older wandb
                run = wandb.init(**kwargs, reinit=True)
            except TypeError:
                # very old
                run = wandb.init(**kwargs)

        # try to log a compact summary/config
        try:
            cfg_summary = {
                "device": self.config.DEVICE, "epochs": self.config.EPOCHS, "lr": self.config.LR,
                "batch_size": self.config.BATCH_SIZE, "optimizer": self.config.OPTIMIZER,
                "loss": self.config.LOSS, "weight_decay": self.config.WEIGHT_DECAY, "model": self.config.MODEL_NAME
            }
            if extra_cfg:
                # safe: don't pass the whole hydra tree if your wandb is picky
                wandb.config.update(cfg_summary, allow_val_change=True)
        except Exception:
            pass

        # parameter counts
        try:
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            wandb.log({"model/total_parameters": total, "model/trainable_parameters": trainable})
        except Exception:
            pass

        return run

    def _load_hydra_cfg(self) -> Optional[Dict[str, Any]]:
        """Try to read the composed Hydra config saved at ./.hydra/config.yaml."""
        if not _HAS_OMEGACONF:
            return None
        path = os.path.join(os.getcwd(), ".hydra", "config.yaml")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = OmegaConf.load(f)
            return OmegaConf.to_container(cfg, resolve=True)
        except Exception:
            return None

    def _get_cfg(self, path: str, default=None):
        """
        Nested getter into composed Hydra cfg (dict), e.g. "trainer.focal.alpha".
        Returns default if not available.
        """
        if self.hydra_cfg is None:
            return default
        cur: Any = self.hydra_cfg
        for k in path.split("."):
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return default
        return cur
