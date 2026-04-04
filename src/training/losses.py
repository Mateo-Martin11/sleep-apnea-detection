"""Loss functions for binary segmentation with class imbalance."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEWithLogitsLoss(nn.Module):
    """Weighted BCE to handle class imbalance."""

    def __init__(self, pos_weight: float = 3.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pw = torch.tensor(self.pos_weight, device=logits.device)
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)


class DiceLoss(nn.Module):
    """Soft Dice loss (works on probabilities, not logits)."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        # Flatten over batch and time
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice


class FocalLoss(nn.Module):
    """Focal loss to down-weight easy negatives."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = alpha_t * (1 - p_t) ** self.gamma * bce
        return focal.mean()


class CombinedLoss(nn.Module):
    """BCE + Dice combination — best of both worlds."""

    def __init__(self, pos_weight: float = 3.0, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()
        self.bce_w = bce_weight
        self.dice_w = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_w * self.bce(logits, targets) + self.dice_w * self.dice(logits, targets)


def build_loss(name: str, **kwargs) -> nn.Module:
    losses = {
        "bce": BCEWithLogitsLoss,
        "dice": DiceLoss,
        "focal": FocalLoss,
        "combined": CombinedLoss,
    }
    if name not in losses:
        raise ValueError(f"Unknown loss: {name}. Choose from {list(losses.keys())}")
    return losses[name](**kwargs)
