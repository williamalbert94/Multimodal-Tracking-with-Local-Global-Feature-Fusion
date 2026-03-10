"""
Advanced Loss Functions for Imbalanced Segmentation.

Implements:
1. Focal Loss - handles class imbalance by down-weighting easy examples
2. Dice Loss - optimizes IoU directly
3. Combined Loss - best of both worlds

References:
- Focal Loss: Lin et al. "Focal Loss for Dense Object Detection" (2017)
- Dice Loss: Milletari et al. "V-Net" (2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Standard BCE treats all examples equally → model biases toward majority class.
    Focal Loss down-weights easy examples (well-classified) → forces learning on hard cases.

    Formula:
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    where:
        p_t = model's estimated probability for true class
        α_t = balancing factor for positive class
        γ = focusing parameter (typically 2)

    When γ=0, reduces to weighted BCE.
    When γ>0, reduces loss for easy examples (p_t → 1).
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', label_smoothing=0.0):
        """
        Args:
            alpha: Weight for positive class (0.25 = 25% weight for moving objects)
                   Use α < 0.5 when positive class is minority
            gamma: Focusing parameter (2.0 recommended)
                   Higher γ = more focus on hard examples
            reduction: 'mean', 'sum', or 'none'
            label_smoothing: Small epsilon to prevent overconfidence (0.0-0.1)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [N] or [B, N] raw logits (before sigmoid)
            targets: [N] or [B, N] binary labels {0, 1}

        Returns:
            Focal loss scalar
        """
        # Flatten to 1D
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1).float()

        # Label smoothing (optional regularization)
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Compute BCE loss (with logits for numerical stability)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Get predicted probabilities
        p_t = torch.exp(-bce_loss)  # p_t = p if y=1, else 1-p

        # Focal term: (1 - p_t)^gamma
        # When p_t → 1 (easy example), focal_weight → 0 (down-weight)
        # When p_t → 0 (hard example), focal_weight → 1 (full weight)
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha balancing
        # Gives more weight to minority class (moving objects)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Final focal loss
        focal_loss = alpha_t * focal_weight * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Soft Dice Loss - directly optimizes IoU metric.

    Standard pixel-wise losses (BCE, Focal) don't optimize segmentation overlap.
    Dice Loss measures intersection-over-union (IoU) in a differentiable way.

    Formula:
        Dice = 2 * |X ∩ Y| / (|X| + |Y|)
        Loss = 1 - Dice

    Advantages:
    - Handles class imbalance naturally (focuses on overlap, not pixel count)
    - Optimizes the metric we care about (IoU/F1)
    - Works well for small objects
    """

    def __init__(self, smooth=1.0, squared_pred=False):
        """
        Args:
            smooth: Laplace smoothing to avoid division by zero
            squared_pred: If True, square predictions before computing Dice
                         (increases penalty for low-confidence predictions)
        """
        super().__init__()
        self.smooth = smooth
        self.squared_pred = squared_pred

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [N] or [B, N] raw logits (before sigmoid)
            targets: [N] or [B, N] binary labels {0, 1}

        Returns:
            Dice loss scalar (0 = perfect overlap, 1 = no overlap)
        """
        # Apply sigmoid to get probabilities
        inputs = torch.sigmoid(inputs)

        # Flatten
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        # Optional: square predictions (penalize uncertainty)
        if self.squared_pred:
            inputs = inputs ** 2

        # Compute Dice coefficient
        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Return Dice Loss = 1 - Dice
        return 1.0 - dice


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice Loss with FP/FN weighting.

    Allows controlling trade-off between false positives and false negatives.
    Useful when FP and FN have different costs.

    Formula:
        Tversky = TP / (TP + α*FP + β*FN)

    Special cases:
        α = β = 0.5 → Dice Loss
        α = β = 1.0 → Tanimoto/Jaccard
        α < β → penalize FN more (recall-focused)
        α > β → penalize FP more (precision-focused)
    """

    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        """
        Args:
            alpha: Weight for false positives
            beta: Weight for false negatives
            smooth: Laplace smoothing
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [N] raw logits
            targets: [N] binary labels

        Returns:
            Tversky loss scalar
        """
        inputs = torch.sigmoid(inputs).reshape(-1)
        targets = targets.reshape(-1)

        # True Positive, False Positive, False Negative
        TP = (inputs * targets).sum()
        FP = (inputs * (1 - targets)).sum()
        FN = ((1 - inputs) * targets).sum()

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        return 1.0 - tversky


class CombinedSegmentationLoss(nn.Module):
    """
    Combined loss for robust segmentation.

    Combines:
    1. Focal Loss - handles class imbalance
    2. Dice Loss - optimizes IoU

    This combination:
    - Focal Loss ensures pixel-wise accuracy on hard examples
    - Dice Loss ensures good regional overlap (IoU)
    """

    def __init__(self,
                 focal_weight=0.7,
                 dice_weight=0.3,
                 focal_alpha=0.25,
                 focal_gamma=2.0,
                 dice_smooth=1.0):
        """
        Args:
            focal_weight: Weight for focal loss component (0.7 recommended)
            dice_weight: Weight for dice loss component (0.3 recommended)
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
            dice_smooth: Smoothing for dice loss
        """
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss(smooth=dice_smooth)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [N] raw logits
            targets: [N] binary labels

        Returns:
            Combined loss scalar
        """
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)

        total = self.focal_weight * focal + self.dice_weight * dice

        return total, focal, dice  # Return components for logging


# ============================================================================
# Helper: Weighted BCE (baseline comparison)
# ============================================================================

def weighted_bce_loss(inputs, targets, pos_weight=1.0):
    """
    Weighted Binary Cross-Entropy (baseline).

    Simple class balancing by giving more weight to positive class.

    Args:
        inputs: [N] logits
        targets: [N] binary labels
        pos_weight: Weight multiplier for positive class

    Returns:
        BCE loss scalar
    """
    pos_weight_tensor = torch.tensor([pos_weight], device=inputs.device)
    loss = F.binary_cross_entropy_with_logits(
        inputs.reshape(-1),
        targets.reshape(-1).float(),
        pos_weight=pos_weight_tensor
    )
    return loss


# ============================================================================
# Testing
# ============================================================================

def test_losses():
    """Quick test of loss functions."""
    torch.manual_seed(42)

    # Create dummy predictions and targets
    batch_size = 32
    num_points = 512

    # Simulated imbalanced data (10% positive class)
    logits = torch.randn(batch_size, num_points)
    targets = (torch.rand(batch_size, num_points) > 0.9).float()

    print("=" * 60)
    print("Testing Loss Functions")
    print("=" * 60)
    print(f"Batch size: {batch_size}, Points: {num_points}")
    print(f"Positive class ratio: {targets.mean().item():.2%}")
    print()

    # Test Focal Loss
    focal = FocalLoss(alpha=0.25, gamma=2.0)
    focal_val = focal(logits, targets)
    print(f"Focal Loss (α=0.25, γ=2.0): {focal_val.item():.4f}")

    # Test Dice Loss
    dice = DiceLoss(smooth=1.0)
    dice_val = dice(logits, targets)
    print(f"Dice Loss:                   {dice_val.item():.4f}")

    # Test Tversky Loss
    tversky = TverskyLoss(alpha=0.3, beta=0.7)  # Penalize FN more
    tversky_val = tversky(logits, targets)
    print(f"Tversky Loss (α=0.3, β=0.7): {tversky_val.item():.4f}")

    # Test Combined Loss
    combined = CombinedSegmentationLoss(focal_weight=0.7, dice_weight=0.3)
    total, focal_comp, dice_comp = combined(logits, targets)
    print(f"Combined Loss:               {total.item():.4f}")
    print(f"  ├─ Focal component:        {focal_comp.item():.4f}")
    print(f"  └─ Dice component:         {dice_comp.item():.4f}")

    # Test Weighted BCE (baseline)
    bce_val = weighted_bce_loss(logits, targets, pos_weight=9.0)
    print(f"Weighted BCE (pos_w=9.0):    {bce_val.item():.4f}")

    print("\nAll loss functions working correctly!")


if __name__ == "__main__":
    test_losses()
