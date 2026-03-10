"""
Detection Losses for 3D Object Detection Head

Implements losses for anchor-free 3D object detection (CenterPoint-style):
  - Center Heatmap Loss (Focal Loss)
  - Box Size Loss (L1 Loss)
  - Orientation Loss (Smooth L1 for sin/cos regression)
  - Classification Loss (Cross Entropy)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================
# CENTER HEATMAP LOSS (Focal Loss)
# ============================================

class FocalLoss(nn.Module):
    """
    Focal Loss for center heatmap prediction.

    Focal Loss = -(1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor (default: 2.0)
        beta: Penalty reduction factor (default: 4.0)
    """
    def __init__(self, alpha=2.0, beta=4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        """
        Args:
            pred: [B, 1, N] predicted center heatmap (sigmoid output)
            target: [B, 1, N] GT center heatmap (0 or 1)

        Returns:
            loss: scalar
        """
        # Clamp predictions to avoid log(0)
        pred = torch.clamp(pred, min=1e-4, max=1 - 1e-4)

        # Positive and negative masks
        pos_mask = target.eq(1).float()
        neg_mask = target.lt(1).float()

        # Positive loss (penalize false negatives)
        pos_loss = -((1 - pred) ** self.alpha) * torch.log(pred) * pos_mask

        # Negative loss (penalize false positives, with penalty reduction for hard negatives)
        neg_loss = -(pred ** self.alpha) * ((1 - target) ** self.beta) * torch.log(1 - pred) * neg_mask

        # Normalize by number of positive points
        num_pos = pos_mask.sum().clamp(min=1.0)
        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos

        return loss


# ============================================
# BOX SIZE LOSS (L1 Loss)
# ============================================

def box_size_loss(pred_size, gt_size, mask):
    """
    L1 loss for box dimensions (length, width, height).

    Args:
        pred_size: [B, 3, N] predicted box dimensions (l, w, h)
        gt_size: [B, 3, N] GT box dimensions
        mask: [B, N] binary mask (1 for valid boxes, 0 otherwise)

    Returns:
        loss: scalar
    """
    # Expand mask to match size dimensions
    mask_expanded = mask.unsqueeze(1).expand_as(pred_size)  # [B, 3, N]

    # L1 loss
    loss = F.l1_loss(pred_size * mask_expanded, gt_size * mask_expanded, reduction='sum')

    # Normalize by number of valid boxes
    num_valid = mask.sum().clamp(min=1.0)
    loss = loss / num_valid

    return loss


# ============================================
# ORIENTATION LOSS (Smooth L1 for sin/cos)
# ============================================

def orientation_loss(pred_ori, gt_ori, mask):
    """
    Smooth L1 loss for orientation (sin, cos regression).

    Args:
        pred_ori: [B, 2, N] predicted orientation (sin(yaw), cos(yaw))
        gt_ori: [B, 2, N] GT orientation (sin(yaw), cos(yaw))
        mask: [B, N] binary mask (1 for valid boxes, 0 otherwise)

    Returns:
        loss: scalar
    """
    # Expand mask to match orientation dimensions
    mask_expanded = mask.unsqueeze(1).expand_as(pred_ori)  # [B, 2, N]

    # Smooth L1 loss
    loss = F.smooth_l1_loss(pred_ori * mask_expanded, gt_ori * mask_expanded, reduction='sum')

    # Normalize by number of valid boxes
    num_valid = mask.sum().clamp(min=1.0)
    loss = loss / num_valid

    return loss


# ============================================
# CLASSIFICATION LOSS (Cross Entropy)
# ============================================

def classification_loss(pred_class, gt_class, mask):
    """
    Cross Entropy loss for object classification.

    Args:
        pred_class: [B, num_classes, N] predicted class logits
        gt_class: [B, N] GT class indices (0, 1, 2, ...)
        mask: [B, N] binary mask (1 for valid boxes, 0 otherwise)

    Returns:
        loss: scalar
    """
    B, num_classes, N = pred_class.shape

    # Reshape for cross entropy
    pred_class = pred_class.permute(0, 2, 1).contiguous()  # [B, N, num_classes]
    pred_class = pred_class.view(B * N, num_classes)       # [B*N, num_classes]
    gt_class = gt_class.view(B * N)                        # [B*N]
    mask = mask.view(B * N)                                # [B*N]

    # Cross entropy loss
    loss = F.cross_entropy(pred_class, gt_class, reduction='none')  # [B*N]

    # Apply mask
    loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)

    return loss


# ============================================
# COMBINED DETECTION LOSS
# ============================================

def compute_detection_loss(pred_detection, gt_boxes, gt_classes, pc1, loss_weights=None):
    """
    Compute combined detection loss.

    Args:
        pred_detection: dict with keys
            'center': [B, 1, N]
            'size': [B, 3, N]
            'orientation': [B, 2, N]
            'class': [B, num_classes, N]
        gt_boxes: List[Tensor] - GT boxes per batch [M, 7] (x, y, z, l, w, h, yaw)
        gt_classes: List[Tensor] - GT class indices per batch [M]
        pc1: [B, 3, N] - point cloud coordinates
        loss_weights: dict with keys
            'center': float
            'size': float
            'orientation': float
            'class': float

    Returns:
        total_loss: scalar
        loss_dict: dict with individual losses
    """
    if loss_weights is None:
        loss_weights = {
            'center': 1.0,
            'size': 0.5,
            'orientation': 0.5,
            'class': 1.0,
        }

    B, _, N = pc1.shape
    device = pc1.device

    # ===== CREATE GT HEATMAPS AND MASKS =====
    gt_center_heatmap = torch.zeros(B, 1, N, device=device)
    gt_size_map = torch.zeros(B, 3, N, device=device)
    gt_ori_map = torch.zeros(B, 2, N, device=device)
    gt_class_map = torch.zeros(B, N, device=device, dtype=torch.long)
    valid_mask = torch.zeros(B, N, device=device)

    # OPTIMIZED: Vectorize distance computation for all boxes at once
    for b in range(B):
        if len(gt_boxes[b]) == 0:
            continue

        boxes_b = gt_boxes[b]  # [M, 7]
        classes_b = gt_classes[b]  # [M]
        M = len(boxes_b)

        pc_b = pc1[b].permute(1, 0)  # [N, 3]
        N = pc_b.shape[0]

        # Vectorized: Compute distances for all boxes at once
        centers = boxes_b[:, :3]  # [M, 3]
        # Broadcast: pc_b [N, 3] - centers [M, 3] → [N, M, 3]
        pc_b_expanded = pc_b.unsqueeze(1)  # [N, 1, 3]
        centers_expanded = centers.unsqueeze(0)  # [1, M, 3]
        dists = torch.norm(pc_b_expanded - centers_expanded, dim=2)  # [N, M]

        # Find points within radius for each box
        center_radius = 0.5
        center_masks = dists < center_radius  # [N, M]

        # Process each box
        for m in range(M):
            center_mask = center_masks[:, m]  # [N]

            if center_mask.sum() > 0:
                box = boxes_b[m]
                cls = classes_b[m]

                # Set center heatmap
                gt_center_heatmap[b, 0, center_mask] = 1.0

                # Set size, orientation, class for center points
                gt_size_map[b, :, center_mask] = box[3:6].unsqueeze(1)  # (l, w, h)
                yaw = box[6]
                gt_ori_map[b, 0, center_mask] = torch.sin(yaw)
                gt_ori_map[b, 1, center_mask] = torch.cos(yaw)
                gt_class_map[b, center_mask] = cls

                # Mark as valid
                valid_mask[b, center_mask] = 1.0

    # ===== COMPUTE LOSSES =====
    focal_loss_fn = FocalLoss(alpha=2.0, beta=4.0)

    loss_center = focal_loss_fn(pred_detection['center'], gt_center_heatmap)
    loss_size = box_size_loss(pred_detection['size'], gt_size_map, valid_mask)
    loss_ori = orientation_loss(pred_detection['orientation'], gt_ori_map, valid_mask)
    loss_cls = classification_loss(pred_detection['class'], gt_class_map, valid_mask)

    # Weighted combination
    total_loss = (
        loss_weights['center'] * loss_center +
        loss_weights['size'] * loss_size +
        loss_weights['orientation'] * loss_ori +
        loss_weights['class'] * loss_cls
    )

    loss_dict = {
        'detection_center': loss_center.item(),
        'detection_size': loss_size.item(),
        'detection_orientation': loss_ori.item(),
        'detection_class': loss_cls.item(),
        'detection_total': total_loss.item(),
    }

    return total_loss, loss_dict
