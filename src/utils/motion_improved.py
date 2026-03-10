"""
Improved Motion Module with Ego-Motion Compensation and Box-Level Features
===========================================================================

Key improvements over baseline point-level flow:
1. GT ego-motion compensation (removes sensor motion artifacts)
2. Box-level temporal motion (direction, speed, is_moving)
3. Motion-weighted segmentation loss (focus on moving objects)

Author: Claude Code
Date: 2026-03-06
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compensate_ego_motion(pc_t0, ego_transform_t0_to_t1):
    """
    Transform points from frame t to frame t+1 coordinate system using GT ego-motion.

    This removes the motion component induced by the sensor's own movement,
    isolating only the object motion relative to the scene.

    Args:
        pc_t0: [B, N, 3] points at time t in frame t coordinates
        ego_transform_t0_to_t1: [B, 4, 4] SE(3) rigid transform from t to t+1
            Format: [[R, t], [0, 1]] where R is [3,3] rotation, t is [3,1] translation

    Returns:
        pc_t0_in_t1: [B, N, 3] points from t transformed to t+1 frame
    """
    B, N, _ = pc_t0.shape
    device = pc_t0.device

    # Convert to homogeneous coordinates [B, N, 4]
    pc_t0_hom = torch.cat([
        pc_t0,
        torch.ones(B, N, 1, device=device)
    ], dim=-1)  # [B, N, 4]

    # Apply ego transform: p_t1 = T @ p_t0
    # ego_transform_t0_to_t1: [B, 4, 4]
    # pc_t0_hom: [B, N, 4]
    #
    # Method: batch matrix multiplication
    # Reshape pc_t0_hom to [B, N, 4, 1] for bmm
    pc_t0_hom = pc_t0_hom.unsqueeze(-1)  # [B, N, 4, 1]

    # Expand transform for batch matrix multiplication
    # ego_transform: [B, 4, 4] → [B, 1, 4, 4] → [B, N, 4, 4]
    ego_transform_expanded = ego_transform_t0_to_t1.unsqueeze(1).expand(B, N, 4, 4)

    # Apply transform: [B, N, 4, 4] @ [B, N, 4, 1] → [B, N, 4, 1]
    pc_t0_in_t1_hom = torch.matmul(ego_transform_expanded, pc_t0_hom)  # [B, N, 4, 1]

    # Remove homogeneous coordinate and extra dimension
    pc_t0_in_t1 = pc_t0_in_t1_hom.squeeze(-1)[:, :, :3]  # [B, N, 3]

    return pc_t0_in_t1


def compute_box_motion_features(boxes_t0, boxes_t1, track_ids_t0, track_ids_t1,
                                 delta_t=0.1):
    """
    Compute motion features at box level (direction, speed, is_moving).

    After ego-motion compensation, this extracts object-level motion that is more
    stable and meaningful than point-level flow, especially for sparse radar.

    Args:
        boxes_t0: [M, 7] boxes at time t (x, y, z, l, w, h, yaw)
        boxes_t1: [M', 7] boxes at time t+1 (already in same frame after ego-comp)
        track_ids_t0: [M] GT track IDs at t
        track_ids_t1: [M'] GT track IDs at t+1
        delta_t: float, time between frames (default 0.1s for 10Hz)

    Returns:
        motion_features: Dict {
            'displacement': [M_matched, 3],    # (dx, dy, dz) in meters
            'direction_2d': [M_matched],       # BEV direction angle in radians
            'speed': [M_matched],              # Speed magnitude in m/s
            'is_moving': [M_matched],          # Binary (1 if speed > threshold)
            'matched_indices_t0': List[int],   # Indices in boxes_t0
            'matched_indices_t1': List[int],   # Indices in boxes_t1
        }
        Returns None if no matches found.
    """
    device = boxes_t0.device

    # Match boxes by track ID
    matched_pairs = []
    for i in range(len(track_ids_t0)):
        id_t0 = track_ids_t0[i].item()
        if id_t0 <= 0:  # Skip invalid IDs
            continue

        # Find same ID in t+1
        matches = (track_ids_t1 == id_t0).nonzero(as_tuple=True)[0]
        if len(matches) > 0:
            j = matches[0].item()  # Take first match
            matched_pairs.append((i, j))

    if len(matched_pairs) == 0:
        return None

    indices_t0, indices_t1 = zip(*matched_pairs)
    indices_t0 = list(indices_t0)
    indices_t1 = list(indices_t1)

    boxes_matched_t0 = boxes_t0[indices_t0]  # [M_matched, 7]
    boxes_matched_t1 = boxes_t1[indices_t1]  # [M_matched, 7]

    # Compute displacement (object motion in scene)
    centers_t0 = boxes_matched_t0[:, :3]  # [M_matched, 3]
    centers_t1 = boxes_matched_t1[:, :3]
    displacement = centers_t1 - centers_t0  # [M_matched, 3] in meters

    # Compute speed (magnitude / time)
    speed_3d = torch.norm(displacement, dim=1) / delta_t  # [M_matched] in m/s

    # Compute 2D BEV direction (ignoring z)
    direction_2d = torch.atan2(displacement[:, 1], displacement[:, 0])  # [M_matched] radians

    # Binary moving flag (threshold = 0.5 m/s ≈ 1.8 km/h)
    # NOTE: This is AFTER ego-motion compensation, so only object motion remains
    is_moving = (speed_3d > 0.5).float()  # [M_matched]

    return {
        'displacement': displacement,
        'direction_2d': direction_2d,
        'speed': speed_3d,
        'is_moving': is_moving,
        'matched_indices_t0': indices_t0,
        'matched_indices_t1': indices_t1,
    }


def is_point_in_box(points, box):
    """
    Check if points are inside oriented 3D box.

    Args:
        points: [N, 3] point cloud
        box: [7] (x, y, z, l, w, h, yaw)

    Returns:
        mask: [N] boolean mask
    """
    cx, cy, cz, l, w, h, yaw = box

    # Transform points to box-centric coordinates
    points_centered = points - box[:3].unsqueeze(0)  # [N, 3]

    # Rotate by -yaw (inverse rotation to align with box axes)
    cos_yaw = torch.cos(-yaw)
    sin_yaw = torch.sin(-yaw)

    x_rot = cos_yaw * points_centered[:, 0] - sin_yaw * points_centered[:, 1]
    y_rot = sin_yaw * points_centered[:, 0] + cos_yaw * points_centered[:, 1]
    z_rot = points_centered[:, 2]

    # Check if inside box bounds
    in_x = torch.abs(x_rot) < (l / 2)
    in_y = torch.abs(y_rot) < (w / 2)
    in_z = torch.abs(z_rot) < (h / 2)

    mask = in_x & in_y & in_z

    return mask


def motion_weighted_segmentation_loss(seg_pred, seg_gt, motion_features,
                                       points, boxes_t0, weight_moving=2.0,
                                       weight_static=1.0):
    """
    Segmentation loss weighted by object motion.

    Moving objects receive higher weight → model focuses on tracking-relevant regions.
    Static objects receive lower weight → less penalty for mistakes on background.

    Args:
        seg_pred: [B, N] predicted segmentation logits (before sigmoid)
        seg_gt: [B, N] GT segmentation labels (0=static, 1=moving)
        motion_features: Dict from compute_box_motion_features() or None
        points: [B, N, 3] point cloud
        boxes_t0: List[Tensor] boxes per batch [M_i, 7]
        weight_moving: float, weight for points in moving objects (default 2.0)
        weight_static: float, weight for other points (default 1.0)

    Returns:
        weighted_loss: Scalar loss
    """
    B, N = seg_pred.shape
    device = seg_pred.device

    # Base BCE loss (no reduction yet)
    bce_loss = F.binary_cross_entropy_with_logits(
        seg_pred, seg_gt.float(), reduction='none'
    )  # [B, N]

    # Compute motion weights per point
    motion_weights = torch.ones(B, N, device=device) * weight_static

    if motion_features is not None:
        for b in range(B):
            # Get matched boxes for this batch
            matched_indices = motion_features['matched_indices_t0']
            is_moving = motion_features['is_moving']  # [M_matched]

            if len(matched_indices) == 0:
                continue

            # Assign points to boxes and weight by motion
            for idx, box_idx in enumerate(matched_indices):
                if b >= len(boxes_t0):
                    continue
                if box_idx >= len(boxes_t0[b]):
                    continue

                box = boxes_t0[b][box_idx]  # [7]

                # Find points inside this box
                in_box = is_point_in_box(points[b], box)  # [N]

                # Weight points based on box motion
                if is_moving[idx] > 0.5:
                    motion_weights[b, in_box] = weight_moving  # Moving: 2x weight
                else:
                    motion_weights[b, in_box] = weight_static  # Static: 1x weight

    # Apply weights and compute mean
    weighted_loss = (bce_loss * motion_weights).sum() / motion_weights.sum()

    return weighted_loss


class MotionPredictionHead(nn.Module):
    """
    Optional: Predict box-level motion from point features.

    This can be used to:
    1. Predict displacement/speed from appearance
    2. Regularize motion features
    3. Improve segmentation via motion awareness

    Architecture:
        Input: point features [B, C, N] + box assignment
        Output: motion predictions per box [M, 3] (displacement)
    """

    def __init__(self, feature_dim=128, hidden_dim=256):
        super().__init__()

        # Per-point motion features
        self.motion_conv = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # Box-level aggregation + prediction
        self.motion_fc = nn.Sequential(
            nn.Linear(hidden_dim + 7, 128),  # hidden_dim + box params
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # Predict (dx, dy, dz)
        )

    def forward(self, point_features, boxes, cluster_ids):
        """
        Args:
            point_features: [B, C, N] per-point features from backbone
            boxes: List[Tensor] boxes per batch [M_i, 7]
            cluster_ids: List[Tensor] cluster assignment [N_i]

        Returns:
            motion_pred: List[Tensor] predicted displacement per box [M_i, 3]
        """
        B = point_features.shape[0]
        motion_pred_batch = []

        for b in range(B):
            feats = point_features[b]  # [C, N]
            bxs = boxes[b]  # [M, 7]
            ids = cluster_ids[b]  # [N]

            if len(bxs) == 0:
                motion_pred_batch.append(torch.zeros(0, 3, device=feats.device))
                continue

            # Extract motion features per point
            motion_feats = self.motion_conv(feats.unsqueeze(0))  # [1, hidden_dim, N]
            motion_feats = motion_feats.squeeze(0)  # [hidden_dim, N]

            # Aggregate per box
            box_motions = []
            for box_id in range(len(bxs)):
                # Get points in this box
                mask = (ids == box_id)
                if mask.sum() == 0:
                    # No points - predict zero motion
                    box_motions.append(torch.zeros(3, device=feats.device))
                    continue

                # Max pool motion features
                box_feat = motion_feats[:, mask].max(dim=1)[0]  # [hidden_dim]

                # Concatenate with box geometry
                box_param = bxs[box_id]  # [7]
                combined = torch.cat([box_feat, box_param], dim=0)  # [hidden_dim + 7]

                # Predict motion
                motion = self.motion_fc(combined)  # [3]
                box_motions.append(motion)

            motion_pred_batch.append(torch.stack(box_motions))  # [M, 3]

        return motion_pred_batch


def motion_prediction_loss(motion_pred, motion_gt, valid_mask=None):
    """
    L1 loss for motion prediction.

    Args:
        motion_pred: List[Tensor] predicted displacement [M_i, 3]
        motion_gt: List[Tensor] GT displacement [M_i, 3]
        valid_mask: List[Tensor] validity mask [M_i] (optional)

    Returns:
        loss: scalar
    """
    total_loss = 0.0
    total_count = 0

    for b in range(len(motion_pred)):
        pred = motion_pred[b]  # [M, 3]
        gt = motion_gt[b]      # [M, 3]

        if len(pred) == 0 or len(gt) == 0:
            continue

        # Match lengths
        M = min(len(pred), len(gt))
        pred = pred[:M]
        gt = gt[:M]

        # Compute L1 loss
        loss = F.l1_loss(pred, gt, reduction='none')  # [M, 3]

        # Apply validity mask if provided
        if valid_mask is not None and b < len(valid_mask):
            mask = valid_mask[b][:M].unsqueeze(-1)  # [M, 1]
            loss = loss * mask

        total_loss += loss.sum()
        total_count += M * 3

    if total_count == 0:
        return torch.tensor(0.0, device=motion_pred[0].device if len(motion_pred[0]) > 0 else 'cuda')

    return total_loss / total_count


# =============================================================================
# INTEGRATION EXAMPLE
# =============================================================================
"""
# In trainer_simple.py, modify run_epoch_simple():

def run_epoch_simple(args, net, data_loader, logger, optimizer=None, ...):
    ...

    for batch_idx, batch in enumerate(data_loader):
        pc1 = batch['pc1']  # [B, N, 3]
        pc2 = batch['pc2']  # [B, N, 3] (if available)
        seg_gt = batch['seg_gt']  # [B, N]

        # NEW: Get ego-motion transform
        ego_transform = batch.get('ego_transform', None)  # [B, 4, 4]

        # Forward pass
        pred, features = net(pc1)

        # === MOTION IMPROVEMENTS ===
        if epoch >= args.pretrain_epochs and pc2 is not None and ego_transform is not None:
            # 1. Compensate ego-motion
            from utils.motion_improved import compensate_ego_motion

            pc1_in_t1_frame = compensate_ego_motion(pc1, ego_transform)

            # 2. Compute box-level motion
            if 'boxes_gt' in batch and 'track_ids_gt' in batch:
                from utils.motion_improved import compute_box_motion_features

                boxes_t0 = batch['boxes_gt']  # List of boxes at t
                boxes_t1 = batch['boxes_gt2']  # List of boxes at t+1
                track_ids_t0 = batch['track_ids_gt']
                track_ids_t1 = batch['track_ids_gt2']

                # Compute motion features per batch
                motion_features_list = []
                for b in range(len(boxes_t0)):
                    motion_feats = compute_box_motion_features(
                        boxes_t0[b], boxes_t1[b],
                        track_ids_t0[b], track_ids_t1[b]
                    )
                    motion_features_list.append(motion_feats)

                # 3. Motion-weighted segmentation loss
                from utils.motion_improved import motion_weighted_segmentation_loss

                loss_seg = motion_weighted_segmentation_loss(
                    seg_pred=pred,
                    seg_gt=seg_gt,
                    motion_features=motion_features_list[0],  # First batch
                    points=pc1,
                    boxes_t0=boxes_t0,
                    weight_moving=2.0,
                    weight_static=1.0
                )
            else:
                # Fallback: standard BCE loss
                loss_seg = F.binary_cross_entropy_with_logits(pred, seg_gt.float())
        else:
            # Pretrain: standard BCE loss
            loss_seg = F.binary_cross_entropy_with_logits(pred, seg_gt.float())

        ...
"""
