"""
Simplified trainer following baseline RaTrack approach.
Uses simple BCE loss and EPE loss like the original working baseline.

Now supports Re-ID tracking mode via train_mode flag.
"""

import os
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.models_utils import filter_object_points_batch, get_gt_flow_new_batch
from model.model import load_model
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import glob

# Re-ID module will be imported when needed (to avoid path issues)
from utils.detection_metrics import compute_detection_metrics_epoch, print_detection_metrics
from models.gallery_manager import GalleryManager

# Detection head imports (Phase 1)
from utils.detection_losses import compute_detection_loss
from utils.detection_utils import (
    extract_boxes_from_detection,
    visualize_detection_results,
    convert_o3d_boxes_to_tensor_list
)

# RGB projection visualization
from utils.visualization_rgb import plot_rgb_projection, load_rgb_for_frame


def save_metrics_to_csv(csv_path, epoch, metrics, losses, lr, mode='train'):
    """
    Save epoch metrics to CSV file (cumulative - appends each epoch).

    Args:
        csv_path: Path to CSV file
        epoch: Current epoch number
        metrics: Dict of metrics
        losses: Dict of losses
        lr: Current learning rate
        mode: 'train' or 'val'
    """
    csv_path = Path(csv_path)
    file_exists = csv_path.exists()

    # Prepare row data
    row = {
        'epoch': epoch,
        'lr': lr,
        'mode': mode,
    }

    # Add losses
    for key, value in losses.items():
        if isinstance(value, list) and len(value) > 0:
            row[f'loss_{key}'] = float(np.mean(value))
        elif isinstance(value, (int, float)):
            row[f'loss_{key}'] = float(value)

    # Add metrics
    for key, value in metrics.items():
        if isinstance(value, list) and len(value) > 0:
            row[key] = float(np.mean(value))
        elif isinstance(value, (int, float)):
            row[key] = float(value)

    # Write to CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted(row.keys()))

        # Write header only if file is new
        if not file_exists or csv_path.stat().st_size == 0:
            writer.writeheader()

        writer.writerow(row)


def plot_training_progress(checkpoint_dir, train_mode='segmentation_only'):
    """
    Generate training progress plots from CSV files.

    For tracking mode (reid_only):
        Creates 3x2 grid: Loss, MOTA, IDF1, sAMOTA, ID Switches, Box F1
    For segmentation mode:
        Creates 2x2 grid: Loss, mIoU, F1, Per-class IoU

    Args:
        checkpoint_dir: Path to checkpoint directory
        train_mode: Training mode ('segmentation_only' or 'reid_only')
    """
    try:
        checkpoint_dir = Path(checkpoint_dir)
        train_csv = checkpoint_dir / 'train_metrics.csv'
        val_csv = checkpoint_dir / 'val_metrics.csv'

        # Check if train CSV exists
        if not train_csv.exists():
            return

        # Load training data
        train_df = pd.read_csv(train_csv, on_bad_lines='skip')
        val_df = pd.read_csv(val_csv, on_bad_lines='skip') if val_csv.exists() else None

        # Detect if tracking metrics exist (auto-detect mode)
        is_tracking_mode = 'MOTA' in train_df.columns

        exp_name = checkpoint_dir.parent.name if checkpoint_dir.name == 'models' else checkpoint_dir.name
        current_epoch = train_df["epoch"].iloc[-1]

        if is_tracking_mode:
            # ===== TRACKING MODE: 3x2 Grid =====
            fig = plt.figure(figsize=(18, 10))

            # Row 1, Col 1: Total Loss
            ax1 = plt.subplot(2, 3, 1)
            ax1.plot(train_df['epoch'], train_df['loss_total'], 'o-',
                    label='Train', linewidth=2, markersize=5, color='#1f77b4')
            if val_df is not None and 'loss_total' in val_df.columns:
                ax1.plot(val_df['epoch'], val_df['loss_total'], 's-',
                        label='Val', linewidth=2, markersize=5, color='#ff7f0e')
            ax1.set_xlabel('Epoch', fontsize=11)
            ax1.set_ylabel('Loss', fontsize=11)
            ax1.set_title('Total Loss', fontsize=12, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)

            # Row 1, Col 2: MOTA (Multi-Object Tracking Accuracy)
            ax2 = plt.subplot(2, 3, 2)
            if 'MOTA' in train_df.columns:
                ax2.plot(train_df['epoch'], train_df['MOTA'], 'o-',
                        label='Train', linewidth=2, markersize=5, color='#2ca02c')
            if val_df is not None and 'MOTA' in val_df.columns:
                ax2.plot(val_df['epoch'], val_df['MOTA'], 's-',
                        label='Val', linewidth=2, markersize=5, color='#d62728')
            ax2.set_xlabel('Epoch', fontsize=11)
            ax2.set_ylabel('MOTA (%)', fontsize=11)
            ax2.set_title('MOTA ↑ (PRIMARY METRIC)', fontsize=12, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)

            # Row 1, Col 3: IDF1 (ID F1 Score)
            ax3 = plt.subplot(2, 3, 3)
            if 'IDF1' in train_df.columns:
                ax3.plot(train_df['epoch'], train_df['IDF1'], 'o-',
                        label='Train', linewidth=2, markersize=5, color='#9467bd')
            if val_df is not None and 'IDF1' in val_df.columns:
                ax3.plot(val_df['epoch'], val_df['IDF1'], 's-',
                        label='Val', linewidth=2, markersize=5, color='#e377c2')
            ax3.set_xlabel('Epoch', fontsize=11)
            ax3.set_ylabel('IDF1 (%)', fontsize=11)
            ax3.set_title('IDF1 ↑ (ID Consistency)', fontsize=12, fontweight='bold')
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3)

            # Row 2, Col 1: sAMOTA (Scaled Average MOTA)
            ax4 = plt.subplot(2, 3, 4)
            if 'sAMOTA' in train_df.columns:
                ax4.plot(train_df['epoch'], train_df['sAMOTA'], 'o-',
                        label='Train', linewidth=2, markersize=5, color='#8c564b')
            if val_df is not None and 'sAMOTA' in val_df.columns:
                ax4.plot(val_df['epoch'], val_df['sAMOTA'], 's-',
                        label='Val', linewidth=2, markersize=5, color='#ff9896')
            ax4.set_xlabel('Epoch', fontsize=11)
            ax4.set_ylabel('sAMOTA (%)', fontsize=11)
            ax4.set_title('sAMOTA ↑ (nuScenes Metric)', fontsize=12, fontweight='bold')
            ax4.legend(fontsize=10)
            ax4.grid(True, alpha=0.3)

            # Row 2, Col 2: ID Switches
            ax5 = plt.subplot(2, 3, 5)
            if 'ID_switches' in train_df.columns:
                ax5.plot(train_df['epoch'], train_df['ID_switches'], 'o-',
                        label='Train', linewidth=2, markersize=5, color='#e377c2')
            if val_df is not None and 'ID_switches' in val_df.columns:
                ax5.plot(val_df['epoch'], val_df['ID_switches'], 's-',
                        label='Val', linewidth=2, markersize=5, color='#d62728')
            ax5.set_xlabel('Epoch', fontsize=11)
            ax5.set_ylabel('Count', fontsize=11)
            ax5.set_title('ID Switches ↓ (Tracking Quality)', fontsize=12, fontweight='bold')
            ax5.legend(fontsize=10)
            ax5.grid(True, alpha=0.3)

            # Row 2, Col 3: Box F1 (Detection Quality)
            ax6 = plt.subplot(2, 3, 6)
            if 'box_f1' in train_df.columns:
                ax6.plot(train_df['epoch'], train_df['box_f1'], 'o-',
                        label='Train', linewidth=2, markersize=5, color='#17becf')
            if val_df is not None and 'box_f1' in val_df.columns:
                ax6.plot(val_df['epoch'], val_df['box_f1'], 's-',
                        label='Val', linewidth=2, markersize=5, color='#7f7f7f')
            ax6.set_xlabel('Epoch', fontsize=11)
            ax6.set_ylabel('F1 Score', fontsize=11)
            ax6.set_title('Box F1 ↑ (Detection Quality)', fontsize=12, fontweight='bold')
            ax6.legend(fontsize=10)
            ax6.grid(True, alpha=0.3)

            plt.suptitle(f'Tracking Training Progress: {exp_name} (Epoch {current_epoch})',
                        fontsize=16, fontweight='bold', y=0.995)

        else:
            # ===== SEGMENTATION MODE: 2x3 Grid (IMPROVED) =====
            fig = plt.figure(figsize=(18, 9))

            # Detect pretrain_epochs from data (when motion loss becomes non-zero)
            pretrain_epochs = None
            if 'loss_motion' in train_df.columns:
                motion_nonzero = train_df[train_df['loss_motion'] > 0.01]
                if len(motion_nonzero) > 0:
                    pretrain_epochs = motion_nonzero['epoch'].min()

            # 1. Segmentation Loss (IMPROVED: separate from total)
            ax1 = plt.subplot(2, 3, 1)
            if 'loss_seg' in train_df.columns:
                ax1.plot(train_df['epoch'], train_df['loss_seg'], 'o-',
                        label='Train', linewidth=2, markersize=4, color='#ff7f0e')
                if val_df is not None and 'loss_seg' in val_df.columns:
                    ax1.plot(val_df['epoch'], val_df['loss_seg'], 's-',
                            label='Val', linewidth=2, markersize=4, color='#d62728')
            else:
                # Fallback to loss_total if loss_seg not available
                ax1.plot(train_df['epoch'], train_df['loss_total'], 'o-',
                        label='Train', linewidth=2, markersize=4, color='#1f77b4')
                if val_df is not None and 'loss_total' in val_df.columns:
                    ax1.plot(val_df['epoch'], val_df['loss_total'], 's-',
                            label='Val', linewidth=2, markersize=4, color='#ff7f0e')

            # Add pretrain end line
            if pretrain_epochs is not None:
                ax1.axvline(x=pretrain_epochs, color='gray', linestyle='--',
                           linewidth=1.5, alpha=0.7, label=f'Pretrain End (E{pretrain_epochs})')

            ax1.set_xlabel('Epoch', fontsize=11)
            ax1.set_ylabel('Loss', fontsize=11)
            ax1.set_title('Segmentation Loss', fontsize=12, fontweight='bold')
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)

            # 2. Motion Loss (NEW: separate subplot)
            ax2 = plt.subplot(2, 3, 2)
            if 'loss_motion' in train_df.columns:
                ax2.plot(train_df['epoch'], train_df['loss_motion'], 'o-',
                        label='Train', linewidth=2, markersize=4, color='#d62728')
                if val_df is not None and 'loss_motion' in val_df.columns:
                    ax2.plot(val_df['epoch'], val_df['loss_motion'], 's-',
                            label='Val', linewidth=2, markersize=4, color='#ff9896')

                # Add pretrain end line
                if pretrain_epochs is not None:
                    ax2.axvline(x=pretrain_epochs, color='gray', linestyle='--',
                               linewidth=1.5, alpha=0.7, label=f'Pretrain End (E{pretrain_epochs})')

                ax2.set_xlabel('Epoch', fontsize=11)
                ax2.set_ylabel('Loss', fontsize=11)
                ax2.set_title('Motion Loss (Ego Flow)', fontsize=12, fontweight='bold')
                ax2.legend(fontsize=9)
                ax2.grid(True, alpha=0.3)
            else:
                # No motion loss available
                ax2.text(0.5, 0.5, 'Motion Loss\n(Not Available)',
                        ha='center', va='center', fontsize=14, color='gray',
                        transform=ax2.transAxes)
                ax2.set_title('Motion Loss', fontsize=12, fontweight='bold')
                ax2.grid(True, alpha=0.3)

            # 3. mIoU (Point-Level)
            ax3 = plt.subplot(2, 3, 3)
            if 'miou' in train_df.columns:
                ax3.plot(train_df['epoch'], train_df['miou'], 'o-',
                        label='Train', linewidth=2, markersize=4, color='#2ca02c')
                if val_df is not None and 'miou' in val_df.columns:
                    ax3.plot(val_df['epoch'], val_df['miou'], 's-',
                            label='Val', linewidth=2, markersize=4, color='#1f77b4')
            ax3.set_xlabel('Epoch', fontsize=11)
            ax3.set_ylabel('mIoU', fontsize=11)
            ax3.set_title('Mean IoU (Point-Level)', fontsize=12, fontweight='bold')
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3)

            # 5. F1 Score (Point-Level)
            ax5 = plt.subplot(2, 3, 5)
            if 'f1' in train_df.columns:
                ax5.plot(train_df['epoch'], train_df['f1'], 'o-',
                        label='Train', linewidth=2, markersize=4, color='#e377c2')
                if val_df is not None and 'f1' in val_df.columns:
                    ax5.plot(val_df['epoch'], val_df['f1'], 's-',
                            label='Val', linewidth=2, markersize=4, color='#7f7f7f')
            ax5.set_xlabel('Epoch', fontsize=11)
            ax5.set_ylabel('F1 Score', fontsize=11)
            ax5.set_title('F1 Score (Point-Level)', fontsize=12, fontweight='bold')
            ax5.legend(fontsize=10)
            ax5.grid(True, alpha=0.3)

            # 6. Per-class IoU (Point-Level)
            ax6 = plt.subplot(2, 3, 6)
            if 'IoU_moving' in train_df.columns:
                ax6.plot(train_df['epoch'], train_df['IoU_moving'], 'o-',
                        label='Train Moving', linewidth=2, markersize=4, color='#ff9896')
                if val_df is not None and 'IoU_moving' in val_df.columns:
                    ax6.plot(val_df['epoch'], val_df['IoU_moving'], 's-',
                            label='Val Moving', linewidth=2, markersize=4, color='#d62728')
            if 'IoU_static' in train_df.columns:
                ax6.plot(train_df['epoch'], train_df['IoU_static'], 'o--',
                        label='Train Static', linewidth=1.5, markersize=3, color='#aec7e8')
                if val_df is not None and 'IoU_static' in val_df.columns:
                    ax6.plot(val_df['epoch'], val_df['IoU_static'], 's--',
                            label='Val Static', linewidth=1.5, markersize=3, color='#1f77b4')
            ax6.set_xlabel('Epoch', fontsize=11)
            ax6.set_ylabel('IoU (%)', fontsize=11)
            ax6.set_title('IoU per Class (Point-Level)', fontsize=12, fontweight='bold')
            ax6.legend(fontsize=9)
            ax6.grid(True, alpha=0.3)

            plt.suptitle(f'Training Progress: {exp_name} (Epoch {current_epoch})',
                        fontsize=14, fontweight='bold', y=0.995)

        # Save figure
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plots_dir = checkpoint_dir.parent if checkpoint_dir.name == 'models' else checkpoint_dir
        output_path = plots_dir / 'training_progress.png'
        plt.savefig(output_path, dpi=120, bbox_inches='tight')
        plt.close(fig)

    except Exception as e:
        # Don't crash training if plotting fails
        import traceback
        print(f"Warning: Failed to generate training plot: {e}")
        print(f"Traceback: {traceback.format_exc()}")


def get_cartesian_res(pc, sensor):
    """
    Calculate cartesian resolution for radar or lidar sensor.
    Following baseline RaTrack implementation (main_utils.py:272-309).

    Args:
        pc: [B, N, 3] point cloud in numpy
        sensor: 'radar' or 'lidar'

    Returns:
        xyz_res: [B, N, 3] resolution in x, y, z dimensions
    """
    # Sensor resolution parameters
    if sensor == 'radar':  # LRR30
        r_res = 0.2  # m
        theta_res = 1 * np.pi / 180  # radian
        phi_res = 1.6 * np.pi / 180  # radian
    elif sensor == 'lidar':  # HDL-64E
        r_res = 0.04  # m
        theta_res = 0.4 * np.pi / 180  # radian
        phi_res = 0.08 * np.pi / 180  # radian
    else:
        raise ValueError(f"Unknown sensor: {sensor}")

    res = np.array([r_res, theta_res, phi_res])

    # Extract x, y, z coordinates
    x = pc[:, :, 0]
    y = pc[:, :, 1]
    z = pc[:, :, 2]

    # Convert from xyz to spherical (r/theta/phi)
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arcsin(z / (r + 1e-20))
    phi = np.arctan2(y, x)

    # Compute xyz's gradient about r/theta/phi
    grad_x = np.stack((
        np.cos(phi) * np.cos(theta),
        -r * np.sin(theta) * np.cos(phi),
        -r * np.cos(theta) * np.sin(phi)
    ), axis=2)

    grad_y = np.stack((
        np.sin(phi) * np.cos(theta),
        -r * np.sin(phi) * np.sin(theta),
        r * np.cos(theta) * np.cos(phi)
    ), axis=2)

    grad_z = np.stack((
        np.sin(theta),
        r * np.cos(theta),
        np.zeros((np.size(x, 0), np.size(x, 1)))
    ), axis=2)

    # Measure resolution for xyz (different positions have different resolution)
    x_res = np.sum(np.abs(grad_x) * res, axis=2)
    y_res = np.sum(np.abs(grad_y) * res, axis=2)
    z_res = np.sum(np.abs(grad_z) * res, axis=2)

    xyz_res = np.stack((x_res, y_res, z_res), axis=2)

    return xyz_res


def motion_seg_loss_baseline(pred_cls, gt_cls):
    """
    Baseline segmentation loss from RaTrack.
    Separates moving (40%) and static (60%) points.

    Args:
        pred_cls: [B, 1, N] predicted probabilities
        gt_cls: [B, N] ground truth binary labels

    Returns:
        Weighted BCE loss
    """
    # Create masks for moving and static points
    gt_cls_bool = gt_cls.bool()
    true_mask = gt_cls_bool  # Moving points
    false_mask = ~gt_cls_bool  # Static points

    # Flatten for BCE
    pred_cls = pred_cls.squeeze(1)  # [B, N]

    # Avoid empty masks
    if true_mask.sum() == 0:
        loss_pos = torch.tensor(0.0).cuda()
    else:
        loss_pos = F.binary_cross_entropy(pred_cls[true_mask], gt_cls[true_mask].float())

    if false_mask.sum() == 0:
        loss_neg = torch.tensor(0.0).cuda()
    else:
        loss_neg = F.binary_cross_entropy(pred_cls[false_mask], gt_cls[false_mask].float())

    # Baseline weights: 40% moving, 60% static
    return 0.4 * loss_pos + 0.6 * loss_neg


def soft_dice_loss(pred_cls, gt_cls, smooth=1e-6):
    """
    Soft Dice loss for binary segmentation.
    Directly optimizes Dice coefficient (overlap), complementary to Focal/BCE
    for class-imbalanced data like radar point clouds.

    Args:
        pred_cls: [B, 1, N] predicted probabilities (sigmoid output)
        gt_cls: [B, N] ground truth binary labels

    Returns:
        Soft Dice loss: 1 - Dice coefficient (scalar)
    """
    pred = pred_cls.squeeze(1)  # [B, N]
    gt = gt_cls.float()

    intersection = (pred * gt).sum(dim=-1)        # [B]
    union = pred.sum(dim=-1) + gt.sum(dim=-1)     # [B]

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return (1.0 - dice).mean()


def feature_contrast_loss(features, gt_cls):
    """
    Feature contrast loss: push moving and static feature centroids apart.
    Computes cosine similarity between mean moving and mean static feature
    vectors, encouraging inter-class separation in appearance space.

    Args:
        features: [B, C, N] per-point features from backbone (pc1_features)
        gt_cls:   [B, N] ground truth binary labels (1=moving, 0=static)

    Returns:
        Contrast loss (scalar, in [0, 1])
    """
    total_loss = 0.0
    valid_batches = 0

    for b in range(features.shape[0]):
        feat_b = features[b]           # [C, N]
        moving_mask = gt_cls[b].bool() # [N]
        static_mask = ~moving_mask

        if moving_mask.sum() == 0 or static_mask.sum() == 0:
            continue

        feat_moving = feat_b[:, moving_mask].mean(dim=1)   # [C]
        feat_static = feat_b[:, static_mask].mean(dim=1)   # [C]

        # L2-normalize centroids for cosine similarity
        feat_moving = F.normalize(feat_moving, dim=0)
        feat_static = F.normalize(feat_static, dim=0)

        # Cosine similarity: +1 = same direction, -1 = opposite
        # Want to push apart → minimize cosine_sim → loss = (1 + cosine_sim) / 2
        cosine_sim = (feat_moving * feat_static).sum()
        total_loss += (1.0 + cosine_sim) / 2.0
        valid_batches += 1

    if valid_batches == 0:
        return features.sum() * 0.0  # stays in computation graph

    return total_loss / valid_batches


def object_aware_seg_loss(pred_cls, gt_cls, obj_ids):
    """
    Object-aware segmentation loss.
    Computes loss separately for each object instance and weights by object size.
    Small objects get higher weight (harder to segment).

    Args:
        pred_cls: [B, 1, N] predicted probabilities
        gt_cls: [B, N] ground truth binary labels
        obj_ids: [B, N] object IDs for each point (0 = background/static)

    Returns:
        Weighted BCE loss per object
    """
    batch_size = pred_cls.shape[0]
    pred_cls = pred_cls.squeeze(1)  # [B, N]

    total_loss = 0.0
    total_objects = 0

    for b in range(batch_size):
        pred_b = pred_cls[b]  # [N]
        gt_b = gt_cls[b]  # [N]
        obj_ids_b = obj_ids[b]  # [N]

        # Get unique object IDs (excluding background = 0)
        unique_ids = torch.unique(obj_ids_b)
        unique_ids = unique_ids[unique_ids > 0]  # Remove background

        if len(unique_ids) == 0:
            # No objects, use standard BCE on all points
            loss_b = F.binary_cross_entropy(pred_b, gt_b.float())
            total_loss += loss_b
            total_objects += 1
            continue

        # Compute loss per object with size-based weighting
        for obj_id in unique_ids:
            obj_mask = (obj_ids_b == obj_id)
            obj_size = obj_mask.sum().float()

            if obj_size == 0:
                continue

            # Extract predictions and labels for this object
            obj_pred = pred_b[obj_mask]
            obj_gt = gt_b[obj_mask].float()

            # Compute BCE for this object
            obj_loss = F.binary_cross_entropy(obj_pred, obj_gt)

            # Weight: smaller objects get higher weight (inversely proportional to size)
            # Normalize by sqrt to avoid extreme weights
            weight = 1.0 / torch.sqrt(obj_size)

            total_loss += weight * obj_loss
            total_objects += 1

    # Average over all objects in batch
    if total_objects == 0:
        return torch.tensor(0.0).cuda()

    return total_loss / total_objects


def instance_consistency_loss(pred_cls, obj_ids, margin=0.1):
    """
    Instance consistency loss.
    Encourages points from the same object to have similar predictions.
    Penalizes variance within each object instance.

    Args:
        pred_cls: [B, 1, N] predicted probabilities
        obj_ids: [B, N] object IDs for each point (0 = background/static)
        margin: margin for consistency (points can vary within this margin)

    Returns:
        Consistency loss (variance within objects)
    """
    batch_size = pred_cls.shape[0]
    pred_cls = pred_cls.squeeze(1)  # [B, N]

    total_loss = 0.0
    total_objects = 0

    for b in range(batch_size):
        pred_b = pred_cls[b]  # [N]
        obj_ids_b = obj_ids[b]  # [N]

        # Get unique object IDs (excluding background = 0)
        unique_ids = torch.unique(obj_ids_b)
        unique_ids = unique_ids[unique_ids > 0]  # Remove background

        if len(unique_ids) == 0:
            continue

        # Compute consistency loss per object
        for obj_id in unique_ids:
            obj_mask = (obj_ids_b == obj_id)
            obj_size = obj_mask.sum().float()

            if obj_size <= 1:  # Need at least 2 points for consistency
                continue

            # Extract predictions for this object
            obj_pred = pred_b[obj_mask]

            # Compute mean prediction for this object
            mean_pred = obj_pred.mean()

            # Compute variance (with margin)
            # Only penalize deviations larger than margin
            deviations = torch.abs(obj_pred - mean_pred)
            deviations_clipped = torch.clamp(deviations - margin, min=0.0)

            consistency = (deviations_clipped ** 2).mean()

            total_loss += consistency
            total_objects += 1

    # Average over all objects in batch
    if total_objects == 0:
        return torch.tensor(0.0).cuda()

    return total_loss / total_objects


def scene_flow_loss_baseline(pc1_wrap, gt_flow):
    """
    Baseline scene flow loss from RaTrack.
    Simple EPE (End-Point Error).

    Args:
        pc1_wrap: [B, 3, N] warped point cloud (pc1 + predicted_flow)
        gt_flow: [B, 3, N] ground truth flow

    Returns:
        Mean EPE across all points
    """
    # EPE: sqrt(sum((pred - gt)^2, dim=1))
    epe_per_point = ((pc1_wrap - gt_flow).pow(2).sum(dim=1)).sqrt()  # [B, N]
    epe_mean = torch.mean(epe_per_point, dim=1)  # [B]
    return epe_mean[0] if epe_mean.shape[0] > 0 else epe_mean.mean()


def compute_reid_metrics(pred_boxes, gt_boxes, embeddings=None):
    """
    Compute Re-ID detection metrics (frame-by-frame).

    NOTE: These are DETECTION metrics, NOT tracking metrics.
    For tracking metrics (MOTA, IDF1, etc.), use compute_mot_metrics()
    which requires temporal association across frames.

    Args:
        pred_boxes: List[Tensor] predicted boxes per batch [M_i, 7]
        gt_boxes: List[dict] GT boxes (Open3D OrientedBoundingBox dicts)
        embeddings: List[Tensor] Re-ID embeddings (optional)

    Returns:
        metrics: Dict with detection metrics (precision, recall, IoU, F1)
    """
    from models.reid_module import convert_o3d_boxes_to_tensor

    total_pred_boxes = 0
    total_gt_boxes = 0
    total_matched = 0
    sum_iou = 0.0

    for b in range(len(pred_boxes)):
        pred_b = pred_boxes[b]
        gt_b_dict = gt_boxes[b]

        # Convert GT boxes to tensor
        gt_b, _ = convert_o3d_boxes_to_tensor(gt_b_dict)

        total_pred_boxes += len(pred_b)
        total_gt_boxes += len(gt_b)

        if len(pred_b) == 0 or len(gt_b) == 0:
            continue

        # Compute 2D IoU on bird's eye view
        # Simple version: use center distance as proxy
        if len(pred_b) > 0 and len(gt_b) > 0:
            pred_centers = pred_b[:, :2].cpu().numpy()  # [M, 2] x,y
            gt_centers = gt_b[:, :2].cpu().numpy()      # [N, 2] x,y

            # Greedy matching: match closest boxes
            import numpy as np
            from scipy.spatial.distance import cdist
            if len(pred_centers) > 0 and len(gt_centers) > 0:
                dist_matrix = cdist(pred_centers, gt_centers)
                for i in range(len(pred_centers)):
                    if len(gt_centers) == 0:
                        break
                    min_idx = dist_matrix[i].argmin()
                    min_dist = dist_matrix[i, min_idx]

                    # Match if distance < 3 meters
                    if min_dist < 3.0:
                        total_matched += 1
                        # Estimate IoU from distance (rough approximation)
                        iou_est = max(0, 1.0 - min_dist / 3.0)
                        sum_iou += iou_est

    # Compute detection metrics
    precision = total_matched / max(total_pred_boxes, 1)
    recall = total_matched / max(total_gt_boxes, 1)
    avg_iou = sum_iou / max(total_matched, 1)

    # F1 score (harmonic mean of precision and recall)
    f1 = 2 * (precision * recall) / max(precision + recall, 1e-6)

    return {
        'box_precision': precision,
        'box_recall': recall,
        'box_f1': f1,
        'box_iou': avg_iou,
        'num_pred_boxes': total_pred_boxes / max(len(pred_boxes), 1),
        'num_gt_boxes': total_gt_boxes / max(len(gt_boxes), 1),
    }


def compute_metrics_simple(pred_flow, gt_flow, pred_seg, gt_seg, pc1=None):
    """
    Simple metrics computation following baseline RaTrack approach.

    Baseline metrics from main_utils.py:342-389:
    - RNE (Resolution-Normalized Error): Primary metric
    - SAS (Strict Accuracy Score): % points with rn_error <= 0.10
    - RAS (Relaxed Accuracy Score): % points with rn_error <= 0.20
    - EPE (End-Point Error): Secondary metric
    - acc (accuracy): (TP + TN) / (TP + TN + FP + FN)
    - miou: 0.5 * (IoU_moving + IoU_static)
    - sen (sensitivity/recall): TP / (TP + FN)

    Args:
        pred_flow: [B, 3, N] predicted flow
        gt_flow: [B, 3, N] ground truth flow
        pred_seg: [B, 1, N] predicted segmentation probabilities
        gt_seg: [B, N] ground truth segmentation
        pc1: [B, 3, N] point cloud (needed for RNE calculation)

    Returns:
        Dictionary with metrics (matching baseline format)
    """
    metrics = {}

    # Flow metrics: only computed when gt_flow is available (skipped during pretrain)
    if gt_flow is not None and pred_flow is not None:
        pred_flow_np = pred_flow.cpu().detach().numpy()
        gt_flow_np = gt_flow.cpu().detach().numpy()

        # EPE (secondary metric)
        error = np.sqrt(np.sum((pred_flow_np - gt_flow_np) ** 2, 1) + 1e-20)  # [B, N]
        metrics['EPE'] = np.mean(error)

        # RNE, SAS, RAS (primary metrics - following baseline)
        if pc1 is not None:
            pc1_np = pc1.cpu().detach().numpy().transpose(0, 2, 1)  # [B, 3, N] -> [B, N, 3]
            xyz_res_r = get_cartesian_res(pc1_np, 'radar')
            res_r = np.sqrt(np.sum(xyz_res_r ** 2, 2) + 1e-20)
            xyz_res_l = get_cartesian_res(pc1_np, 'lidar')
            res_l = np.sqrt(np.sum(xyz_res_l ** 2, 2) + 1e-20)
            rn_error = error / (res_r / res_l + 1e-20)
            metrics['RNE'] = np.mean(rn_error)
            gtflow_len = np.sqrt(np.sum(gt_flow_np * gt_flow_np, 1) + 1e-20)
            sas = np.sum(np.logical_or((rn_error <= 0.10), (rn_error / gtflow_len <= 0.10))) / (
                    np.size(pred_flow_np, 0) * np.size(pred_flow_np, 2))
            metrics['SAS'] = sas
            ras = np.sum(np.logical_or((rn_error <= 0.20), (rn_error / gtflow_len <= 0.20))) / (
                    np.size(pred_flow_np, 0) * np.size(pred_flow_np, 2))
            metrics['RAS'] = ras
        else:
            metrics['RNE'] = 0.0
            metrics['SAS'] = 0.0
            metrics['RAS'] = 0.0
    else:
        # Pretrain mode or no flow: set flow metrics to 0
        metrics['RNE'] = 0.0
        metrics['SAS'] = 0.0
        metrics['RAS'] = 0.0
        metrics['EPE'] = 0.0

    # Segmentation metrics (following baseline exactly)
    if pred_seg is not None and gt_seg is not None:
        pred_binary = (pred_seg.squeeze(1) > 0.5).float()  # [B, N]
        gt_binary = gt_seg.float()  # [B, N]

        TP = torch.sum((pred_binary == 1) & (gt_binary == 1)).float() + 1e-20
        FP = torch.sum((pred_binary == 1) & (gt_binary == 0)).float() + 1e-20
        FN = torch.sum((pred_binary == 0) & (gt_binary == 1)).float() + 1e-20
        TN = torch.sum((pred_binary == 0) & (gt_binary == 0)).float() + 1e-20

        # Baseline metrics (exact same formulas)
        # Accuracy
        metrics['acc'] = ((TP + TN) / (TP + TN + FP + FN)).item()

        # mIoU (mean IoU over both classes)
        iou_moving = (TP / (TP + FP + FN)).item()
        iou_static = (TN / (TN + FP + FN)).item()
        metrics['miou'] = 0.5 * (iou_moving + iou_static)

        # Sensitivity (recall for moving class)
        metrics['sen'] = (TP / (TP + FN)).item()

        # F1 score (point-level, following same approach as mIoU)
        precision = (TP / (TP + FP)).item()
        recall = (TP / (TP + FN)).item()
        metrics['f1'] = (2 * precision * recall / (precision + recall + 1e-20))

        # Additional metrics (for reference)
        metrics['IoU_moving'] = iou_moving * 100  # As percentage
        metrics['IoU_static'] = iou_static * 100  # As percentage
    else:
        metrics['acc'] = 0.0
        metrics['miou'] = 0.0
        metrics['sen'] = 0.0
        metrics['f1'] = 0.0
        metrics['IoU_moving'] = 0.0
        metrics['IoU_static'] = 0.0

    return metrics


def sample_points(pc_list, num_points):
    """
    Sample/pad point clouds to a fixed number of points.

    Args:
        pc_list: List of numpy arrays [N_i, C] with variable N_i
        num_points: Target number of points

    Returns:
        Tensor of shape [batch_size, num_points, C]
    """
    batch_size = len(pc_list)
    num_channels = pc_list[0].shape[1]
    sampled_batch = np.zeros((batch_size, num_points, num_channels), dtype=np.float32)

    for i, pc in enumerate(pc_list):
        n_points = pc.shape[0]
        if n_points >= num_points:
            # Subsample
            indices = np.random.choice(n_points, num_points, replace=False)
        else:
            # Oversample
            indices = np.random.choice(n_points, num_points, replace=True)

        sampled_batch[i] = pc[indices]

    return torch.from_numpy(sampled_batch).float()


def create_video_from_frames(frames_dir, output_path, fps=2, logger=None):
    """
    Create a video from saved visualization frames.

    Args:
        frames_dir: Directory containing frame_XXXX.png files
        output_path: Path to save output video (e.g., 'video.mp4')
        fps: Frames per second (default=2 for slow playback)
        logger: Logger instance (optional)

    Returns:
        bool: True if video was created successfully, False otherwise
    """
    try:
        # Get all frame files sorted by number
        frame_pattern = os.path.join(frames_dir, 'frame_*.png')
        frame_files = sorted(glob.glob(frame_pattern))

        if len(frame_files) == 0:
            if logger:
                logger.warning(f'No frames found in {frames_dir}')
            return False

        if logger:
            logger.info(f'🎬 Creating video from {len(frame_files)} frames...')
            logger.info(f'   FPS: {fps} (slow playback)')

        # Read first frame to get dimensions
        first_frame = cv2.imread(frame_files[0])
        if first_frame is None:
            if logger:
                logger.error(f'❌ Could not read first frame: {frame_files[0]}')
            return False

        height, width, _ = first_frame.shape

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Write all frames
        for frame_file in frame_files:
            frame = cv2.imread(frame_file)
            if frame is not None:
                video_writer.write(frame)

        video_writer.release()

        if logger:
            logger.info(f'Video saved to: {output_path}')
            logger.info(f'   Duration: {len(frame_files)/fps:.1f} seconds at {fps} FPS')

        return True

    except Exception as e:
        if logger:
            logger.error(f'❌ Failed to create video: {e}')
            import traceback
            logger.error(traceback.format_exc())
        return False


def run_epoch_simple(args, net, train_loader, logger, optimizer, mode='train', ep_num=0, pretrain=False,
                     save_visualizations=False, vis_dir=None, reid_module=None, train_mode='segmentation_only'):
    """
    Simplified training loop following baseline approach.
    Now supports Re-ID tracking mode.

    Args:
        args: Configuration arguments
        net: Neural network model
        train_loader: DataLoader for training data
        logger: Logger for tracking progress
        optimizer: Optimizer for training
        mode: 'train' or 'val'
        ep_num: Current epoch number
        pretrain: If True, only train segmentation (like baseline)
        save_visualizations: If True, save visualizations every 50 batches
        vis_dir: Directory to save visualizations
    """
    progess_bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))

    # ===== DETECT DETECTION HEAD MODE (PHASE 1) =====
    use_detection_head = getattr(args, 'use_detection_head', False)
    is_detection_phase = use_detection_head and train_mode == 'reid_only'

    if is_detection_phase:
        logger.info('🎯 DETECTION TRAINING MODE (Phase 1)')
        logger.info('   → Metrics: detection_loss, mAP, box_IoU')
        logger.info('   → Visualization: Boxes pred vs GT only')
    elif train_mode == 'reid_only':
        logger.info('🔍 RE-ID TRAINING MODE (Phase 2)')
        logger.info('   → Metrics: MOTA, IDF1, sAMOTA, triplet_loss')
        logger.info('   → Visualization: Track IDs')

    # Initialize hidden state for temporal GRU (if enabled)
    # Will be reset on new sequences
    hidden_state = None

    # Loss structure: 'total', 'motion' (reid loss in reid_only), 'seg', 'dice', 'feat'
    # Add detection losses for Phase 1
    if is_detection_phase:
        epoch_losses = {
            'total': [], 'motion': [], 'seg': [], 'dice': [], 'feat': [],
            'detection_total': [], 'detection_center': [], 'detection_size': [],
            'detection_orientation': [], 'detection_class': []
        }
    else:
        epoch_losses = {'total': [], 'motion': [], 'seg': [], 'dice': [], 'feat': []}

    # Accumulate boxes for detection metrics (mAP, mIoU)
    boxes_pred_accumulated = []
    boxes_gt_accumulated = []
    seg_pred_accumulated = []
    seg_gt_accumulated = []

    # Initialize metrics based on training mode
    if train_mode == 'reid_only':
        epoch_metrics = {
            'box_precision': [], 'box_recall': [], 'box_f1': [], 'box_iou': [],  # Detection metrics
            'num_pred_boxes': [], 'num_gt_boxes': [],  # Box counts
            # Flow metrics for verification (GT flow → RNE ~0)
            'RNE': [], 'EPE': [], 'SAS': [], 'RAS': [],
            # Classification metrics
            'classification_acc': [],  # Classification accuracy (vehicle/pedestrian/cyclist)
            # NOTE: For tracking metrics (MOTA, IDF1), we need temporal association
            # which will be implemented in compute_mot_metrics()
        }
        # Initialize tracker and MOT accumulator for reid_only mode (both train and val)
        tracker = None
        mot_accumulator = None
        if train_mode == 'reid_only':
            try:
                from utils.mot_metrics import MOTMetricsAccumulator
                # Use tracker config from yaml (args is a dict-like object)
                track_cfg = getattr(args, 'tracking', {}) if hasattr(args, 'tracking') else {}
                max_age = track_cfg.get('max_age', 5) if isinstance(track_cfg, dict) else 5
                min_hits = track_cfg.get('min_hits', 1) if isinstance(track_cfg, dict) else 1
                iou_thresh = track_cfg.get('iou_threshold', 0.3) if isinstance(track_cfg, dict) else 0.3
                use_enhanced = track_cfg.get('use_enhanced_tracker', False) if isinstance(track_cfg, dict) else False
                matching_thresh = track_cfg.get('matching_threshold', 0.3) if isinstance(track_cfg, dict) else 0.3

                # Choose tracker based on config
                use_gallery = track_cfg.get('use_gallery_tracker', False) if isinstance(track_cfg, dict) else False

                if use_gallery:
                    # Use GalleryTracker with multi-cue + spatial configuration
                    from models.gallery_tracker import GalleryTracker
                    tracker = GalleryTracker(
                        max_age=max_age,
                        min_hits=min_hits,
                        matching_threshold=matching_thresh,
                        weight_appearance=track_cfg.get('appearance_weight', 0.30) if isinstance(track_cfg, dict) else 0.30,
                        weight_geometry=track_cfg.get('geometry_weight', 0.20) if isinstance(track_cfg, dict) else 0.20,
                        weight_density=track_cfg.get('density_weight', 0.10) if isinstance(track_cfg, dict) else 0.10,
                        weight_motion=track_cfg.get('motion_weight', 0.20) if isinstance(track_cfg, dict) else 0.20,
                        weight_spatial=track_cfg.get('spatial_weight', 0.20) if isinstance(track_cfg, dict) else 0.20,
                    )
                    if mode == 'train':
                        logger.info(f'🎯 Initialized GalleryTracker for TRAINING (max_age={max_age}, min_hits={min_hits}, matching_threshold={matching_thresh})')
                        logger.info(f'   → Multi-cue matching: appearance + motion + geometry + density + spatial')
                        logger.info(f'   → Spatial configuration: distances + ordering (left/right, front/back)')
                        logger.info(f'   → Memory persistence: {max_age} frames')
                        logger.info(f'   → MOTA/IDF1/sAMOTA will be computed every epoch')
                    else:
                        logger.info(f'🎯 Initialized GalleryTracker for VALIDATION (max_age={max_age}, min_hits={min_hits}, matching_threshold={matching_thresh})')

                elif use_enhanced:
                    # Use EnhancedTracker with motion-based matching
                    from models.enhanced_tracker import EnhancedTracker
                    tracker = EnhancedTracker(max_age=max_age, min_hits=min_hits, matching_threshold=matching_thresh)
                    if mode == 'train':
                        logger.info(f'🎯 Initialized EnhancedTracker for TRAINING (max_age={max_age}, min_hits={min_hits}, matching_threshold={matching_thresh})')
                        logger.info(f'   → Multi-cue matching: appearance + motion + geometry + size')
                        logger.info(f'   → Memory persistence: {max_age} frames')
                        logger.info(f'   → MOTA/IDF1/sAMOTA will be computed every epoch')
                    else:
                        logger.info(f'🎯 Initialized EnhancedTracker for VALIDATION (max_age={max_age}, min_hits={min_hits}, matching_threshold={matching_thresh})')
                else:
                    # Use SimpleTracker (legacy)
                    from models.simple_tracker import SimpleTracker
                    tracker = SimpleTracker(max_age=max_age, min_hits=min_hits, iou_threshold=iou_thresh)
                    if mode == 'train':
                        logger.info(f'🎯 Initialized SimpleTracker for TRAINING (max_age={max_age}, min_hits={min_hits}, iou={iou_thresh})')
                        logger.info(f'   → MOTA/IDF1/sAMOTA will be computed every epoch')
                    else:
                        logger.info(f'🎯 Initialized SimpleTracker for VALIDATION (max_age={max_age}, min_hits={min_hits}, iou={iou_thresh})')

                mot_accumulator = MOTMetricsAccumulator()
            except Exception as e:
                import traceback
                logger.error(f'❌ Failed to load tracking modules: {e}')
                logger.error(f'Traceback: {traceback.format_exc()}')
                logger.warning('MOT metrics will not be available - continuing without tracking')
    else:
        epoch_metrics = {
            'RNE': [], 'SAS': [], 'RAS': [], 'EPE': [],  # Scene flow metrics (RNE is primary)
            'acc': [], 'miou': [], 'sen': [], 'f1': [],  # Segmentation metrics (point-level)
            'IoU_moving': [], 'IoU_static': [],  # Detailed IoU
            # Instance-level metrics (NEW - for object detection evaluation)
            'box_precision': [], 'box_recall': [], 'box_f1': [], 'box_iou': [],
            'num_pred_boxes': [], 'num_gt_boxes': []
        }
        tracker = None
        mot_accumulator = None

    # ===== INITIALIZE IMPROVED BOX PROPOSAL (ONLY for EVALUATION) =====
    improved_box_proposal = None
    use_improved_boxes = (
        mode == 'val' and  # ← ONLY in validation/evaluation mode
        train_mode == 'reid_only' and
        not getattr(args, 'use_gt_boxes', True)
    )

    if use_improved_boxes:
        try:
            # Get config from args
            box_config = getattr(args, 'box_proposal', {})
            if isinstance(box_config, dict):
                detection_mode = box_config.get('detection_mode', 'segmentation_matcher')
                min_points = box_config.get('min_points_per_box', 5)
                min_cluster_size = box_config.get('min_cluster_size', 5)
                min_samples = box_config.get('min_samples', 3)
                moving_threshold = box_config.get('moving_threshold', 0.3)
                motion_threshold = box_config.get('motion_threshold', 0.1)
                use_hdbscan = box_config.get('use_hdbscan', True)
            else:
                detection_mode = getattr(box_config, 'detection_mode', 'segmentation_matcher')
                min_points = getattr(box_config, 'min_points_per_box', 5)
                min_cluster_size = getattr(box_config, 'min_cluster_size', 5)
                min_samples = getattr(box_config, 'min_samples', 3)
                moving_threshold = getattr(box_config, 'moving_threshold', 0.3)
                motion_threshold = getattr(box_config, 'motion_threshold', 0.1)
                use_hdbscan = getattr(box_config, 'use_hdbscan', True)

            if detection_mode == 'temporal_motion':
                # Full inference mode: Motion + Clustering
                from models.temporal_motion_detector import TemporalMotionDetector
                improved_box_proposal = TemporalMotionDetector(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    moving_threshold=moving_threshold,
                    motion_threshold=motion_threshold,
                    use_hdbscan=use_hdbscan
                )
                logger.info('🔧 TemporalMotionDetector initialized (FULL INFERENCE)')
                logger.info(f'   → Clustering: {"HDBSCAN" if use_hdbscan else "DBSCAN"}')
                logger.info(f'   → Min cluster size: {min_cluster_size}')
                logger.info(f'   → Motion threshold: {motion_threshold} m/s')
                logger.info(f'   → Generates boxes from segmentation + motion + clustering')
            else:
                # Segmentation matcher mode: GT boxes + segmentation filter
                from models.segmentation_gt_matcher import SegmentationGTMatcher
                improved_box_proposal = SegmentationGTMatcher(
                    min_points_per_box=min_points,
                    moving_threshold=moving_threshold
                )
                logger.info('🔧 SegmentationGTMatcher initialized')
                logger.info(f'   → Min points per box: {min_points}')
                logger.info(f'   → Moving threshold: {moving_threshold}')
                logger.info('   → Matches GT boxes with predicted segmentation points')

        except Exception as e:
            logger.error(f'❌ Failed to initialize box detector: {e}')
            logger.warning('Falling back to GT boxes in evaluation')
            use_improved_boxes = False
            improved_box_proposal = None

    # Import visualization functions if needed (mode-specific)
    if save_visualizations:
        if train_mode == 'segmentation_only' or train_mode == 'full':
            # Only import segmentation visualization modules if in segmentation mode
            try:
                from visualization_advanced import plot_advanced_bev, reset_object_color_map
            except ImportError:
                logger.warning('visualization_advanced module not found - segmentation visualizations disabled')
                save_visualizations = False

    # Track last sequence to detect real sequence changes
    last_seq = None
    frame_counter_in_seq = 0  # Frame counter within current sequence

    for batch_idx, batch in progess_bar:
        # Unpack batch (check if motion_features is included)
        if len(batch) == 17:
            # New format with motion_features (17 elements)
            pc1, pc2, ft1, ft2, pc1_compensated, index, seq, ego_motion, pc_last_lidar, pc0_lidar, pc1_lidar, is_new_seq, lbl1, lbl2, transforms1, transforms2, motion_features = batch
        else:
            # Legacy format without motion_features (16 elements)
            pc1, pc2, ft1, ft2, pc1_compensated, index, seq, ego_motion, pc_last_lidar, pc0_lidar, pc1_lidar, is_new_seq, lbl1, lbl2, transforms1, transforms2 = batch
            motion_features = {}  # Empty dict if not available

        # Detect REAL sequence change (not just dataloader's is_new_seq which is buggy)
        # Only reset when sequence actually changes
        actual_new_seq = (last_seq is None) or (seq != last_seq)

        # Reset color map and hidden state when starting a new sequence
        if actual_new_seq:
            # Reset hidden state for GRU temporal aggregation
            hidden_state = None

            # Reset tracker for new sequence (clear all tracks)
            if tracker is not None:
                if hasattr(tracker, 'reset'):
                    # GalleryTracker and EnhancedTracker have .reset() method
                    tracker.reset()
                    logger.info(f'[NEW SEQUENCE] Resetting tracker (GalleryTracker/EnhancedTracker) for sequence {seq} (was: {last_seq})')
                elif hasattr(tracker, 'trackers'):
                    # SimpleTracker uses .trackers attribute
                    tracker.trackers = []
                    logger.info(f'[NEW SEQUENCE] Resetting tracker (SimpleTracker) for sequence {seq} (was: {last_seq})')
                else:
                    logger.warning(f'[NEW SEQUENCE] Tracker type unknown, cannot reset!')

            # Reset MOT accumulator for new sequence (compute metrics per sequence)
            if mot_accumulator is not None and len(mot_accumulator.frame_data) > 0:
                # Compute metrics for previous sequence before resetting
                seq_metrics = mot_accumulator.compute_metrics()
                logger.info(f'[SEQUENCE {last_seq} COMPLETE] sAMOTA: {seq_metrics["sAMOTA"]:.2f}%, MOTA: {seq_metrics["MOTA"]:.2f}%, ID Sw: {seq_metrics["ID_switches"]}')

                # Reset accumulator for new sequence
                from utils.mot_metrics import MOTMetricsAccumulator
                mot_accumulator = MOTMetricsAccumulator()
                logger.info(f'[NEW SEQUENCE] Resetting MOT accumulator for sequence {seq}')

            # Reset frame counter for new sequence
            frame_counter_in_seq = 0

            # Reset color map for segmentation mode only
            if save_visualizations and train_mode in ['segmentation_only', 'full']:
                reset_object_color_map()
                logger.info(f'[NEW SEQUENCE] Resetting object color map for sequence {seq}')

            # Update last sequence
            last_seq = seq
        else:
            # Increment frame counter within sequence
            frame_counter_in_seq += 1

        # Sample to fixed size
        num_points = args.num_points
        pc1 = sample_points(pc1, num_points).permute(0, 2, 1)[:, :3, :].cuda()
        pc2 = sample_points(pc2, num_points).permute(0, 2, 1)[:, :3, :].cuda()
        ft1 = sample_points(ft1, num_points).permute(0, 2, 1)[:, :2, :].cuda()
        ft2 = sample_points(ft2, num_points).permute(0, 2, 1)[:, :2, :].cuda()
        pc1_compensated = sample_points(pc1_compensated, num_points).permute(0, 2, 1)[:, :3, :].cuda()

        # Sample LiDAR data for visualization (keep on GPU for now, will move to CPU in visualization)
        pc1_lidar_sampled = sample_points(pc1_lidar, num_points).permute(0, 2, 1)[:, :3, :].cuda() if pc1_lidar is not None and len(pc1_lidar) > 0 else None
        # For pc2, we use pc0_lidar as the next frame (assuming sequential frames)
        pc2_lidar_sampled = sample_points(pc0_lidar, num_points).permute(0, 2, 1)[:, :3, :].cuda() if pc0_lidar is not None and len(pc0_lidar) > 0 else None

        # Filter object points
        gt_mov_pts1_batch, gt_cls1_batch, _, _, _, cls_obj_id1_batch, boxes1_batch, objs_combined1_batch, objs_idx_combined1_batch, objs_centre_combined1_batch, class_labels1_batch = filter_object_points_batch(
            args, lbl1, pc1, transforms1
        )
        gt_mov_pts2_batch, gt_cls2_batch, _, _, _, cls_obj_id2_batch, boxes2_batch, objs_combined2_batch, objs_idx_combined2_batch, objs_centre_combined2_batch, class_labels2_batch = filter_object_points_batch(
            args, lbl2, pc2, transforms2
        )

        # Compute ground truth flow — SKIPPED during pretrain (flow loss = 0, ego motion not needed)
        # In reid_only mode, compute GT flow for verification/metrics only (not used in loss)
        if not pretrain and train_mode in ['segmentation_only', 'full', 'reid_only']:
            gt_flow_batch = get_gt_flow_new_batch(
                objs_centre_combined1_batch, objs_centre_combined2_batch, gt_cls1_batch,
                cls_obj_id1_batch, cls_obj_id2_batch, pc1,
                pc1_compensated, boxes1_batch, boxes2_batch
            )
            if isinstance(gt_flow_batch, list):
                gt_flow_batch = torch.stack(gt_flow_batch)
        else:
            gt_flow_batch = None  # Not needed: pretrain uses only seg loss

        if isinstance(gt_cls1_batch, list):
            gt_cls1_batch = torch.stack(gt_cls1_batch)

        # Forward pass
        if mode == 'train':
            optimizer.zero_grad()

        # Pass hidden state to model (for GRU temporal aggregation)
        outputs = net(pc1, pc2, ft1, ft2, hidden_state)

        # Update hidden state from outputs (for next iteration)
        # CRITICAL: detach() prevents backprop through ALL previous frames (gradient explosion)
        # This implements Truncated BPTT — GRU still uses temporal context, gradients don't accumulate
        if outputs.get('h') is not None:
            hidden_state = outputs['h'].detach()

        # ===== INITIALIZE LOSS COMPONENTS =====
        # Renamed: loss_flow → loss_motion (ego motion)
        #          loss_seg → loss_seg (segmentation)
        #          loss_box + loss_triplet → loss_motion (reid_only mode)
        # Initialize with zeros connected to computation graph (using model parameters)
        if is_detection_phase:
            # Detection phase: use main network parameters
            dummy_param = next(net.parameters())
        elif train_mode == 'reid_only' and reid_module is not None:
            # Reid phase: use reid module parameters
            try:
                dummy_param = next(reid_module.parameters())
            except StopIteration:
                # Fallback if reid_module has no parameters
                dummy_param = next(net.parameters())
        else:
            dummy_param = next(net.parameters())
        loss_motion = dummy_param.sum() * 0.0
        loss_seg = dummy_param.sum() * 0.0
        loss_dice = dummy_param.sum() * 0.0
        loss_feat = dummy_param.sum() * 0.0

        # Initialize pred_flow and pred_seg (may not be computed in reid_only mode)
        pred_flow = None
        pred_seg = None

        # ===== DETECTION PHASE: Compute Detection Loss (PHASE 1) =====
        if batch_idx == 0 and ep_num == 0:
            logger.info(f'🔍 DEBUG: is_detection_phase={is_detection_phase}, "detection" in outputs={"detection" in outputs}, outputs.keys()={list(outputs.keys())}')

        if is_detection_phase and 'detection' in outputs and outputs['detection'] is not None:
            # Convert GT boxes from Open3D format to tensor format FIRST
            boxes_gt_tensor_list = convert_o3d_boxes_to_tensor_list(boxes1_batch, device=pc1.device)

            # Extract GT classes (convert dict to tensor if needed)
            classes_gt_list = []
            for b_idx in range(len(boxes1_batch)):
                num_boxes = len(boxes1_batch[b_idx])

                if num_boxes == 0:
                    # No boxes, create empty tensor
                    classes_gt_list.append(torch.zeros(0, dtype=torch.long, device=pc1.device))
                elif class_labels1_batch is not None and b_idx < len(class_labels1_batch) and class_labels1_batch[b_idx] is not None:
                    # class_labels is dict: {box_idx: class_id}
                    class_dict = class_labels1_batch[b_idx]
                    classes = torch.zeros(num_boxes, dtype=torch.long, device=pc1.device)
                    for box_idx, class_id in class_dict.items():
                        if box_idx < num_boxes:
                            classes[box_idx] = class_id
                    classes_gt_list.append(classes)
                else:
                    # No class labels available, default to class 0
                    classes_gt_list.append(torch.zeros(num_boxes, dtype=torch.long, device=pc1.device))

            # Compute detection loss
            detection_loss, detection_loss_dict = compute_detection_loss(
                pred_detection=outputs['detection'],
                gt_boxes=boxes_gt_tensor_list,  # Now in tensor format [M, 7]
                gt_classes=classes_gt_list,
                pc1=pc1,
                loss_weights={
                    'center': getattr(args, 'loss_weights', {}).get('box_center', 1.0),
                    'size': getattr(args, 'loss_weights', {}).get('box_size', 0.5),
                    'orientation': getattr(args, 'loss_weights', {}).get('box_orientation', 0.5),
                    'class': getattr(args, 'loss_weights', {}).get('classification', 1.0),
                }
            )

            # Add to loss_motion (will be used as total loss in reid_only mode)
            loss_motion = detection_loss

            # Log detection losses
            for key, value in detection_loss_dict.items():
                if key in epoch_losses:
                    epoch_losses[key].append(value)

            # Extract boxes from detection predictions for mAP computation
            boxes_pred_list, scores_list, pred_classes_list = extract_boxes_from_detection(
                detection_pred=outputs['detection'],
                pc1=pc1,
                threshold=getattr(args, 'detection_threshold', 0.3),
                nms_iou_threshold=getattr(args, 'nms_iou_threshold', 0.5)
            )

            if batch_idx == 0:
                num_pred = sum([len(b) for b in boxes_pred_list])
                num_gt = sum([len(b) for b in boxes_gt_tensor_list])
                center_max = outputs['detection']['center'].max().item()
                center_mean = outputs['detection']['center'].mean().item()
                logger.info(f'🔍 Batch 0: Predicted {num_pred} boxes, GT {num_gt} boxes')
                logger.info(f'   Center heatmap: max={center_max:.4f}, mean={center_mean:.6f}, threshold={getattr(args, "detection_threshold", 0.3)}')

            # Accumulate for mAP computation at epoch end
            for b_idx in range(len(boxes_pred_list)):
                boxes_pred_accumulated.append(boxes_pred_list[b_idx])
                boxes_gt_accumulated.append(boxes_gt_tensor_list[b_idx])

        # ===== REID_ONLY MODE: Use GT inputs for verification =====
        if train_mode == 'reid_only':
            # Use GT segmentation for metrics (already handled in reid_module)
            if args.use_gt_segmentation:
                # Create pred_seg from GT for metrics computation
                pred_seg = gt_cls1_batch.unsqueeze(1).float()  # [B, N] → [B, 1, N]
            else:
                # Use predicted segmentation from model (for visualization)
                if outputs.get('cls_supervised') is not None:
                    pred_seg = outputs['cls_supervised']  # [B, 1, N]

            # Use GT flow for RNE verification (should be ~0)
            if args.use_gt_ego_motion and gt_flow_batch is not None:
                # "Predict" GT flow for metrics computation (RNE should be ~0)
                pred_flow = gt_flow_batch.clone()

        # ===== COMPUTE SEGMENTATION & EGO MOTION LOSSES =====
        if train_mode in ['segmentation_only', 'full']:
            pred_flow = outputs['flow']  # [B, 3, N]
            pc1_warp = pc1 + pred_flow

            # Ego motion loss (scene flow EPE) — only in 'full' mode
            if train_mode == 'full' and gt_flow_batch is not None:
                loss_motion = scene_flow_loss_baseline(pc1_warp, gt_flow_batch)
            # segmentation_only: no flow supervision, loss_motion stays at 0

            # Segmentation loss (BCE + object-aware)
            if outputs['cls_supervised'] is not None:
                pred_seg = outputs['cls_supervised']  # [B, 1, N]
                loss_seg_base = motion_seg_loss_baseline(pred_seg, gt_cls1_batch)

                # Convert obj_ids to tensor if it's a list
                if isinstance(cls_obj_id1_batch, list):
                    obj_ids_tensor = torch.stack(cls_obj_id1_batch) if len(cls_obj_id1_batch) > 0 else None
                else:
                    obj_ids_tensor = cls_obj_id1_batch

                # Object-aware segmentation loss
                if obj_ids_tensor is not None and hasattr(args, 'use_object_aware_loss') and args.use_object_aware_loss:
                    loss_obj_aware = object_aware_seg_loss(pred_seg, gt_cls1_batch, obj_ids_tensor)
                else:
                    loss_obj_aware = torch.tensor(0.0).cuda()

                # Instance consistency loss
                if obj_ids_tensor is not None and hasattr(args, 'use_consistency_loss') and args.use_consistency_loss:
                    consistency_margin = args.consistency_margin if hasattr(args, 'consistency_margin') else 0.1
                    loss_consistency = instance_consistency_loss(pred_seg, obj_ids_tensor, margin=consistency_margin)
                else:
                    loss_consistency = torch.tensor(0.0).cuda()

                # Combine segmentation components into loss_seg
                weight_obj_aware = getattr(args, 'weight_obj_aware', 0.3)
                weight_consistency = getattr(args, 'weight_consistency', 0.2)
                loss_seg = loss_seg_base + weight_obj_aware * loss_obj_aware + weight_consistency * loss_consistency

                # Soft Dice loss: directly optimizes overlap, complements BCE
                loss_dice = soft_dice_loss(pred_seg, gt_cls1_batch)

                # Feature contrast loss: appearance-based inter-class separation
                pc1_features = outputs['pc1_features']  # [B, 256, N]
                loss_feat = feature_contrast_loss(pc1_features, gt_cls1_batch)

        # ===== COMPUTE RE-ID LOSSES (SKIP IN DETECTION PHASE) =====
        if train_mode in ['reid_only', 'full'] and reid_module is not None and not is_detection_phase:
            # ===== BOX GENERATION: GT or Predicted =====
            if use_improved_boxes and improved_box_proposal is not None:
                # Get predicted segmentation
                with torch.no_grad():
                    seg_pred1 = outputs['cls_supervised']  # [B, 1, N]
                    seg_pred2 = outputs.get('cls_supervised2', None)

                # Check detector type
                from models.temporal_motion_detector import TemporalMotionDetector
                is_temporal_detector = isinstance(improved_box_proposal, TemporalMotionDetector)

                # Initialize defaults (in case both paths fail)
                boxes1_pred = boxes1_batch
                boxes2_pred = boxes2_batch
                track_ids1_pred = cls_obj_id1_batch
                track_ids2_pred = cls_obj_id2_batch
                boxes1_info = None

                if is_temporal_detector:
                    # TemporalMotionDetector needs both frames
                    if seg_pred2 is None:
                        logger.info("🔄 Generating seg_pred2 for TemporalMotionDetector...")
                        # Generate segmentation for frame 2 using the model
                        with torch.no_grad():
                            # Run model on pc2 to get segmentation
                            # Use same model as pc1 but swap frames
                            outputs2 = net(pc2, pc1, ft2, ft1, hidden_state)
                            seg_pred2 = outputs2['cls_supervised']

                        if seg_pred2 is None:
                            logger.warning("Failed to generate seg_pred2, falling back")
                            is_temporal_detector = False

                    # If still temporal detector after generation attempt
                    if is_temporal_detector:
                        # Temporal Motion Detector: Use motion between frames
                        # Get ego motion (if available)
                        ego_motion = None
                        if hasattr(train_loader.dataset, 'get_ego_motion'):
                            # Try to get ego motion from dataset
                            try:
                                ego_motion = torch.stack([train_loader.dataset.get_ego_motion(i) for i in range(len(pc1))]).to(pc1.device)
                            except:
                                logger.warning("Could not get ego motion, using None")
                                ego_motion = None

                        boxes1_pred, track_ids1_pred, boxes1_info = improved_box_proposal(
                            points_t=pc1,
                            points_t1=pc2,
                            seg_pred_t=seg_pred1,
                            seg_pred_t1=seg_pred2,
                            ego_motion=ego_motion,
                            gt_boxes_t=boxes1_batch,
                            gt_boxes_t1=boxes2_batch,
                            gt_classes_t=class_labels1_batch,
                            gt_track_ids_t=cls_obj_id1_batch,
                            time_delta=0.1  # 10 Hz
                        )

                        # For consistency, also process frame 2 (can reuse or skip)
                        boxes2_pred = boxes2_batch  # Fallback to GT for frame 2
                        track_ids2_pred = cls_obj_id2_batch

                if not is_temporal_detector:
                    # Segmentation Matcher: Match GT boxes with segmentation
                    boxes1_pred, track_ids1_pred, boxes1_info = improved_box_proposal(
                        points=pc1,
                        seg_pred=seg_pred1,
                        gt_boxes=boxes1_batch,
                        gt_track_ids=cls_obj_id1_batch,
                        gt_classes=class_labels1_batch
                    )

                    if seg_pred2 is not None:
                        boxes2_pred, track_ids2_pred, boxes2_info = improved_box_proposal(
                            points=pc2,
                            seg_pred=seg_pred2,
                            gt_boxes=boxes2_batch,
                            gt_track_ids=cls_obj_id2_batch,
                            gt_classes=class_labels2_batch
                        )
                    else:
                        boxes2_pred = boxes2_batch
                        track_ids2_pred = cls_obj_id2_batch

                # ===== REFUERZA SEGMENTACIÓN CON BOXES =====
                # Marca puntos dentro de boxes predichos como "moving" (segmentation=1)
                # Esto hace que aparezcan naranjas en visualización y refuerza la segmentación
                if pred_seg is not None and boxes1_pred is not None:
                    if batch_idx == 0:
                        logger.info(f'🔧 Starting segmentation reinforcement (pred_seg shape: {pred_seg.shape}, boxes: {len(boxes1_pred)})')
                    # Clone pred_seg to avoid modifying original
                    pred_seg_reinforced = pred_seg.clone()

                    # For each batch
                    for b_idx in range(pred_seg.shape[0]):
                        if b_idx >= len(boxes1_pred) or boxes1_pred[b_idx] is None:
                            continue

                        boxes_b = boxes1_pred[b_idx]

                        # Handle both tensor and dict formats
                        if isinstance(boxes_b, torch.Tensor):
                            # Tensor format: [M, 7] where each row is [x, y, z, l, w, h, yaw]
                            if boxes_b.numel() == 0:
                                continue
                            boxes_array = boxes_b.cpu().numpy()
                        elif isinstance(boxes_b, dict):
                            # Dict format: {box_id: Object_3D}
                            if len(boxes_b) == 0:
                                continue
                            # Convert dict to array [M, 7]
                            boxes_list = []
                            for box_obj in boxes_b.values():
                                center = box_obj.get_center()
                                size = box_obj.get_lwh()
                                yaw = box_obj.get_yaw()
                                boxes_list.append([center[0], center[1], center[2],
                                                 size[0], size[1], size[2], yaw])
                            boxes_array = np.array(boxes_list)
                        else:
                            continue

                        # Get points for this batch [N, 3]
                        points_b = pc1[b_idx].permute(1, 0).cpu().numpy()  # [3, N] -> [N, 3]
                        num_points = points_b.shape[0]

                        total_points_marked = 0

                        # For each box, mark points inside as moving
                        for box_idx in range(len(boxes_array)):
                            try:
                                # Box format: [x, y, z, l, w, h, yaw]
                                box = boxes_array[box_idx]
                                center = box[:3]
                                size = box[3:6]
                                yaw = box[6]

                                # Transform points to box frame
                                cos_yaw = np.cos(-yaw)
                                sin_yaw = np.sin(-yaw)
                                rot_mat = np.array([[cos_yaw, -sin_yaw, 0],
                                                   [sin_yaw, cos_yaw, 0],
                                                   [0, 0, 1]])

                                points_centered = points_b - center
                                points_rotated = points_centered @ rot_mat.T

                                # Check if inside box
                                half_size = size / 2.0
                                inside_x = np.abs(points_rotated[:, 0]) < half_size[0]
                                inside_y = np.abs(points_rotated[:, 1]) < half_size[1]
                                inside_z = np.abs(points_rotated[:, 2]) < half_size[2]
                                inside_mask = inside_x & inside_y & inside_z

                                # Mark these points as inside box (yellow in visualization)
                                num_inside = inside_mask.sum()
                                if num_inside > 0:
                                    # Convert boolean mask to tensor indices
                                    inside_indices = torch.from_numpy(inside_mask).to(pred_seg_reinforced.device)
                                    # pred_seg shape: [B, 1, N]
                                    # Mark as 2.0 = yellow (inside box), keeping 1.0 = orange (motion)
                                    pred_seg_reinforced[b_idx, 0, inside_indices] = 2.0
                                    total_points_marked += num_inside

                                    if batch_idx == 0 and box_idx < 3:  # Log first 3 boxes
                                        logger.debug(f'  Box {box_idx}: {num_inside} points marked inside')
                            except Exception as e:
                                if batch_idx == 0:
                                    logger.warning(f'  Box {box_idx} failed: {e}')
                                continue

                        if batch_idx == 0 and total_points_marked > 0:
                            logger.info(f'  Batch {b_idx}: {total_points_marked}/{num_points} points marked in {len(boxes_array)} boxes')

                    # Update pred_seg with reinforced version
                    pred_seg = pred_seg_reinforced

                    if batch_idx == 0:
                        # Log statistics for first batch
                        num_boxes = sum([len(boxes1_pred[b]) if boxes1_pred[b] is not None else 0
                                        for b in range(min(len(boxes1_pred), pred_seg.shape[0]))])
                        total_moving_before = (pred_seg_reinforced[0, 0] == 0).sum().item()  # Background points
                        total_moving_after = (pred_seg[0, 0] > 0.5).sum().item()
                        logger.info(f'Segmentation reinforcement: {num_boxes} boxes → {total_moving_after} moving points (was background)')
                        if total_moving_after == 0:
                            logger.warning('No points marked as moving! Check box generation.')

                # Use matched boxes in reid_batch
                reid_batch = {
                    'pc1': pc1,
                    'pc2': pc2,
                    'seg_gt': gt_cls1_batch,
                    'seg_gt2': gt_cls2_batch,
                    'boxes_gt': boxes1_pred,  # GT boxes that have segmented points
                    'boxes_gt2': boxes2_pred,
                    'track_ids_gt': track_ids1_pred,  # Corresponding GT track IDs
                    'track_ids_gt2': track_ids2_pred,
                    'class_labels_gt': class_labels1_batch,
                    'class_labels_gt2': class_labels2_batch,
                }

                # Log statistics (first batch only)
                if batch_idx == 0:
                    total_gt_boxes = sum([len(b) for b in boxes1_batch if b is not None])
                    total_matched = sum([len(b) for b in boxes1_pred if b is not None])
                    if total_gt_boxes > 0:
                        match_rate = total_matched / total_gt_boxes * 100
                        logger.info(f'GT boxes: {total_gt_boxes}, matched with segmentation: {total_matched} ({match_rate:.1f}%)')
                    else:
                        logger.info(f'Matched {total_matched} boxes with segmentation')

            else:
                # Use GT boxes (original behavior)
                reid_batch = {
                    'pc1': pc1,  # [B, 3, N] Frame t
                    'pc2': pc2,  # [B, 3, N] Frame t+1 (for temporal triplet loss)
                    'seg_gt': gt_cls1_batch,  # [B, N] GT segmentation frame t
                    'seg_gt2': gt_cls2_batch,  # [B, N] GT segmentation frame t+1
                    'boxes_gt': boxes1_batch,  # List[boxes] GT boxes frame t
                    'boxes_gt2': boxes2_batch,  # List[boxes] GT boxes frame t+1
                    'track_ids_gt': cls_obj_id1_batch,  # List[track_ids] GT track IDs frame t
                    'track_ids_gt2': cls_obj_id2_batch,  # List[track_ids] GT track IDs frame t+1
                    'class_labels_gt': class_labels1_batch,  # List[dict] GT class labels frame t
                    'class_labels_gt2': class_labels2_batch,  # List[dict] GT class labels frame t+1
                }

            # Forward Re-ID module
            # Pass epoch for progressive training strategy (detection pretraining vs Re-ID)
            reid_outputs = reid_module(reid_batch, epoch=ep_num)

            # Extract Re-ID losses (if computed)
            if 'losses' in reid_outputs:
                reid_losses = reid_outputs['losses']
                # Use model parameters for defaults to ensure gradient flow
                dummy_reid_param = next(reid_module.parameters())
                loss_box_reid = reid_losses.get('box', dummy_reid_param.sum() * 0.0)
                loss_triplet_reid = reid_losses.get('reid_triplet', dummy_reid_param.sum() * 0.0)
                loss_classification = reid_losses.get('classification', dummy_reid_param.sum() * 0.0)

                # Get loss weights
                weight_box = getattr(args, 'loss_weights', {}).get('box_center', 1.0)
                weight_triplet = getattr(args, 'loss_weights', {}).get('reid_triplet', 1.0)
                weight_classification = getattr(args, 'loss_weights', {}).get('classification', 0.5)

                loss_reid = weight_box * loss_box_reid + weight_triplet * loss_triplet_reid + weight_classification * loss_classification
                loss_motion = loss_reid   # Re-ID mode: use motion slot for reid loss

        # ===== COMPUTE TOTAL LOSS BASED ON TRAINING MODE =====
        if train_mode == 'reid_only':
            loss_total = loss_motion   # Re-ID loss stored in motion slot
        elif train_mode == 'segmentation_only':
            weight_seg = getattr(args, 'weight_seg', 1.0)
            weight_dice = getattr(args, 'weight_dice', 0.3)
            weight_feat = getattr(args, 'weight_feat_contrast', 0.2)
            if pretrain:
                loss_total = loss_seg
            else:
                # No flow/motion term — pure segmentation with appearance cues
                loss_total = weight_seg * loss_seg + weight_dice * loss_dice + weight_feat * loss_feat
        elif train_mode == 'full':
            weight_seg = getattr(args, 'weight_seg', 1.0)
            weight_motion = getattr(args, 'weight_flow', 0.5)
            weight_dice = getattr(args, 'weight_dice', 0.3)
            weight_feat = getattr(args, 'weight_feat_contrast', 0.2)
            if pretrain:
                loss_total = loss_seg
            else:
                loss_total = weight_seg * loss_seg + weight_motion * loss_motion + weight_dice * loss_dice + weight_feat * loss_feat
        else:
            loss_total = loss_seg + loss_motion

        # Backward pass
        if mode == 'train':
            loss_total.backward()

            # Clip gradients
            if train_mode == 'reid_only' and reid_module is not None:
                torch.nn.utils.clip_grad_norm_(reid_module.parameters(), max_norm=1.0)
            else:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

            optimizer.step()

        # Track losses (using unified naming)
        epoch_losses['total'].append(loss_total.item())
        epoch_losses['motion'].append(loss_motion.item())
        epoch_losses['seg'].append(loss_seg.item())
        epoch_losses['dice'].append(loss_dice.item())
        epoch_losses['feat'].append(loss_feat.item())

        # Compute metrics based on training mode
        with torch.no_grad():
            if train_mode == 'reid_only':
                # For Re-ID mode: compute tracking metrics (box IoU, etc.)
                batch_metrics = {}

                # Skip reid_outputs references in detection phase (not defined)
                if not is_detection_phase:
                    if 'boxes' in reid_outputs and 'boxes_gt' in reid_batch:
                        batch_metrics = compute_reid_metrics(
                            pred_boxes=reid_outputs['boxes'],
                            gt_boxes=reid_batch['boxes_gt'],
                            embeddings=reid_outputs.get('embeddings', [])
                        )

                    # Add classification accuracy
                    if 'losses' in reid_outputs and 'classification_acc' in reid_outputs['losses']:
                        batch_metrics['classification_acc'] = reid_outputs['losses']['classification_acc']

                # Compute flow metrics for verification (should be ~0 with GT flow)
                # Skip in detection phase - not relevant
                if not is_detection_phase and pred_flow is not None and gt_flow_batch is not None:
                    flow_metrics = compute_metrics_simple(
                        pred_flow=pred_flow,
                        gt_flow=gt_flow_batch,
                        pred_seg=pred_seg,
                        gt_seg=gt_cls1_batch,
                        pc1=pc1
                    )
                    # Add flow metrics to batch_metrics
                    batch_metrics.update({
                        'RNE': flow_metrics['RNE'],
                        'EPE': flow_metrics['EPE'],
                        'SAS': flow_metrics['SAS'],
                        'RAS': flow_metrics['RAS'],
                    })

                # ===== PREPARE VISUALIZATION DATA (REID_ONLY, NOT DETECTION PHASE) =====
                # Skip in detection phase - reid_outputs not defined
                if not is_detection_phase:
                    # Extract GT boxes and IDs (FOR MOT EVALUATION)
                    # IMPORTANT: Use REAL GT boxes (boxes1_batch), NOT filtered boxes (reid_batch['boxes_gt'])
                    # This ensures MOT metrics compare against true ground truth
                    from models.reid_module import convert_o3d_boxes_to_tensor
                    gt_boxes_dict_real = boxes1_batch[0] if len(boxes1_batch) > 0 else {}
                    gt_boxes_b, gt_ids_b = convert_o3d_boxes_to_tensor(gt_boxes_dict_real)

                    # Extract predicted detections
                    pred_boxes_b = reid_outputs['boxes'][0] if len(reid_outputs['boxes']) > 0 else torch.zeros((0, 7))
                    embeddings_b = reid_outputs['embeddings'][0] if len(reid_outputs.get('embeddings', [])) > 0 else None

                    # TRAINING MODE: Use GT boxes and GT IDs (supervised learning)
                    # VALIDATION MODE: Use predicted boxes + tracker (evaluation)
                    if mode == 'train':
                        # Supervised: Use GT directly
                        pred_boxes_tracked = gt_boxes_b
                        pred_ids_tracked = gt_ids_b
                        if batch_idx == 0:
                            logger.info(f'📚 Training mode: Using GT boxes ({len(gt_boxes_b)}) and GT IDs for supervision')
                    else:
                        # Initialize with raw detections (no tracking IDs yet)
                        pred_boxes_tracked = pred_boxes_b
                        pred_ids_tracked = torch.arange(len(pred_boxes_b)).long()  # Sequential IDs

                    # ===== UPDATE TRACKER FOR MOT METRICS (TRAIN & VAL) =====
                    # Skip tracking in detection phase (Phase 1 - only detection metrics)
                    # Note: Training metrics will be low (small batches), validation is more meaningful
                    # IMPORTANT: If use_gt_boxes=True, feed GT boxes to tracker for fair evaluation
                    if not is_detection_phase and tracker is not None and mot_accumulator is not None:
                        # Determine which boxes to feed to tracker
                        if args.use_gt_boxes:
                            # Use GT boxes for tracking (ensures tracker gets all detections)
                            boxes_for_tracker = gt_boxes_b
                            if batch_idx == 0 and mode == 'val':
                                logger.info(f'🎯 Validation: Using GT boxes ({len(gt_boxes_b)}) for tracker (use_gt_boxes=True)')
                        else:
                            # Use predicted boxes from ReID module (DBSCAN)
                            boxes_for_tracker = pred_boxes_b
                            if batch_idx == 0 and mode == 'val':
                                logger.info(f'🔍 Validation: Using predicted boxes ({len(pred_boxes_b)}) for tracker (DBSCAN)')

                        # Calculate num_points per box for GalleryTracker
                        num_points_array = None
                        if use_gallery:
                            from utils.instance_assignment import assign_points_to_boxes
                            # Assign points to boxes to count them
                            temp_track_ids = torch.arange(len(boxes_for_tracker)).long()
                            point_assignments = assign_points_to_boxes(
                                points=pc1[0].T,  # [N, 3]
                                boxes=boxes_for_tracker,
                                track_ids=temp_track_ids
                            )
                            # Count points per box
                            num_points_array = np.zeros(len(boxes_for_tracker), dtype=np.int32)
                            for box_idx in range(len(boxes_for_tracker)):
                                num_points_array[box_idx] = np.sum(point_assignments == box_idx)

                        # Update tracker (pass motion_dict and num_points if using GalleryTracker)
                        if use_gallery:
                            # GalleryTracker accepts boxes, embeddings, num_points, motion_dict
                            tracks = tracker.update(
                                boxes=boxes_for_tracker,
                                embeddings=embeddings_b,
                                num_points_per_box=num_points_array,
                                motion_dict=motion_features
                            )
                        elif use_enhanced:
                            # EnhancedTracker accepts motion_dict
                            tracks = tracker.update(boxes_for_tracker, embeddings_b, motion_dict=motion_features)
                        else:
                            # SimpleTracker doesn't use motion
                            tracks = tracker.update(boxes_for_tracker, embeddings_b)

                        # Use tracked boxes and IDs if available
                        if len(tracks) > 0:
                            pred_boxes_tracked = torch.stack([torch.from_numpy(box) for box, _ in tracks]).float()
                            pred_ids_tracked = torch.tensor([id for _, id in tracks]).long()
                        else:
                            pred_boxes_tracked = torch.zeros((0, 7))
                            pred_ids_tracked = torch.zeros(0).long()

                        # Accumulate for MOT metrics (use frame counter within sequence)
                        mot_accumulator.update(
                            frame_id=frame_counter_in_seq,
                            gt_boxes=gt_boxes_b,
                            gt_ids=gt_ids_b,
                            pred_boxes=pred_boxes_tracked,
                            pred_ids=pred_ids_tracked
                        )

                    # ===== ASSIGN POINTS TO INSTANCES (FOR VISUALIZATION) =====
                    point_instance_ids = None
                    if len(pred_boxes_tracked) > 0:
                        from utils.instance_assignment import assign_points_to_boxes
                        # Assign each point to a tracked instance
                        point_instance_ids = assign_points_to_boxes(
                            points=pc1[0].T,  # [N, 3]
                            boxes=pred_boxes_tracked,
                            track_ids=pred_ids_tracked
                        )
                else:
                    batch_metrics = {}
            else:
                # For segmentation mode: compute segmentation + flow metrics
                batch_metrics = compute_metrics_simple(
                    pred_flow=pred_flow,
                    gt_flow=gt_flow_batch,
                    pred_seg=pred_seg if outputs['cls_supervised'] is not None else None,
                    gt_seg=gt_cls1_batch if outputs['cls_supervised'] is not None else None,
                    pc1=pc1  # Pass pc1 for RNE calculation
                )

                # ===== COMPUTE INSTANCE-LEVEL METRICS (NEW) =====
                # Extract bounding boxes from segmentation predictions using DBSCAN
                if pred_seg is not None and boxes1_batch is not None:
                    try:
                        from models.box_proposal import BoxProposalNetwork

                        # Initialize DBSCAN box proposal (reuse config from reid if available)
                        box_cfg = getattr(args, 'reid', {})
                        if isinstance(box_cfg, dict) and 'box_proposal' in box_cfg:
                            box_proposal_cfg = box_cfg['box_proposal']
                            eps = box_proposal_cfg.get('eps', 2.0)
                            min_samples = box_proposal_cfg.get('min_samples', 3)
                        else:
                            # Default DBSCAN parameters for segmentation mode
                            eps = 2.0
                            min_samples = 3

                        box_proposer = BoxProposalNetwork(
                            method='dbscan',
                            eps=eps,
                            min_samples=min_samples
                        )

                        # Convert predicted segmentation to binary mask
                        # pred_seg is [B, 1, N], threshold at 0.5
                        pred_mask = (torch.sigmoid(pred_seg) > 0.5).squeeze(1)  # [B, N]

                        # Extract boxes from predicted moving points
                        pc1_transposed = pc1.permute(0, 2, 1)  # [B, N, 3]
                        pred_boxes_list, _ = box_proposer(pc1_transposed, pred_mask)

                        # Compute instance-level metrics
                        instance_metrics = compute_reid_metrics(
                            pred_boxes=pred_boxes_list,
                            gt_boxes=boxes1_batch,
                            embeddings=[]  # No embeddings in segmentation mode
                        )

                        # Add instance-level metrics to batch_metrics
                        batch_metrics.update(instance_metrics)

                    except Exception as e:
                        # If instance-level metrics fail, log warning and continue
                        if batch_idx == 0:
                            logger.warning(f'Could not compute instance-level metrics: {str(e)}')
                        # Add zeros for missing metrics
                        batch_metrics.update({
                            'box_precision': 0.0, 'box_recall': 0.0, 'box_f1': 0.0,
                            'box_iou': 0.0, 'num_pred_boxes': 0.0, 'num_gt_boxes': 0.0
                        })

            # ===== ACCUMULATE BOXES FOR mAP/mIoU COMPUTATION =====
            if train_mode == 'reid_only' and not is_detection_phase:
                # Accumulate predicted and GT boxes for detection metrics (Phase 2 only)
                # Note: In detection phase, boxes are already accumulated after detection head forward pass
                if 'boxes' in reid_outputs and len(reid_outputs['boxes']) > 0:
                    for b_idx in range(len(reid_outputs['boxes'])):
                        boxes_pred_accumulated.append(reid_outputs['boxes'][b_idx])

                        # Get GT boxes (use different variable to avoid overwriting gt_boxes_b for visualization)
                        from models.reid_module import convert_o3d_boxes_to_tensor
                        gt_boxes_dict = reid_batch['boxes_gt'][b_idx] if b_idx < len(reid_batch['boxes_gt']) else {}
                        gt_boxes_for_map, _ = convert_o3d_boxes_to_tensor(gt_boxes_dict)
                        boxes_gt_accumulated.append(gt_boxes_for_map)

                # Accumulate segmentation for mIoU
                if pred_seg is not None and gt_cls1_batch is not None:
                    seg_pred = (torch.sigmoid(pred_seg) > 0.5).long().squeeze(1)  # [B, N]
                    for b_idx in range(seg_pred.shape[0]):
                        seg_pred_accumulated.append(seg_pred[b_idx])
                        seg_gt_accumulated.append(gt_cls1_batch[b_idx])

            for key in epoch_metrics.keys():
                if key in batch_metrics:
                    epoch_metrics[key].append(batch_metrics[key])

        # ===== GENERATE VISUALIZATIONS DURING VALIDATION =====
        # Save ALL frames during validation to see complete temporal evolution
        if save_visualizations and mode == 'val':
            # Prepare bounding boxes - handle different formats
            # boxes_batch can be: list of boxes, list of lists, or None
            try:
                # Check if boxes1_batch is a list and not empty
                if isinstance(boxes1_batch, list) and len(boxes1_batch) > 0:
                    # If first element is a list/array, take it (batch format)
                    if isinstance(boxes1_batch[0], (list, np.ndarray)):
                        boxes1 = boxes1_batch[0]
                    # If first element is a number or None, use the list as is
                    else:
                        boxes1 = boxes1_batch if boxes1_batch[0] is not None else None
                else:
                    boxes1 = None

                if isinstance(boxes2_batch, list) and len(boxes2_batch) > 0:
                    if isinstance(boxes2_batch[0], (list, np.ndarray)):
                        boxes2 = boxes2_batch[0]
                    else:
                        boxes2 = boxes2_batch if boxes2_batch[0] is not None else None
                else:
                    boxes2 = None
            except Exception as e:
                logger.warning(f'Error preparing boxes for visualization: {e}')
                boxes1 = None
                boxes2 = None

            # Prepare object IDs - handle batch format
            try:
                if isinstance(cls_obj_id1_batch, list) and len(cls_obj_id1_batch) > 0:
                    gt_obj_ids = cls_obj_id1_batch[0]  # Take first element of batch
                else:
                    gt_obj_ids = cls_obj_id1_batch if cls_obj_id1_batch is not None else None
            except Exception as e:
                logger.warning(f'Error preparing object IDs for visualization: {e}')
                gt_obj_ids = None

            # Prepare visualization directory for this epoch
            epoch_vis_dir = os.path.join(vis_dir, f'epoch_{ep_num:03d}')
            os.makedirs(epoch_vis_dir, exist_ok=True)

            # Prepare metrics dict for visualization (mode-dependent)
            if train_mode == 'reid_only':
                # Re-ID mode: use detection metrics
                vis_metrics = {
                    'box_f1': batch_metrics.get('box_f1', 0.0),
                    'box_precision': batch_metrics.get('box_precision', 0.0),
                    'box_recall': batch_metrics.get('box_recall', 0.0),
                    'box_iou': batch_metrics.get('box_iou', 0.0),
                }
            else:
                # Segmentation mode: use scene flow metrics
                vis_metrics = {
                    'RNE': batch_metrics['RNE'],
                    'EPE': batch_metrics['EPE'],
                    'SAS': batch_metrics.get('SAS', 0.0),
                    'RAS': batch_metrics.get('RAS', 0.0),
                    'mIoU': batch_metrics['miou'],
                    'IoU_moving': batch_metrics.get('IoU_moving', 0.0),
                    'IoU_static': batch_metrics.get('IoU_static', 0.0),
                    'F1': batch_metrics['f1']
                }

            # Generate visualization (mode-specific)
            if is_detection_phase:
                # PHASE 1: Detection visualization (boxes pred vs GT only)
                try:
                    if batch_idx < 5:  # Save first 5 samples only
                        logger.info(f'🎯 Generating detection visualization (boxes pred vs GT)')

                        # Get predicted boxes from accumulated list
                        if len(boxes_pred_accumulated) > 0:
                            boxes_pred_vis = boxes_pred_accumulated[-1]  # Last accumulated (current batch)
                        else:
                            boxes_pred_vis = torch.empty(0, 7)

                        # Get GT boxes
                        if len(boxes_gt_accumulated) > 0:
                            boxes_gt_vis = boxes_gt_accumulated[-1]
                        else:
                            boxes_gt_vis = torch.empty(0, 7)

                        # Save visualization
                        save_path = Path(epoch_vis_dir) / f'sample_{batch_idx:03d}_detection.png'
                        visualize_detection_results(
                            pc1=pc1,
                            boxes_pred=[boxes_pred_vis],
                            boxes_gt=[boxes_gt_vis],
                            save_path=save_path,
                            sample_idx=0
                        )

                        if batch_idx == 0:
                            logger.info(f'Saved detection visualization to {epoch_vis_dir}')

                except Exception as e:
                    import traceback
                    logger.warning(f'Detection visualization failed: {e}')
                    logger.warning(f'Traceback: {traceback.format_exc()}')

            elif train_mode == 'reid_only':
                # Re-ID visualization: 3-panel evaluation (GT BEV | RGB | Pred BEV)
                try:
                    if pred_boxes_tracked is not None and gt_boxes_b is not None:
                        if batch_idx == 0:  # Log only for first batch
                            logger.info(f'Generating 3-panel evaluation visualization to: {epoch_vis_dir}')

                        # Import 3-panel visualization
                        from utils.visualization_eval import plot_3panel_evaluation

                        # Prepare frame number (5-digit zero-padded)
                        frame_number = str(index[0].item()).zfill(5)

                        # Prepare GT segmentation labels (0=background, 1=moving)
                        gt_seg_labels = gt_cls1_batch[0].cpu().numpy() if gt_cls1_batch is not None else None

                        # Prepare predicted segmentation labels (AFTER reinforcement)
                        # Now supports: 0=background, 1=motion, 2=inside_box
                        pred_seg_labels = None
                        if pred_seg is not None:
                            pred_seg_cpu = pred_seg[0].cpu()
                            # Convert to discrete labels (0, 1, 2)
                            if pred_seg_cpu.ndim == 2:  # [1, N] or [num_classes, N]
                                if pred_seg_cpu.shape[0] == 1:
                                    # Segmentation with reinforcement [1, N]
                                    # Values: 0.0, 1.0, 2.0 → round to get discrete labels
                                    pred_seg_labels = torch.round(pred_seg_cpu.squeeze(0)).int().numpy()
                                else:
                                    # Multi-class [num_classes, N]
                                    pred_seg_labels = torch.argmax(pred_seg_cpu, dim=0).numpy()
                            else:  # [N]
                                pred_seg_labels = torch.round(pred_seg_cpu).int().numpy()

                        # Get kitti_locations for RGB loading
                        # Need to get from args or construct from dataset path
                        from external.vod.configuration.file_locations import VodTrackLocations
                        kitti_locations = VodTrackLocations(root_dir=args.dataset_path)

                        # Assign class labels to each radar point based on box membership
                        point_class_labels_gt = None
                        if class_labels1_batch is not None and len(class_labels1_batch) > 0:
                            class_labels_boxes = class_labels1_batch[0]  # Box-level class labels
                            if class_labels_boxes is not None and len(gt_boxes_b) > 0:
                                from utils.instance_assignment import assign_points_to_boxes
                                radar_points = pc1[0].T.cpu().numpy()  # [N, 3]

                                # Assign each point to a box (returns track ID or -1)
                                # assign_points_to_boxes handles both tensors and numpy arrays
                                point_track_ids = assign_points_to_boxes(
                                    points=radar_points,
                                    boxes=gt_boxes_b,  # Can be tensor or numpy
                                    track_ids=gt_ids_b  # Can be tensor or numpy
                                )

                                # class_labels_boxes is a dict with track_id as keys
                                # Map each point's track ID to its class label
                                point_class_labels_gt = []
                                for track_id in point_track_ids:
                                    if track_id >= 0 and int(track_id) in class_labels_boxes:
                                        point_class_labels_gt.append(class_labels_boxes[int(track_id)])
                                    else:
                                        point_class_labels_gt.append('background')
                                point_class_labels_gt = np.array(point_class_labels_gt)

                        # Prepare GT data
                        gt_data = {
                            'boxes': gt_boxes_b,
                            'ids': gt_ids_b,
                            'seg_labels': gt_seg_labels,
                            'class_labels': point_class_labels_gt  # Now point-level
                        }

                        # Prepare prediction data
                        pred_data = {
                            'boxes': pred_boxes_tracked,
                            'ids': pred_ids_tracked,  # Already remapped by MOT metrics
                            'seg_labels': pred_seg_labels
                        }

                        # Generate 3-panel visualization
                        save_path = os.path.join(epoch_vis_dir, f'frame_{batch_idx:04d}.png')
                        plot_3panel_evaluation(
                            frame_number=frame_number,
                            pc_lidar=pc1_lidar_sampled[0].permute(1, 0).cpu().numpy() if pc1_lidar_sampled is not None else None,  # [3, N] -> [N, 3]
                            pc_radar=pc1[0].permute(1, 0).cpu().numpy(),  # [3, N] -> [N, 3]
                            gt_data=gt_data,
                            pred_data=pred_data,
                            kitti_locations=kitti_locations,
                            save_path=save_path
                        )
                    else:
                        if batch_idx == 0:  # Log only for first batch
                            logger.warning(f'Skipping visualization - pred_boxes_tracked or gt_boxes_b is None')
                except Exception as e:
                    import traceback
                    logger.error(f'❌ Failed to generate 3-panel visualization for batch {batch_idx}: {e}')
                    logger.error(f'Traceback: {traceback.format_exc()}')

            else:
                # Segmentation visualization: show scene flow and segmentation
                try:
                    plot_advanced_bev(
                        pc1=pc1,
                        pc2=pc2,
                        pred_flow=pred_flow,
                        gt_flow=gt_flow_batch,
                        pred_seg=pred_seg,
                        gt_seg=gt_cls1_batch,
                        boxes1=boxes1,
                        boxes2=boxes2,
                        save_path=epoch_vis_dir,
                        index=batch_idx,
                        metrics=vis_metrics,
                        pc1_lidar=pc1_lidar_sampled,
                        pc2_lidar=pc2_lidar_sampled,
                        gt_obj_ids=gt_obj_ids,  # Object IDs extracted from batch
                        pred_obj_ids=None  # No predicted object IDs yet (binary segmentation only)
                    )
                except Exception as e:
                    import traceback
                    logger.warning(f'Failed to generate visualization for batch {batch_idx}: {e}')
                    logger.debug(f'Traceback: {traceback.format_exc()}')

        # Update progress bar (show primary metrics based on mode)
        if train_mode == 'reid_only':
            progess_bar.set_description(
                f"Epoch {ep_num} [{mode}] | Loss: {loss_total.item():.4f} | "
                f"F1: {batch_metrics.get('box_f1', 0.0):.4f} | "
                f"Prec: {batch_metrics.get('box_precision', 0.0):.3f} | "
                f"Rec: {batch_metrics.get('box_recall', 0.0):.3f}"
            )
        else:
            progess_bar.set_description(
                f"Epoch {ep_num} [{mode}] | Loss: {loss_total.item():.4f} | "
                f"RNE: {batch_metrics.get('RNE', 0.0):.4f} | "
                f"mIoU: {batch_metrics.get('miou', 0.0):.4f}"
            )

    # Log epoch statistics
    avg_loss = np.mean(epoch_losses['total'])
    avg_motion = np.mean(epoch_losses['motion'])
    avg_seg = np.mean(epoch_losses['seg'])
    avg_dice = np.mean(epoch_losses['dice'])
    avg_feat = np.mean(epoch_losses['feat'])
    avg_metrics = {key: np.mean(values) if len(values) > 0 else 0.0
                   for key, values in epoch_metrics.items()}
    # Average losses dict for detection components
    avg_losses = {key: np.mean(values) if len(values) > 0 else 0.0
                  for key, values in epoch_losses.items()}

    # ===== COMPUTE DETECTION METRICS (mAP, mIoU) =====
    if len(boxes_pred_accumulated) > 0 and len(boxes_gt_accumulated) > 0:
        try:
            detection_metrics = compute_detection_metrics_epoch(
                boxes_pred_list=boxes_pred_accumulated,
                boxes_gt_list=boxes_gt_accumulated,
                seg_pred_list=seg_pred_accumulated if len(seg_pred_accumulated) > 0 else None,
                seg_gt_list=seg_gt_accumulated if len(seg_gt_accumulated) > 0 else None
            )
            # Add detection metrics to avg_metrics
            avg_metrics.update(detection_metrics)

            # Print detection metrics
            mode_label = mode.upper()
            print_detection_metrics(detection_metrics, prefix=mode_label)

        except Exception as e:
            logger.warning(f'Could not compute detection metrics: {str(e)}')

    # ===== COMPUTE MOT METRICS (REID_ONLY MODE - TRAIN & VAL) =====
    # Skip MOT metrics in detection phase (Phase 1 - only detection metrics)
    mot_metrics = {}
    if train_mode == 'reid_only' and not is_detection_phase:
        if mot_accumulator is not None:
            try:
                mode_label = 'TRAINING' if mode == 'train' else 'VALIDATION'
                logger.info(f'🔄 Computing MOT metrics from temporal tracking ({mode_label})...')
                mot_metrics = mot_accumulator.compute_metrics()
                logger.info(f'MOT metrics computed successfully ({mode_label})')
            except Exception as e:
                import traceback
                logger.error(f'❌ Failed to compute MOT metrics: {e}')
                logger.error(f'Traceback: {traceback.format_exc()}')
                mot_metrics = {}
        else:
            logger.warning('MOT accumulator is None - metrics will not be available')

        # Add MOT metrics to avg_metrics for CSV logging
        if mot_metrics:
            avg_metrics.update(mot_metrics)

    logger.info(f'\n{"="*80}')
    if is_detection_phase:
        logger.info(f'Epoch {ep_num} [{mode.upper()}] SUMMARY (PHASE 1: DETECTION TRAINING)')
    elif train_mode == 'reid_only':
        logger.info(f'Epoch {ep_num} [{mode.upper()}] SUMMARY (PHASE 2: RE-ID TRACKING MODE)')
    elif train_mode == 'full':
        logger.info(f'Epoch {ep_num} [{mode.upper()}] SUMMARY (FULL MODE: SEG + MOTION + REID)')
    else:
        logger.info(f'Epoch {ep_num} [{mode.upper()}] SUMMARY (SEGMENTATION + EGO MOTION)')
    logger.info(f'{"="*80}')
    logger.info(f'LOSSES:')
    logger.info(f'  Total Loss:        {avg_loss:.4f}')

    # Show loss components based on mode
    if is_detection_phase:
        # PHASE 1: Detection losses only
        logger.info(f'  └─ Detection Loss:       {avg_motion:.4f}')
        logger.info(f'\n📉 Detection Loss Components:')
        logger.info(f'  Center Loss:     {avg_losses.get("detection_center", 0.0):.4f}')
        logger.info(f'  Size Loss:       {avg_losses.get("detection_size", 0.0):.4f}')
        logger.info(f'  Orientation Loss:{avg_losses.get("detection_orientation", 0.0):.4f}')
        logger.info(f'  Class Loss:      {avg_losses.get("detection_class", 0.0):.4f}')
    elif train_mode == 'reid_only':
        logger.info(f'  └─ Re-ID Loss:           {avg_motion:.4f}')
    elif train_mode == 'segmentation_only':
        weight_seg = getattr(args, 'weight_seg', 1.0)
        weight_dice = getattr(args, 'weight_dice', 0.3)
        weight_feat = getattr(args, 'weight_feat_contrast', 0.2)
        logger.info(f'  ├─ BCE Seg Loss:         {avg_seg:.4f}  (weight: {weight_seg:.2f})')
        logger.info(f'  ├─ Dice Loss:            {avg_dice:.4f}  (weight: {weight_dice:.2f})')
        logger.info(f'  └─ Feat Contrast Loss:   {avg_feat:.4f}  (weight: {weight_feat:.2f})')
    elif train_mode == 'full':
        weight_seg = getattr(args, 'weight_seg', 1.0)
        weight_motion = getattr(args, 'weight_flow', 0.5)
        weight_dice = getattr(args, 'weight_dice', 0.3)
        weight_feat = getattr(args, 'weight_feat_contrast', 0.2)
        logger.info(f'  ├─ BCE Seg Loss:         {avg_seg:.4f}  (weight: {weight_seg:.2f})')
        logger.info(f'  ├─ Dice Loss:            {avg_dice:.4f}  (weight: {weight_dice:.2f})')
        logger.info(f'  ├─ Feat Contrast Loss:   {avg_feat:.4f}  (weight: {weight_feat:.2f})')
        logger.info(f'  └─ Motion Loss (Flow):   {avg_motion:.4f}  (weight: {weight_motion:.2f})')

    # Show metrics based on mode
    if is_detection_phase:
        # PHASE 1: Detection metrics only (mAP, box_IoU)
        logger.info(f'\nDETECTION METRICS:')
        logger.info(f'  mAP:            {avg_metrics.get("mAP", 0.0):.4f}  ← Mean Average Precision (PRIMARY)')
        logger.info(f'  mAP@0.3:        {avg_metrics.get("mAP@0.3", 0.0):.4f}  ← mAP at IoU=0.3')
        logger.info(f'  mAP@0.5:        {avg_metrics.get("mAP@0.5", 0.0):.4f}  ← mAP at IoU=0.5')
        logger.info(f'  mAP@0.7:        {avg_metrics.get("mAP@0.7", 0.0):.4f}  ← mAP at IoU=0.7')
        logger.info(f'  Box IoU:        {avg_metrics.get("box_iou", 0.0):.4f}  ← Average IoU of matched boxes')
        logger.info(f'  Box Recall:     {avg_metrics.get("box_recall", 0.0):.4f}  ← Detection recall')
        logger.info(f'  Box Precision:  {avg_metrics.get("box_precision", 0.0):.4f}  ← Detection precision')
        logger.info(f'  Box F1:         {avg_metrics.get("box_f1", 0.0):.4f}  ← F1 score')
        logger.info(f'  Avg Pred Boxes: {avg_metrics.get("num_pred_boxes", 0.0):.1f}  ← Predicted boxes per frame')
        logger.info(f'  Avg GT Boxes:   {avg_metrics.get("num_gt_boxes", 0.0):.1f}  ← GT boxes per frame')
    elif train_mode == 'reid_only':
        logger.info(f'\nDETECTION METRICS (Frame-by-Frame):')
        logger.info(f'  Box F1 Score:   {avg_metrics["box_f1"]:.4f}  ← Harmonic mean of Prec & Rec')
        logger.info(f'  Box Precision:  {avg_metrics["box_precision"]:.4f}  ← Ratio of correct predictions')
        logger.info(f'  Box Recall:     {avg_metrics["box_recall"]:.4f}  ← Ratio of detected GT boxes')
        logger.info(f'  Box IoU:        {avg_metrics["box_iou"]:.4f}  ← Average IoU of matched boxes')
        logger.info(f'  Avg Pred Boxes: {avg_metrics["num_pred_boxes"]:.1f}  ← Predicted boxes per frame')
        logger.info(f'  Avg GT Boxes:   {avg_metrics["num_gt_boxes"]:.1f}  ← GT boxes per frame')

        # Show classification metrics (class-aware embeddings)
        if avg_metrics.get('classification_acc', 0.0) > 0:
            logger.info(f'\nCLASSIFICATION METRICS (Class-Aware Embeddings):')
            logger.info(f'  Classification Accuracy: {avg_metrics["classification_acc"]:.4f}  ← vehicle/pedestrian/cyclist/unknown')

        # Show GT flow verification metrics (should be ~0 with GT flow)
        if avg_metrics.get('RNE', 0.0) > 0:
            logger.info(f'\nGT FLOW VERIFICATION (Should be ~0):')
            logger.info(f'  RNE:            {avg_metrics["RNE"]:.4f}  ← Resolution-Normalized Error (GT flow)')
            logger.info(f'  EPE:            {avg_metrics["EPE"]:.4f}m ← End-Point Error (GT flow)')
            logger.info(f'  SAS:            {avg_metrics["SAS"]:.4f}  ← Strict Accuracy Score')
            logger.info(f'  RAS:            {avg_metrics["RAS"]:.4f}  ← Relaxed Accuracy Score')

        # Show MOT metrics if available (training and validation)
        if mot_metrics:
            mode_label = '(TRAINING)' if mode == 'train' else '(VALIDATION)'
            logger.info(f'\nPRIMARY TRACKING METRICS {mode_label} (Standard MOT):')
            logger.info(f'  MOTA:           {mot_metrics["MOTA"]:.2f}%  ↑  ← Multi-Object Tracking Accuracy (PRIMARY)')
            logger.info(f'  IDF1:           {mot_metrics["IDF1"]:.2f}%  ↑  ← ID F1 Score (identity preservation)')
            logger.info(f'  MOTP:           {mot_metrics["MOTP"]:.2f}%  ↑  ← Multi-Object Tracking Precision')

            logger.info(f'\nAVERAGE METRICS (Multi-Threshold):')
            logger.info(f'  sAMOTA:         {mot_metrics["sAMOTA"]:.2f}%  ↑  ← Scaled Average MOTA (nuScenes PRIMARY)')
            logger.info(f'  AMOTA:          {mot_metrics["AMOTA"]:.2f}%  ↑  ← Average MOTA over IoU thresholds')
            logger.info(f'  AMOTP:          {mot_metrics["AMOTP"]:.2f}%  ↑  ← Average MOTP over IoU thresholds')

            logger.info(f'\nDETECTION-ONLY:')
            logger.info(f'  MODA:           {mot_metrics["MODA"]:.2f}%  ↑  ← MOTA without ID switches')

            logger.info(f'\nTRACK QUALITY:')
            logger.info(f'  MT (Mostly Tracked):     {mot_metrics["MT"]:.1f}%  ↑  ← Objects tracked >80% of lifetime')
            logger.info(f'  PT (Partially Tracked):  {mot_metrics["PT"]:.1f}%      ← Objects tracked 20-80%')
            logger.info(f'  ML (Mostly Lost):        {mot_metrics["ML"]:.1f}%  ↓  ← Objects tracked <20% of lifetime')

            logger.info(f'\nERROR BREAKDOWN:')
            logger.info(f'  ID Switches:    {int(mot_metrics["ID_switches"])}  ↓  ← Identity changes')
            logger.info(f'  Fragmentations: {int(mot_metrics["Fragmentations"])}  ↓  ← Track interruptions')
            logger.info(f'  False Positives: {int(mot_metrics["FP"])}  ↓  ← Incorrect detections')
            logger.info(f'  False Negatives: {int(mot_metrics["FN"])}  ↓  ← Missed detections')
    else:
        logger.info(f'\nSCENE FLOW METRICS (PRIMARY):')
        logger.info(f'  RNE:            {avg_metrics["RNE"]:.4f}  ← Resolution-Normalized Error (PRIMARY)')
        logger.info(f'  SAS:            {avg_metrics["SAS"]:.4f}  ← Strict Accuracy Score (10% threshold)')
        logger.info(f'  RAS:            {avg_metrics["RAS"]:.4f}  ← Relaxed Accuracy Score (20% threshold)')
        logger.info(f'  EPE:            {avg_metrics["EPE"]:.4f}m ← End-Point Error (secondary)')
        logger.info(f'\nSEGMENTATION METRICS (Point-Level):')
        logger.info(f'  mIoU:           {avg_metrics["miou"]:.4f}  ← Mean IoU (moving + static) / 2')
        logger.info(f'  F1 Score:       {avg_metrics["f1"]:.4f}  ← Harmonic mean of precision & recall (POINT-LEVEL)')
        logger.info(f'  Accuracy:       {avg_metrics["acc"]:.4f}  ← (TP + TN) / Total')
        logger.info(f'  Sensitivity:    {avg_metrics["sen"]:.4f}  ← Recall for moving class')
        logger.info(f'\nDetailed IoU:')
        logger.info(f'  IoU (moving):   {avg_metrics["IoU_moving"]:.2f}%')
        logger.info(f'  IoU (static):   {avg_metrics["IoU_static"]:.2f}%')
    logger.info(f'{"="*80}\n')

    # ===== CREATE VIDEO FROM VISUALIZATIONS (VALIDATION ONLY) =====
    if mode == 'val' and save_visualizations and vis_dir is not None:
        try:
            epoch_vis_dir = os.path.join(vis_dir, f'epoch_{ep_num:03d}')
            if os.path.exists(epoch_vis_dir):
                # Create video with low FPS for slow playback
                fps = getattr(args, 'video_fps', 2)  # Default 2 FPS (slow)
                video_path = os.path.join(vis_dir, f'epoch_{ep_num:03d}_tracking.mp4')

                logger.info(f'\n🎬 Generating tracking video...')
                success = create_video_from_frames(
                    frames_dir=epoch_vis_dir,
                    output_path=video_path,
                    fps=fps,
                    logger=logger
                )

                if success:
                    logger.info(f'Tracking video generated successfully!')
                    logger.info(f'   📹 Video: {video_path}')
                    logger.info(f'   ⏱️  Speed: {fps} FPS (slow motion for detailed inspection)')
        except Exception as e:
            logger.warning(f'Could not create video: {e}')

    return epoch_losses, avg_metrics


def run_train_simple(args, logger, train_loader, val_loader=None):
    """
    Simplified training following baseline approach.
    Validates every 2 epochs and saves best models by mIoU and F1.
    Generates visualizations during validation like RaTrack baseline.

    Args:
        args: Configuration arguments
        logger: Logger instance
        train_loader: DataLoader for training
        val_loader: DataLoader for validation (optional)
    """
    # Detect training mode
    train_mode = getattr(args, 'train_mode', 'segmentation_only')
    logger.info(f'Training Mode: {train_mode}')

    # Load model
    net = load_model(args, logger)
    net.train()

    # Initialize Re-ID module if in reid mode
    reid_module = None
    if train_mode == 'reid_only':
        try:
            from models.reid_module import ReIDTrackingModule
            logger.info('Initializing Re-ID Tracking Module...')
            reid_module = ReIDTrackingModule(args).cuda()
        except ImportError as e:
            raise ImportError(f"Re-ID module not available. Check models/reid_module.py. Error: {e}")
        reid_module.train()
        logger.info(f'   - Box Proposal: {args.reid.get("box_proposal", {}).get("method", "dbscan")}')
        logger.info(f'   - Embedding dim: {args.reid.get("embedding_dim", 256)}')
        logger.info(f'   - Using GT segmentation: {args.use_gt_segmentation}')
        logger.info(f'   - Using GT ego motion: {args.use_gt_ego_motion}')
        logger.info(f'   - Using GT boxes: {args.use_gt_boxes}')

        if args.use_gt_boxes:
            logger.info(f'')
            logger.info(f'   🎯 GT BOXES MODE ENABLED:')
            logger.info(f'      → Box detection metrics will be PERFECT (F1=1.0, IoU=1.0)')
            logger.info(f'      → Model will ONLY train Re-ID embeddings (triplet loss)')
            logger.info(f'      → Box proposal network is NOT being trained')
        else:
            logger.info(f'')
            logger.info(f'   DBSCAN BOX PROPOSAL MODE:')
            logger.info(f'      → Boxes generated by DBSCAN (NOT trainable)')
            logger.info(f'      → Detection metrics depend on DBSCAN performance')
            logger.info(f'      → May have low recall if objects are small/sparse')

        logger.info(f'')
        logger.info(f'   📍 NOTE: In reid_only mode, scene flow / ego motion is NOT used')
        logger.info(f'      → Re-ID module only needs: point clouds, segmentation, boxes, track IDs')
        logger.info(f'      → use_gt_ego_motion flag has no effect in this mode')
        logger.info(f'      → Training focuses purely on Re-ID embeddings (appearance features)')

    # Setup optimizer
    # Detect detection phase mode
    use_detection_head = getattr(args, 'use_detection_head', False)
    is_detection_phase = use_detection_head and train_mode == 'reid_only'

    if is_detection_phase:
        # PHASE 1: Detection training - optimize main model (includes detection head)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.0001)
        logger.info('Optimizer: Main model (Detection Head) parameters')
    elif train_mode == 'reid_only' and reid_module is not None:
        # PHASE 2: Only optimize Re-ID parameters
        optimizer = torch.optim.Adam(reid_module.parameters(), lr=args.lr, weight_decay=0.0001)
        logger.info('Optimizer: Re-ID module parameters only')
    elif train_mode == 'full' and reid_module is not None:
        # Optimize both main model and Re-ID
        params = list(net.parameters()) + list(reid_module.parameters())
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=0.0001)
        logger.info('Optimizer: Main model + Re-ID module parameters')
    else:
        # Only main model
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.0001)
        logger.info('Optimizer: Main model parameters only')

    # Setup learning rate scheduler (adaptive based on validation metrics)
    scheduler_patience = int(getattr(args, 'scheduler_patience', 3))
    scheduler_factor = float(getattr(args, 'scheduler_factor', 0.5))
    scheduler_min_lr = float(getattr(args, 'scheduler_min_lr', 1e-6))

    # Determine which metric to monitor based on training mode
    if is_detection_phase:
        # PHASE 1: Monitor mAP for detection training
        scheduler_metric = 'mAP'
        scheduler_mode = 'max'  # Maximize mAP
        logger.info(f'LR Scheduler: ReduceLROnPlateau monitoring mAP (maximize)')
        logger.info(f'   → Target: mAP ≥ 0.65 to beat RaTrack baseline')
    elif train_mode == 'reid_only':
        # PHASE 2: Monitor LOSS (not Box F1) because we use GT boxes → Box F1 is always perfect
        # Loss reflects actual training progress (triplet + classification)
        scheduler_metric = 'loss'
        scheduler_mode = 'min'  # Minimize loss
        logger.info(f'LR Scheduler: ReduceLROnPlateau monitoring Total Loss (minimize)')
        logger.info(f'   → Box F1 not used (always perfect with GT boxes)')
        logger.info(f'   → Loss reflects Re-ID training progress (triplet + classification)')
    else:
        scheduler_metric = 'box_f1'  # Monitor instance-level Box F1 for segmentation
        scheduler_mode = 'max'  # Maximize F1
        logger.info(f'LR Scheduler: ReduceLROnPlateau monitoring Instance-Level Box F1')

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=scheduler_mode,  # 'min' for loss, 'max' for metrics
        factor=scheduler_factor,  # Reduce LR by this factor (default 0.5)
        patience=scheduler_patience,  # Wait N epochs without improvement
        verbose=True,
        min_lr=scheduler_min_lr,  # Minimum LR
        threshold=0.001,  # Minimum change to qualify as improvement
        threshold_mode='abs'
    )
    logger.info(f'   - Patience: {scheduler_patience} epochs')
    logger.info(f'   - Reduction factor: {scheduler_factor}')
    logger.info(f'   - Minimum LR: {scheduler_min_lr}')

    # Track best metrics for checkpointing
    best_miou = 0.0
    best_f1 = 0.0  # Point-level F1 for segmentation
    best_box_f1 = 0.0  # Instance-level Box F1 for segmentation mode
    best_mota = 0.0  # MOTA for Re-ID mode
    best_amota = 0.0  # sAMOTA for Re-ID mode (PRIMARY METRIC)
    final_metrics = None

    # Create checkpoint directory
    checkpoint_dir = os.path.join('checkpoints', args.exp_name, 'models')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create visualization directory
    vis_dir = os.path.join('checkpoints', args.exp_name, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # Training loop
    for epoch in range(args.epochs):
        logger.info(f'\n🚀 Starting Epoch {epoch}/{args.epochs-1} | Learning rate: {args.lr:.6f}')

        # ===== DETECTION PRETRAINING vs RE-ID TRAINING STRATEGY (NEW) =====
        if train_mode == 'reid_only' and reid_module is not None:
            reid_config = getattr(args, 'reid', {})
            detection_pretraining_epochs = reid_config.get('detection_pretraining_epochs', 0)

            if epoch == 0 and detection_pretraining_epochs > 0:
                logger.info(f'🎯 DETECTION PRETRAINING PHASE: Epochs 0-{detection_pretraining_epochs-1}')
                logger.info(f'   → Focus: Learn object classification + appearance features')
                logger.info(f'   → Triplet loss: DISABLED')
                logger.info(f'   → Loss: Only classification loss')
            elif epoch == detection_pretraining_epochs:
                logger.info(f'🔄 SWITCHING TO RE-ID TRAINING at Epoch {epoch}')
                logger.info(f'   → Focus: Learn discriminative temporal embeddings')
                logger.info(f'   → Triplet loss: ENABLED (margin = {reid_config.get("triplet_loss", {}).get("margin", 0.5)})')
                logger.info(f'   → Loss: Classification + Triplet loss')

            if epoch < detection_pretraining_epochs:
                logger.info(f'   📍 PHASE: Detection Pretraining [{epoch+1}/{detection_pretraining_epochs}]')
            else:
                logger.info(f'   📍 PHASE: Re-ID Training [{epoch-detection_pretraining_epochs+1}/{args.epochs-detection_pretraining_epochs}]')

        # ===== PROGRESSIVE TRAINING: Freeze/Unfreeze Backbone for Re-ID =====
        if train_mode == 'reid_only' and reid_module is not None:
            reid_config = getattr(args, 'reid', {})
            freeze_epochs = reid_config.get('freeze_backbone_epochs', 0)
            unfreeze_epoch = reid_config.get('unfreeze_backbone_epoch', freeze_epochs)

            if epoch == 0 and freeze_epochs > 0:
                # Freeze backbone at start
                reid_module.reid_extractor.freeze_backbone()
                # Ensure reid_module is in train mode (backbone will stay in eval mode)
                reid_module.train()
                logger.info(f'📌 Progressive Training: Backbone FROZEN for first {freeze_epochs} epochs (warm-up Re-ID head)')
            elif epoch == unfreeze_epoch and freeze_epochs > 0:
                # Unfreeze backbone for fine-tuning
                reid_module.reid_extractor.unfreeze_backbone()
                # Set everything to train mode
                reid_module.train()
                logger.info(f'🔥 Progressive Training: Backbone UNFROZEN at epoch {epoch} (end-to-end fine-tuning)')

        # Determine if pretrain phase (RaTrack: 16/24 epochs = 66% of total)
        # Use pretrain_epochs from config, fallback to RaTrack ratio (66% of total epochs)
        pretrain_epochs = getattr(args, 'pretrain_epochs', int(args.epochs * 0.66))
        pretrain = (epoch < pretrain_epochs and train_mode != 'reid_only')

        if pretrain:
            logger.info(f'PRETRAIN MODE (epoch {epoch}/{pretrain_epochs-1}): ONLY segmentation loss, flow=0 (like RaTrack)')

        # Run training epoch
        epoch_losses, epoch_metrics = run_epoch_simple(
            args, net, train_loader, logger, optimizer,
            mode='train', ep_num=epoch, pretrain=pretrain,
            reid_module=reid_module, train_mode=train_mode
        )

        # Save final metrics
        final_metrics = epoch_metrics

        # ===== SAVE TRAINING METRICS TO CSV =====
        train_csv_path = os.path.join(checkpoint_dir, '..', 'train_metrics.csv')
        save_metrics_to_csv(
            csv_path=train_csv_path,
            epoch=epoch,
            metrics=epoch_metrics,
            losses=epoch_losses,
            lr=args.lr,
            mode='train'
        )

        # ===== GENERATE TRAINING PROGRESS PLOTS =====
        try:
            # Plot function expects the experiment directory (parent of models/)
            exp_dir = os.path.join(checkpoint_dir, '..')
            plot_training_progress(exp_dir, train_mode=train_mode)
            logger.info(f'Training progress plot saved to {os.path.join(exp_dir, "training_progress.png")}')
        except Exception as e:
            logger.warning(f'Could not generate training plot: {e}')

        # ===== VALIDATION (configurable via validation_epochs) =====
        validation_freq = getattr(args, 'validation_epochs', 2)
        if val_loader is not None and epoch % validation_freq == 0 and epoch > 0:
            logger.info(f'\nRunning Validation at Epoch {epoch} (every {validation_freq} epochs)...')

            # Switch to eval mode
            net.eval()

            # Run validation with visualizations
            val_losses, val_metrics = run_epoch_simple(
                args, net, val_loader, logger, optimizer=None,
                mode='val', ep_num=epoch, pretrain=False,
                save_visualizations=True, vis_dir=vis_dir,
                reid_module=reid_module, train_mode=train_mode
            )

            # Switch back to train mode
            net.train()

            # Log validation metrics (mode-dependent)
            logger.info('\n' + "="*80)
            logger.info(f'Epoch {epoch} [VALIDATION] SUMMARY')
            logger.info("="*80)

            if train_mode == 'reid_only':
                # Re-ID mode: show detection metrics
                logger.info('RE-ID DETECTION METRICS:')
                logger.info(f'  Box F1 Score:   {val_metrics["box_f1"]:.4f}  ← PRIMARY')
                logger.info(f'  Box Precision:  {val_metrics["box_precision"]:.4f}')
                logger.info(f'  Box Recall:     {val_metrics["box_recall"]:.4f}')
                logger.info(f'  Box IoU:        {val_metrics["box_iou"]:.4f}')
                logger.info(f'  Avg Pred Boxes: {val_metrics["num_pred_boxes"]:.1f}')
                logger.info(f'  Avg GT Boxes:   {val_metrics["num_gt_boxes"]:.1f}')
            else:
                # Segmentation mode: show scene flow metrics
                logger.info('SCENE FLOW METRICS:')
                logger.info(f'  RNE:            {val_metrics["RNE"]:.4f}')
                logger.info(f'  EPE:            {val_metrics["EPE"]:.2f}m')
                logger.info(f'  SAS:            {val_metrics["SAS"]:.4f}')
                logger.info(f'  RAS:            {val_metrics["RAS"]:.4f}')
                logger.info('')
                logger.info('SEGMENTATION METRICS (Point-Level):')
                logger.info(f'  mIoU:           {val_metrics["miou"]:.4f}')
                logger.info(f'  F1 Score:       {val_metrics["f1"]:.4f}  ← POINT-LEVEL')
                logger.info(f'  Accuracy:       {val_metrics["acc"]:.4f}')
                logger.info(f'  Sensitivity:    {val_metrics["sen"]:.4f}')
                logger.info('')
                logger.info('Detailed IoU:')
                logger.info(f'  IoU (moving):   {val_metrics["IoU_moving"]*100:.2f}%')
                logger.info(f'  IoU (static):   {val_metrics["IoU_static"]*100:.2f}%')
                logger.info('')
                logger.info('INSTANCE-LEVEL DETECTION METRICS:')
                logger.info(f'  Box F1 Score:   {val_metrics["box_f1"]:.4f}  ← INSTANCE-LEVEL (PRIMARY)')
                logger.info(f'  Box Precision:  {val_metrics["box_precision"]:.4f}')
                logger.info(f'  Box Recall:     {val_metrics["box_recall"]:.4f}')
                logger.info(f'  Box IoU:        {val_metrics["box_iou"]:.4f}')
                logger.info(f'  Avg Pred Boxes: {val_metrics["num_pred_boxes"]:.1f}')
                logger.info(f'  Avg GT Boxes:   {val_metrics["num_gt_boxes"]:.1f}')

            logger.info("="*80)

            # ===== SAVE BEST MODEL (mode-dependent) =====
            if train_mode == 'reid_only':
                # Track best MOTA and sAMOTA (for logging, even if we don't save)
                if 'MOTA' in val_metrics and val_metrics['MOTA'] > best_mota:
                    best_mota = val_metrics['MOTA']

                # Re-ID mode: save best by sAMOTA (PRIMARY TRACKING METRIC)
                if 'sAMOTA' in val_metrics and val_metrics['sAMOTA'] > best_amota:
                    best_amota = val_metrics['sAMOTA']

                    checkpoint_path = os.path.join(checkpoint_dir, 'best_samota_model.pth')
                    checkpoint = {
                        'epoch': epoch,
                        'model_state': reid_module.state_dict() if reid_module is not None else net.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'metrics': val_metrics,
                        'args': vars(args)
                    }

                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f'\n🏆 NEW BEST sAMOTA: {best_amota:.2f}%')
                    logger.info(f'Saved best sAMOTA model to: {checkpoint_path}')
            else:
                # Segmentation mode: save best by mIoU and F1
                if val_metrics['miou'] > best_miou:
                    best_miou = val_metrics['miou']

                    checkpoint_path = os.path.join(checkpoint_dir, 'best_miou_model.pth')
                    checkpoint = {
                        'epoch': epoch,
                        'model_state': net.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'metrics': val_metrics,
                        'args': vars(args)
                    }

                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f'\n🏆 NEW BEST mIoU: {best_miou:.4f}')
                    logger.info(f'Saved best mIoU model to: {checkpoint_path}')

                # ===== SAVE BEST MODEL BY F1 (Point-Level) =====
                if val_metrics['f1'] > best_f1:
                    best_f1 = val_metrics['f1']

                    checkpoint_path = os.path.join(checkpoint_dir, 'best_f1_model.pth')
                    checkpoint = {
                        'epoch': epoch,
                        'model_state': net.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'metrics': val_metrics,
                        'args': vars(args)
                    }

                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f'\n🏆 NEW BEST F1 (Point-Level): {best_f1:.4f}')
                    logger.info(f'Saved best F1 model to: {checkpoint_path}')

                # ===== SAVE BEST MODEL BY INSTANCE-LEVEL BOX F1 (NEW) =====
                if 'box_f1' in val_metrics and val_metrics['box_f1'] > best_box_f1:
                    best_box_f1 = val_metrics['box_f1']

                    checkpoint_path = os.path.join(checkpoint_dir, 'best_instance_f1_model.pth')
                    checkpoint = {
                        'epoch': epoch,
                        'model_state': net.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'metrics': val_metrics,
                        'args': vars(args)
                    }

                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f'\n🏆 NEW BEST Box F1 (Instance-Level): {best_box_f1:.4f}')
                    logger.info(f'Saved best instance-level F1 model to: {checkpoint_path}')

            # ===== UPDATE LEARNING RATE SCHEDULER =====
            # Step scheduler based on validation metric
            if scheduler_metric == 'loss':
                # Use validation loss (average of val_losses['total'])
                monitored_metric = np.mean(val_losses['total']) if len(val_losses['total']) > 0 else 0.0
            else:
                # Use validation metric (F1, mIoU, etc.)
                monitored_metric = val_metrics.get(scheduler_metric, 0.0)

            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(monitored_metric)
            new_lr = optimizer.param_groups[0]['lr']

            if new_lr < old_lr:
                logger.info(f'\n📉 Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}')
                logger.info(f'   Reason: {scheduler_metric} did not improve for {scheduler_patience} epochs')
                if scheduler_metric == 'loss':
                    logger.info(f'   Current loss: {monitored_metric:.4f}')

            # Update args.lr for logging in next epoch
            args.lr = new_lr

            # ===== SAVE VALIDATION METRICS TO CSV =====
            val_csv_path = os.path.join(checkpoint_dir, '..', 'val_metrics.csv')
            save_metrics_to_csv(
                csv_path=val_csv_path,
                epoch=epoch,
                metrics=val_metrics,
                losses=val_losses,
                lr=args.lr,
                mode='val'
            )

            # ===== UPDATE TRAINING PROGRESS PLOTS AFTER VALIDATION =====
            try:
                # Plot function expects the experiment directory (parent of models/)
                exp_dir = os.path.join(checkpoint_dir, '..')
                plot_training_progress(exp_dir, train_mode=train_mode)
                logger.info(f'Training progress plot updated with validation data')
            except Exception as e:
                logger.warning(f'Could not update training plot: {e}')

    # Save final model
    checkpoint_path = os.path.join(checkpoint_dir, 'final_model.pth')
    checkpoint = {
        'epoch': args.epochs - 1,
        'model_state': net.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'metrics': final_metrics,
        'best_miou': best_miou,
        'best_f1': best_f1,
        'args': vars(args)
    }

    torch.save(checkpoint, checkpoint_path)
    logger.info(f'\nSaved final model to: {checkpoint_path}')

    # Show final metrics based on training mode
    if train_mode == 'reid_only':
        logger.info(f'   Final metrics: Box F1={final_metrics.get("box_f1", 0.0):.4f}, MOTA={final_metrics.get("MOTA", 0.0):.2f}%, IDF1={final_metrics.get("IDF1", 0.0):.2f}%')
        logger.info(f'\nBest validation metrics:')
        logger.info(f'   Best MOTA:  {best_mota:.2f}%')
        logger.info(f'   Best sAMOTA: {best_amota:.2f}%')
    else:
        logger.info(f'   Final metrics: mIoU={final_metrics.get("miou", 0.0):.4f}, RNE={final_metrics.get("RNE", 0.0):.4f}, F1={final_metrics.get("f1", 0.0):.4f}')
        logger.info(f'\nBest validation metrics:')
        logger.info(f'   Best mIoU: {best_miou:.4f}')
        logger.info(f'   Best F1:   {best_f1:.4f}')

    logger.info('\nTraining completed!')
    return epoch_losses, final_metrics
