"""
Utilities for Pseudo Box Proposal Training
===========================================

Helper functions to integrate PseudoBoxProposal with the training pipeline.
Handles conversion between different data formats and batch processing.

Author: Multimodal Tracking System
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def process_batch_with_pseudo_boxes(
    pc1_batch: torch.Tensor,
    seg_pred_batch: torch.Tensor,
    gt_boxes_batch: List,
    gt_track_ids_batch: List,
    gt_classes_batch: List,
    box_proposal_module,
    gt_matcher_module,
    device: str = 'cuda'
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict]]:
    """
    Process batch to generate pseudo boxes and match with GT.

    Args:
        pc1_batch: [B, 3, N] point cloud (channel-first)
        seg_pred_batch: [B, 1, N] predicted segmentation probabilities
        gt_boxes_batch: List[B] of GT boxes per sample
        gt_track_ids_batch: List[B] of GT track IDs per sample
        gt_classes_batch: List[B] of GT classes per sample
        box_proposal_module: PseudoBoxProposal instance
        gt_matcher_module: GTTrackMatcher instance
        device: Device to put tensors on

    Returns:
        boxes_list: List[B] of matched boxes (as tensors [M, 7])
        track_ids_list: List[B] of matched track IDs (as tensors [M])
        boxes_info_list: List[B] of box info dicts (with masks, confidence, etc.)
    """
    batch_size = pc1_batch.shape[0]
    num_points = pc1_batch.shape[2]

    boxes_list = []
    track_ids_list = []
    boxes_info_list = []

    for b in range(batch_size):
        # Get single sample
        pc1 = pc1_batch[b].permute(1, 0)  # [3, N] → [N, 3]
        seg_pred = seg_pred_batch[b].squeeze(0)  # [1, N] → [N]

        gt_boxes = gt_boxes_batch[b] if b < len(gt_boxes_batch) else []
        gt_track_ids = gt_track_ids_batch[b] if b < len(gt_track_ids_batch) else []
        gt_classes = gt_classes_batch[b] if b < len(gt_classes_batch) else []

        # Step 1: Generate pseudo boxes from predicted segmentation
        pred_boxes = box_proposal_module(
            points=pc1,
            segmentation_pred=seg_pred,
            return_labels=False
        )

        if len(pred_boxes) == 0:
            logger.debug(f"Batch {b}: No boxes generated from segmentation")
            boxes_list.append(torch.empty(0, 7, device=device))
            track_ids_list.append(torch.empty(0, dtype=torch.long, device=device))
            boxes_info_list.append([])
            continue

        # Step 2: Match with GT boxes to assign track IDs
        if len(gt_boxes) > 0:
            matched_boxes = gt_matcher_module.match(
                pred_boxes=pred_boxes,
                gt_boxes=gt_boxes,
                gt_track_ids=gt_track_ids,
                gt_classes=gt_classes
            )
        else:
            matched_boxes = []

        if len(matched_boxes) == 0:
            logger.debug(f"Batch {b}: No matches found (generated {len(pred_boxes)} proposals, {len(gt_boxes)} GT boxes)")
            boxes_list.append(torch.empty(0, 7, device=device))
            track_ids_list.append(torch.empty(0, dtype=torch.long, device=device))
            boxes_info_list.append([])
            continue

        # Step 3: Convert to tensors
        boxes_tensor = []
        track_ids_tensor = []

        for matched_box in matched_boxes:
            # Box format: [x, y, z, l, w, h, yaw]
            box = np.concatenate([
                matched_box['center'],       # [x, y, z]
                matched_box['size'],         # [l, w, h]
                [matched_box['yaw']]         # [yaw]
            ])
            boxes_tensor.append(box)
            track_ids_tensor.append(matched_box['track_id'])

        boxes_tensor = torch.tensor(np.array(boxes_tensor), dtype=torch.float32, device=device)
        track_ids_tensor = torch.tensor(track_ids_tensor, dtype=torch.long, device=device)

        boxes_list.append(boxes_tensor)
        track_ids_list.append(track_ids_tensor)
        boxes_info_list.append(matched_boxes)

        logger.debug(f"Batch {b}: Matched {len(matched_boxes)} boxes (from {len(pred_boxes)} proposals, {len(gt_boxes)} GT)")

    return boxes_list, track_ids_list, boxes_info_list


def extract_box_points(
    pc1: torch.Tensor,
    boxes_info: List[Dict],
    num_points_target: int = 512
) -> Tuple[List[torch.Tensor], List[int]]:
    """
    Extract points inside each box for Re-ID embedding.

    Args:
        pc1: [N, 3] point cloud
        boxes_info: List of box dicts with 'points_mask' key
        num_points_target: Target number of points per box

    Returns:
        box_points_list: List[M] of point tensors [num_points, 3]
        track_ids_list: List[M] of track IDs
    """
    box_points_list = []
    track_ids_list = []

    for box in boxes_info:
        mask = box['points_mask']  # Boolean numpy array [N]

        if isinstance(mask, np.ndarray):
            mask = torch.tensor(mask, device=pc1.device)

        # Extract points
        if isinstance(pc1, torch.Tensor):
            points = pc1[mask]  # [num_points_in_box, 3]
        else:
            points = torch.tensor(pc1[mask.cpu().numpy()], device='cuda')

        # Resample to fixed size (if needed)
        num_points = points.shape[0]

        if num_points == 0:
            continue

        if num_points > num_points_target:
            # Downsample (random)
            indices = torch.randperm(num_points, device=points.device)[:num_points_target]
            points = points[indices]
        elif num_points < num_points_target:
            # Upsample (repeat with noise)
            repeat_factor = (num_points_target + num_points - 1) // num_points
            points = points.repeat(repeat_factor, 1)[:num_points_target]

            # Add small noise to avoid identical points
            noise = torch.randn_like(points) * 0.01
            points = points + noise

        box_points_list.append(points)
        track_ids_list.append(box['track_id'])

    return box_points_list, track_ids_list


def visualize_boxes_comparison(
    pc1: np.ndarray,
    pred_boxes: List[Dict],
    gt_boxes: np.ndarray,
    matched_boxes: List[Dict],
    save_path: str = None
) -> None:
    """
    Visualize predicted boxes vs GT boxes for debugging.

    Args:
        pc1: [N, 3] point cloud
        pred_boxes: List of predicted box dicts
        gt_boxes: [M, 7] GT boxes
        matched_boxes: List of matched box dicts
        save_path: Path to save visualization (optional)
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(15, 5))

        # Plot 1: Point cloud with predicted boxes
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(pc1[:, 0], pc1[:, 1], pc1[:, 2], c='gray', s=1, alpha=0.3)
        for box in pred_boxes:
            center = box['center']
            ax1.scatter([center[0]], [center[1]], [center[2]], c='red', s=50, marker='x')
        ax1.set_title(f'Predicted Boxes ({len(pred_boxes)})')

        # Plot 2: Point cloud with GT boxes
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(pc1[:, 0], pc1[:, 1], pc1[:, 2], c='gray', s=1, alpha=0.3)
        for box in gt_boxes:
            center = box[:3]
            ax2.scatter([center[0]], [center[1]], [center[2]], c='blue', s=50, marker='o')
        ax2.set_title(f'GT Boxes ({len(gt_boxes)})')

        # Plot 3: Matched boxes
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(pc1[:, 0], pc1[:, 1], pc1[:, 2], c='gray', s=1, alpha=0.3)
        for box in matched_boxes:
            center = box['center']
            ax3.scatter([center[0]], [center[1]], [center[2]], c='green', s=50, marker='*')
        ax3.set_title(f'Matched Boxes ({len(matched_boxes)})')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"Saved box comparison to {save_path}")
        else:
            plt.show()

        plt.close()

    except Exception as e:
        logger.warning(f"Failed to visualize boxes: {e}")
