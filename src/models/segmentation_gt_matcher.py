"""
Segmentation-GT Matcher (Simple & Direct)
=========================================

Simple box proposal for Re-ID evaluation:
  1. Use predicted segmentation → points with predicted class
  2. Match segmented points with GT boxes
  3. Return GT boxes that have enough segmented points

NO DBSCAN, NO clustering - just direct point-in-box matching.

Pipeline:
  - For each GT box: count how many segmented points fall inside
  - If enough points → return GT box + GT track ID + GT class
  - This evaluates Re-ID with predicted segmentation but GT boxes/IDs

Author: Multimodal Tracking System
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SegmentationGTMatcher(nn.Module):
    """
    Direct matching between predicted segmentation and GT boxes.

    Returns GT boxes that have sufficient segmented points inside them.
    No DBSCAN, no clustering - just point-in-box matching.

    Args:
        min_points_per_box: Minimum segmented points required for valid box
        moving_threshold: Segmentation confidence threshold
    """

    def __init__(self, min_points_per_box=5, moving_threshold=0.3):
        super().__init__()
        self.min_points_per_box = min_points_per_box
        self.moving_threshold = moving_threshold

        logger.info(f"SegmentationGTMatcher initialized: "
                   f"min_points={min_points_per_box}, "
                   f"threshold={moving_threshold}")

    def forward(
        self,
        points: torch.Tensor,
        seg_pred: torch.Tensor,
        gt_boxes: Optional[List] = None,
        gt_track_ids: Optional[List] = None,
        gt_classes: Optional[List] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict]]:
        """
        Match predicted segmentation with GT boxes.

        Args:
            points: [B, N, 3] or [B, 3, N] point cloud
            seg_pred: [B, N] or [B, 1, N] predicted segmentation probabilities
            gt_boxes: List[B] of GT boxes (required)
            gt_track_ids: List[B] of GT track IDs (required)
            gt_classes: List[B] of GT classes (required)

        Returns:
            boxes_list: List[B] of box tensors [M, 7] (x, y, z, l, w, h, yaw)
            track_ids_list: List[B] of track ID tensors [M]
            boxes_info_list: List[B] of box info dicts (with points_mask)
        """
        batch_size = points.shape[0]

        # Ensure correct point cloud format [B, N, 3]
        if points.shape[1] == 3 and points.shape[2] != 3:
            points = points.permute(0, 2, 1)  # [B, 3, N] → [B, N, 3]

        # Ensure segmentation format [B, N]
        if seg_pred.dim() == 3:
            seg_pred = seg_pred.squeeze(1)  # [B, 1, N] → [B, N]

        # Apply moving threshold to get binary mask
        seg_mask = (seg_pred > self.moving_threshold)  # [B, N]

        boxes_list = []
        track_ids_list = []
        boxes_info_list = []

        total_gt_boxes = 0
        total_matched_boxes = 0

        for b in range(batch_size):
            pts = points[b]  # [N, 3]
            mask = seg_mask[b]  # [N]

            # Get GT boxes for this sample
            gt_boxes_b = gt_boxes[b] if b < len(gt_boxes) else []
            gt_ids_b = gt_track_ids[b] if b < len(gt_track_ids) else []
            gt_classes_b = gt_classes[b] if b < len(gt_classes) else []

            # Convert GT boxes to array format
            gt_boxes_array, gt_ids_array, gt_classes_array = self._extract_gt_boxes(
                gt_boxes_b, gt_ids_b, gt_classes_b
            )

            total_gt_boxes += len(gt_boxes_array)

            if len(gt_boxes_array) == 0:
                boxes_list.append(torch.empty(0, 7, device=points.device))
                track_ids_list.append(torch.empty(0, dtype=torch.long, device=points.device))
                boxes_info_list.append([])
                continue

            # Match: which GT boxes have segmented points?
            matched_boxes, matched_ids, matched_info = self._match_points_to_gt_boxes(
                pts, mask, gt_boxes_array, gt_ids_array, gt_classes_array, points.device
            )

            total_matched_boxes += len(matched_boxes)

            boxes_list.append(matched_boxes)
            track_ids_list.append(matched_ids)
            boxes_info_list.append(matched_info)

        # Log statistics
        if total_gt_boxes > 0:
            match_rate = total_matched_boxes / total_gt_boxes * 100
            logger.debug(f"GT boxes: {total_gt_boxes}, "
                        f"matched with segmentation: {total_matched_boxes} ({match_rate:.1f}%)")

        return boxes_list, track_ids_list, boxes_info_list

    def _match_points_to_gt_boxes(
        self,
        points: torch.Tensor,
        seg_mask: torch.Tensor,
        gt_boxes: List[np.ndarray],
        gt_ids: List[int],
        gt_classes: List[int],
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
        """
        For each GT box, count segmented points inside.
        Return boxes with sufficient points.

        Args:
            points: [N, 3] point cloud
            seg_mask: [N] binary segmentation mask
            gt_boxes: List of GT box arrays [7]
            gt_ids: List of GT track IDs
            gt_classes: List of GT classes
            device: torch device

        Returns:
            boxes: [M, 7] tensor of matched GT boxes
            track_ids: [M] tensor of GT track IDs
            boxes_info: List[M] of box dicts with points_mask
        """
        matched_boxes = []
        matched_ids = []
        matched_info = []

        # Get segmented points only
        seg_points = points[seg_mask]  # [K, 3] where K = num segmented points

        if len(seg_points) == 0:
            return (torch.empty(0, 7, device=device),
                   torch.empty(0, dtype=torch.long, device=device),
                   [])

        for i, box in enumerate(gt_boxes):
            # box format: [x, y, z, l, w, h, yaw]
            center = box[:3]
            size = box[3:6]
            yaw = box[6]

            # Find points inside this box
            points_mask_full = self._point_in_box(
                points.cpu().numpy(),
                center,
                size,
                yaw
            )  # [N] boolean array

            # Count segmented points inside this box
            points_in_box_and_segmented = points_mask_full & seg_mask.cpu().numpy()
            num_seg_points = points_in_box_and_segmented.sum()

            if num_seg_points >= self.min_points_per_box:
                # This GT box has enough segmented points → include it
                matched_boxes.append(box)
                matched_ids.append(gt_ids[i])

                # Create box info dict
                box_info = {
                    'center': center,
                    'size': size,
                    'yaw': yaw,
                    'track_id': gt_ids[i],
                    'class': gt_classes[i] if i < len(gt_classes) else 0,
                    'points_mask': points_in_box_and_segmented,
                    'num_points': int(num_seg_points),
                    'confidence': 1.0
                }
                matched_info.append(box_info)

        # Convert to tensors
        if len(matched_boxes) == 0:
            return (torch.empty(0, 7, device=device),
                   torch.empty(0, dtype=torch.long, device=device),
                   [])

        boxes_tensor = torch.tensor(
            np.array(matched_boxes),
            dtype=torch.float32,
            device=device
        )
        ids_tensor = torch.tensor(
            matched_ids,
            dtype=torch.long,
            device=device
        )

        return boxes_tensor, ids_tensor, matched_info

    def _point_in_box(
        self,
        points: np.ndarray,
        center: np.ndarray,
        size: np.ndarray,
        yaw: float
    ) -> np.ndarray:
        """
        Check which points are inside a 3D oriented bounding box.

        Args:
            points: [N, 3] point cloud
            center: [3] box center (x, y, z)
            size: [3] box size (l, w, h)
            yaw: rotation angle around z-axis

        Returns:
            mask: [N] boolean array (True = point inside box)
        """
        # Translate points to box frame
        points_local = points - center[np.newaxis, :]  # [N, 3]

        # Rotate points to box-aligned frame
        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)

        # Rotation matrix (2D in XY plane)
        x_rot = points_local[:, 0] * cos_yaw - points_local[:, 1] * sin_yaw
        y_rot = points_local[:, 0] * sin_yaw + points_local[:, 1] * cos_yaw
        z_rot = points_local[:, 2]

        # Check if inside axis-aligned box
        half_size = size / 2.0

        inside_x = np.abs(x_rot) <= half_size[0]
        inside_y = np.abs(y_rot) <= half_size[1]
        inside_z = np.abs(z_rot) <= half_size[2]

        mask = inside_x & inside_y & inside_z

        return mask

    def _extract_gt_boxes(
        self,
        gt_boxes: any,
        gt_track_ids: any,
        gt_classes: any
    ) -> Tuple[List[np.ndarray], List[int], List[int]]:
        """
        Extract GT boxes from various formats (dict, list, etc.)

        Returns:
            boxes: List of box arrays [7] (x, y, z, l, w, h, yaw)
            ids: List of track IDs
            classes: List of class IDs
        """
        boxes_list = []
        ids_list = []
        classes_list = []

        # Handle dict format (Open3D boxes)
        if isinstance(gt_boxes, dict):
            for obj_id, box in gt_boxes.items():
                # Extract box parameters
                center = box.get_center()
                extent = box.extent
                R = np.array(box.R)

                # Extract yaw from rotation matrix
                yaw = np.arctan2(R[1, 0], R[0, 0])

                # Create box array
                box_array = np.array([
                    center[0], center[1], center[2],
                    extent[0], extent[1], extent[2],
                    yaw
                ])

                boxes_list.append(box_array)
                ids_list.append(obj_id)

                # Get class
                if isinstance(gt_classes, dict) and obj_id in gt_classes:
                    classes_list.append(gt_classes[obj_id])
                else:
                    classes_list.append(0)

        # Handle list format
        elif isinstance(gt_boxes, list):
            for i, box in enumerate(gt_boxes):
                if isinstance(box, np.ndarray):
                    boxes_list.append(box)
                elif isinstance(box, torch.Tensor):
                    boxes_list.append(box.cpu().numpy())
                else:
                    boxes_list.append(np.array(box))

                # Get ID
                if isinstance(gt_track_ids, list) and i < len(gt_track_ids):
                    ids_list.append(gt_track_ids[i])
                else:
                    ids_list.append(i)

                # Get class
                if isinstance(gt_classes, list) and i < len(gt_classes):
                    classes_list.append(gt_classes[i])
                else:
                    classes_list.append(0)

        return boxes_list, ids_list, classes_list


def extract_box_points(
    points: torch.Tensor,
    boxes_info: List[Dict],
    num_points_target: int = 512
) -> List[torch.Tensor]:
    """
    Extract points inside each box for Re-ID embedding.

    Args:
        points: [N, 3] point cloud
        boxes_info: List of box dicts with 'points_mask' key
        num_points_target: Target number of points per box

    Returns:
        box_points_list: List[M] of point tensors [num_points, 3]
    """
    box_points_list = []

    for box in boxes_info:
        if not isinstance(box, dict) or 'points_mask' not in box:
            continue

        mask = box['points_mask']  # Boolean numpy array [N]

        if isinstance(mask, np.ndarray):
            mask = torch.tensor(mask, device=points.device)

        # Extract points
        if isinstance(points, torch.Tensor):
            box_points = points[mask]  # [num_points_in_box, 3]
        else:
            box_points = torch.tensor(
                points[mask.cpu().numpy()],
                device=points.device if torch.is_tensor(points) else 'cuda'
            )

        num_points = box_points.shape[0]

        if num_points == 0:
            continue

        # Resample to fixed size
        if num_points > num_points_target:
            # Downsample (random)
            indices = torch.randperm(num_points, device=box_points.device)[:num_points_target]
            box_points = box_points[indices]
        elif num_points < num_points_target:
            # Upsample (repeat with noise)
            repeat_factor = (num_points_target + num_points - 1) // num_points
            box_points = box_points.repeat(repeat_factor, 1)[:num_points_target]

            # Add small noise to avoid identical points
            noise = torch.randn_like(box_points) * 0.01
            box_points = box_points + noise

        box_points_list.append(box_points)

    return box_points_list
