"""
Improved Box Proposal System (Hybrid)
=====================================

Combines the best of both implementations:
  1. Robust box fitting from BoxProposalNetwork
  2. Hungarian matching from GTTrackMatcher

Used for Re-ID training with predicted boxes (not GT boxes).

Pipeline:
  - TRAINING: Generate boxes → Match with GT → Return matched boxes + GT IDs
  - EVALUATION: Generate boxes → Return boxes (no GT matching)

Author: Multimodal Tracking System
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

from .box_proposal import BoxProposalNetwork
from .pseudo_box_proposal import GTTrackMatcher

logger = logging.getLogger(__name__)


class ImprovedBoxProposal(nn.Module):
    """
    Hybrid box proposal system that combines:
      - Robust DBSCAN clustering + PCA box fitting
      - Hungarian matching with GT (for training only)

    Auto-detects mode based on GT availability:
      - If gt_boxes provided → TRAINING mode (match with GT)
      - If gt_boxes is None → EVALUATION mode (no matching)

    Args:
        args: Configuration object with box_proposal and gt_matching configs
    """

    def __init__(self, args):
        super().__init__()

        # Get box proposal config
        box_config = getattr(args, 'box_proposal', {})
        if isinstance(box_config, dict):
            self.eps = box_config.get('eps', 2.0)
            self.min_samples = box_config.get('min_samples', 5)
            self.moving_threshold = box_config.get('moving_threshold', 0.5)
            self.min_points_per_box = box_config.get('min_points_per_box', 10)
        else:
            self.eps = getattr(box_config, 'eps', 2.0)
            self.min_samples = getattr(box_config, 'min_samples', 5)
            self.moving_threshold = getattr(box_config, 'moving_threshold', 0.5)
            self.min_points_per_box = getattr(box_config, 'min_points_per_box', 10)

        # Get GT matching config
        gt_config = getattr(args, 'gt_matching', {})
        if isinstance(gt_config, dict):
            self.iou_threshold = gt_config.get('iou_threshold', 0.3)
            self.distance_threshold = gt_config.get('distance_threshold', 3.0)
            self.matching_enabled = gt_config.get('enabled', True)
        else:
            self.iou_threshold = getattr(gt_config, 'iou_threshold', 0.3)
            self.distance_threshold = getattr(gt_config, 'distance_threshold', 3.0)
            self.matching_enabled = getattr(gt_config, 'enabled', True)

        # Initialize box generator (robust DBSCAN + PCA)
        self.box_generator = BoxProposalNetwork(
            method='dbscan',
            eps=self.eps,
            min_samples=self.min_samples,
            embedding_dim=256
        )

        # Initialize GT matcher (Hungarian algorithm)
        if self.matching_enabled:
            self.gt_matcher = GTTrackMatcher(
                iou_threshold=self.iou_threshold,
                distance_threshold=self.distance_threshold
            )

        logger.info(f"ImprovedBoxProposal initialized: eps={self.eps}, min_samples={self.min_samples}, "
                   f"moving_threshold={self.moving_threshold}, iou_threshold={self.iou_threshold}")

    def forward(
        self,
        points: torch.Tensor,
        seg_pred: torch.Tensor,
        gt_boxes: Optional[List] = None,
        gt_track_ids: Optional[List] = None,
        gt_classes: Optional[List] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict]]:
        """
        Generate box proposals and optionally match with GT.

        Args:
            points: [B, 3, N] or [B, N, 3] point cloud
            seg_pred: [B, 1, N] or [B, N] predicted segmentation probabilities
            gt_boxes: List[B] of GT boxes (optional, for training)
            gt_track_ids: List[B] of GT track IDs (optional, for training)
            gt_classes: List[B] of GT classes (optional, for training)

        Returns:
            boxes_list: List[B] of box tensors [M, 7] (x, y, z, l, w, h, yaw)
            track_ids_list: List[B] of track ID tensors [M] (or None if eval mode)
            boxes_info_list: List[B] of box info dicts (metadata)
        """
        batch_size = points.shape[0]

        # Ensure correct point cloud format [B, N, 3]
        if points.shape[1] == 3 and points.shape[2] != 3:
            points = points.permute(0, 2, 1)  # [B, 3, N] → [B, N, 3]

        # Ensure segmentation format [B, N]
        if seg_pred.dim() == 3:
            seg_pred = seg_pred.squeeze(1)  # [B, 1, N] → [B, N]

        # Apply moving threshold to get binary mask
        seg_mask = (seg_pred > self.moving_threshold).long()

        # Generate boxes using robust DBSCAN
        boxes_batch, cluster_ids_batch = self.box_generator(points, seg_mask)

        # Check if GT is available (TRAINING mode)
        training_mode = (gt_boxes is not None and
                        gt_track_ids is not None and
                        self.matching_enabled)

        if training_mode:
            return self._process_with_gt_matching(
                points, boxes_batch, cluster_ids_batch,
                gt_boxes, gt_track_ids, gt_classes, batch_size
            )
        else:
            return self._process_without_gt(
                points, boxes_batch, cluster_ids_batch, batch_size
            )

    def _process_with_gt_matching(
        self,
        points: torch.Tensor,
        boxes_batch: List[torch.Tensor],
        cluster_ids_batch: List[torch.Tensor],
        gt_boxes: List,
        gt_track_ids: List,
        gt_classes: List,
        batch_size: int
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict]]:
        """
        TRAINING mode: Match predicted boxes with GT to assign track IDs.
        """
        matched_boxes_list = []
        matched_ids_list = []
        boxes_info_list = []

        total_generated = 0
        total_matched = 0

        for b in range(batch_size):
            pred_boxes_tensor = boxes_batch[b]  # [M, 7]

            if len(pred_boxes_tensor) == 0:
                matched_boxes_list.append(torch.empty(0, 7, device=points.device))
                matched_ids_list.append(torch.empty(0, dtype=torch.long, device=points.device))
                boxes_info_list.append([])
                continue

            # Convert tensor boxes to dict format for GT matcher
            pred_boxes_dicts = self._tensor_boxes_to_dicts(
                pred_boxes_tensor,
                cluster_ids_batch[b],
                points[b]
            )

            total_generated += len(pred_boxes_dicts)

            # Get GT for this sample
            gt_boxes_b = gt_boxes[b] if b < len(gt_boxes) else []
            gt_ids_b = gt_track_ids[b] if b < len(gt_track_ids) else []
            gt_classes_b = gt_classes[b] if b < len(gt_classes) else []

            # Convert GT boxes from Open3D dict format to array list
            gt_boxes_array, gt_ids_array, gt_classes_array = self._convert_o3d_boxes_to_arrays(
                gt_boxes_b, gt_ids_b, gt_classes_b
            )

            if len(gt_boxes_array) == 0 or len(pred_boxes_dicts) == 0:
                matched_boxes_list.append(torch.empty(0, 7, device=points.device))
                matched_ids_list.append(torch.empty(0, dtype=torch.long, device=points.device))
                boxes_info_list.append([])
                continue

            # Hungarian matching with GT
            matched_boxes = self.gt_matcher.match(
                pred_boxes=pred_boxes_dicts,
                gt_boxes=gt_boxes_array,
                gt_track_ids=gt_ids_array,
                gt_classes=gt_classes_array
            )

            total_matched += len(matched_boxes)

            if len(matched_boxes) == 0:
                matched_boxes_list.append(torch.empty(0, 7, device=points.device))
                matched_ids_list.append(torch.empty(0, dtype=torch.long, device=points.device))
                boxes_info_list.append([])
                continue

            # Convert matched dicts back to tensors
            boxes_tensor, ids_tensor = self._matched_dicts_to_tensors(
                matched_boxes, points.device
            )

            matched_boxes_list.append(boxes_tensor)
            matched_ids_list.append(ids_tensor)
            boxes_info_list.append(matched_boxes)

        # Log statistics
        if total_generated > 0:
            match_rate = total_matched / total_generated * 100
            logger.debug(f"[TRAINING] Generated {total_generated} boxes, "
                        f"matched {total_matched} ({match_rate:.1f}%)")

        return matched_boxes_list, matched_ids_list, boxes_info_list

    def _process_without_gt(
        self,
        points: torch.Tensor,
        boxes_batch: List[torch.Tensor],
        cluster_ids_batch: List[torch.Tensor],
        batch_size: int
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict]]:
        """
        EVALUATION mode: Return predicted boxes without GT matching.
        """
        boxes_list = []
        boxes_info_list = []

        total_generated = 0

        for b in range(batch_size):
            pred_boxes_tensor = boxes_batch[b]  # [M, 7]
            total_generated += len(pred_boxes_tensor)

            # Convert to dict format for consistency
            pred_boxes_dicts = self._tensor_boxes_to_dicts(
                pred_boxes_tensor,
                cluster_ids_batch[b],
                points[b]
            )

            boxes_list.append(pred_boxes_tensor)
            boxes_info_list.append(pred_boxes_dicts)

        logger.debug(f"[EVALUATION] Generated {total_generated} box proposals")

        # Return None for track IDs in eval mode
        track_ids_list = [None] * batch_size

        return boxes_list, track_ids_list, boxes_info_list

    def _tensor_boxes_to_dicts(
        self,
        boxes_tensor: torch.Tensor,
        cluster_ids: torch.Tensor,
        points: torch.Tensor
    ) -> List[Dict]:
        """
        Convert tensor boxes [M, 7] to dict format for GT matching.
        """
        boxes_dicts = []

        for i, box in enumerate(boxes_tensor):
            # Create points mask for this box
            points_mask = (cluster_ids == i).cpu().numpy()
            num_points = points_mask.sum()

            box_dict = {
                'center': box[:3].cpu().numpy(),
                'size': box[3:6].cpu().numpy(),
                'yaw': box[6].item(),
                'points_mask': points_mask,
                'num_points': int(num_points),
                'confidence': 1.0  # Default confidence
            }
            boxes_dicts.append(box_dict)

        return boxes_dicts

    def _matched_dicts_to_tensors(
        self,
        matched_boxes: List[Dict],
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert matched box dicts to tensors.
        """
        boxes_list = []
        ids_list = []

        for box_dict in matched_boxes:
            # Box format: [x, y, z, l, w, h, yaw]
            box = np.concatenate([
                box_dict['center'],      # [x, y, z]
                box_dict['size'],        # [l, w, h]
                [box_dict['yaw']]        # [yaw]
            ])
            boxes_list.append(box)
            ids_list.append(box_dict['track_id'])

        boxes_tensor = torch.tensor(
            np.array(boxes_list),
            dtype=torch.float32,
            device=device
        )
        ids_tensor = torch.tensor(
            ids_list,
            dtype=torch.long,
            device=device
        )

        return boxes_tensor, ids_tensor

    def _convert_o3d_boxes_to_arrays(
        self,
        gt_boxes: any,
        gt_track_ids: any,
        gt_classes: any
    ) -> Tuple[List[np.ndarray], List[int], List[int]]:
        """
        Convert GT boxes from Open3D dict format to array list format.

        Args:
            gt_boxes: dict{obj_id: Open3D_Box} or list of arrays
            gt_track_ids: list of track IDs or dict
            gt_classes: list of classes or dict

        Returns:
            boxes_array_list: List of box arrays [7] (x, y, z, l, w, h, yaw)
            ids_list: List of track IDs
            classes_list: List of classes
        """
        boxes_array_list = []
        ids_list = []
        classes_list = []

        # Check if gt_boxes is a dict (Open3D format)
        if isinstance(gt_boxes, dict):
            # Format: {obj_id: Open3D_Box}
            for obj_id, box in gt_boxes.items():
                # Extract box parameters from Open3D box
                center = box.get_center()  # [3]
                extent = box.extent  # [3] (l, w, h)
                R = np.array(box.R)  # [3, 3] rotation matrix

                # Extract yaw from rotation matrix
                yaw = np.arctan2(R[2, 0], R[0, 0])

                # Create box array [x, y, z, l, w, h, yaw]
                box_array = np.array([
                    center[0], center[1], center[2],
                    extent[0], extent[1], extent[2],
                    yaw
                ])

                boxes_array_list.append(box_array)
                ids_list.append(obj_id)

                # Get class if available
                if isinstance(gt_classes, dict) and obj_id in gt_classes:
                    classes_list.append(gt_classes[obj_id])
                elif isinstance(gt_classes, list) and len(gt_classes) > len(classes_list):
                    classes_list.append(gt_classes[len(classes_list)])
                else:
                    classes_list.append(0)  # Default class

        elif isinstance(gt_boxes, list) and len(gt_boxes) > 0:
            # Already in array format
            for i, box in enumerate(gt_boxes):
                if isinstance(box, np.ndarray):
                    boxes_array_list.append(box)
                elif isinstance(box, torch.Tensor):
                    boxes_array_list.append(box.cpu().numpy())
                else:
                    # Try to convert to array
                    boxes_array_list.append(np.array(box))

                # Get track ID
                if isinstance(gt_track_ids, list) and i < len(gt_track_ids):
                    ids_list.append(gt_track_ids[i])
                else:
                    ids_list.append(i)

                # Get class
                if isinstance(gt_classes, list) and i < len(gt_classes):
                    classes_list.append(gt_classes[i])
                else:
                    classes_list.append(0)

        return boxes_array_list, ids_list, classes_list


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
