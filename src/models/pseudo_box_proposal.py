"""
Pseudo Box Proposal Module
===========================

Generates 3D bounding boxes from predicted segmentation masks using clustering.
This allows training Re-ID without GT boxes, simulating real-world pipeline.

Pipeline:
    1. Predicted Segmentation → Moving Points
    2. DBSCAN Clustering → Object Instances
    3. Bounding Box Fitting → Boxes (center, size, orientation)
    4. Filtering → Remove noise/invalid boxes

Author: Multimodal Tracking System
"""

import numpy as np
import torch
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class PseudoBoxProposal:
    """
    Generates 3D bounding box proposals from predicted segmentation.

    Uses DBSCAN clustering on moving points to identify object instances,
    then fits oriented bounding boxes around each cluster.

    Args:
        eps: DBSCAN epsilon (max distance between points in same cluster)
        min_samples: Minimum points per cluster
        moving_threshold: Threshold for moving/static classification (0-1)
        min_points_per_box: Minimum points required for valid box
        max_points_per_box: Maximum points per box (split large clusters)
    """

    def __init__(
        self,
        eps: float = 1.5,
        min_samples: int = 5,
        moving_threshold: float = 0.5,
        min_points_per_box: int = 10,
        max_points_per_box: int = 500
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.moving_threshold = moving_threshold
        self.min_points_per_box = min_points_per_box
        self.max_points_per_box = max_points_per_box

    def __call__(
        self,
        points: torch.Tensor,
        segmentation_pred: torch.Tensor,
        return_labels: bool = False
    ) -> List[Dict]:
        """
        Generate box proposals from predicted segmentation.

        Args:
            points: [N, 3] point cloud coordinates
            segmentation_pred: [N] predicted moving probabilities (0-1)
            return_labels: If True, return cluster labels for each point

        Returns:
            List of dicts, each containing:
                - 'center': [3] box center (x, y, z)
                - 'size': [3] box dimensions (l, w, h)
                - 'yaw': float, rotation around z-axis
                - 'points_mask': [N] boolean mask of points in this box
                - 'confidence': float, confidence score (0-1)
                - 'num_points': int, number of points in box
        """
        # Convert to numpy
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
        if isinstance(segmentation_pred, torch.Tensor):
            segmentation_pred = segmentation_pred.detach().cpu().numpy()

        # Ensure 1D segmentation
        if segmentation_pred.ndim > 1:
            segmentation_pred = segmentation_pred.squeeze()

        # Step 1: Extract moving points
        moving_mask = segmentation_pred > self.moving_threshold
        moving_points = points[moving_mask]

        if len(moving_points) < self.min_points_per_box:
            logger.debug(f"Too few moving points ({len(moving_points)}), returning empty proposals")
            return []

        # Step 2: DBSCAN clustering
        clustering = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            n_jobs=-1
        )
        cluster_labels = clustering.fit_predict(moving_points)

        # Step 3: Fit boxes for each cluster
        boxes = []
        unique_labels = set(cluster_labels)
        unique_labels.discard(-1)  # Remove noise label

        for label in unique_labels:
            cluster_mask = cluster_labels == label
            cluster_points = moving_points[cluster_mask]

            # Filter by size
            if len(cluster_points) < self.min_points_per_box:
                continue
            if len(cluster_points) > self.max_points_per_box:
                # Large cluster, possibly multiple objects merged
                # Could split it, but for now just skip
                logger.debug(f"Cluster {label} too large ({len(cluster_points)} points), skipping")
                continue

            # Fit oriented bounding box
            box_dict = self._fit_oriented_box(cluster_points)

            if box_dict is not None:
                # Create full point cloud mask
                full_mask = np.zeros(len(points), dtype=bool)
                moving_indices = np.where(moving_mask)[0]
                cluster_indices_in_moving = np.where(cluster_mask)[0]
                full_mask[moving_indices[cluster_indices_in_moving]] = True

                box_dict['points_mask'] = full_mask
                box_dict['num_points'] = len(cluster_points)
                box_dict['confidence'] = segmentation_pred[full_mask].mean()

                boxes.append(box_dict)

        logger.debug(f"Generated {len(boxes)} box proposals from {len(moving_points)} moving points")

        return boxes

    def _fit_oriented_box(self, points: np.ndarray) -> Dict:
        """
        Fit oriented 3D bounding box to point cluster.

        Uses PCA to find principal axes, then fits box aligned to those axes.

        Args:
            points: [N, 3] cluster points

        Returns:
            Dict with 'center', 'size', 'yaw' or None if fitting fails
        """
        if len(points) < 3:
            return None

        try:
            # Compute center
            center = points.mean(axis=0)

            # PCA for orientation (on XY plane only for yaw)
            points_2d = points[:, :2] - center[:2]  # Center in XY

            if len(points_2d) < 2:
                return None

            # Covariance matrix
            cov = np.cov(points_2d.T)

            # Eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(cov)

            # Sort by eigenvalue (largest first)
            idx = eigenvalues.argsort()[::-1]
            eigenvectors = eigenvectors[:, idx]

            # Principal axis (largest eigenvalue)
            principal_axis = eigenvectors[:, 0]

            # Yaw angle (rotation around z-axis)
            yaw = np.arctan2(principal_axis[1], principal_axis[0])

            # Rotate points to axis-aligned frame
            cos_yaw = np.cos(-yaw)
            sin_yaw = np.sin(-yaw)
            rotation_matrix = np.array([
                [cos_yaw, -sin_yaw],
                [sin_yaw, cos_yaw]
            ])

            points_aligned = points_2d @ rotation_matrix.T

            # Compute size (length, width) in aligned frame
            min_xy = points_aligned.min(axis=0)
            max_xy = points_aligned.max(axis=0)
            length = max_xy[0] - min_xy[0]  # Along principal axis
            width = max_xy[1] - min_xy[1]   # Perpendicular

            # Height (z-axis, always aligned)
            min_z = points[:, 2].min()
            max_z = points[:, 2].max()
            height = max_z - min_z

            # Sanity checks
            if length < 0.5 or width < 0.5 or height < 0.5:
                # Too small, probably noise
                return None

            if length > 20.0 or width > 20.0 or height > 5.0:
                # Too large, probably merged objects
                return None

            return {
                'center': center,
                'size': np.array([length, width, height]),
                'yaw': yaw
            }

        except Exception as e:
            logger.warning(f"Failed to fit box: {e}")
            return None


class GTTrackMatcher:
    """
    Matches predicted boxes with GT boxes to assign track IDs.

    Uses Hungarian algorithm with IoU-based cost matrix to find optimal
    one-to-one assignment between predicted and GT boxes.

    Args:
        iou_threshold: Minimum IoU for valid match (default: 0.3)
        distance_threshold: Maximum center distance for valid match (meters)
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        distance_threshold: float = 3.0
    ):
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold

    def match(
        self,
        pred_boxes: List[Dict],
        gt_boxes: List[np.ndarray],
        gt_track_ids: List[int],
        gt_classes: List[int]
    ) -> List[Dict]:
        """
        Match predicted boxes with GT boxes and assign track IDs.

        Args:
            pred_boxes: List of predicted box dicts from PseudoBoxProposal
            gt_boxes: List of GT boxes [M, 7] (x, y, z, l, w, h, yaw)
            gt_track_ids: List of GT track IDs [M]
            gt_classes: List of GT class labels [M]

        Returns:
            List of matched box dicts, each with added:
                - 'track_id': int, matched GT track ID
                - 'class': int, matched GT class
                - 'match_iou': float, IoU with matched GT box
        """
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            return []

        # Compute cost matrix (negative IoU for minimization)
        cost_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))

        for i, pred_box in enumerate(pred_boxes):
            for j, gt_box in enumerate(gt_boxes):
                iou = self._compute_iou_3d(pred_box, gt_box)
                distance = np.linalg.norm(pred_box['center'] - gt_box[:3])

                # Cost = -IoU (we want to maximize IoU)
                # Add penalty for large distance
                cost = -iou + 0.1 * min(distance, 10.0)
                cost_matrix[i, j] = cost

        # Hungarian algorithm (scipy.optimize.linear_sum_assignment)
        from scipy.optimize import linear_sum_assignment
        pred_indices, gt_indices = linear_sum_assignment(cost_matrix)

        # Filter matches by IoU threshold
        matched_boxes = []
        for pred_idx, gt_idx in zip(pred_indices, gt_indices):
            iou = -cost_matrix[pred_idx, gt_idx]  # Convert back to IoU

            if iou >= self.iou_threshold:
                matched_box = pred_boxes[pred_idx].copy()
                matched_box['track_id'] = gt_track_ids[gt_idx]
                matched_box['class'] = gt_classes[gt_idx]
                matched_box['match_iou'] = iou
                matched_boxes.append(matched_box)

        logger.debug(f"Matched {len(matched_boxes)}/{len(pred_boxes)} predicted boxes with GT (IoU >= {self.iou_threshold})")

        return matched_boxes

    def _compute_iou_3d(self, pred_box: Dict, gt_box: np.ndarray) -> float:
        """
        Compute 3D IoU between predicted and GT box.

        Simplified axis-aligned approximation for speed.

        Args:
            pred_box: Dict with 'center' [3], 'size' [3], 'yaw'
            gt_box: [7] array (x, y, z, l, w, h, yaw)

        Returns:
            IoU score (0-1)
        """
        # Extract box parameters
        pred_center = pred_box['center']
        pred_size = pred_box['size']

        gt_center = gt_box[:3]
        gt_size = gt_box[3:6]

        # Axis-aligned approximation (ignore rotation for speed)
        pred_min = pred_center - pred_size / 2
        pred_max = pred_center + pred_size / 2

        gt_min = gt_center - gt_size / 2
        gt_max = gt_center + gt_size / 2

        # Intersection
        inter_min = np.maximum(pred_min, gt_min)
        inter_max = np.minimum(pred_max, gt_max)
        inter_size = np.maximum(0, inter_max - inter_min)
        inter_volume = np.prod(inter_size)

        # Union
        pred_volume = np.prod(pred_size)
        gt_volume = np.prod(gt_size)
        union_volume = pred_volume + gt_volume - inter_volume

        # IoU
        if union_volume > 0:
            return inter_volume / union_volume
        else:
            return 0.0
