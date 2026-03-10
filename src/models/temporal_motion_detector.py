"""
Temporal Motion Detector
========================

Full inference pipeline for box detection using:
  1. Predicted segmentation (from trained model)
  2. Point-level motion (frame t → t+1)
  3. Ego motion compensation (GT)
  4. HDBSCAN clustering
  5. Box generation

Used for realistic Re-ID evaluation with inferred boxes (not GT boxes).

Pipeline:
  Segmentation Inference → Motion Calculation → Ego Compensation
  → HDBSCAN Clustering → Box Generation → Class Assignment (GT)

Author: Multimodal Tracking System
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

try:
    from hdbscan import HDBSCAN
    HDBSCAN_AVAILABLE = True
except ImportError:
    from sklearn.cluster import DBSCAN
    HDBSCAN_AVAILABLE = False
    logging.warning("HDBSCAN not available, using DBSCAN as fallback")

logger = logging.getLogger(__name__)


class TemporalMotionDetector(nn.Module):
    """
    Detects boxes using temporal motion and clustering.

    Full inference mode - no GT boxes used, only:
      - Predicted segmentation
      - Point motion between frames
      - Ego motion (GT)
      - HDBSCAN clustering

    Args:
        min_cluster_size: Minimum points per cluster (HDBSCAN)
        min_samples: Core points threshold (HDBSCAN)
        moving_threshold: Segmentation confidence threshold
        motion_threshold: Minimum motion magnitude (m/s)
        use_hdbscan: Use HDBSCAN if available, else DBSCAN
    """

    def __init__(
        self,
        min_cluster_size=5,
        min_samples=3,
        moving_threshold=0.3,
        motion_threshold=0.1,
        use_hdbscan=True
    ):
        super().__init__()
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.moving_threshold = moving_threshold
        self.motion_threshold = motion_threshold
        self.use_hdbscan = use_hdbscan and HDBSCAN_AVAILABLE

        logger.info(f"TemporalMotionDetector initialized:")
        logger.info(f"  → Clustering: {'HDBSCAN' if self.use_hdbscan else 'DBSCAN'}")
        logger.info(f"  → Min cluster size: {min_cluster_size}")
        logger.info(f"  → Motion threshold: {motion_threshold} m/s")
        logger.info(f"  → Segmentation threshold: {moving_threshold}")

    def forward(
        self,
        points_t: torch.Tensor,
        points_t1: torch.Tensor,
        seg_pred_t: torch.Tensor,
        seg_pred_t1: torch.Tensor,
        ego_motion: torch.Tensor,
        gt_boxes_t: Optional[List] = None,
        gt_boxes_t1: Optional[List] = None,
        gt_classes_t: Optional[List] = None,
        gt_track_ids_t: Optional[List] = None,
        time_delta: float = 0.1
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict]]:
        """
        Generate box detections using temporal motion.

        Args:
            points_t: [B, N, 3] points at frame t
            points_t1: [B, N, 3] points at frame t+1
            seg_pred_t: [B, N] segmentation at frame t
            seg_pred_t1: [B, N] segmentation at frame t+1
            ego_motion: [B, 4, 4] ego motion transformation (GT)
            gt_boxes_t: GT boxes (only for class assignment)
            gt_classes_t: GT classes (for evaluation)
            gt_track_ids_t: GT track IDs (for evaluation)
            time_delta: Time between frames (seconds)

        Returns:
            boxes_list: List[B] of detected boxes [M, 7]
            track_ids_list: List[B] of placeholder IDs [M] (Re-ID will assign)
            boxes_info_list: List[B] of box metadata
        """
        batch_size = points_t.shape[0]

        # Ensure correct format
        if points_t.shape[1] == 3 and points_t.shape[2] != 3:
            points_t = points_t.permute(0, 2, 1)
        if points_t1.shape[1] == 3 and points_t1.shape[2] != 3:
            points_t1 = points_t1.permute(0, 2, 1)

        if seg_pred_t.dim() == 3:
            seg_pred_t = seg_pred_t.squeeze(1)
        if seg_pred_t1.dim() == 3:
            seg_pred_t1 = seg_pred_t1.squeeze(1)

        boxes_list = []
        track_ids_list = []
        boxes_info_list = []

        total_clusters = 0

        for b in range(batch_size):
            pts_t = points_t[b]      # [N, 3]
            pts_t1 = points_t1[b]    # [N, 3]
            seg_t = seg_pred_t[b]    # [N]
            seg_t1 = seg_pred_t1[b]  # [N]
            ego = ego_motion[b] if ego_motion is not None else None  # [4, 4]

            # Step 1: Get segmented points (moving objects)
            seg_mask_t = (seg_t > self.moving_threshold)
            seg_mask_t1 = (seg_t1 > self.moving_threshold)

            # Step 2: Calculate point motion
            motion_vectors = self._calculate_point_motion(
                pts_t, pts_t1, seg_mask_t, seg_mask_t1, ego, time_delta
            )

            # Step 3: Filter points by motion threshold
            moving_mask = self._filter_by_motion(
                motion_vectors, seg_mask_t, self.motion_threshold
            )

            if moving_mask.sum() < self.min_cluster_size:
                # Not enough moving points
                boxes_list.append(torch.empty(0, 7, device=points_t.device))
                track_ids_list.append(torch.empty(0, dtype=torch.long, device=points_t.device))
                boxes_info_list.append([])
                continue

            # Step 4: Cluster moving points
            clusters, labels = self._cluster_points(
                pts_t[moving_mask].cpu().numpy()
            )

            if len(clusters) == 0:
                boxes_list.append(torch.empty(0, 7, device=points_t.device))
                track_ids_list.append(torch.empty(0, dtype=torch.long, device=points_t.device))
                boxes_info_list.append([])
                continue

            total_clusters += len(clusters)

            # Step 5: Generate boxes from clusters
            boxes, box_infos = self._generate_boxes_from_clusters(
                pts_t, moving_mask, labels, clusters, motion_vectors
            )

            # Step 6: Assign classes from GT (for evaluation)
            if gt_boxes_t is not None and b < len(gt_boxes_t):
                boxes, box_infos = self._assign_gt_classes(
                    boxes, box_infos,
                    gt_boxes_t[b],
                    gt_classes_t[b] if gt_classes_t and b < len(gt_classes_t) else None
                )

            # Step 7: Create placeholder track IDs (Re-ID will assign real IDs)
            track_ids = torch.arange(len(boxes), dtype=torch.long, device=points_t.device)

            boxes_list.append(boxes)
            track_ids_list.append(track_ids)
            boxes_info_list.append(box_infos)

        logger.debug(f"[INFERENCE] Generated {total_clusters} box detections from motion+clustering")

        return boxes_list, track_ids_list, boxes_info_list

    def _calculate_point_motion(
        self,
        pts_t: torch.Tensor,
        pts_t1: torch.Tensor,
        seg_mask_t: torch.Tensor,
        seg_mask_t1: torch.Tensor,
        ego_motion: Optional[torch.Tensor],
        time_delta: float
    ) -> torch.Tensor:
        """
        Calculate point-level motion between frames.

        Args:
            pts_t: [N, 3] points at frame t
            pts_t1: [N, 3] points at frame t+1
            seg_mask_t: [N] segmentation mask at t
            seg_mask_t1: [N] segmentation mask at t+1
            ego_motion: [4, 4] ego motion transformation
            time_delta: Time between frames

        Returns:
            motion_vectors: [N, 3] motion vector per point (m/s)
        """
        N = pts_t.shape[0]

        # Compensate ego motion in frame t+1
        if ego_motion is not None:
            # Apply inverse ego motion to pts_t1
            pts_t1_compensated = self._apply_ego_motion(pts_t1, ego_motion)
        else:
            pts_t1_compensated = pts_t1

        # Calculate displacement
        displacement = pts_t1_compensated - pts_t  # [N, 3]

        # Convert to velocity (m/s)
        velocity = displacement / time_delta  # [N, 3]

        # Only keep for segmented points
        motion_vectors = torch.zeros_like(velocity)
        motion_vectors[seg_mask_t & seg_mask_t1] = velocity[seg_mask_t & seg_mask_t1]

        return motion_vectors

    def _apply_ego_motion(
        self,
        points: torch.Tensor,
        ego_motion: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply inverse ego motion transformation to compensate vehicle movement.

        Args:
            points: [N, 3] points
            ego_motion: [4, 4] transformation matrix (frame t → t+1)

        Returns:
            compensated_points: [N, 3] points in frame t coordinates
        """
        # Inverse transformation (t+1 → t)
        ego_inv = torch.inverse(ego_motion)

        # Convert to homogeneous coordinates
        N = points.shape[0]
        ones = torch.ones((N, 1), device=points.device, dtype=points.dtype)
        points_hom = torch.cat([points, ones], dim=1)  # [N, 4]

        # Apply transformation
        points_transformed = (ego_inv @ points_hom.T).T  # [N, 4]

        # Convert back to 3D
        return points_transformed[:, :3]

    def _filter_by_motion(
        self,
        motion_vectors: torch.Tensor,
        seg_mask: torch.Tensor,
        threshold: float
    ) -> torch.Tensor:
        """
        Filter points by motion magnitude.

        Args:
            motion_vectors: [N, 3] velocity vectors
            seg_mask: [N] segmentation mask
            threshold: Minimum motion magnitude (m/s)

        Returns:
            moving_mask: [N] boolean mask (True = moving object)
        """
        # Calculate motion magnitude
        motion_magnitude = torch.norm(motion_vectors, dim=1)  # [N]

        # Filter by threshold
        moving_mask = seg_mask & (motion_magnitude > threshold)

        return moving_mask

    def _cluster_points(
        self,
        points: np.ndarray
    ) -> Tuple[List[int], np.ndarray]:
        """
        Cluster points using HDBSCAN or DBSCAN.

        Args:
            points: [K, 3] moving points

        Returns:
            clusters: List of cluster IDs
            labels: [K] cluster label per point (-1 = noise)
        """
        if len(points) < self.min_cluster_size:
            return [], np.full(len(points), -1)

        if self.use_hdbscan:
            # HDBSCAN - adaptive clustering
            clusterer = HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                cluster_selection_epsilon=0.5
            )
        else:
            # DBSCAN fallback
            clusterer = DBSCAN(
                eps=2.0,
                min_samples=self.min_samples
            )

        # Cluster in XY plane (BEV)
        labels = clusterer.fit_predict(points[:, :2])

        # Get unique cluster IDs (exclude noise = -1)
        unique_clusters = np.unique(labels)
        unique_clusters = unique_clusters[unique_clusters >= 0]

        return list(unique_clusters), labels

    def _generate_boxes_from_clusters(
        self,
        points: torch.Tensor,
        moving_mask: torch.Tensor,
        labels: np.ndarray,
        clusters: List[int],
        motion_vectors: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Generate oriented bounding boxes from clusters.

        Args:
            points: [N, 3] all points
            moving_mask: [N] boolean mask of moving points
            labels: [K] cluster labels for moving points
            clusters: List of cluster IDs
            motion_vectors: [N, 3] motion vectors

        Returns:
            boxes: [M, 7] boxes (x, y, z, l, w, h, yaw)
            box_infos: List[M] of box metadata
        """
        boxes = []
        box_infos = []

        moving_indices = torch.where(moving_mask)[0]
        moving_points = points[moving_mask].cpu().numpy()
        moving_motion = motion_vectors[moving_mask].cpu().numpy()

        for cluster_id in clusters:
            cluster_mask = (labels == cluster_id)
            cluster_points = moving_points[cluster_mask]
            cluster_motion = moving_motion[cluster_mask]

            if len(cluster_points) < 3:
                continue

            # Fit oriented box
            box = self._fit_oriented_box(cluster_points)

            # Calculate average motion
            avg_motion = cluster_motion.mean(axis=0)
            motion_magnitude = np.linalg.norm(avg_motion)

            # Create box info
            points_mask = np.zeros(len(points), dtype=bool)
            points_mask[moving_indices[cluster_mask].cpu().numpy()] = True

            box_info = {
                'center': box[:3],
                'size': box[3:6],
                'yaw': box[6],
                'points_mask': points_mask,
                'num_points': int(cluster_mask.sum()),
                'motion_vector': avg_motion,
                'motion_magnitude': motion_magnitude,
                'confidence': 1.0,
                'class': -1  # Will be assigned from GT
            }

            boxes.append(box)
            box_infos.append(box_info)

        if len(boxes) == 0:
            return torch.empty(0, 7, device=points.device), []

        boxes_tensor = torch.tensor(
            np.array(boxes),
            dtype=torch.float32,
            device=points.device
        )

        return boxes_tensor, box_infos

    def _fit_oriented_box(self, points: np.ndarray) -> np.ndarray:
        """
        Fit oriented bounding box using PCA.

        Args:
            points: [P, 3] cluster points

        Returns:
            box: [7] (x, y, z, l, w, h, yaw)
        """
        # Center
        center = points.mean(axis=0)

        if len(points) == 1:
            return np.array([
                center[0], center[1], center[2],
                0.5, 0.5, 0.3, 0.0
            ])

        # PCA for orientation
        points_xy = points[:, :2] - center[:2]

        if len(points) == 2 or np.allclose(points_xy, 0):
            yaw = 0.0
            points_rotated = points_xy
        else:
            cov = np.cov(points_xy.T)

            if not np.all(np.isfinite(cov)):
                yaw = 0.0
                points_rotated = points_xy
            else:
                try:
                    eigenvalues, eigenvectors = np.linalg.eig(cov)
                    main_axis = eigenvectors[:, np.argmax(eigenvalues)]
                    yaw = np.arctan2(main_axis[1], main_axis[0])

                    # Rotate to box frame
                    cos_yaw = np.cos(-yaw)
                    sin_yaw = np.sin(-yaw)
                    rotation_matrix = np.array([
                        [cos_yaw, -sin_yaw],
                        [sin_yaw, cos_yaw]
                    ])
                    points_rotated = points_xy @ rotation_matrix.T
                except:
                    yaw = 0.0
                    points_rotated = points_xy

        # Size
        min_xy = points_rotated.min(axis=0)
        max_xy = points_rotated.max(axis=0)
        size_xy = np.maximum(max_xy - min_xy, 0.3)

        # Height
        min_z = points[:, 2].min()
        max_z = points[:, 2].max()
        height = max(max_z - min_z, 0.3)

        return np.array([
            center[0], center[1], center[2],
            size_xy[0], size_xy[1], height, yaw
        ])

    def _assign_gt_classes(
        self,
        boxes: torch.Tensor,
        box_infos: List[Dict],
        gt_boxes: any,
        gt_classes: any
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Assign GT classes to detected boxes (for evaluation).

        Args:
            boxes: [M, 7] detected boxes
            box_infos: List[M] of box metadata
            gt_boxes: GT boxes (various formats)
            gt_classes: GT classes

        Returns:
            boxes: [M, 7] same boxes
            box_infos: List[M] with classes assigned
        """
        if gt_boxes is None or len(boxes) == 0:
            return boxes, box_infos

        # Extract GT boxes
        gt_boxes_array = self._extract_gt_boxes(gt_boxes, gt_classes)

        if len(gt_boxes_array) == 0:
            return boxes, box_infos

        # Match detected boxes with GT using IoU
        boxes_np = boxes.cpu().numpy()

        for i, box_info in enumerate(box_infos):
            best_iou = 0.0
            best_class = -1

            for gt_box, gt_class in gt_boxes_array:
                iou = self._compute_iou_2d(boxes_np[i], gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_class = gt_class

            if best_iou > 0.1:  # Minimum IoU threshold
                box_info['class'] = best_class
                box_info['gt_iou'] = best_iou

        return boxes, box_infos

    def _extract_gt_boxes(self, gt_boxes: any, gt_classes: any) -> List[Tuple[np.ndarray, int]]:
        """Extract GT boxes and classes."""
        boxes_with_classes = []

        if isinstance(gt_boxes, dict):
            for obj_id, box in gt_boxes.items():
                center = box.get_center()
                extent = box.extent
                R = np.array(box.R)
                yaw = np.arctan2(R[1, 0], R[0, 0])

                box_array = np.array([
                    center[0], center[1], center[2],
                    extent[0], extent[1], extent[2], yaw
                ])

                cls = gt_classes.get(obj_id, 0) if isinstance(gt_classes, dict) else 0
                boxes_with_classes.append((box_array, cls))

        elif isinstance(gt_boxes, list):
            for i, box in enumerate(gt_boxes):
                if isinstance(box, (np.ndarray, torch.Tensor)):
                    box_array = box.cpu().numpy() if isinstance(box, torch.Tensor) else box
                    cls = gt_classes[i] if isinstance(gt_classes, list) and i < len(gt_classes) else 0
                    boxes_with_classes.append((box_array, cls))

        return boxes_with_classes

    def _compute_iou_2d(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute 2D IoU (simplified, axis-aligned)."""
        x1, y1, l1, w1 = box1[0], box1[1], box1[3], box1[4]
        x2, y2, l2, w2 = box2[0], box2[1], box2[3], box2[4]

        min_x1, max_x1 = x1 - l1/2, x1 + l1/2
        min_y1, max_y1 = y1 - w1/2, y1 + w1/2

        min_x2, max_x2 = x2 - l2/2, x2 + l2/2
        min_y2, max_y2 = y2 - w2/2, y2 + w2/2

        inter_min_x = max(min_x1, min_x2)
        inter_max_x = min(max_x1, max_x2)
        inter_min_y = max(min_y1, min_y2)
        inter_max_y = min(max_y1, max_y2)

        inter_w = max(0, inter_max_x - inter_min_x)
        inter_h = max(0, inter_max_y - inter_min_y)
        inter_area = inter_w * inter_h

        area1 = l1 * w1
        area2 = l2 * w2
        union_area = area1 + area2 - inter_area

        return inter_area / (union_area + 1e-6)
