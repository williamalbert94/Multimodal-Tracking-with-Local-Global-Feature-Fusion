"""
Box Proposal Network
Generates bounding boxes from segmented point cloud.
Supports two methods:
  1. DBSCAN clustering (simple, fast)
  2. Learned clustering (more advanced)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import DBSCAN


class BoxProposalNetwork(nn.Module):
    """
    Propone bounding boxes a partir de puntos segmentados (moving only).

    Método DBSCAN (simple):
      - Clustering espacial de puntos
      - Caja orientada por PCA
      - Rápido, sin aprendizaje

    Método Learned (avanzado):
      - PointNet para features
      - Clustering aprendido
      - Regresión de parámetros
    """

    def __init__(self, method='dbscan', eps=2.0, min_samples=5, embedding_dim=128):
        super().__init__()
        self.method = method
        self.eps = eps
        self.min_samples = min_samples

        if method == 'learned':
            # PointNet encoder (para método aprendido)
            self.point_encoder = nn.Sequential(
                nn.Conv1d(3, 64, 1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 256, 1),
            )

            # Instance embedding (para clustering)
            self.instance_mlp = nn.Sequential(
                nn.Linear(256, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim)
            )

            # Box regression head
            self.box_regressor = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 7)  # x, y, z, l, w, h, yaw
            )

    def forward(self, points, seg_mask):
        """
        Args:
            points: [B, N, 3] point cloud
            seg_mask: [B, N] binary mask (1 = moving, 0 = static)

        Returns:
            boxes: List[Tensor] - boxes per batch [M_i, 7] (x,y,z,l,w,h,yaw)
            cluster_ids: List[Tensor] - cluster ID per point [N_i]
        """
        batch_size = points.shape[0]
        boxes_batch = []
        cluster_ids_batch = []

        for b in range(batch_size):
            pts = points[b]  # [N, 3]
            mask = seg_mask[b]  # [N]

            # Get moving points only
            moving_pts = pts[mask.bool()]  # [M, 3]

            if len(moving_pts) < self.min_samples:
                # No hay suficientes puntos en movimiento
                boxes_batch.append(torch.zeros(0, 7).to(points.device))
                cluster_ids_batch.append(torch.zeros(len(pts)).long().to(points.device) - 1)
                continue

            if self.method == 'dbscan':
                boxes, cluster_ids = self._dbscan_boxes(pts, mask)
            else:
                boxes, cluster_ids = self._learned_boxes(pts, mask)

            boxes_batch.append(boxes)
            cluster_ids_batch.append(cluster_ids)

        return boxes_batch, cluster_ids_batch

    def _dbscan_boxes(self, points, mask):
        """
        DBSCAN clustering + oriented bounding box fitting.

        Args:
            points: [N, 3] all points
            mask: [N] moving mask

        Returns:
            boxes: [M, 7] boxes (x, y, z, l, w, h, yaw)
            cluster_ids: [N] cluster ID per point (-1 = noise)
        """
        # Get moving points
        moving_pts = points[mask.bool()].cpu().numpy()  # [K, 3]

        if len(moving_pts) < self.min_samples:
            return (torch.zeros(0, 7).to(points.device),
                   torch.zeros(len(points)).long().to(points.device) - 1)

        # DBSCAN clustering (solo en XY para radar)
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = clustering.fit_predict(moving_pts[:, :2])  # Cluster en XY

        # Convert labels to full point cloud
        cluster_ids = torch.zeros(len(points)).long().to(points.device) - 1
        moving_indices = torch.where(mask.bool())[0]
        cluster_ids[moving_indices] = torch.from_numpy(labels).long().to(points.device)

        # Get unique clusters (exclude noise = -1)
        unique_clusters = np.unique(labels)
        unique_clusters = unique_clusters[unique_clusters >= 0]

        if len(unique_clusters) == 0:
            return torch.zeros(0, 7).to(points.device), cluster_ids

        # Fit box per cluster
        boxes = []
        for cluster_id in unique_clusters:
            cluster_mask = (labels == cluster_id)
            cluster_pts = moving_pts[cluster_mask]  # [P, 3]

            # Fit oriented bounding box
            box = self._fit_oriented_box(cluster_pts)
            boxes.append(box)

        boxes = torch.stack(boxes).to(points.device)  # [M, 7]
        return boxes, cluster_ids

    def _fit_oriented_box(self, points):
        """
        Fit oriented bounding box using PCA.

        Args:
            points: [P, 3] numpy array

        Returns:
            box: [7] tensor (x, y, z, l, w, h, yaw)
        """
        # Center
        center = points.mean(axis=0)  # [3]

        # Handle edge cases for small clusters
        if len(points) == 1:
            # Single point cluster - create small default box
            box = torch.tensor([
                center[0],      # x
                center[1],      # y
                center[2],      # z
                0.5,            # length (default small size)
                0.5,            # width
                0.3,            # height
                0.0             # yaw (default orientation)
            ], dtype=torch.float32)
            return box

        # PCA para orientación (solo en XY)
        points_xy = points[:, :2] - center[:2]

        # Check for collinear points or zero variance
        if len(points) == 2 or np.allclose(points_xy, 0):
            # Two points or collinear - use simple axis-aligned box
            yaw = 0.0
            points_rotated = points_xy
        else:
            # Standard PCA for 3+ non-collinear points
            cov = np.cov(points_xy.T)

            # Check for NaN/Inf in covariance matrix
            if not np.all(np.isfinite(cov)):
                yaw = 0.0
                points_rotated = points_xy
            else:
                try:
                    eigenvalues, eigenvectors = np.linalg.eig(cov)

                    # Eje principal (mayor eigenvalue)
                    main_axis_idx = np.argmax(eigenvalues)
                    main_axis = eigenvectors[:, main_axis_idx]

                    # Yaw (ángulo del eje principal)
                    yaw = np.arctan2(main_axis[1], main_axis[0])

                    # Rotar puntos al frame del objeto
                    cos_yaw = np.cos(-yaw)
                    sin_yaw = np.sin(-yaw)
                    rotation_matrix = np.array([
                        [cos_yaw, -sin_yaw],
                        [sin_yaw, cos_yaw]
                    ])

                    points_rotated = points_xy @ rotation_matrix.T
                except np.linalg.LinAlgError:
                    # Fallback if eigenvalue decomposition fails
                    yaw = 0.0
                    points_rotated = points_xy

        # Tamaño (min/max en frame rotado)
        min_xy = points_rotated.min(axis=0)
        max_xy = points_rotated.max(axis=0)
        size_xy = max_xy - min_xy

        # Add small epsilon to avoid zero-size boxes
        size_xy = np.maximum(size_xy, 0.3)  # Min 30cm in each dimension

        # Altura
        min_z = points[:, 2].min()
        max_z = points[:, 2].max()
        height = max(max_z - min_z, 0.3)  # Min 30cm height

        # Box: [x, y, z, length, width, height, yaw]
        box = torch.tensor([
            center[0],      # x
            center[1],      # y
            center[2],      # z (mean)
            size_xy[0],     # length
            size_xy[1],     # width
            height,         # height
            yaw             # yaw
        ], dtype=torch.float32)

        return box

    def _learned_boxes(self, points, mask):
        """
        Learned clustering and box regression.
        (Placeholder - to be implemented)
        """
        raise NotImplementedError("Learned clustering not implemented yet")


class BoxRegressionLoss(nn.Module):
    """
    Combined box regression loss.
    """

    def __init__(self, weight_center=1.0, weight_size=0.5, weight_iou=2.0):
        super().__init__()
        self.weight_center = weight_center
        self.weight_size = weight_size
        self.weight_iou = weight_iou

    def forward(self, pred_boxes, gt_boxes):
        """
        Args:
            pred_boxes: [M, 7] predicted boxes (tensor, list, or numpy array)
            gt_boxes: [M, 7] ground truth boxes (tensor, list, or numpy array)

        Returns:
            loss: scalar
        """
        # Ensure inputs are tensors
        if not isinstance(pred_boxes, torch.Tensor):
            pred_boxes = torch.tensor(pred_boxes, dtype=torch.float32)
        if not isinstance(gt_boxes, torch.Tensor):
            gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)

        # Ensure same device
        device = pred_boxes.device if len(pred_boxes) > 0 else (gt_boxes.device if len(gt_boxes) > 0 else torch.device('cpu'))
        pred_boxes = pred_boxes.to(device)
        gt_boxes = gt_boxes.to(device)

        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            return torch.tensor(0.0).to(device)

        # Match predicted boxes to GT (simple nearest neighbor for now)
        # TODO: Use Hungarian algorithm for optimal matching
        matched_pred, matched_gt = self._match_boxes(pred_boxes, gt_boxes)

        if len(matched_pred) == 0:
            return torch.tensor(0.0).to(pred_boxes.device)

        # Center loss (x, y, z)
        loss_center = F.smooth_l1_loss(matched_pred[:, :3], matched_gt[:, :3])

        # Size loss (l, w, h)
        loss_size = F.smooth_l1_loss(matched_pred[:, 3:6], matched_gt[:, 3:6])

        # Orientation loss (yaw)
        loss_orient = self._orientation_loss(matched_pred[:, 6], matched_gt[:, 6])

        # IoU loss (optional, more expensive)
        if self.weight_iou > 0:
            loss_iou = self._iou_loss(matched_pred, matched_gt)
        else:
            loss_iou = torch.tensor(0.0).to(pred_boxes.device)

        # Combined loss
        loss = (self.weight_center * loss_center +
                self.weight_size * (loss_size + loss_orient) +
                self.weight_iou * loss_iou)

        return loss

    def _match_boxes(self, pred_boxes, gt_boxes):
        """
        Match predicted boxes to GT using nearest center distance.
        (Simple version - can be improved with Hungarian algorithm)
        """
        # Ensure inputs are tensors
        if not isinstance(pred_boxes, torch.Tensor):
            pred_boxes = torch.tensor(pred_boxes, dtype=torch.float32)
        if not isinstance(gt_boxes, torch.Tensor):
            gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)

        # Ensure correct device
        if pred_boxes.device != gt_boxes.device:
            gt_boxes = gt_boxes.to(pred_boxes.device)

        # Compute pairwise distances (center only)
        pred_centers = pred_boxes[:, :3]  # [M, 3]
        gt_centers = gt_boxes[:, :3]      # [N, 3]

        # Distance matrix [M, N]
        dist_matrix = torch.cdist(pred_centers, gt_centers)

        # Greedy matching (nearest neighbor)
        matched_pred = []
        matched_gt = []

        for i in range(len(pred_boxes)):
            if len(gt_boxes) == 0:
                break

            # Find nearest GT
            distances = dist_matrix[i]
            min_idx = distances.argmin()

            if distances[min_idx] < 5.0:  # Threshold: 5 meters
                matched_pred.append(pred_boxes[i])
                matched_gt.append(gt_boxes[min_idx])

                # Remove matched GT
                dist_matrix = torch.cat([dist_matrix[:, :min_idx],
                                        dist_matrix[:, min_idx+1:]], dim=1)
                gt_boxes = torch.cat([gt_boxes[:min_idx],
                                     gt_boxes[min_idx+1:]], dim=0)

        if len(matched_pred) == 0:
            return torch.zeros(0, 7).to(pred_boxes.device), torch.zeros(0, 7).to(pred_boxes.device)

        return torch.stack(matched_pred), torch.stack(matched_gt)

    def _orientation_loss(self, pred_yaw, gt_yaw):
        """
        Orientation loss (handles angle wrapping).
        """
        # Normalize angles to [-pi, pi]
        diff = pred_yaw - gt_yaw
        diff = torch.atan2(torch.sin(diff), torch.cos(diff))
        return F.smooth_l1_loss(diff, torch.zeros_like(diff))

    def _iou_loss(self, pred_boxes, gt_boxes):
        """
        IoU loss (placeholder - 3D IoU is expensive).
        Using simplified 2D IoU on BEV.
        """
        # Simplified: 1 - IoU
        iou = self._compute_iou_2d(pred_boxes, gt_boxes)
        return (1 - iou).mean()

    def _compute_iou_2d(self, boxes1, boxes2):
        """
        Compute 2D IoU on Bird's Eye View.
        Simplified version (axis-aligned for speed).
        """
        # Extract x, y, l, w
        x1, y1, l1, w1 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 3], boxes1[:, 4]
        x2, y2, l2, w2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 3], boxes2[:, 4]

        # Axis-aligned boxes (ignore rotation for speed)
        min_x1, max_x1 = x1 - l1/2, x1 + l1/2
        min_y1, max_y1 = y1 - w1/2, y1 + w1/2

        min_x2, max_x2 = x2 - l2/2, x2 + l2/2
        min_y2, max_y2 = y2 - w2/2, y2 + w2/2

        # Intersection
        inter_min_x = torch.max(min_x1, min_x2)
        inter_max_x = torch.min(max_x1, max_x2)
        inter_min_y = torch.max(min_y1, min_y2)
        inter_max_y = torch.min(max_y1, max_y2)

        inter_w = torch.clamp(inter_max_x - inter_min_x, min=0)
        inter_h = torch.clamp(inter_max_y - inter_min_y, min=0)
        inter_area = inter_w * inter_h

        # Union
        area1 = l1 * w1
        area2 = l2 * w2
        union_area = area1 + area2 - inter_area

        # IoU
        iou = inter_area / (union_area + 1e-6)
        return iou
