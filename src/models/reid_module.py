"""
Re-ID Tracking Module (Integrated)
Módulo completo de Re-ID que integra:
  1. Box Proposal
  2. Feature Extraction
  3. Track Association

Puede funcionar en modo aislado (GT inputs) o integrado (predicted inputs).
"""

import torch
import torch.nn as nn
import numpy as np
from .box_proposal import BoxProposalNetwork, BoxRegressionLoss
from .reid_features import ReIDFeatureExtractor, TripletLoss, ClassificationLoss


def convert_o3d_boxes_to_tensor(boxes_dict):
    """
    Convert Open3D OrientedBoundingBox dictionary to tensor format.

    Args:
        boxes_dict: dict {object_id: OrientedBoundingBox} OR tensor [M, 7] (already converted)

    Returns:
        boxes_tensor: [M, 7] tensor (x, y, z, l, w, h, yaw)
        track_ids: [M] tensor of object IDs (or None if already tensor)
    """
    if boxes_dict is None or len(boxes_dict) == 0:
        return torch.zeros(0, 7), torch.zeros(0).long()

    # Check if already a tensor (from ImprovedBoxProposal)
    if isinstance(boxes_dict, torch.Tensor):
        # Already converted, return as-is
        # Note: track_ids should be passed separately in this case
        return boxes_dict, None

    # Check if it's a list/array (also pre-converted)
    if isinstance(boxes_dict, (list, np.ndarray)):
        if isinstance(boxes_dict, list) and len(boxes_dict) > 0:
            if isinstance(boxes_dict[0], torch.Tensor):
                # List of tensors, stack them
                return torch.stack(boxes_dict), None
            elif isinstance(boxes_dict[0], np.ndarray):
                # List of numpy arrays, convert to tensor
                return torch.from_numpy(np.stack(boxes_dict)).float(), None
        elif isinstance(boxes_dict, np.ndarray):
            # Numpy array, convert to tensor
            return torch.from_numpy(boxes_dict).float(), None

    # Original behavior: dict of Open3D boxes
    boxes_list = []
    ids_list = []

    for obj_id, obb in boxes_dict.items():
        # Get center from Open3D box
        center = obb.center  # [3]

        # Get extent (size) from Open3D box
        extent = obb.extent  # [3] - [length, height, width] in Open3D

        # Get rotation matrix and extract yaw
        R = obb.R  # [3, 3] rotation matrix
        # Extract yaw from rotation matrix (assuming rotation around Y-axis)
        # yaw = atan2(R[2,0], R[0,0]) for Y-up coordinate system
        yaw = np.arctan2(R[2, 0], R[0, 0])

        # Create box in format [x, y, z, l, w, h, yaw]
        box = np.array([
            center[0],   # x
            center[1],   # y
            center[2],   # z
            extent[0],   # length
            extent[2],   # width
            extent[1],   # height
            yaw          # yaw
        ])

        boxes_list.append(box)
        ids_list.append(obj_id)

    if len(boxes_list) == 0:
        return torch.zeros(0, 7), torch.zeros(0).long()

    boxes_tensor = torch.from_numpy(np.stack(boxes_list)).float()
    ids_tensor = torch.tensor(ids_list).long()

    return boxes_tensor, ids_tensor


class ReIDTrackingModule(nn.Module):
    """
    Módulo completo de Re-ID y Tracking.

    Modos de operación:
      1. reid_only: Usa GT segmentation/ego_motion, aprende box + Re-ID
      2. full: Usa predicciones del modelo principal

    Architecture:
      Input → Box Proposal → Re-ID Features → Track Association
    """

    def __init__(self, args):
        super().__init__()

        # Configuration
        self.train_mode = getattr(args, 'train_mode', 'segmentation_only')
        self.use_gt_segmentation = getattr(args, 'use_gt_segmentation', False)
        self.use_gt_ego_motion = getattr(args, 'use_gt_ego_motion', False)
        self.use_gt_boxes = getattr(args, 'use_gt_boxes', False)

        # Re-ID config
        reid_config = getattr(args, 'reid', None)
        self.reid_config = reid_config  # Store for later use in compute_losses
        if reid_config is None:
            # Default config
            self.reid_enabled = False
            self.embedding_dim = 256
            self.box_method = 'dbscan'
            self.dbscan_eps = 2.0
            self.dbscan_min_samples = 5
        else:
            self.reid_enabled = reid_config.get('enabled', True)
            self.embedding_dim = reid_config.get('embedding_dim', 256)

            box_config = reid_config.get('box_proposal', {})
            self.box_method = box_config.get('method', 'dbscan')
            self.dbscan_eps = box_config.get('eps', 2.0)
            self.dbscan_min_samples = box_config.get('min_samples', 5)

        # Modules
        if self.reid_enabled:
            self.box_proposal = BoxProposalNetwork(
                method=self.box_method,
                eps=self.dbscan_eps,
                min_samples=self.dbscan_min_samples,
                embedding_dim=self.embedding_dim
            )

            # Get sample_point_num from config (usado por el backbone PointNet++)
            sample_point_num = getattr(args, 'num_points', 512)
            use_pretrained = reid_config.get('use_pretrained_backbone', False) if reid_config else False
            checkpoint_path = reid_config.get('pretrained_checkpoint', None) if reid_config else None
            backbone_name = reid_config.get('backbone', 'LocalGlobalFusionSimple') if reid_config else 'LocalGlobalFusionSimple'

            self.reid_extractor = ReIDFeatureExtractor(
                embedding_dim=self.embedding_dim,
                sample_point_num=sample_point_num,
                use_pretrained_backbone=use_pretrained,
                checkpoint_path=checkpoint_path,
                backbone_name=backbone_name
            )

        # Losses
        loss_weights = getattr(args, 'loss_weights', {})
        self.box_loss_fn = BoxRegressionLoss(
            weight_center=loss_weights.get('box_center', 1.0),
            weight_size=loss_weights.get('box_size', 0.5),
            weight_iou=loss_weights.get('box_iou', 2.0)
        )

        triplet_config = getattr(args, 'reid', {}).get('triplet_loss', {}) if hasattr(args, 'reid') else {}
        self.triplet_loss_fn = TripletLoss(
            margin=triplet_config.get('margin', 0.3),
            hard_mining=triplet_config.get('hard_mining', True)
        )

        # Classification loss for class-aware embeddings
        self.classification_loss_fn = ClassificationLoss()

    def forward(self, batch, seg_pred=None, ego_motion_pred=None, epoch=None):
        """
        Forward pass del Re-ID module.

        Args:
            batch: Dict con datos del batch (pc1, [pc2], seg_gt, [seg_gt2], boxes_gt, track_ids_gt, etc.)
            seg_pred: [B, N] segmentation prediction (si train_mode='full')
            ego_motion_pred: [B, 6] ego motion prediction (si train_mode='full')
            epoch: Current epoch number (for progressive training strategy)

        Returns:
            outputs: Dict con boxes, embeddings, losses
        """
        if not self.reid_enabled:
            return {}

        # Determine input source based on mode (frame t)
        if self.train_mode == 'reid_only' or self.use_gt_segmentation:
            seg_mask = batch['seg_gt']  # Use GT
        else:
            seg_mask = seg_pred  # Use prediction

        # Extract point clouds (frame t)
        pc1 = batch['pc1']  # [B, 3, N] o [B, N, 3]

        # Ensure correct shape [B, N, 3]
        if pc1.shape[1] == 3:  # [B, 3, N]
            pc1 = pc1.permute(0, 2, 1)  # → [B, N, 3]

        # Box Proposal (frame t)
        if self.use_gt_boxes and 'boxes_gt' in batch:
            # Use GT boxes directly (perfect detection for Re-ID training)
            pred_boxes = []
            cluster_ids = []
            for b in range(pc1.shape[0]):
                boxes_b, ids_b = convert_o3d_boxes_to_tensor(batch['boxes_gt'][b])
                pred_boxes.append(boxes_b.cuda() if boxes_b.device.type != 'cuda' else boxes_b)
                # For GT boxes, we need cluster_ids to extract features
                # Assign points to GT boxes based on distance
                cluster_ids_b = self._assign_points_to_gt_boxes(pc1[b], boxes_b.cuda() if boxes_b.device.type != 'cuda' else boxes_b)
                cluster_ids.append(cluster_ids_b)
        else:
            # Use DBSCAN box proposal (learning mode)
            pred_boxes, cluster_ids = self.box_proposal(pc1, seg_mask)

        # Re-ID Feature Extraction (frame t)
        embeddings, class_logits = self.reid_extractor(pc1, pred_boxes, cluster_ids)

        # Process frame t+1 if available (for triplet loss)
        pred_boxes2 = None
        embeddings2 = None
        class_logits2 = None
        if 'pc2' in batch and 'seg_gt2' in batch:
            # Frame t+1 segmentation
            if self.train_mode == 'reid_only' or self.use_gt_segmentation:
                seg_mask2 = batch['seg_gt2']
            else:
                seg_mask2 = seg_pred  # Would need seg_pred2 in full mode

            pc2 = batch['pc2']
            if pc2.shape[1] == 3:  # [B, 3, N]
                pc2 = pc2.permute(0, 2, 1)  # → [B, N, 3]

            # Box Proposal (frame t+1)
            if self.use_gt_boxes and 'boxes_gt2' in batch:
                # Use GT boxes directly for frame t+1
                pred_boxes2 = []
                cluster_ids2 = []
                for b in range(pc2.shape[0]):
                    boxes_b, ids_b = convert_o3d_boxes_to_tensor(batch['boxes_gt2'][b])
                    pred_boxes2.append(boxes_b.cuda() if boxes_b.device.type != 'cuda' else boxes_b)
                    # Assign points to GT boxes
                    cluster_ids_b = self._assign_points_to_gt_boxes(pc2[b], boxes_b.cuda() if boxes_b.device.type != 'cuda' else boxes_b)
                    cluster_ids2.append(cluster_ids_b)
            else:
                # Use DBSCAN box proposal
                pred_boxes2, cluster_ids2 = self.box_proposal(pc2, seg_mask2)

            # Re-ID Feature Extraction (frame t+1)
            embeddings2, class_logits2 = self.reid_extractor(pc2, pred_boxes2, cluster_ids2)

        # Prepare outputs
        outputs = {
            'boxes': pred_boxes,
            'cluster_ids': cluster_ids,
            'embeddings': embeddings,
            'class_logits': class_logits,  # NEW: class predictions
        }

        # Compute losses (if GT available)
        if 'boxes_gt' in batch and 'track_ids_gt' in batch:
            # Pass both frames if available
            boxes_gt2 = batch.get('boxes_gt2', None)
            track_ids_gt2 = batch.get('track_ids_gt2', None)
            class_labels_gt = batch.get('class_labels_gt', None)
            class_labels_gt2 = batch.get('class_labels_gt2', None)

            losses = self.compute_losses(
                pred_boxes=pred_boxes,
                embeddings=embeddings,
                boxes_gt=batch['boxes_gt'],
                track_ids_gt=batch['track_ids_gt'],
                # Frame t+1 data for triplet loss
                pred_boxes2=pred_boxes2,
                embeddings2=embeddings2,
                boxes_gt2=boxes_gt2,
                track_ids_gt2=track_ids_gt2,
                # Classification data
                class_logits=class_logits,
                class_logits2=class_logits2,
                class_labels_gt=class_labels_gt,
                class_labels_gt2=class_labels_gt2,
                # Epoch for progressive training
                epoch=epoch
            )
            outputs['losses'] = losses

        return outputs

    def compute_losses(self, pred_boxes, embeddings, boxes_gt, track_ids_gt,
                      pred_boxes2=None, embeddings2=None, boxes_gt2=None, track_ids_gt2=None,
                      class_logits=None, class_logits2=None, class_labels_gt=None, class_labels_gt2=None,
                      epoch=None):
        """
        Compute Re-ID losses.

        Args:
            pred_boxes: List[Tensor] predicted boxes per frame at time t
            embeddings: List[Tensor] Re-ID embeddings per frame at time t
            boxes_gt: List[dict] GT boxes per frame at time t (Open3D OrientedBoundingBox dicts)
            track_ids_gt: List[Tensor] GT track IDs per frame at time t
            pred_boxes2: List[Tensor] predicted boxes at time t+1 (optional, for triplet loss)
            embeddings2: List[Tensor] embeddings at time t+1 (optional, for triplet loss)
            boxes_gt2: List[dict] GT boxes at time t+1 (optional, for triplet loss)
            track_ids_gt2: List[Tensor] GT track IDs at time t+1 (optional, for triplet loss)
            class_logits: List[Tensor] class predictions at time t [M_i, NUM_CLASSES]
            class_logits2: List[Tensor] class predictions at time t+1 (optional)
            class_labels_gt: List[dict] GT class labels at time t {obj_id: 'car'}
            class_labels_gt2: List[dict] GT class labels at time t+1 (optional)
            epoch: Current epoch number (for progressive training strategy)

        Returns:
            losses: Dict with individual losses (tensors with gradients)
        """
        batch_size = len(pred_boxes)
        device = pred_boxes[0].device if len(pred_boxes) > 0 and len(pred_boxes[0]) > 0 else torch.device('cuda')

        # Convert GT boxes from Open3D dict format to tensor
        boxes_gt_tensors = []
        track_ids_from_boxes = []
        for b in range(batch_size):
            boxes_tensor, ids_tensor = convert_o3d_boxes_to_tensor(boxes_gt[b])
            boxes_gt_tensors.append(boxes_tensor)

            # Use track_ids_gt if available (from ImprovedBoxProposal), otherwise use ids from dict
            if track_ids_gt is not None and b < len(track_ids_gt) and track_ids_gt[b] is not None:
                track_ids_from_boxes.append(track_ids_gt[b])
            else:
                track_ids_from_boxes.append(ids_tensor)

        # Box regression loss
        # NOTE: When use_gt_boxes=True, pred_boxes ARE GT boxes, so loss should be 0
        #       When use_gt_boxes=False, pred_boxes come from DBSCAN (not differentiable)
        #       In both cases, box_loss doesn't contribute gradients (GT = perfect, DBSCAN = not trainable)
        # We compute it for logging/monitoring only
        dummy_param = next(self.reid_extractor.parameters())

        if self.use_gt_boxes:
            # Using GT boxes - perfect detection, no box loss needed
            box_loss = dummy_param.sum() * 0.0
        else:
            # Using DBSCAN boxes - compute loss for monitoring
            box_loss_value = None
            num_boxes = 0
            for b in range(batch_size):
                if len(pred_boxes[b]) > 0 and len(boxes_gt_tensors[b]) > 0:
                    loss_b = self.box_loss_fn(pred_boxes[b], boxes_gt_tensors[b])
                    if box_loss_value is None:
                        box_loss_value = loss_b
                    else:
                        box_loss_value = box_loss_value + loss_b
                    num_boxes += 1

            if box_loss_value is not None and num_boxes > 0:
                # Box loss exists but has no gradients (DBSCAN is not differentiable)
                # Add dummy gradient component so backward() doesn't fail
                box_loss_value = box_loss_value / num_boxes
                box_loss = dummy_param.sum() * 0.0 + box_loss_value.detach()  # Detach to prevent grad errors
            else:
                # No boxes - use pure zero loss with gradients
                box_loss = dummy_param.sum() * 0.0

        # Triplet loss (temporal - between consecutive frames t and t+1)
        # DISABLED during detection pretraining phase (first 25 epochs)
        triplet_loss = None
        detection_pretraining_epochs = 0
        if self.reid_config and isinstance(self.reid_config, dict):
            detection_pretraining_epochs = self.reid_config.get('detection_pretraining_epochs', 0)

        # Only compute triplet loss after detection pretraining phase
        if epoch is None or epoch >= detection_pretraining_epochs:
            if embeddings2 is not None and track_ids_gt2 is not None:
                # We have frame t+1 data - compute triplet loss between frames
                # Convert GT boxes and IDs for frame t+1
                boxes_gt2_tensors = []
                track_ids_from_boxes2_gt = []
                for b in range(len(boxes_gt2) if boxes_gt2 is not None else 0):
                    boxes_tensor, ids_tensor = convert_o3d_boxes_to_tensor(boxes_gt2[b])
                    boxes_gt2_tensors.append(boxes_tensor)

                    # Use track_ids_gt2 if available, otherwise use ids from dict
                    if track_ids_gt2 is not None and b < len(track_ids_gt2) and track_ids_gt2[b] is not None:
                        track_ids_from_boxes2_gt.append(track_ids_gt2[b])
                    else:
                        track_ids_from_boxes2_gt.append(ids_tensor)

                # Assign GT track IDs to predicted boxes (via IoU matching)
                # This ensures embeddings and track IDs have the same length
                track_ids_assigned_t = self._assign_gt_ids_to_pred_boxes(
                    pred_boxes, boxes_gt_tensors, track_ids_from_boxes
                )
                track_ids_assigned_t1 = self._assign_gt_ids_to_pred_boxes(
                    pred_boxes2, boxes_gt2_tensors, track_ids_from_boxes2_gt
                )

                # Compute triplet loss between frame t and t+1
                if len(embeddings) > 0 and len(embeddings2) > 0:
                    triplet_loss = self.triplet_loss_fn(
                        embeddings_t=embeddings,           # Frame t predicted embeddings
                        embeddings_t1=embeddings2,         # Frame t+1 predicted embeddings
                        track_ids_t=track_ids_assigned_t,  # Assigned GT IDs for frame t
                        track_ids_t1=track_ids_assigned_t1 # Assigned GT IDs for frame t+1
                    )

        if triplet_loss is None:
            # Create zero loss connected to computation graph (no frame t+1 or no matches)
            # Use model parameters to maintain gradient flow
            dummy_param = next(self.reid_extractor.parameters())
            triplet_loss = dummy_param.sum() * 0.0  # Connected to graph via parameters, results in 0

        # ===== CLASSIFICATION LOSS (NEW) =====
        # Classify objects into: vehicle, pedestrian, cyclist, unknown
        # Makes embeddings class-aware → better Re-ID performance
        classification_loss = None
        classification_acc = 0.0
        if class_logits is not None and class_labels_gt is not None:
            # Compute classification loss for frame t
            classification_loss, classification_acc = self.classification_loss_fn(
                class_logits_batch=class_logits,
                class_labels_batch=class_labels_gt
            )

        if classification_loss is None:
            # Create zero loss connected to computation graph
            dummy_param = next(self.reid_extractor.parameters())
            classification_loss = dummy_param.sum() * 0.0

        losses = {
            'box': box_loss,
            'reid_triplet': triplet_loss,
            'classification': classification_loss,  # NEW
            'classification_acc': classification_acc,  # For logging
        }

        return losses

    def _assign_gt_ids_to_pred_boxes(self, pred_boxes_list, gt_boxes_list, gt_ids_list):
        """
        Assign GT track IDs to predicted boxes based on IoU matching.

        Args:
            pred_boxes_list: List[Tensor] predicted boxes [M_i, 7] per batch
            gt_boxes_list: List[Tensor] GT boxes [N_i, 7] per batch
            gt_ids_list: List[Tensor] GT track IDs [N_i] per batch

        Returns:
            assigned_ids_list: List[Tensor] assigned track IDs [M_i] per batch
                               (ID = -1 for unmatched predictions)
        """
        assigned_ids_list = []

        for b in range(len(pred_boxes_list)):
            pred_boxes = pred_boxes_list[b]  # [M, 7]
            gt_boxes = gt_boxes_list[b]      # [N, 7]
            gt_ids = gt_ids_list[b]          # [N] or None

            M = len(pred_boxes)
            N = len(gt_boxes) if gt_boxes is not None else 0

            # If no GT IDs (evaluation mode), assign -1 to all predictions
            if gt_ids is None or M == 0 or N == 0:
                # No predictions or no GT - assign -1 to all
                assigned_ids = torch.full((M,), -1, dtype=torch.long, device=pred_boxes.device if M > 0 else torch.device('cuda'))
                assigned_ids_list.append(assigned_ids)
                continue

            # Compute IoU matrix [M, N]
            iou_matrix = self._compute_iou_matrix(pred_boxes, gt_boxes)

            # Greedy matching: assign each prediction to best GT match (if IoU > threshold)
            assigned_ids = torch.full((M,), -1, dtype=torch.long, device=pred_boxes.device)
            iou_threshold = 0.3  # Minimum IoU for valid match

            for i in range(M):
                ious = iou_matrix[i]  # [N]
                max_iou, max_idx = ious.max(dim=0)

                if max_iou > iou_threshold:
                    assigned_ids[i] = gt_ids[max_idx]

            assigned_ids_list.append(assigned_ids)

        return assigned_ids_list

    def _compute_iou_matrix(self, boxes1, boxes2):
        """
        Compute pairwise IoU matrix between two sets of boxes.
        Simplified 2D IoU (ignoring rotation for speed).

        Args:
            boxes1: [M, 7] boxes (x, y, z, l, w, h, yaw)
            boxes2: [N, 7] boxes

        Returns:
            iou_matrix: [M, N] pairwise IoU values
        """
        # Ensure both tensors are on the same device
        if boxes1.device != boxes2.device:
            boxes2 = boxes2.to(boxes1.device)

        M = len(boxes1)
        N = len(boxes2)

        # Extract x, y, l, w for BEV IoU
        x1, y1, l1, w1 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 3], boxes1[:, 4]
        x2, y2, l2, w2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 3], boxes2[:, 4]

        # Expand to [M, N] for pairwise computation
        x1 = x1.unsqueeze(1).expand(M, N)  # [M, N]
        y1 = y1.unsqueeze(1).expand(M, N)
        l1 = l1.unsqueeze(1).expand(M, N)
        w1 = w1.unsqueeze(1).expand(M, N)

        x2 = x2.unsqueeze(0).expand(M, N)  # [M, N]
        y2 = y2.unsqueeze(0).expand(M, N)
        l2 = l2.unsqueeze(0).expand(M, N)
        w2 = w2.unsqueeze(0).expand(M, N)

        # Axis-aligned bounding boxes (ignore rotation for simplicity)
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

    def _assign_points_to_gt_boxes(self, points, gt_boxes):
        """
        Assign each point to the nearest GT box.

        Args:
            points: [N, 3] point cloud
            gt_boxes: [M, 7] GT boxes (x, y, z, l, w, h, yaw)

        Returns:
            cluster_ids: [N] cluster assignment for each point (-1 for unassigned)
        """
        N = points.shape[0]
        M = gt_boxes.shape[0]

        if M == 0:
            # No GT boxes - all points unassigned
            return torch.full((N,), -1, dtype=torch.long, device=points.device)

        # Compute distance from each point to each box center
        # points: [N, 3], gt_boxes[:, :3]: [M, 3]
        box_centers = gt_boxes[:, :3]  # [M, 3]

        # Expand for broadcasting: [N, 1, 3] - [1, M, 3] -> [N, M, 3]
        distances = torch.cdist(points, box_centers)  # [N, M]

        # Assign each point to nearest box
        min_distances, cluster_ids = distances.min(dim=1)  # [N]

        # Only assign points within a reasonable distance (e.g., 5 meters)
        max_distance = 5.0
        cluster_ids[min_distances > max_distance] = -1

        return cluster_ids

    def is_reid_mode(self):
        """Check if we're in Re-ID training mode."""
        return self.train_mode == 'reid_only'

    def use_gt_inputs(self):
        """Check if we should use GT inputs."""
        return self.use_gt_segmentation or self.train_mode == 'reid_only'
