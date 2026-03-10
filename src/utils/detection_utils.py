"""
Detection Training Utilities for Phase 1

Functions for:
  - Extracting boxes from detection head predictions
  - Non-Maximum Suppression (NMS)
  - Visualization of detection results (boxes pred vs GT)
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def extract_boxes_from_detection(detection_pred, pc1, threshold=0.3, nms_iou_threshold=0.5):
    """
    Extract 3D boxes from detection head predictions with NMS.

    Args:
        detection_pred: dict with keys
            'center': [B, 1, N] - center heatmap (probability)
            'size': [B, 3, N] - box dimensions (l, w, h)
            'orientation': [B, 2, N] - (sin(yaw), cos(yaw))
            'class': [B, num_classes, N] - class logits
        pc1: [B, 3, N] point cloud coordinates (channel-first)
        threshold: center heatmap threshold (default: 0.3)
        nms_iou_threshold: IoU threshold for NMS (default: 0.5)

    Returns:
        boxes_list: List[Tensor] boxes per batch [M, 7] (x, y, z, l, w, h, yaw)
        scores_list: List[Tensor] scores per batch [M]
        classes_list: List[Tensor] class indices per batch [M]
    """
    B, _, N = pc1.shape
    device = pc1.device

    boxes_list = []
    scores_list = []
    classes_list = []

    for b in range(B):
        # Get center scores
        center_scores = detection_pred['center'][b, 0]  # [N]

        # Threshold centers
        center_mask = center_scores > threshold

        if center_mask.sum() == 0:
            # No detections
            boxes_list.append(torch.empty(0, 7, device=device))
            scores_list.append(torch.empty(0, device=device))
            classes_list.append(torch.empty(0, dtype=torch.long, device=device))
            continue

        # Get box parameters for center points
        centers = pc1[b, :, center_mask].T  # [M, 3]
        sizes = detection_pred['size'][b, :, center_mask].T  # [M, 3]
        ori = detection_pred['orientation'][b, :, center_mask].T  # [M, 2]
        class_logits = detection_pred['class'][b, :, center_mask].T  # [M, num_classes]
        scores = center_scores[center_mask]  # [M]

        # Convert sin/cos to yaw
        yaw = torch.atan2(ori[:, 0], ori[:, 1])  # [M]

        # Get predicted classes
        pred_classes = class_logits.argmax(dim=1)  # [M]

        # Concatenate to boxes [M, 7] (x, y, z, l, w, h, yaw)
        boxes = torch.cat([centers, sizes, yaw.unsqueeze(1)], dim=1)  # [M, 7]

        # Apply NMS
        keep_indices = nms_3d_boxes(boxes, scores, nms_iou_threshold)

        boxes_list.append(boxes[keep_indices])
        scores_list.append(scores[keep_indices])
        classes_list.append(pred_classes[keep_indices])

    return boxes_list, scores_list, classes_list


def nms_3d_boxes(boxes, scores, iou_threshold=0.5):
    """
    Non-Maximum Suppression for 3D boxes (2D IoU in BEV).

    Args:
        boxes: [M, 7] (x, y, z, l, w, h, yaw)
        scores: [M] confidence scores
        iou_threshold: IoU threshold for NMS

    Returns:
        keep: [K] indices to keep
    """
    if len(boxes) == 0:
        return torch.empty(0, dtype=torch.long)

    # Sort by scores descending
    sorted_indices = torch.argsort(scores, descending=True)

    keep = []
    while len(sorted_indices) > 0:
        # Take box with highest score
        idx = sorted_indices[0]
        keep.append(idx.item())

        if len(sorted_indices) == 1:
            break

        # Compute IoU with remaining boxes
        ious = compute_iou_2d_bev(boxes[idx:idx+1], boxes[sorted_indices[1:]])  # [1, M-1]

        # Remove boxes with IoU > threshold
        mask = ious[0] <= iou_threshold
        sorted_indices = sorted_indices[1:][mask]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def compute_iou_2d_bev(boxes1, boxes2):
    """
    Compute 2D IoU in Bird's Eye View (BEV) - VECTORIZED.

    Args:
        boxes1: [N, 7] (x, y, z, l, w, h, yaw)
        boxes2: [M, 7]

    Returns:
        iou: [N, M] IoU matrix
    """
    N = boxes1.shape[0]
    M = boxes2.shape[0]

    if N == 0 or M == 0:
        return torch.zeros(N, M, device=boxes1.device)

    # VECTORIZED: Compute all box corners at once (axis-aligned approximation)
    # boxes1: [N, 7] → expand to [N, 1, 7] for broadcasting
    # boxes2: [M, 7] → expand to [1, M, 7] for broadcasting
    boxes1_exp = boxes1.unsqueeze(1)  # [N, 1, 7]
    boxes2_exp = boxes2.unsqueeze(0)  # [1, M, 7]

    # Extract coordinates (broadcast to [N, M])
    x1_min = boxes1_exp[..., 0] - boxes1_exp[..., 3] / 2  # [N, 1]
    x1_max = boxes1_exp[..., 0] + boxes1_exp[..., 3] / 2
    y1_min = boxes1_exp[..., 1] - boxes1_exp[..., 4] / 2
    y1_max = boxes1_exp[..., 1] + boxes1_exp[..., 4] / 2

    x2_min = boxes2_exp[..., 0] - boxes2_exp[..., 3] / 2  # [1, M]
    x2_max = boxes2_exp[..., 0] + boxes2_exp[..., 3] / 2
    y2_min = boxes2_exp[..., 1] - boxes2_exp[..., 4] / 2
    y2_max = boxes2_exp[..., 1] + boxes2_exp[..., 4] / 2

    # Intersection (broadcast to [N, M])
    inter_x_min = torch.max(x1_min, x2_min)  # [N, M]
    inter_x_max = torch.min(x1_max, x2_max)
    inter_y_min = torch.max(y1_min, y2_min)
    inter_y_max = torch.min(y1_max, y2_max)

    # Intersection area (clamp to 0 if no overlap)
    inter_w = torch.clamp(inter_x_max - inter_x_min, min=0)
    inter_h = torch.clamp(inter_y_max - inter_y_min, min=0)
    inter_area = inter_w * inter_h  # [N, M]

    # Box areas
    area1 = (x1_max - x1_min) * (y1_max - y1_min)  # [N, 1]
    area2 = (x2_max - x2_min) * (y2_max - y2_min)  # [1, M]

    # Union area
    union_area = area1 + area2 - inter_area  # [N, M]

    # IoU
    ious = inter_area / (union_area + 1e-6)  # [N, M]

    return ious


def visualize_detection_results(pc1, boxes_pred, boxes_gt, save_path, sample_idx=0):
    """
    Visualize detection results (boxes pred vs GT) in BEV.

    Args:
        pc1: [B, 3, N] point cloud
        boxes_pred: List[Tensor] predicted boxes per batch [M, 7]
        boxes_gt: List[Tensor] GT boxes per batch [M_gt, 7]
        save_path: Path to save visualization
        sample_idx: Index of sample to visualize (default: 0)
    """
    # Extract sample
    if isinstance(pc1, torch.Tensor):
        pc1_np = pc1[sample_idx].cpu().numpy()  # [3, N]
    else:
        pc1_np = pc1  # Already numpy

    boxes_pred_np = boxes_pred[sample_idx].cpu().numpy() if len(boxes_pred[sample_idx]) > 0 else np.empty((0, 7))
    boxes_gt_np = boxes_gt[sample_idx].cpu().numpy() if len(boxes_gt[sample_idx]) > 0 else np.empty((0, 7))

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ===== Plot 1: Predicted Boxes =====
    ax = axes[0]
    ax.scatter(pc1_np[0], pc1_np[1], c='gray', s=1, alpha=0.3, label='Points')

    # Draw predicted boxes
    for box in boxes_pred_np:
        x, y, z, l, w, h, yaw = box
        # Get 4 corners
        corners = get_box_corners_2d(x, y, l, w, yaw)
        ax.plot(corners[:, 0], corners[:, 1], 'b-', linewidth=2, label='Pred' if box is boxes_pred_np[0] else '')

    ax.set_title(f'Predicted Boxes ({len(boxes_pred_np)} detections)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axis('equal')

    # ===== Plot 2: Ground Truth Boxes =====
    ax = axes[1]
    ax.scatter(pc1_np[0], pc1_np[1], c='gray', s=1, alpha=0.3, label='Points')

    # Draw GT boxes
    for box in boxes_gt_np:
        x, y, z, l, w, h, yaw = box
        corners = get_box_corners_2d(x, y, l, w, yaw)
        ax.plot(corners[:, 0], corners[:, 1], 'r-', linewidth=2, label='GT' if box is boxes_gt_np[0] else '')

    ax.set_title(f'Ground Truth Boxes ({len(boxes_gt_np)} objects)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axis('equal')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def get_box_corners_2d(x, y, l, w, yaw):
    """
    Get 2D box corners in BEV.

    Args:
        x, y: center coordinates
        l, w: length, width
        yaw: rotation angle

    Returns:
        corners: [5, 2] corners (closed polygon)
    """
    # Half dimensions
    half_l, half_w = l / 2, w / 2

    # Corners in local frame
    corners_local = np.array([
        [-half_l, -half_w],
        [half_l, -half_w],
        [half_l, half_w],
        [-half_l, half_w],
        [-half_l, -half_w]  # Close polygon
    ])

    # Rotation matrix
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    rot_matrix = np.array([
        [cos_yaw, -sin_yaw],
        [sin_yaw, cos_yaw]
    ])

    # Rotate and translate
    corners_global = corners_local @ rot_matrix.T
    corners_global[:, 0] += x
    corners_global[:, 1] += y

    return corners_global


def convert_o3d_boxes_to_tensor_list(boxes_o3d_list, device='cuda'):
    """
    Convert Open3D boxes to tensor list format.

    Args:
        boxes_o3d_list: List of Open3D box dicts per batch
                       Each element is dict{obj_id: Open3D_Box}
        device: Device to place tensors on (default: 'cuda')

    Returns:
        boxes_tensor_list: List[Tensor] boxes per batch [M, 7]
    """
    boxes_tensor_list = []

    for boxes_o3d in boxes_o3d_list:
        if len(boxes_o3d) == 0 or boxes_o3d is None:
            boxes_tensor_list.append(torch.empty(0, 7, device=device))
            continue

        boxes_batch = []

        # Check if boxes_o3d is a dict (obj_id -> box) or list of boxes
        if isinstance(boxes_o3d, dict):
            # boxes_o3d is dict{obj_id: Open3D_Box}
            # OPTIMIZATION: Batch convert all boxes at once to reduce CPU→GPU transfers
            boxes_np_list = []
            for obj_id, box in boxes_o3d.items():
                # Extract box parameters from Open3D box (CPU operations)
                center = box.get_center()  # [3]
                extent = box.extent  # [3] (l, w, h)
                R = np.array(box.R)  # [3, 3] rotation matrix

                # Extract yaw from rotation matrix
                yaw = np.arctan2(R[1, 0], R[0, 0])

                # Concatenate [x, y, z, l, w, h, yaw] as numpy array
                box_np = np.array([
                    center[0], center[1], center[2],
                    extent[0], extent[1], extent[2],
                    yaw
                ], dtype=np.float32)

                boxes_np_list.append(box_np)

            # Single CPU→GPU transfer for all boxes
            if len(boxes_np_list) > 0:
                boxes_np_array = np.stack(boxes_np_list)  # [M, 7]
                boxes_batch_tensor = torch.from_numpy(boxes_np_array).to(device)  # Single transfer
                # Split back into list for compatibility
                for i in range(len(boxes_batch_tensor)):
                    boxes_batch.append(boxes_batch_tensor[i])
        else:
            # boxes_o3d is a list of box dicts (fallback for other formats)
            # OPTIMIZATION: Batch convert all boxes at once
            boxes_np_list = []
            for box_dict in boxes_o3d:
                if isinstance(box_dict, dict) and 'center' in box_dict:
                    center = box_dict['center']  # [3]
                    extent = box_dict['extent']  # [3] (l, w, h)
                    R = box_dict['R']  # [3, 3] rotation matrix

                    # Extract yaw from rotation matrix
                    yaw = np.arctan2(R[1, 0], R[0, 0])

                    # Concatenate [x, y, z, l, w, h, yaw] as numpy array
                    box_np = np.array([
                        center[0], center[1], center[2],
                        extent[0], extent[1], extent[2],
                        yaw
                    ], dtype=np.float32)

                    boxes_np_list.append(box_np)

            # Single CPU→GPU transfer for all boxes
            if len(boxes_np_list) > 0:
                boxes_np_array = np.stack(boxes_np_list)  # [M, 7]
                boxes_batch_tensor = torch.from_numpy(boxes_np_array).to(device)
                for i in range(len(boxes_batch_tensor)):
                    boxes_batch.append(boxes_batch_tensor[i])

        if len(boxes_batch) == 0:
            boxes_tensor_list.append(torch.empty(0, 7, device=device))
        else:
            boxes_tensor_list.append(torch.stack(boxes_batch))

    return boxes_tensor_list
