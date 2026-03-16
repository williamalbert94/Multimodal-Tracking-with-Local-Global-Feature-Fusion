"""
Detection Metrics for Object Detection and Segmentation.

Metrics:
  - mAP (mean Average Precision) for 3D box detection
  - mIoU (mean Intersection over Union) for segmentation
  - Per-class metrics
"""

import numpy as np
import torch
from collections import defaultdict


def compute_iou_3d_boxes(box1, box2):
    """
    Compute 3D IoU between two boxes (simplified 2D BEV IoU).

    Args:
        box1: [7] (x, y, z, l, w, h, yaw)
        box2: [7]

    Returns:
        iou: float
    """
    # Extract x, y, l, w for Bird's Eye View
    x1, y1, l1, w1 = box1[0], box1[1], box1[3], box1[4]
    x2, y2, l2, w2 = box2[0], box2[1], box2[3], box2[4]

    # Axis-aligned bounding boxes (ignoring yaw for simplicity)
    min_x1, max_x1 = x1 - l1/2, x1 + l1/2
    min_y1, max_y1 = y1 - w1/2, y1 + w1/2

    min_x2, max_x2 = x2 - l2/2, x2 + l2/2
    min_y2, max_y2 = y2 - w2/2, y2 + w2/2

    # Intersection
    inter_min_x = max(min_x1, min_x2)
    inter_max_x = min(max_x1, max_x2)
    inter_min_y = max(min_y1, min_y2)
    inter_max_y = min(max_y1, max_y2)

    inter_w = max(0, inter_max_x - inter_min_x)
    inter_h = max(0, inter_max_y - inter_min_y)
    inter_area = inter_w * inter_h

    # Union
    area1 = l1 * w1
    area2 = l2 * w2
    union_area = area1 + area2 - inter_area

    # IoU
    iou = inter_area / (union_area + 1e-6)
    return iou


def compute_iou_matrix(boxes_pred, boxes_gt):
    """
    Compute IoU matrix between predicted and GT boxes.

    Args:
        boxes_pred: [M, 7] predicted boxes
        boxes_gt: [N, 7] GT boxes

    Returns:
        iou_matrix: [M, N]
    """
    if isinstance(boxes_pred, torch.Tensor):
        boxes_pred = boxes_pred.cpu().numpy()
    if isinstance(boxes_gt, torch.Tensor):
        boxes_gt = boxes_gt.cpu().numpy()

    M = len(boxes_pred)
    N = len(boxes_gt)

    iou_matrix = np.zeros((M, N))

    for i in range(M):
        for j in range(N):
            iou_matrix[i, j] = compute_iou_3d_boxes(boxes_pred[i], boxes_gt[j])

    return iou_matrix


def compute_average_precision(recalls, precisions):
    """
    Compute Average Precision (AP) given precision-recall curve.

    Uses 11-point interpolation.

    Args:
        recalls: [K] recall values
        precisions: [K] precision values

    Returns:
        ap: float, average precision
    """
    # 11-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        # Get max precision for recall >= t
        mask = recalls >= t
        if np.any(mask):
            p = np.max(precisions[mask])
        else:
            p = 0
        ap += p / 11.0

    return ap


def compute_map_3d(boxes_pred_list, boxes_gt_list, class_pred_list=None, class_gt_list=None,
                   iou_thresholds=[0.3, 0.5, 0.7], per_class=False):
    """
    Compute mAP (mean Average Precision) for 3D box detection.

    Args:
        boxes_pred_list: List[Tensor] predicted boxes per sample [M_i, 7]
        boxes_gt_list: List[Tensor] GT boxes per sample [N_i, 7]
        class_pred_list: List[Tensor] predicted class labels per sample [M_i] (optional)
        class_gt_list: List[Tensor] GT class labels per sample [N_i] (optional)
        iou_thresholds: List[float] IoU thresholds for matching
        per_class: bool, compute per-class AP

    Returns:
        metrics: dict with mAP@IoU for each threshold
    """
    metrics = {}

    for iou_thresh in iou_thresholds:
        # Collect all predictions and GT across samples
        all_tp = []  # True positives
        all_fp = []  # False positives
        all_scores = []  # Confidence scores (use IoU as score)
        all_num_gt = 0

        for i in range(len(boxes_pred_list)):
            boxes_pred = boxes_pred_list[i]
            boxes_gt = boxes_gt_list[i]

            if isinstance(boxes_pred, torch.Tensor):
                boxes_pred = boxes_pred.cpu().numpy()
            if isinstance(boxes_gt, torch.Tensor):
                boxes_gt = boxes_gt.cpu().numpy()

            M = len(boxes_pred)
            N = len(boxes_gt)

            all_num_gt += N

            if M == 0:
                continue

            if N == 0:
                # All predictions are false positives
                all_fp.extend([1] * M)
                all_tp.extend([0] * M)
                all_scores.extend([0.0] * M)
                continue

            # Compute IoU matrix
            iou_matrix = compute_iou_matrix(boxes_pred, boxes_gt)

            # Match predictions to GT (greedy by IoU)
            matched_gt = set()

            for pred_idx in range(M):
                ious = iou_matrix[pred_idx]
                max_iou = np.max(ious)
                max_gt_idx = np.argmax(ious)

                if max_iou >= iou_thresh and max_gt_idx not in matched_gt:
                    # True positive
                    all_tp.append(1)
                    all_fp.append(0)
                    matched_gt.add(max_gt_idx)
                else:
                    # False positive
                    all_tp.append(0)
                    all_fp.append(1)

                all_scores.append(max_iou)

        # Sort by score (descending)
        sorted_indices = np.argsort(all_scores)[::-1]
        tp_sorted = np.array(all_tp)[sorted_indices]
        fp_sorted = np.array(all_fp)[sorted_indices]

        # Compute precision-recall curve
        tp_cumsum = np.cumsum(tp_sorted)
        fp_cumsum = np.cumsum(fp_sorted)

        recalls = tp_cumsum / max(all_num_gt, 1)
        precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1)

        # Compute AP
        ap = compute_average_precision(recalls, precisions)

        metrics[f'mAP@{iou_thresh}'] = ap

        # Add F1 score at best threshold
        if len(precisions) > 0 and len(recalls) > 0:
            f1_scores = 2 * (precisions * recalls) / np.maximum(precisions + recalls, 1e-6)
            best_f1 = np.max(f1_scores)
            metrics[f'F1@{iou_thresh}'] = best_f1

    # Average mAP across thresholds
    metrics['mAP'] = np.mean([metrics[f'mAP@{t}'] for t in iou_thresholds])

    return metrics


def compute_miou_segmentation(seg_pred_list, seg_gt_list, num_classes=2):
    """
    Compute mIoU (mean Intersection over Union) for segmentation.

    Args:
        seg_pred_list: List[Tensor] predicted segmentation masks [N_i]
        seg_gt_list: List[Tensor] GT segmentation masks [N_i]
        num_classes: int, number of classes (2 for binary: bg/fg)

    Returns:
        metrics: dict with mIoU and per-class IoU
    """
    # Accumulate confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes))

    for i in range(len(seg_pred_list)):
        seg_pred = seg_pred_list[i]
        seg_gt = seg_gt_list[i]

        if isinstance(seg_pred, torch.Tensor):
            seg_pred = seg_pred.cpu().numpy()
        if isinstance(seg_gt, torch.Tensor):
            seg_gt = seg_gt.cpu().numpy()

        # Flatten
        seg_pred = seg_pred.flatten()
        seg_gt = seg_gt.flatten()

        # Update confusion matrix
        for c_pred in range(num_classes):
            for c_gt in range(num_classes):
                mask_pred = (seg_pred == c_pred)
                mask_gt = (seg_gt == c_gt)
                confusion_matrix[c_pred, c_gt] += np.sum(mask_pred & mask_gt)

    # Compute IoU per class
    iou_per_class = []
    for c in range(num_classes):
        intersection = confusion_matrix[c, c]
        union = np.sum(confusion_matrix[c, :]) + np.sum(confusion_matrix[:, c]) - intersection
        iou = intersection / max(union, 1)
        iou_per_class.append(iou)

    # mIoU
    miou = np.mean(iou_per_class)

    metrics = {
        'mIoU': miou,
        'IoU_background': iou_per_class[0] if num_classes > 0 else 0,
        'IoU_foreground': iou_per_class[1] if num_classes > 1 else 0,
    }

    return metrics


def compute_detection_metrics_epoch(boxes_pred_list, boxes_gt_list, seg_pred_list=None, seg_gt_list=None):
    """
    Compute detection metrics for an entire epoch.

    Args:
        boxes_pred_list: List[Tensor] predicted boxes per sample
        boxes_gt_list: List[Tensor] GT boxes per sample
        seg_pred_list: List[Tensor] predicted segmentation (optional)
        seg_gt_list: List[Tensor] GT segmentation (optional)

    Returns:
        metrics: dict with mAP, mIoU, F1, etc.
    """
    metrics = {}

    # mAP for box detection
    if boxes_pred_list and boxes_gt_list:
        map_metrics = compute_map_3d(
            boxes_pred_list, boxes_gt_list,
            iou_thresholds=[0.3, 0.5, 0.7]
        )
        metrics.update(map_metrics)

    # mIoU for segmentation
    if seg_pred_list and seg_gt_list:
        miou_metrics = compute_miou_segmentation(seg_pred_list, seg_gt_list)
        metrics.update(miou_metrics)

    return metrics


def print_detection_metrics(metrics, prefix=''):
    """
    Print detection metrics in a formatted way.

    Args:
        metrics: dict with metric values
        prefix: str, prefix for logging (e.g., 'Train', 'Val')
    """
    # print(f"\n{'='*60}")
    # print(f"{prefix} Detection Metrics")
    # print(f"{'='*60}")
    #
    # # Box detection metrics
    # if 'mAP' in metrics:
    #     print(f"  Box Detection:")
    #     print(f"     mAP (avg):     {metrics['mAP']:.4f}")
    #     if 'mAP@0.3' in metrics:
    #         print(f"     mAP@0.3:       {metrics['mAP@0.3']:.4f}")
    #     if 'mAP@0.5' in metrics:
    #         print(f"     mAP@0.5:       {metrics['mAP@0.5']:.4f}")
    #     if 'mAP@0.7' in metrics:
    #         print(f"     mAP@0.7:       {metrics['mAP@0.7']:.4f}")
    #
    #     if 'F1@0.5' in metrics:
    #         print(f"     F1@0.5:        {metrics['F1@0.5']:.4f}")
    #
    # # Segmentation metrics
    # if 'mIoU' in metrics:
    #     print(f"\n  Segmentation:")
    #     print(f"     mIoU:          {metrics['mIoU']:.4f}")
    #     if 'IoU_foreground' in metrics:
    #         print(f"     IoU (fg):      {metrics['IoU_foreground']:.4f}")
    #     if 'IoU_background' in metrics:
    #         print(f"     IoU (bg):      {metrics['IoU_background']:.4f}")
    #
    # print(f"{'='*60}\n")
    pass
