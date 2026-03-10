"""
Instance Point Assignment utilities.
Assign each foreground point to a detected instance (tracked box).
"""

import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


def point_in_box_2d(point, box):
    """
    Check if a point is inside a 2D bounding box (Bird's Eye View).

    Args:
        point: [2] or [3] - (x, y) or (x, y, z)
        box: [7] - (x, y, z, l, w, h, yaw)

    Returns:
        bool - True if point is inside box
    """
    px, py = point[0], point[1]
    bx, by, l, w = box[0], box[1], box[3], box[4]

    # Simplified axis-aligned check (ignoring yaw for now)
    # TODO: Add rotation handling for yaw
    min_x, max_x = bx - l/2, bx + l/2
    min_y, max_y = by - w/2, by + w/2

    return (min_x <= px <= max_x) and (min_y <= py <= max_y)


def assign_points_to_boxes(points, boxes, track_ids, return_mask=False):
    """
    Assign each point to the nearest detected box instance.

    Args:
        points: [N, 3] tensor or ndarray - Point cloud (x, y, z)
        boxes: [M, 7] tensor or ndarray - Detected boxes (x, y, z, l, w, h, yaw)
        track_ids: [M] tensor or ndarray - Track ID for each box
        return_mask: bool - If True, return boolean mask per instance

    Returns:
        point_instance_ids: [N] ndarray - Instance ID for each point (-1 = background)
        OR
        instance_masks: dict {track_id: [N] boolean mask} if return_mask=True
    """
    # Convert to numpy if needed
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(track_ids, torch.Tensor):
        track_ids = track_ids.cpu().numpy()

    N = len(points)
    M = len(boxes)

    # Initialize all points as background (-1)
    point_instance_ids = np.full(N, -1, dtype=np.int64)

    if M == 0:
        # No boxes detected
        if return_mask:
            return {}
        return point_instance_ids

    # Handle None track_ids (can happen when no GT matches in evaluation)
    if track_ids is None:
        logger.warning(f"track_ids is None, cannot assign points to boxes")
        if return_mask:
            return {}
        return point_instance_ids

    # Validate that boxes and track_ids have same length
    if len(track_ids) != M:
        logger.warning(f"Mismatch: {M} boxes but {len(track_ids)} track_ids. Using min length.")
        M = min(M, len(track_ids))
        boxes = boxes[:M]
        track_ids = track_ids[:M]

    # For each point, find which box it belongs to
    for i, point in enumerate(points):
        # Check all boxes
        for box_idx, box in enumerate(boxes):
            if point_in_box_2d(point, box):
                # Safety check: validate box_idx is within track_ids bounds
                if box_idx >= len(track_ids):
                    logger.warning(f"box_idx {box_idx} >= len(track_ids) {len(track_ids)} - skipping point {i}")
                    continue
                # Assign track ID to this point
                point_instance_ids[i] = track_ids[box_idx]
                break  # Point can only belong to one box

    if return_mask:
        # Return masks per instance
        instance_masks = {}
        unique_ids = np.unique(track_ids)
        for track_id in unique_ids:
            instance_masks[int(track_id)] = (point_instance_ids == track_id)
        return instance_masks

    return point_instance_ids


def assign_points_to_boxes_nearest(points, boxes, track_ids):
    """
    Assign each foreground point to the nearest box center (fallback method).

    Args:
        points: [N, 3] - Point cloud
        boxes: [M, 7] - Detected boxes
        track_ids: [M] - Track IDs

    Returns:
        point_instance_ids: [N] - Instance ID per point
    """
    # Convert to numpy
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(track_ids, torch.Tensor):
        track_ids = track_ids.cpu().numpy()

    N = len(points)
    M = len(boxes)

    point_instance_ids = np.full(N, -1, dtype=np.int64)

    if M == 0:
        return point_instance_ids

    # Extract box centers
    box_centers = boxes[:, :2]  # [M, 2] - (x, y)
    point_xy = points[:, :2]     # [N, 2]

    # Compute distances from each point to each box center
    # distances[i, j] = distance from point i to box j
    distances = np.linalg.norm(
        point_xy[:, np.newaxis, :] - box_centers[np.newaxis, :, :],
        axis=2
    )  # [N, M]

    # Assign each point to nearest box (if within reasonable distance)
    nearest_box_idx = np.argmin(distances, axis=1)  # [N]
    nearest_dist = np.min(distances, axis=1)         # [N]

    # Only assign if within box dimensions (rough heuristic)
    for i in range(N):
        box_idx = nearest_box_idx[i]
        # Safety check: validate box_idx is within track_ids bounds
        if box_idx >= len(track_ids):
            logger.warning(f"box_idx {box_idx} >= len(track_ids) {len(track_ids)} in nearest assignment")
            continue
        box = boxes[box_idx]
        max_dist = max(box[3], box[4])  # max(length, width)

        if nearest_dist[i] <= max_dist:
            point_instance_ids[i] = track_ids[box_idx]

    return point_instance_ids


def get_instance_statistics(point_instance_ids, points):
    """
    Get statistics about detected instances.

    Args:
        point_instance_ids: [N] - Instance ID per point
        points: [N, 3] - Point cloud

    Returns:
        stats: dict with instance statistics
    """
    unique_ids = np.unique(point_instance_ids)
    unique_ids = unique_ids[unique_ids >= 0]  # Remove background

    stats = {
        'num_instances': len(unique_ids),
        'num_background_points': np.sum(point_instance_ids < 0),
        'num_instance_points': np.sum(point_instance_ids >= 0),
        'instances': {}
    }

    for instance_id in unique_ids:
        mask = point_instance_ids == instance_id
        instance_points = points[mask]

        stats['instances'][int(instance_id)] = {
            'num_points': np.sum(mask),
            'centroid': instance_points.mean(axis=0),
            'bbox_min': instance_points.min(axis=0),
            'bbox_max': instance_points.max(axis=0),
        }

    return stats
