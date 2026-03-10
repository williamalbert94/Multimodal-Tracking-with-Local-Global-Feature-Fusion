"""
Utilities for computing size-based metrics in tracking evaluation.

This module provides functions to categorize objects by size (small/medium/large)
based on their 3D bounding box volumes, and assign these categories to points.
"""

import torch
import numpy as np


def compute_object_volumes(labels_batch):
    """
    Compute volume for each object in the batch.

    Args:
        labels_batch: list of dicts {obj_id: Object_3D}, one per batch element
                     Each Object_3D has attributes: h (height), w (width), l (length)

    Returns:
        volumes_batch: list of dicts {obj_id: volume_m3}, one per batch element

    Example:
        >>> labels_batch = [{1: Object_3D(h=1.5, w=1.6, l=4.5), 2: Object_3D(h=1.8, w=0.6, l=1.8)}]
        >>> volumes = compute_object_volumes(labels_batch)
        >>> volumes
        [{1: 10.8, 2: 1.944}]  # Car (medium) and Cyclist (small)
    """
    volumes_batch = []

    for labels in labels_batch:
        volumes = {}
        for obj_id, label in labels.items():
            # Volume = height × width × length
            volume = label.h * label.w * label.l
            volumes[obj_id] = volume

        volumes_batch.append(volumes)

    return volumes_batch


def categorize_objects_by_size(volumes_batch, size_thresholds):
    """
    Categorize objects into small/medium/large based on volume.

    Args:
        volumes_batch: list of dicts {obj_id: volume}, one per batch element
        size_thresholds: dict with keys 'small_max' and 'medium_max' (in m³)
                        Example: {'small_max': 10.0, 'medium_max': 50.0}

    Returns:
        categories_batch: list of dicts {obj_id: 'small'|'medium'|'large'}

    Size categories:
        - Small: volume < small_max (e.g., < 10 m³) - pedestrians, cyclists
        - Medium: small_max <= volume < medium_max (e.g., 10-50 m³) - cars, vans
        - Large: volume >= medium_max (e.g., >= 50 m³) - trucks, buses

    Example:
        >>> volumes_batch = [{1: 10.8, 2: 1.944, 3: 75.0}]
        >>> thresholds = {'small_max': 10.0, 'medium_max': 50.0}
        >>> categories = categorize_objects_by_size(volumes_batch, thresholds)
        >>> categories
        [{1: 'medium', 2: 'small', 3: 'large'}]
    """
    small_max = size_thresholds['small_max']
    medium_max = size_thresholds['medium_max']

    categories_batch = []

    for volumes in volumes_batch:
        categories = {}
        for obj_id, volume in volumes.items():
            if volume < small_max:
                categories[obj_id] = 'small'
            elif volume < medium_max:
                categories[obj_id] = 'medium'
            else:
                categories[obj_id] = 'large'

        categories_batch.append(categories)

    return categories_batch


def assign_size_to_points(cls_obj_id_batch, categories_batch):
    """
    Assign size category to each point based on which object it belongs to.

    Args:
        cls_obj_id_batch: list of tensors [N] with object ID per point
                         -1 or negative values indicate static/background points
        categories_batch: list of dicts {obj_id: 'small'|'medium'|'large'}

    Returns:
        size_per_point_batch: tensor [B, N] with size category per point
                             0 = static/unknown (not in any object)
                             1 = small
                             2 = medium
                             3 = large

    Example:
        >>> cls_obj_id_batch = [torch.tensor([-1, 1, 1, 2, -1, 3])]  # 6 points
        >>> categories_batch = [{1: 'medium', 2: 'small', 3: 'large'}]
        >>> size_per_point = assign_size_to_points(cls_obj_id_batch, categories_batch)
        >>> size_per_point
        tensor([[0, 2, 2, 1, 0, 3]])  # static, medium, medium, small, static, large
    """
    # Category to integer mapping
    category_to_int = {
        'small': 1,
        'medium': 2,
        'large': 3
    }

    size_per_point_list = []

    for cls_obj_id, categories in zip(cls_obj_id_batch, categories_batch):
        # Initialize all points as 0 (static/unknown)
        num_points = cls_obj_id.shape[0]
        size_per_point = torch.zeros(num_points, dtype=torch.long, device=cls_obj_id.device)

        # Assign size category to each point based on its object ID
        for obj_id, category in categories.items():
            # Find all points belonging to this object
            mask = (cls_obj_id == obj_id)
            # Assign the category integer
            size_per_point[mask] = category_to_int[category]

        size_per_point_list.append(size_per_point)

    # Stack into batch tensor [B, N]
    if len(size_per_point_list) > 0:
        size_per_point_batch = torch.stack(size_per_point_list)
    else:
        # Empty batch
        size_per_point_batch = torch.empty((0, 0), dtype=torch.long)

    return size_per_point_batch
