"""
3-Panel Evaluation Visualization
Shows GT BEV | RGB Projection | Prediction BEV side-by-side.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.visualization_rgb import load_rgb_for_frame, plot_rgb_projection_simple
from external.gnd.module.lidar_projection import birds_eye_point_cloud


# Class color scheme (matching VOD dataset classes)
CLASS_COLORS = {
    'car': 'red',
    'van': 'red',
    'truck': 'red',
    'pedestrian': 'green',
    'person_sitting': 'green',
    'cyclist': 'blue',
    'rider': 'blue',
    'unknown': 'gray',
    'background': 'lightgray'
}

# Segmentation label colors (0=background, 1=moving, 2=in_box)
SEG_COLORS = {
    0: 'lightgray',  # Background/static
    1: 'orange',     # Moving (motion detected)
    2: 'yellow'      # Inside inferred box
}


def plot_bev_with_segmentation(ax, pc_lidar, pc_radar, seg_labels, class_labels,
                               boxes, ids, title, is_gt=True):
    """
    Plot Bird's Eye View with lidar background, colored radar segmentation, and boxes.

    Args:
        ax: matplotlib axis
        pc_lidar: [N, 3] lidar points (x, y, z)
        pc_radar: [M, 3] radar points (x, y, z)
        seg_labels: [M] segmentation labels (0=background, 1=moving) for radar points
        class_labels: [M] class names for radar points (or None)
        boxes: [K, 7] boxes (x, y, z, l, w, h, yaw)
        ids: [K] track IDs
        title: str - subplot title
        is_gt: bool - whether this is GT (affects styling)
    """
    # BEV parameters
    side_range = (-30, 30)  # left-right
    fwd_range = (0, 75)      # forward range
    res = 0.1                # meters per pixel

    # Generate BEV background from lidar
    if pc_lidar is not None and len(pc_lidar) > 0:
        try:
            bev_image = birds_eye_point_cloud(
                pc_lidar,
                side_range=side_range,
                fwd_range=fwd_range,
                res=res
            )
            # Normalize and display as grayscale background
            bev_image = (bev_image - bev_image.min()) / (bev_image.max() - bev_image.min() + 1e-6)
            ax.imshow(bev_image, cmap='gray', origin='lower',
                     extent=[side_range[0], side_range[1], fwd_range[0], fwd_range[1]],
                     alpha=0.3)
        except Exception as e:
            print(f"Warning: Could not generate BEV: {e}")

    # Overlay radar points colored by class (for GT) or segmentation (for pred)
    if pc_radar is not None and len(pc_radar) > 0:
        x_radar = pc_radar[:, 0]
        y_radar = pc_radar[:, 1]

        if is_gt and class_labels is not None:
            # GT: Color by object class
            colors = []
            for label in class_labels:
                label_str = label if isinstance(label, str) else 'unknown'
                colors.append(CLASS_COLORS.get(label_str.lower(), 'gray'))

            ax.scatter(y_radar, x_radar, c=colors, s=20, alpha=0.7,
                      edgecolors='white', linewidths=0.5, label='Radar (by class)')
        elif seg_labels is not None:
            # Prediction: Color by segmentation (moving vs static)
            colors = [SEG_COLORS.get(int(label), 'gray') for label in seg_labels]
            ax.scatter(y_radar, x_radar, c=colors, s=20, alpha=0.7,
                      edgecolors='white', linewidths=0.5,
                      label='Radar (moving=orange, static=gray)')
        else:
            # Fallback: uniform color
            ax.scatter(y_radar, x_radar, c='blue', s=10, alpha=0.5)

    # Draw 3D boxes as 2D rectangles (BEV projection)
    draw_boxes_on_bev(ax, boxes, ids, color='lime' if not is_gt else 'yellow')

    # Formatting
    ax.set_xlim(side_range)
    ax.set_ylim(fwd_range)
    ax.set_xlabel('Y (m) - Left/Right', fontsize=10)
    ax.set_ylabel('X (m) - Forward', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')


def draw_boxes_on_bev(ax, boxes, ids, color='yellow'):
    """
    Draw 2D bounding boxes on BEV plot.

    Args:
        ax: matplotlib axis
        boxes: [K, 7] boxes (x, y, z, l, w, h, yaw)
        ids: [K] track IDs
        color: str - box edge color
    """
    if boxes is None or len(boxes) == 0:
        return

    # Handle None ids (no matches case)
    if ids is None:
        ids = list(range(len(boxes)))

    # Convert to numpy if tensor (and move to CPU if needed)
    if hasattr(boxes, 'cpu'):
        boxes = boxes.cpu().numpy()
    if hasattr(ids, 'cpu'):
        ids = ids.cpu().numpy()

    for i, box in enumerate(boxes):
        x, y, z, l, w, h, yaw = box

        # Compute box corners (simplified 2D rectangle)
        # For proper rotation, would need to compute rotated corners
        # Here we use axis-aligned approximation for simplicity
        rect = patches.Rectangle(
            (y - w/2, x - l/2),  # (x, y) in BEV is (y, x) in world
            w, l,
            linewidth=2,
            edgecolor=color,
            facecolor='none',
            alpha=0.8
        )
        ax.add_patch(rect)

        # Add track ID label
        track_id = ids[i] if i < len(ids) else i
        ax.text(y, x, f'{int(track_id)}',
               color='yellow', fontsize=10, fontweight='bold',
               ha='center', va='center',
               bbox=dict(boxstyle='circle,pad=0.1', facecolor='black',
                        alpha=0.7, edgecolor=color, linewidth=1.5))


def plot_3panel_evaluation(frame_number, pc_lidar, pc_radar,
                           gt_data, pred_data, kitti_locations, save_path):
    """
    Create 3-panel evaluation visualization:
    Left: GT BEV | Middle: RGB Projection | Right: Prediction BEV

    Args:
        frame_number: str - Frame number (e.g., '00142')
        pc_lidar: [N, 3] lidar points
        pc_radar: [M, 3] radar points
        gt_data: dict with keys:
            - 'boxes': [K_gt, 7] GT boxes
            - 'ids': [K_gt] GT track IDs
            - 'seg_labels': [M] GT segmentation labels for radar points
            - 'class_labels': [M] GT class labels for radar points (optional)
        pred_data: dict with keys:
            - 'boxes': [K_pred, 7] predicted boxes
            - 'ids': [K_pred] predicted track IDs
            - 'seg_labels': [M] predicted segmentation labels
        kitti_locations: VodTrackLocations object
        save_path: str - Path to save figure

    Returns:
        None - Saves figure to save_path
    """
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(24, 8))

    # Panel 1 (Left): GT BEV
    ax1 = fig.add_subplot(131)
    plot_bev_with_segmentation(
        ax1,
        pc_lidar=pc_lidar,
        pc_radar=pc_radar,
        seg_labels=gt_data.get('seg_labels'),
        class_labels=gt_data.get('class_labels'),
        boxes=gt_data.get('boxes'),
        ids=gt_data.get('ids'),
        title=f'Ground Truth BEV\n(Frame {frame_number})',
        is_gt=True
    )

    # Panel 2 (Middle): RGB Projection with GT boxes (RED) and Predicted boxes (CYAN)
    ax2 = fig.add_subplot(132)
    try:
        rgb_image, transforms = load_rgb_for_frame(frame_number, kitti_locations)

        gt_boxes = gt_data.get('boxes')
        gt_ids = gt_data.get('ids')
        pred_boxes = pred_data.get('boxes')
        pred_ids = pred_data.get('ids')

        plot_rgb_projection_simple(
            rgb_image, transforms,
            pc_radar=pc_radar,
            pc_lidar=pc_lidar,
            boxes=gt_boxes,  # GT boxes (first set)
            track_ids=gt_ids,  # GT track IDs
            ax=ax2,
            box_color='red',  # RED for GT boxes and IDs
            boxes2=pred_boxes,  # Predicted boxes (second set)
            track_ids2=pred_ids,  # Predicted track IDs
            box_color2='cyan'  # CYAN for predicted boxes and IDs
        )
        ax2.set_title(f'RGB Camera Projection\n(Frame {frame_number})',
                     fontsize=12, fontweight='bold')
    except Exception as e:
        print(f"Warning: Could not generate RGB projection: {e}")
        ax2.text(0.5, 0.5, 'RGB Projection Failed',
                ha='center', va='center', fontsize=14)
        ax2.axis('off')

    # Panel 3 (Right): Prediction BEV
    ax3 = fig.add_subplot(133)
    plot_bev_with_segmentation(
        ax3,
        pc_lidar=pc_lidar,
        pc_radar=pc_radar,
        seg_labels=pred_data.get('seg_labels'),
        class_labels=None,  # Predictions don't have class labels, use segmentation
        boxes=pred_data.get('boxes'),
        ids=pred_data.get('ids'),  # Already remapped by MOT metrics
        title=f'Prediction BEV\n(Frame {frame_number})',
        is_gt=False
    )

    # Add overall title
    fig.suptitle(f'Tracking Evaluation - Frame {frame_number}',
                fontsize=16, fontweight='bold', y=0.98)

    # Add legend for class colors
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Car/Van/Truck',
              markerfacecolor='red', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Pedestrian',
              markerfacecolor='green', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Cyclist/Rider',
              markerfacecolor='blue', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Moving (Motion)',
              markerfacecolor='orange', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Inside Box (Inferred)',
              markerfacecolor='yellow', markersize=8),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
              ncol=5, fontsize=10, framealpha=0.9,
              bbox_to_anchor=(0.5, -0.02))

    # Save figure
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.close(fig)


def plot_simple_bev_comparison(pc_lidar, pc_radar, gt_boxes, gt_ids,
                               pred_boxes, pred_ids, save_path):
    """
    Simplified 2-panel BEV comparison (no RGB, no segmentation).

    Args:
        pc_lidar: [N, 3] lidar points
        pc_radar: [M, 3] radar points
        gt_boxes: [K_gt, 7] GT boxes
        gt_ids: [K_gt] GT IDs
        pred_boxes: [K_pred, 7] predicted boxes
        pred_ids: [K_pred] predicted IDs
        save_path: str

    Returns:
        None - Saves figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Left: GT
    plot_bev_with_segmentation(
        ax1, pc_lidar, pc_radar,
        seg_labels=None, class_labels=None,
        boxes=gt_boxes, ids=gt_ids,
        title='Ground Truth', is_gt=True
    )

    # Right: Prediction
    plot_bev_with_segmentation(
        ax2, pc_lidar, pc_radar,
        seg_labels=None, class_labels=None,
        boxes=pred_boxes, ids=pred_ids,
        title='Prediction', is_gt=False
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.close(fig)
