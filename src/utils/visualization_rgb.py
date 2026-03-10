"""
RGB Projection Visualization
Project 3D radar/lidar points and boxes onto RGB camera images.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os

# Add paths for external modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from external.vod.frame.data_loader import FrameDataLoader
from external.vod.frame.transformations import FrameTransformMatrix, project_pcl_to_image
from external.kitti.box import Box3D
from utils.instance_assignment import assign_points_to_boxes


def load_rgb_for_frame(frame_number, kitti_locations):
    """
    Load RGB image for a given frame.

    Args:
        frame_number: str - Frame number (e.g., '00142')
        kitti_locations: VodTrackLocations object

    Returns:
        rgb_image: numpy array - RGB image
        transforms: FrameTransformMatrix object
    """
    frame_loader = FrameDataLoader(kitti_locations, frame_number)
    rgb_image = frame_loader.image
    transforms = FrameTransformMatrix(frame_loader)
    return rgb_image, transforms


def project_box_to_image(box_arr, transforms):
    """
    Project 3D box corners to 2D image coordinates.

    Args:
        box_arr: [7] array - box (x, y, z, l, w, h, yaw) in radar frame
        transforms: FrameTransformMatrix object

    Returns:
        corners_2d: [8, 2] array - 2D corner coordinates (u, v)
        valid: bool - whether all corners are in front of camera
    """
    # Convert box array to Box3D object and get 3D corners
    box = Box3D.array2bbox(box_arr)
    corners_3d = Box3D.box2corners3d_camcoord(box)  # [8, 3] in camera frame

    # Transform from radar frame to camera frame
    corners_homo = np.hstack([corners_3d, np.ones((8, 1))])  # [8, 4]
    corners_cam = corners_homo @ transforms.t_camera_radar.T  # [8, 4]

    # Check if all corners are in front of camera (positive Z)
    if np.any(corners_cam[:, 2] <= 0):
        return None, False

    # Project to 2D using camera intrinsics
    proj_matrix = transforms.camera_projection_matrix  # [3, 4]
    uvs_homo = corners_cam[:, :3] @ proj_matrix.T  # [8, 3]
    uvs = uvs_homo[:, :2] / (uvs_homo[:, 2:3] + 1e-6)  # [8, 2]

    return uvs, True


def plot_rgb_projection(frame_number, pc_radar, pc_lidar, boxes, track_ids,
                       kitti_locations, save_path, class_labels=None):
    """
    Plot RGB image with projected radar/lidar points and 3D boxes with track IDs.

    Args:
        frame_number: str - Frame number (e.g., '00142')
        pc_radar: [N, 3] numpy array - Radar points in radar frame
        pc_lidar: [M, 3] numpy array - Lidar points in radar frame (optional)
        boxes: [K, 7] numpy array - 3D boxes (x, y, z, l, w, h, yaw) in radar frame
        track_ids: [K] numpy array - Track IDs for each box
        kitti_locations: VodTrackLocations object
        save_path: str - Path to save figure
        class_labels: [K] list - Class labels for each box (optional)

    Returns:
        None - Saves figure to save_path
    """
    # Load RGB image and transforms
    rgb_image, transforms = load_rgb_for_frame(frame_number, kitti_locations)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 10), dpi=120)
    ax.imshow(rgb_image)

    # Project and plot radar points
    if pc_radar is not None and len(pc_radar) > 0:
        try:
            uvs_radar, depths_radar = project_pcl_to_image(
                pc_radar,
                transforms.t_camera_radar,
                transforms.camera_projection_matrix,
                rgb_image.shape
            )
            if len(uvs_radar) > 0:
                # Color by negative depth (closer = warmer colors)
                # Variable size based on depth (closer = larger)
                ax.scatter(uvs_radar[:, 0], uvs_radar[:, 1],
                          c=-depths_radar, cmap='jet',
                          s=(70 / depths_radar) ** 2, alpha=0.8,
                          marker='o', edgecolors='white', linewidths=0.5,
                          label='Radar', vmin=-50, vmax=0)
        except Exception as e:
            print(f"Warning: Could not project radar points: {e}")

    # Project and plot lidar points (smaller, more transparent)
    if pc_lidar is not None and len(pc_lidar) > 0:
        try:
            uvs_lidar, depths_lidar = project_pcl_to_image(
                pc_lidar,
                transforms.t_camera_lidar,
                transforms.camera_projection_matrix,
                rgb_image.shape
            )
            if len(uvs_lidar) > 0:
                ax.scatter(uvs_lidar[:, 0], uvs_lidar[:, 1],
                          c=-depths_lidar, cmap='jet', s=1, alpha=0.4,
                          marker='.', label='Lidar')
        except Exception as e:
            print(f"Warning: Could not project lidar points: {e}")

    # Project and draw 3D boxes
    if boxes is not None and len(boxes) > 0:
        for i, box_arr in enumerate(boxes):
            try:
                corners_2d, valid = project_box_to_image(box_arr, transforms)

                if not valid or corners_2d is None:
                    continue

                # Define box edges (wireframe)
                edges = [
                    [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
                    [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
                    [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
                ]

                # Draw edges
                box_color = 'lime' if class_labels is None else 'lime'
                for edge in edges:
                    p1, p2 = corners_2d[edge[0]], corners_2d[edge[1]]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                           color=box_color, linewidth=2, alpha=0.8)

                # Draw track ID label at bottom-front corner
                track_id = track_ids[i] if i < len(track_ids) else i
                label_pos = corners_2d[0]  # Front-bottom-left corner

                # Add background box for readability
                ax.text(label_pos[0], label_pos[1], f'ID:{int(track_id)}',
                       color='yellow', fontsize=14, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black',
                                alpha=0.7, edgecolor='yellow', linewidth=2),
                       ha='left', va='bottom')

            except Exception as e:
                print(f"Warning: Could not project box {i}: {e}")
                continue

    # Add legend if points were plotted
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='upper right', fontsize=12, framealpha=0.8)

    # Add title
    ax.set_title(f'RGB Projection - Frame {frame_number}',
                fontsize=16, fontweight='bold', pad=10)
    ax.axis('off')

    # Save figure
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.close(fig)


def draw_2d_boxes_around_points(ax, uvs_points, point_track_ids, track_ids, box_color='lime'):
    """
    Draw 2D bounding boxes around projected points that belong to the same track ID.

    Args:
        ax: matplotlib axis
        uvs_points: [N, 2] projected 2D coordinates of points
        point_track_ids: [N] track ID for each point (-1 for background)
        track_ids: [K] list of valid track IDs to draw
        box_color: str - color for boxes
    """
    import matplotlib.patches as patches

    for track_id in track_ids:
        # Find points belonging to this track
        mask = point_track_ids == track_id
        if not np.any(mask):
            continue

        # Get 2D coordinates of points for this track
        track_points_2d = uvs_points[mask]

        if len(track_points_2d) < 2:  # Need at least 2 points for a box
            continue

        # Compute 2D bounding box (min/max in x and y)
        x_min = np.min(track_points_2d[:, 0])
        x_max = np.max(track_points_2d[:, 0])
        y_min = np.min(track_points_2d[:, 1])
        y_max = np.max(track_points_2d[:, 1])

        # Add some padding
        padding = 10  # pixels
        x_min -= padding
        x_max += padding
        y_min -= padding
        y_max += padding

        width = x_max - x_min
        height = y_max - y_min

        # Draw rectangle
        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=2, edgecolor=box_color, facecolor='none',
            alpha=0.8, linestyle='-'
        )
        ax.add_patch(rect)

        # Add track ID label at top-left corner
        ax.text(
            x_min, y_min - 5,  # Slightly above the box
            f'ID:{int(track_id)}',
            color=box_color, fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black',
                     alpha=0.8, edgecolor=box_color, linewidth=2),
            ha='left', va='bottom'
        )


def draw_boxes_on_rgb(ax, boxes, track_ids, transforms, pc_radar,
                      uvs_radar_final, radar_valid_indices, box_color, id_offset=(0, 0)):
    """
    Helper function to draw one set of boxes on RGB image.
    Draws both 3D boxes projected to 2D and 2D boxes around radar points.

    Args:
        ax: matplotlib axis
        boxes: [K, 7] 3D boxes
        track_ids: [K] track IDs
        transforms: FrameTransformMatrix object
        pc_radar: [N, 3] original radar points (3D)
        uvs_radar_final: [M, 2] projected radar points (2D)
        radar_valid_indices: indices of valid projected points
        box_color: color for drawing boxes
        id_offset: (x, y) pixel offset for ID position to avoid overlap
    """
    if boxes is None or len(boxes) == 0:
        return

    # Draw 3D boxes projected to 2D
    for i, box_arr in enumerate(boxes):
        try:
            corners_2d, valid = project_box_to_image(box_arr, transforms)
            if not valid or corners_2d is None:
                continue

            edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],
                    [0,4],[1,5],[2,6],[3,7]]

            for edge in edges:
                p1, p2 = corners_2d[edge[0]], corners_2d[edge[1]]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                       color=box_color, linewidth=2, alpha=0.9)

            track_id = track_ids[i] if i < len(track_ids) else i
            # Apply offset to avoid ID overlap between GT and predictions
            ax.text(corners_2d[0, 0] + id_offset[0], corners_2d[0, 1] + id_offset[1],
                   f'{int(track_id)}',
                   color=box_color, fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black',
                            alpha=0.8, edgecolor=box_color, linewidth=2))
        except:
            continue

    # Draw 2D bounding boxes around radar points grouped by track ID
    if (uvs_radar_final is not None and len(uvs_radar_final) > 0 and
        radar_valid_indices is not None and pc_radar is not None):
        try:
            # Convert boxes to numpy if needed
            boxes_np = boxes.cpu().numpy() if hasattr(boxes, 'cpu') else np.array(boxes)
            track_ids_np = track_ids.cpu().numpy() if hasattr(track_ids, 'cpu') else np.array(track_ids)

            # Assign each radar point to a track ID
            all_point_track_ids = assign_points_to_boxes(
                points=pc_radar,
                boxes=boxes_np,
                track_ids=track_ids_np
            )

            # Filter to only valid projected points
            point_track_ids = all_point_track_ids[radar_valid_indices]

            # Draw 2D boxes around projected points for each track
            draw_2d_boxes_around_points(
                ax=ax,
                uvs_points=uvs_radar_final,
                point_track_ids=point_track_ids,
                track_ids=track_ids_np,
                box_color=box_color
            )
        except Exception as e:
            print(f"Warning: Could not draw 2D boxes around radar points: {e}")


def plot_rgb_projection_simple(rgb_image, transforms, pc_radar, pc_lidar,
                               boxes, track_ids, ax=None, box_color='cyan',
                               boxes2=None, track_ids2=None, box_color2='cyan'):
    """
    Simplified version for use in multi-panel figures.
    Now includes 2D boxes around projected radar points grouped by track ID.
    Supports two sets of boxes with different colors (e.g., GT and predictions).

    Args:
        rgb_image: numpy array - RGB image
        transforms: FrameTransformMatrix object
        pc_radar: [N, 3] radar points in 3D
        pc_lidar: [M, 3] lidar points in 3D
        boxes: [K, 7] 3D boxes (x, y, z, l, w, h, yaw) - first set
        track_ids: [K] track IDs for each box - first set
        ax: matplotlib axis (if None, creates new figure)
        box_color: str - color for first set of boxes and IDs (default 'cyan')
        boxes2: [K2, 7] 3D boxes - second set (optional)
        track_ids2: [K2] track IDs for second set (optional)
        box_color2: str - color for second set of boxes and IDs (default 'cyan')

    Returns:
        ax: matplotlib axis with plot
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    ax.imshow(rgb_image)

    # Debug: Check input data
    print(f"[DEBUG RGB PROJECTION] pc_radar: {pc_radar.shape if pc_radar is not None else None}")
    print(f"[DEBUG RGB PROJECTION] pc_lidar: {pc_lidar.shape if pc_lidar is not None else None}")
    print(f"[DEBUG RGB PROJECTION] boxes: {boxes.shape if boxes is not None and hasattr(boxes, 'shape') else len(boxes) if boxes is not None else None}")
    print(f"[DEBUG RGB PROJECTION] rgb_image: {rgb_image.shape}")

    # Variables to store projected points and indices for later use
    uvs_radar_final = None
    radar_valid_indices = None

    # Project radar points (manual projection to keep track of valid indices)
    if pc_radar is not None and len(pc_radar) > 0:
        try:
            # Do projection manually to keep track of which points are valid
            from external.vod.frame.transformations import homogeneous_transformation, project_3d_to_2d, canvas_crop

            # Transform to camera frame
            point_homo = np.hstack((pc_radar[:, :3],
                                   np.ones((pc_radar.shape[0], 1), dtype=np.float32)))
            radar_camera_frame = homogeneous_transformation(point_homo,
                                                           transform=transforms.t_camera_radar)
            point_depth = radar_camera_frame[:, 2]

            # Project to 2D
            uvs_radar = project_3d_to_2d(points=radar_camera_frame,
                                        projection_matrix=transforms.camera_projection_matrix)

            # Filter points outside image (canvas_crop returns boolean mask)
            valid_mask = canvas_crop(points=uvs_radar,
                                    image_size=rgb_image.shape,
                                    points_depth=point_depth)

            # Get valid indices and filtered points
            radar_valid_indices = np.where(valid_mask)[0]
            uvs_radar = uvs_radar[valid_mask]
            depths_radar = point_depth[valid_mask]

            print(f"[DEBUG RGB PROJECTION] Radar: {len(pc_radar)} input points -> {len(uvs_radar)} projected points")
            if len(uvs_radar) > 0:
                # Store for later use (2D box drawing)
                uvs_radar_final = uvs_radar

                # Variable size based on depth (closer = larger)
                ax.scatter(uvs_radar[:, 0], uvs_radar[:, 1],
                          c=-depths_radar, cmap='jet',
                          s=(70 / depths_radar) ** 2, alpha=0.8,
                          vmin=-50, vmax=0)
                print(f"[DEBUG RGB PROJECTION] Radar scatter plot added successfully")
            else:
                print(f"[DEBUG RGB PROJECTION] WARNING: No radar points after projection (all filtered out)")
        except Exception as e:
            print(f"Warning: Could not project radar points in simple plot: {e}")
            import traceback
            traceback.print_exc()

    # Project lidar points
    if pc_lidar is not None and len(pc_lidar) > 0:
        try:
            uvs_lidar, depths_lidar = project_pcl_to_image(
                pc_lidar, transforms.t_camera_lidar,
                transforms.camera_projection_matrix, rgb_image.shape
            )
            print(f"[DEBUG RGB PROJECTION] Lidar: {len(pc_lidar)} input points -> {len(uvs_lidar)} projected points")
            if len(uvs_lidar) > 0:
                ax.scatter(uvs_lidar[:, 0], uvs_lidar[:, 1],
                          c=-depths_lidar, cmap='jet', s=1, alpha=0.4)
                print(f"[DEBUG RGB PROJECTION] Lidar scatter plot added successfully")
            else:
                print(f"[DEBUG RGB PROJECTION] WARNING: No lidar points after projection (all filtered out)")
        except Exception as e:
            print(f"Warning: Could not project lidar points in simple plot: {e}")
            import traceback
            traceback.print_exc()

    # Draw first set of boxes (GT boxes - no offset)
    draw_boxes_on_rgb(
        ax=ax,
        boxes=boxes,
        track_ids=track_ids,
        transforms=transforms,
        pc_radar=pc_radar,
        uvs_radar_final=uvs_radar_final,
        radar_valid_indices=radar_valid_indices,
        box_color=box_color,
        id_offset=(0, 0)  # No offset for first set
    )

    # Draw second set of boxes (Predicted boxes - offset to avoid ID overlap)
    if boxes2 is not None and track_ids2 is not None:
        draw_boxes_on_rgb(
            ax=ax,
            boxes=boxes2,
            track_ids=track_ids2,
            transforms=transforms,
            pc_radar=pc_radar,
            uvs_radar_final=uvs_radar_final,
            radar_valid_indices=radar_valid_indices,
            box_color=box_color2,
            id_offset=(40, 0)  # Shift right by 40 pixels to avoid overlap with GT IDs
        )

    ax.axis('off')
    return ax
