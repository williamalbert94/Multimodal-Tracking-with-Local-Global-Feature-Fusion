"""
Simple Tracker for Multi-Object Tracking
Combines IoU-based matching with Re-ID embeddings for robust tracking.

Reference:
- SORT: Simple Online and Realtime Tracking (Bewley et al., 2016)
- DeepSORT: Simple Online and Realtime Tracking with Deep Association Metric (Wojke et al., 2017)
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from .kalman_filter_simple import KalmanFilter


class KalmanBoxTracker:
    """
    Kalman Filter for tracking a single 3D bounding box.
    State: [x, y, z, l, w, h, yaw, vx, vy, vz]
    """
    count = 0

    def __init__(self, bbox, embedding=None):
        """
        Initialize tracker with a bounding box.

        Args:
            bbox: [7] array (x, y, z, l, w, h, yaw)
            embedding: [D] Re-ID embedding (optional)
        """
        # Initialize Kalman filter (10D state: position + velocity)
        self.kf = KalmanFilter(dim_x=10, dim_z=7)

        # State transition matrix (constant velocity model)
        self.kf.F = np.eye(10)
        self.kf.F[0, 7] = 1.0  # x += vx
        self.kf.F[1, 8] = 1.0  # y += vy
        self.kf.F[2, 9] = 1.0  # z += vz

        # Measurement matrix (observe position, size, yaw)
        self.kf.H = np.zeros((7, 10))
        self.kf.H[:7, :7] = np.eye(7)

        # Measurement noise
        self.kf.R[0:3, 0:3] *= 0.1  # Position measurement noise (low)
        self.kf.R[3:6, 3:6] *= 1.0  # Size measurement noise (medium)
        self.kf.R[6, 6] *= 0.5      # Yaw measurement noise

        # Process noise
        self.kf.Q[0:3, 0:3] *= 0.01  # Position process noise
        self.kf.Q[3:6, 3:6] *= 0.1   # Size process noise
        self.kf.Q[6, 6] *= 0.1       # Yaw process noise
        self.kf.Q[7:10, 7:10] *= 1.0 # Velocity process noise

        # Covariance matrix
        self.kf.P[0:3, 0:3] *= 1.0   # Initial position uncertainty
        self.kf.P[3:6, 3:6] *= 10.0  # Initial size uncertainty
        self.kf.P[6, 6] *= 5.0       # Initial yaw uncertainty
        self.kf.P[7:10, 7:10] *= 100.0  # Initial velocity uncertainty (high)

        # Initialize state
        self.kf.x[:7] = bbox.reshape((7, 1))

        # Track management
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        # Re-ID embedding
        self.embedding = embedding
        self.embedding_history = [embedding] if embedding is not None else []

    def update(self, bbox, embedding=None):
        """
        Update tracker with new measurement.

        Args:
            bbox: [7] array (x, y, z, l, w, h, yaw)
            embedding: [D] Re-ID embedding (optional)
        """
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(bbox)

        # Update embedding (exponential moving average)
        if embedding is not None:
            if self.embedding is None:
                self.embedding = embedding
            else:
                alpha = 0.9  # EMA coefficient
                self.embedding = alpha * self.embedding + (1 - alpha) * embedding

            self.embedding_history.append(embedding)
            # Keep only last 10 embeddings
            if len(self.embedding_history) > 10:
                self.embedding_history = self.embedding_history[-10:]

    def predict(self):
        """
        Predict next state.

        Returns:
            bbox: [7] predicted box
        """
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.get_state()

    def get_state(self):
        """
        Get current state as bounding box.

        Returns:
            bbox: [7] array (x, y, z, l, w, h, yaw)
        """
        return self.kf.x[:7].flatten()


class SimpleTracker:
    """
    Simple Multi-Object Tracker using Kalman filter + Re-ID embeddings.

    Combines:
    - Motion model (Kalman filter)
    - Appearance model (Re-ID embeddings)
    - Hungarian algorithm for optimal matching
    """

    def __init__(self, max_age=3, min_hits=3, iou_threshold=0.3, embedding_threshold=0.5):
        """
        Initialize tracker.

        Args:
            max_age: Maximum frames to keep alive without updates
            min_hits: Minimum hits to consider track confirmed
            iou_threshold: IoU threshold for matching
            embedding_threshold: Cosine similarity threshold for Re-ID
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.embedding_threshold = embedding_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, boxes, embeddings=None):
        """
        Update tracker with new detections.

        Args:
            boxes: [M, 7] detected boxes (x, y, z, l, w, h, yaw)
            embeddings: [M, D] Re-ID embeddings (optional)

        Returns:
            tracks: List of (bbox, track_id) for confirmed tracks
        """
        self.frame_count += 1

        # Convert to numpy
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        # Predict next state for existing tracks
        trks = np.zeros((len(self.trackers), 7))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trks[t] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)

        # Remove tracks with invalid predictions
        for t in reversed(to_del):
            self.trackers.pop(t)
        trks = np.delete(trks, to_del, axis=0)

        # Associate detections to tracks
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
            boxes, trks, embeddings
        )

        # Update matched tracks
        for m in matched:
            det_idx, trk_idx = m[0], m[1]
            emb = embeddings[det_idx] if embeddings is not None else None
            self.trackers[trk_idx].update(boxes[det_idx], emb)

        # Create new tracks for unmatched detections
        for i in unmatched_dets:
            emb = embeddings[i] if embeddings is not None else None
            trk = KalmanBoxTracker(boxes[i], emb)
            self.trackers.append(trk)

        # Return confirmed tracks
        ret = []
        for i, trk in enumerate(self.trackers):
            # Only return tracks that have been updated enough times
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append((trk.get_state(), trk.id))

        # Remove dead tracks
        self.trackers = [t for t in self.trackers if t.time_since_update < self.max_age]

        return ret

    def _associate_detections_to_trackers(self, detections, trackers, embeddings=None):
        """
        Associate detections to tracked objects using Hungarian algorithm.

        Args:
            detections: [M, 7] detected boxes
            trackers: [N, 7] predicted boxes from tracks
            embeddings: [M, D] Re-ID embeddings for detections

        Returns:
            matches: [(det_idx, trk_idx), ...]
            unmatched_detections: [det_idx, ...]
            unmatched_trackers: [trk_idx, ...]
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty(0, dtype=int)

        if len(detections) == 0:
            return np.empty((0, 2), dtype=int), np.empty(0, dtype=int), np.arange(len(trackers))

        # Compute cost matrix: (1 - IoU) + λ * (1 - cosine_similarity)
        iou_matrix = self._compute_iou_matrix(detections, trackers)

        if embeddings is not None and len(self.trackers) > 0:
            # Compute appearance similarity
            appearance_matrix = np.zeros((len(detections), len(trackers)))
            for i in range(len(detections)):
                for j in range(len(trackers)):
                    if self.trackers[j].embedding is not None:
                        # Cosine similarity
                        sim = np.dot(embeddings[i], self.trackers[j].embedding)
                        sim /= (np.linalg.norm(embeddings[i]) * np.linalg.norm(self.trackers[j].embedding) + 1e-8)
                        appearance_matrix[i, j] = sim

            # Combined cost: 0.7 * (1 - IoU) + 0.3 * (1 - appearance)
            cost_matrix = 0.7 * (1 - iou_matrix) + 0.3 * (1 - appearance_matrix)
        else:
            # Only IoU-based matching
            cost_matrix = 1 - iou_matrix

        # Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Filter matches with high cost
        matches = []
        for row, col in zip(row_indices, col_indices):
            if iou_matrix[row, col] >= self.iou_threshold:
                matches.append([row, col])

        # Ensure matches is always 2D array (even if empty)
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.array(matches, dtype=int)

        unmatched_detections = [d for d in range(len(detections)) if d not in matches[:, 0]]
        unmatched_trackers = [t for t in range(len(trackers)) if t not in matches[:, 1]]

        return matches, unmatched_detections, unmatched_trackers

    def _compute_iou_matrix(self, boxes1, boxes2):
        """
        Compute IoU matrix between two sets of boxes (2D Bird's Eye View).

        Args:
            boxes1: [M, 7] boxes
            boxes2: [N, 7] boxes

        Returns:
            iou_matrix: [M, N]
        """
        M, N = len(boxes1), len(boxes2)
        iou_matrix = np.zeros((M, N))

        for i in range(M):
            for j in range(N):
                iou_matrix[i, j] = self._compute_iou_2d(boxes1[i], boxes2[j])

        return iou_matrix

    def _compute_iou_2d(self, box1, box2):
        """
        Compute 2D IoU on Bird's Eye View.

        Args:
            box1: [7] (x, y, z, l, w, h, yaw)
            box2: [7]

        Returns:
            iou: float
        """
        # Extract x, y, l, w (ignore z, h for BEV)
        x1, y1, l1, w1 = box1[0], box1[1], box1[3], box1[4]
        x2, y2, l2, w2 = box2[0], box2[1], box2[3], box2[4]

        # Axis-aligned bounding boxes (simplified, ignores yaw)
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

    def reset(self):
        """Reset tracker (clear all tracks)."""
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0
