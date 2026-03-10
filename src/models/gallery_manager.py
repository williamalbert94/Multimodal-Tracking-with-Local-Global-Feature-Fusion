"""
Gallery Manager for Re-ID Tracking
Manages embeddings buffer per sequence with efficient disk-based storage.

Features:
  - Pre-compute embeddings offline per sequence
  - Efficient query/gallery sampling for training
  - Re-ID metrics: Precision@k, Recall@k, mAP
  - Analysis logging for debugging
"""

import os
import pickle
import numpy as np
import torch
from collections import defaultdict
from pathlib import Path


class GalleryManager:
    """
    Manages Re-ID embeddings gallery with disk-based storage.

    Structure per sequence:
        {
            'sequence_id': str,
            'tracks': {
                track_id_1: {
                    'embeddings': np.array [T, D],     # T timesteps, D embedding dim
                    'boxes': np.array [T, 7],           # x, y, z, l, w, h, yaw
                    'timestamps': np.array [T],         # frame indices
                    'class': str,                       # 'car', 'pedestrian', etc.
                    'points': List[np.array],           # segmented points (optional)
                    'ego_motion': np.array [T, 3],      # ego motion (dx, dy, dz) per frame
                    'velocities': np.array [T, 3],      # object velocities (vx, vy, vz)
                    'motion_map': np.array [T, M],      # motion representation features
                },
                track_id_2: {...},
                ...
            },
            'metadata': {
                'num_frames': int,
                'num_tracks': int,
                'embedding_dim': int,
                'motion_enabled': bool,              # Whether motion features are included
            }
        }
    """

    def __init__(self, gallery_dir='checkpoints/reid_gallery', device='cuda'):
        """
        Initialize Gallery Manager.

        Args:
            gallery_dir: Directory to store pre-computed galleries
            device: Device for tensor operations
        """
        self.gallery_dir = Path(gallery_dir)
        self.gallery_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        # In-memory cache (LRU style - keep only recent sequences)
        self.cache = {}
        self.max_cache_size = 5  # Keep max 5 sequences in RAM

        # Statistics
        self.stats = {
            'queries_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
        }

    def save_sequence_gallery(self, sequence_id, tracks_data):
        """
        Save pre-computed gallery for a sequence to disk.

        Args:
            sequence_id: str, unique sequence identifier
            tracks_data: dict with structure described in class docstring
        """
        file_path = self.gallery_dir / f"{sequence_id}.pkl"

        # Add metadata
        if 'metadata' not in tracks_data:
            tracks_data['metadata'] = {}

        tracks_data['metadata']['num_tracks'] = len(tracks_data.get('tracks', {}))

        # Get embedding dim from first track
        if tracks_data.get('tracks'):
            first_track = list(tracks_data['tracks'].values())[0]
            if len(first_track['embeddings']) > 0:
                tracks_data['metadata']['embedding_dim'] = first_track['embeddings'].shape[1]

        # Save to disk
        with open(file_path, 'wb') as f:
            pickle.dump(tracks_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[GalleryManager] Saved gallery: {sequence_id} → {file_path}")
        print(f"                 Tracks: {tracks_data['metadata']['num_tracks']}, "
              f"Frames: {tracks_data['metadata'].get('num_frames', 'N/A')}")

    def load_sequence_gallery(self, sequence_id, use_cache=True):
        """
        Load pre-computed gallery for a sequence.

        Args:
            sequence_id: str, unique sequence identifier
            use_cache: bool, whether to use in-memory cache

        Returns:
            tracks_data: dict with gallery data (or None if not found)
        """
        # Check cache first
        if use_cache and sequence_id in self.cache:
            self.stats['cache_hits'] += 1
            return self.cache[sequence_id]

        # Load from disk
        file_path = self.gallery_dir / f"{sequence_id}.pkl"

        if not file_path.exists():
            print(f"[GalleryManager] WARNING: Gallery not found for sequence {sequence_id}")
            return None

        with open(file_path, 'rb') as f:
            tracks_data = pickle.load(f)

        # Update cache (LRU eviction)
        if use_cache:
            self.cache[sequence_id] = tracks_data

            # Evict oldest if cache is full
            if len(self.cache) > self.max_cache_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]

        self.stats['cache_misses'] += 1
        return tracks_data

    def sample_query_gallery_pairs(self, sequence_id, num_queries=16, gallery_size=50,
                                   min_temporal_gap=5, max_temporal_gap=30):
        """
        Sample query-gallery pairs for training Re-ID.

        Strategy:
          - Query: Random track at random timestamp t
          - Gallery: All other tracks + same track at different timestamps
          - Positive: Same track_id (temporal gap between t and t')
          - Negative: Different track_ids

        Args:
            sequence_id: str, sequence to sample from
            num_queries: int, number of query samples
            gallery_size: int, max gallery size per query
            min_temporal_gap: int, minimum frames between query and positive
            max_temporal_gap: int, maximum frames between query and positive

        Returns:
            query_data: List[dict], each with:
                - 'embedding': [D]
                - 'track_id': int
                - 'timestamp': int
                - 'box': [7]
                - 'class': str
            gallery_data: List[dict], similar structure
            labels: List[int], 1 if query matches gallery[i], 0 otherwise
        """
        tracks_data = self.load_sequence_gallery(sequence_id)

        if tracks_data is None or 'tracks' not in tracks_data:
            return [], [], []

        tracks = tracks_data['tracks']
        all_track_ids = list(tracks.keys())

        if len(all_track_ids) < 2:
            # Need at least 2 tracks for positive/negative pairs
            return [], [], []

        query_samples = []
        gallery_samples = []
        labels_list = []

        for _ in range(num_queries):
            # Sample random track as query
            query_track_id = np.random.choice(all_track_ids)
            query_track = tracks[query_track_id]

            # Sample random timestamp for query
            num_timestamps = len(query_track['embeddings'])
            if num_timestamps == 0:
                continue

            query_t = np.random.randint(0, num_timestamps)

            # Create query sample
            query_sample = {
                'embedding': query_track['embeddings'][query_t],
                'track_id': query_track_id,
                'timestamp': query_track['timestamps'][query_t],
                'box': query_track['boxes'][query_t],
                'class': query_track['class'],
            }

            # Build gallery
            gallery_batch = []
            labels_batch = []

            # Add positive samples (same track, different time)
            valid_positive_indices = [
                t for t in range(num_timestamps)
                if abs(t - query_t) >= min_temporal_gap and abs(t - query_t) <= max_temporal_gap
            ]

            if len(valid_positive_indices) > 0:
                # Add 1-3 positive samples
                num_positives = min(3, len(valid_positive_indices))
                positive_indices = np.random.choice(valid_positive_indices, size=num_positives, replace=False)

                for pos_t in positive_indices:
                    gallery_batch.append({
                        'embedding': query_track['embeddings'][pos_t],
                        'track_id': query_track_id,
                        'timestamp': query_track['timestamps'][pos_t],
                        'box': query_track['boxes'][pos_t],
                        'class': query_track['class'],
                    })
                    labels_batch.append(1)  # Positive match

            # Add negative samples (different tracks)
            other_track_ids = [tid for tid in all_track_ids if tid != query_track_id]
            num_negatives = min(gallery_size - len(gallery_batch), len(other_track_ids) * 3)

            for _ in range(num_negatives):
                neg_track_id = np.random.choice(other_track_ids)
                neg_track = tracks[neg_track_id]

                if len(neg_track['embeddings']) == 0:
                    continue

                neg_t = np.random.randint(0, len(neg_track['embeddings']))

                gallery_batch.append({
                    'embedding': neg_track['embeddings'][neg_t],
                    'track_id': neg_track_id,
                    'timestamp': neg_track['timestamps'][neg_t],
                    'box': neg_track['boxes'][neg_t],
                    'class': neg_track['class'],
                })
                labels_batch.append(0)  # Negative match

            # Add to overall batch
            query_samples.append(query_sample)
            gallery_samples.append(gallery_batch)
            labels_list.append(labels_batch)

        self.stats['queries_processed'] += len(query_samples)

        return query_samples, gallery_samples, labels_list

    def compute_reid_metrics(self, query_embeddings, gallery_embeddings, gallery_labels,
                            k_values=[1, 5, 10]):
        """
        Compute Re-ID metrics: Precision@k, Recall@k, mAP.

        Args:
            query_embeddings: [Q, D] query embeddings
            gallery_embeddings: [G, D] gallery embeddings
            gallery_labels: [G] binary labels (1 = match, 0 = no match)
            k_values: list of k values for Precision@k and Recall@k

        Returns:
            metrics: dict with precision@k, recall@k, mAP
        """
        if isinstance(query_embeddings, list):
            query_embeddings = np.array(query_embeddings)
        if isinstance(gallery_embeddings, list):
            gallery_embeddings = np.array(gallery_embeddings)
        if isinstance(gallery_labels, list):
            gallery_labels = np.array(gallery_labels)

        # Compute pairwise distances (cosine similarity → distance)
        # Normalize embeddings
        query_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
        gallery_norm = gallery_embeddings / (np.linalg.norm(gallery_embeddings, axis=1, keepdims=True) + 1e-8)

        # Similarity matrix [Q, G]
        similarity_matrix = query_norm @ gallery_norm.T

        # Distance matrix (1 - similarity for cosine)
        distance_matrix = 1 - similarity_matrix

        # Sort gallery by distance (ascending - closer is better)
        sorted_indices = np.argsort(distance_matrix, axis=1)  # [Q, G]

        # Compute metrics
        metrics = {}

        # Precision@k and Recall@k
        for k in k_values:
            precision_k = []
            recall_k = []

            for q_idx in range(len(query_embeddings)):
                top_k_indices = sorted_indices[q_idx, :k]
                top_k_labels = gallery_labels[top_k_indices]

                # Precision@k: fraction of top-k that are relevant
                num_relevant_k = np.sum(top_k_labels)
                precision_k.append(num_relevant_k / k)

                # Recall@k: fraction of all relevant retrieved in top-k
                total_relevant = np.sum(gallery_labels)
                recall_k.append(num_relevant_k / total_relevant if total_relevant > 0 else 0)

            metrics[f'precision@{k}'] = np.mean(precision_k)
            metrics[f'recall@{k}'] = np.mean(recall_k)

        # mAP (mean Average Precision)
        ap_list = []
        for q_idx in range(len(query_embeddings)):
            sorted_labels = gallery_labels[sorted_indices[q_idx]]

            # Compute Average Precision
            num_relevant = np.sum(sorted_labels)
            if num_relevant == 0:
                ap_list.append(0)
                continue

            precisions = []
            num_correct = 0
            for i, label in enumerate(sorted_labels):
                if label == 1:
                    num_correct += 1
                    precision_at_i = num_correct / (i + 1)
                    precisions.append(precision_at_i)

            ap = np.mean(precisions) if len(precisions) > 0 else 0
            ap_list.append(ap)

        metrics['mAP'] = np.mean(ap_list)

        return metrics

    def log_analysis(self, query_sample, gallery_samples, labels, distances):
        """
        Log detailed analysis of query-gallery matching for debugging.

        Args:
            query_sample: dict with query info
            gallery_samples: List[dict] with gallery info
            labels: List[int] ground truth labels
            distances: List[float] predicted distances
        """
        # Sort by distance
        sorted_indices = np.argsort(distances)

        print(f"\n[Gallery Analysis] Query: Track {query_sample['track_id']}, "
              f"Time {query_sample['timestamp']}, Class {query_sample['class']}")
        print("  Top-5 Gallery Matches:")

        for rank, idx in enumerate(sorted_indices[:5], 1):
            g_sample = gallery_samples[idx]
            label = "✓ MATCH" if labels[idx] == 1 else "✗ NO MATCH"

            print(f"    {rank}. Track {g_sample['track_id']}, Time {g_sample['timestamp']}, "
                  f"Dist: {distances[idx]:.3f} {label}")

    def get_statistics(self):
        """Get gallery statistics."""
        return {
            'queries_processed': self.stats['queries_processed'],
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'cache_hit_rate': self.stats['cache_hits'] / max(1, self.stats['cache_hits'] + self.stats['cache_misses']),
            'cached_sequences': len(self.cache),
        }

    def clear_cache(self):
        """Clear in-memory cache."""
        self.cache = {}
        print("[GalleryManager] Cache cleared")

    def list_available_sequences(self):
        """List all pre-computed sequence galleries."""
        gallery_files = sorted(self.gallery_dir.glob("*.pkl"))
        sequence_ids = [f.stem for f in gallery_files]
        return sequence_ids

    def get_sequence_info(self, sequence_id):
        """Get metadata for a sequence gallery."""
        tracks_data = self.load_sequence_gallery(sequence_id, use_cache=False)

        if tracks_data is None:
            return None

        info = {
            'sequence_id': sequence_id,
            'metadata': tracks_data.get('metadata', {}),
            'num_tracks': len(tracks_data.get('tracks', {})),
        }

        # Compute additional stats
        tracks = tracks_data.get('tracks', {})
        total_embeddings = sum(len(t['embeddings']) for t in tracks.values())

        info['total_embeddings'] = total_embeddings
        info['avg_embeddings_per_track'] = total_embeddings / max(1, len(tracks))

        return info


# ============================================
# MOTION REPRESENTATION UTILITIES
# ============================================

def compute_motion_representation(boxes, ego_motion=None, timestamps=None):
    """
    Compute motion representation features for a track.

    Args:
        boxes: np.array [T, 7] - boxes (x, y, z, l, w, h, yaw) over time
        ego_motion: np.array [T, 3] - ego motion (dx, dy, dz) per frame
        timestamps: np.array [T] - frame indices

    Returns:
        motion_map: np.array [T, M] - motion features where M includes:
            - velocities (3D): vx, vy, vz
            - acceleration (3D): ax, ay, az
            - ego_motion (3D): ego_dx, ego_dy, ego_dz (if provided)
            - angular velocity (1D): d_yaw
            - Total: M = 10 dimensions
    """
    T = len(boxes)

    if T < 2:
        # Not enough frames for motion
        return np.zeros((T, 10))

    motion_features = []

    for t in range(T):
        # ===== VELOCITY (from box displacement) =====
        if t == 0:
            # First frame: no previous frame
            velocity = np.zeros(3)
            acceleration = np.zeros(3)
            angular_vel = 0.0
        else:
            # Compute displacement
            dt = 1.0  # Assume 1 frame = 1 timestep (adjust if you have real timestamps)
            if timestamps is not None:
                dt = max(timestamps[t] - timestamps[t-1], 1.0)

            # Position displacement
            dx = boxes[t, 0] - boxes[t-1, 0]
            dy = boxes[t, 1] - boxes[t-1, 1]
            dz = boxes[t, 2] - boxes[t-1, 2]
            velocity = np.array([dx, dy, dz]) / dt

            # Angular displacement
            d_yaw = boxes[t, 6] - boxes[t-1, 6]
            # Normalize to [-pi, pi]
            d_yaw = np.arctan2(np.sin(d_yaw), np.cos(d_yaw))
            angular_vel = d_yaw / dt

            # Acceleration (second derivative)
            if t == 1:
                acceleration = np.zeros(3)
            else:
                prev_dx = boxes[t-1, 0] - boxes[t-2, 0]
                prev_dy = boxes[t-1, 1] - boxes[t-2, 1]
                prev_dz = boxes[t-1, 2] - boxes[t-2, 2]
                prev_velocity = np.array([prev_dx, prev_dy, prev_dz]) / dt
                acceleration = (velocity - prev_velocity) / dt

        # ===== EGO MOTION =====
        if ego_motion is not None and t < len(ego_motion):
            ego_vel = ego_motion[t]  # [dx, dy, dz]
        else:
            ego_vel = np.zeros(3)

        # Concatenate all motion features: [vx, vy, vz, ax, ay, az, ego_dx, ego_dy, ego_dz, d_yaw]
        motion_feat = np.concatenate([
            velocity,       # 3D
            acceleration,   # 3D
            ego_vel,        # 3D
            [angular_vel]   # 1D
        ])  # Total: 10D

        motion_features.append(motion_feat)

    motion_map = np.array(motion_features)  # [T, 10]
    return motion_map


def compute_velocities_from_boxes(boxes, timestamps=None):
    """
    Compute object velocities from box positions over time.

    Args:
        boxes: np.array [T, 7] - boxes (x, y, z, l, w, h, yaw)
        timestamps: np.array [T] - frame indices (optional)

    Returns:
        velocities: np.array [T, 3] - (vx, vy, vz) velocities
    """
    T = len(boxes)
    velocities = np.zeros((T, 3))

    for t in range(1, T):
        dt = 1.0
        if timestamps is not None:
            dt = max(timestamps[t] - timestamps[t-1], 1.0)

        dx = boxes[t, 0] - boxes[t-1, 0]
        dy = boxes[t, 1] - boxes[t-1, 1]
        dz = boxes[t, 2] - boxes[t-1, 2]

        velocities[t] = np.array([dx, dy, dz]) / dt

    return velocities


def extract_ego_motion_from_batch(batch):
    """
    Extract ego motion from batch data.

    Args:
        batch: dict with 'ego_motion' or 'flow_gt' keys

    Returns:
        ego_motion: np.array [B, 3] - ego motion (dx, dy, dz)
    """
    if 'ego_motion' in batch:
        ego_motion = batch['ego_motion']  # [B, 3] or similar
        if isinstance(ego_motion, torch.Tensor):
            ego_motion = ego_motion.cpu().numpy()
        return ego_motion

    if 'flow_gt' in batch:
        # Compute average flow as ego motion approximation
        flow_gt = batch['flow_gt']  # [B, N, 3]
        if isinstance(flow_gt, torch.Tensor):
            flow_gt = flow_gt.cpu().numpy()

        # Average flow across all points
        ego_motion = flow_gt.mean(axis=1)  # [B, 3]
        return ego_motion

    # No ego motion available
    return None
