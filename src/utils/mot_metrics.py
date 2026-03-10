"""
MOT Metrics Implementation
Compute standard tracking metrics: MOTA, IDF1, AMOTA, sAMOTA, etc.

Reference:
- MOTA/MOTP: Bernardin & Stiefelhagen (2008)
- IDF1: Ristani et al. (2016)
- AMOTA/sAMOTA: Weng & Kitani (2019) - nuScenes tracking
"""

import numpy as np
import torch


class MOTMetricsAccumulator:
    """
    Accumulate tracking results for MOT metrics computation.

    Usage:
        accumulator = MOTMetricsAccumulator()

        for frame in sequence:
            pred_boxes, pred_ids = tracker.update(detections)
            accumulator.update(frame_id, gt_boxes, gt_ids, pred_boxes, pred_ids)

        metrics = accumulator.compute_metrics()
    """

    def __init__(self):
        self.frame_data = []  # List of (frame_id, gt_boxes, gt_ids, pred_boxes, pred_ids)
        self.gt_tracks = {}   # {track_id: [frame_ids]}
        self.pred_tracks = {}  # {track_id: [frame_ids]}

    def update(self, frame_id, gt_boxes, gt_ids, pred_boxes, pred_ids, iou_threshold=0.5):
        """
        Update accumulator with results from one frame.

        Args:
            frame_id: int - frame number
            gt_boxes: [N, 7] tensor - GT boxes (x, y, z, l, w, h, yaw)
            gt_ids: [N] tensor - GT track IDs
            pred_boxes: [M, 7] tensor - predicted boxes
            pred_ids: [M] tensor - predicted track IDs
            iou_threshold: float - IoU threshold for matching
        """
        # Convert to numpy
        if isinstance(gt_boxes, torch.Tensor):
            gt_boxes = gt_boxes.cpu().numpy()
        if isinstance(gt_ids, torch.Tensor):
            gt_ids = gt_ids.cpu().numpy()
        if isinstance(pred_boxes, torch.Tensor):
            pred_boxes = pred_boxes.cpu().numpy()
        if isinstance(pred_ids, torch.Tensor):
            pred_ids = pred_ids.cpu().numpy()

        # Store frame data
        self.frame_data.append({
            'frame_id': frame_id,
            'gt_boxes': gt_boxes,
            'gt_ids': gt_ids,
            'pred_boxes': pred_boxes,
            'pred_ids': pred_ids,
        })

        # Track GT trajectories
        if gt_ids is not None:
            for track_id in gt_ids:
                if track_id not in self.gt_tracks:
                    self.gt_tracks[track_id] = []
                self.gt_tracks[track_id].append(frame_id)

        # Track predicted trajectories
        if pred_ids is not None:
            for track_id in pred_ids:
                if track_id not in self.pred_tracks:
                    self.pred_tracks[track_id] = []
                self.pred_tracks[track_id].append(frame_id)

    def compute_metrics(self, iou_thresholds=None):
        """
        Compute all MOT metrics.

        Args:
            iou_thresholds: list of float - IoU thresholds for average metrics
                           Default: [0.5, 0.55, 0.6, ..., 0.95] (nuScenes-style)

        Returns:
            metrics: dict with all MOT metrics
        """
        if iou_thresholds is None:
            iou_thresholds = np.arange(0.5, 1.0, 0.05)  # [0.5, 0.55, ..., 0.95]

        # STEP 1: Global ID remapping FIRST
        # This ensures predicted IDs are optimally aligned with GT IDs
        # The model gets credit for consistent tracking regardless of ID numbers
        id_map = self._remap_predicted_ids_globally()

        # STEP 2: Compute metrics with remapped IDs
        # Compute standard metrics at IoU=0.5
        metrics_0_5 = self._compute_single_threshold_metrics(iou_threshold=0.5)

        # Compute average metrics over multiple thresholds
        motas = []
        motps = []

        for iou_thresh in iou_thresholds:
            m = self._compute_single_threshold_metrics(iou_threshold=iou_thresh)
            motas.append(m['MOTA'])
            motps.append(m['MOTP'])

        amota = np.mean(motas)
        amotp = np.mean(motps)
        # sAMOTA = AMOTA × (1 - TP_error), where TP_error = 1 - MOTP_precision
        # Since MOTP is returned as precision (higher = better), we need: TP_error = 1 - (MOTP/100)
        # Therefore: sAMOTA = AMOTA × (1 - (1 - MOTP/100)) = AMOTA × (MOTP/100)
        # Note: MOTP here is the average IoU over matched boxes (precision metric)
        samota = (amota / 100.0) * (amotp / 100.0) * 100.0  # Scaled AMOTA = AMOTA × MOTP_precision

        # Combine all metrics
        all_metrics = {
            # Primary metrics (IoU=0.5)
            'MOTA': metrics_0_5['MOTA'],
            'IDF1': metrics_0_5['IDF1'],
            'MOTP': metrics_0_5['MOTP'],

            # Average metrics
            'sAMOTA': samota,
            'AMOTA': amota,
            'AMOTP': amotp,

            # Detection-only
            'MODA': metrics_0_5['MODA'],

            # Track quality
            'MT': metrics_0_5['MT'],
            'PT': metrics_0_5['PT'],
            'ML': metrics_0_5['ML'],

            # Error breakdown
            'ID_switches': metrics_0_5['ID_switches'],
            'Fragmentations': metrics_0_5['Fragmentations'],
            'FP': metrics_0_5['FP'],
            'FN': metrics_0_5['FN'],
        }

        return all_metrics

    def _compute_single_threshold_metrics(self, iou_threshold=0.5):
        """
        Compute MOT metrics for a single IoU threshold.

        Args:
            iou_threshold: float - IoU threshold for matching

        Returns:
            metrics: dict with metrics at this threshold
        """
        total_gt = 0
        total_pred = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_id_switches = 0
        total_fragmentations = 0
        sum_iou = 0.0

        # Track last matched ID for each GT track (for ID switch detection)
        gt_track_last_match = {}  # {gt_id: last_matched_pred_id}

        for frame_data in self.frame_data:
            gt_boxes = frame_data.get('gt_boxes', [])
            gt_ids = frame_data.get('gt_ids', None)
            pred_boxes = frame_data.get('pred_boxes', [])
            pred_ids = frame_data.get('pred_ids', None)

            # Skip frames with no data
            if gt_ids is None or pred_ids is None:
                continue

            total_gt += len(gt_ids)
            total_pred += len(pred_ids)

            # Match predictions to GT using Hungarian algorithm
            matches, unmatched_gt, unmatched_pred = self._match_boxes(
                pred_boxes, pred_ids, gt_boxes, gt_ids, iou_threshold
            )

            # True Positives
            for pred_idx, gt_idx, iou in matches:
                total_tp += 1
                sum_iou += iou

                # Check for ID switch
                gt_id = gt_ids[gt_idx]
                pred_id = pred_ids[pred_idx]

                if gt_id in gt_track_last_match:
                    if gt_track_last_match[gt_id] != pred_id:
                        total_id_switches += 1

                gt_track_last_match[gt_id] = pred_id

            # False Positives
            total_fp += len(unmatched_pred)

            # False Negatives
            total_fn += len(unmatched_gt)

        # Compute fragmentations (track interruptions)
        for gt_id, frame_list in self.gt_tracks.items():
            frame_list = sorted(frame_list)
            for i in range(len(frame_list) - 1):
                # If frames are not consecutive, it's a gap
                if frame_list[i+1] - frame_list[i] > 1:
                    total_fragmentations += 1

        # Compute MT, PT, ML (track quality)
        num_gt_tracks = len(self.gt_tracks)
        mt_count = 0  # Mostly Tracked
        pt_count = 0  # Partially Tracked
        ml_count = 0  # Mostly Lost

        for gt_id, frame_list in self.gt_tracks.items():
            total_frames = len(frame_list)
            # Count how many frames this GT track was successfully matched
            matched_frames = 0
            for frame_data in self.frame_data:
                if gt_id in frame_data['gt_ids']:
                    # Check if this GT was matched in this frame
                    gt_idx = np.where(frame_data['gt_ids'] == gt_id)[0][0]
                    matches, _, _ = self._match_boxes(
                        frame_data['pred_boxes'],
                        frame_data['pred_ids'],
                        frame_data['gt_boxes'],
                        frame_data['gt_ids'],
                        iou_threshold
                    )
                    for pred_idx, matched_gt_idx, iou in matches:
                        if matched_gt_idx == gt_idx:
                            matched_frames += 1
                            break

            ratio = matched_frames / max(total_frames, 1)
            if ratio > 0.8:
                mt_count += 1
            elif ratio < 0.2:
                ml_count += 1
            else:
                pt_count += 1

        # Compute metrics
        mota = 1 - (total_fn + total_fp + total_id_switches) / max(total_gt, 1)
        motp = sum_iou / max(total_tp, 1)
        moda = 1 - (total_fn + total_fp) / max(total_gt, 1)  # MOTA without ID switches

        # IDF1 (ID F1 Score) - requires IDTP, IDFP, IDFN
        # Simplified version: approximate using TP and ID switches
        idf1 = total_tp / max(total_tp + 0.5 * (total_fp + total_fn + total_id_switches), 1)

        mt = mt_count / max(num_gt_tracks, 1)
        pt = pt_count / max(num_gt_tracks, 1)
        ml = ml_count / max(num_gt_tracks, 1)

        return {
            'MOTA': mota * 100,  # Convert to percentage
            'MOTP': motp * 100,
            'MODA': moda * 100,
            'IDF1': idf1 * 100,
            'MT': mt * 100,
            'PT': pt * 100,
            'ML': ml * 100,
            'ID_switches': total_id_switches,
            'Fragmentations': total_fragmentations,
            'FP': total_fp,
            'FN': total_fn,
            'TP': total_tp,
        }

    def _match_boxes(self, pred_boxes, pred_ids, gt_boxes, gt_ids, iou_threshold):
        """
        Match predicted boxes to GT using Hungarian algorithm based on IoU.

        Args:
            pred_boxes: [M, 7] numpy array
            pred_ids: [M] numpy array
            gt_boxes: [N, 7] numpy array
            gt_ids: [N] numpy array
            iou_threshold: float

        Returns:
            matches: list of (pred_idx, gt_idx, iou)
            unmatched_gt: list of gt indices
            unmatched_pred: list of pred indices
        """
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            return [], list(range(len(gt_boxes))), list(range(len(pred_boxes)))

        # Compute IoU matrix [M, N]
        iou_matrix = self._compute_iou_matrix(pred_boxes, gt_boxes)

        # Hungarian algorithm
        from scipy.optimize import linear_sum_assignment
        pred_indices, gt_indices = linear_sum_assignment(-iou_matrix)  # Maximize IoU

        # Filter matches below threshold
        matches = []
        matched_pred = set()
        matched_gt = set()

        for pred_idx, gt_idx in zip(pred_indices, gt_indices):
            iou = iou_matrix[pred_idx, gt_idx]
            if iou >= iou_threshold:
                matches.append((pred_idx, gt_idx, iou))
                matched_pred.add(pred_idx)
                matched_gt.add(gt_idx)

        unmatched_gt = [i for i in range(len(gt_boxes)) if i not in matched_gt]
        unmatched_pred = [i for i in range(len(pred_boxes)) if i not in matched_pred]

        return matches, unmatched_gt, unmatched_pred

    def _compute_iou_matrix(self, boxes1, boxes2):
        """
        Compute IoU matrix between two sets of boxes (2D Bird's Eye View).

        Args:
            boxes1: [M, 7] boxes (x, y, z, l, w, h, yaw)
            boxes2: [N, 7] boxes

        Returns:
            iou_matrix: [M, N] IoU values
        """
        M = len(boxes1)
        N = len(boxes2)
        iou_matrix = np.zeros((M, N))

        for i in range(M):
            for j in range(N):
                iou_matrix[i, j] = self._compute_iou_2d(boxes1[i], boxes2[j])

        return iou_matrix

    def _compute_iou_2d(self, box1, box2):
        """
        Compute 2D IoU on Bird's Eye View (simplified axis-aligned version).

        Args:
            box1: [7] box (x, y, z, l, w, h, yaw)
            box2: [7] box

        Returns:
            iou: float
        """
        # Extract x, y, l, w (ignore z, h, yaw for simplicity)
        x1, y1, l1, w1 = box1[0], box1[1], box1[3], box1[4]
        x2, y2, l2, w2 = box2[0], box2[1], box2[3], box2[4]

        # Axis-aligned bounding boxes
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
        iou = inter_area / max(union_area, 1e-6)
        return iou

    def _build_track_trajectories(self):
        """
        Build trajectory dictionaries for GT and predicted tracks.

        Returns:
            gt_tracks: dict {track_id: {'frames': [f1, f2, ...], 'boxes': [box1, box2, ...]}}
            pred_tracks: dict {track_id: {'frames': [...], 'boxes': [...]}}
        """
        gt_tracks = {}
        pred_tracks = {}

        for frame_data in self.frame_data:
            frame_id = frame_data['frame_id']

            # GT trajectories
            gt_ids = frame_data.get('gt_ids', None)
            if gt_ids is not None:
                for i, gt_id in enumerate(gt_ids):
                    if gt_id not in gt_tracks:
                        gt_tracks[gt_id] = {'frames': [], 'boxes': []}
                    gt_tracks[gt_id]['frames'].append(frame_id)
                    gt_tracks[gt_id]['boxes'].append(frame_data['gt_boxes'][i])

            # Predicted trajectories
            pred_ids = frame_data.get('pred_ids', None)
            if pred_ids is not None:
                for i, pred_id in enumerate(pred_ids):
                    if pred_id not in pred_tracks:
                        pred_tracks[pred_id] = {'frames': [], 'boxes': []}
                    pred_tracks[pred_id]['frames'].append(frame_id)
                    pred_tracks[pred_id]['boxes'].append(frame_data['pred_boxes'][i])

        return gt_tracks, pred_tracks

    def _compute_track_similarity_matrix(self, gt_tracks, pred_tracks):
        """
        Compute similarity matrix between predicted and GT track trajectories.

        Similarity = avg_IoU_over_overlapping_frames * overlap_ratio

        Args:
            gt_tracks: dict from _build_track_trajectories
            pred_tracks: dict from _build_track_trajectories

        Returns:
            similarity_matrix: [M, N] where M=num_pred_tracks, N=num_gt_tracks
            pred_ids_list: list of pred track IDs (row indices)
            gt_ids_list: list of GT track IDs (column indices)
        """
        gt_ids_list = list(gt_tracks.keys())
        pred_ids_list = list(pred_tracks.keys())

        if len(pred_ids_list) == 0 or len(gt_ids_list) == 0:
            return np.zeros((len(pred_ids_list), len(gt_ids_list))), pred_ids_list, gt_ids_list

        M = len(pred_ids_list)
        N = len(gt_ids_list)
        similarity_matrix = np.zeros((M, N))

        for i, pred_id in enumerate(pred_ids_list):
            pred_frames = set(pred_tracks[pred_id]['frames'])
            pred_boxes = pred_tracks[pred_id]['boxes']

            for j, gt_id in enumerate(gt_ids_list):
                gt_frames = set(gt_tracks[gt_id]['frames'])
                gt_boxes = gt_tracks[gt_id]['boxes']

                # Find overlapping frames
                overlap_frames = pred_frames & gt_frames

                if len(overlap_frames) == 0:
                    similarity_matrix[i, j] = 0.0
                    continue

                # Compute average IoU over overlapping frames
                ious = []
                for frame_id in overlap_frames:
                    # Find box indices for this frame
                    pred_frame_idx = pred_tracks[pred_id]['frames'].index(frame_id)
                    gt_frame_idx = gt_tracks[gt_id]['frames'].index(frame_id)

                    pred_box = pred_boxes[pred_frame_idx]
                    gt_box = gt_boxes[gt_frame_idx]

                    iou = self._compute_iou_2d(pred_box, gt_box)
                    ious.append(iou)

                avg_iou = np.mean(ious)
                # Overlap ratio: what fraction of GT track lifetime overlaps with prediction
                overlap_ratio = len(overlap_frames) / max(len(gt_frames), 1)

                # Similarity = IoU quality × temporal overlap
                similarity_matrix[i, j] = avg_iou * overlap_ratio

        return similarity_matrix, pred_ids_list, gt_ids_list

    def _remap_predicted_ids_globally(self):
        """
        Globally remap predicted track IDs to GT track IDs using Hungarian matching.

        This ensures that the model gets credit for consistent tracking even when
        using different ID numbers. After remapping, MOTA/IDF1 metrics will be more fair.

        The method:
        1. Builds track trajectories for GT and predictions
        2. Computes similarity matrix based on IoU and temporal overlap
        3. Uses Hungarian algorithm to find optimal global ID assignment
        4. Remaps all predicted IDs in frame_data to matched GT IDs

        Returns:
            id_map: dict {old_pred_id: new_remapped_id}
        """
        # Build track trajectories
        gt_tracks, pred_tracks = self._build_track_trajectories()

        # Compute similarity matrix
        similarity_matrix, pred_ids_list, gt_ids_list = self._compute_track_similarity_matrix(
            gt_tracks, pred_tracks
        )

        if len(pred_ids_list) == 0 or len(gt_ids_list) == 0:
            return {}

        # Hungarian matching (maximize similarity)
        from scipy.optimize import linear_sum_assignment
        pred_indices, gt_indices = linear_sum_assignment(-similarity_matrix)

        # Build ID mapping
        id_map = {}
        max_gt_id = int(max(gt_ids_list)) if gt_ids_list else 0
        next_new_id = max_gt_id + 1

        # Threshold for accepting a match (require at least 10% similarity)
        SIMILARITY_THRESHOLD = 0.1

        for pred_idx, gt_idx in zip(pred_indices, gt_indices):
            pred_id = pred_ids_list[pred_idx]
            gt_id = gt_ids_list[gt_idx]

            if similarity_matrix[pred_idx, gt_idx] > SIMILARITY_THRESHOLD:
                # Good match - map to GT ID
                id_map[pred_id] = gt_id
            else:
                # Poor match - assign new ID
                id_map[pred_id] = next_new_id
                next_new_id += 1

        # Assign IDs to remaining unmatched predictions
        matched_pred = set(id_map.keys())
        for pred_id in pred_ids_list:
            if pred_id not in matched_pred:
                id_map[pred_id] = next_new_id
                next_new_id += 1

        # Apply remapping to all frame data
        for frame_data in self.frame_data:
            remapped_ids = []
            for pred_id in frame_data['pred_ids']:
                # Remap if ID exists in map, otherwise keep original
                remapped_id = id_map.get(pred_id, pred_id)
                remapped_ids.append(remapped_id)
            frame_data['pred_ids'] = np.array(remapped_ids)

        # Update pred_tracks dict with remapped IDs
        new_pred_tracks = {}
        for old_id, frame_list in self.pred_tracks.items():
            new_id = id_map.get(old_id, old_id)
            if new_id in new_pred_tracks:
                # Merge frame lists if multiple old IDs map to same new ID
                new_pred_tracks[new_id].extend(frame_list)
            else:
                new_pred_tracks[new_id] = frame_list
        self.pred_tracks = new_pred_tracks

        return id_map


def compute_mot_metrics_simple(pred_tracks_all, gt_tracks_all):
    """
    Simple wrapper to compute MOT metrics from full sequence predictions.

    Args:
        pred_tracks_all: dict {frame_id: {'boxes': [M, 7], 'ids': [M]}}
        gt_tracks_all: dict {frame_id: {'boxes': [N, 7], 'ids': [N]}}

    Returns:
        metrics: dict with all MOT metrics

    Example:
        pred_tracks = {
            0: {'boxes': tensor([[...], [...]]), 'ids': tensor([1, 2])},
            1: {'boxes': tensor([[...]]), 'ids': tensor([1])},
            ...
        }
        metrics = compute_mot_metrics_simple(pred_tracks, gt_tracks)
        print(f"MOTA: {metrics['MOTA']:.2f}%")
    """
    accumulator = MOTMetricsAccumulator()

    # Get all frame IDs
    all_frame_ids = sorted(set(pred_tracks_all.keys()) | set(gt_tracks_all.keys()))

    for frame_id in all_frame_ids:
        # Get predictions (or empty if frame not in predictions)
        pred_data = pred_tracks_all.get(frame_id, {'boxes': np.zeros((0, 7)), 'ids': np.zeros(0)})
        gt_data = gt_tracks_all.get(frame_id, {'boxes': np.zeros((0, 7)), 'ids': np.zeros(0)})

        accumulator.update(
            frame_id=frame_id,
            gt_boxes=gt_data['boxes'],
            gt_ids=gt_data['ids'],
            pred_boxes=pred_data['boxes'],
            pred_ids=pred_data['ids']
        )

    return accumulator.compute_metrics()
