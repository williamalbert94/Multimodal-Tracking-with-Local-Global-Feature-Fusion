"""
model.py
========
Multimodal Tracking Network with Local-Global Feature Fusion.

Architecture
------------
1. Feature Extraction  : LocalGlobalFusionSimple (our contribution)
                         Input : pc [B,3,N] + radar_features [B,2,N]
                         Output: per-point features [B, 128, N]

2. Local-Global Fusion : max-pool global descriptor + concat with local
                         pc_features = cat(local, global) = [B, 256, N]

3. Feature Correlator  : KNN-based cost volume (RaTrack baseline pattern)
                         Finds K nearest pc2 neighbors for each pc1 point,
                         aggregates with learnable position-dependent weights.
                         Output: [B, 256, N]

4. Flow Decoder        : Simplified Conv1D decoder — no GRU, no DBSCAN.
                         Input : cat(coords, radar, pc1_feat, cor_feat) [B,517,N]
                         Outputs: flow[B,3,N], seg[B,1,N], prop_features[B,128,N]

5. Supervised Seg Head : Deeper head with residual on shared backbone (our contribution).
                         Output: [B, 1, N]

Losses (trainer_simple.py)
---------------------------
  L_seg  : motion_seg_loss — weighted BCE (0.4*moving + 0.6*static)  [baseline]
  L_flow : flow_loss       — EPE = ||pc1 + pred_flow - gt_warped||_2  [baseline]
  L_total = L_seg + 0.5 * L_flow  (pretrain: only L_seg)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.feature_extractor import *           # LocalGlobalFusionSimple, PNHead
from utils.models_utils import log_optimal_transport, arange_like, obj_centre


# ─────────────────────────────────────────────────────────────────────────────
# KNN helper  (pure PyTorch — no compiled CUDA op required)
# ─────────────────────────────────────────────────────────────────────────────

def _knn_gather(pc2_xyz, pc1_xyz, feature2, K):
    """
    For each point in pc1_xyz, find K nearest neighbors in pc2_xyz.

    Args:
        pc2_xyz:  [B, N2, 3]
        pc1_xyz:  [B, N1, 3]
        feature2: [B, N2, C]
        K       : int

    Returns:
        neighbor_xyz:  [B, N1, K, 3]
        neighbor_feat: [B, N1, K, C]
    """
    B, N1, _ = pc1_xyz.shape
    C = feature2.shape[-1]

    dists = torch.cdist(pc1_xyz, pc2_xyz)         # [B, N1, N2]
    _, idx = dists.topk(K, dim=-1, largest=False)  # [B, N1, K]

    idx_xyz  = idx.unsqueeze(-1).expand(-1, -1, -1, 3)
    idx_feat = idx.unsqueeze(-1).expand(-1, -1, -1, C)

    neighbor_xyz  = pc2_xyz.unsqueeze(1).expand(-1, N1, -1, -1).gather(2, idx_xyz)
    neighbor_feat = feature2.unsqueeze(1).expand(-1, N1, -1, -1).gather(2, idx_feat)

    return neighbor_xyz, neighbor_feat   # [B, N1, K, 3/C]


# ─────────────────────────────────────────────────────────────────────────────
# WeightNet — learnable position-to-weight mapping (baseline)
# ─────────────────────────────────────────────────────────────────────────────

class WeightNet(nn.Module):
    """Maps relative XYZ offsets → per-neighbor aggregation weights."""

    def __init__(self, in_dim: int = 3, out_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, 8, 1), nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Conv2d(8, out_dim, 1),
        )

    def forward(self, xyz):
        """xyz: [B, 3, K, N] → [B, out_dim, K, N]"""
        return self.net(xyz)


# ─────────────────────────────────────────────────────────────────────────────
# FeatureCorrelator — KNN cost volume (RaTrack baseline, adapted)
# ─────────────────────────────────────────────────────────────────────────────

class FeatureCorrelator(nn.Module):
    """
    Two-stage KNN cost volume.

    Stage 1 (point-to-patch):
        For each pc1 point, gather K nearest pc2 neighbors.
        Concatenate [pc1_feat | pc2_neighbor_feat | relative_xyz] per neighbor.
        Process with Conv2D MLP → weighted sum over K → [B, C_out, N].

    Stage 2 (patch-to-patch):
        Self-aggregation in pc1 space using the same K-NN pattern.
        Refines spatial context before passing to the flow decoder.
    """

    def __init__(self, nsample: int, in_channel: int, mlp: list):
        """
        nsample   : K neighbors
        in_channel: D1 + D2 + 3  (e.g. 256+256+3 = 515)
        mlp       : output channel sizes, e.g. [256, 256, 256]
        """
        super().__init__()
        self.nsample = nsample
        self.relu = nn.LeakyReLU(0.1, inplace=True)

        self.mlp_convs = nn.ModuleList()
        last_ch = in_channel
        for out_ch in mlp:
            self.mlp_convs.append(nn.Conv2d(last_ch, out_ch, 1))
            last_ch = out_ch                        # last_ch = mlp[-1]

        self.weightnet1 = WeightNet(3, last_ch)     # cross-frame weights
        self.weightnet2 = WeightNet(3, last_ch)     # self-aggregation weights

    def forward(self, pc1, pc2, feature1, feature2):
        """
        pc1, pc2:           [B, 3, N]   (channel-first coordinates)
        feature1, feature2: [B, C, N]

        Returns: [B, mlp[-1], N]
        """
        B, _, N1 = pc1.shape
        K = self.nsample

        # Convert to [B, N, ...] for _knn_gather
        pc1_t   = pc1.permute(0, 2, 1)           # [B, N1, 3]
        pc2_t   = pc2.permute(0, 2, 1)           # [B, N2, 3]
        feat1_t = feature1.permute(0, 2, 1)      # [B, N1, C1]
        feat2_t = feature2.permute(0, 2, 1)      # [B, N2, C2]

        # ── Stage 1: point-to-patch ──────────────────────────────────────────
        nbr_xyz, nbr_feat2 = _knn_gather(pc2_t, pc1_t, feat2_t, K)
        # [B, N1, K, 3]  [B, N1, K, C2]

        direction = nbr_xyz - pc1_t.unsqueeze(2)               # [B, N1, K, 3]
        feat1_exp = feat1_t.unsqueeze(2).expand(-1, -1, K, -1) # [B, N1, K, C1]

        x = torch.cat([feat1_exp, nbr_feat2, direction], dim=-1)  # [B, N1, K, C1+C2+3]
        x = x.permute(0, 3, 2, 1)                                 # [B, C1+C2+3, K, N1]

        for conv in self.mlp_convs:
            x = self.relu(conv(x))                                 # [B, last_ch, K, N1]

        w1 = self.weightnet1(direction.permute(0, 3, 2, 1))       # [B, last_ch, K, N1]
        x  = (w1 * x).sum(dim=2)                                   # [B, last_ch, N1]

        # ── Stage 2: patch-to-patch self-aggregation ─────────────────────────
        nbr_xyz2, nbr_feat_self = _knn_gather(pc1_t, pc1_t, x.permute(0, 2, 1), K)
        # x.permute: [B, N1, last_ch]
        # nbr_xyz2:  [B, N1, K, 3]
        # nbr_feat_self: [B, N1, K, last_ch]

        direction2 = nbr_xyz2 - pc1_t.unsqueeze(2)                 # [B, N1, K, 3]
        w2 = self.weightnet2(direction2.permute(0, 3, 2, 1))       # [B, last_ch, K, N1]
        x2 = nbr_feat_self.permute(0, 3, 2, 1)                     # [B, last_ch, K, N1]
        x  = (w2 * x2).sum(dim=2)                                   # [B, last_ch, N1]

        return x   # [B, mlp[-1], N]


# ─────────────────────────────────────────────────────────────────────────────
# FlowDecoder — simple Conv1D, no GRU
# ─────────────────────────────────────────────────────────────────────────────

class FlowDecoder(nn.Module):
    """
    Lightweight scene-flow and segmentation decoder (no GRU, no clustering).

    Input : cat(pc1_coords[3], radar_feat[2], pc1_fused[256], cor[256]) = [B,517,N]
    Output: flow [B,3,N], seg [B,1,N], prop_features [B,128,N]
    """

    def __init__(self, in_dim: int = 517):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Conv1d(in_dim, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(256, 256, 1),   nn.BatchNorm1d(256), nn.ReLU(),
        )

        self.flow_head = nn.Sequential(
            nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 3, 1),
        )

        self.seg_head = nn.Sequential(
            nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1, 1),
            nn.Sigmoid(),
        )

        self.prop_head = nn.Sequential(
            nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
        )

    def forward(self, pc1, feature1, pc1_features, cor_features):
        """
        pc1:          [B, 3,   N]
        feature1:     [B, 2,   N]
        pc1_features: [B, 256, N]
        cor_features: [B, 256, N]
        """
        x = torch.cat([pc1, feature1, pc1_features, cor_features], dim=1)
        x = self.shared(x)                       # [B, 256, N]
        return self.flow_head(x), self.seg_head(x), self.prop_head(x)


# ─────────────────────────────────────────────────────────────────────────────
# SupervisedSegmentationHead — residual head (our contribution)
# ─────────────────────────────────────────────────────────────────────────────

class SupervisedSegmentationHead(nn.Module):
    """
    Deeper supervised segmentation head with skip connection.
    Input : pc1_features [B, 256, N]
    Output: seg_pred     [B, 1,   N]
    """

    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.conv1     = nn.Sequential(nn.Conv1d(feature_dim, 256, 1), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2))
        self.conv2     = nn.Sequential(nn.Conv1d(256, 256, 1),         nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2))
        self.skip_proj = nn.Conv1d(feature_dim, 256, 1)
        self.conv3     = nn.Sequential(nn.Conv1d(256, 128, 1),         nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1))
        self.final     = nn.Sequential(
            nn.Conv1d(128, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 1, 1), nn.Sigmoid(),
        )

    def forward(self, pc1_features):
        x    = self.conv1(pc1_features)
        skip = self.skip_proj(pc1_features)
        x    = self.conv2(x) + skip          # residual
        x    = self.conv3(x)
        return self.final(x)                 # [B, 1, N]


# ─────────────────────────────────────────────────────────────────────────────
# DetectionHead — 3D Object Detection (CenterPoint-style)
# ─────────────────────────────────────────────────────────────────────────────

class DetectionHead(nn.Module):
    """
    3D Object Detection Head (anchor-free, CenterPoint-style).

    Predicts per-point:
      - Center heatmap: [B, 1, N] - probability of object center
      - Box dimensions: [B, 3, N] - (length, width, height)
      - Orientation:    [B, 2, N] - (sin(yaw), cos(yaw))
      - Class logits:   [B, num_classes, N] - object classification

    Input : pc1_features [B, 256, N]
    """

    def __init__(self, feature_dim: int = 256, num_classes: int = 4):
        """
        Args:
            feature_dim: Input feature dimension
            num_classes: Number of object classes (e.g., car, ped, cyclist, unknown)
        """
        super().__init__()
        self.num_classes = num_classes

        # Shared backbone
        self.shared = nn.Sequential(
            nn.Conv1d(feature_dim, 256, 1), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(256, 256, 1),         nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
        )

        # Residual connection
        self.skip_proj = nn.Conv1d(feature_dim, 256, 1)

        # Task-specific heads
        self.center_head = nn.Sequential(
            nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 64, 1),  nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Conv1d(64, 1, 1),    nn.Sigmoid(),  # Probability [0, 1]
        )

        self.size_head = nn.Sequential(
            nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 64, 1),  nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Conv1d(64, 3, 1),    nn.ReLU(),  # Length, Width, Height (positive values)
        )

        self.orientation_head = nn.Sequential(
            nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 64, 1),  nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Conv1d(64, 2, 1),    # sin(yaw), cos(yaw) - no activation (regression)
        )

        self.class_head = nn.Sequential(
            nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 64, 1),  nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Conv1d(64, num_classes, 1),  # Class logits (no softmax - use CrossEntropy)
        )

    def forward(self, pc1_features):
        """
        Args:
            pc1_features: [B, 256, N] per-point features

        Returns dict:
            'center':      [B, 1, N] - center heatmap (probability)
            'size':        [B, 3, N] - box dimensions (l, w, h)
            'orientation': [B, 2, N] - (sin(yaw), cos(yaw))
            'class':       [B, num_classes, N] - class logits
        """
        # Shared backbone with residual
        x = self.shared(pc1_features)
        skip = self.skip_proj(pc1_features)
        x = x + skip  # Residual

        # Task-specific predictions
        center_heatmap = self.center_head(x)       # [B, 1, N]
        box_size       = self.size_head(x)         # [B, 3, N]
        orientation    = self.orientation_head(x)  # [B, 2, N]
        class_logits   = self.class_head(x)        # [B, num_classes, N]

        # Normalize orientation to unit vector
        orientation = F.normalize(orientation, dim=1)  # sin^2 + cos^2 = 1

        return {
            'center': center_heatmap,
            'size': box_size,
            'orientation': orientation,
            'class': class_logits,
        }


# ─────────────────────────────────────────────────────────────────────────────
# rastreador — main model
# ─────────────────────────────────────────────────────────────────────────────

class rastreador(nn.Module):
    """
    Multimodal Tracking Network — Local-Global Feature Fusion.

    Backward-compatible forward signature:
        net(pc1, pc2, feature1, feature2, h=None) → dict
    h is accepted but unused (no GRU).
    """

    def __init__(self, args):
        super().__init__()

        self.npoints           = args.num_points
        self.use_supervised_seg = getattr(args, 'use_supervised_seg', True)
        self.use_detection_head = getattr(args, 'use_detection_head', False)  # NEW: Detection head for Phase 1

        # ── 1. Feature extractor (our contribution) ──────────────────────────
        if args.extractor == 'PNHead':
            print(f'Using feature extractor: {args.extractor}')
            self.pn_head = PNHead(args.num_points, 2)
        elif args.extractor == 'LocalGlobalFusionSimple':
            print(f'Using feature extractor: {args.extractor}')
            self.pn_head = LocalGlobalFusionSimple(args.num_points, 2)
        else:
            raise ValueError(f'Unknown extractor: {args.extractor}')

        # ── 2. KNN cost volume (baseline FeatureCorrelator) ──────────────────
        # Per-frame fused features = 256  →  D1 + D2 + 3 = 515
        fc_inch = 256
        self.fc_layer = FeatureCorrelator(
            nsample=16,
            in_channel=fc_inch * 2 + 3,      # 515
            mlp=[fc_inch, fc_inch, fc_inch],  # output = 256
        )

        # ── 3. Flow decoder (no GRU) ─────────────────────────────────────────
        # Input: coords(3) + radar(2) + pc1_fused(256) + cor(256) = 517
        self.fd_layer = FlowDecoder(in_dim=3 + 2 + fc_inch + fc_inch)

        # ── 4. Supervised segmentation head (our contribution) ───────────────
        self.supervised_seg = SupervisedSegmentationHead(feature_dim=fc_inch)

        # ── 5. Detection head (NEW - for Phase 1 detector training) ──────────
        if self.use_detection_head:
            self.detection_head = DetectionHead(feature_dim=fc_inch, num_classes=4)
            print('Detection Head enabled (anchor-free 3D object detection)')

    # ── Public forward ───────────────────────────────────────────────────────

    def forward(self, pc1, pc2, feature1, feature2, h=None):
        """
        Args:
            pc1, pc2:       [B, 3, N]  point coordinates (channel-first)
            feature1/2:     [B, 2, N]  radar features (velocity, RCS)
            h:              ignored    (no GRU; kept for caller compatibility)

        Returns dict with keys:
            'flow'           [B, 3, N]
            'h'              None
            'cls_flow'       [B, 1, N]
            'cls_supervised' [B, 1, N] or None
            'pc1_features'   [B, 256, N]
            'pc2_features'   [B, 256, N]
            'prop_features'  [B, 128, N]
            'detection'      dict or None (if use_detection_head=True)
                'center':      [B, 1, N]
                'size':        [B, 3, N]
                'orientation': [B, 2, N]
                'class':       [B, num_classes, N]
        """
        flow, cls_flow, pc1_features, pc2_features, prop_features = \
            self._backbone(pc1, pc2, feature1, feature2)

        seg_pred_supervised = None
        if self.use_supervised_seg:
            seg_pred_supervised = self.supervised_seg(pc1_features)

        detection_outputs = None
        if self.use_detection_head:
            detection_outputs = self.detection_head(pc1_features)

        return {
            'flow'           : flow,
            'h'              : None,
            'cls_flow'       : cls_flow,
            'cls_supervised' : seg_pred_supervised,
            'pc1_features'   : pc1_features,
            'pc2_features'   : pc2_features,
            'prop_features'  : prop_features,
            'detection'      : detection_outputs,  # NEW: detection predictions
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _backbone(self, pc1, pc2, feature1, feature2):
        # 1. Extract per-point local features from both frames
        pc1_local, pc2_local = self._extract(pc1, pc2, feature1, feature2)
        # [B, 128, N] each

        # 2. Fuse local + global max-pool descriptor → [B, 256, N]
        pc1_fused, pc2_fused = self._fuse_local_global(pc1_local, pc2_local)

        # 3. KNN cost volume → correlation features [B, 256, N]
        cor = self.fc_layer(pc1, pc2, pc1_fused, pc2_fused)

        # 4. Decode flow + segmentation
        flow, cls_flow, prop = self.fd_layer(pc1, feature1, pc1_fused, cor)

        return flow, cls_flow, pc1_fused, pc2_fused, prop

    def _extract(self, pc1, pc2, feature1, feature2):
        """Run pn_head on both frames. Returns [B, 128, N] each."""
        _, f1 = self.pn_head(pc1.permute(0, 2, 1).contiguous(),
                             feature1.permute(0, 2, 1).contiguous())
        _, f2 = self.pn_head(pc2.permute(0, 2, 1).contiguous(),
                             feature2.permute(0, 2, 1).contiguous())
        return f1, f2

    @staticmethod
    def _fuse_local_global(f1, f2):
        """Append global max-pool to local features: [B,128,N] → [B,256,N]."""
        g1 = f1.max(dim=-1, keepdim=True)[0].expand_as(f1)
        g2 = f2.max(dim=-1, keepdim=True)[0].expand_as(f2)
        return torch.cat([f1, g1], dim=1), torch.cat([f2, g2], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def load_model(args, logger):
    logger.info('Loading rastreador model...')
    net = rastreador(args).cuda()
    logger.info(
        f'Model loaded. Total params: '
        f'{sum(p.numel() for p in net.parameters()) / 1e6:.2f}M'
    )
    return net
